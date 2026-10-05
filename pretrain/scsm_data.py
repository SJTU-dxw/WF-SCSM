from __future__ import annotations

import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SCSMDataset(Dataset):
    """Create the uploaded SCSM's two WTCM views directly from GTT23 HDF5."""

    def __init__(self, path: Path, config: dict, max_day: int | None):
        self.path = str(path)
        self.config = config
        self._file = None
        with h5py.File(path, "r") as source:
            day = np.asarray(source["day"])
            self.indices = np.flatnonzero(day <= max_day)
        if not len(self.indices):
            raise ValueError(f"no samples in {path} satisfy day <= {max_day}")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def _source(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r", swmr=True)
        return self._file

    def __len__(self) -> int:
        return len(self.indices)

    def _trace(self, logical_index: int) -> np.ndarray:
        source_index = int(self.indices[logical_index])
        source = self._source()
        length = int(source["lengths"][source_index])
        times = np.asarray(source["times"][source_index, :length], dtype=np.float64)
        directions = np.asarray(source["directions"][source_index, :length], dtype=np.float32)
        if not length:
            return np.empty((0, 2), dtype=np.float64)
        times -= times[0]
        times[0] = 1e-6
        return np.column_stack((times, directions))

    @staticmethod
    def _crop(trace: np.ndarray, duration: float) -> np.ndarray:
        if len(trace) == 0 or trace[-1, 0] <= duration:
            return trace
        start = random.uniform(0.0, max(0.0, trace[-1, 0] - duration))
        selected = trace[(trace[:, 0] >= start) & (trace[:, 0] <= start + duration)].copy()
        if len(selected):
            selected[:, 0] -= selected[0, 0]
            selected[0, 0] = 1e-6
        return selected

    @staticmethod
    def _merge(traces: list[np.ndarray], overlap: float) -> np.ndarray:
        merged, offset = [], 0.0
        for trace in traces:
            if not len(trace):
                continue
            current = trace.copy()
            current[:, 0] -= current[0, 0]
            if merged:
                offset -= overlap * min(merged[-1][-1, 0] - merged[-1][0, 0], current[-1, 0])
            current[:, 0] += max(offset, 0.0)
            merged.append(current)
            offset = current[-1, 0]
        if not merged:
            return np.empty((0, 2), dtype=np.float64)
        output = np.concatenate(merged)
        output = output[np.argsort(output[:, 0], kind="stable")]
        output[output[:, 0] == 0, 0] = 1e-6
        return output

    @staticmethod
    def _segment(trace: np.ndarray, split_num: int) -> np.ndarray:
        """Split a trace into packet-contiguous segments and shuffle the segments."""
        if len(trace) < 2 or split_num < 2:
            return trace.copy()
        split_num = min(split_num, len(trace))
        cut_points = np.sort(
            np.random.choice(np.arange(1, len(trace)), split_num - 1, replace=False)
        )
        segments = list(np.split(trace.copy(), cut_points))
        random.shuffle(segments)
        output, offset = [], 0.0
        for segment in segments:
            segment[:, 0] -= segment[0, 0]
            segment[:, 0] += offset
            output.append(segment)
            offset = segment[-1, 0]
        result = np.concatenate(output)
        result[result[:, 0] == 0, 0] = 1e-6
        return result

    def _augment_segment(self, trace: np.ndarray) -> np.ndarray:
        if not bool(self.config.get("segment_aug", False)):
            return trace
        return self._segment(trace, int(self.config.get("segment_aug_num")))

    def _combined_view(self, indices: list[int], maximum_load_time: float) -> np.ndarray:
        count = len(indices)
        traces = [self._augment_segment(self._trace(index)) for index in indices]
        overlap = random.uniform(-float(self.config.get("max_overlap_ratio", 0.5)),
                                 float(self.config.get("max_overlap_ratio", 0.5)))
        per_trace = maximum_load_time / (count - (count - 1) * overlap)
        traces = [self._crop(trace, per_trace) for trace in traces]
        random.shuffle(traces)
        return self._merge(traces, overlap)

    def _matrix(self, trace: np.ndarray, slot: float | None) -> tuple[torch.Tensor, torch.Tensor]:
        length = int(self.config.get("max_matrix_length"))
        maximum_cells = int(self.config.get("maximum_cell_number"))
        if slot is None:
            slot = float(self.config.get("maximum_load_time")) / length
        feature = np.zeros((2 * (maximum_cells + 2), length), dtype=np.float32)
        if len(trace):
            times, directions = trace[:, 0], np.sign(trace[:, 1]).astype(np.int64)
            columns = np.floor(times / slot).astype(np.int64).clip(0, length - 1)
            rows = (directions > 0).astype(np.int64)  # only-direction mode: cell-size group is zero
            np.add.at(feature, (rows, columns), 1)
            unique, starts = np.unique(columns, return_index=True)
            feature[2 * maximum_cells + 2, unique[1:]] = np.diff(unique)
            threshold = slot * float(self.config.get("time_interval_threshold"))
            for group, begin in enumerate(starts):
                end = starts[group + 1] if group + 1 < len(starts) else len(times)
                feature[2 * maximum_cells + 3, unique[group]] = 1 + np.sum(np.diff(times[begin:end]) > threshold)
            last_index = int(unique[-1])
        else:
            last_index = 0
        if bool(self.config.get("log_transform", True)):
            feature = np.log1p(feature)
        if bool(self.config.get("mask", True)):
            ratio = float(self.config.get("mask_ratio", 0.5))
            mask_count = int(length * ratio)
            masked = np.random.choice(length, mask_count, replace=False)
            feature[:, masked] = 0
        return torch.from_numpy(feature[None]), torch.tensor(last_index, dtype=torch.float32)

    def __getitem__(self, index: int):
        length = int(self.config.get("max_matrix_length"))
        if bool(self.config.get("adaptive_slot")):
            first_slot = random.uniform(float(self.config.get("min_slot_time")),
                                        float(self.config.get("max_slot_time")))
            second_slot = random.uniform(float(self.config.get("min_slot_time")),
                                         float(self.config.get("max_slot_time")))
            first_maximum_load_time = first_slot * length
            second_maximum_load_time = second_slot * length
        else:
            first_slot = second_slot = None
            first_maximum_load_time = second_maximum_load_time = float(
                self.config.get("maximum_load_time")
            )

        if bool(self.config.get("combine_aug")):
            minimum = int(self.config.get("combine_min"))
            maximum = int(self.config.get("combine_max"))
            count = random.randint(minimum, maximum)
            # Positive views must describe the same collection of traces.
            indices = [index]
            indices.extend(random.randrange(len(self)) for _ in range(count - 1))
            first_trace = self._combined_view(indices, first_maximum_load_time)
            second_trace = self._combined_view(indices, second_maximum_load_time)
        else:
            trace = self._trace(index)
            first_trace = self._augment_segment(trace)
            second_trace = self._augment_segment(trace)
        return self._matrix(first_trace, first_slot), self._matrix(second_trace, second_slot)
