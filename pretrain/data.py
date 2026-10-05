from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from augment.netclr_augmentor import Augmentor as NetCLRAugmentor
from augment.netclr_augmentor import find_bursts as find_netclr_bursts
from augment.swallow_augmentor import Augmentor


def _load(path: Path) -> dict:
    # 加载NetCLR和Swallow数据集
    dataset_path = path / "dataset.npy"
    result = {"dataset": np.load(dataset_path, mmap_mode="r", allow_pickle=False)}
    for name in ("label", "day", "load_time", "slot_duration"):
        field = path / f"{name}.npy"
        if field.exists():
            result[name] = np.load(field, mmap_mode="r", allow_pickle=False)
    return result


def _day_selection(day: np.ndarray, sample_count: int, max_day: int) -> tuple[np.ndarray, int]:
    """Build a logical-to-physical index without modifying the stored arrays."""
    if len(day) != sample_count:
        raise ValueError(f"day has {len(day)} rows, but dataset has {sample_count}")

    indices = np.flatnonzero(np.asarray(day) <= max_day)
    if not len(indices):
        raise ValueError(f"no samples satisfy day <= {max_day}")

    return indices, len(indices)


class DayFilteredDataset(Dataset):
    _indices: np.ndarray
    _length: int

    def _configure_day_filter(self, day: np.ndarray, sample_count: int, max_day: int) -> None:
        self._indices, self._length = _day_selection(day, sample_count, max_day)

    def __len__(self) -> int:
        return self._length

    def _source_index(self, index: int) -> int:
        return int(self._indices[index])


class DirectionDataset(DayFilteredDataset):
    def __init__(self, path: Path, augmentation: dict, max_day: int | None = None):
        data = _load(path)
        self.features = data["dataset"]
        self._configure_day_filter(data.get("day"), len(self.features), max_day)
        sample_count = min(int(augmentation.get("cdf_sample_size", 1000)), self._length)
        sampled_logical = np.random.choice(self._length, size=sample_count, replace=False)
        sampled_source = self._indices[sampled_logical]
        outgoing = []
        for source_index in sampled_source:
            bursts = find_netclr_bursts(self.features[int(source_index)])
            outgoing.extend(burst[2] for burst in bursts if burst[2] > 0)
        if not outgoing:
            raise ValueError("cannot build NetAugment CDF: sampled traces contain no outgoing bursts")
        maximum = int(max(outgoing))
        counts = np.bincount(outgoing, minlength=maximum + 1)[1:]
        cdf = np.cumsum(counts, dtype=np.float64)
        cdf /= cdf[-1]
        self.augmentor = NetCLRAugmentor(
            max_outgoing_burst_size=maximum,
            outgoing_burst_sizes=outgoing,
            OUTGOING_BURST_SIZE_CDF=cdf,
            output_length=int(self.features.shape[-1]),
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.features[self._source_index(index)]
        return self._augment(source), self._augment(source)

    def _augment(self, source: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(self.augmentor.augment(source))


class CIFDataset(DayFilteredDataset):
    def __init__(self, path: Path, augmentation: dict, max_day: int | None = None):
        data = _load(path)
        self.features = data["dataset"]
        self.load_times = data.get("load_time")
        self.slot_durations = data.get("slot_duration")
        if self.load_times is None or self.slot_durations is None:
            raise ValueError(f"{path} must contain load_time.npy and slot_duration.npy")
        if len(self.load_times) != len(self.features) or len(self.slot_durations) != len(self.features):
            raise ValueError("CIF features, load times, and slot durations have incompatible lengths")
        self.augmentor = Augmentor()
        self._configure_day_filter(data.get("day"), len(self.features), max_day)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source_index = self._source_index(index)
        source = np.array(self.features[source_index], copy=True)
        load_time = float(self.load_times[source_index])
        slot_duration = float(self.slot_durations[source_index])

        def augment() -> torch.Tensor:
            view = self.augmentor.augment(source.copy(), load_time, slot_duration)
            if view.shape != source.shape:
                raise ValueError(f"Swallow augmentation changed CIF shape from {source.shape} to {view.shape}")
            return torch.from_numpy(np.asarray(view, dtype=np.float32))

        return augment(), augment()


class PromptDataset(DayFilteredDataset):
    def __init__(self, path: Path, max_day: int | None = None):
        offsets_path = path / "offsets.npy"
        if not offsets_path.exists():
            raise ValueError(f"{path} does not contain offsets.npy")
        self.byte_data = np.memmap(path / "dataset.bin", mode="r", dtype=np.uint8)
        self.offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
        day_path = path / "day.npy"
        day = np.load(day_path, mmap_mode="r", allow_pickle=False)
        self._configure_day_filter(day, len(self.offsets) - 1, max_day)

    def __getitem__(self, index: int) -> str:
        index = self._source_index(index)
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return self.byte_data[start:end].tobytes().decode("utf-8")


class MaskedPromptCollator:
    def __init__(self, tokenizer, max_length: int, mask_ratio: float):
        self.tokenizer, self.max_length, self.mask_ratio = tokenizer, max_length, mask_ratio
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between 0 and 1")
        if tokenizer.mask_token_id is None:
            tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    @staticmethod
    def _random_mask(candidates: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """Mask a fixed fraction per sample using MAE-style random shuffling."""
        masked = torch.zeros_like(candidates)
        for row_index, row in enumerate(candidates):
            positions = torch.nonzero(row, as_tuple=False).flatten()
            token_count = positions.numel()
            if token_count == 0:
                raise ValueError("prompt contains no maskable tokens")
            keep_count = int(token_count * (1.0 - mask_ratio))
            order = torch.argsort(torch.rand(token_count))
            masked[row_index, positions[order[keep_count:]]] = True
        return masked

    def __call__(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer(prompts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        original = batch["input_ids"].clone()
        special = torch.tensor([self.tokenizer.get_special_tokens_mask(row, already_has_special_tokens=True) for row in original.tolist()], dtype=torch.bool)
        candidates = batch["attention_mask"].bool() & ~special
        masked = self._random_mask(candidates, self.mask_ratio)
        batch["input_ids"][masked] = self.tokenizer.mask_token_id
        labels = torch.full_like(original, -100)
        labels[masked] = original[masked]
        batch["labels"] = labels
        return batch
