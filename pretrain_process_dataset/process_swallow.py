#!/usr/bin/env python3
"""Convert GTT23 traces to the pre-training format used by Swallow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import yaml
from tqdm import tqdm

from feature_extract.feature_swallow import fun


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def extract_cif_features(
    times: np.ndarray,
    directions: np.ndarray,
    lengths: np.ndarray,
    slot_count: int,
    window_multiplier: float,
    min_slot_duration: float,
    max_slot_duration: float,
    feature_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized equivalent of Swallow's CIF.fun for one HDF5 batch."""
    batch_size, sequence_length = times.shape

    counts = np.zeros(
        (batch_size, 2, slot_count),
        dtype=np.int64,
    )
    load_times = np.zeros(batch_size, dtype=np.float32)
    slot_durations = np.zeros(batch_size, dtype=np.float32)

    for batch_index in range(batch_size):
        valid_length = int(lengths[batch_index])

        trace_times = np.asarray(
            times[batch_index, :valid_length],
            dtype=np.float64,
        )
        trace_directions = np.asarray(
            directions[batch_index, :valid_length]
        )

        feature, load_time, slot_duration = fun(
            times=trace_times,
            sizes=trace_directions,
            min_slot_duration=min_slot_duration,
            max_slot_duration=max_slot_duration,
            time_window_multiplier=window_multiplier,
            slot_len=slot_count,
        )

        counts[batch_index] = np.asarray(feature, dtype=np.int64)
        load_times[batch_index] = load_time
        slot_durations[batch_index] = slot_duration


    max_value = np.iinfo(feature_dtype).max
    min_value = np.iinfo(feature_dtype).min
    count_min = counts.min(initial=0)
    count_max = counts.max(initial=0)
    if count_min < min_value or count_max > max_value:
        raise OverflowError(
            f"CIF count range [{count_min}, {count_max}] cannot be "
            f"represented by {feature_dtype}; choose a wider dtype "
            "in config/swallow.yaml"
        )
    return counts.astype(feature_dtype), load_times, slot_durations


def process(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    dataset_config = config["dataset"]
    cif_config = config["cif"]
    input_path = resolve_path(dataset_config["input_path"])
    output_path = resolve_path(dataset_config["output_path"])
    batch_size = int(dataset_config["batch_size"])
    slot_count = int(cif_config["slot_count"])
    feature_dtype = np.dtype(cif_config["feature_dtype"])

    if output_path.exists():
        if not output_path.is_dir():
            raise FileExistsError(
                f"output_path must be a directory, but a file already exists: {output_path}"
            )
        if any(output_path.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite non-empty output directory: {output_path}"
            )
    else:
        output_path.mkdir(parents=True)

    with h5py.File(input_path, "r") as source:
        total = len(source["lengths"])
        sequence_length = source["times"].shape[1]
        if source["directions"].shape != (total, sequence_length):
            raise ValueError("times and directions have incompatible shapes")
        if len(source["labels"]) != total:
            raise ValueError("labels and lengths have incompatible shapes")
        if "day" not in source:
            raise ValueError(
                "Input dataset does not contain 'day'; regenerate it with "
                "pretrain_dataset/process_GTT.py"
            )
        if len(source["day"]) != total:
            raise ValueError("day and lengths have incompatible shapes")

        # Write each batch directly to disk. The resulting .npy files can be
        # memory-mapped by pretrain/data.py without loading the full dataset.
        features = np.lib.format.open_memmap(
            output_path / "dataset.npy",
            mode="w+",
            dtype=feature_dtype,
            shape=(total, 2, slot_count),
        )
        load_times = np.lib.format.open_memmap(
            output_path / "load_time.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total,),
        )
        slot_durations = np.lib.format.open_memmap(
            output_path / "slot_duration.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total,),
        )
        labels = np.lib.format.open_memmap(
            output_path / "label.npy",
            mode="w+",
            dtype=source["labels"].dtype,
            shape=(total,),
        )
        days = np.lib.format.open_memmap(
            output_path / "day.npy",
            mode="w+",
            dtype=source["day"].dtype,
            shape=(total,),
        )

        progress = tqdm(total=total, unit="trace", desc="Extracting Swallow CIF")
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            lengths = source["lengths"][start:end]
            if np.any(lengths > sequence_length):
                raise ValueError(f"Trace length exceeds sequence size near row {start}")

            batch = extract_cif_features(
                source["times"][start:end],
                source["directions"][start:end],
                lengths,
                slot_count=slot_count,
                window_multiplier=float(cif_config["time_window_multiplier"]),
                min_slot_duration=float(cif_config["min_slot_duration"]),
                max_slot_duration=float(cif_config["max_slot_duration"]),
                feature_dtype=feature_dtype,
            )
            features[start:end], load_times[start:end], slot_durations[start:end] = batch
            labels[start:end] = source["labels"][start:end]
            days[start:end] = source["day"][start:end]
            progress.update(end - start)
        progress.close()
        for array in (features, labels, days, load_times, slot_durations):
            array.flush()

    print(f"Saved {total} traces with shape {features.shape} to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default="pretrain_process_dataset/config_pretrain_dataset/swallow_single.yaml")
    args = parser.parse_args()
    process(args.config.resolve())


if __name__ == "__main__":
    main()
