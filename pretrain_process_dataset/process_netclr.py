#!/usr/bin/env python3
"""Convert GTT23 traces to fixed-length direction sequences for NetCLR."""

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

from feature_extract.feature_length import fun


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def extract_direction_features(
    directions: np.ndarray,
    lengths: np.ndarray,
    output_length: int,
    feature_dtype: np.dtype,
) -> np.ndarray:
    """Truncate direction sequences and pad shorter sequences with zeros."""
    if directions.ndim != 2:
        raise ValueError("directions must be a two-dimensional array")
    if len(lengths) != len(directions):
        raise ValueError("directions and lengths have incompatible shapes")
    if output_length <= 0:
        raise ValueError("sequence_length must be positive")

    batch_size, input_length = directions.shape
    features = np.zeros((batch_size, output_length), dtype=feature_dtype)

    for batch_index, raw_length in enumerate(lengths):
        valid_length = int(raw_length)
        if valid_length < 0 or valid_length > input_length:
            raise ValueError(
                f"Invalid trace length {valid_length} at batch row {batch_index}"
            )

        trace = directions[batch_index, :valid_length]
        output_trace = fun(trace, output_length)
        features[batch_index] = output_trace

    return features


def process(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    dataset_config = config["dataset"]
    netclr_config = config["netclr"]
    input_path = resolve_path(dataset_config["input_path"])
    output_path = resolve_path(dataset_config["output_path"])
    batch_size = int(dataset_config["batch_size"])
    output_length = int(netclr_config["sequence_length"])
    feature_dtype = np.dtype(netclr_config["feature_dtype"])

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not np.issubdtype(feature_dtype, np.signedinteger):
        raise ValueError("feature_dtype must be a signed integer dtype")

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
        if source["directions"].ndim != 2:
            raise ValueError("directions must be a two-dimensional dataset")
        if source["directions"].shape[0] != total or len(source["labels"]) != total:
            raise ValueError("directions, lengths, and labels have incompatible shapes")
        if "day" not in source:
            raise ValueError(
                "Input dataset does not contain 'day'; regenerate it with "
                "pretrain_dataset/process_GTT.py"
            )
        if len(source["day"]) != total:
            raise ValueError("day and lengths have incompatible shapes")

        features = np.lib.format.open_memmap(
            output_path / "dataset.npy", mode="w+", dtype=feature_dtype,
            shape=(total, output_length),
        )
        labels = np.lib.format.open_memmap(
            output_path / "label.npy", mode="w+", dtype=source["labels"].dtype,
            shape=(total,),
        )
        days = np.lib.format.open_memmap(
            output_path / "day.npy", mode="w+", dtype=source["day"].dtype,
            shape=(total,),
        )

        progress = tqdm(total=total, unit="trace", desc="Extracting NetCLR directions")
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            features[start:end] = extract_direction_features(
                source["directions"][start:end],
                source["lengths"][start:end],
                output_length=output_length,
                feature_dtype=feature_dtype,
            )
            labels[start:end] = source["labels"][start:end]
            days[start:end] = source["day"][start:end]
            progress.update(end - start)
        progress.close()
        for array in (features, labels, days):
            array.flush()

    print(f"Saved {total} traces with shape {features.shape} to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default="pretrain_process_dataset/config_pretrain_dataset/netclr.yaml",
    )
    args = parser.parse_args()
    process(args.config.resolve())


if __name__ == "__main__":
    main()
