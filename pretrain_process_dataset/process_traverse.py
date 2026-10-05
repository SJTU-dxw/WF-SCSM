#!/usr/bin/env python3
"""Convert GTT23 traces to burst prompts used by TraVerse."""

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

from feature_extract.feature_burst import fun


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def extract_burst_prompts(
    directions: np.ndarray,
    lengths: np.ndarray,
    max_bursts: int,
    template: str,
) -> np.ndarray:
    """Extract signed bursts and place them into the prompt template."""
    if directions.ndim != 2:
        raise ValueError("directions must be a two-dimensional array")
    if len(lengths) != len(directions):
        raise ValueError("directions and lengths have incompatible shapes")
    if "{burst feature}" not in template:
        raise ValueError("template must contain '{burst feature}'")

    batch_size, input_length = directions.shape
    prompts = np.empty(batch_size, dtype=object)

    for batch_index, raw_length in enumerate(lengths):
        valid_length = int(raw_length)
        if valid_length < 0 or valid_length > input_length:
            raise ValueError(
                f"Invalid trace length {valid_length} at batch row {batch_index}"
            )

        bursts = fun(directions[batch_index, :valid_length], max_bursts)
        burst_feature = ", ".join(str(value) for value in bursts)
        prompts[batch_index] = template.replace(
            "{burst feature}", burst_feature
        )

    return prompts


def process(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    dataset_config = config["dataset"]
    traverse_config = config["traverse"]
    input_path = resolve_path(dataset_config["input_path"])
    output_path = resolve_path(dataset_config["output_path"])
    batch_size = int(dataset_config["batch_size"])
    max_bursts = int(traverse_config["max_bursts"])
    template = str(traverse_config["template"])

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_bursts <= 0:
        raise ValueError("max_bursts must be positive")

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

        offsets = np.lib.format.open_memmap(
            output_path / "offsets.npy", mode="w+", dtype=np.uint64,
            shape=(total + 1,),
        )
        labels = np.lib.format.open_memmap(
            output_path / "label.npy", mode="w+", dtype=source["labels"].dtype,
            shape=(total,),
        )
        days = np.lib.format.open_memmap(
            output_path / "day.npy", mode="w+", dtype=source["day"].dtype,
            shape=(total,),
        )
        offsets[0] = 0

        progress = tqdm(total=total, unit="trace", desc="Extracting TraVerse bursts")
        with (output_path / "dataset.bin").open("wb") as prompt_stream:
            byte_offset = 0
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                prompts = extract_burst_prompts(
                    source["directions"][start:end],
                    source["lengths"][start:end],
                    max_bursts=max_bursts,
                    template=template,
                )
                for index, prompt in enumerate(prompts, start=start):
                    encoded = str(prompt).encode("utf-8")
                    prompt_stream.write(encoded)
                    byte_offset += len(encoded)
                    offsets[index + 1] = byte_offset
                labels[start:end] = source["labels"][start:end]
                days[start:end] = source["day"][start:end]
                progress.update(end - start)
        progress.close()
        for array in (offsets, labels, days):
            array.flush()

    print(f"Saved {total} TraVerse prompts to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default="pretrain_process_dataset/config_pretrain_dataset/traverse.yaml",
    )
    args = parser.parse_args()
    process(args.config.resolve())


if __name__ == "__main__":
    main()
