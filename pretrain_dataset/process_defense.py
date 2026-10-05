#!/usr/bin/env python3
"""Convert one flat DF defense trace directory to training HDF5."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


SEQUENCE_LENGTH = 10000
BATCH_SIZE = 1024
CHUNK_ROWS = 64
DAY = 56
TRACE_PATTERN = re.compile(r"^(\d+)-(\d+)$")
SPLIT_TRACE_PATTERN = re.compile(r"^(\d+)-(\d+)_split_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="flat directory containing defended trace files",
    )
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="output HDF5 path",
    )
    parser.add_argument(
        "--defense-name",
        required=True,
        help="defense name stored in the HDF5 metadata",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite the target if it already exists",
    )
    return parser.parse_args()


def parse_trace_name(path: Path) -> tuple[int, int, int]:
    match = TRACE_PATTERN.fullmatch(path.name)
    if match is not None:
        return int(match.group(1)), int(match.group(2)), -1

    match = SPLIT_TRACE_PATTERN.fullmatch(path.name)
    if match is not None:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))

    raise ValueError(f"unexpected trace filename: {path}")


def collect_files(source_root: Path) -> tuple[list[Path], int]:
    paths: list[Path] = []
    skipped_empty = 0

    for path in source_root.iterdir():
        if not path.is_file():
            continue
        parse_trace_name(path)
        if path.stat().st_size == 0:
            skipped_empty += 1
            continue
        paths.append(path)

    paths.sort(key=parse_trace_name)
    if not paths:
        raise RuntimeError(f"no non-empty trace files found in {source_root}")

    labels = {parse_trace_name(path)[0] for path in paths}
    split_traces = sum(parse_trace_name(path)[2] >= 0 for path in paths)
    print(
        f"Defense traces: {len(paths)}, websites: {len(labels)}, "
        f"split traces: {split_traces}, skipped empty: {skipped_empty}"
    )
    return paths, skipped_empty


def create_target_file(
    target: h5py.File,
    total: int,
    source_root: Path,
    defense_name: str,
    skipped_empty: int,
) -> None:
    matrix_chunk_rows = min(CHUNK_ROWS, total)
    vector_chunk_rows = min(8192, total)
    target.create_dataset(
        "times",
        shape=(total, SEQUENCE_LENGTH),
        dtype=np.float32,
        chunks=(matrix_chunk_rows, SEQUENCE_LENGTH),
        compression="lzf",
        shuffle=True,
    )
    target.create_dataset(
        "directions",
        shape=(total, SEQUENCE_LENGTH),
        dtype=np.int8,
        chunks=(matrix_chunk_rows, SEQUENCE_LENGTH),
        compression="lzf",
        shuffle=True,
    )
    target.create_dataset(
        "lengths",
        shape=(total,),
        dtype=np.uint16,
        chunks=(vector_chunk_rows,),
        compression="lzf",
        shuffle=True,
    )
    target.create_dataset(
        "labels",
        shape=(total,),
        dtype="S44",
        chunks=(vector_chunk_rows,),
        compression="lzf",
    )
    target.create_dataset(
        "day",
        shape=(total,),
        dtype=np.uint8,
        chunks=(vector_chunk_rows,),
        compression="lzf",
    )
    target.attrs["total_rows"] = total
    target.attrs["sequence_length"] = SEQUENCE_LENGTH
    target.attrs["source_root"] = str(source_root)
    target.attrs["dataset"] = f"DF Defense ({defense_name})"
    target.attrs["defense"] = defense_name
    target.attrs["day_mapping"] = f"all traces={DAY}"
    target.attrs["packet_order"] = "stable timestamp sort"
    target.attrs["traffic_silver_split_policy"] = (
        "each non-empty _split_N file is an independent sample"
    )
    target.attrs["skipped_empty_traces"] = skipped_empty
    target.attrs["complete"] = False


def read_trace(path: Path) -> tuple[np.ndarray, np.ndarray, int, bytes]:
    trace = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if trace.shape[0] == 0 or trace.shape[1] < 2:
        raise ValueError(f"{path}: expected a non-empty trace with at least 2 columns")

    raw_times = trace[:, 0]
    raw_sizes = trace[:, 1]
    if not np.all(np.isfinite(raw_times)):
        raise ValueError(f"{path}: time contains NaN or inf")
    if not np.all(np.isfinite(raw_sizes)):
        raise ValueError(f"{path}: packet size contains NaN or inf")

    directions = np.sign(raw_sizes)
    if not np.all((directions == -1) | (directions == 1)):
        raise ValueError(f"{path}: packet size must be non-zero")

    order = np.argsort(raw_times, kind="stable")
    raw_times = raw_times[order]
    directions = directions[order]

    length = min(trace.shape[0], SEQUENCE_LENGTH)
    times = np.zeros(SEQUENCE_LENGTH, dtype=np.float32)
    padded_directions = np.zeros(SEQUENCE_LENGTH, dtype=np.int8)
    times[:length] = raw_times[:length]
    padded_directions[:length] = directions[:length].astype(np.int8, copy=False)

    label, _, _ = parse_trace_name(path)
    return times, padded_directions, length, str(label).encode("ascii")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    target_path = args.target.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {source_root}")

    paths, skipped_empty = collect_files(source_root)
    if target_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"target already exists: {target_path}; pass --overwrite to replace it"
            )
        target_path.unlink()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(paths)
    print(f"Creating: {target_path}")
    print(f"Defense: {args.defense_name}; sequence length: {SEQUENCE_LENGTH}; day: {DAY}")
    started = time.time()

    with h5py.File(target_path, "w", libver="latest") as target:
        create_target_file(
            target,
            total,
            source_root,
            args.defense_name,
            skipped_empty,
        )
        with tqdm(total=total, unit="trace", desc="Converting defense") as progress:
            for start in range(0, total, BATCH_SIZE):
                batch = paths[start : start + BATCH_SIZE]
                count = len(batch)
                times = np.zeros((count, SEQUENCE_LENGTH), dtype=np.float32)
                directions = np.zeros((count, SEQUENCE_LENGTH), dtype=np.int8)
                lengths = np.empty(count, dtype=np.uint16)
                labels = np.empty(count, dtype="S44")

                for row, path in enumerate(batch):
                    times[row], directions[row], lengths[row], labels[row] = read_trace(path)

                end = start + count
                target["times"][start:end] = times
                target["directions"][start:end] = directions
                target["lengths"][start:end] = lengths
                target["labels"][start:end] = labels
                target["day"][start:end] = DAY
                progress.update(count)

        target.attrs["complete"] = True
        target.flush()

    elapsed = time.time() - started
    print(f"Done in {elapsed / 3600:.2f} hours")
    print(f"Saved to: {target_path}")


if __name__ == "__main__":
    main()
