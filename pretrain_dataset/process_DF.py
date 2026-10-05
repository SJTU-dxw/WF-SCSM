#!/usr/bin/env python3
"""Convert the DF Undefended traces to the GTT-style HDF5 format."""

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
FILENAME_PATTERN = re.compile(r"^(\d+)-(\d+)$")


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=base_dir / "raw-data-50-1000",
        help="directory containing DF trace files named <label>-<index>",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=base_dir / "DF_train.hdf5",
        help="output HDF5 path",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite the target if it already exists",
    )
    return parser.parse_args()


def trace_sort_key(path: Path) -> tuple[int, int]:
    match = FILENAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unexpected trace filename: {path}")
    return int(match.group(1)), int(match.group(2))


def collect_files(source_root: Path) -> list[Path]:
    paths = sorted(
        (path for path in source_root.iterdir() if path.is_file()),
        key=trace_sort_key,
    )
    if not paths:
        raise RuntimeError(f"no trace files found in {source_root}")
    labels = {trace_sort_key(path)[0] for path in paths}
    print(f"DF traces: {len(paths)}, websites: {len(labels)}")
    return paths


def create_target_file(target: h5py.File, total: int, source_root: Path) -> None:
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
    target.attrs["dataset"] = "DF Undefended"
    target.attrs["day_mapping"] = f"all traces={DAY}"
    target.attrs["packet_order"] = "stable timestamp sort"
    target.attrs["complete"] = False


def read_trace(path: Path) -> tuple[np.ndarray, np.ndarray, int, bytes]:
    trace = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if trace.shape[0] == 0:
        raise ValueError(f"{path}: empty trace")
    if trace.shape[1] < 2:
        raise ValueError(f"{path}: expected at least 2 columns, got {trace.shape[1]}")

    raw_times = trace[:, 0]
    raw_sizes = trace[:, 1]
    if not np.all(np.isfinite(raw_times)):
        raise ValueError(f"{path}: time contains NaN or inf")
    if not np.all(np.isfinite(raw_sizes)):
        raise ValueError(f"{path}: packet size contains NaN or inf")

    directions = np.sign(raw_sizes)
    if not np.all((directions == -1) | (directions == 1)):
        raise ValueError(f"{path}: packet size must be non-zero")

    # Stable sorting fixes timestamp regressions while retaining source order
    # among packets with identical timestamps. Apply the same permutation to
    # directions so that packets remain paired with their timestamps.
    order = np.argsort(raw_times, kind="stable")
    raw_times = raw_times[order]
    directions = directions[order]

    length = min(trace.shape[0], SEQUENCE_LENGTH)
    times = np.zeros(SEQUENCE_LENGTH, dtype=np.float32)
    padded_directions = np.zeros(SEQUENCE_LENGTH, dtype=np.int8)
    times[:length] = raw_times[:length]
    padded_directions[:length] = directions[:length].astype(np.int8, copy=False)

    label, _ = trace_sort_key(path)
    return times, padded_directions, length, str(label).encode("ascii")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    target_path = args.target.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {source_root}")

    paths = collect_files(source_root)
    if target_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"target already exists: {target_path}; pass --overwrite to replace it"
            )
        target_path.unlink()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(paths)
    print(f"Creating: {target_path}")
    print(f"Sequence length: {SEQUENCE_LENGTH}; day: {DAY}")
    started = time.time()

    with h5py.File(target_path, "w", libver="latest") as target:
        create_target_file(target, total, source_root)
        with tqdm(total=total, unit="trace", desc="Converting DF") as progress:
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
