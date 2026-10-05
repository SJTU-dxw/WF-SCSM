#!/usr/bin/env python3
"""Merge one ARES multi-tab dataset into the GTT-style HDF5 format."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import BinaryIO, Iterator
from zipfile import ZipFile

import h5py
import numpy as np
from numpy.lib import format as npy_format
from tqdm import tqdm


SEQUENCE_LENGTH = 10000
BATCH_SIZE = 64
CHUNK_ROWS = 64
DAY = 56
SOURCE_FILES = ("train.npz", "valid.npz", "test.npz")
SPLIT_CODES = {"train": 0, "valid": 1, "test": 2}
DATASET_PATTERN = re.compile(r"^(Closed|Open)_([2-5])tab$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="one ARES dataset directory containing train/valid/test.npz",
    )
    parser.add_argument("--target", type=Path, required=True, help="output HDF5 path")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite the target if it already exists",
    )
    return parser.parse_args()


def read_npy_header(stream: BinaryIO) -> tuple[tuple[int, ...], bool, np.dtype]:
    version = npy_format.read_magic(stream)
    if version == (1, 0):
        return npy_format.read_array_header_1_0(stream)
    if version in {(2, 0), (3, 0)}:
        return npy_format.read_array_header_2_0(stream)
    raise ValueError(f"unsupported NPY format version: {version}")


def inspect_npz(
    path: Path, expected_labels_per_trace: int
) -> tuple[int, int, np.dtype, int]:
    with ZipFile(path) as archive:
        members = set(archive.namelist())
        missing = {"X.npy", "y.npy"} - members
        if missing:
            raise ValueError(f"{path}: missing arrays: {sorted(missing)}")
        with archive.open("X.npy") as stream:
            shape, fortran_order, dtype = read_npy_header(stream)
        with archive.open("y.npy") as stream:
            labels = np.load(stream, allow_pickle=False)

    if len(shape) != 3 or shape[2] != 2:
        raise ValueError(f"{path}: expected X shape (N, L, 2), got {shape}")
    if fortran_order:
        raise ValueError(f"{path}: Fortran-order X arrays are not supported")
    if not np.issubdtype(dtype, np.number):
        raise ValueError(f"{path}: X must be numeric, got {dtype}")
    if labels.ndim != 2 or labels.shape[0] != shape[0]:
        raise ValueError(
            f"{path}: expected two-dimensional y with {shape[0]} rows, got {labels.shape}"
        )
    if labels.dtype != np.uint8 or np.any((labels != 0) & (labels != 1)):
        raise ValueError(f"{path}: y must be a uint8 multi-hot matrix")
    label_counts = labels.sum(axis=1)
    if np.any(label_counts != expected_labels_per_trace):
        observed = np.unique(label_counts).tolist()
        raise ValueError(
            f"{path}: expected {expected_labels_per_trace} active labels per trace, "
            f"got {observed}"
        )
    return shape[0], shape[1], dtype, labels.shape[1]


def collect_archives(
    source: Path, expected_labels_per_trace: int
) -> list[tuple[Path, int, int, np.dtype, int, int]]:
    expected_paths = [source / name for name in SOURCE_FILES]
    missing = [path.name for path in expected_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{source}: missing required files: {missing}")
    unexpected = sorted(
        path.name
        for path in source.glob("*.npz")
        if path.name not in SOURCE_FILES
    )
    if unexpected:
        raise ValueError(f"{source}: unexpected NPZ files: {unexpected}")

    archives = []
    for path in expected_paths:
        rows, width, dtype, label_width = inspect_npz(
            path, expected_labels_per_trace
        )
        split = path.stem
        archives.append(
            (path, rows, width, dtype, label_width, SPLIT_CODES[split])
        )
        print(
            f"{path.name}: samples={rows}, trace_width={width}, "
            f"labels={label_width}, dtype={dtype}"
        )

    label_widths = {item[4] for item in archives}
    if len(label_widths) != 1:
        raise ValueError(f"label width differs across splits: {sorted(label_widths)}")
    return archives


def read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"unexpected end of X.npy; needed {remaining} more bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def iter_npz_batches(
    path: Path,
    expected_rows: int,
    expected_width: int,
    expected_dtype: np.dtype,
    expected_label_width: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    with ZipFile(path) as archive:
        with archive.open("y.npy") as stream:
            labels = np.load(stream, allow_pickle=False)
        if labels.shape != (expected_rows, expected_label_width):
            raise ValueError(f"{path}: y shape changed while reading: {labels.shape}")

        with archive.open("X.npy") as stream:
            shape, fortran_order, dtype = read_npy_header(stream)
            if shape != (expected_rows, expected_width, 2):
                raise ValueError(f"{path}: X shape changed while reading: {shape}")
            if fortran_order or dtype != expected_dtype:
                raise ValueError(f"{path}: X metadata changed while reading")

            row_bytes = expected_width * 2 * dtype.itemsize
            for start in range(0, expected_rows, BATCH_SIZE):
                count = min(BATCH_SIZE, expected_rows - start)
                raw = read_exact(stream, count * row_bytes)
                traces = np.frombuffer(raw, dtype=dtype).reshape(
                    count, expected_width, 2
                )
                yield traces, labels[start : start + count]


def process_batch(traces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_times = traces[:, :, 0]
    raw_sizes = traces[:, :, 1]
    if not np.all(np.isfinite(raw_times)):
        raise ValueError("time contains NaN or inf")
    if not np.all(np.isfinite(raw_sizes)):
        raise ValueError("packet size contains NaN or inf")

    valid = raw_sizes != 0
    lengths = np.minimum(valid.sum(axis=1), SEQUENCE_LENGTH).astype(np.uint16)
    sort_times = np.where(valid, raw_times, np.inf)
    order = np.argsort(sort_times, axis=1, kind="stable")
    sorted_times = np.take_along_axis(raw_times, order, axis=1).astype(
        np.float32, copy=False
    )
    sorted_sizes = np.take_along_axis(raw_sizes, order, axis=1)

    source_columns = np.arange(traces.shape[1])[None, :]
    valid_after_first = (source_columns > 0) & (source_columns < lengths[:, None])
    first_times = sorted_times[:, :1]
    duplicate_start = valid_after_first & (sorted_times == first_times)
    sorted_times = np.where(
        duplicate_start,
        np.nextafter(first_times, np.float32(np.inf)),
        sorted_times,
    )
    if np.any(valid_after_first & (sorted_times == sorted_times[:, :1])):
        raise RuntimeError("valid packets still contain duplicate start timestamps")

    count = len(traces)
    times = np.zeros((count, SEQUENCE_LENGTH), dtype=np.float32)
    directions = np.zeros((count, SEQUENCE_LENGTH), dtype=np.int8)
    copy_width = min(traces.shape[1], SEQUENCE_LENGTH)
    times[:, :copy_width] = sorted_times[:, :copy_width]
    directions[:, :copy_width] = np.sign(sorted_sizes[:, :copy_width]).astype(
        np.int8
    )

    columns = np.arange(SEQUENCE_LENGTH)[None, :]
    padding = columns >= lengths[:, None]
    times[padding] = 0
    directions[padding] = 0
    return times, directions, lengths


def create_target_file(
    target: h5py.File,
    total: int,
    label_width: int,
    source: Path,
    files: list[Path],
    dataset_kind: str,
    tabs: int,
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
        "labels",
        shape=(total, label_width),
        dtype=np.uint8,
        chunks=(matrix_chunk_rows, label_width),
        compression="lzf",
        shuffle=True,
    )
    for name, dtype, shuffle in (
        ("lengths", np.uint16, True),
        ("day", np.uint8, False),
        ("split", np.uint8, False),
    ):
        target.create_dataset(
            name,
            shape=(total,),
            dtype=dtype,
            chunks=(vector_chunk_rows,),
            compression="lzf",
            shuffle=shuffle,
        )

    target.attrs["total_rows"] = total
    target.attrs["sequence_length"] = SEQUENCE_LENGTH
    target.attrs["source_root"] = str(source)
    target.attrs["source_files"] = ",".join(path.name for path in files)
    target.attrs["day_mapping"] = f"all traces={DAY}"
    target.attrs["packet_order"] = "valid packets, stable timestamp sort"
    target.attrs["label_format"] = "multi-hot"
    target.attrs["label_count"] = label_width
    target.attrs["labels_per_trace"] = tabs
    target.attrs["dataset_kind"] = dataset_kind.lower()
    target.attrs["open_world_label"] = label_width - 1 if dataset_kind == "Open" else -1
    target.attrs["split_mapping"] = "train=0,valid=1,test=2"
    target.attrs["complete"] = False


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    target_path = args.target.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source}")

    match = DATASET_PATTERN.fullmatch(source.name)
    if match is None:
        raise ValueError(
            "source directory name must match Closed_[2-5]tab or Open_[2-5]tab"
        )
    dataset_kind, tabs_text = match.groups()
    tabs = int(tabs_text)
    archives = collect_archives(source, tabs)
    total = sum(item[1] for item in archives)
    label_width = archives[0][4]
    expected_label_width = 100 if dataset_kind == "Closed" else 101
    if label_width != expected_label_width:
        raise ValueError(
            f"{source}: expected {expected_label_width} label columns for "
            f"{dataset_kind}, got {label_width}"
        )

    if target_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"target already exists: {target_path}; pass --overwrite to replace it"
            )
        target_path.unlink()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Total samples: {total}")
    print(f"Creating: {target_path}")
    print(
        f"Sequence length: {SEQUENCE_LENGTH}; labels: multi-hot ({label_width}); "
        f"tabs: {tabs}; day: {DAY}"
    )
    started = time.time()
    written = 0

    with h5py.File(target_path, "w", libver="latest") as target:
        create_target_file(
            target,
            total,
            label_width,
            source,
            [item[0] for item in archives],
            dataset_kind,
            tabs,
        )
        with tqdm(total=total, unit="trace", desc="Merging NPZ") as progress:
            for path, rows, width, dtype, _, split_code in archives:
                for traces, labels in iter_npz_batches(
                    path, rows, width, dtype, label_width
                ):
                    times, directions, lengths = process_batch(traces)
                    count = len(traces)
                    end = written + count
                    target["times"][written:end] = times
                    target["directions"][written:end] = directions
                    target["lengths"][written:end] = lengths
                    target["labels"][written:end] = labels
                    target["day"][written:end] = DAY
                    target["split"][written:end] = split_code
                    written = end
                    progress.update(count)

        if written != total:
            raise RuntimeError(f"wrote {written} rows, expected {total}")
        target.attrs["complete"] = True
        target.flush()

    elapsed = time.time() - started
    print(f"Done in {elapsed / 3600:.2f} hours")
    print(f"Saved to: {target_path}")


if __name__ == "__main__":
    main()
