#!/usr/bin/env python3
"""Merge NPZ trace datasets into the GTT-style HDF5 training format."""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default="pretrain_dataset/N_Finetune_R-Precision",
        help="directory containing the NPZ files to merge",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default="pretrain_dataset/R-Precision_train.hdf5",
        help="output HDF5 path",
    )
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


def inspect_npz(path: Path) -> tuple[int, int, np.dtype]:
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
    if labels.ndim != 1 or len(labels) != shape[0]:
        raise ValueError(
            f"{path}: y shape {labels.shape} does not match X rows {shape[0]}"
        )
    return shape[0], shape[1], dtype


def collect_archives(source: Path) -> list[tuple[Path, int, int, np.dtype]]:
    paths = sorted(path for path in source.iterdir() if path.is_file() and path.suffix == ".npz")
    if not paths:
        raise RuntimeError(f"no .npz files found in {source}")
    archives = [(path, *inspect_npz(path)) for path in paths]
    for path, rows, width, dtype in archives:
        print(f"{path.name}: samples={rows}, trace_width={width}, dtype={dtype}")
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
    path: Path, expected_rows: int, expected_width: int, expected_dtype: np.dtype
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    with ZipFile(path) as archive:
        with archive.open("y.npy") as stream:
            labels = np.load(stream, allow_pickle=False)
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
                traces = np.frombuffer(raw, dtype=dtype).reshape(count, expected_width, 2)
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
    # Padding timestamps are often zero. Move padding behind all real packets,
    # then stably order valid packets by timestamp.
    sort_times = np.where(valid, raw_times, np.inf)
    order = np.argsort(sort_times, axis=1, kind="stable")
    sorted_times = np.take_along_axis(raw_times, order, axis=1).astype(
        np.float32, copy=False
    )
    sorted_sizes = np.take_along_axis(raw_sizes, order, axis=1)

    # A packet after the first one may legitimately have the same timestamp as
    # the first packet.  Downstream time-aware features subtract the first
    # timestamp and combine elapsed time with direction, which would turn such
    # packets into internal zeros.  Preserve the packets and their order while
    # moving only those timestamps by the smallest representable float32 step.
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
    directions[:, :copy_width] = np.sign(sorted_sizes[:, :copy_width]).astype(np.int8)

    columns = np.arange(SEQUENCE_LENGTH)[None, :]
    padding = columns >= lengths[:, None]
    times[padding] = 0
    directions[padding] = 0
    return times, directions, lengths


def encode_labels(labels: np.ndarray, source: Path) -> np.ndarray:
    encoded = np.asarray([str(value).encode("utf-8") for value in labels], dtype="S44")
    if any(len(str(value).encode("utf-8")) > 44 for value in labels):
        raise ValueError(f"{source}: a label exceeds the 44-byte HDF5 limit")
    return encoded


def create_target_file(target: h5py.File, total: int, source: Path, files: list[Path]) -> None:
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
    for name, dtype, shuffle in (
        ("lengths", np.uint16, True),
        ("labels", "S44", False),
        ("day", np.uint8, False),
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
    target.attrs["complete"] = False


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    target_path = args.target.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source}")

    archives = collect_archives(source)
    total = sum(rows for _, rows, _, _ in archives)
    if target_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"target already exists: {target_path}; pass --overwrite to replace it"
            )
        target_path.unlink()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Total samples: {total}")
    print(f"Creating: {target_path}")
    print(f"Sequence length: {SEQUENCE_LENGTH}; day: {DAY}")
    started = time.time()
    written = 0

    with h5py.File(target_path, "w", libver="latest") as target:
        create_target_file(target, total, source, [item[0] for item in archives])
        with tqdm(total=total, unit="trace", desc="Merging NPZ") as progress:
            for path, rows, width, dtype in archives:
                for traces, labels in iter_npz_batches(path, rows, width, dtype):
                    times, directions, lengths = process_batch(traces)
                    count = len(traces)
                    end = written + count
                    target["times"][written:end] = times
                    target["directions"][written:end] = directions
                    target["lengths"][written:end] = lengths
                    target["labels"][written:end] = encode_labels(labels, path)
                    target["day"][written:end] = DAY
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
