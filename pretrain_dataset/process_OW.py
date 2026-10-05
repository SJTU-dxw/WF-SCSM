#!/usr/bin/env python3
"""Merge DF monitored traces and DF open-world traces into one HDF5 file."""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

try:
    from .process_DF import (
        BATCH_SIZE,
        CHUNK_ROWS,
        DAY,
        FILENAME_PATTERN,
        SEQUENCE_LENGTH,
        trace_sort_key,
    )
except ImportError:
    from process_DF import (
        BATCH_SIZE,
        CHUNK_ROWS,
        DAY,
        FILENAME_PATTERN,
        SEQUENCE_LENGTH,
        trace_sort_key,
    )


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--df-source-root",
        type=Path,
        default=base_dir / "raw-data-50-1000",
        help="directory containing monitored DF traces named <label>-<index>",
    )
    parser.add_argument(
        "--ow-source-root",
        type=Path,
        default=base_dir / "open-world-traces-50-40716",
        help="directory containing unmonitored open-world traces",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=base_dir / "OW_train.hdf5",
        help="combined output HDF5 path",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite the target if it already exists",
    )
    return parser.parse_args()


def collect_files(df_root: Path, ow_root: Path) -> tuple[list[Path], list[Path], bytes]:
    df_paths = sorted(
        (path for path in df_root.iterdir() if path.is_file()),
        key=trace_sort_key,
    )
    ow_paths = sorted(
        (path for path in ow_root.iterdir() if path.is_file()),
        key=lambda path: path.name,
    )
    if not df_paths:
        raise RuntimeError(f"no DF trace files found in {df_root}")
    if not ow_paths:
        raise RuntimeError(f"no open-world trace files found in {ow_root}")

    df_labels = sorted({trace_sort_key(path)[0] for path in df_paths})
    if df_labels != list(range(df_labels[-1] + 1)):
        raise ValueError(f"DF labels must be contiguous from 0: {df_labels}")
    ow_label = str(df_labels[-1] + 1).encode("ascii")
    print(f"DF traces: {len(df_paths)}, monitored classes: {len(df_labels)}")
    print(f"Open-world traces: {len(ow_paths)}, shared class label: {ow_label.decode()}")
    return df_paths, ow_paths, ow_label


def create_target_file(
    target: h5py.File,
    total: int,
    df_root: Path,
    ow_root: Path,
    ow_label: bytes,
    ow_count: int,
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
    for name, dtype in (("lengths", np.uint16), ("day", np.uint8)):
        target.create_dataset(
            name,
            shape=(total,),
            dtype=dtype,
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
    target.attrs["total_rows"] = total
    target.attrs["sequence_length"] = SEQUENCE_LENGTH
    target.attrs["df_source_root"] = str(df_root)
    target.attrs["open_world_source_root"] = str(ow_root)
    target.attrs["dataset"] = "DF Undefended Open World"
    target.attrs["open_world_label"] = ow_label.decode("ascii")
    target.attrs["accepted_open_world_footer"] = f"Traces: {ow_count}"
    target.attrs["day_mapping"] = f"all traces={DAY}"
    target.attrs["packet_order"] = "stable timestamp sort"
    target.attrs["complete"] = False


def load_numeric_trace(path: Path, footer_count: int | None) -> np.ndarray:
    try:
        return np.loadtxt(path, dtype=np.float64, ndmin=2)
    except ValueError as error:
        if footer_count is None:
            raise
        lines = path.read_text().splitlines()
        nonempty = [index for index, line in enumerate(lines) if line.strip()]
        expected = f"Traces: {footer_count}"
        if not nonempty or lines[nonempty[-1]].strip() != expected:
            raise error
        del lines[nonempty[-1]]
        # Retrying remains strict: any malformed content other than the one
        # verified terminal dataset-count line still raises ValueError.
        return np.loadtxt(io.StringIO("\n".join(lines)), dtype=np.float64, ndmin=2)


def read_trace(
    path: Path,
    label: bytes,
    footer_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, bytes]:
    trace = load_numeric_trace(path, footer_count)
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

    order = np.argsort(raw_times, kind="stable")
    raw_times = raw_times[order]
    directions = directions[order]
    length = min(len(raw_times), SEQUENCE_LENGTH)
    times = np.zeros(SEQUENCE_LENGTH, dtype=np.float32)
    padded_directions = np.zeros(SEQUENCE_LENGTH, dtype=np.int8)
    times[:length] = raw_times[:length]
    padded_directions[:length] = directions[:length].astype(np.int8, copy=False)
    return times, padded_directions, length, label


def write_group(
    target: h5py.File,
    paths: list[Path],
    start: int,
    fixed_label: bytes | None,
    footer_count: int | None,
    progress: tqdm,
) -> int:
    for batch_start in range(0, len(paths), BATCH_SIZE):
        batch = paths[batch_start : batch_start + BATCH_SIZE]
        count = len(batch)
        times = np.zeros((count, SEQUENCE_LENGTH), dtype=np.float32)
        directions = np.zeros((count, SEQUENCE_LENGTH), dtype=np.int8)
        lengths = np.empty(count, dtype=np.uint16)
        labels = np.empty(count, dtype="S44")
        for row, path in enumerate(batch):
            label = fixed_label
            if label is None:
                match = FILENAME_PATTERN.fullmatch(path.name)
                if match is None:
                    raise ValueError(f"unexpected DF trace filename: {path}")
                label = match.group(1).encode("ascii")
            times[row], directions[row], lengths[row], labels[row] = read_trace(
                path, label, footer_count
            )
        end = start + count
        target["times"][start:end] = times
        target["directions"][start:end] = directions
        target["lengths"][start:end] = lengths
        target["labels"][start:end] = labels
        target["day"][start:end] = DAY
        start = end
        progress.update(count)
    return start


def main() -> None:
    args = parse_args()
    df_root = args.df_source_root.expanduser().resolve()
    ow_root = args.ow_source_root.expanduser().resolve()
    target_path = args.target.expanduser().resolve()
    for source in (df_root, ow_root):
        if not source.is_dir():
            raise FileNotFoundError(f"source root does not exist: {source}")
    df_paths, ow_paths, ow_label = collect_files(df_root, ow_root)
    if target_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"target already exists: {target_path}; pass --overwrite to replace it"
            )
        target_path.unlink()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(df_paths) + len(ow_paths)
    print(f"Creating: {target_path}")
    print(f"Total traces: {total}; classes: {int(ow_label) + 1}; day: {DAY}")
    started = time.time()
    with h5py.File(target_path, "w", libver="latest") as target:
        create_target_file(target, total, df_root, ow_root, ow_label, len(ow_paths))
        with tqdm(total=total, unit="trace", desc="Converting DF Open World") as progress:
            offset = write_group(target, df_paths, 0, None, None, progress)
            offset = write_group(
                target, ow_paths, offset, ow_label, len(ow_paths), progress
            )
        if offset != total:
            raise RuntimeError(f"wrote {offset} rows, expected {total}")
        target.attrs["complete"] = True
        target.flush()
    elapsed = time.time() - started
    print(f"Done in {elapsed / 3600:.2f} hours")
    print(f"Saved to: {target_path}")


if __name__ == "__main__":
    main()
