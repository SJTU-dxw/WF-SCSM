#!/usr/bin/env python3
"""Fix repeated initial timestamps in Defense HDF5 files in place."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np


BASE = Path(__file__).resolve().parent
DEFAULT_DATASETS = (
    BASE / "DF_WTF-PAD_train.hdf5",
    BASE / "DF_FRONT_train.hdf5",
    BASE / "DF_RegulaTor_train.hdf5",
    BASE / "DF_TrafficSilver_BD_train.hdf5",
    BASE / "DF_TrafficSilver_BWR_train.hdf5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        type=Path,
        help="HDF5 files to process; defaults to the five affected Defense datasets",
    )
    return parser.parse_args()


def validate_file(target: h5py.File, path: Path) -> None:
    for name in ("times", "directions", "lengths"):
        if name not in target:
            raise ValueError(f"{path}: missing dataset {name!r}")
    times = target["times"]
    directions = target["directions"]
    lengths = target["lengths"]
    if times.ndim != 2 or directions.shape != times.shape:
        raise ValueError(f"{path}: incompatible times/directions shapes")
    if lengths.shape != (times.shape[0],):
        raise ValueError(f"{path}: incompatible lengths shape")
    if not np.issubdtype(times.dtype, np.floating):
        raise ValueError(f"{path}: times must use a floating-point dtype")
    if not target.attrs.get("complete", True):
        raise ValueError(f"{path}: HDF5 conversion is incomplete")


def repair_copy(path: Path, temporary: Path) -> tuple[int, int]:
    changed_rows = 0
    changed_timestamps = 0
    with h5py.File(temporary, "r+") as target:
        validate_file(target, path)
        times = target["times"]
        lengths = target["lengths"]
        row_count, width = times.shape
        chunk_rows = times.chunks[0] if times.chunks else 256
        columns = np.arange(width)[None, :]

        for start in range(0, row_count, chunk_rows):
            end = min(start + chunk_rows, row_count)
            values = times[start:end]
            valid = columns < np.minimum(lengths[start:end], width)[:, None]
            repeated = valid & (columns > 0) & (values == values[:, :1])
            affected = np.any(repeated, axis=1)
            if not np.any(affected):
                continue

            replacement = np.nextafter(
                values[:, :1], np.asarray(np.inf, dtype=values.dtype)
            )
            values[repeated] = np.broadcast_to(replacement, values.shape)[repeated]
            times[start:end] = values
            changed_rows += int(np.count_nonzero(affected))
            changed_timestamps += int(np.count_nonzero(repeated))

        target.flush()

        # Re-read every valid trace to ensure no later packet still equals its first.
        for start in range(0, row_count, chunk_rows):
            end = min(start + chunk_rows, row_count)
            values = times[start:end]
            valid = columns < np.minimum(lengths[start:end], width)[:, None]
            repeated = valid & (columns > 0) & (values == values[:, :1])
            if np.any(repeated):
                row, column = np.argwhere(repeated)[0]
                raise RuntimeError(
                    f"{path}: validation failed at row {start + row}, column {column}"
                )

    return changed_rows, changed_timestamps


def replace_in_place(path: Path) -> tuple[int, int]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    temporary = path.with_name(f".{path.name}.initial-timestamps.tmp")
    if temporary.exists():
        raise FileExistsError(
            f"temporary file already exists; inspect or remove it first: {temporary}"
        )

    try:
        shutil.copy2(path, temporary)
        result = repair_copy(path, temporary)
        os.replace(temporary, path)
        return result
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    paths = args.datasets or DEFAULT_DATASETS
    total_rows = 0
    total_timestamps = 0
    for path in paths:
        changed_rows, changed_timestamps = replace_in_place(path)
        total_rows += changed_rows
        total_timestamps += changed_timestamps
        print(
            f"Replaced {path}: fixed {changed_timestamps} timestamps "
            f"across {changed_rows} traces",
            flush=True,
        )
    print(
        f"Done: fixed {total_timestamps} timestamps across {total_rows} traces "
        f"in {len(paths)} files",
        flush=True,
    )


if __name__ == "__main__":
    main()
