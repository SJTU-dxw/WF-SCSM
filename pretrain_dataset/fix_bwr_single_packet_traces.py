#!/usr/bin/env python3
"""Remove single-packet traces from the TrafficSilver-BWR HDF5 in place."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np


BASE = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE / "DF_TrafficSilver_BWR_train.hdf5"
ROW_DATASETS = ("times", "directions", "lengths", "labels", "day")
COPY_ROWS = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET,
        help="BWR HDF5 to repair in place",
    )
    return parser.parse_args()


def validate_source(source: h5py.File, path: Path) -> int:
    missing = [name for name in ROW_DATASETS if name not in source]
    if missing:
        raise ValueError(f"{path}: missing datasets: {', '.join(missing)}")
    if not source.attrs.get("complete", True):
        raise ValueError(f"{path}: HDF5 conversion is incomplete")

    row_count = source["lengths"].shape[0]
    for name in ROW_DATASETS:
        dataset = source[name]
        if dataset.ndim < 1 or dataset.shape[0] != row_count:
            raise ValueError(
                f"{path}: dataset {name!r} has incompatible shape {dataset.shape}"
            )
    return row_count


def creation_options(source: h5py.Dataset, rows: int) -> dict:
    options = {}
    if source.chunks is not None:
        chunks = list(source.chunks)
        chunks[0] = min(chunks[0], rows)
        options["chunks"] = tuple(chunks)
    if source.compression is not None:
        options["compression"] = source.compression
        options["compression_opts"] = source.compression_opts
    if source.shuffle:
        options["shuffle"] = True
    if source.fletcher32:
        options["fletcher32"] = True
    if source.scaleoffset is not None:
        options["scaleoffset"] = source.scaleoffset
    return options


def copy_filtered_rows(
    source: h5py.Dataset,
    target: h5py.Dataset,
    keep: np.ndarray,
) -> None:
    output_start = 0
    for input_start in range(0, len(keep), COPY_ROWS):
        input_end = min(input_start + COPY_ROWS, len(keep))
        selected = keep[input_start:input_end]
        count = int(np.count_nonzero(selected))
        if count == 0:
            continue
        target[output_start : output_start + count] = source[
            input_start:input_end
        ][selected]
        output_start += count
    if output_start != target.shape[0]:
        raise RuntimeError(
            f"copy count mismatch for {source.name}: "
            f"wrote {output_start}, expected {target.shape[0]}"
        )


def build_repaired_file(path: Path, temporary: Path) -> tuple[int, int]:
    with h5py.File(path, "r") as source:
        original_rows = validate_source(source, path)
        lengths = source["lengths"][:]
        keep = lengths > 1
        removed_rows = int(np.count_nonzero(~keep))
        if removed_rows == 0:
            return original_rows, 0

        repaired_rows = int(np.count_nonzero(keep))
        with h5py.File(temporary, "w", libver="latest") as target:
            for key, value in source.attrs.items():
                target.attrs[key] = value
            target.attrs["complete"] = False
            target.attrs["total_rows"] = repaired_rows
            target.attrs["removed_single_packet_traces"] = removed_rows

            for name, dataset in source.items():
                if name not in ROW_DATASETS:
                    source.copy(dataset, target, name=name)
                    continue
                shape = (repaired_rows, *dataset.shape[1:])
                output = target.create_dataset(
                    name,
                    shape=shape,
                    dtype=dataset.dtype,
                    **creation_options(dataset, repaired_rows),
                )
                for key, value in dataset.attrs.items():
                    output.attrs[key] = value
                copy_filtered_rows(dataset, output, keep)

            target.attrs["complete"] = True
            target.flush()

    return repaired_rows, removed_rows


def validate_repaired_file(
    temporary: Path,
    expected_rows: int,
    expected_removed: int,
) -> None:
    with h5py.File(temporary, "r") as target:
        row_count = validate_source(target, temporary)
        if row_count != expected_rows:
            raise RuntimeError(
                f"{temporary}: row count {row_count}, expected {expected_rows}"
            )
        if int(target.attrs.get("total_rows", -1)) != expected_rows:
            raise RuntimeError(f"{temporary}: total_rows attribute is incorrect")
        if int(target.attrs.get("removed_single_packet_traces", -1)) != expected_removed:
            raise RuntimeError(
                f"{temporary}: removed_single_packet_traces attribute is incorrect"
            )
        lengths = target["lengths"][:]
        if np.any(lengths <= 1):
            index = int(np.flatnonzero(lengths <= 1)[0])
            raise RuntimeError(
                f"{temporary}: validation found length {lengths[index]} at row {index}"
            )


def replace_in_place(path: Path) -> tuple[int, int]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    temporary = path.with_name(f".{path.name}.remove-single-packet.tmp")
    if temporary.exists():
        raise FileExistsError(
            f"temporary file already exists; inspect or remove it first: {temporary}"
        )

    try:
        repaired_rows, removed_rows = build_repaired_file(path, temporary)
        if removed_rows == 0:
            temporary.unlink(missing_ok=True)
            return repaired_rows, 0
        validate_repaired_file(temporary, repaired_rows, removed_rows)
        os.replace(temporary, path)
        return repaired_rows, removed_rows
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    repaired_rows, removed_rows = replace_in_place(args.dataset)
    if removed_rows == 0:
        print(f"No single-packet traces found: {args.dataset}", flush=True)
    else:
        print(
            f"Replaced {args.dataset}: removed {removed_rows} traces; "
            f"remaining rows: {repaired_rows}",
            flush=True,
        )


if __name__ == "__main__":
    main()
