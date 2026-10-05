#!/usr/bin/env python3
"""Convert Swallow D1-D7 Undefence traces to the GTT-style HDF5 format.

The collection days are aligned after the GTT pre-training interval:
D1=49, D2=56, ..., D7=91.  Consequently, using ``max_day: 49`` in a
pre-training configuration selects D1 only.
"""

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
FIRST_DAY = 49
DAY_STEP = 7
PERIODS = range(1, 8)
FILENAME_PATTERN = re.compile(r"^(\d+)-(\d+)$")


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=base_dir,
        help="pretrain_dataset directory containing Swallow/Traces and D6/D7",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=base_dir / "Swallow_train.hdf5",
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


def find_period_dir(source_root: Path, period: int) -> Path:
    """Locate one Undefence directory without considering defended variants."""
    name = f"D{period}-Undefence"
    candidates = (source_root / "Swallow" / "Traces" / name, source_root / name)
    existing = [path for path in candidates if path.is_dir()]
    if not existing:
        locations = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"could not find {name}; checked: {locations}")
    if len(existing) > 1:
        raise RuntimeError(f"ambiguous source for {name}: {existing}")
    return existing[0]


def collect_files(source_root: Path) -> list[tuple[Path, int]]:
    records: list[tuple[Path, int]] = []
    reference_labels: set[int] | None = None
    for period in PERIODS:
        directory = find_period_dir(source_root, period)
        paths = sorted(
            (path for path in directory.iterdir() if path.is_file()),
            key=trace_sort_key,
        )
        if not paths:
            raise RuntimeError(f"no trace files found in {directory}")
        labels = {trace_sort_key(path)[0] for path in paths}
        if reference_labels is None:
            reference_labels = labels
        else:
            missing = reference_labels - labels
            if missing:
                raise RuntimeError(
                    f"D{period} is missing reference website labels: {sorted(missing)}"
                )
            extra = labels - reference_labels
            if extra:
                before = len(paths)
                paths = [
                    path for path in paths
                    if trace_sort_key(path)[0] in reference_labels
                ]
                print(
                    f"D{period}: filtering extra website labels {sorted(extra)} "
                    f"({before - len(paths)} traces)"
                )
        day = FIRST_DAY + (period - 1) * DAY_STEP
        print(
            f"D{period}: day={day}, traces={len(paths)}, "
            f"websites={len(reference_labels)}, source={directory}"
        )
        records.extend((path, day) for path in paths)
    return records


def create_target_file(target: h5py.File, total: int, source_root: Path) -> None:
    target.create_dataset(
        "times",
        shape=(total, SEQUENCE_LENGTH),
        dtype=np.float32,
        chunks=(CHUNK_ROWS, SEQUENCE_LENGTH),
        compression="lzf",
        shuffle=True,
    )
    target.create_dataset(
        "directions",
        shape=(total, SEQUENCE_LENGTH),
        dtype=np.int8,
        chunks=(CHUNK_ROWS, SEQUENCE_LENGTH),
        compression="lzf",
        shuffle=True,
    )
    target.create_dataset(
        "lengths",
        shape=(total,),
        dtype=np.uint16,
        chunks=(8192,),
        compression="lzf",
        shuffle=True,
    )
    target.create_dataset(
        "labels",
        shape=(total,),
        dtype="S44",
        chunks=(8192,),
        compression="lzf",
    )
    target.create_dataset(
        "day",
        shape=(total,),
        dtype=np.uint8,
        chunks=(8192,),
        compression="lzf",
    )
    target.attrs["total_rows"] = total
    target.attrs["sequence_length"] = SEQUENCE_LENGTH
    target.attrs["source_root"] = str(source_root)
    target.attrs["dataset_variant"] = "Undefence"
    target.attrs["website_filter"] = "D1 label set"
    target.attrs["day_mapping"] = "D1=49,D2=56,D3=63,D4=70,D5=77,D6=84,D7=91"
    target.attrs["pretrain_max_day_for_d1_only"] = FIRST_DAY
    target.attrs["complete"] = False


def read_trace(path: Path) -> tuple[np.ndarray, np.ndarray, int, bytes]:
    # D6 contains an extra third column; only timestamp and direction are used.
    trace = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if trace.shape[1] < 2:
        raise ValueError(f"{path}: expected at least 2 columns, got {trace.shape[1]}")
    if trace.shape[0] == 0:
        raise ValueError(f"{path}: empty trace")

    raw_times = trace[:, 0]
    raw_directions = trace[:, 1]
    if not np.all(np.isfinite(raw_times)):
        raise ValueError(f"{path}: time contains NaN or inf")
    directions = np.sign(raw_directions)
    if not np.all((directions == -1) | (directions == 1)):
        raise ValueError(f"{path}: packet direction must be -1 or 1")

    # Sort every packet by timestamp instead of accepting or flattening time
    # regressions.  Stable sorting preserves source order for equal timestamps,
    # and the same permutation keeps packet directions paired with their times.
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

    records = collect_files(source_root)
    if target_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"target already exists: {target_path}; pass --overwrite to replace it"
            )
        target_path.unlink()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(records)
    print(f"Total Undefence traces: {total}")
    print(f"Creating: {target_path}")
    started = time.time()

    with h5py.File(target_path, "w", libver="latest") as target:
        create_target_file(target, total, source_root)
        with tqdm(total=total, unit="trace", desc="Converting Swallow") as progress:
            for start in range(0, total, BATCH_SIZE):
                batch = records[start : start + BATCH_SIZE]
                count = len(batch)
                times = np.zeros((count, SEQUENCE_LENGTH), dtype=np.float32)
                directions = np.zeros((count, SEQUENCE_LENGTH), dtype=np.int8)
                lengths = np.empty(count, dtype=np.uint16)
                labels = np.empty(count, dtype="S44")
                days = np.empty(count, dtype=np.uint8)

                for row, (path, day) in enumerate(batch):
                    times[row], directions[row], lengths[row], labels[row] = read_trace(path)
                    days[row] = day

                end = start + count
                target["times"][start:end] = times
                target["directions"][start:end] = directions
                target["lengths"][start:end] = lengths
                target["labels"][start:end] = labels
                target["day"][start:end] = days
                progress.update(count)

        target.attrs["complete"] = True
        target.flush()

    elapsed = time.time() - started
    print(f"Done in {elapsed / 3600:.2f} hours")
    print(f"Saved to: {target_path}")
    print("Use max_day: 49 during pre-training to select D1 only.")


if __name__ == "__main__":
    main()
