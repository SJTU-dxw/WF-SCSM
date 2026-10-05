"""Persistent k-shot split for the DF Open-World dataset."""

from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


OPEN_WORLD_LABEL = "95"


def prepare_split(args):
    path = Path(args.dataset).resolve()
    stat = path.stat()
    spec = {
        "dataset": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "protocol": "df_open_world_kshot_all_remaining_v1",
        "day": 56,
        "open_world_label": OPEN_WORLD_LABEL,
        "min_trace_length": args.min_trace_length,
        "k": args.k,
        "open_world_train_ratio": 95,
        "seed": args.seed,
        "split_seed": args.split_seed,
    }
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    directory = Path(args.split_dir) / key
    manifest_path = directory / "manifest.json"

    def load_manifest():
        manifest = json.loads(manifest_path.read_text())
        if manifest["spec"] != spec:
            raise ValueError("Split specification mismatch")
        print(
            f"Split: {len(manifest['classes'])} classes, {args.k}-shot, "
            f"unmonitored class={manifest['open_world_label']}",
            flush=True,
        )
        return directory, manifest

    if manifest_path.exists():
        return load_manifest()

    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".split.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if manifest_path.exists():
            return load_manifest()

        with h5py.File(path, "r") as source:
            if not source.attrs.get("complete", True):
                raise ValueError("HDF5 conversion is incomplete")
            if source.attrs.get("open_world_label") != OPEN_WORLD_LABEL:
                raise ValueError(
                    f"Expected HDF5 open_world_label={OPEN_WORLD_LABEL}"
                )
            days, labels, lengths = (
                source[name][:] for name in ("day", "labels", "lengths")
            )
        observed_days = set(np.unique(days).tolist())
        if observed_days != {56}:
            raise ValueError(f"Expected every trace at day 56, got {observed_days}")

        valid = lengths >= args.min_trace_length
        classes = sorted(np.unique(labels[valid]).tolist())
        decoded_classes = [label.decode("utf-8") for label in classes]
        if OPEN_WORLD_LABEL not in decoded_classes:
            raise ValueError(f"Missing unmonitored class {OPEN_WORLD_LABEL}")
        if len(classes) != 96:
            raise ValueError(f"Expected 96 classes, got {len(classes)}")

        monitored_class_count = len(classes) - 1
        open_world_train_count = args.k * monitored_class_count
        train_rows = []
        test_rows = []
        class_counts = {}
        for y, label in enumerate(classes):
            indices = np.flatnonzero(valid & (labels == label))
            decoded = label.decode("utf-8")
            class_counts[decoded] = int(len(indices))
            train_count = (
                open_world_train_count if decoded == OPEN_WORLD_LABEL else args.k
            )
            if len(indices) <= train_count:
                raise ValueError(
                    f"class {decoded}: {len(indices)} valid samples; "
                    f"need more than training count {train_count}"
                )
            rng = np.random.default_rng(
                np.random.SeedSequence([args.split_seed, args.seed, y])
            )
            indices = rng.permutation(indices)
            records = np.column_stack(
                (indices, np.full(len(indices), y), np.ones(len(indices)))
            )
            train_rows.extend(records[:train_count])
            test_rows.extend(records[train_count:])

        arrays = {
            "pool_1": np.asarray(train_rows, dtype=np.int64),
            "test": np.asarray(test_rows, dtype=np.int64),
        }
        open_index = decoded_classes.index(OPEN_WORLD_LABEL)
        manifest = {
            "spec": spec,
            "eligible_count": len(classes),
            "classes": decoded_classes,
            "weeks": [[56, 56]],
            "test_count": len(test_rows),
            "train_per_class": args.k,
            "monitored_train_count": args.k * monitored_class_count,
            "open_world_train_count": open_world_train_count,
            "total_train_count": args.k * monitored_class_count + open_world_train_count,
            "test_policy": "all_remaining",
            "open_world_label": OPEN_WORLD_LABEL,
            "open_world_class_index": open_index,
            "class_counts": class_counts,
        }
        np.savez_compressed(directory / "indices.npz", **arrays)
        temporary = directory / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(manifest_path)
        print(
            f"Split: {args.k * monitored_class_count} monitored + "
            f"{open_world_train_count} unmonitored training traces, "
            f"{len(test_rows)} remaining traces for testing",
            flush=True,
        )
        return directory, manifest


def use_preselected_train(pool, k, seed):
    """Return the persisted imbalanced training set without k-shot resampling."""
    labels, counts = np.unique(pool[:, 1], return_counts=True)
    if len(labels) != 96 or labels[-1] != 95:
        raise ValueError("Expected training records for classes 0..95")
    if not np.all(counts[:-1] == k):
        raise ValueError("Each monitored class must contain exactly k training traces")
    if counts[-1] != 95 * k:
        raise ValueError("Unmonitored training count must equal all monitored training traces")
    return np.asarray(pool, dtype=np.int64)
