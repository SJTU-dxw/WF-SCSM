"""Persistent single-stage k-shot splits for Closed-World datasets."""

from __future__ import annotations

import hashlib
import fcntl
import json
from pathlib import Path

import h5py
import numpy as np


def prepare_split(args):
    path = Path(args.dataset).resolve()
    stat = path.stat()
    spec = {
        "dataset": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "protocol": "closed_world_kshot_all_remaining_v1",
        "day": 56,
        "min_trace_length": args.min_trace_length,
        "k": args.k,
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
            "all remaining traces for testing",
            flush=True,
        )
        return directory, manifest

    if manifest_path.exists():
        return load_manifest()

    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".split.lock"
    # The first process creates the split while concurrent workers block here.
    # A flock is released automatically if its owner exits, so crashes cannot
    # leave a stale lock that permanently prevents later runs.
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if manifest_path.exists():
            return load_manifest()

        with h5py.File(path, "r") as source:
            if not source.attrs.get("complete", True):
                raise ValueError("HDF5 conversion is incomplete")
            days, labels, lengths = (
                source[name][:] for name in ("day", "labels", "lengths")
            )
        observed_days = set(np.unique(days).tolist())
        if observed_days != {56}:
            raise ValueError(
                f"Expected every Closed-World trace at day 56, got {observed_days}"
            )

        valid = lengths >= args.min_trace_length
        classes = sorted(np.unique(labels[valid]).tolist())
        if len(classes) < 2:
            raise ValueError("Fewer than two classes remain after length filtering")

        train_rows = []
        test_rows = []
        for y, label in enumerate(classes):
            indices = np.flatnonzero(valid & (labels == label))
            if len(indices) <= args.k:
                decoded = label.decode("utf-8")
                raise ValueError(
                    f"class {decoded}: {len(indices)} valid samples; need more than k={args.k}"
                )
            rng = np.random.default_rng(
                np.random.SeedSequence([args.split_seed, args.seed, y])
            )
            indices = rng.permutation(indices)
            records = np.column_stack(
                (indices, np.full(len(indices), y), np.ones(len(indices)))
            )
            train_rows.extend(records[: args.k])
            test_rows.extend(records[args.k :])

        arrays = {
            "pool_1": np.asarray(train_rows, dtype=np.int64),
            "test": np.asarray(test_rows, dtype=np.int64),
        }
        manifest = {
            "spec": spec,
            "eligible_count": len(classes),
            "classes": [label.decode("utf-8") for label in classes],
            "weeks": [[56, 56]],
            "test_count": len(test_rows),
            "train_per_class": args.k,
            "test_policy": "all_remaining",
        }
        np.savez_compressed(directory / "indices.npz", **arrays)
        temporary = directory / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(manifest_path)
        print(
            f"Split: {len(classes)} classes, {args.k}-shot, "
            f"{len(test_rows)} remaining traces for testing",
            flush=True,
        )
        return directory, manifest
