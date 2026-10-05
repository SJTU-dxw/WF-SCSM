"""Shared k-shot membership across models; all remaining traces are test data."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def save_split(args, days, labels):
    path = Path(args.dataset).resolve()
    stat = path.stat()
    spec = dict(dataset=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                protocol="kshot_all_remaining_v1", k=args.k, seed=args.seed,
                split_seed=args.split_seed, classes=list(range(100)))
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    directory = Path(args.split_dir) / key
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["spec"] != spec:
            raise ValueError("Split specification mismatch")
        return directory, manifest
    arrays = {}
    tests = []
    weeks = [[start, start + 6] for start in range(50, 86, 7)]
    classes = sorted(str(i).encode("ascii") for i in range(100))
    for week, (start, end) in enumerate(weeks, 1):
        train_rows = []
        for y, label in enumerate(classes):
            indices = np.flatnonzero((days >= start) & (days <= end) & (labels == label))
            rng = np.random.default_rng(np.random.SeedSequence([args.split_seed, args.seed, week, y]))
            indices = rng.permutation(indices)
            rows = np.column_stack((indices, np.full(len(indices), y), np.full(len(indices), week)))
            train_rows.extend(rows[:args.k])
            tests.extend(rows[args.k:])
        # The reused sample_train() receives exactly k rows per class and only
        # changes their order, never their membership.
        arrays[f"pool_{week}"] = np.asarray(train_rows, dtype=np.int64)
    arrays["test"] = np.asarray(tests, dtype=np.int64)
    manifest = dict(spec=spec, eligible_count=100, classes=[c.decode("ascii") for c in classes],
                    weeks=weeks, test_count=len(tests), train_per_class=args.k,
                    test_policy="all_remaining")
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "creating.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        if not manifest_path.exists():
            np.savez_compressed(directory / "indices.npz", **arrays)
            temp = directory / "manifest.tmp"
            temp.write_text(json.dumps(manifest, indent=2))
            temp.replace(manifest_path)
    finally:
        os.close(fd)
        lock.unlink()
    print(f"Split: 100 classes, {args.k}-shot, all remaining traces for testing", flush=True)
    return directory, manifest
