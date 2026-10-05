"""Persistent global-k splits and multi-label dataset adapters for ARES."""

from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from finetune.data import WeeklyDataset


def prepare_split(args):
    path = Path(args.dataset).resolve()
    stat = path.stat()
    spec = {
        "dataset": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "protocol": "ares_multitab_global_k_all_remaining_v1",
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
            f"Split: {manifest['train_count']} global training traces, "
            f"{manifest['test_count']} remaining test traces",
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
            days = source["day"][:]
            lengths = source["lengths"][:]
            labels = source["labels"][:]
            source_splits = source["split"][:]
            label_format = source.attrs.get("label_format", "")
            labels_per_trace = int(source.attrs.get("labels_per_trace", -1))
            dataset_kind = source.attrs.get("dataset_kind", "")

        if label_format != "multi-hot" or labels.ndim != 2:
            raise ValueError("Expected a two-dimensional multi-hot labels dataset")
        if labels.shape[1] != args.class_count:
            raise ValueError(
                f"Expected {args.class_count} label columns, got {labels.shape[1]}"
            )
        if labels.dtype != np.uint8 or np.any((labels != 0) & (labels != 1)):
            raise ValueError("Labels must be a uint8 multi-hot matrix")
        if labels_per_trace != args.num_tabs:
            raise ValueError(
                f"Expected {args.num_tabs} positive labels per trace, "
                f"got HDF5 metadata {labels_per_trace}"
            )
        if np.any(labels.sum(axis=1) != args.num_tabs):
            raise ValueError("A trace has a positive-label count different from num_tabs")
        if dataset_kind != args.world:
            raise ValueError(f"Expected {args.world} dataset, got {dataset_kind!r}")
        if set(np.unique(days).tolist()) != {56}:
            raise ValueError("Expected every ARES trace at day 56")
        if not set(np.unique(source_splits).tolist()).issubset({0, 1, 2}):
            raise ValueError("Unexpected source split code")

        valid_indices = np.flatnonzero(lengths >= args.min_trace_length)
        if len(valid_indices) <= args.k:
            raise ValueError(
                f"Only {len(valid_indices)} valid traces; need more than k={args.k}"
            )
        rng = np.random.default_rng(
            np.random.SeedSequence([args.split_seed, args.seed])
        )
        shuffled = rng.permutation(valid_indices)
        train_indices = shuffled[: args.k]
        test_indices = shuffled[args.k :]

        def records(indices):
            return np.column_stack(
                (
                    indices,
                    np.zeros(len(indices), dtype=np.int64),
                    np.ones(len(indices), dtype=np.int64),
                )
            )

        arrays = {
            "pool_1": records(train_indices).astype(np.int64, copy=False),
            "test": records(test_indices).astype(np.int64, copy=False),
        }
        source_split_counts = {
            name: int(np.sum(source_splits == code))
            for name, code in (("train", 0), ("valid", 1), ("test", 2))
        }
        manifest = {
            "spec": spec,
            "classes": [str(index) for index in range(args.class_count)],
            "weeks": [[56, 56]],
            "train_count": int(len(train_indices)),
            "test_count": int(len(test_indices)),
            "eligible_count": int(len(valid_indices)),
            "test_policy": "all valid traces not selected for global-k training",
            "num_tabs": args.num_tabs,
            "world": args.world,
            "class_count": args.class_count,
            "source_split_counts": source_split_counts,
        }
        np.savez_compressed(directory / "indices.npz", **arrays)
        temporary = directory / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(manifest_path)
        print(
            f"Split: sampled {len(train_indices)} training traces globally; "
            f"using {len(test_indices)} remaining traces for testing",
            flush=True,
        )
        return directory, manifest


def use_preselected_train(pool, k, seed):
    if len(pool) != k:
        raise ValueError(f"Expected exactly k={k} persisted training traces, got {len(pool)}")
    return np.asarray(pool, dtype=np.int64)


class MultiLabelDataset(Dataset):
    """Reuse feature extraction and its HDF5 handle for multi-hot targets."""

    def __init__(self, path, records, method, config, train=False):
        self.path = str(path)
        self.records = np.asarray(records, dtype=np.int64)
        self.base = WeeklyDataset(path, records, method, config, train=train)

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        return self.__dict__.copy()

    def _source(self):
        if self.base.delegate is not None:
            return self.base.delegate.source()
        if self.base.file is None:
            self.base.file = h5py.File(self.path, "r", swmr=True)
        return self.base.file

    def close(self):
        if self.base.delegate is not None:
            source = self.base.delegate._file
            self.base.delegate._file = None
        else:
            source = self.base.file
            self.base.file = None
        if source is not None:
            source.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getitem__(self, index):
        row = int(self.records[index, 0])
        target = torch.from_numpy(
            np.asarray(self._source()["labels"][row], dtype=np.float32)
        )
        sample = self.base[index]
        return (*sample[:-1], target)


class MultiLabelTextCollator:
    def __init__(self, tokenizer, config):
        from feature_similarity.evaluate import TraverseCollator

        self.base = TraverseCollator(
            tokenizer,
            config["max_length"],
            config["prompt_prefix"],
            config["prompt_suffix"],
        )

    def __call__(self, samples):
        data, mask, _, _, _ = self.base(
            [(text, 0, b"", index) for index, (text, _) in enumerate(samples)]
        )
        return data, mask, torch.stack([target for _, target in samples])
