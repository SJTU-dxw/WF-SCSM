#!/usr/bin/env python3
"""Evaluate SCSM ablations on DF with frozen-encoder classification metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_similarity.evaluate import (  # noqa: E402
    TraceDataset,
    TRAVERSE_MAX_BURSTS,
    TRAVERSE_PROMPT_PREFIX,
    TRAVERSE_PROMPT_SUFFIX,
    TraverseCollator,
    extract_embeddings,
    load_config,
    load_model,
    macro_accuracy,
    swallow_cif_config,
)


MODEL_DIRS = {
    "full": PROJECT_ROOT / "output/pretrain/swallow_dataset/scsm_single",
    "no_segmentation": PROJECT_ROOT / "output/pretrain/swallow_dataset/ablation/no_segmentation",
    "no_combination": PROJECT_ROOT / "output/pretrain/swallow_dataset/ablation/no_combination",
    "no_scaling": PROJECT_ROOT / "output/pretrain/swallow_dataset/ablation/no_scaling",
    "no_masking": PROJECT_ROOT / "output/pretrain/swallow_dataset/ablation/no_masking",
    "mask_only": PROJECT_ROOT / "output/pretrain/swallow_dataset/ablation/mask_only",
    "no_augmentation": PROJECT_ROOT / "output/pretrain/swallow_dataset/ablation/no_augmentation",
    "netclr": PROJECT_ROOT / "output/pretrain/swallow_dataset/netclr",
    "swallow": PROJECT_ROOT / "output/pretrain/swallow_dataset/swallow-origin",
    "traverse": PROJECT_ROOT / "output/pretrain/swallow_dataset/traverse",
}
DEFAULT_SLOT_DURATION = 120.0 / 2700.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=PROJECT_ROOT / "pretrain_dataset/DF_train.hdf5")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_DIRS),
                        default=list(MODEL_DIRS))
    parser.add_argument("--gallery-per-class", type=int, default=100,
                        help="Number of labeled reference traces per class")
    parser.add_argument("--min-trace-length", type=int, default=1)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--knn-query-block-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--slot-duration", type=float, default=DEFAULT_SLOT_DURATION,
                        help="Fixed inference slot width in seconds (default: 120/2700)")
    parser.add_argument("--slot-strategies", nargs="+", choices=("fixed", "ensemble"),
                        default=["fixed", "ensemble"],
                        help="Inference slot protocols to evaluate")
    parser.add_argument("--slot-count", type=int, default=5,
                        help="Number of evenly spaced slots for ensemble inference")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "feature_similarity/results/ablation_df")
    args = parser.parse_args()
    if args.gallery_per_class < 1 or args.min_trace_length < 1:
        parser.error("gallery-per-class and min-trace-length must be positive")
    if args.slot_duration <= 0:
        parser.error("slot-duration must be positive")
    if args.slot_count < 1:
        parser.error("slot-count must be positive")
    return args


def stratified_records(path: Path, gallery_per_class: int, min_trace_length: int,
                       seed: int):
    """Return one deterministic, class-stratified gallery/query split."""
    with h5py.File(path, "r", swmr=True) as source:
        labels = np.asarray(source["labels"])
        lengths = np.asarray(source["lengths"])
    classes = sorted(bytes(label) for label in np.unique(labels))
    rng = np.random.default_rng(seed)
    gallery, query = [], []
    for label in classes:
        indices = np.flatnonzero((labels == label) & (lengths >= min_trace_length))
        if len(indices) <= gallery_per_class:
            raise ValueError(
                f"class {label!r} has {len(indices)} valid traces; it needs more than "
                f"gallery-per-class={gallery_per_class}"
            )
        indices = rng.permutation(indices)
        gallery.extend((0, int(index), label) for index in indices[:gallery_per_class])
        query.extend((1, int(index), label) for index in indices[gallery_per_class:])
    return classes, gallery, query


def flatten(embeddings, period: int, classes: list[bytes]):
    parts, targets = [], []
    for class_id, label in enumerate(classes):
        features = embeddings[(period, label)]
        parts.append(features)
        targets.append(np.full(len(features), class_id, dtype=np.int64))
    return np.concatenate(parts), np.concatenate(targets)


def knn_predictions(query: np.ndarray, gallery: np.ndarray,
                    gallery_targets: np.ndarray, class_count: int, k: int,
                    block_size: int, device: torch.device) -> np.ndarray:
    """Exact cosine k-NN on the selected device with vectorized voting."""
    if k < 1 or k > len(gallery):
        raise ValueError(f"knn-k must be in [1, {len(gallery)}]")
    gallery_tensor = torch.from_numpy(gallery).to(device)
    target_tensor = torch.from_numpy(gallery_targets).to(device)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(query), block_size):
            batch = torch.from_numpy(query[start:start + block_size]).to(device)
            similarities, indices = torch.topk(batch @ gallery_tensor.T, k, dim=1)
            labels = target_tensor[indices]
            votes = torch.zeros(
                len(batch), class_count, dtype=torch.float32, device=device,
            )
            similarity_sums = torch.zeros_like(votes)
            votes.scatter_add_(1, labels, torch.ones_like(similarities))
            similarity_sums.scatter_add_(1, labels, similarities)
            # Vote count is primary; summed cosine similarity breaks ties.
            scores = votes * (2 * k + 1) + similarity_sums
            predictions.append(scores.argmax(dim=1).cpu().numpy())
            print(
                f"kNN queries: {min(start + block_size, len(query)):,}/{len(query):,}",
                end="\r", flush=True,
            )
    print()
    return np.concatenate(predictions)


def evaluate_model(embeddings, classes: list[bytes], knn_k: int,
                   query_block_size: int, device: torch.device) -> dict[str, float]:
    gallery, gallery_targets = flatten(embeddings, 0, classes)
    query, query_targets = flatten(embeddings, 1, classes)

    centroids = np.stack([
        gallery[gallery_targets == class_id].mean(axis=0)
        for class_id in range(len(classes))
    ])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-12)
    centroid_predictions = (query @ centroids.T).argmax(axis=1)
    knn_result = knn_predictions(
        query, gallery, gallery_targets, len(classes), knn_k,
        query_block_size, device,
    )
    return {
        "centroid_macro_accuracy": macro_accuracy(query_targets, centroid_predictions),
        f"knn_{knn_k}_macro_accuracy": macro_accuracy(query_targets, knn_result),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    classes, gallery_records, query_records = stratified_records(
        dataset_path, args.gallery_per_class, args.min_trace_length, args.seed,
    )
    records = gallery_records + query_records
    print(
        f"dataset={dataset_path} classes={len(classes)} gallery={len(gallery_records):,} "
        f"query={len(query_records):,} seed={args.seed}"
    )

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name in args.models:
        model_dir = MODEL_DIRS[model_name]
        config = load_config(model_dir)
        method = config["method"].lower()
        strategies = args.slot_strategies if method == "scsm" else ["native"]
        tokenizer = None
        if method == "traverse":
            pretrained = PROJECT_ROOT / "pretrain" / Path(
                config["model"]["pretrained_model"]
            ).name
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained,
                trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
            )
            if tokenizer.mask_token_id is None:
                tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
        model, weights_path = load_model(config, model_dir, device, tokenizer)
        for slot_strategy in strategies:
            collate_fn = None
            batch_size = args.batch_size
            if method == "scsm":
                feature_config = {
                    **config["model"],
                    **config["augmentation"],
                    "mask": False,
                    "slot_strategy": slot_strategy,
                    "slot_ensemble": slot_strategy == "ensemble",
                    "test_slot_count": args.slot_count,
                    "maximum_load_time": (
                        args.slot_duration * int(config["model"]["max_matrix_length"])
                    ),
                }
            elif method == "netclr":
                feature_config = config["model"]
            elif method == "swallow":
                feature_config = swallow_cif_config(config)
            elif method == "traverse":
                feature_config = {
                    "max_bursts": TRAVERSE_MAX_BURSTS,
                    "prompt_prefix": TRAVERSE_PROMPT_PREFIX,
                    "prompt_suffix": TRAVERSE_PROMPT_SUFFIX,
                }
                collate_fn = TraverseCollator(
                    tokenizer,
                    int(config["dataset"]["max_length"]),
                    feature_config["prompt_prefix"],
                    feature_config["prompt_suffix"],
                )
                batch_size = min(batch_size, int(config["dataset"]["batch_size"]))
            else:
                raise ValueError(f"unsupported evaluation method: {method}")
            dataset = TraceDataset(dataset_path, records, method, feature_config)
            loader_options = {
                "batch_size": batch_size,
                "shuffle": False,
                "num_workers": args.num_workers,
                "pin_memory": args.device.startswith("cuda"),
                "persistent_workers": False,
                "collate_fn": collate_fn,
            }
            if args.num_workers > 0:
                loader_options["multiprocessing_context"] = "spawn"
            loader = DataLoader(dataset, **loader_options)
            print(f"evaluating {model_name}/{slot_strategy}: {weights_path}")
            embeddings = extract_embeddings(model, loader, device, method)
            metrics = evaluate_model(
                embeddings, classes, args.knn_k, args.knn_query_block_size, device,
            )
            row = {
                "model": model_name,
                "inference_strategy": slot_strategy,
                "classes": len(classes),
                "gallery_samples": len(gallery_records),
                "query_samples": len(query_records),
                **metrics,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
            del embeddings, loader
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(args.output_dir / "metrics.csv", rows)
    (args.output_dir / "protocol.json").write_text(
        json.dumps({
            "dataset": str(dataset_path),
            "models": args.models,
            "seed": args.seed,
            "minimum_trace_length": args.min_trace_length,
            "gallery_per_class": args.gallery_per_class,
            "query_policy": "all remaining valid traces",
            "knn_k": args.knn_k,
            "slot_strategies": args.slot_strategies,
            "fixed_slot_duration_seconds": args.slot_duration,
            "ensemble_slot_range_seconds": [0.005, 0.05],
            "ensemble_slot_count": args.slot_count,
            "metrics": ["centroid macro accuracy", f"{args.knn_k}-NN macro accuracy"],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(args.output_dir / "metrics.csv")


if __name__ == "__main__":
    main()
