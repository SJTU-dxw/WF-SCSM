#!/usr/bin/env python3
"""Evaluate weekly temporal drift using one GTT dataset for both periods."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.netclr import DFEncoder
from model.scsm import SCSM
from model.swallow import SwallowNetwork
from model.traverse import TraVerseMLM
from feature_extract.feature_burst import fun as extract_bursts
from feature_extract.feature_length import fun as extract_length
from feature_extract.feature_swallow import fun as extract_cif


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-trace-length", type=int, default=80, help="trace最小长度")
    parser.add_argument("--min-reference-samples", type=int, default=1000, help="参考时期网站总共最小样本数")
    parser.add_argument("--min-week-samples", type=int, default=25, help="预测时期网站每周最小样本数")
    parser.add_argument("--website-count", type=int, default=200,
                        help="按全部有效样本数保留最多的 N 个网站")
    parser.add_argument(
        "--scsm-slot-strategy",
        choices=("fixed", "adaptive", "ensemble", "both", "all"),
        default="all",
        help="SCSM 特征评测的窗口策略",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="只评测指定的模型目录名；默认评测全部模型",
    )
    parser.add_argument("--max-samples-per-site", type=int, default=1000,
                        help="每个网站在每个时期最多保留的样本数")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="样本抽样随机种子")
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--knn-query-block-size", type=int, default=128,
                        help="Number of query samples processed per kNN similarity block")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.min_trace_length < 1:
        parser.error("min-trace-length must be positive")
    if args.website_count < 2:
        parser.error("website-count must be at least 2")
    return args


MODEL_DIRS = {
    "GTT_scsm_single": PROJECT_ROOT / "output/pretrain/GTT_dataset/scsm_single",
    "GTT_scsm_multi": PROJECT_ROOT / "output/pretrain/GTT_dataset/scsm_multi",
    "GTT_netclr": PROJECT_ROOT / "output/pretrain/GTT_dataset/netclr",
    "GTT_swallow_origin": PROJECT_ROOT / "output/pretrain/GTT_dataset/swallow-origin",
    "GTT_swallow_single": PROJECT_ROOT / "output/pretrain/GTT_dataset/swallow-single",
    "GTT_swallow_multi": PROJECT_ROOT / "output/pretrain/GTT_dataset/swallow-multi",
    "GTT_traverse": PROJECT_ROOT / "output/pretrain/GTT_dataset/traverse",
    # Explicit aliases keep SwallowDataset-pretrained results separate.
    "swallow_dataset_scsm_single": PROJECT_ROOT / "output/pretrain/swallow_dataset/scsm_single",
    "swallow_dataset_scsm_no_segmentation": PROJECT_ROOT / "output/pretrain/swallow_dataset/ablation/no_segmentation",
    "swallow_dataset_scsm_no_combination": PROJECT_ROOT / "output/pretrain/swallow_dataset/ablation/no_combination",
    "swallow_dataset_scsm_no_scaling": PROJECT_ROOT / "output/pretrain/swallow_dataset/ablation/no_scaling",
    "swallow_dataset_scsm_no_masking": PROJECT_ROOT / "output/pretrain/swallow_dataset/ablation/no_masking",
    "swallow_dataset_scsm_mask_only": PROJECT_ROOT / "output/pretrain/swallow_dataset/ablation/mask_only",
    "swallow_dataset_scsm_no_augmentation": PROJECT_ROOT / "output/pretrain/swallow_dataset/ablation/no_augmentation",
    "swallow_dataset_scsm_multi": PROJECT_ROOT / "output/pretrain/swallow_dataset/scsm_multi",
    "swallow_dataset_netclr": PROJECT_ROOT / "output/pretrain/swallow_dataset/netclr",
    "swallow_dataset_swallow_origin": PROJECT_ROOT / "output/pretrain/swallow_dataset/swallow-origin",
    "swallow_dataset_swallow_single": PROJECT_ROOT / "output/pretrain/swallow_dataset/swallow-single",
    "swallow_dataset_swallow_multi": PROJECT_ROOT / "output/pretrain/swallow_dataset/swallow-multi",
    "swallow_dataset_traverse": PROJECT_ROOT / "output/pretrain/swallow_dataset/traverse",
}

SWALLOW_ORIGIN_CIF_CONFIG = {
    "slot_count": 1000,
    "time_window_multiplier": 3.0,
    "min_slot_duration": 0.02,
    "max_slot_duration": 0.08,
}

SWALLOW_SINGLE_CIF_CONFIG = {
    "slot_count": 2700,
    "time_window_multiplier": 3.0,
    "min_slot_duration": 0.0074074,
    "max_slot_duration": 0.0296296,
}

SWALLOW_MULTI_CIF_CONFIG = {
    "slot_count": 7200,
    "time_window_multiplier": 3.0,
    "min_slot_duration": 0.0055555,
    "max_slot_duration": 0.0444444,
}


def swallow_cif_config(config: dict) -> dict:
    dataset_name = Path(config["dataset"]["path"]).name.lower()
    if dataset_name.endswith("-swallow-origin"):
        return SWALLOW_ORIGIN_CIF_CONFIG
    if dataset_name.endswith("-swallow-single"):
        return SWALLOW_SINGLE_CIF_CONFIG
    if dataset_name.endswith("-swallow-multi"):
        return SWALLOW_MULTI_CIF_CONFIG
    raise ValueError(f"unsupported Swallow dataset variant: {config['dataset']['path']}")

TRAVERSE_MAX_BURSTS = 300
TRAVERSE_PROMPT_PREFIX = (
    "Instruction:\n"
    "Given a traffic burst sequence as input, predict the website label.\n"
    "Input:\n"
    "<flow>: <burst sequence>: "
)
TRAVERSE_PROMPT_SUFFIX = "."


def load_config(model_dir: Path) -> dict:
    with (model_dir / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    return config


def week_ranges(first: int, last: int, size: int) -> list[tuple[int, int]]:
    return [(start, min(start + size - 1, last)) for start in range(first, last + 1, size)]


def period_ids(days: np.ndarray, weeks: list[tuple[int, int]]) -> np.ndarray:
    result = np.full(days.shape, -1, dtype=np.int16)
    for period, (start, end) in enumerate(weeks, start=1):
        result[(days >= start) & (days <= end)] = period
    return result


def scan_counts(path: Path, reference_end: int, weeks: list[tuple[int, int]], chunk: int,
                min_trace_length: int):
    counts: list[defaultdict[bytes, int]] = [defaultdict(int) for _ in range(len(weeks) + 1)]
    with h5py.File(path, "r", swmr=True) as source:
        total = len(source["day"])
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            days = np.asarray(source["day"][start:end])
            labels = np.asarray(source["labels"][start:end])
            valid_lengths = np.asarray(source["lengths"][start:end]) >= min_trace_length
            periods = period_ids(days, weeks)
            for period in range(len(counts)):
                mask = (days <= reference_end) if period == 0 else (periods == period)
                selected = labels[mask & valid_lengths]
                if not len(selected):
                    continue
                values, occurrences = np.unique(selected, return_counts=True)
                for label, count in zip(values, occurrences):
                    counts[period][bytes(label)] += int(count)
            print(f"count scan: {end:,}/{total:,}", end="\r", flush=True)
    print()
    return counts


def choose_sites(counts, weeks: list[tuple[int, int]], min_reference: int,
                 min_week: int, website_count: int) -> tuple[list[bytes], int]:
    eligible = set(label for label, count in counts[0].items() if count >= min_reference)

    qualified_per_week = [
        {
            label
            for label, count in counts[period].items()
            if count >= min_week
        }
        for period in range(1, len(weeks) + 1)
    ]
    eligible &= set.intersection(*qualified_per_week)

    eligible_count = len(eligible)
    if website_count > eligible_count:
        raise ValueError(
            f"website-count {website_count} exceeds {eligible_count} eligible websites"
        )
    totals = {
        site: sum(period_counts.get(site, 0) for period_counts in counts)
        for site in eligible
    }
    selected = sorted(
        sorted(eligible, key=lambda site: (-totals[site], site))[:website_count]
    )
    return selected, eligible_count


def sample_indices(path: Path, sites: list[bytes], reference_end: int,
                   weeks: list[tuple[int, int]], chunk: int, min_trace_length: int,
                   max_samples_per_site: int, seed: int):
    selected_sites = np.asarray(sites)
    candidates: dict[tuple[int, bytes], list[int]] = defaultdict(list)
    with h5py.File(path, "r", swmr=True) as source:
        total = len(source["day"])
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            days = np.asarray(source["day"][start:end])
            labels = np.asarray(source["labels"][start:end])
            valid_lengths = np.asarray(source["lengths"][start:end]) >= min_trace_length
            periods = period_ids(days, weeks)
            selected = np.isin(labels, selected_sites) & valid_lengths
            for period in range(len(weeks) + 1):
                mask = (days <= reference_end) if period == 0 else (periods == period)
                for offset in np.flatnonzero(selected & mask):
                    source_index = start + int(offset)
                    key = (period, bytes(labels[offset]))
                    candidates[key].append(source_index)
            print(f"sample scan: {end:,}/{total:,}", end="\r", flush=True)
    print()

    rng = np.random.default_rng(seed)
    sampled = {}
    for key, indices in candidates.items():
        if max_samples_per_site and len(indices) > max_samples_per_site:
            selected = rng.choice(indices, size=max_samples_per_site, replace=False)
            sampled[key] = sorted(int(index) for index in selected)
        else:
            sampled[key] = indices
    return sampled


class TraceDataset(Dataset):
    def __init__(self, path: Path, records: list[tuple[int, int, bytes]],
                 method: str, feature_config: dict):
        self.path, self.records = str(path), records
        self.method, self.config = method, feature_config
        self._file = None

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def source(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r", swmr=True)
        return self._file

    def _scsm_matrix(self, times: np.ndarray, directions: np.ndarray, slot: float):
        matrix_length = int(self.config["max_matrix_length"])
        maximum_cells = int(self.config["maximum_cell_number"])
        feature = np.zeros((2 * (maximum_cells + 2), matrix_length), dtype=np.float32)
        columns = np.floor(times / slot).astype(np.int64).clip(0, matrix_length - 1)
        rows = (np.sign(directions) > 0).astype(np.int64)
        np.add.at(feature, (rows, columns), 1)
        unique, starts = np.unique(columns, return_index=True)
        feature[2 * maximum_cells + 2, unique[1:]] = np.diff(unique)
        threshold = slot * float(self.config["time_interval_threshold"])
        for group, begin in enumerate(starts):
            finish = starts[group + 1] if group + 1 < len(starts) else len(times)
            feature[2 * maximum_cells + 3, unique[group]] = 1 + np.sum(
                np.diff(times[begin:finish]) > threshold
            )
        if bool(self.config.get("log_transform", True)):
            feature = np.log1p(feature)
        return torch.from_numpy(feature[None]), torch.tensor(
            int(unique[-1]), dtype=torch.float32
        )

    def __getitem__(self, logical_index: int):
        period, index, label = self.records[logical_index]
        source = self.source()
        length = int(source["lengths"][index])
        directions = np.asarray(source["directions"][index, :length], dtype=np.float32)
        times = np.asarray(source["times"][index, :length], dtype=np.float64)

        if self.method == "traverse":
            bursts = extract_bursts(directions, int(self.config["max_bursts"]))
            prompt = (self.config["prompt_prefix"]
                      + ", ".join(str(value) for value in bursts)
                      + self.config["prompt_suffix"])
            return prompt, period, label, index

        if self.method == "netclr":
            feature = extract_length(
                directions,
                int(self.config["input_length"]),
            ).astype(np.float32, copy=False)
            return (torch.from_numpy(feature), torch.tensor(0.0), period, label, index)

        if self.method == "swallow":
            feature, _, _ = extract_cif(
                times,
                directions,
                float(self.config["min_slot_duration"]),
                float(self.config["max_slot_duration"]),
                float(self.config["time_window_multiplier"]),
                int(self.config["slot_count"]),
            )
            feature = np.asarray(feature, dtype=np.float32)
            return (torch.from_numpy(feature), torch.tensor(0.0), period, label, index)

        if self.method == "scsm":
            times -= times[0]
            times[0] = 1e-6
            matrix_length = int(self.config["max_matrix_length"])
            strategy = self.config.get(
                "slot_strategy",
                "ensemble" if bool(self.config.get("slot_ensemble", False)) else "fixed",
            )
            if strategy == "ensemble":
                slots = np.linspace(
                    float(self.config["min_slot_time"]),
                    float(self.config["max_slot_time"]),
                    int(self.config["test_slot_count"]),
                )
                views = [self._scsm_matrix(times, directions, slot) for slot in slots]
                return (torch.stack([view[0] for view in views]),
                        torch.stack([view[1] for view in views]), period, label, index)
            if strategy == "adaptive":
                load_time = float(times[-1])
                slot = np.clip(
                    float(self.config["test_slot_multiplier"])
                    * load_time / matrix_length,
                    float(self.config["min_slot_time"]),
                    float(self.config["max_slot_time"]),
                )
            else:
                slot = float(self.config["maximum_load_time"]) / matrix_length
            feature, last_index = self._scsm_matrix(times, directions, slot)
            return feature, last_index, period, label, index


def find_model_weights(model_dir: Path) -> Path:
    """Prefer the exported encoder, then the latest available training checkpoint."""
    encoder = model_dir / "encoder.pt"
    if encoder.exists():
        return encoder
    final = model_dir / "checkpoint-final.pt"
    if final.exists():
        return final

    raise FileNotFoundError(f"no model weights found in {model_dir}")


def load_model(config: dict, model_dir: Path, device: torch.device,
               tokenizer=None) -> tuple[torch.nn.Module, Path]:
    method = config["method"].lower()
    if method == "scsm":
        model = SCSM(config["model"])
    elif method == "netclr":
        model = DFEncoder(config["model"])
    elif method == "swallow":
        model = SwallowNetwork(config["model"]).encoder
    else:
        if tokenizer is None:
            raise ValueError("TraVerse evaluation requires a tokenizer")
        model_config = dict(config["model"])
        pretrained = Path(model_config["pretrained_model"])
        if not pretrained.exists():
            local_pretrained = PROJECT_ROOT / "pretrain" / pretrained.name
            if not local_pretrained.exists():
                raise FileNotFoundError(f"pretrained model not found: {pretrained}")
            model_config["pretrained_model"] = str(local_pretrained)
        model = TraVerseMLM(model_config)
        model.model.resize_token_embeddings(len(tokenizer))
    weights_path = find_model_weights(model_dir)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), weights_path


class TraverseCollator:
    """Tokenize prompts and mark only traffic-feature tokens for pooling."""

    def __init__(self, tokenizer, max_length: int, prompt_prefix: str,
                 prompt_suffix: str):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix

    def __call__(self, samples):
        prompts, periods, labels, indices = zip(*samples)
        batch = self.tokenizer(
            list(prompts), padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = batch.pop("offset_mapping")
        starts, ends = offsets.unbind(dim=-1)
        traffic_start = len(self.prompt_prefix)
        traffic_ends = torch.tensor(
            [len(prompt) - len(self.prompt_suffix) for prompt in prompts]
        ).unsqueeze(1)
        traffic_mask = (
            batch["attention_mask"].bool()
            & (starts >= traffic_start)
            & (ends <= traffic_ends)
            & (ends > starts)
        )
        return batch, traffic_mask, torch.tensor(periods), list(labels), torch.tensor(indices)


def linear_position_pool(hidden: torch.Tensor, traffic_mask: torch.Tensor) -> torch.Tensor:
    """Pool valid traffic tokens with normalized weights L, L-1, ..., 1."""
    lengths = traffic_mask.sum(dim=1, keepdim=True)
    if torch.any(lengths == 0):
        raise ValueError("a TraVerse prompt contains no traffic tokens after tokenization")
    ranks = traffic_mask.cumsum(dim=1)
    weights = (lengths - ranks + 1).clamp_min(0) * traffic_mask
    weights = weights.to(hidden.dtype)
    weights /= weights.sum(dim=1, keepdim=True)
    return torch.sum(hidden * weights.unsqueeze(-1), dim=1)


def extract_embeddings(model, loader, device: torch.device, method: str):
    output: dict[tuple[int, bytes], list[np.ndarray]] = defaultdict(list)
    with torch.inference_mode():
        for batch_index, batch_data in enumerate(loader, start=1):
            if method == "traverse":
                tokenized, traffic_mask, periods, labels, indices = batch_data
                tokenized = {key: value.to(device, non_blocking=True)
                             for key, value in tokenized.items()}
                traffic_mask = traffic_mask.to(device, non_blocking=True)
            else:
                matrix, last_index, periods, labels, indices = batch_data
                matrix = matrix.to(device, non_blocking=True)
                last_index = last_index.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=False):
                if method == "scsm":
                    if matrix.ndim == 5:
                        views = matrix.shape[1]
                        features = None
                        for view_index in range(views):
                            view_features = model(
                                matrix[:, view_index], last_index[:, view_index]
                            )
                            features = (
                                view_features
                                if features is None
                                else features + view_features
                            )
                        features = features / views
                    else:
                        features = model(matrix, last_index)
                elif method == "swallow":
                    features = model(matrix.unsqueeze(1)).flatten(1)
                elif method == "traverse":
                    result = model.model(
                        **tokenized, output_hidden_states=True, return_dict=True,
                    )
                    features = linear_position_pool(result.hidden_states[-1], traffic_mask)
                else:
                    features = model(matrix)
                features = features.float()

            if not torch.isfinite(features).all():
                raise FloatingPointError(
                    f"non-finite {method} features in embedding batch {batch_index}"
                )
            features = F.normalize(features, dim=-1)
            for feature, period, label in zip(
                    features.cpu().numpy(), periods.tolist(), labels):
                key = (int(period), bytes(label))
                output[key].append(feature)
            print(f"embedding batches: {batch_index:,}/{len(loader):,}", end="\r", flush=True)
    print()
    return {key: np.asarray(value, dtype=np.float32) for key, value in output.items()}


def knn_predictions(query: np.ndarray, gallery: np.ndarray, gallery_labels: np.ndarray,
                    k: int, exclude_self: bool = False,
                    query_block_size: int = 128,
                    progress_desc: str = "kNN") -> np.ndarray:
    """Cosine kNN with majority vote and summed-similarity tie breaking."""
    if query.ndim != 2 or gallery.ndim != 2 or query.shape[1] != gallery.shape[1]:
        raise ValueError("query and gallery must be 2-D arrays with the same feature dimension")
    if len(gallery) != len(gallery_labels):
        raise ValueError("gallery and gallery_labels have incompatible lengths")
    if k <= 0 or query_block_size <= 0:
        raise ValueError("k and query_block_size must be positive")
    if exclude_self and len(query) != len(gallery):
        raise ValueError("leave-one-out kNN requires identical query and gallery arrays")
    available = len(gallery) - int(exclude_self)
    if available < 1:
        raise ValueError("kNN gallery has no eligible neighbor")
    actual_k = min(k, available)
    predictions = np.empty(len(query), dtype=np.int64)

    query_starts = range(0, len(query), query_block_size)
    for query_start in tqdm(
        query_starts,
        total=len(query_starts),
        desc=progress_desc,
        unit="block",
    ):
        query_end = min(query_start + query_block_size, len(query))
        similarities = query[query_start:query_end] @ gallery.T

        if exclude_self:
            rows = np.arange(query_end - query_start)
            similarities[rows, query_start + rows] = -np.inf

        finite_counts = np.isfinite(similarities).sum(axis=1)
        if np.any(finite_counts < actual_k):
            raise ValueError("a kNN query has fewer eligible neighbors than actual_k")

        partition = len(gallery) - actual_k
        neighbors = np.argpartition(similarities, partition, axis=1)[:, -actual_k:]
        for local_row, indices in enumerate(neighbors):
            labels = gallery_labels[indices]
            classes, votes = np.unique(labels, return_counts=True)
            finalists = classes[votes == votes.max()]
            output_row = query_start + local_row
            if len(finalists) == 1:
                predictions[output_row] = finalists[0]
            else:
                predictions[output_row] = max(
                    finalists,
                    key=lambda label: float(
                        similarities[local_row, indices[labels == label]].sum()
                    ),
                )
    return predictions


def macro_accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean([
        np.mean(predictions[targets == label] == label) for label in np.unique(targets)
    ]))


def same_site_pairwise_cosine(features: np.ndarray) -> float:
    count = len(features)
    if count < 2:
        return float("nan")
    return (float(np.square(features.sum(axis=0)).sum()) - count) / (count * (count - 1))


def evaluate(embeddings, sites: list[bytes], weeks: list[tuple[int, int]],
             min_week: int, knn_k: int, knn_query_block_size: int):
    reference_sites = [site for site in sites if (0, site) in embeddings]
    centroids = np.stack([embeddings[(0, site)].mean(axis=0) for site in reference_sites])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-12)
    site_ids = {site: index for index, site in enumerate(reference_sites)}
    gallery_parts, gallery_label_parts = [], []
    for site in reference_sites:
        features = embeddings[(0, site)]
        gallery_parts.append(features)
        gallery_label_parts.append(np.full(len(features), site_ids[site], dtype=np.int64))
    reference_gallery = np.concatenate(gallery_parts)
    reference_gallery_labels = np.concatenate(gallery_label_parts)
    rows = []
    for period, (start, end) in enumerate(weeks, start=1):
        active = [site for site in reference_sites if len(embeddings.get((period, site), ())) >= min_week]
        if not active:
            continue
        query = np.concatenate([embeddings[(period, site)] for site in active])
        targets = np.concatenate([
            np.full(len(embeddings[(period, site)]), site_ids[site], dtype=np.int64)
            for site in active
        ])
        similarities = query @ centroids.T
        positive = similarities[np.arange(len(query)), targets]
        centroid_predictions = similarities.argmax(axis=1)
        reference_predictions = knn_predictions(
            query, reference_gallery, reference_gallery_labels, knn_k,
            query_block_size=knn_query_block_size,
            progress_desc=f"Week {period} reference kNN",
        )
        within_predictions = knn_predictions(
            query, query, targets, knn_k, exclude_self=True,
            query_block_size=knn_query_block_size,
            progress_desc=f"Week {period} within-week kNN",
        )
        per_site_cosine, within_site_cosine = [], []
        cursor = 0
        for site in active:
            count = len(embeddings[(period, site)])
            per_site_cosine.append(float(positive[cursor:cursor + count].mean()))
            within_site_cosine.append(same_site_pairwise_cosine(embeddings[(period, site)]))
            cursor += count
        rows.append({
            "week": period, "start_day": start, "end_day": end,
            "sites": len(active), "samples": len(query),
            "site_coverage": len(active) / len(reference_sites),
            "与参考时期质心余弦相似度": float(np.mean(per_site_cosine)),
            "同时期样本平均余弦相似度": float(np.nanmean(within_site_cosine)),
            "质心分类准确率 (macro)": macro_accuracy(targets, centroid_predictions),
            f"与参考时期样本的KNN_{knn_k}": macro_accuracy(targets, reference_predictions),
            f"同时期样本的KNN_{knn_k}": macro_accuracy(targets, within_predictions),
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    # Preserve the original default: evaluate all GTT-pretrained models.
    # SwallowDataset-pretrained models are selected explicitly with their aliases.
    model_specs = [
        item for item in MODEL_DIRS.items()
        if not item[0].startswith("swallow_dataset_")
    ]
    if args.models is not None:
        requested = set(args.models)
        available = set(MODEL_DIRS)
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError(
                f"unknown models {unknown}; available models: {sorted(available)}"
            )
        model_specs = [
            (model_name, model_dir)
            for model_name, model_dir in MODEL_DIRS.items()
            if model_name in requested
        ]
    print(f"Selected models: {[model_name for model_name, _ in model_specs]}")

    dataset_kinds = {
        "swallow" if model_name.startswith("swallow_dataset_") else "gtt"
        for model_name, _ in model_specs
    }
    if len(dataset_kinds) != 1:
        raise ValueError(
            "one evaluation command cannot mix GTT-pretrained and "
            "SwallowDataset-pretrained models; run the two groups separately"
        )
    dataset_kind = dataset_kinds.pop()
    evaluation_dataset = PROJECT_ROOT / "pretrain_dataset" / (
        "Swallow_train.hdf5" if dataset_kind == "swallow" else "GTT23_train.hdf5"
    )
    print(f"Evaluation dataset: {evaluation_dataset}")

    # 各模型的配置
    configs = [
        (model_name, model_dir, load_config(model_dir))
        for model_name, model_dir in model_specs
    ]

    # 时间分段设置
    configured_max_days = {
        int(config["dataset"]["max_day"]) for _, _, config in configs
    }
    assert len(configured_max_days) == 1, "all models must have the same max_day in their config.json"
    reference_end = configured_max_days.pop()
    with h5py.File(evaluation_dataset, "r", swmr=True) as source:
        dataset_last_day = int(np.max(source["day"]))
    weeks = week_ranges(reference_end + 1, dataset_last_day, 7)

    # 计算各period的各站点样本数量，其中reference_end之前的统一作为period 0，之后的每周作为一个period
    # 只统计长度>=min_trace_length的样本
    counts = scan_counts(
        evaluation_dataset, reference_end, weeks, 262144,
        args.min_trace_length,
    )
    if dataset_kind == "swallow":
        # Swallow_train.hdf5 has already been normalized to the D1 website set
        # by pretrain_dataset/process_swallow.py.  Do not apply GTT's website
        # thresholds or Top-N selection a second time.
        sites = sorted(set().union(*(period_counts.keys() for period_counts in counts)))
        missing = {
            period: [site for site in sites if counts[period].get(site, 0) == 0]
            for period in range(len(weeks) + 1)
        }
        missing = {period: labels for period, labels in missing.items() if labels}
        if missing:
            raise ValueError(
                "Swallow contains websites with no valid trace after "
                f"min-trace-length filtering: {missing}"
            )
        eligible_count = len(sites)
        print(f"Swallow 使用全部网站，不执行阈值或 Top-N 筛选: {len(sites)}")
    else:
        # GTT website selection:
        # reference samples >= min_reference_samples, every prediction week
        # >= min_week_samples, followed by Top-N effective sample count.
        sites, eligible_count = choose_sites(
            counts, weeks, args.min_reference_samples, args.min_week_samples,
            args.website_count,
        )
        print(f"按 reference/每周阈值筛选后的网站数量: {eligible_count}")
        print(
            f"按有效样本总数选择 Top-{args.website_count} 后的网站数量: "
            f"{len(sites)}"
        )
    # 收集符合条件站点的全部有效样本下标，保留长度>=min_trace_length的样本
    sampled = sample_indices(evaluation_dataset, sites,
                             reference_end, weeks, 262144, args.min_trace_length,
                             args.max_samples_per_site, args.seed)
    period_sample_counts = [0] * (len(weeks) + 1)
    for (period, _), indices in sampled.items():
        period_sample_counts[period] += len(indices)
    print(f"参考期 (Day <= {reference_end}) 样本数: {period_sample_counts[0]:,}")
    for period, (start_day, end_day) in enumerate(weeks, start=1):
        print(
            f"Week {period} (Day {start_day}-{end_day}) 样本数: "
            f"{period_sample_counts[period]:,}"
        )
    print(f"测试期总样本数: {sum(period_sample_counts[1:]):,}")
    print(f"参考期与测试期总样本数: {sum(period_sample_counts):,}")

    records = [(period, index, label) for (period, label), indices in sampled.items() for index in indices]
    records.sort()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"selected_websites_{dataset_kind}.json").write_text(
        json.dumps(
            {
                "eligible_count": eligible_count,
                "selected_count": len(sites),
                "websites": [site.decode("utf-8") for site in sites],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    evaluation_runs = []
    for model_name, model_dir, config in configs:
        method = config["method"].lower()
        if (method == "scsm" and bool(config["model"].get("simple_mode"))
                and args.scsm_slot_strategy in {"both", "all"}):
            strategies = (
                ("fixed", "adaptive", "ensemble")
                if args.scsm_slot_strategy == "all"
                else ("fixed", "ensemble")
            )
            evaluation_runs.extend(
                (model_name, model_dir, config, strategy) for strategy in strategies
            )
        else:
            strategy = (
                "ensemble" if method == "scsm" and args.scsm_slot_strategy in {"both", "all"}
                else args.scsm_slot_strategy if method == "scsm" else None
            )
            evaluation_runs.append((model_name, model_dir, config, strategy))

    for model_name, model_dir, config, scsm_slot_strategy in evaluation_runs:
        print(f"evaluating model: {model_name} ({model_dir})")
        method = config["method"].lower()
        if method == "scsm":
            feature_config = {
                **config["model"], **config["augmentation"], "mask": False,
                "slot_strategy": scsm_slot_strategy,
                "slot_ensemble": scsm_slot_strategy == "ensemble",
                "test_slot_multiplier": 3.0,
                "test_slot_count": 5,
            }
            print(f"SCSM slot strategy: {scsm_slot_strategy}")
        elif method == "netclr":
            feature_config = config["model"]
        elif method == "traverse":
            feature_config = {
                "max_bursts": TRAVERSE_MAX_BURSTS,
                "prompt_prefix": TRAVERSE_PROMPT_PREFIX,
                "prompt_suffix": TRAVERSE_PROMPT_SUFFIX
            }
        elif method == "swallow":
            feature_config = swallow_cif_config(config)
        else:
            raise ValueError(f"unsupported evaluation method: {method}")
        dataset = TraceDataset(evaluation_dataset, records, method, feature_config)
        tokenizer = None
        collate_fn = None
        batch_size = args.batch_size
        if method == "traverse":
            pretrained = PROJECT_ROOT / "pretrain" / Path(config["model"]["pretrained_model"]).name
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained,
                trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
            )
            if tokenizer.mask_token_id is None:
                tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            collate_fn = TraverseCollator(
                tokenizer, int(config["dataset"]["max_length"]),
                feature_config["prompt_prefix"], feature_config["prompt_suffix"],
            )
            batch_size = min(batch_size, int(config["dataset"].get("batch_size", batch_size)))
            print(f"TraVerse evaluation batch size: {batch_size}")
        loader_options = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": args.num_workers,
            "pin_memory": args.device.startswith("cuda"),
            "persistent_workers": False,
            "collate_fn": collate_fn,
        }
        if args.num_workers > 0:
            # Forking after Mamba/Triton has initialized LLVM can abort workers
            # when evaluation advances to the next model.
            loader_options["multiprocessing_context"] = "spawn"
        loader = DataLoader(dataset, **loader_options)
        model, weights_path = load_model(config, model_dir, device, tokenizer)
        print(f"loaded weights: {weights_path}")
        embeddings = extract_embeddings(model, loader, device, method)
        evaluation_min_week = 1 if dataset_kind == "swallow" else args.min_week_samples
        rows = evaluate(embeddings, sites, weeks, evaluation_min_week,
                        args.knn_k, args.knn_query_block_size)
        result_model_name = model_name
        if method == "scsm" and args.scsm_slot_strategy in {"both", "all"}:
            result_model_name = f"{model_name}-{scsm_slot_strategy}"
        rows = [{"model": result_model_name, **row} for row in rows]
        model_output = args.output_dir / model_name
        model_output.mkdir(parents=True, exist_ok=True)
        output_name = (
            f"weekly_metrics_{scsm_slot_strategy}.csv"
            if method == "scsm" else "weekly_metrics.csv"
        )
        write_csv(model_output / output_name, rows)
        del model, embeddings, loader
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
