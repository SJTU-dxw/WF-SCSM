"""Evaluate saved Week 1 models on every week's held-out test set; no training."""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from finetune.data import TextCollator, WeeklyDataset
from finetune.main import cuda_device, evaluate, model_name
from finetune.models import DEFAULTS, build_model

BASE = Path(__file__).resolve().parent
MODELS = (
    "awf", "tmwf", "ares", "df", "tiktok", "varcnn", "rf", "countmamba",
    "netclr", "swallow-single", "traverse", "scsm-single",
)


def test_checkpoint(source, destination, device, workers):
    checkpoint_path = source / "final.pt"
    # Only load trusted local checkpoints produced by finetune/main.py.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = checkpoint["config"]
    if saved["week"] != 1:
        raise ValueError(f"Not a Week 1 checkpoint: {source}")
    split = Path(saved["split"])
    manifest = json.loads((split / "manifest.json").read_text())
    if saved["classes"] != manifest["classes"]:
        raise ValueError("Checkpoint/split class ordering mismatch")
    with np.load(split / "indices.npz") as archive:
        tests = archive["test"].copy()
    train_records = np.load(source / "train_indices.npy")
    if np.intersect1d(train_records[:, 0], tests[:, 0]).size:
        raise ValueError("Training records overlap test records")
    random.seed(saved["seed"])
    np.random.seed(saved["seed"])
    torch.manual_seed(saved["seed"])
    torch.cuda.manual_seed_all(saved["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    method = saved["model"]
    model, _, tokenizer, _ = build_model(
        method, len(saved["classes"]), device, saved["model_dir"], saved["freeze"]
    )
    if method == "traverse":
        # Frozen base/adapter weights are restored by build_model from pretraining.
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        buffers = {name for name, _ in model.named_buffers()}
        # named_buffers also includes non-persistent buffers (e.g. rotary caches),
        # which state_dict intentionally excludes and checkpoints need not save.
        required = (trainable | buffers) & set(model.state_dict())
        if set(checkpoint["model"]) != required:
            raise ValueError("TraVerse checkpoint lacks required classifier/buffer state")
        incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(incompatible.unexpected_keys)
    else:
        model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    features = dict(saved["features"])
    strategies = ("fixed", "adaptive", "ensemble") if method == "scsm-single" else ("fixed",)
    serialized_config = json.dumps(
        dict(source_checkpoint=str(checkpoint_path), train_week=1,
             test_weeks=list(range(1, len(manifest["weeks"]) + 1)),
             device=device, training_config=saved), indent=2, default=str
    )
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "config.json").write_text(serialized_config)
    results = {}
    for strategy in strategies:
        features["test_slot_strategy"] = strategy
        by_week = {}
        for week in range(1, len(manifest["weeks"]) + 1):
            records = tests[tests[:, 2] == week]
            if not len(records):
                raise ValueError(f"No test records for Week {week}")
            loader = DataLoader(
                WeeklyDataset(saved["dataset"], records, method, features, train=False),
                batch_size=saved["batch_size"], shuffle=False, num_workers=workers,
                collate_fn=TextCollator(tokenizer, features) if tokenizer else None,
                generator=torch.Generator().manual_seed(saved["seed"]),
            )
            metrics, predictions, confusion = evaluate(model, loader, device, len(saved["classes"]))
            output = destination / f"week{week:02d}"
            output.mkdir(exist_ok=True)
            suffix = f"_{strategy}" if method == "scsm-single" else ""
            (output / f"test_metrics{suffix}.json").write_text(json.dumps(
                dict(train_week=1, test_week=week, strategy=strategy, overall=metrics), indent=2
            ))
            np.savez_compressed(output / f"predictions{suffix}.npz",
                                records=records, predictions=predictions, confusion=confusion)
            by_week[str(week)] = metrics
            print(f"{method} {source.parent.name} -> Week {week} ({strategy}): {metrics}", flush=True)
        results[strategy] = by_week
    (destination / "test_metrics.json").write_text(json.dumps(
        dict(train_week=1, by_strategy=results), indent=2
    ))
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=cuda_device, default="cuda:3")
    parser.add_argument("--models", nargs="+", type=model_name, choices=MODELS, default=list(MODELS))
    parser.add_argument("--config", default="k15_seed0_sites200_minlen80")
    parser.add_argument("--results-root", type=Path, default=BASE / "results")
    parser.add_argument("--output-root", type=Path, default=BASE / "results_week1_alltest")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("num-workers must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tasks = []
    for method in args.models:
        variants = ("_randomslot_alltest", "_alltest") if method == "scsm-single" else (
            "_freeze" if method in ("traverse", "swallow-single") else "",
        )
        for variant in variants:
            name = args.config + variant
            source = args.results_root / method / name / "week01"
            destination = args.output_root / method / name
            for filename in ("final.pt", "train_indices.npy"):
                if not (source / filename).is_file():
                    raise FileNotFoundError(source / filename)
            if destination.exists():
                metadata_path = destination / "config.json"
                summary_path = destination / "test_metrics.json"
                complete = metadata_path.is_file() and summary_path.is_file()
                if complete:
                    metadata = json.loads(metadata_path.read_text())
                    complete = (
                        Path(metadata["source_checkpoint"]).resolve() == (source / "final.pt").resolve()
                        and metadata["train_week"] == 1
                    )
                    suffixes = ("_fixed", "_adaptive", "_ensemble") if method == "scsm-single" else ("",)
                    for week in metadata["test_weeks"]:
                        for suffix in suffixes:
                            for filename in (f"test_metrics{suffix}.json", f"predictions{suffix}.npz"):
                                complete &= (destination / f"week{week:02d}" / filename).is_file()
                if complete:
                    print(f"Skip completed: {destination}", flush=True)
                    continue
                raise FileExistsError(f"Refusing to overwrite incomplete/existing results: {destination}")
            tasks.append((source, destination))
    for source, destination in tasks:
        test_checkpoint(source, destination, args.device, args.num_workers)
    print(f"Saved {len(tasks)} Week 1 models' cross-week tests to {args.output_root}")


if __name__ == "__main__":
    main()
