"""Evaluate one saved GTT23 Week 1 checkpoint on every held-out week."""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from finetune.data import TextCollator, WeeklyDataset
from finetune.main import cuda_device, evaluate
from finetune.models import build_model


def existing_path(saved_path, fallback):
    path = Path(saved_path)
    return path if path.exists() else fallback


def resolve_split(saved, source):
    path = Path(saved["split"])
    if (path / "manifest.json").is_file() and (path / "indices.npz").is_file():
        return path
    matches = []
    for manifest_path in (ROOT / "finetune/splits").glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("classes") == saved["classes"] and len(manifest.get("weeks", [])) >= 1:
            matches.append(manifest_path.parent)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Cannot uniquely replace missing saved split {path} for {source}; matches={matches}"
        )
    return matches[0]


def strategies_for(saved):
    if saved["model"] != "scsm-single":
        return ("fixed",)
    strategy = saved.get("test_slot_strategy", "fixed")
    return ("fixed", "adaptive", "ensemble") if strategy == "all" else (strategy,)


def expected_files(destination, weeks, strategies, scsm):
    for week in weeks:
        for strategy in strategies:
            suffix = f"_{strategy}" if scsm else ""
            yield destination / f"week{week:02d}" / f"test_metrics{suffix}.json"
            yield destination / f"week{week:02d}" / f"predictions{suffix}.npz"


def load_checkpoint(source, device):
    checkpoint_path = source / "final.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = checkpoint["config"]
    if saved.get("week") != 1:
        raise ValueError(f"Not a Week 1 checkpoint: {checkpoint_path}")
    model, _, tokenizer, _ = build_model(
        saved["model"], len(saved["classes"]), device,
        saved.get("model_dir"), saved.get("freeze", False),
    )
    if saved["model"] == "traverse":
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        buffers = {name for name, _ in model.named_buffers()}
        required = (trainable | buffers) & set(model.state_dict())
        if set(checkpoint["model"]) != required:
            raise ValueError(f"TraVerse checkpoint state mismatch: {checkpoint_path}")
        incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(incompatible.unexpected_keys)
    else:
        model.load_state_dict(checkpoint["model"], strict=True)
    return model, tokenizer, saved


def run(source, destination, device, workers):
    checkpoint_path = source / "final.pt"
    train_indices_path = source / "train_indices.npy"
    for path in (checkpoint_path, source / "config.json", train_indices_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    model, tokenizer, saved = load_checkpoint(source, device)
    split = resolve_split(saved, source)
    manifest = json.loads((split / "manifest.json").read_text())
    if saved["classes"] != manifest["classes"]:
        raise ValueError(f"Checkpoint/split class ordering mismatch: {source}")
    with np.load(split / "indices.npz") as archive:
        tests = archive["test"].copy()
    train_records = np.load(train_indices_path)
    if np.intersect1d(train_records[:, 0], tests[:, 0]).size:
        raise ValueError(f"Training records overlap held-out tests: {source}")

    dataset = existing_path(saved["dataset"], ROOT / "pretrain_dataset" / Path(saved["dataset"]).name)
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    weeks = tuple(range(1, len(manifest["weeks"]) + 1))
    strategies = strategies_for(saved)
    scsm = saved["model"] == "scsm-single"
    source_stat = checkpoint_path.stat()
    metadata = dict(
        source_checkpoint=str(checkpoint_path.resolve()),
        source_size=source_stat.st_size,
        source_mtime_ns=source_stat.st_mtime_ns,
        train_week=1,
        test_weeks=list(weeks),
        strategies=list(strategies),
        resolved_dataset=str(dataset.resolve()),
        resolved_split=str(split.resolve()),
        training_config=saved,
    )
    config_path = destination / "config.json"
    if destination.exists():
        if not config_path.is_file() or json.loads(config_path.read_text()) != metadata:
            raise FileExistsError(f"Refusing to mix results in {destination}")
    else:
        destination.mkdir(parents=True)
        config_path.write_text(json.dumps(metadata, indent=2, default=str))

    random.seed(saved["seed"])
    np.random.seed(saved["seed"])
    torch.manual_seed(saved["seed"])
    torch.cuda.manual_seed_all(saved["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    features = dict(saved["features"])
    collate = TextCollator(tokenizer, features) if tokenizer else None

    for strategy in strategies:
        features["test_slot_strategy"] = strategy
        suffix = f"_{strategy}" if scsm else ""
        for week in weeks:
            output = destination / f"week{week:02d}"
            metrics_path = output / f"test_metrics{suffix}.json"
            predictions_path = output / f"predictions{suffix}.npz"
            if metrics_path.is_file() and predictions_path.is_file():
                print(f"Skip completed: {source.parent.parent.name}/{source.parent.name} Week {week} ({strategy})", flush=True)
                continue
            if metrics_path.exists() or predictions_path.exists():
                raise FileExistsError(f"Incomplete result pair in {output}")
            records = tests[tests[:, 2] == week]
            if not len(records):
                raise ValueError(f"No test records for Week {week}")
            loader = DataLoader(
                WeeklyDataset(dataset, records, saved["model"], features, train=False),
                batch_size=saved["batch_size"], shuffle=False, num_workers=workers,
                collate_fn=collate,
                generator=torch.Generator().manual_seed(saved["seed"]),
            )
            result, predictions, confusion = evaluate(
                model, loader, device, len(saved["classes"])
            )
            output.mkdir(exist_ok=True)
            metrics_path.write_text(json.dumps(dict(
                train_week=1, test_week=week, strategy=strategy, overall=result,
            ), indent=2))
            np.savez_compressed(
                predictions_path, records=records, predictions=predictions, confusion=confusion,
            )
            print(
                f"{saved['model']} {source.parent.name} -> Week {week} ({strategy}): {result}",
                flush=True,
            )

    summary = {}
    for strategy in strategies:
        suffix = f"_{strategy}" if scsm else ""
        summary[strategy] = {
            str(week): json.loads(
                (destination / f"week{week:02d}" / f"test_metrics{suffix}.json").read_text()
            )["overall"]
            for week in weeks
        }
    (destination / "test_metrics.json").write_text(json.dumps(
        dict(train_week=1, by_strategy=summary), indent=2,
    ))
    missing = [str(path) for path in expected_files(destination, weeks, strategies, scsm) if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing outputs after evaluation: {missing}")
    print(f"Saved complete cross-week evaluation to {destination}", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=cuda_device, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("num-workers must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    run(args.source.resolve(), args.output.resolve(), args.device, args.num_workers)


if __name__ == "__main__":
    main()
