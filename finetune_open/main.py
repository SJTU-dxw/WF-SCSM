"""Fine-tune a supported method on DF Open-World and compute pi_20 curves."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from finetune import main as training
from finetune_closed.main import PRETRAINED_METHODS, build_model
from finetune_open.data import OPEN_WORLD_LABEL, prepare_split, use_preselected_train
from finetune_open.metrics import serializable_curve, threshold_curve


BASE = ROOT / "finetune_open"
DATASET = ROOT / "pretrain_dataset/OW_train.hdf5"
PRETRAIN_ROOTS = {
    "gtt": ROOT / "output/pretrain/GTT_dataset",
    "swallow": ROOT / "output/pretrain/swallow_dataset",
}
MODEL_DIRECTORIES = {"scsm-single": "scsm_single"}
THRESHOLDS = np.linspace(0.0, 1.0, 1001, dtype=np.float64)
R_PRECISION_RATIO = 20.0
_shared_parse_args = training.parse_args
_active_args = None


def parse_args(argv=None):
    global _active_args
    argv = list(sys.argv[1:] if argv is None else argv)
    if "-h" in argv or "--help" in argv:
        print(
            "Open-World options:\n"
            "  --pretrain-source {none,gtt,swallow}  checkpoint family\n"
            "Threshold protocol: 1001 thresholds over [0,1], Holmes r=20.\n"
        )
        _shared_parse_args(["--help"])
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument(
        "--pretrain-source",
        choices=("none", "gtt", "swallow"),
        default="none",
    )
    selected, remaining = custom.parse_known_args(argv)
    if not any(arg == "--k" or arg.startswith("--k=") for arg in remaining):
        remaining.extend(["--k", "10"])
    if not any(
        arg == "--min-trace-length" or arg.startswith("--min-trace-length=")
        for arg in remaining
    ):
        remaining.extend(["--min-trace-length", "1"])
    args = _shared_parse_args(remaining)
    is_pretrained = args.model in PRETRAINED_METHODS
    if is_pretrained and selected.pretrain_source == "none":
        raise ValueError(f"{args.model} requires --pretrain-source gtt or swallow")
    if not is_pretrained and selected.pretrain_source != "none":
        raise ValueError(f"{args.model} is not pretrained; use --pretrain-source none")

    args.dataset_name = "df-open-world"
    args.pretrain_source = selected.pretrain_source
    args.dataset = DATASET
    args.reference_end = 49
    args.min_reference_samples = 1
    args.min_week_samples = args.k + 1
    args.test_per_class = None
    args.website_count = None
    args.split_dir = BASE / "splits"
    args.output_dir = BASE / "results" / selected.pretrain_source
    args.open_world_label = OPEN_WORLD_LABEL
    args.r_precision_ratio = R_PRECISION_RATIO
    args.threshold_count = len(THRESHOLDS)
    if is_pretrained:
        folder = MODEL_DIRECTORIES.get(args.model, args.model)
        args.model_dir = PRETRAIN_ROOTS[selected.pretrain_source] / folder
        config_path = args.model_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"missing {selected.pretrain_source} checkpoint config: {config_path}"
            )
    else:
        args.model_dir = None
    _active_args = args
    return args


def output_directory(args) -> Path:
    strategy = (
        "ensemblerange5"
        if args.test_slot_strategy == "ensemble"
        else args.test_slot_strategy
    )
    test_suffix = (
        f"_{strategy}test" if args.model == "scsm-single" else ""
    )
    name = (
        f"k{args.k}_seed{args.seed}_minlen{args.min_trace_length}"
        f"{'_randomslot' if args.random_slot else ''}"
        f"{test_suffix}"
        f"{'_freeze' if args.freeze else ''}"
    )
    return Path(args.output_dir) / args.model / name / "week01"


@torch.no_grad()
def evaluate_open_world(model, loader, device, classes):
    if _active_args is None:
        raise RuntimeError("Open-World arguments were not initialized")
    # The active split key is persisted in the training config later; class 95
    # is the final class for this validated 0..95 dataset.
    open_world_index = classes - 1
    model.eval()
    confusion = np.zeros((classes, classes), dtype=np.int64)
    predictions = []
    targets = []
    monitored_predictions = []
    monitored_confidence = []
    top3_hits = 0
    for x, extra, y in loader:
        if isinstance(x, torch.Tensor) and x.ndim == 5:
            batch_size, views = x.shape[:2]
            logits = model(
                training.move(x.flatten(0, 1), device),
                training.move(extra.flatten(0, 1), device),
            ).reshape(batch_size, views, -1).mean(1)
        else:
            logits = model(training.move(x, device), training.move(extra, device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation logits")
        probabilities = logits.softmax(-1)
        prediction = logits.argmax(-1).cpu().numpy()
        target = y.numpy()
        monitored_prediction = probabilities[:, :open_world_index].argmax(-1).cpu().numpy()
        confidence = (1.0 - probabilities[:, open_world_index]).cpu().numpy()
        top3 = logits.topk(min(3, classes), dim=-1).indices.cpu().numpy()
        top3_hits += int((top3 == target[:, None]).any(1).sum())
        np.add.at(confusion, (target, prediction), 1)
        predictions.extend(prediction.tolist())
        targets.extend(target.tolist())
        monitored_predictions.extend(monitored_prediction.tolist())
        monitored_confidence.extend(confidence.tolist())

    result = training.metrics(confusion)
    result["accuracy_at_3"] = top3_hits / int(confusion.sum())
    targets_array = np.asarray(targets, dtype=np.int64)
    monitored_predictions_array = np.asarray(monitored_predictions, dtype=np.int64)
    monitored_confidence_array = np.asarray(monitored_confidence, dtype=np.float64)
    curve = threshold_curve(
        targets_array,
        monitored_predictions_array,
        monitored_confidence_array,
        open_world_index,
        THRESHOLDS,
        R_PRECISION_RATIO,
    )
    strategy = loader.dataset.config.get("test_slot_strategy", "fixed")
    multi_strategy = (
        _active_args.model == "scsm-single"
        and _active_args.test_slot_strategy == "all"
    )
    suffix = f"_{strategy}" if multi_strategy else ""
    destination = output_directory(_active_args)
    destination.mkdir(parents=True, exist_ok=True)
    serialized = serializable_curve(curve)
    serialized.update(
        train_week=1,
        test_week=1,
        strategy=strategy,
        open_world_label=OPEN_WORLD_LABEL,
        open_world_class_index=open_world_index,
        confidence="1 - P(unmonitored class)",
        monitored_prediction="argmax probability over monitored classes",
        threshold_rule="accept as monitored when confidence >= threshold",
    )
    (destination / f"open_world_metrics{suffix}.json").write_text(
        json.dumps(serialized, indent=2)
    )
    np.savez_compressed(
        destination / f"open_world_predictions{suffix}.npz",
        targets=targets_array,
        monitored_predictions=monitored_predictions_array,
        monitored_confidence=monitored_confidence_array,
        **{key: value for key, value in curve.items() if isinstance(value, np.ndarray)},
    )
    return result, np.asarray(predictions), confusion


def main():
    with patch.object(training, "parse_args", parse_args), patch.object(
        training, "prepare_split", prepare_split
    ), patch.object(training, "build_model", build_model), patch.object(
        training, "evaluate", evaluate_open_world
    ), patch.object(
        training, "sample_train", use_preselected_train
    ):
        training.main()


if __name__ == "__main__":
    main()
