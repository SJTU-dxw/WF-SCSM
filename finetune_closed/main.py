"""Fine-tune all supported methods on one Closed-World dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from finetune import main as training
from finetune import models
from finetune_closed.data import prepare_split


BASE = ROOT / "finetune_closed"
DATASETS = {
    "df": ROOT / "pretrain_dataset/DF_train.hdf5",
    "r-precision": ROOT / "pretrain_dataset/R-Precision_train.hdf5",
    "w-t": ROOT / "pretrain_dataset/W-T_train.hdf5",
    "k-nn": ROOT / "pretrain_dataset/k-NN_train.hdf5",
}
PRETRAIN_ROOTS = {
    "gtt": ROOT / "output/pretrain/GTT_dataset",
    "swallow": ROOT / "output/pretrain/swallow_dataset",
}
PRETRAINED_METHODS = {
    "scsm-single",
    "netclr",
    "swallow-origin",
    "swallow-single",
    "traverse",
}
MODEL_DIRECTORIES = {"scsm-single": "scsm_single"}
_shared_parse_args = training.parse_args


def closed_name(value: str) -> str:
    value = value.lower().replace("_", "-")
    aliases = {"rprecision": "r-precision", "wt": "w-t", "knn": "k-nn"}
    return aliases.get(value, value)


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "-h" in argv or "--help" in argv:
        print(
            "Closed-World options:\n"
            "  --dataset-name {df,r-precision,w-t,k-nn}  required dataset\n"
            "  --pretrain-source {none,gtt,swallow}       checkpoint family\n"
        )
        _shared_parse_args(["--help"])
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--dataset-name", type=closed_name, choices=DATASETS, required=True)
    custom.add_argument(
        "--pretrain-source",
        choices=("none", "gtt", "swallow"),
        default="none",
        help="checkpoint family for pretrained methods; baselines require none",
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

    args.dataset_name = selected.dataset_name
    args.pretrain_source = selected.pretrain_source
    args.dataset = DATASETS[selected.dataset_name]
    args.reference_end = 49
    args.min_reference_samples = 1
    args.min_week_samples = args.k + 1
    args.test_per_class = None
    args.website_count = None
    args.split_dir = BASE / "splits" / selected.dataset_name
    args.output_dir = (
        BASE / "results" / selected.dataset_name / selected.pretrain_source
    )
    if is_pretrained:
        folder = MODEL_DIRECTORIES.get(args.model, args.model)
        args.model_dir = PRETRAIN_ROOTS[selected.pretrain_source] / folder
    else:
        args.model_dir = None
    return args


def validate_pretrained(method: str, folder: Path, source: str) -> dict:
    config_path = folder / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing {source} checkpoint config: {config_path}")
    pretrained = json.loads(config_path.read_text())
    expected_method = {
        "scsm-single": "scsm",
        "swallow-origin": "swallow",
        "swallow-single": "swallow",
    }.get(method, method)
    if pretrained.get("method") != expected_method:
        raise ValueError(f"{folder}: expected method {expected_method}")
    if method == "scsm-single" and not pretrained.get("model", {}).get("simple_mode"):
        raise ValueError(f"{folder}: expected an SCSM single checkpoint")

    expected_datasets = {
        "gtt": {
            "scsm-single": "GTT23_train.hdf5",
            "netclr": "GTT23-NetCLR",
            "swallow-origin": "GTT23-Swallow-origin",
            "swallow-single": "GTT23-Swallow-single",
            "traverse": "GTT23-TraVerse",
        },
        "swallow": {
            "scsm-single": "Swallow_train.hdf5",
            "netclr": "Swallow-NetCLR",
            "swallow-origin": "Swallow-Swallow-origin",
            "swallow-single": "Swallow-Swallow-single",
            "traverse": "Swallow-TraVerse",
        },
    }
    actual = Path(pretrained.get("dataset", {}).get("path", "")).name
    expected = expected_datasets[source][method]
    if actual.lower() != expected.lower():
        raise ValueError(f"{folder}: expected dataset {expected}, got {actual}")
    if pretrained["dataset"].get("max_day", 0) > 49:
        raise ValueError(f"{folder}: pretraining extends beyond day 49")
    return pretrained


def build_model(method, classes, device, model_dir=None, freeze=False):
    if method not in PRETRAINED_METHODS:
        return models.build_model(method, classes, device, model_dir, freeze)
    folder = Path(model_dir)
    source = "swallow" if folder.parent.name == "swallow_dataset" else "gtt"
    pretrained = validate_pretrained(method, folder, source)
    if source == "gtt" or method not in {"swallow-origin", "swallow-single"}:
        return models.build_model(method, classes, device, folder, freeze)

    # The shared builder deliberately validates GTT CIF paths. Reuse its
    # encoder/head construction here after validating the SwallowDataset path.
    from feature_similarity.evaluate import (
        SWALLOW_ORIGIN_CIF_CONFIG,
        SWALLOW_SINGLE_CIF_CONFIG,
        load_model,
    )

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and resolved_device.index is not None:
        torch.cuda.set_device(resolved_device)
    encoder, checkpoint = load_model(pretrained, folder, torch.device("cpu"))
    dim = 2048 if pretrained["model"]["name"] == "resnet50" else 512
    model = models.Classifier(encoder, dim, classes, method)
    if freeze:
        model.encoder.requires_grad_(False)
        model.freeze = True
    cif = (
        SWALLOW_ORIGIN_CIF_CONFIG
        if method == "swallow-origin"
        else SWALLOW_SINGLE_CIF_CONFIG
    )
    config = {**cif, "pretrained_config": pretrained}
    return model.to(device), config, None, str(checkpoint)


def main():
    with patch.object(training, "parse_args", parse_args), patch.object(
        training, "prepare_split", prepare_split
    ), patch.object(training, "build_model", build_model):
        training.main()


if __name__ == "__main__":
    main()
