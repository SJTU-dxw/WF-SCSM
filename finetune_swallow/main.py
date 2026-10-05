"""SwallowDataset adapter; reuse finetune training without editing its files."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import h5py
import numpy as np
from finetune import main as training
from finetune import models
from finetune_swallow.data import save_split

BASE = ROOT / "finetune_swallow"
PRETRAIN_ROOT = ROOT / "output/pretrain/swallow_dataset"
SETTINGS = {
    **training.SETTINGS,
    "dataset": ROOT / "pretrain_dataset/Swallow_train.hdf5",
    "min_reference_samples": 1,
    "min_week_samples": 6,
    "split_dir": BASE / "splits",
    "output_dir": BASE / "results/all_remaining",
}
_parse_args = training.parse_args


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(arg == "--k" or arg.startswith("--k=") for arg in argv):
        argv.extend(["--k", "10"])
    if not any(arg == "--website-count" or arg.startswith("--website-count=") for arg in argv):
        argv.extend(["--website-count", "100"])
    if not any(arg == "--min-trace-length" or arg.startswith("--min-trace-length=") for arg in argv):
        argv.extend(["--min-trace-length", "1"])
    # Restore the imported module's settings immediately after parsing.
    with patch.object(training, "SETTINGS", SETTINGS):
        args = _parse_args(argv)
    if args.website_count != 100 or args.min_trace_length != 1:
        raise ValueError("This entry does not support filtering: website-count must be 100 and min-trace-length must be 1")
    args.website_count = None  # Disable top-N selection in the shared splitter.
    args.test_per_class = None  # Variable-size test set: every non-training trace.
    if args.model in ("scsm-single", "netclr", "swallow-origin", "swallow-single", "traverse"):
        args.model_dir = PRETRAIN_ROOT / {"scsm-single": "scsm_single"}.get(args.model, args.model)
    return args


def prepare_split(args):
    """Validate the same 100 pre-filtered website classes in every period."""
    expected = {str(i).encode("ascii") for i in range(100)}
    with h5py.File(args.dataset, "r") as source:
        if not source.attrs.get("complete", True):
            raise ValueError("HDF5 conversion is incomplete")
        days, labels, lengths = (source[key][:] for key in ("day", "labels", "lengths"))
    if np.any(lengths < 1):
        raise ValueError("Empty traces are unsupported; refusing to silently filter them")
    if set(np.unique(days).tolist()) != {49, 56, 63, 70, 77, 84, 91}:
        raise ValueError("Unexpected SwallowDataset collection days")
    for day in np.unique(days):
        values, counts = np.unique(labels[days == day], return_counts=True)
        if set(values.tolist()) != expected:
            raise ValueError(
                f"Expected website labels 0-99 at day {day}; "
                "regenerate Swallow_train.hdf5 with the current preprocessing script"
            )
        if day != 49 and np.any(counts <= args.k):
            raise ValueError(f"Insufficient train/test samples at day {day}")
    return save_split(args, days, labels)


def build_model(method, classes, device, model_dir=None, freeze=False):
    if method == "traverse":
        folder = Path(model_dir) if model_dir else PRETRAIN_ROOT / "traverse"
        pretrained = json.loads((folder / "config.json").read_text())
        if pretrained["method"] != "traverse" or Path(pretrained["dataset"]["path"]).name != "Swallow-TraVerse":
            raise ValueError("Expected SwallowDataset TraVerse checkpoint")
        if pretrained["dataset"]["max_day"] > 49:
            raise ValueError("TraVerse pretraining overlaps evaluation stages")
        return models.build_model(method, classes, device, folder, freeze)
    if method not in ("swallow-origin", "swallow-single"):
        return models.build_model(method, classes, device, model_dir, freeze)
    # The original builder's Swallow CIF checks are specific to GTT23.
    # Keep SwallowDataset validation here and reuse the encoder loader/head.
    from feature_similarity.evaluate import load_model, SWALLOW_ORIGIN_CIF_CONFIG, SWALLOW_SINGLE_CIF_CONFIG
    variant = method.removeprefix("swallow-")
    folder = Path(model_dir) if model_dir else PRETRAIN_ROOT / method
    pretrained = json.loads((folder / "config.json").read_text())
    expected_dataset = f"Swallow-Swallow-{variant}"
    if pretrained["method"] != "swallow" or Path(pretrained["dataset"]["path"]).name != expected_dataset:
        raise ValueError(f"Expected SwallowDataset {variant}-CIF checkpoint")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and resolved_device.index is not None:
        torch.cuda.set_device(resolved_device)
    encoder, source = load_model(pretrained, folder, torch.device("cpu"))
    dim = 2048 if pretrained["model"]["name"] == "resnet50" else 512
    model = models.Classifier(encoder, dim, classes, method)
    if freeze:
        model.encoder.requires_grad_(False)
        model.freeze = True
    cif_config = SWALLOW_ORIGIN_CIF_CONFIG if method == "swallow-origin" else SWALLOW_SINGLE_CIF_CONFIG
    config = {**cif_config, "pretrained_config": pretrained}
    return model.to(device), config, None, str(source)


def main():
    # Only this process's invocation uses the adapter; original files stay intact.
    with patch.object(training, "parse_args", parse_args), patch.object(training, "build_model", build_model), patch.object(training, "prepare_split", prepare_split):
        training.main()


if __name__ == "__main__":
    main()
