"""Fine-tune one model on one defended DF dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finetune import main as training
from finetune_closed.main import PRETRAINED_METHODS, build_model
from finetune_defense.data import prepare_split


BASE = ROOT / "finetune_defense"
DATASETS = {
    "palette": ROOT / "pretrain_dataset/DF_Palette_train.hdf5",
    "wtfpad": ROOT / "pretrain_dataset/DF_WTF-PAD_train.hdf5",
    "front": ROOT / "pretrain_dataset/DF_FRONT_train.hdf5",
    "tamaraw": ROOT / "pretrain_dataset/DF_Tamaraw_train.hdf5",
    "regulator": ROOT / "pretrain_dataset/DF_RegulaTor_train.hdf5",
    "trafficsilver-rr": ROOT / "pretrain_dataset/DF_TrafficSilver_RR_train.hdf5",
    "trafficsilver-bd": ROOT / "pretrain_dataset/DF_TrafficSilver_BD_train.hdf5",
    "trafficsilver-bwr": ROOT / "pretrain_dataset/DF_TrafficSilver_BWR_train.hdf5",
}
PRETRAIN_ROOTS = {
    "gtt": ROOT / "output/pretrain/GTT_dataset",
    "swallow": ROOT / "output/pretrain/swallow_dataset",
}
MODEL_DIRECTORIES = {"scsm-single": "scsm_single"}
_shared_parse_args = training.parse_args


def defense_name(value: str) -> str:
    value = value.lower().replace("_", "-")
    aliases = {
        "wtf-pad": "wtfpad",
        "traffic-silver-rr": "trafficsilver-rr",
        "traffic-silver-bd": "trafficsilver-bd",
        "traffic-silver-bwr": "trafficsilver-bwr",
        "rr": "trafficsilver-rr",
        "bd": "trafficsilver-bd",
        "bwr": "trafficsilver-bwr",
    }
    return aliases.get(value, value)


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "-h" in argv or "--help" in argv:
        print(
            "Defense options:\n"
            "  --defense-name {palette,wtfpad,front,tamaraw,regulator,"
            "trafficsilver-rr,trafficsilver-bd,trafficsilver-bwr}\n"
            "  --pretrain-source {none,gtt,swallow}\n"
        )
        _shared_parse_args(["--help"])

    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument(
        "--defense-name", type=defense_name, choices=DATASETS, required=True
    )
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

    args.defense_name = selected.defense_name
    args.pretrain_source = selected.pretrain_source
    args.dataset = DATASETS[selected.defense_name]
    if not args.dataset.is_file():
        raise FileNotFoundError(f"missing Defense HDF5: {args.dataset}")
    args.reference_end = 49
    args.min_reference_samples = 1
    args.min_week_samples = args.k + 1
    args.test_per_class = None
    args.website_count = None
    args.split_dir = BASE / "splits" / selected.defense_name
    args.output_dir = (
        BASE / "results" / selected.defense_name / selected.pretrain_source
    )
    if is_pretrained:
        folder = MODEL_DIRECTORIES.get(args.model, args.model)
        args.model_dir = PRETRAIN_ROOTS[selected.pretrain_source] / folder
    else:
        args.model_dir = None
    return args


def main():
    with patch.object(training, "parse_args", parse_args), patch.object(
        training, "prepare_split", prepare_split
    ), patch.object(training, "build_model", build_model):
        training.main()


if __name__ == "__main__":
    main()
