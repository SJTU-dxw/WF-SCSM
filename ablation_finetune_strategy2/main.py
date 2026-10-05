"""GTT SCSM fine-tuning-strategy ablation without editing the shared trainer."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finetune import main as training
from finetune import models
from model.scsm import SCSM


def parse_ablation_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--initialization",
        choices=("pretrained", "scratch"),
        default="scratch",
        help="initialize SCSM from the GTT checkpoint or from scratch",
    )
    return parser.parse_known_args(argv)


def build_scratch_model(method, classes, device, model_dir=None, freeze=False):
    if method != "scsm-single":
        raise ValueError("The fine-tuning-strategy ablation only supports scsm-single")
    if freeze:
        raise ValueError("--freeze is invalid for a randomly initialized encoder")

    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise ValueError("scsm-single requires CUDA for the Mamba2 kernels")
    if resolved_device.index is not None:
        torch.cuda.set_device(resolved_device)

    architecture_path = ROOT / "output/pretrain/GTT_dataset/scsm_single/config.json"
    source_config = json.loads(architecture_path.read_text())
    if source_config["method"] != "scsm" or not source_config["model"].get("simple_mode"):
        raise ValueError("Expected the GTT SCSM-single architecture config")

    # Reuse only the architecture and input protocol; do not load checkpoint weights.
    config = {**source_config["model"], **source_config.get("augmentation", {})}
    encoder = SCSM(config)
    model = models.Classifier(encoder, encoder.output_dim, classes, method)
    return model.to(resolved_device), config, None, None


def main():
    ablation, remaining = parse_ablation_args(sys.argv[1:])
    sys.argv[1:] = remaining
    settings = {
        **training.SETTINGS,
        "output_dir": ROOT / "ablation_finetune_strategy2/results" / ablation.initialization,
    }
    builder = models.build_model if ablation.initialization == "pretrained" else build_scratch_model
    with patch.object(training, "SETTINGS", settings), patch.object(training, "build_model", builder):
        training.main()


if __name__ == "__main__":
    main()
