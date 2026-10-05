#!/usr/bin/env python3
"""Evaluate GTT-pretrained baselines and SCSM ablations across weeks."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_similarity import evaluate as weekly_evaluate


GTT_MODEL_DIRS = {
    "GTT_scsm_single": PROJECT_ROOT / "output/pretrain/GTT_dataset/scsm_single",
    "GTT_scsm_no_segmentation": (
        PROJECT_ROOT / "output/pretrain/GTT_dataset/ablation/no_segmentation"
    ),
    "GTT_scsm_no_combination": (
        PROJECT_ROOT / "output/pretrain/GTT_dataset/ablation/no_combination"
    ),
    "GTT_scsm_no_scaling": (
        PROJECT_ROOT / "output/pretrain/GTT_dataset/ablation/no_scaling"
    ),
    "GTT_scsm_no_masking": (
        PROJECT_ROOT / "output/pretrain/GTT_dataset/ablation/no_masking"
    ),
    "GTT_scsm_mask_only": (
        PROJECT_ROOT / "output/pretrain/GTT_dataset/ablation/mask_only"
    ),
    "GTT_scsm_no_augmentation": (
        PROJECT_ROOT / "output/pretrain/GTT_dataset/ablation/no_augmentation"
    ),
    "GTT_swallow_origin": (
        PROJECT_ROOT / "output/pretrain/GTT_dataset/swallow-origin"
    ),
    "GTT_netclr": PROJECT_ROOT / "output/pretrain/GTT_dataset/netclr",
    "GTT_traverse": PROJECT_ROOT / "output/pretrain/GTT_dataset/traverse",
}


def _add_default_argument(name: str, value: str) -> None:
    """Append a default CLI option while still allowing an explicit override."""
    option = f"--{name}"
    if option not in sys.argv and not any(
        argument.startswith(f"{option}=") for argument in sys.argv[1:]
    ):
        sys.argv.extend((option, value))


def main() -> None:
    # Keep this entry point GTT-only and make its sampling protocol explicit.
    weekly_evaluate.MODEL_DIRS = GTT_MODEL_DIRS
    _add_default_argument("min-trace-length", "80")
    _add_default_argument("website-count", "200")
    _add_default_argument(
        "output-dir", "feature_similarity/results/ablation_GTT_weekly"
    )
    weekly_evaluate.main()


if __name__ == "__main__":
    main()
