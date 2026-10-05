#!/usr/bin/env python3
"""Evaluate SwallowDataset SCSM parameter-sensitivity models across weeks."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_similarity import evaluate as weekly_evaluate


SENSITIVITY_MODEL_DIRS = {
    "swallow_dataset_scsm_single": (
        PROJECT_ROOT / "output/pretrain/swallow_dataset/scsm_single"
    ),
    "swallow_dataset_scsm_segment_aug_num_1": (
        PROJECT_ROOT
        / "output/pretrain/swallow_dataset/parameter_sensitivity/segment_aug_num_1"
    ),
    "swallow_dataset_scsm_segment_aug_num_3": (
        PROJECT_ROOT
        / "output/pretrain/swallow_dataset/parameter_sensitivity/segment_aug_num_3"
    ),
    "swallow_dataset_scsm_combine_max_2": (
        PROJECT_ROOT
        / "output/pretrain/swallow_dataset/parameter_sensitivity/combine_max_2"
    ),
    "swallow_dataset_scsm_combine_max_3": (
        PROJECT_ROOT
        / "output/pretrain/swallow_dataset/parameter_sensitivity/combine_max_3"
    ),
    "swallow_dataset_scsm_combine_max_4": (
        PROJECT_ROOT
        / "output/pretrain/swallow_dataset/parameter_sensitivity/combine_max_4"
    ),
    "swallow_dataset_scsm_max_overlap_ratio_0_0": (
        PROJECT_ROOT
        / "output/pretrain/swallow_dataset/parameter_sensitivity/max_overlap_ratio_0_0"
    ),
    "swallow_dataset_scsm_max_overlap_ratio_0_1": (
        PROJECT_ROOT
        / "output/pretrain/swallow_dataset/parameter_sensitivity/max_overlap_ratio_0_1"
    ),
    "swallow_dataset_scsm_max_overlap_ratio_0_3": (
        PROJECT_ROOT
        / "output/pretrain/swallow_dataset/parameter_sensitivity/max_overlap_ratio_0_3"
    ),
    "swallow_dataset_scsm_mask_ratio_0_1": (
        PROJECT_ROOT
        / "output/pretrain/swallow_dataset/parameter_sensitivity/mask_ratio_0_1"
    ),
    "swallow_dataset_scsm_mask_ratio_0_9": (
        PROJECT_ROOT
        / "output/pretrain/swallow_dataset/parameter_sensitivity/mask_ratio_0_9"
    ),
}


def _add_default_argument(name: str, value: str) -> None:
    """Append a default CLI option while still allowing an explicit override."""
    option = f"--{name}"
    if option not in sys.argv and not any(
        argument.startswith(f"{option}=") for argument in sys.argv[1:]
    ):
        sys.argv.extend((option, value))


def main() -> None:
    weekly_evaluate.MODEL_DIRS = SENSITIVITY_MODEL_DIRS
    _add_default_argument("min-trace-length", "80")
    _add_default_argument(
        "output-dir", "feature_similarity/results/sensitivity_swallow_weekly"
    )
    weekly_evaluate.main()


if __name__ == "__main__":
    main()
