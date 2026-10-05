#!/usr/bin/env python3
"""Evaluate sensitivity to the number of SCSM ensemble views at inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_similarity import evaluate as weekly_evaluate
from feature_similarity.evaluate_sensitivity import SENSITIVITY_MODEL_DIRS


MODEL_NAME = "swallow_dataset_scsm_single"
DEFAULT_R_VALUES = (2, 3, 5, 7, 9)


class InferenceRTraceDataset(weekly_evaluate.TraceDataset):
    """Inject a selected ensemble-view count into the reusable dataset."""

    inference_r: int | None = None

    def __init__(self, path, records, method, feature_config):
        patched_config = dict(feature_config)
        if method == "scsm" and patched_config.get("slot_strategy") == "ensemble":
            if self.inference_r is None:
                raise RuntimeError("InferenceRTraceDataset.inference_r is not set")
            patched_config["test_slot_count"] = self.inference_r
        super().__init__(path, records, method, patched_config)


def parse_args(argv=None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Unrecognized options are forwarded to feature_similarity/evaluate.py; "
            "--models, --scsm-slot-strategy, and its --output-dir are managed here."
        ),
    )
    parser.add_argument(
        "--r-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_R_VALUES),
        help="Numbers of uniformly spaced ensemble slot widths (default: 2 3 5 7 9)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "feature_similarity/results/sensitivity_inference_r",
    )
    args, forwarded = parser.parse_known_args(argv)
    if any(value < 2 for value in args.r_values):
        parser.error("every R value must be at least 2")
    if len(set(args.r_values)) != len(args.r_values):
        parser.error("R values must be unique")
    reserved = {"--models", "--scsm-slot-strategy"}
    conflicts = sorted(
        option for option in reserved
        if option in forwarded or any(item.startswith(f"{option}=") for item in forwarded)
    )
    if conflicts:
        parser.error(f"options managed by this entry point cannot be forwarded: {conflicts}")
    return args, forwarded


def has_option(arguments: list[str], option: str) -> bool:
    return option in arguments or any(item.startswith(f"{option}=") for item in arguments)


def read_result(path: Path, inference_r: int) -> list[dict[str, str | int]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    return [
        {
            "model": row.pop("model"),
            "inference_strategy": "ensemble",
            "inference_r": inference_r,
            **row,
        }
        for row in rows
    ]


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    if not rows:
        raise ValueError("no R-sensitivity results were produced")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> None:
    args, forwarded = parse_args(argv)
    r_values = sorted(args.r_values)
    output_dir = args.output_dir.resolve()
    model_dir = SENSITIVITY_MODEL_DIRS[MODEL_NAME]
    config = weekly_evaluate.load_config(model_dir)
    minimum_slot = float(config["augmentation"]["min_slot_time"])
    maximum_slot = float(config["augmentation"]["max_slot_time"])

    original_dataset_class = weekly_evaluate.TraceDataset
    original_model_dirs = weekly_evaluate.MODEL_DIRS
    original_argv = sys.argv
    all_rows: list[dict[str, str | int]] = []
    try:
        weekly_evaluate.MODEL_DIRS = {MODEL_NAME: model_dir}
        weekly_evaluate.TraceDataset = InferenceRTraceDataset
        for current_r in r_values:
            InferenceRTraceDataset.inference_r = current_r
            run_output = output_dir / f"r_{current_r:02d}"
            run_arguments = list(forwarded)
            if not has_option(run_arguments, "--min-trace-length"):
                run_arguments.extend(("--min-trace-length", "80"))
            run_arguments.extend((
                "--models", MODEL_NAME,
                "--scsm-slot-strategy", "ensemble",
                "--output-dir", str(run_output),
            ))
            print(f"\n=== inference ensemble R={current_r} ===")
            sys.argv = [original_argv[0], *run_arguments]
            weekly_evaluate.main()
            result_path = run_output / MODEL_NAME / "weekly_metrics_ensemble.csv"
            all_rows.extend(read_result(result_path, current_r))
    finally:
        InferenceRTraceDataset.inference_r = None
        weekly_evaluate.TraceDataset = original_dataset_class
        weekly_evaluate.MODEL_DIRS = original_model_dirs
        sys.argv = original_argv

    write_csv(output_dir / "weekly_metrics.csv", all_rows)
    protocol = {
        "model": MODEL_NAME,
        "checkpoint_directory": str(model_dir),
        "inference_strategy": "ensemble",
        "r_values": r_values,
        "slot_range_seconds": [minimum_slot, maximum_slot],
        "slot_widths_seconds": {
            str(value): np.linspace(minimum_slot, maximum_slot, value).tolist()
            for value in r_values
        },
        "aggregation": "arithmetic mean of the R encoder embeddings",
        "individual_result_directories": {
            str(value): str(output_dir / f"r_{value:02d}") for value in r_values
        },
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(output_dir / "weekly_metrics.csv")


if __name__ == "__main__":
    main()
