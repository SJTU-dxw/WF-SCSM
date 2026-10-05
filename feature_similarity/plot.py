#!/usr/bin/env python3
"""Plot weekly centroid metrics in the style of the TraVerse paper.

Usage: python feature_similarity/plot.py
Requires matplotlib. Reads the existing CSV without rerunning evaluation.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


ROOT = Path(__file__).resolve().parent
WEEK_OFFSET = 7  # Evaluation weeks 1–6 correspond to calendar weeks 8–13.
STYLES = {
    "scsm_single": ("SCSM", "#7040A0", "o", "-"),
    "scsm_single-fixed": ("SCSM-fixed", "#8E77A8", "o", "--"),
    "scsm_single-adaptive": ("SCSM-adaptive", "#A45C9D", "s", "-."),
    "scsm_single-ensemble": ("SCSM-ensemble", "#7040A0", "o", "-"),
    "traverse": ("TraVerse", "#E47743", "h", "-"),
    "swallow-origin": ("Swallow-origin", "#8B5A2B", "^", ":"),
    "swallow-single": ("Swallow", "#B13D62", "D", "--"),
    "netclr": ("NetCLR", "#4886B5", "P", "--"),
}
METRICS = (
    ("与参考时期质心余弦相似度", "Mean cosine similarity",
     "(a) Cosine similarity"),
    ("质心分类准确率 (macro)", "Macro accuracy",
     "(b) Classification accuracy"),
)


def default_input_paths(strategy: str) -> list[Path]:
    """Find per-model result files now that evaluation no longer combines them."""
    paths = []
    if not ROOT.joinpath("results").is_dir():
        return paths
    for model_dir in sorted(ROOT.joinpath("results").iterdir()):
        if not model_dir.is_dir():
            continue
        regular = model_dir / "weekly_metrics.csv"
        if regular.is_file():
            paths.append(regular)
        selected_strategies = {
            "all": ("fixed", "adaptive", "ensemble"),
            "both": ("fixed", "ensemble"),
        }.get(strategy, (strategy,))
        paths.extend(
            path for selected in selected_strategies
            if (path := model_dir / f"weekly_metrics_{selected}.csv").is_file()
        )
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", default=None,
                        help="One or more result CSV files; defaults to per-model outputs")
    parser.add_argument(
        "--scsm-slot-strategy",
        choices=("fixed", "adaptive", "ensemble", "both", "all"),
        default="all",
    )
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path stem; saves both PNG and PDF")
    args = parser.parse_args()
    if args.input is None:
        args.input = default_input_paths(args.scsm_slot_strategy)
    if args.output is None:
        args.output = ROOT / f"weekly_centroid_metrics_scsm-{args.scsm_slot_strategy}"
    if not args.input:
        raise FileNotFoundError(f"No per-model metric CSV files found under {ROOT / 'results'}")

    grouped = defaultdict(list)
    for input_path in args.input:
        with input_path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if (row["model"].endswith(("swallow-multi", "swallow_multi"))
                        or "scsm_multi" in row["model"]):
                    continue
                grouped[row["model"]].append(row)
    if not grouped:
        raise ValueError(f"No metric rows found in {args.input}")
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["week"]))
        weeks = [int(row["week"]) for row in rows]
        if len(set(weeks)) != len(weeks):
            raise ValueError("Duplicate weeks found for a model")

    plt.rcParams.update({
        "font.family": "STIXGeneral", "font.size": 16,
        "mathtext.fontset": "stix",
        "axes.labelsize": 17, "axes.linewidth": 0.7,
        "axes.edgecolor": "#444444", "text.color": "#222222",
        "axes.labelcolor": "#222222",
        "xtick.color": "#444444", "ytick.color": "#444444",
        "xtick.labelsize": 15, "ytick.labelsize": 15,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1))
    models = [name for name in STYLES if name in grouped]
    models.extend(sorted(set(grouped) - set(STYLES)))
    weeks = sorted({int(row["week"]) + WEEK_OFFSET
                    for rows in grouped.values() for row in rows})
    for ax, (column, ylabel, caption) in zip(axes, METRICS):
        for index, model in enumerate(models):
            label, color, marker, linestyle = STYLES.get(
                model, (model, f"C{index % 10}", "o", "--"))
            rows = grouped[model]
            ax.plot([int(row["week"]) + WEEK_OFFSET for row in rows],
                    [float(row[column]) for row in rows],
                    label=label, color=color, marker=marker,
                    linestyle=linestyle, linewidth=1.65, markersize=4.8,
                    markeredgecolor="white", markeredgewidth=0.45,
                    solid_capstyle="round")
        ax.set_xlabel("Week", labelpad=3)
        if column == "质心分类准确率 (macro)":
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_ylabel(ylabel, labelpad=6)
        ax.set_xticks(weeks)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.55, color="#D8D8D8")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.margins(x=0.04, y=0.12)
        ax.text(0.5, -0.32, caption, transform=ax.transAxes,
                ha="center", va="top", fontsize=17)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.995),
               ncol=len(models), frameon=False, fontsize=16,
               handlelength=3.6, handletextpad=0.5, columnspacing=0.9,
               numpoints=1, markerscale=0.85)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.30, wspace=0.34)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = args.output.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.12)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
