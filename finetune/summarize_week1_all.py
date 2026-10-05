"""Summarize saved Week 1 models' cross-week evaluations (no training)."""

import argparse
import csv
import json
from pathlib import Path

from summarize_results import BASE, CONFIG, MODELS, WEEKS


def collect(root, config):
    rows = []
    checkpoints = 0
    evaluations = 0
    for method in MODELS:
        variants = ("_randomslot_alltest", "_alltest") if method == "scsm-single" else (
            "_freeze" if method in ("traverse", "swallow-single") else "",
        )
        for variant in variants:
            directory = root / method / (config + variant)
            metadata = json.loads((directory / "config.json").read_text())
            summary = json.loads((directory / "test_metrics.json").read_text())
            if metadata["train_week"] != 1 or summary["train_week"] != 1:
                raise ValueError(f"Not trained on Week 1: {directory}")
            if metadata["test_weeks"] != list(WEEKS):
                raise ValueError(f"Expected Week 1–6 evaluations: {directory}")
            strategies = ("fixed", "adaptive", "ensemble") if method == "scsm-single" else ("fixed",)
            for strategy in strategies:
                pairs = []
                suffix = f"_{strategy}" if method == "scsm-single" else ""
                for week in WEEKS:
                    weekly = directory / f"week{week:02d}"
                    data = json.loads((weekly / f"test_metrics{suffix}.json").read_text())
                    if (data["train_week"], data["test_week"], data["strategy"]) != (1, week, strategy):
                        raise ValueError(f"Evaluation metadata mismatch: {weekly}")
                    if not (weekly / f"predictions{suffix}.npz").is_file():
                        raise FileNotFoundError(weekly / f"predictions{suffix}.npz")
                    metrics = data["overall"]
                    if metrics != summary["by_strategy"][strategy][str(week)]:
                        raise ValueError(f"Weekly/summary metrics mismatch: {weekly}")
                    pair = (float(metrics["accuracy_at_3"]), float(metrics["macro_f1"]))
                    if not all(0 <= value <= 1 for value in pair):
                        raise ValueError(f"Invalid metrics: {weekly}")
                    pairs.append(pair)
                    evaluations += 1
                label = method
                if method == "scsm-single":
                    slot = "random-slot" if variant == "_randomslot_alltest" else "no-random-slot"
                    label += f" ({slot}, {strategy})"
                elif variant == "_freeze":
                    label += " (freeze)"
                rows.append((label, pairs))
            checkpoints += 1
    return rows, checkpoints, evaluations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--results-root", type=Path, default=BASE / "results_week1_alltest")
    args = parser.parse_args()
    rows, checkpoints, evaluations = collect(args.results_root, args.config)
    destination = args.results_root / "summary"
    destination.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Week 1 微调模型跨周测试：{args.config}", "",
        "所有列均使用同一个 Week 1 微调模型，分别测试对应周的固定测试集，不进行逐周微调。",
        f"已验证 {checkpoints}/13 个模型配置、{evaluations}/102 次分周评测。",
        "单元格为 Accuracy@3 / macro-F1，单位为 %，保留两位小数。",
        "SCSM 的 random-slot 表示训练槽位配置；fixed/adaptive/ensemble 表示测试策略。", "",
        "| 方法 | " + " | ".join(f"Week {week}" for week in WEEKS) + " |",
        "| " + " | ".join(["---"] + ["---:"] * 6) + " |",
    ]
    for label, pairs in rows:
        lines.append("| " + label + " | " + " | ".join(
            f"{accuracy * 100:.2f} / {f1 * 100:.2f}" for accuracy, f1 in pairs
        ) + " |")
    markdown = destination / f"{args.config}.md"
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (destination / f"{args.config}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Method"] + [
            f"Week {week} {metric}" for week in WEEKS for metric in ("Accuracy@3 (%)", "macro-F1 (%)")
        ])
        for label, pairs in rows:
            writer.writerow([label] + [f"{value * 100:.2f}" for pair in pairs for value in pair])
    print(markdown.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
