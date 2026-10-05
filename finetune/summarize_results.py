"""Summarize the default run_all.sh experiment as Markdown and CSV tables."""

import csv
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
CONFIG = "k15_seed0_sites200_minlen80"
MODELS = (
    "awf", "tmwf", "ares", "df", "tiktok", "varcnn", "rf",
    "countmamba", "netclr", "swallow-single", "traverse", "scsm-single",
)
WEEKS = range(1, 7)


def collect():
    rows = []
    tasks = 0
    for model in MODELS:
        variants = ("_randomslot_alltest", "_alltest") if model == "scsm-single" else (
            "_freeze" if model in ("traverse", "swallow-single") else "",
        )
        for variant in variants:
            strategies = ("fixed", "adaptive", "ensemble") if model == "scsm-single" else (None,)
            values = {strategy: [] for strategy in strategies}
            for week in WEEKS:
                directory = BASE / "results" / model / (CONFIG + variant) / f"week{week:02d}"
                if not (directory / "final.pt").is_file():
                    raise FileNotFoundError(directory / "final.pt")
                for strategy in strategies:
                    suffix = f"_{strategy}" if strategy else ""
                    data = json.loads((directory / f"test_metrics{suffix}.json").read_text())
                    if strategy and not (directory / f"predictions{suffix}.npz").is_file():
                        raise FileNotFoundError(directory / f"predictions{suffix}.npz")
                    metrics = data["overall"]
                    pair = (float(metrics["accuracy_at_3"]), float(metrics["macro_f1"]))
                    if not all(0 <= value <= 1 for value in pair):
                        raise ValueError(f"Invalid metrics: {directory}")
                    values[strategy].append(pair)
                tasks += 1
            for strategy in strategies:
                label = model
                if model == "scsm-single":
                    slot = "random-slot" if variant == "_randomslot_alltest" else "no-random-slot"
                    label += f" ({slot}, {strategy})"
                elif variant == "_freeze":
                    label += " (freeze)"
                rows.append((label, values[strategy]))
    return rows, tasks


def main():
    rows, tasks = collect()
    destination = BASE / "results" / "summary"
    destination.mkdir(exist_ok=True)
    lines = [
        f"# {CONFIG}", "",
        f"已完成 {tasks}/78 个训练任务。单元格为 Accuracy@3 / macro-F1，单位为 %，保留两位小数。",
        "SCSM 的 random-slot 表示训练槽位配置；fixed/adaptive/ensemble 表示测试策略。",
        "",
        "| 方法 | " + " | ".join(f"Week {week}" for week in WEEKS) + " |",
        "| " + " | ".join(["---"] + ["---:"] * 6) + " |",
    ]
    for label, pairs in rows:
        lines.append("| " + label + " | " + " | ".join(
            f"{accuracy * 100:.2f} / {f1 * 100:.2f}" for accuracy, f1 in pairs
        ) + " |")
    markdown = destination / f"{CONFIG}.md"
    markdown.write_text("\n".join(lines) + "\n")
    with (destination / f"{CONFIG}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Method"] + [
            f"Week {week} {metric}" for week in WEEKS for metric in ("Accuracy@3 (%)", "macro-F1 (%)")
        ])
        for label, pairs in rows:
            writer.writerow([label] + [f"{value * 100:.2f}" for pair in pairs for value in pair])
    print(markdown.read_text())


if __name__ == "__main__":
    main()
