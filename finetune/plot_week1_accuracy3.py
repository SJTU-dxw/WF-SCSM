"""Plot Week 1 models' Accuracy@3 across all test weeks."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from summarize_week1_all import collect
from summarize_results import BASE, CONFIG, WEEKS


def main():
    root = BASE / "results_week1_alltest"
    rows, _, _ = collect(root, CONFIG)
    names = {
        "awf": "AWF", "tmwf": "TMWF", "ares": "ARES", "df": "DF",
        "tiktok": "TikTok", "varcnn": "VarCNN", "rf": "RF",
        "countmamba": "CountMamba", "netclr": "NetCLR",
        "swallow-single (freeze)": "Swallow (frozen)", "traverse (freeze)": "TraVerse (frozen)",
        "scsm-single (random-slot, ensemble)": "SCSM (random-slot, ensemble)",
    }
    fig, ax = plt.subplots(figsize=(11, 6.5), layout="constrained")
    colors = list(plt.get_cmap("tab20").colors)
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "p"]
    index = 0
    for label, pairs in rows:
        if label not in names:
            continue
        scsm = label.startswith("scsm-single")
        ax.plot(
            list(WEEKS), [accuracy * 100 for accuracy, _ in pairs],
            label=names[label], color="#c81d25" if scsm else colors[index],
            marker="*" if scsm else markers[index],
            markersize=11 if scsm else 5, linewidth=3 if scsm else 1.6,
            linestyle="-" if scsm or index % 2 == 0 else "--",
            alpha=1 if scsm else 0.85, zorder=5 if scsm else 2,
        )
        if not scsm:
            index += 1
    ax.set(title="Week 1 fine-tuned models: cross-week Accuracy@3",
           xlabel="Test week", ylabel="Accuracy@3 (%)", ylim=(0, 65),
           xticks=list(WEEKS), xticklabels=[f"Week {week}" for week in WEEKS])
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9)
    destination = root / "summary"
    destination.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        path = destination / f"{CONFIG}_accuracy3.{extension}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
