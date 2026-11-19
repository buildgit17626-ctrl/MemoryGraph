"""Plot measured retrieval scores from the generated synthetic result JSON."""

from __future__ import annotations

import json
import os

import numpy as np

from experiments.config import RESULTS_DIR, ROOT

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "mplconfig"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    metrics = json.loads((RESULTS_DIR / "semantic_metrics.json").read_text(encoding="utf-8"))
    methods = ["TF-IDF", "Pretrained embeddings"]
    keys = ["tfidf", "semantic"]
    mrr = [metrics[key]["mrr"] for key in keys]
    precision = [metrics[key]["precision_at_k"] for key in keys]
    x = np.arange(2)
    width = 0.28
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    fig.patch.set_facecolor("#f7f8f6")
    ax.set_facecolor("#f7f8f6")
    bars_mrr = ax.bar(x - width / 2, mrr, width, label="MRR", color="#0f4c47")
    bars_precision = ax.bar(x + width / 2, precision, width, label="Precision@3", color="#7aa9a1")
    for bars in (bars_mrr, bars_precision):
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
    ax.set_xticks(x, methods)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Synthetic memory retrieval comparison", loc="left", fontsize=14, pad=16)
    ax.text(
        0,
        -0.20,
        "Fictional templated users; these scores do not estimate real-user performance.",
        transform=ax.transAxes,
        fontsize=9,
        color="#475569",
    )
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=0.28, left=0.10, right=0.97, top=0.84)
    fig.savefig(RESULTS_DIR / "synthetic_retrieval.png", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
