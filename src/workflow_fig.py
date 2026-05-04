\
"""Generate the revised Figure 12 workflow diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .utils import ensure_dir


BOXES = [
    ("Raw merged UCI Heart Disease dataset", "(Cleveland + Hungary + Switzerland + VA Long Beach)"),
    ("Unified preprocessing", '("?"→NaN, numeric casting, target binarization,\nmedian imputation; scaling only when required)'),
    ("Primary external evaluation protocol", "Stratified outer Train/Test split (80/20, seed=42)\nInner Train/Val for selection / early stopping"),
    ("Primary benchmark training", "ML: Light vs Full\nDL: Light vs Full\nAutoML: Light vs Full"),
    ("Evaluate on held-out test set", "Accuracy, ROC-AUC, F1, Precision, Recall, LogLoss,\nConfusion Matrix"),
    ("Resource logging", "Observed runtime (wall-clock), RAM / memory usage,\nModel size (serialized file MB)"),
    ("Repeated lightweight holdout analysis", "5 stratified 80/20 splits\nSeeds: [42, 7, 21, 84, 123]\nReport mean ± SD"),
    ("Additional robustness analyses", "Budget compliance\nWilcoxon-Holm + Friedman tests\nTarget sensitivity: binary / ordinal 3-class / 5-class"),
    ("Report results", "Tables / Figures + exact configurations +\nreproducibility materials"),
]


def make_workflow_figure(output_path: str | Path, dpi: int = 300) -> None:
    """Create a vertical flowchart matching the revised manuscript workflow."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n = len(BOXES)
    box_h = 0.082
    gap = 0.020
    top = 0.965
    x = 0.08
    w = 0.84

    for i, (title, subtitle) in enumerate(BOXES):
        y_top = top - i * (box_h + gap)
        y = y_top - box_h

        rect = FancyBboxPatch(
            (x, y),
            w,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=1.6,
            edgecolor="black",
            facecolor="white",
        )
        ax.add_patch(rect)

        ax.text(
            0.5,
            y + box_h * 0.64,
            title,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        ax.text(
            0.5,
            y + box_h * 0.34,
            subtitle,
            ha="center",
            va="center",
            fontsize=8.8,
        )

        if i < n - 1:
            ax.annotate(
                "",
                xy=(0.5, y - gap * 0.55),
                xytext=(0.5, y - gap * 0.05),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color="black"),
            )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
