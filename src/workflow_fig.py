# src/workflow_fig.py
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from .utils import ensure_dir


def _save_900(fig, out_path: str, dpi: int = 150):
    fig.set_size_inches(6, 6, forward=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _box(ax, xy, w, h, text, fontsize=9):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        facecolor="white"
    )
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def _arrow(ax, p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="->", mutation_scale=12, linewidth=1.2))


def make_fig3_workflow(out_dir="figures_900"):
    ensure_dir(out_dir)

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    W, H = 0.82, 0.10
    X = 0.09

    b1 = _box(ax, (X, 0.86), W, H, "Raw merged CSV\n(Cleveland + Hungary + Switzerland + VA Long Beach)")
    b2 = _box(ax, (X, 0.72), W, H, "Unified preprocessing\n(“?”→NaN, numeric casting, target binarization,\nmedian imputation for features)")
    b3 = _box(ax, (X, 0.58), W, H, "Stratified split\nOuter: Train/Test\nInner: Train/Val (selection / early stopping)")
    b4 = _box(ax, (X, 0.44), W, H, "Train (two comparable regimes)\nML: Light vs Full\nDL: Light vs Full\nAutoML: Light vs Full")
    b5 = _box(ax, (X, 0.30), W, H, "Evaluate on held-out test set\nAccuracy, AUC, F1, Precision, Recall, LogLoss,\nConfusion Matrix")
    b6 = _box(ax, (X, 0.16), W, H, "Resource logging\nRuntime (wall-clock), RAM (process RSS & system delta),\nModel size (serialized file MB)")
    b7 = _box(ax, (X, 0.02), W, H, "Report results\nTables/Figures + exact configs for reproducibility")

    def mid_bottom(box):
        x, y = box.get_x(), box.get_y()
        w, h = box.get_width(), box.get_height()
        return (x + w/2, y)

    def mid_top(box):
        x, y = box.get_x(), box.get_y()
        w, h = box.get_width(), box.get_height()
        return (x + w/2, y + h)

    for a, b in [(b1, b2), (b2, b3), (b3, b4), (b4, b5), (b5, b6), (b6, b7)]:
        _arrow(ax, mid_bottom(a), mid_top(b))

    ax.set_title("Figure 3. End-to-end experimental workflow", fontsize=11)
    out_path = os.path.join(out_dir, "Fig3_workflow_900.png")
    _save_900(fig, out_path)
    print("[OK] Saved:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures_900")
    args = ap.parse_args()
    make_fig3_workflow(args.out)


if __name__ == "__main__":
    main()
