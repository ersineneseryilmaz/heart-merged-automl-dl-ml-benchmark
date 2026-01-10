# src/figures_clean.py
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

from .utils import ensure_dir


def _save_900(fig, out_path: str, dpi: int = 150):
    fig.set_size_inches(6, 6, forward=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_figures_900(clean_csv_path: str, target_col: str, out_dir: str):
    ensure_dir(out_dir)
    df = pd.read_csv(clean_csv_path)
    df[target_col] = df[target_col].astype(int)
    feature_cols = [c for c in df.columns if c != target_col]

    # Fig1
    fig, ax = plt.subplots()
    counts = df[target_col].value_counts().sort_index()
    ax.bar([str(i) for i in counts.index], counts.values)
    ax.set_title("Target Class Distribution")
    ax.set_xlabel("Class (target)")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    _save_900(fig, os.path.join(out_dir, "Fig1_target_distribution_900.png"))

    # Fig2
    fig, ax = plt.subplots()
    nan_counts = df[feature_cols].isna().sum().sort_values(ascending=False)
    ax.bar(nan_counts.index.astype(str), nan_counts.values)
    ax.set_title("Missing Values per Feature")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Missing count")
    ax.tick_params(axis="x", rotation=90)
    _save_900(fig, os.path.join(out_dir, "Fig2_missing_values_900.png"))

    # Fig3
    fig, ax = plt.subplots()
    data0 = df[df[target_col] == 0][feature_cols]
    data1 = df[df[target_col] == 1][feature_cols]
    plot_data, positions = [], []
    pos = 1
    for f in feature_cols:
        plot_data.append(data0[f].values)
        plot_data.append(data1[f].values)
        positions.extend([pos, pos + 0.35])
        pos += 1

    ax.boxplot(plot_data, positions=positions, widths=0.25, showfliers=False)
    ax.set_title("Feature Distributions by Target (Boxplot)")
    ax.set_xlabel("Feature (pairs: 0 then 1)")
    ax.set_ylabel("Value")
    ax.set_xticks([i + 0.175 for i in range(1, len(feature_cols) + 1)])
    ax.set_xticklabels([str(f) for f in feature_cols], rotation=90)
    ax.text(0.02, 0.98, "Each feature has two boxes: target=0 (left), target=1 (right)",
            transform=ax.transAxes, ha="left", va="top")
    _save_900(fig, os.path.join(out_dir, "Fig3_boxplot_by_target_900.png"))

    # Fig4
    corr = df[feature_cols + [target_col]].corr(numeric_only=True)
    fig, ax = plt.subplots()
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_title("Correlation Matrix (Pearson)")
    ax.set_xticks(range(corr.shape[1]))
    ax.set_yticks(range(corr.shape[0]))
    ax.set_xticklabels(corr.columns.astype(str), rotation=90)
    ax.set_yticklabels(corr.index.astype(str))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_900(fig, os.path.join(out_dir, "Fig4_correlation_matrix_900.png"))

    # Fig5 (age vs thalach)
    if "age" in df.columns and "thalach" in df.columns:
        fig, ax = plt.subplots()
        df0 = df[df[target_col] == 0]
        df1 = df[df[target_col] == 1]
        ax.scatter(df0["age"], df0["thalach"], label="target=0", alpha=0.8)
        ax.scatter(df1["age"], df1["thalach"], label="target=1", alpha=0.8)
        ax.set_title("age vs thalach by Target")
        ax.set_xlabel("age")
        ax.set_ylabel("thalach")
        ax.legend()
        _save_900(fig, os.path.join(out_dir, "Fig5_scatter_age_thalach_900.png"))

    print(f"[OK] Clean figures saved in: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--target", default="target")
    ap.add_argument("--out", default="figures_900")
    args = ap.parse_args()
    make_figures_900(args.input, args.target, args.out)


if __name__ == "__main__":
    main()
