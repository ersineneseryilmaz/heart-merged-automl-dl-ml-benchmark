# src/figures_raw.py
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import ensure_dir

EXPECTED = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]


def _save_900(fig, out_path: str, dpi: int = 150):
    fig.set_size_inches(6, 6, forward=True)  # 6*150=900px
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_raw_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        na_values=["?", "??", "???", "????", " ?"],
        skipinitialspace=True,
        engine="python",
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def make_figures_raw_900(raw_csv_path: str, target_col: str, out_dir: str):
    ensure_dir(out_dir)
    df_raw = load_raw_csv(raw_csv_path)

    cols = [c for c in EXPECTED if c in df_raw.columns] or list(df_raw.columns)

    df_num = df_raw.copy()
    for c in cols:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")

    feature_cols = [c for c in cols if c != target_col and c in df_num.columns]

    # FigR1
    fig, ax = plt.subplots()
    nan_counts = df_num[cols].isna().sum().sort_values(ascending=False)
    ax.bar(nan_counts.index.astype(str), nan_counts.values)
    ax.set_title("Raw Data: Missing Values per Column")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing count")
    ax.tick_params(axis="x", rotation=90)
    _save_900(fig, os.path.join(out_dir, "FigR1_raw_missing_values_900.png"))

    # FigR2
    fig, ax = plt.subplots()
    non_null = df_raw[cols].notna().sum().sort_values(ascending=False)
    ax.bar(non_null.index.astype(str), non_null.values)
    ax.set_title("Raw Data: Non-null Counts per Column")
    ax.set_xlabel("Column")
    ax.set_ylabel("Non-null count")
    ax.tick_params(axis="x", rotation=90)
    _save_900(fig, os.path.join(out_dir, "FigR2_raw_nonnull_counts_900.png"))

    # FigR3
    if target_col in df_num.columns:
        fig, ax = plt.subplots()
        t = df_num[target_col]
        counts = t.value_counts(dropna=False).sort_index()
        labels, values = [], []
        for idx, v in counts.items():
            if pd.isna(idx):
                labels.append("NaN")
            else:
                labels.append(str(int(idx)) if float(idx).is_integer() else str(idx))
            values.append(int(v))

        ax.bar(labels, values)
        ax.set_title("Raw Data: Target Distribution (Original)")
        ax.set_xlabel("Target value")
        ax.set_ylabel("Count")
        for i, v in enumerate(values):
            ax.text(i, v, str(v), ha="center", va="bottom")
        _save_900(fig, os.path.join(out_dir, "FigR3_raw_target_distribution_900.png"))

    # FigR4
    max_rows = 200
    show_cols = feature_cols[:13]
    if show_cols:
        mat = df_num[show_cols].notna().astype(int).head(max_rows).values
        fig, ax = plt.subplots()
        im = ax.imshow(mat, aspect="auto")
        ax.set_title(f"Raw Data: Feature Completeness (first {min(len(df_num), max_rows)} rows)")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Row index")
        ax.set_xticks(range(len(show_cols)))
        ax.set_xticklabels([str(c) for c in show_cols], rotation=90)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _save_900(fig, os.path.join(out_dir, "FigR4_raw_completeness_heatmap_900.png"))

    # FigR5
    if feature_cols:
        fcols = feature_cols[:13]
        plot_data = [df_num[c].dropna().values for c in fcols]
        fig, ax = plt.subplots()
        ax.boxplot(plot_data, showfliers=False)
        ax.set_title("Raw Data: Feature Distributions (Boxplot)")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Value")
        ax.set_xticks(range(1, len(fcols) + 1))
        ax.set_xticklabels([str(c) for c in fcols], rotation=90)
        _save_900(fig, os.path.join(out_dir, "FigR5_raw_boxplot_900.png"))

    # FigR6
    if feature_cols and target_col in df_num.columns:
        corr = df_num[feature_cols + [target_col]].corr(numeric_only=True)
        fig, ax = plt.subplots()
        im = ax.imshow(corr.values, aspect="auto")
        ax.set_title("Raw Data: Correlation Matrix (Numeric Only)")
        ax.set_xticks(range(corr.shape[1]))
        ax.set_yticks(range(corr.shape[0]))
        ax.set_xticklabels(corr.columns.astype(str), rotation=90)
        ax.set_yticklabels(corr.index.astype(str))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _save_900(fig, os.path.join(out_dir, "FigR6_raw_correlation_matrix_900.png"))

    print(f"[OK] Raw figures saved in: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--target", default="target")
    ap.add_argument("--out", default="figures_raw_900")
    args = ap.parse_args()
    make_figures_raw_900(args.input, args.target, args.out)

if __name__ == "__main__":
    main()
