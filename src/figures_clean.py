\
"""Clean-data figures for the benchmark manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import FEATURE_COLUMNS
from .utils import ensure_dir


def save_fig(path: str | Path, dpi: int = 300) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def make_clean_figures(clean_csv: str | Path, output_dir: str | Path) -> None:
    """Generate clean-data diagnostic figures."""
    out = ensure_dir(output_dir)
    df = pd.read_csv(clean_csv)

    plt.figure(figsize=(6, 5))
    df["target"].value_counts().sort_index().plot(kind="bar")
    plt.xlabel("Binary target")
    plt.ylabel("Count")
    plt.title("Class balance after preprocessing")
    save_fig(out / "clean_class_balance.png")

    # Boxplots by target
    n = len(FEATURE_COLUMNS)
    for col in FEATURE_COLUMNS:
        plt.figure(figsize=(6, 5))
        df.boxplot(column=col, by="target")
        plt.suptitle("")
        plt.title(f"{col} by target")
        plt.xlabel("Target")
        plt.ylabel(col)
        save_fig(out / f"boxplot_{col}_by_target.png")

    corr = df[FEATURE_COLUMNS + ["target"]].corr()
    plt.figure(figsize=(8, 7))
    plt.imshow(corr, aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar(label="Pearson correlation")
    plt.title("Clean-data correlation matrix")
    save_fig(out / "clean_correlation_matrix.png")

    if {"age", "thalach", "target"}.issubset(df.columns):
        plt.figure(figsize=(6, 5))
        for target, group in df.groupby("target"):
            plt.scatter(group["age"], group["thalach"], label=f"target={target}", alpha=0.75)
        plt.xlabel("Age")
        plt.ylabel("Max heart rate achieved (thalach)")
        plt.legend()
        plt.title("Age vs. thalach by target")
        save_fig(out / "scatter_age_thalach_by_target.png")
