\
"""Raw data diagnostic figures for the merged heart disease dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import FEATURE_COLUMNS, TARGET_COLUMN
from .preprocess import load_raw_heart_csv
from .utils import ensure_dir


def save_fig(path: str | Path, dpi: int = 300) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def make_raw_figures(raw_csv: str | Path, output_dir: str | Path) -> None:
    """Generate simple raw-data diagnostic figures."""
    out = ensure_dir(output_dir)
    df = load_raw_heart_csv(raw_csv)

    # Missingness density
    plt.figure(figsize=(9, 5))
    df[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().mean().sort_values().plot(kind="bar")
    plt.ylabel("Missing fraction")
    plt.title("Raw missingness by variable")
    save_fig(out / "raw_missingness_by_variable.png")

    # Original target distribution
    plt.figure(figsize=(7, 5))
    df[TARGET_COLUMN].value_counts(dropna=False).sort_index().plot(kind="bar")
    plt.xlabel("Original target")
    plt.ylabel("Count")
    plt.title("Original target distribution")
    save_fig(out / "raw_target_distribution.png")

    # Feature completeness heatmap-like image
    plt.figure(figsize=(9, 5))
    plt.imshow(df[FEATURE_COLUMNS].notna().astype(int).values, aspect="auto")
    plt.xlabel("Feature index")
    plt.ylabel("Row index")
    plt.title("Raw feature completeness pattern")
    save_fig(out / "raw_completeness_pattern.png")

    # Raw correlation matrix
    corr = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").corr()
    plt.figure(figsize=(8, 7))
    plt.imshow(corr, aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar(label="Pearson correlation")
    plt.title("Raw feature correlation matrix")
    save_fig(out / "raw_correlation_matrix.png")
