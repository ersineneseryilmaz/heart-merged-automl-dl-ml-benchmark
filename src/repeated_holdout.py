\
"""Repeated stratified holdout utilities for lightweight framework comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from .config import REPEATED_HOLDOUT_SEEDS
from .utils import ensure_dir, mean_sd


Runner = Callable[[pd.DataFrame, str | Path, str, int], pd.DataFrame]


def run_repeated_lightweight_holdout(
    df: pd.DataFrame,
    output_dir: str | Path,
    runners: dict[str, Runner],
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Run repeated lightweight benchmark for selected framework runners.

    Each runner must accept (df, output_dir, regime, seed) and return a one-row
    or multi-row DataFrame. For ClassicML, pass a wrapper that selects the best
    lightweight model per seed if framework-level summaries are desired.
    """
    out = ensure_dir(output_dir)
    seeds = seeds or REPEATED_HOLDOUT_SEEDS
    all_rows = []

    for seed in seeds:
        for name, runner in runners.items():
            try:
                res = runner(df, out / name, "Light", seed)
                res["analysis_seed"] = seed
                res["analysis_framework"] = name
                all_rows.append(res)
            except Exception as exc:
                all_rows.append(
                    pd.DataFrame(
                        [
                            {
                                "analysis_seed": seed,
                                "analysis_framework": name,
                                "framework": name,
                                "seed": seed,
                                "status": "error",
                                "error": repr(exc),
                            }
                        ]
                    )
                )

    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(out / "repeated_holdout_light_results.csv", index=False)
    return results


def select_best_per_framework_seed(
    results: pd.DataFrame,
    score_col: str = "f1_pct",
) -> pd.DataFrame:
    """Select the best model row within each framework/seed pair."""
    ok = results[results["status"].fillna("ok").eq("ok")].copy()
    idx = ok.groupby(["analysis_framework", "analysis_seed"])[score_col].idxmax()
    return ok.loc[idx].reset_index(drop=True)


def summarize_repeated_holdout(
    selected: pd.DataFrame,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Create mean ± SD summary for repeated lightweight selected runs."""
    rows = []
    metric_map = {
        "accuracy_pct": "Accuracy (%)",
        "auc_pct": "AUC (%)",
        "f1_pct": "F1 (%)",
        "precision_pct": "Precision (%)",
        "recall_pct": "Recall (%)",
        "logloss": "LogLoss",
        "runtime_s": "Runtime (s)",
        "model_size_mb": "Model size (MB)",
    }

    for framework, g in selected.groupby("analysis_framework"):
        row = {
            "Framework": framework,
            "Regime": "Lightweight",
            "n": int(len(g)),
            "Most frequent best model": g["model"].mode().iloc[0] if "model" in g and not g["model"].mode().empty else "",
        }
        for col, label in metric_map.items():
            row[label] = mean_sd(g[col]) if col in g else ""
        rows.append(row)

    summary = pd.DataFrame(rows)
    if output_csv is not None:
        summary.to_csv(output_csv, index=False, encoding="utf-8")
    return summary
