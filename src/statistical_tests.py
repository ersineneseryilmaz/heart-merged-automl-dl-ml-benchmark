\
"""Statistical tests for repeated lightweight benchmark results."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests


def pivot_metric(
    selected_runs: pd.DataFrame,
    metric: str,
    method_col: str = "analysis_framework",
    split_col: str = "analysis_seed",
) -> pd.DataFrame:
    """Pivot selected repeated runs as split x method."""
    return selected_runs.pivot_table(index=split_col, columns=method_col, values=metric, aggfunc="mean").dropna(axis=0)


def friedman_tests(
    selected_runs: pd.DataFrame,
    metrics: list[str],
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Run Friedman omnibus tests for each metric."""
    rows = []
    for metric in metrics:
        pivot = pivot_metric(selected_runs, metric)
        methods = list(pivot.columns)
        if pivot.shape[0] < 2 or pivot.shape[1] < 3:
            rows.append(
                {
                    "analysis_set": "framework_level_selected_runs",
                    "metric": metric,
                    "n_paired_splits": int(pivot.shape[0]),
                    "n_methods": int(pivot.shape[1]),
                    "methods": "; ".join(methods),
                    "friedman_statistic": np.nan,
                    "p_value": np.nan,
                    "significant_at_0_05": False,
                    "note": "insufficient paired data",
                }
            )
            continue

        stat, p = friedmanchisquare(*[pivot[m].values for m in methods])
        rows.append(
            {
                "analysis_set": "framework_level_selected_runs",
                "metric": metric,
                "n_paired_splits": int(pivot.shape[0]),
                "n_methods": int(pivot.shape[1]),
                "methods": "; ".join(methods),
                "friedman_statistic": stat,
                "p_value": p,
                "significant_at_0_05": bool(p < 0.05),
                "note": "ok",
            }
        )

    out = pd.DataFrame(rows)
    if output_csv is not None:
        out.to_csv(output_csv, index=False, encoding="utf-8")
    return out


def wilcoxon_holm_tests(
    selected_runs: pd.DataFrame,
    metrics: list[str],
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Run pairwise Wilcoxon signed-rank tests with Holm correction."""
    all_rows = []

    for metric in metrics:
        pivot = pivot_metric(selected_runs, metric)
        methods = list(pivot.columns)
        metric_rows = []

        for a, b in combinations(methods, 2):
            diff = pivot[a] - pivot[b]
            diff_nonzero = diff[diff != 0]
            if len(diff_nonzero) == 0:
                stat, p, note = 0.0, 1.0, "all paired differences are zero"
            elif len(diff_nonzero) < 2:
                stat, p, note = np.nan, np.nan, "insufficient nonzero paired differences"
            else:
                stat, p = wilcoxon(pivot[a], pivot[b], zero_method="wilcox", alternative="two-sided")
                note = "ok"

            metric_rows.append(
                {
                    "analysis_set": "framework_level_selected_runs",
                    "metric": metric,
                    "n_paired_splits": int(pivot.shape[0]),
                    "method_a": a,
                    "method_b": b,
                    "mean_a": pivot[a].mean(),
                    "mean_b": pivot[b].mean(),
                    "mean_diff_a_minus_b": pivot[a].mean() - pivot[b].mean(),
                    "better_method_descriptive": a if pivot[a].mean() >= pivot[b].mean() else b,
                    "wilcoxon_statistic": stat,
                    "p_value_raw": p,
                    "note": note,
                }
            )

        pvals = [r["p_value_raw"] for r in metric_rows]
        valid = np.array([pd.notna(p) for p in pvals])
        adj = np.full(len(metric_rows), np.nan)
        reject = np.full(len(metric_rows), False)

        if valid.any():
            reject_valid, adj_valid, _, _ = multipletests(np.array(pvals)[valid], alpha=0.05, method="holm")
            adj[valid] = adj_valid
            reject[valid] = reject_valid

        for i, r in enumerate(metric_rows):
            r["p_value_holm"] = adj[i]
            r["significant_raw_0_05"] = bool(pd.notna(r["p_value_raw"]) and r["p_value_raw"] < 0.05)
            r["significant_holm_0_05"] = bool(reject[i])

        all_rows.extend(metric_rows)

    out = pd.DataFrame(all_rows)
    if output_csv is not None:
        out.to_csv(output_csv, index=False, encoding="utf-8")
    return out
