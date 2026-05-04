\
"""Target sensitivity analysis for binary, ordinal, and original five-class labels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .classic_ml import build_classic_models, evaluate_model
from .config import FEATURE_COLUMNS, REPEATED_HOLDOUT_SEEDS, TARGET_FORMULATIONS
from .preprocess import class_distribution, load_raw_heart_csv, preprocess_heart_csv, make_train_test_imputed
from .utils import ensure_dir, resource_tracker, set_global_seed, mean_sd


SENSITIVITY_MODELS = [
    "ExtraTrees_balanced",
    "LightGBM_fixed",
    "LogisticRegression_balanced",
    "RandomForest_balanced",
    "XGBoost_fixed",
]


def run_target_sensitivity(
    raw_csv: str | Path,
    output_dir: str | Path,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Run lightweight classical models under alternative target formulations."""
    out = ensure_dir(output_dir)
    seeds = seeds or REPEATED_HOLDOUT_SEEDS
    raw = load_raw_heart_csv(raw_csv)

    dist = class_distribution(raw, TARGET_FORMULATIONS)
    dist.to_csv(out / "target_sensitivity_class_distribution.csv", index=False)

    rows = []
    for formulation in TARGET_FORMULATIONS:
        df = preprocess_heart_csv(raw_csv, output_csv=None, formulation=formulation)

        for seed in seeds:
            set_global_seed(seed)
            X_train, X_test, y_train, y_test, _, _ = make_train_test_imputed(df, seed=seed, scale=False)
            models = build_classic_models(seed)

            for model_name in SENSITIVITY_MODELS:
                if model_name not in models:
                    continue

                model = models[model_name]
                with resource_tracker() as res:
                    model.fit(X_train, y_train)
                    metrics = evaluate_model(model, X_test, y_test)

                rows.append(
                    {
                        "formulation": formulation,
                        "model": model_name,
                        "seed": seed,
                        "status": "ok",
                        **metrics,
                        **res,
                    }
                )

    results = pd.DataFrame(rows)
    results.to_csv(out / "target_sensitivity_results.csv", index=False)
    summarize_target_sensitivity(results, out / "target_sensitivity_summary_formatted.csv")
    best_by_formulation(results, out / "target_sensitivity_best_by_formulation.csv")
    return results


def summarize_target_sensitivity(
    results: pd.DataFrame,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Summarize target sensitivity as mean ± SD by formulation and model."""
    metric_cols = {
        "accuracy_pct": "Accuracy (%)",
        "balanced_accuracy_pct": "Balanced Accuracy (%)",
        "macro_f1_pct": "Macro-F1 (%)",
        "weighted_f1_pct": "Weighted-F1 (%)",
        "auc_pct": "ROC-AUC (%)",
        "logloss": "LogLoss",
        "runtime_s": "Runtime (s)",
    }
    rows = []
    for (formulation, model), g in results.groupby(["formulation", "model"]):
        row = {
            "formulation": formulation,
            "model": model,
            "n_runs": int(len(g)),
        }
        for col, label in metric_cols.items():
            if col in g:
                row[label] = mean_sd(g[col])
        rows.append(row)

    out = pd.DataFrame(rows)
    if output_csv is not None:
        out.to_csv(output_csv, index=False, encoding="utf-8")
    return out


def best_by_formulation(
    results: pd.DataFrame,
    output_csv: str | Path | None = None,
    selection_metric: str = "macro_f1_pct",
) -> pd.DataFrame:
    """Select the best model within each target formulation by mean Macro-F1."""
    summaries = []
    for (formulation, model), g in results.groupby(["formulation", "model"]):
        summaries.append(
            {
                "formulation": formulation,
                "model": model,
                "mean_selection_metric": g[selection_metric].mean(),
                "n_runs": len(g),
                "Accuracy (%)": mean_sd(g["accuracy_pct"]),
                "Balanced Accuracy (%)": mean_sd(g["balanced_accuracy_pct"]),
                "Macro-F1 (%)": mean_sd(g["macro_f1_pct"]),
                "Weighted-F1 (%)": mean_sd(g["weighted_f1_pct"]),
                "ROC-AUC (%)": mean_sd(g["auc_pct"]),
                "LogLoss": mean_sd(g["logloss"]),
                "Runtime (s)": mean_sd(g["runtime_s"]),
            }
        )
    summary = pd.DataFrame(summaries)
    idx = summary.groupby("formulation")["mean_selection_metric"].idxmax()
    best = summary.loc[idx].drop(columns=["mean_selection_metric"]).reset_index(drop=True)
    if output_csv is not None:
        best.to_csv(output_csv, index=False, encoding="utf-8")
    return best
