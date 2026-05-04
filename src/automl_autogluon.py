\
"""AutoGluon runner for the time-aware benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import LIGHT_BUDGET_SECONDS, FULL_BUDGET_SECONDS
from .preprocess import make_train_test_imputed
from .utils import compute_metrics, ensure_dir, get_path_size_mb, resource_tracker, set_global_seed


def run_autogluon(
    df: pd.DataFrame,
    output_dir: str | Path,
    regime: str = "Light",
    seed: int = 42,
) -> pd.DataFrame:
    """Run AutoGluon under a nominal framework-level time limit."""
    try:
        from autogluon.tabular import TabularPredictor
    except Exception as exc:
        raise ImportError("AutoGluon is required for run_autogluon().") from exc

    set_global_seed(seed)
    out = ensure_dir(output_dir)
    time_limit = LIGHT_BUDGET_SECONDS if regime.lower() == "light" else FULL_BUDGET_SECONDS
    preset = "medium_quality_faster_train" if regime.lower() == "light" else "best_quality"

    X_train, X_test, y_train, y_test, _, _ = make_train_test_imputed(df, seed=seed, scale=False)
    train_df = X_train.copy()
    train_df["target"] = y_train.values

    model_dir = out / f"autogluon_{regime.lower()}_seed{seed}"

    with resource_tracker() as res:
        predictor = TabularPredictor(
            label="target",
            path=str(model_dir),
            eval_metric="f1",
            verbosity=0,
        ).fit(
            train_df,
            time_limit=time_limit,
            presets=preset,
        )
        y_pred = predictor.predict(X_test)
        proba_df = predictor.predict_proba(X_test)
        y_proba = proba_df.values if hasattr(proba_df, "values") else proba_df
        metrics = compute_metrics(y_test, y_pred, y_proba)

    row = {
        "framework": f"AutoGluon_{regime}",
        "model": str(predictor.model_best),
        "seed": seed,
        "nominal_budget_s": time_limit,
        "status": "ok",
        **metrics,
        **res,
        "model_size_mb": get_path_size_mb(model_dir),
    }
    results = pd.DataFrame([row])
    results.to_csv(out / f"autogluon_{regime.lower()}_seed{seed}.csv", index=False)
    return results
