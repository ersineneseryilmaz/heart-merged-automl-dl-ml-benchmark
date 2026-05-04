\
"""MLJAR AutoML runner for the time-aware benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import LIGHT_BUDGET_SECONDS, FULL_BUDGET_SECONDS
from .preprocess import make_train_test_imputed
from .utils import compute_metrics, ensure_dir, get_path_size_mb, resource_tracker, safe_predict_proba, set_global_seed


def run_mljar(
    df: pd.DataFrame,
    output_dir: str | Path,
    regime: str = "Light",
    seed: int = 42,
) -> pd.DataFrame:
    """Run MLJAR AutoML under a nominal framework-level time limit."""
    try:
        from supervised.automl import AutoML
    except Exception as exc:
        raise ImportError("mljar-supervised is required for run_mljar().") from exc

    set_global_seed(seed)
    out = ensure_dir(output_dir)
    time_limit = LIGHT_BUDGET_SECONDS if regime.lower() == "light" else FULL_BUDGET_SECONDS
    mode = "Explain" if regime.lower() == "light" else "Compete"

    X_train, X_test, y_train, y_test, _, _ = make_train_test_imputed(df, seed=seed, scale=False)
    model_dir = out / f"mljar_{regime.lower()}_seed{seed}"

    with resource_tracker() as res:
        automl = AutoML(
            mode=mode,
            total_time_limit=time_limit,
            algorithms=["Baseline", "Linear", "Decision Tree", "Random Forest", "Xgboost", "LightGBM"]
            if regime.lower() == "light"
            else "auto",
            results_path=str(model_dir),
            random_state=seed,
            eval_metric="f1",
        )
        automl.fit(X_train, y_train)
        y_pred = automl.predict(X_test)
        y_proba = safe_predict_proba(automl, X_test)
        metrics = compute_metrics(y_test, y_pred, y_proba)

    row = {
        "framework": f"MLJAR_{regime}",
        "model": "Ensemble_or_best_model",
        "seed": seed,
        "nominal_budget_s": time_limit,
        "status": "ok",
        **metrics,
        **res,
        "model_size_mb": get_path_size_mb(model_dir),
    }
    results = pd.DataFrame([row])
    results.to_csv(out / f"mljar_{regime.lower()}_seed{seed}.csv", index=False)
    return results
