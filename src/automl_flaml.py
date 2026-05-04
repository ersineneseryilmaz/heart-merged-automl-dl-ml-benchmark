\
"""FLAML runner for the time-aware benchmark."""

from __future__ import annotations

from pathlib import Path
import pickle

import pandas as pd

from .config import LIGHT_BUDGET_SECONDS, FULL_BUDGET_SECONDS
from .preprocess import make_train_test_imputed
from .utils import compute_metrics, ensure_dir, get_path_size_mb, resource_tracker, safe_predict_proba, set_global_seed


def run_flaml(
    df: pd.DataFrame,
    output_dir: str | Path,
    regime: str = "Light",
    seed: int = 42,
) -> pd.DataFrame:
    """Run FLAML under a nominal framework-level time limit."""
    try:
        from flaml import AutoML
    except Exception as exc:
        raise ImportError("FLAML is required for run_flaml().") from exc

    set_global_seed(seed)
    out = ensure_dir(output_dir)
    time_limit = LIGHT_BUDGET_SECONDS if regime.lower() == "light" else FULL_BUDGET_SECONDS
    estimators = ["lgbm", "xgboost", "rf", "extra_tree", "lrl1"] if regime.lower() == "light" else "auto"

    X_train, X_test, y_train, y_test, _, _ = make_train_test_imputed(df, seed=seed, scale=False)
    model_path = out / f"flaml_{regime.lower()}_seed{seed}.pkl"

    with resource_tracker() as res:
        automl = AutoML()
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            task="classification",
            metric="f1",
            time_budget=time_limit,
            estimator_list=estimators,
            seed=seed,
            verbose=0,
        )
        y_pred = automl.predict(X_test)
        y_proba = safe_predict_proba(automl, X_test)
        metrics = compute_metrics(y_test, y_pred, y_proba)

    with model_path.open("wb") as f:
        pickle.dump(automl, f)

    row = {
        "framework": f"FLAML_{regime}",
        "model": str(getattr(automl, "best_estimator", "")),
        "seed": seed,
        "nominal_budget_s": time_limit,
        "status": "ok",
        **metrics,
        **res,
        "model_size_mb": get_path_size_mb(model_path),
    }
    results = pd.DataFrame([row])
    results.to_csv(out / f"flaml_{regime.lower()}_seed{seed}.csv", index=False)
    return results
