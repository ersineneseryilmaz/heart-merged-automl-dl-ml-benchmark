\
"""H2O AutoML runner for the time-aware benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import LIGHT_BUDGET_SECONDS, FULL_BUDGET_SECONDS
from .preprocess import make_train_test_imputed
from .utils import compute_metrics, ensure_dir, get_path_size_mb, resource_tracker, set_global_seed


def run_h2o(
    df: pd.DataFrame,
    output_dir: str | Path,
    regime: str = "Light",
    seed: int = 42,
) -> pd.DataFrame:
    """Run H2O AutoML under a nominal framework-level time limit."""
    try:
        import h2o
        from h2o.automl import H2OAutoML
    except Exception as exc:
        raise ImportError("h2o is required for run_h2o().") from exc

    set_global_seed(seed)
    out = ensure_dir(output_dir)
    time_limit = LIGHT_BUDGET_SECONDS if regime.lower() == "light" else FULL_BUDGET_SECONDS

    X_train, X_test, y_train, y_test, _, _ = make_train_test_imputed(df, seed=seed, scale=False)
    train_df = X_train.copy()
    test_df = X_test.copy()
    train_df["target"] = y_train.astype(str).values
    test_df["target"] = y_test.astype(str).values

    h2o.init(max_mem_size="2G", nthreads=-1, silent=True)
    train_h2o = h2o.H2OFrame(train_df)
    test_h2o = h2o.H2OFrame(test_df)
    train_h2o["target"] = train_h2o["target"].asfactor()
    test_h2o["target"] = test_h2o["target"].asfactor()

    model_dir = out / f"h2o_{regime.lower()}_seed{seed}"
    ensure_dir(model_dir)

    with resource_tracker() as res:
        aml = H2OAutoML(
            max_runtime_secs=time_limit,
            seed=seed,
            sort_metric="F1",
            verbosity="warn",
            nfolds=0,
        )
        aml.train(x=list(X_train.columns), y="target", training_frame=train_h2o)
        leader = aml.leader
        pred = leader.predict(test_h2o).as_data_frame()
        y_pred = pred["predict"].astype(int).values
        if "p1" in pred.columns:
            y_proba = np.column_stack([pred["p0"].values, pred["p1"].values])
        else:
            y_proba = None
        metrics = compute_metrics(y_test, y_pred, y_proba)

    model_path = h2o.save_model(leader, path=str(model_dir), force=True)
    row = {
        "framework": f"H2O_{regime}",
        "model": str(leader.model_id),
        "seed": seed,
        "nominal_budget_s": time_limit,
        "status": "ok",
        **metrics,
        **res,
        "model_size_mb": get_path_size_mb(model_path),
    }
    results = pd.DataFrame([row])
    results.to_csv(out / f"h2o_{regime.lower()}_seed{seed}.csv", index=False)
    h2o.shutdown(prompt=False)
    return results
