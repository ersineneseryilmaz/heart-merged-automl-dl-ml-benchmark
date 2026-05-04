\
"""Classical ML benchmark runners for the time-aware heart disease benchmark."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .preprocess import make_train_test_imputed
from .utils import compute_metrics, ensure_dir, get_path_size_mb, resource_tracker, safe_predict_proba, set_global_seed

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


def build_classic_models(seed: int = 42) -> dict[str, Any]:
    """Return lightweight classical ML baselines."""
    models: dict[str, Any] = {
        "LogisticRegression_balanced": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "RandomForest_balanced": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "ExtraTrees_balanced": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "DecisionTree_balanced": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=seed,
        ),
        "GaussianNB": GaussianNB(),
        "SVM_RBF_balanced": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost_fixed"] = XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )

    if LGBMClassifier is not None:
        models["LightGBM_fixed"] = LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

    return models


def classic_param_spaces(seed: int = 42) -> dict[str, dict[str, list[Any]]]:
    """Small randomized-search spaces for the full classical ML regime."""
    spaces: dict[str, dict[str, list[Any]]] = {
        "RandomForest_balanced": {
            "n_estimators": [200, 300, 500],
            "max_depth": [None, 3, 5, 8],
            "min_samples_split": [2, 5, 10],
        },
        "ExtraTrees_balanced": {
            "n_estimators": [200, 300, 500],
            "max_depth": [None, 3, 5, 8],
            "min_samples_split": [2, 5, 10],
        },
        "DecisionTree_balanced": {
            "max_depth": [None, 3, 5, 8],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    }

    if XGBClassifier is not None:
        spaces["XGBoost_fixed"] = {
            "n_estimators": [100, 150, 250, 400],
            "max_depth": [2, 3, 4, 5],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
        }

    if LGBMClassifier is not None:
        spaces["LightGBM_fixed"] = {
            "n_estimators": [100, 150, 250, 400],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
            "max_depth": [-1, 3, 5, 8],
        }

    return spaces


def evaluate_model(model: Any, X_test, y_test) -> dict[str, Any]:
    """Evaluate a fitted sklearn-like model."""
    y_pred = model.predict(X_test)
    y_proba = safe_predict_proba(model, X_test)
    return compute_metrics(y_test, y_pred, y_proba)


def run_classic_ml_light(
    df: pd.DataFrame,
    output_dir: str | Path,
    seed: int = 42,
    save_models: bool = True,
) -> pd.DataFrame:
    """Run lightweight fixed classical ML baselines."""
    set_global_seed(seed)
    out = ensure_dir(output_dir)
    X_train, X_test, y_train, y_test, _, _ = make_train_test_imputed(df, seed=seed, scale=False)
    rows = []

    for name, model in build_classic_models(seed).items():
        model_path = out / f"{name}_seed{seed}.pkl"
        with resource_tracker() as res:
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
        if save_models:
            with model_path.open("wb") as f:
                pickle.dump(model, f)
        rows.append(
            {
                "framework": "ClassicML_Light",
                "model": name,
                "seed": seed,
                "status": "ok",
                **metrics,
                **res,
                "model_size_mb": get_path_size_mb(model_path),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(out / f"classic_ml_light_seed{seed}.csv", index=False)
    return results


def run_classic_ml_full(
    df: pd.DataFrame,
    output_dir: str | Path,
    seed: int = 42,
    n_iter: int = 12,
    cv: int = 3,
    save_models: bool = True,
) -> pd.DataFrame:
    """Run full classical ML baselines with small randomized searches."""
    set_global_seed(seed)
    out = ensure_dir(output_dir)
    X_train, X_test, y_train, y_test, _, _ = make_train_test_imputed(df, seed=seed, scale=False)
    models = build_classic_models(seed)
    spaces = classic_param_spaces(seed)
    rows = []

    for name, base_model in models.items():
        model_path = out / f"{name}_full_seed{seed}.pkl"
        with resource_tracker() as res:
            if name in spaces:
                search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=spaces[name],
                    n_iter=n_iter,
                    cv=cv,
                    scoring="f1",
                    random_state=seed,
                    n_jobs=-1,
                    error_score="raise",
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_
                best_params = search.best_params_
            else:
                model = base_model.fit(X_train, y_train)
                best_params = {}
            metrics = evaluate_model(model, X_test, y_test)

        if save_models:
            with model_path.open("wb") as f:
                pickle.dump(model, f)

        rows.append(
            {
                "framework": "ClassicML_Full",
                "model": name,
                "seed": seed,
                "status": "ok",
                "best_params": str(best_params),
                **metrics,
                **res,
                "model_size_mb": get_path_size_mb(model_path),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(out / f"classic_ml_full_seed{seed}.csv", index=False)
    return results
