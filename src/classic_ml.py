# src/classic_ml.py
import argparse
import os
import time
import warnings
import random
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, log_loss, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

from .utils import set_seed, take_snapshot, diff_snapshot, model_file_size_mb, ensure_dir

warnings.filterwarnings("ignore")

# Optional libs
XGB_AVAILABLE = False
LGB_AVAILABLE = False
CAT_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    pass

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    pass

try:
    from catboost import CatBoostClassifier
    CAT_AVAILABLE = True
except Exception:
    pass


def predict_proba_safe(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1 / (1 + np.exp(-s))
    p = model.predict(X)
    return p.astype(float)


def compute_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    pr = precision_score(y_true, y_pred)
    rc = recall_score(y_true, y_pred)
    ll = log_loss(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    return acc, auc, f1, pr, rc, ll, cm


def build_candidates(seed=42, mode="full"):
    candidates = {}

    candidates["LogReg"] = {
        "base": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=-1))
        ]),
        "space": {
            "clf__C": np.logspace(-2, 2, 30),
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs", "liblinear"],
        }
    }

    candidates["RF"] = {
        "base": RandomForestClassifier(random_state=seed, n_jobs=-1),
        "space": {
            "n_estimators": [200, 400, 800, 1200] if mode == "full" else [300],
            "max_depth": [None, 3, 5, 7, 10],
            "max_features": ["sqrt", "log2", None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
        }
    }

    candidates["ExtraTrees"] = {
        "base": ExtraTreesClassifier(random_state=seed, n_jobs=-1),
        "space": {
            "n_estimators": [300, 600, 1000, 1500] if mode == "full" else [600],
            "max_depth": [None, 3, 5, 7, 10],
            "max_features": ["sqrt", "log2", None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    }

    candidates["MLP_sklearn"] = {
        "base": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                activation="relu",
                alpha=1e-4,
                max_iter=300 if mode == "full" else 150,
                random_state=seed
            ))
        ]),
        "space": {
            "clf__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
            "clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "clf__learning_rate_init": [1e-4, 3e-4, 1e-3, 3e-3],
        }
    }

    if XGB_AVAILABLE:
        candidates["XGBoost"] = {
            "base": XGBClassifier(
                eval_metric="logloss",
                n_jobs=-1,
                random_state=seed,
                tree_method="hist",
            ),
            "space": {
                "n_estimators": [150, 300, 600, 1000] if mode == "full" else [300],
                "max_depth": [2, 3, 4, 5, 6],
                "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "min_child_weight": [1, 3, 5, 10],
                "reg_lambda": [0.5, 1.0, 2.0, 5.0],
            }
        }

    if LGB_AVAILABLE:
        candidates["LightGBM"] = {
            "base": lgb.LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1),
            "space": {
                "n_estimators": [200, 500, 900, 1500] if mode == "full" else [400],
                "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                "num_leaves": [7, 15, 31, 63],
                "min_child_samples": [5, 10, 20, 40],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "reg_lambda": [0.0, 1.0, 3.0, 10.0],
            }
        }

    if CAT_AVAILABLE:
        candidates["CatBoost"] = {
            "base": CatBoostClassifier(
                loss_function="Logloss",
                verbose=False,
                random_seed=seed,
                allow_writing_files=False,
            ),
            "space": {
                "iterations": [300, 800, 1500, 2500] if mode == "full" else [600],
                "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                "depth": [3, 4, 5, 6, 7, 8],
                "l2_leaf_reg": [1, 3, 5, 10],
            }
        }

    return candidates


def timed_search(base_model, param_space, X_tr, y_tr, X_val, y_val, time_left_s, seed=42, max_trials=100000):
    t0 = time.time()
    best = {"f1": -1.0, "params": None, "fit_time_s": np.nan, "trials": 0}
    sampler = ParameterSampler(param_space, n_iter=max_trials, random_state=seed)

    for params in sampler:
        if time.time() - t0 >= time_left_s:
            break
        m = clone(base_model)
        try:
            m.set_params(**params)
        except Exception:
            continue

        t_fit = time.time()
        try:
            m.fit(X_tr, y_tr)
            fit_time = time.time() - t_fit

            proba = predict_proba_safe(m, X_val)
            pred = (proba >= 0.5).astype(int)
            f1v = f1_score(y_val, pred)

            best["trials"] += 1
            if f1v > best["f1"]:
                best.update({"f1": f1v, "params": params, "fit_time_s": fit_time})
        except Exception:
            continue

    return best


def run_classic_ml(data_path: str, mode: str, budget_s: int, seed: int, out_dir: str, topk: int = 3):
    ensure_dir(out_dir)
    set_seed(seed)

    df = pd.read_csv(data_path)
    ycol = "target"
    X = df.drop(columns=[ycol])
    y = df[ycol].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
    )

    snap0 = take_snapshot()
    start_time = time.time()

    candidates = build_candidates(seed=seed, mode=mode)
    rows = []

    # baseline pass
    baseline = []
    for name, item in candidates.items():
        m = clone(item["base"])
        t_fit = time.time()
        try:
            m.fit(X_tr, y_tr)
            fit_time = time.time() - t_fit
            proba = predict_proba_safe(m, X_val)
            pred = (proba >= 0.5).astype(int)
            f1v = f1_score(y_val, pred)
        except Exception:
            fit_time = np.nan
            f1v = -1.0

        baseline.append((name, f1v))
        rows.append({"model": name, "phase": "baseline", "val_f1": f1v, "params": None, "fit_time_s": fit_time})

    baseline.sort(key=lambda x: x[1], reverse=True)
    best_name, best_f1 = baseline[0][0], baseline[0][1]
    best_params = None

    # full: timed search on topK
    if mode == "full":
        top_models = [m for m, _ in baseline[:max(1, topk)]]
        for name in top_models:
            remaining = budget_s - (time.time() - start_time)
            if remaining <= 5:
                break
            per_model = max(5.0, remaining / max(1, len(top_models)))
            item = candidates[name]
            best = timed_search(item["base"], item["space"], X_tr, y_tr, X_val, y_val, per_model, seed=seed)

            rows.append({
                "model": name,
                "phase": "timed_search",
                "val_f1": best["f1"],
                "params": best["params"],
                "fit_time_s": best["fit_time_s"],
                "trials": best["trials"],
            })

            if best["f1"] > best_f1:
                best_name, best_f1 = name, best["f1"]
                best_params = best["params"]

    # retrain on outer train
    best_model = clone(candidates[best_name]["base"])
    if best_params:
        try:
            best_model.set_params(**best_params)
        except Exception:
            pass
    best_model.fit(X_train, y_train)

    test_proba = predict_proba_safe(best_model, X_test)
    test_pred = (test_proba >= 0.5).astype(int)
    acc, auc, f1, pr, rc, ll, cm = compute_metrics(y_test, test_pred, test_proba)

    # save artifacts
    out_model = os.path.join(out_dir, f"classicML_{mode}_best_{best_name}.pkl")
    joblib.dump(best_model, out_model)
    size_mb = model_file_size_mb(out_model)

    lb = pd.DataFrame(rows)
    lb.to_csv(os.path.join(out_dir, f"classicML_{mode}_leaderboard.csv"), index=False)

    snap1 = take_snapshot()
    delta = diff_snapshot(snap0, snap1)

    print(f"\n=== Classic ML {mode.upper()} Best = {best_name} (Test) ===")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"AUC:       {auc*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    print(f"Precision: {pr*100:.2f}%")
    print(f"Recall:    {rc*100:.2f}%")
    print(f"LogLoss:   {ll:.2f}")
    print("\nConfusion Matrix:")
    print(cm)

    print("\n=== Resources ===")
    print(f"Runtime (wall): {delta.wall_time_s:.2f} s (Budget={budget_s}s)")
    print(f"Process RSS Δ:  {delta.proc_rss_delta_mb:.2f} MB")
    print(f"System RAM Δ:   {delta.sys_used_delta_gb:.2f} GB")
    print(f"Model size:     {size_mb:.4f} MB")
    print(f"Saved model:    {out_model}")
    print(f"Saved LB:       {os.path.join(out_dir, f'classicML_{mode}_leaderboard.csv')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", choices=["light", "full"], required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/classic_ml")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.mode)
    run_classic_ml(args.data, args.mode, args.budget, args.seed, out_dir, topk=args.topk)


if __name__ == "__main__":
    main()
