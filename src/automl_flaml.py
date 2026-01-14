# src/automl_flaml.py
import argparse
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, log_loss, confusion_matrix

from flaml import AutoML

from .utils import set_seed, take_snapshot, diff_snapshot, model_file_size_mb


def pick_pos_proba(proba: np.ndarray) -> np.ndarray:
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1]
    if proba.ndim == 1:
        return proba
    return proba[:, -1]


def run_flaml(data_path: str, mode: str, budget_s: int, seed: int, out_dir: str):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    target_col = "target"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    snap0 = take_snapshot()

    automl = AutoML()

    if mode == "full":
        settings = {
            "time_budget": budget_s,
            "task": "classification",
            "metric": "f1",
            "seed": seed,
            "log_file_name": os.path.join(out_dir, "flaml_full.log"),
            # ✅ FULL: holdout
            "eval_method": "holdout",
            "n_splits": 5,
            "n_jobs": -1,
        }
    else:
        settings = {
            "time_budget": budget_s,
            "task": "classification",
            "metric": "f1",
            "seed": seed,
            "log_file_name": os.path.join(out_dir, "flaml_light.log"),
            # LIGHT: holdout
            "eval_method": "holdout",
            "split_ratio": 0.2,
            "estimator_list": ["lgbm", "rf", "extra_tree", "lrl1"],
            "n_jobs": -1,
        }

    automl.fit(X_train=X_train, y_train=y_train, **settings)

    y_pred = automl.predict(X_test).astype(int)
    y_proba = pick_pos_proba(automl.predict_proba(X_test))

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    pr = precision_score(y_test, y_pred)
    rc = recall_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    # save model
    best_model = automl.model
    out_model = os.path.join(out_dir, f"flaml_{mode}_best_model.pkl")
    joblib.dump(best_model, out_model)
    size_mb = model_file_size_mb(out_model)

    snap1 = take_snapshot()
    delta = diff_snapshot(snap0, snap1)

    print(f"\n=== FLAML {mode.upper()} (Test) ===")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="heart_merged_clean.csv")
    ap.add_argument("--mode", choices=["full", "light"], required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/flaml")
    args = ap.parse_args()

    run_flaml(args.data, args.mode, args.budget, args.seed, args.out)


if __name__ == "__main__":
    main()
