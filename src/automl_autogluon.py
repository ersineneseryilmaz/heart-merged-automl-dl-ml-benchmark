# src/automl_autogluon.py
import argparse
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, log_loss, confusion_matrix
)

from autogluon.tabular import TabularPredictor

from .utils import set_seed, take_snapshot, diff_snapshot, ensure_dir


def get_dir_size_mb(path: str) -> float:
    total_bytes = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total_bytes += os.path.getsize(fp)
    return total_bytes / (1024 * 1024)


def pick_positive_proba(proba_df: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    labels = sorted(pd.unique(y_train))
    pos_label = labels[-1]  # usually 1
    if pos_label in proba_df.columns:
        return proba_df[pos_label].to_numpy()
    return proba_df.iloc[:, -1].to_numpy()


def run_autogluon(data_path: str, mode: str, budget_s: int, seed: int, out_dir: str):
    ensure_dir(out_dir)
    set_seed(seed)

    df = pd.read_csv(data_path)
    ycol = "target"

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df[ycol]
    )

    predictor_path = os.path.join(out_dir, f"autogluon_{mode}_predictor")
    ensure_dir(predictor_path)

    snap0 = take_snapshot()

    predictor = TabularPredictor(
        label=ycol,
        eval_metric="f1",
        path=predictor_path
    )

    if mode == "light":
        hyperparameters = {
            "GBM": {},  # LightGBM
            "RF": {},
            "XT": {},
            "LR": {},
        }
        predictor.fit(
            train_data=train_df,
            presets="medium_quality_faster_train",
            time_limit=budget_s,
            num_bag_folds=0,
            num_stack_levels=0,
            hyperparameters=hyperparameters,
        )
    else:
        predictor.fit(
            train_data=train_df,
            presets="best_quality",
            time_limit=budget_s,
            num_bag_folds=5,
            num_stack_levels=0,
            hyperparameters=None,
        )

    y_pred = predictor.predict(test_df).astype(int)
    proba_df = predictor.predict_proba(test_df)
    y_proba = pick_positive_proba(proba_df, train_df[ycol])

    y_test = test_df[ycol].astype(int).to_numpy()
    y_pred_np = np.asarray(y_pred).astype(int)

    acc = accuracy_score(y_test, y_pred_np)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred_np)
    pr = precision_score(y_test, y_pred_np)
    rc = recall_score(y_test, y_pred_np)
    ll = log_loss(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred_np)

    lb = predictor.leaderboard(test_df, silent=True)
    lb_path = os.path.join(out_dir, f"autogluon_{mode}_leaderboard.csv")
    lb.to_csv(lb_path, index=False)

    size_mb = get_dir_size_mb(predictor_path)

    snap1 = take_snapshot()
    delta = diff_snapshot(snap0, snap1)

    print(f"\n=== AutoGluon {mode.upper()} (Test) ===")
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
    print(f"Predictor size: {size_mb:.2f} MB")
    print(f"Predictor path: {predictor_path}")
    print(f"Saved LB:       {lb_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", choices=["light", "full"], required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/autogluon")
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.mode)
    run_autogluon(args.data, args.mode, args.budget, args.seed, out_dir)


if __name__ == "__main__":
    main()
