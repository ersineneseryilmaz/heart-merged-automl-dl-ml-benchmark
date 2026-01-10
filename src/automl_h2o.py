# src/automl_h2o.py
import argparse
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, log_loss, confusion_matrix
)

import h2o
from h2o.automl import H2OAutoML

from .utils import set_seed, take_snapshot, diff_snapshot, model_file_size_mb, ensure_dir


def run_h2o(data_path: str, mode: str, budget_s: int, seed: int, out_dir: str):
    ensure_dir(out_dir)
    set_seed(seed)

    df = pd.read_csv(data_path)
    ycol = "target"

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df[ycol]
    )

    snap0 = take_snapshot()

    # H2O init
    h2o.init(max_mem_size="4G")
    h2o.no_progress()

    train_h2o = h2o.H2OFrame(train_df)
    test_h2o = h2o.H2OFrame(test_df)

    train_h2o[ycol] = train_h2o[ycol].asfactor()
    test_h2o[ycol] = test_h2o[ycol].asfactor()

    x_cols = [c for c in df.columns if c != ycol]

    if mode == "light":
        aml = H2OAutoML(
            max_runtime_secs=budget_s,
            nfolds=0,
            seed=seed,
            sort_metric="F1",
            exclude_algos=["StackedEnsemble", "DeepLearning"],
        )
    else:
        aml = H2OAutoML(
            max_runtime_secs=budget_s,
            nfolds=5,
            seed=seed,
            sort_metric="F1",
            exclude_algos=None,
        )

    aml.train(x=x_cols, y=ycol, training_frame=train_h2o)
    best = aml.leader

    preds = best.predict(test_h2o).as_data_frame()  # predict, p0, p1
    y_pred = preds["predict"].astype(int).to_numpy()

    proba_col = "p1" if "p1" in preds.columns else preds.columns[-1]
    y_proba = preds[proba_col].to_numpy()

    y_test = test_df[ycol].astype(int).to_numpy()

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    pr = precision_score(y_test, y_pred)
    rc = recall_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    # save model
    model_file = h2o.save_model(best, path=out_dir, force=True)
    size_mb = model_file_size_mb(model_file)

    # leaderboard
    lb = aml.leaderboard.as_data_frame()
    lb_path = os.path.join(out_dir, f"h2o_{mode}_leaderboard.csv")
    lb.to_csv(lb_path, index=False)

    # shutdown
    h2o.shutdown(prompt=False)

    snap1 = take_snapshot()
    delta = diff_snapshot(snap0, snap1)

    print(f"\n=== H2O AutoML {mode.upper()} (Test) ===")
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
    print(f"Model size:     {size_mb:.2f} MB")
    print(f"Saved model:    {model_file}")
    print(f"Saved LB:       {lb_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", choices=["light", "full"], required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/h2o")
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.mode)
    run_h2o(args.data, args.mode, args.budget, args.seed, out_dir)


if __name__ == "__main__":
    main()
