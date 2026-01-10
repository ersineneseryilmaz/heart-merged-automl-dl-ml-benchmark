# src/automl_mljar.py
import argparse
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, log_loss, confusion_matrix
)

from supervised.automl import AutoML

from .utils import set_seed, take_snapshot, diff_snapshot, model_file_size_mb, ensure_dir


def run_mljar(data_path: str, mode: str, budget_s: int, seed: int, out_dir: str):
    ensure_dir(out_dir)
    set_seed(seed)

    df = pd.read_csv(data_path)
    ycol = "target"

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df[ycol]
    )

    X_train = train_df.drop(columns=[ycol])
    y_train = train_df[ycol]
    X_test = test_df.drop(columns=[ycol])
    y_test = test_df[ycol].astype(int).to_numpy()

    snap0 = take_snapshot()

    if mode == "light":
        automl = AutoML(
            mode="Explain",
            algorithms=["LightGBM", "Xgboost", "CatBoost", "Random Forest"],
            total_time_limit=budget_s,
            explain_level=1,
            random_state=seed,
            start_random_models=2,
            hill_climbing_steps=0,
        )
    else:
        automl = AutoML(
            mode="Compete",
            algorithms=["LightGBM", "Xgboost", "CatBoost", "Random Forest", "Extra Trees", "Neural Network"],
            total_time_limit=budget_s,
            model_time_limit=max(30, int(budget_s * 0.55)),
            start_random_models=5,
            hill_climbing_steps=3,
            stack_models=True,
            golden_features=True,
            features_selection=True,
            explain_level=2,
            random_state=seed,
        )

    automl.fit(X_train, y_train)

    y_pred = automl.predict(X_test).astype(int)
    y_proba = automl.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    pr = precision_score(y_test, y_pred)
    rc = recall_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    # leaderboard
    lb = automl.get_leaderboard()
    lb_path = os.path.join(out_dir, f"mljar_{mode}_leaderboard.csv")
    lb.to_csv(lb_path, index=False)

    # save best model object
    best_model = automl._best_model
    out_model = os.path.join(out_dir, f"mljar_{mode}_best_model.pkl")
    joblib.dump(best_model, out_model)
    size_mb = model_file_size_mb(out_model)

    snap1 = take_snapshot()
    delta = diff_snapshot(snap0, snap1)

    print(f"\n=== MLJAR {mode.upper()} (Test) ===")
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
    print(f"Saved LB:       {lb_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", choices=["light", "full"], required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/mljar")
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.mode)
    run_mljar(args.data, args.mode, args.budget, args.seed, out_dir)


if __name__ == "__main__":
    main()
