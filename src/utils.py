\
"""Utility functions for metrics, logging, reproducibility, and file handling."""

from __future__ import annotations

import gc
import json
import os
import random
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_global_seed(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and PyTorch when available."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def now_stamp() -> str:
    """Return a filesystem-friendly timestamp."""
    return time.strftime("%Y%m%d_%H%M%S")


def get_process_memory_mb() -> float:
    """Return current Python process RSS in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)


def get_system_used_memory_mb() -> float:
    """Return system used memory in MB."""
    vm = psutil.virtual_memory()
    return vm.used / (1024**2)


def get_path_size_mb(path: str | Path) -> float:
    """Return file or directory size in MB."""
    p = Path(path)
    if not p.exists():
        return 0.0
    if p.is_file():
        return p.stat().st_size / (1024**2)
    total = 0
    for item in p.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total / (1024**2)


@contextmanager
def resource_tracker() -> Iterator[dict[str, float]]:
    """Track runtime and memory deltas for a block."""
    gc.collect()
    start_time = time.perf_counter()
    rss_before = get_process_memory_mb()
    sys_before = get_system_used_memory_mb()
    info: dict[str, float] = {
        "python_rss_before_mb": rss_before,
        "system_used_before_mb": sys_before,
    }
    try:
        yield info
    finally:
        gc.collect()
        runtime = time.perf_counter() - start_time
        rss_after = get_process_memory_mb()
        sys_after = get_system_used_memory_mb()
        info.update(
            {
                "runtime_s": runtime,
                "python_rss_after_mb": rss_after,
                "python_rss_delta_mb": rss_after - rss_before,
                "system_used_after_mb": sys_after,
                "system_used_delta_mb": sys_after - sys_before,
            }
        )


def safe_predict_proba(model: Any, X: Any) -> np.ndarray | None:
    """Return probability estimates if the model exposes them."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return np.asarray(proba)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
        if scores.ndim == 1:
            # Convert scores to pseudo-probabilities for binary ranking metrics only.
            exp = np.exp(scores - np.max(scores))
            p1 = exp / (1.0 + exp)
            return np.column_stack([1 - p1, p1])
    return None


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_proba: np.ndarray | None = None,
    average: str = "binary",
) -> dict[str, Any]:
    """Compute predictive metrics for binary, ordinal, or multiclass tasks."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels = np.unique(y_true_arr)

    if len(labels) > 2:
        f1_avg = "macro"
        precision_avg = "macro"
        recall_avg = "macro"
    else:
        f1_avg = average
        precision_avg = average
        recall_avg = average

    out: dict[str, Any] = {
        "accuracy_pct": accuracy_score(y_true_arr, y_pred_arr) * 100,
        "balanced_accuracy_pct": balanced_accuracy_score(y_true_arr, y_pred_arr) * 100,
        "f1_pct": f1_score(y_true_arr, y_pred_arr, average=f1_avg, zero_division=0) * 100,
        "macro_f1_pct": f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0) * 100,
        "weighted_f1_pct": f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0) * 100,
        "precision_pct": precision_score(y_true_arr, y_pred_arr, average=precision_avg, zero_division=0) * 100,
        "recall_pct": recall_score(y_true_arr, y_pred_arr, average=recall_avg, zero_division=0) * 100,
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr).tolist(),
    }

    if y_proba is not None:
        proba = np.asarray(y_proba)
        try:
            if len(labels) == 2:
                score = proba[:, 1] if proba.ndim == 2 else proba
                out["auc_pct"] = roc_auc_score(y_true_arr, score) * 100
                out["logloss"] = log_loss(y_true_arr, proba, labels=[0, 1])
            else:
                out["auc_pct"] = roc_auc_score(
                    y_true_arr, proba, multi_class="ovr", average="macro"
                ) * 100
                out["logloss"] = log_loss(y_true_arr, proba, labels=list(labels))
        except Exception:
            out["auc_pct"] = np.nan
            out["logloss"] = np.nan
    else:
        out["auc_pct"] = np.nan
        out["logloss"] = np.nan

    return out


def save_json(obj: Any, path: str | Path) -> None:
    """Save an object as JSON."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame as UTF-8 CSV."""
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False, encoding="utf-8")


def mean_sd(series: pd.Series, decimals: int = 2) -> str:
    """Format a series as mean ± SD."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return ""
    return f"{s.mean():.{decimals}f} ± {s.std(ddof=1):.{decimals}f}"


def copy_or_remove_dir(path: str | Path) -> None:
    """Remove an output directory if it exists."""
    p = Path(path)
    if p.exists() and p.is_dir():
        shutil.rmtree(p)
