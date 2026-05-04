\
"""Project-wide constants for the time-aware heart disease benchmark."""

from __future__ import annotations

PRIMARY_SEED: int = 42
REPEATED_HOLDOUT_SEEDS: list[int] = [42, 7, 21, 84, 123]

TEST_SIZE: float = 0.20
VALID_SIZE: float = 0.20

FULL_BUDGET_SECONDS: int = 180
LIGHT_BUDGET_SECONDS: int = 60

RAW_DATASET = "cleveland_hungarian_long-beach-va_switzerland.csv"
CLEAN_DATASET = "heart_merged_clean.csv"

FEATURE_COLUMNS: list[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET_COLUMN: str = "num"

BINARY_TARGET = "binary_0_vs_1_4"
ORDINAL_TARGET = "ordinal_3class_0_vs_1_2_vs_3_4"
MULTICLASS_TARGET = "multiclass_5class_0_1_2_3_4"

TARGET_FORMULATIONS: list[str] = [
    BINARY_TARGET,
    ORDINAL_TARGET,
    MULTICLASS_TARGET,
]

METRIC_COLUMNS: list[str] = [
    "accuracy_pct",
    "auc_pct",
    "f1_pct",
    "precision_pct",
    "recall_pct",
    "logloss",
]

RESOURCE_COLUMNS: list[str] = [
    "runtime_s",
    "python_rss_before_mb",
    "python_rss_after_mb",
    "python_rss_delta_mb",
    "system_used_before_mb",
    "system_used_after_mb",
    "system_used_delta_mb",
    "model_size_mb",
]
