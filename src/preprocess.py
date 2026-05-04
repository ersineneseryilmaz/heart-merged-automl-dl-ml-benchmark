\
"""Preprocessing utilities for the merged UCI Heart Disease dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    BINARY_TARGET,
    FEATURE_COLUMNS,
    MULTICLASS_TARGET,
    ORDINAL_TARGET,
    PRIMARY_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
)
from .utils import ensure_dir


@dataclass
class SplitData:
    """Container for split arrays and fitted transformers."""

    X_train: pd.DataFrame | np.ndarray
    X_valid: pd.DataFrame | np.ndarray
    X_test: pd.DataFrame | np.ndarray
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series
    imputer: SimpleImputer
    scaler: StandardScaler | None


def load_raw_heart_csv(path: str | Path) -> pd.DataFrame:
    """Load the raw merged CSV and standardize missing placeholders."""
    df = pd.read_csv(path, na_values=["?", "??", "NA", "N/A", "", " "])
    df.columns = [str(c).strip() for c in df.columns]
    return df


def create_target(df: pd.DataFrame, formulation: str = BINARY_TARGET) -> pd.Series:
    """Create the requested target formulation from the original 0-4 label."""
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    if formulation == BINARY_TARGET:
        return (y > 0).astype("Int64")
    if formulation == ORDINAL_TARGET:
        # 0 = no disease, 1 = original 1-2, 2 = original 3-4
        mapped = pd.Series(np.nan, index=df.index, dtype="float")
        mapped[y == 0] = 0
        mapped[y.isin([1, 2])] = 1
        mapped[y.isin([3, 4])] = 2
        return mapped.astype("Int64")
    if formulation == MULTICLASS_TARGET:
        return y.astype("Int64")

    raise ValueError(f"Unknown target formulation: {formulation}")


def preprocess_heart_csv(
    input_csv: str | Path,
    output_csv: str | Path | None = None,
    formulation: str = BINARY_TARGET,
) -> pd.DataFrame:
    """Create a cleaned feature table with a selected target formulation.

    This function performs schema normalization and target creation. Split-based
    imputation/scaling should be fitted later on training partitions only.
    """
    df = load_raw_heart_csv(input_csv)

    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["target"] = create_target(df, formulation=formulation)
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)

    clean = df[FEATURE_COLUMNS + ["target"]].copy()

    if output_csv is not None:
        out = Path(output_csv)
        ensure_dir(out.parent)
        clean.to_csv(out, index=False, encoding="utf-8")

    return clean


def class_distribution(
    raw_df: pd.DataFrame,
    formulations: list[str],
) -> pd.DataFrame:
    """Return class distribution for alternative target formulations."""
    rows = []
    for formulation in formulations:
        y = create_target(raw_df, formulation).dropna().astype(int)
        total = len(y)
        for label, count in y.value_counts().sort_index().items():
            rows.append(
                {
                    "formulation": formulation,
                    "class_label": int(label),
                    "count": int(count),
                    "percentage": count / total * 100 if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def make_outer_split(
    df: pd.DataFrame,
    seed: int = PRIMARY_SEED,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified outer train/test split."""
    X = df[FEATURE_COLUMNS].copy()
    y = df["target"].copy()
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )


def make_train_valid_test(
    df: pd.DataFrame,
    seed: int = PRIMARY_SEED,
    test_size: float = TEST_SIZE,
    valid_size: float = 0.20,
    scale: bool = False,
) -> SplitData:
    """Create train/valid/test splits with imputation fitted on train only."""
    X_train_outer, X_test, y_train_outer, y_test = make_outer_split(
        df, seed=seed, test_size=test_size
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_outer,
        y_train_outer,
        test_size=valid_size,
        random_state=seed,
        stratify=y_train_outer,
    )

    imputer = SimpleImputer(strategy="median")
    X_train_i = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURE_COLUMNS)
    X_valid_i = pd.DataFrame(imputer.transform(X_valid), columns=FEATURE_COLUMNS)
    X_test_i = pd.DataFrame(imputer.transform(X_test), columns=FEATURE_COLUMNS)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train_out = scaler.fit_transform(X_train_i)
        X_valid_out = scaler.transform(X_valid_i)
        X_test_out = scaler.transform(X_test_i)
    else:
        X_train_out = X_train_i
        X_valid_out = X_valid_i
        X_test_out = X_test_i

    return SplitData(
        X_train=X_train_out,
        X_valid=X_valid_out,
        X_test=X_test_out,
        y_train=y_train.reset_index(drop=True),
        y_valid=y_valid.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        imputer=imputer,
        scaler=scaler,
    )


def make_train_test_imputed(
    df: pd.DataFrame,
    seed: int = PRIMARY_SEED,
    test_size: float = TEST_SIZE,
    scale: bool = False,
):
    """Create an imputed train/test split for frameworks with native validation."""
    X_train, X_test, y_train, y_test = make_outer_split(df, seed=seed, test_size=test_size)
    imputer = SimpleImputer(strategy="median")
    X_train_i = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURE_COLUMNS)
    X_test_i = pd.DataFrame(imputer.transform(X_test), columns=FEATURE_COLUMNS)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train_i = pd.DataFrame(scaler.fit_transform(X_train_i), columns=FEATURE_COLUMNS)
        X_test_i = pd.DataFrame(scaler.transform(X_test_i), columns=FEATURE_COLUMNS)

    return X_train_i, X_test_i, y_train.reset_index(drop=True), y_test.reset_index(drop=True), imputer, scaler
