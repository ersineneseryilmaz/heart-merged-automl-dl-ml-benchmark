# src/preprocess.py
import argparse
import pandas as pd

from .utils import ensure_dir


EXPECTED_COLS = [
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach",
    "exang","oldpeak","slope","ca","thal","target"
]


def preprocess_heart_csv(input_csv_path: str, output_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        input_csv_path,
        na_values=["?", "??", "???", "????", " ?"],
        skipinitialspace=True,
        engine="python",
    )
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nFound: {list(df.columns)}")

    for c in EXPECTED_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows with missing target
    df = df[df["target"].notna()].copy()

    # binary target: 0 -> 0, 1-4 -> 1
    df["target"] = (df["target"] > 0).astype(int)

    feature_cols = [c for c in EXPECTED_COLS if c != "target"]
    medians = df[feature_cols].median(numeric_only=True)
    df[feature_cols] = df[feature_cols].fillna(medians)

    out_dir = str(pd.Path(output_csv_path).parent) if hasattr(pd, "Path") else None
    # Safe: ensure parent exists if provided like "data/heart_merged_clean.csv"
    # If user gives "heart_merged_clean.csv" parent is "."
    import os
    ensure_dir(os.path.dirname(output_csv_path) or ".")

    df.to_csv(output_csv_path, index=False)
    print(f"[OK] Saved: {output_csv_path} | shape={df.shape}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Raw merged CSV path")
    ap.add_argument("--output", required=True, help="Clean output CSV path")
    args = ap.parse_args()
    preprocess_heart_csv(args.input, args.output)


if __name__ == "__main__":
    main()
