\
"""Example orchestration script for the revised benchmark.

This file is intentionally lightweight. Heavy framework runs should usually be
executed step by step in notebooks or separate jobs.
"""

from __future__ import annotations

from pathlib import Path

from .automl_autogluon import run_autogluon
from .automl_flaml import run_flaml
from .automl_h2o import run_h2o
from .automl_mljar import run_mljar
from .classic_ml import run_classic_ml_full, run_classic_ml_light
from .config import RAW_DATASET
from .preprocess import preprocess_heart_csv
from .workflow_fig import make_workflow_figure


def main(raw_csv: str = RAW_DATASET, output_dir: str = "outputs") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    clean = preprocess_heart_csv(raw_csv, out / "heart_merged_clean.csv")

    run_classic_ml_light(clean, out / "classic_ml_light", seed=42)
    run_classic_ml_full(clean, out / "classic_ml_full", seed=42)

    # AutoML examples. Uncomment frameworks installed in your environment.
    # run_autogluon(clean, out / "autogluon", regime="Light", seed=42)
    # run_flaml(clean, out / "flaml", regime="Light", seed=42)
    # run_h2o(clean, out / "h2o", regime="Light", seed=42)
    # run_mljar(clean, out / "mljar", regime="Light", seed=42)

    make_workflow_figure(out / "figure_12_revised_workflow.png")


if __name__ == "__main__":
    main()
