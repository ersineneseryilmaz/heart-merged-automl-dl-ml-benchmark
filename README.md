# heart-merged-automl-dl-ml-benchmark

Heart disease benchmark on the merged UCI datasets (Cleveland, Hungary, Switzerland, VA Long Beach) comparing **Classic ML**, **Deep Learning (PyTorch)**, and **AutoML frameworks** (AutoGluon, H2O, MLJAR, FLAML) under **Full vs Lightweight**, time-budgeted, reproducible settings with metric & resource logging.

## Dataset
This repository expects a merged CSV named:

- `cleveland_hungarian_long-beach-va_switzerland.csv`

The dataset file is **not** committed to the repository (see `.gitignore`). Place it in the project root (or in `data/` if you prefer, then update paths accordingly).

### Target definition
The original `num`/`target` in UCI can be multi-class (0–4). In this project it is binarized as:
- `0 → 0` (no disease)
- `1–4 → 1` (disease)

## What is measured
For each run, we report:
- **Performance**: Accuracy, AUC, F1, Precision, Recall, LogLoss, Confusion Matrix
- **Resources**: wall-clock runtime, process RSS (MB), system RAM delta (GB), serialized model size (MB)

## Project structure (recommended)
├─ src/
│ ├─ preprocess.py
│ ├─ figures_raw.py
│ ├─ figures_clean.py
│ ├─ workflow_fig.py
│ ├─ classic_ml.py
│ ├─ dl_models.py
│ ├─ automl_mljar.py
│ ├─ automl_autogluon.py
│ ├─ automl_h2o.py
│ └─ automl_flaml.py
├─ scripts/
│ ├─ run_preprocess.sh
│ ├─ run_figures.sh
│ ├─ run_classic_ml.sh
│ ├─ run_dl.sh
│ └─ run_automl.sh
├─ requirements.txt
└─ README.md

> We will create these files incrementally from your Colab notebook code.

## Environment
- OS / CPU / RAM: see paper’s **Table 1**
- Key libraries (from your environment):
  - numpy==2.0.2, pandas==2.2.2, scikit-learn==1.6.1, lightgbm==4.6.0, xgboost==3.1.2

## Quickstart
1) Install dependencies:
```bash
pip install -r requirements.txt

2) Preprocess (creates heart_merged_clean.csv):
python -m src.preprocess \
  --input cleveland_hungarian_long-beach-va_switzerland.csv \
  --output heart_merged_clean.csv

3) Generate figures (900x900 px):  
python -m src.figures_raw  --input cleveland_hungarian_long-beach-va_switzerland.csv --out figures_raw_900
python -m src.figures_clean --input heart_merged_clean.csv --out figures_900
python -m src.workflow_fig  --out figures_900

4) Run benchmarks:
python -m src.classic_ml --data heart_merged_clean.csv --full_budget 180 --light_budget 60
python -m src.dl_models  --data heart_merged_clean.csv --full_budget 180 --light_budget 60
# AutoML:
python -m src.automl_flaml --data heart_merged_clean.csv --mode full  --budget 180 --cv 1
python -m src.automl_flaml --data heart_merged_clean.csv --mode light --budget 60  --cv 0

Notes on reproducibility

Fixed random seed (seed=42) is used consistently across splits and model training.

Outer train/test split is stratified; an inner validation split is used for selection/early stopping without touching the test set.

Full vs Lightweight are defined by explicit budgets and constrained search/model pools (see paper’s Table 2).

License
MIT License
