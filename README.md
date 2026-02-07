# Heart Disease Benchmark — AutoML vs ML vs DL

## Time-Budgeted & Reproducible Benchmark Framework

---

## Overview

This project provides a **fully reproducible, time-budgeted benchmark** comparing:

* AutoML frameworks (MLJAR, AutoGluon, H2O, FLAML)
* Classical Machine Learning models
* Deep Learning architectures (CNN, RNN, GRU, LSTM, AutoEncoder)

for **binary heart disease detection** using a merged clinical dataset derived from multiple UCI Heart Disease sources.

The goal is to evaluate **performance vs computational cost** under controlled runtime budgets.

---

## Dataset

### Raw dataset

```
cleveland_hungarian_long-beach-va_switzerland.csv
```

Merged from:

* Cleveland
* Hungarian
* Long Beach VA
* Switzerland heart disease datasets

---

### Preprocessed dataset

```
heart_merged_clean.csv
```

Generated via deterministic preprocessing.

---

## Preprocessing Pipeline

The preprocessing is **fully deterministic and reproducible**.

Steps:

1. Convert missing placeholders (`?`, `??`, etc.) → NaN
2. Strip column names
3. Cast all columns to numeric
4. Validate expected schema
5. Binarize target

   ```
   0 → 0 (no disease)
   1–4 → 1 (disease)
   ```
6. Remove rows with missing target
7. Impute feature NaNs using column medians
8. Save cleaned dataset

Run:

```python
df_clean = preprocess_heart_csv(
    "cleveland_hungarian_long-beach-va_switzerland.csv",
    "heart_merged_clean.csv"
)
```

---

## Dataset Visualization

Two figure sets are automatically generated.

### Raw dataset figures

Directory:

```
figures_raw_900/
```

Includes:

* Missing value distribution
* Target distribution
* Feature completeness heatmap
* Boxplots
* Correlation matrix

---

### Clean dataset figures

Directory:

```
figures_900/
```

Includes:

* Class balance
* Feature distributions
* Correlation heatmap
* Scatter plots
* Workflow diagram

Generate figures:

```python
make_peerj_figures_raw_900(...)
make_peerj_figures_900(...)
make_fig3_workflow(...)
```

All figures are exported as **900×900 PNG**, suitable for journal submission.

---

## Experimental Protocol

Each framework/model is evaluated under two runtime regimes:

### Light regime

```
60 seconds total budget
```

Focus:

* fast convergence
* minimal tuning
* lightweight configurations

---

### Full regime

```
180 seconds total budget
```

Allows:

* broader hyperparameter search
* stacking/bagging (AutoML)
* deeper DL training

---

## Train/Test Protocol

Strict reproducibility:

```
Outer split: Train/Test (80/20, stratified, seed=42)
Inner split: Train/Validation for tuning
```

No test leakage is allowed.

---

## AutoML Benchmarks

Frameworks:

* MLJAR AutoML
* AutoGluon Tabular
* H2O AutoML
* FLAML

Each framework runs in:

```
Light mode → 60 seconds
Full mode → 180 seconds
```

Metrics:

* Accuracy
* AUC
* F1 score
* Precision
* Recall
* LogLoss
* Confusion matrix

Resource logging:

* Runtime
* Python memory usage
* System RAM delta
* Serialized model size

---

## Classical ML Benchmark

Models include:

* Logistic Regression
* Random Forest
* Extra Trees
* MLP
* XGBoost *(optional)*
* LightGBM *(optional)*
* CatBoost *(optional)*

Timed hyperparameter search is applied only in **full mode**.

---

## Deep Learning Benchmark

Architectures:

* 1D CNN
* RNN
* GRU
* LSTM
* AutoEncoder + classifier

Training features:

* early stopping
* validation-based selection
* strict time budgeting
* deterministic seeds

---

## Reproducibility Guarantees

✔ Fixed seeds (42)
✔ Deterministic preprocessing
✔ Explicit train/validation/test splits
✔ Time-budget controlled training
✔ Saved models
✔ Resource logging
✔ Automatic figure generation

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

To regenerate requirements:

```bash
pip freeze > requirements.txt
```

Key libraries:

```
pandas
numpy
scikit-learn
mljar-supervised
autogluon
h2o
flaml
torch
psutil
matplotlib
```

---

## Running the Benchmark

1. Preprocess dataset
2. Generate figures
3. Run AutoML benchmarks
4. Run ML benchmarks
5. Run DL benchmarks

Each stage produces saved models and metrics logs.

---

## Output Artifacts

* Clean dataset CSV
* Saved models (.pkl / .pt)
* Performance metrics
* Leaderboards
* Publication-ready figures

---

## Intended Use

This benchmark is designed for:

* AutoML vs ML vs DL comparisons
* reproducibility studies
* time-constrained model evaluation
* academic research

---

## Citation

If used in research, please cite:

**GitHub:**
[https://github.com/ersineneseryilmaz/heart-merged-automl-dl-ml-benchmark](https://github.com/ersineneseryilmaz/heart-merged-automl-dl-ml-benchmark)

**Zenodo DOI:**
10.5281/zenodo.18283625

---

## License

Academic/research use. See repository license for details.

---
