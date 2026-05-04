# Heart Disease Benchmark — Time-Aware AutoML vs ML vs DL

## Reproducible full/lightweight benchmark framework for heart disease detection

This repository provides a reproducible, time-aware benchmarking framework comparing:

- Classical machine learning (ML)
- Deep learning (DL)
- AutoML frameworks: AutoGluon, FLAML, H2O AutoML, and MLJAR

for heart disease detection using a merged UCI Heart Disease dataset derived from:

- Cleveland
- Hungary
- Switzerland
- VA Long Beach

The benchmark is designed to evaluate not only predictive performance, but also deployment-oriented behavior, including observed wall-clock runtime, memory usage, serialized model size, budget compliance, and robustness across repeated data partitions.

---

## Main purpose

The project supports a revised experimental protocol for the manuscript:

**A time-aware reproducible benchmark of full and lightweight AutoML, machine learning, and deep learning frameworks for heart disease detection**

The benchmark addresses the following methodological questions:

1. How do classical ML, DL, and AutoML frameworks perform under comparable nominal time-budget regimes?
2. Do nominal framework-level time limits translate into actual end-to-end wall-clock compliance?
3. How stable are lightweight benchmark results across repeated stratified holdout splits?
4. Are observed differences statistically supported or only descriptive trends?
5. How does target formulation affect conclusions when the original UCI target is simplified from five levels to binary disease absence/presence?

---

## Dataset

### Raw merged dataset

```text
cleveland_hungarian_long-beach-va_switzerland.csv
```

The raw dataset is constructed by merging four UCI Heart Disease sources:

```text
Cleveland + Hungary + Switzerland + VA Long Beach
```

The benchmark uses the commonly adopted 14-variable schema:

- 13 predictors
- 1 outcome variable

The original target variable is `num`, with values from 0 to 4.

---

## Target formulations

### Primary binary formulation

The primary benchmark uses a binary disease absence/presence target:

```text
0   -> 0  no disease
1-4 -> 1  disease present
```

This follows common UCI Heart Disease benchmark practice, where target values 1-4 are treated as disease presence and 0 as disease absence.

### Target sensitivity formulations

To evaluate the impact of target simplification, the repository also supports:

```text
Binary formulation:
0 vs. 1-4

Ordinal three-class formulation:
0 vs. 1-2 vs. 3-4

Original five-class formulation:
0 vs. 1 vs. 2 vs. 3 vs. 4
```

The target sensitivity analysis is reported as mean ± standard deviation across repeated stratified holdout splits.

---

## Preprocessing pipeline

The preprocessing pipeline is deterministic and shared across model families.

Main steps:

1. Convert missing placeholders such as `?` to `NaN`
2. Strip and normalize column names
3. Cast all selected variables to numeric values
4. Remove rows with missing target labels
5. Create the primary binary target
6. Preserve alternative target formulations for sensitivity analysis
7. Apply median imputation to features within split-based evaluation
8. Apply feature scaling only when required by the model family

Important leakage-control rule:

```text
Any data-dependent preprocessing step, including imputation and scaling, must be fitted on the training partition only and then applied to validation/test partitions.
```

Tree-based models use the numeric-coded feature representation. Scale-sensitive models, including linear and neural models, use scaling fitted only on the training partition.

---

## Exploratory figures

The project generates exploratory figures for both raw and cleaned data.

### Raw dataset diagnostics

Typical raw-data figures include:

- Missingness density
- Non-null counts
- Original target distribution
- Row-wise completeness patterns
- Raw numeric feature distributions
- Raw correlation matrix

### Clean dataset diagnostics

Typical clean-data figures include:

- Binary class balance
- Feature distributions stratified by class
- Correlation heatmap
- Clinically interpretable scatter plots
- End-to-end workflow diagram

The revised workflow figure includes:

1. Raw merged UCI Heart Disease dataset
2. Unified preprocessing
3. Primary external evaluation protocol
4. Primary benchmark training
5. Held-out test evaluation
6. Resource logging
7. Repeated lightweight holdout analysis
8. Budget compliance, statistical testing, and target sensitivity
9. Final reporting and reproducibility materials

---

## Experimental protocol

The framework uses two nominal framework-level time-budget regimes:

```text
Full regime:        180 seconds
Lightweight regime: 60 seconds
```

These are **nominal framework-level time-limit settings**, not operating-system-level hard timeouts.

Each framework is configured through its native time-limit, runtime, or budget parameter where available. Because AutoML tools differ in internal scheduling, validation, ensembling, logging, checkpointing, and artifact generation, observed end-to-end runtime may exceed the nominal time-limit parameter.

Therefore, the benchmark reports both:

```text
nominal time budget
observed end-to-end wall-clock runtime
```

---

## Primary train/test protocol

The primary benchmark uses:

```text
Outer split: Train/Test, 80/20, stratified, seed=42
Inner split: Train/Validation for model selection and early stopping
```

The held-out test set is used only for final reporting.

No test data are used during model selection, tuning, early stopping, imputation fitting, or scaling fitting.

---

## Repeated lightweight holdout protocol

To reduce dependence on a single train/test split, the lightweight framework-level comparison is repeated over five stratified 80/20 holdout splits.

The pre-specified seeds are:

```text
[42, 7, 21, 84, 123]
```

The same seeds are used for all lightweight frameworks, enabling paired comparisons across identical split partitions.

Reported repeated-holdout values are:

```text
mean ± standard deviation
```

---

## Budget compliance analysis

Budget compliance is evaluated for repeated lightweight runs.

A run is considered compliant if:

```text
observed end-to-end runtime <= 60 seconds
```

The budget compliance summary includes:

- Number of runs
- Number of compliant runs
- Mean runtime
- Runtime standard deviation
- Mean overhead
- Maximum overhead
- Runtime-to-budget ratio
- Compliance rate

This analysis distinguishes nominal framework-level time-limit settings from actual end-to-end runtime behavior.

---

## Statistical testing

Statistical testing is applied to repeated lightweight results.

### Friedman omnibus tests

Friedman tests assess whether there are overall differences among lightweight frameworks across repeated splits.

### Pairwise Wilcoxon-Holm tests

Pairwise Wilcoxon signed-rank tests compare frameworks on identical split-level results.

Holm correction is applied within each metric family to control for multiple comparisons.

The tests are interpreted as robustness indicators because the number of repeated splits is limited.

They should not be interpreted as definitive evidence of clinical superiority or equivalence.

---

## Modeling families

### Classical ML

Classical ML models include:

- Logistic Regression
- Random Forest
- Extra Trees
- LightGBM
- XGBoost

The lightweight regime uses fixed fast configurations.

The full regime uses time-limited hyperparameter search where applicable.

### Deep learning

DL models include:

- 1D-CNN
- RNN
- GRU
- LSTM
- AutoEncoder + classification head

DL models use validation-based early stopping.

The benchmark evaluates DL as a short-budget tabular modeling family rather than as an exhaustive architecture search.

### AutoML frameworks

AutoML frameworks include:

- AutoGluon
- FLAML
- H2O AutoML
- MLJAR

Each framework is evaluated in both lightweight and full regimes.

Framework-specific settings, presets, enabled/disabled components, validation mechanisms, and search constraints should be reported in the configuration tables and code.

---

## Metrics

Predictive metrics include:

- Accuracy
- ROC-AUC
- F1-score
- Precision
- Recall
- LogLoss
- Confusion matrix

Deployment-oriented and resource metrics include:

- Observed wall-clock runtime
- Python process memory usage
- System-level memory usage
- RAM delta
- Serialized model artifact size

---

## Key output files and tables

The revised manuscript uses the following supplementary table structure.

```text
Table S1   Execution environment and software versions
Table S2   Framework-specific configurations and search constraints
Table S3   Metric and resource-measurement definitions
Table S4   Main classical ML benchmark results
Table S5   Main deep learning benchmark results
Table S6   Full AutoML benchmark results
Table S7   Lightweight AutoML benchmark results
Table S8   Single-split best full vs. best lightweight summary
Table S9   Classical ML Light
Table S10  Classical ML Full
Table S11  Repeated stratified holdout summary for lightweight configurations
Table S12  Budget compliance summary for repeated lightweight runs
Table S13  Pairwise Wilcoxon signed-rank tests with Holm correction
Table S14  Friedman omnibus tests across repeated lightweight runs
Table S15  Class distribution under alternative target formulations
Table S16  Target sensitivity analysis across binary, ordinal three-class, and original five-class formulations
Table S17  Best-performing model within each target formulation
```

Raw CSV outputs may include:

```text
repeated_holdout_light_results.csv
repeated_holdout_light_summary.csv
repeated_holdout_light_summary_formatted.csv
repeated_holdout_light_budget_compliance.csv
statistical_tests_wilcoxon_holm.csv
statistical_tests_friedman.csv
target_sensitivity_class_distribution.csv
target_sensitivity_results.csv
target_sensitivity_summary.csv
target_sensitivity_summary_formatted.csv
target_sensitivity_best_by_formulation.csv
target_sensitivity_confusion_matrices_long.csv
```

---

## Reproducibility materials

The repository is intended to include:

- Raw merged dataset or instructions for obtaining it
- Preprocessing scripts
- Benchmark notebooks or scripts
- Framework configuration files
- Saved output CSV files
- Generated figures
- Supplementary tables
- Codebook
- Requirements file
- README file

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Core libraries include:

```text
pandas
numpy
scikit-learn
lightgbm
xgboost
autogluon
flaml
h2o
mljar-supervised
torch
psutil
matplotlib
scipy
```

Exact package versions should be documented in the execution-environment table and `requirements.txt`.

---

## Suggested execution order

A typical reproduction workflow is:

1. Prepare the raw merged UCI Heart Disease CSV
2. Run deterministic preprocessing and data diagnostics
3. Generate exploratory figures
4. Run primary single-split classical ML, DL, and AutoML benchmarks
5. Run repeated lightweight holdout experiments
6. Run budget compliance analysis
7. Run Wilcoxon-Holm and Friedman statistical tests
8. Run target sensitivity analysis
9. Export final tables and figures
10. Compare outputs against the manuscript results

---

## Interpretation notes

This benchmark is an engineering-oriented comparison of full and lightweight ML, DL, and AutoML behavior under short nominal time-budget settings.

Important cautions:

- The 60-second and 180-second regimes are nominal framework-level settings, not operating-system-level hard timeouts.
- Observed runtime should be interpreted alongside predictive performance.
- Repeated holdout uses five pre-specified seeds and should be interpreted as a robustness check, not a full external validation study.
- Statistical tests are robustness indicators because the number of paired splits is limited.
- The binary target formulation is a pragmatic disease absence/presence benchmark and does not preserve the full ordinal disease-severity structure.
- Results do not establish clinical effectiveness, clinical utility, or external validity.

---

## Intended use

This repository is intended for:

- Reproducible AutoML benchmarking
- Time-aware ML/DL/AutoML comparison
- Lightweight AutoML evaluation
- Deployment-oriented performance/resource analysis
- Sensitivity analysis of target formulation
- Academic research and manuscript reproduction

It is not intended for direct clinical deployment without independent validation, calibration, clinical utility assessment, and prospective evaluation.

---

## Citation

If you use this repository, please cite the associated Zenodo archive and manuscript.

**Zenodo DOI**

```text
https://doi.org/10.5281/zenodo.18514862
```

**GitHub repository**

```text
https://github.com/ersineneseryilmaz/heart-merged-automl-dl-ml-benchmark
```

---

## License

Academic and research use. See the repository license file for details.
