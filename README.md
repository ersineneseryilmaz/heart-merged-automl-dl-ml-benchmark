# heart-merged-automl-dl-ml-benchmark
Heart disease benchmark on the merged UCI datasets (Cleveland, Hungary, Switzerland, VA Long Beach) comparing Classic ML, Deep Learning (PyTorch), and AutoML frameworks (AutoGluon, H2O, MLJAR, FLAML) under full vs lightweight, time-budgeted, reproducible settings with metric &amp; resource logging.
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

