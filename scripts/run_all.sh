#!/usr/bin/env bash
set -e

RAW="cleveland_hungarian_long-beach-va_switzerland.csv"
CLEAN="heart_merged_clean.csv"

# 0) preprocess + figures
python -m src.preprocess --input "$RAW" --output "$CLEAN"
python -m src.figures_raw --input "$RAW" --out figures_raw_900
python -m src.figures_clean --input "$CLEAN" --out figures_900
python -m src.workflow_fig --out figures_900

# 1) Classic ML
python -m src.classic_ml --data "$CLEAN" --mode light --budget 60  --out outputs/classic_ml
python -m src.classic_ml --data "$CLEAN" --mode full  --budget 180 --out outputs/classic_ml

# 2) DL (run all models)
for M in CNN RNN GRU LSTM AE; do
  python -m src.dl_models --data "$CLEAN" --model "$M" --mode light --budget 60  --out outputs/dl
  python -m src.dl_models --data "$CLEAN" --model "$M" --mode full  --budget 180 --out outputs/dl
done

# 3) AutoML stacks
# FLAML: full=CV, light=holdout (already implemented)
python -m src.automl_flaml --data "$CLEAN" --mode light --budget 60  --out outputs/flaml_light
python -m src.automl_flaml --data "$CLEAN" --mode full  --budget 180 --out outputs/flaml_full

# MLJAR
python -m src.automl_mljar --data "$CLEAN" --mode light --budget 60  --out outputs/mljar
python -m src.automl_mljar --data "$CLEAN" --mode full  --budget 180 --out outputs/mljar

# AutoGluon
python -m src.automl_autogluon --data "$CLEAN" --mode light --budget 60  --out outputs/autogluon
python -m src.automl_autogluon --data "$CLEAN" --mode full  --budget 180 --out outputs/autogluon

# H2O
python -m src.automl_h2o --data "$CLEAN" --mode light --budget 60  --out outputs/h2o
python -m src.automl_h2o --data "$CLEAN" --mode full  --budget 180 --out outputs/h2o

echo "[OK] All done."

chmod +x scripts/run_all.sh

git add .gitignore README.md requirements.txt src scripts
git commit -m "Add full benchmark: Classic ML, DL, AutoML (FLAML/MLJAR/AutoGluon/H2O) with Light vs Full budgets"
git push

