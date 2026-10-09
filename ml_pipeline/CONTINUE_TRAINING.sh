#!/usr/bin/env bash
# CONTINUE TRAINING — single command to resume the food classifier work.
# 1) Download + index the HF datasets (fresh_rotten_fruit_classification + snacks)
# 2) Train EfficientNet-Lite0 on the resulting index.csv (resumeable, interruptible)
#
# Interruptibility / resume contract (same as ml_cv_train_food.py docs):
#   - Ctrl-C / closing this chat kills the foreground Python process only.
#     Nothing is detached, so there is nothing to "stop" on the machine.
#   - Re-running this same command continues from annapurna/ml_pipeline/runs2/food_efficientnet/last.pt
#     exactly where it left off (model + EMA + optimizer + scheduler + AMP scaler + RNG).
#
# GPU health is wired inside ml_cv_train_food.py (VRAM-fitting batch size, OOM retry,
# tf32/bf16 autocast, --max-temp / --max-minutes caps).
#
# Food-label mapping (edit here if you want a different default):
#   fresh -> GOOD, rotten -> REJECTED  (fresh_rotten_fruit_classification)
#   snacks -> NOT_FOOD by default (--snack-label GOOD to treat snacks as acceptable food)
#   MM-Food-100K is NOT included here: it needs your kaggle.json. Add it later with:
#     python ml_cv_prepare_food_data.py --kaggle-dataset jaisal228/MM-Food-100K --kaggle-map km.json
#
set -euo pipefail

ML=${ML_PIPELINE_DIR:-.}
cd "$ML"

echo "=== STEP 1: prepare food dataset (download + index) ==="
python ml_cv_prepare_food_data.py \
  --hf-config augmented \
  --snack-label NOT_FOOD \
  --max-workers 8 \
  --test-source snacks \
  --val-frac 0.15

if [ ! -f data/cv/index.csv ]; then
  echo "ERROR: data/cv/index.csv was not produced -- aborting before training"
  exit 2
fi

echo
echo "=== STEP 2: train food classifier ==="
python ml_cv_train_food.py \
  --index data/cv/index.csv \
  --out runs2/food_efficientnet \
  --arch tf_efficientnet_lite0 \
  --size 224 \
  --epochs 30 \
  --head-epochs 3 \
  --lr 3e-4 \
  --wd 0.05 \
  --smoothing 0.05 \
  --mix-prob 0.5 \
  --aug auto \
  --ema-decay 0.999 \
  --patience 10 \
  --log-every 100 \
  --health-every 200 \
  --device cuda

echo
echo "done. run artifacts -> $ML/runs/food_efficientnet"
echo "to resume later:  $ML/CONTINUE_TRAINING.sh"
echo "or explicitly:   python ml_cv_train_food.py --index data/cv/index.csv --out runs/food_efficientnet --resume"
