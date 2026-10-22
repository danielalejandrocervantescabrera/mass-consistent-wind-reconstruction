#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
[[ -f "$PREPARED/train_X.npy" && -f "$PREPARED/train_Yuv.npy" ]] || { echo 'Run scripts/01_prepare.sh first' >&2; exit 1; }
for SEED in ${SEEDS:-1234 2345 3456}; do
  OUT="$MODEL_ROOT/seed_${SEED}"
  if [[ -f "$OUT/final_model.pt" && "${FORCE:-0}" != 1 ]]; then echo "Skipping existing $OUT"; continue; fi
  mkdir -p "$OUT"
  "$PYTHON" "$ROOT/src/train_melissa_reconstruction_final.py" \
     --data_dir "$PREPARED" --outdir "$OUT" --seed "$SEED" \
     --width 80 --depth 5 --nEpoch 2500 --nBatch 25 \
     --dataBatchSize 512 --divBatchSize 2048 --bottomBatchSize 1024 \
     --evalDataSize 100000 --evalDivSize 8192 --evalBdySize 8192 \
     --lr 0.01 --weight_decay 0.000001 --step_size 200 --gamma 0.5 \
     --grad_clip 1.0 --beta_div 0.03 --beta_bdy 1.0 \
     --real_fraction 0.5 --alpha_interp 0.5 --eval_every 20 --log_every 20 \
     2>&1 | tee "$OUT/training.log"
done
