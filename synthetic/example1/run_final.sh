#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cpu}"
OUTDIR="${OUTDIR:-$HERE/outputs/final_n1024}"
mkdir -p "$OUTDIR"
for SEED in 0 1 2; do
 TAG="final_n1024_seed${SEED}"
 if [[ -s "$OUTDIR/summary/${TAG}_eval_summary.csv" && "${FORCE:-0}" != 1 ]]; then
  echo "Skipping existing $TAG"
  continue
 fi
 "$PYTHON" "$HERE/deepWind3D_vortex_autograd_fixed_observations_lhs.py" \
  --width 60 --depth 2 --nEpoch 2000 --nBatch 25 \
  --innerBatchSize 512 --bdyBatchSize 64 --nObs 1024 \
  --obsSampling lhs --obsSeed 12345 --nVal 5000 \
  --nTest 10000 --evalSeed 54321 \
  --penalty 1 --penalty2 1 --epsilon 0.1 \
  --lr 0.01 --decay 1e-6 --step_size 200 --gamma 0.5 \
  --valEvery 10 --logEvery 30 --seed "$SEED" --device "$DEVICE" \
  --outdir "$OUTDIR" --tag "$TAG" --noPlots
 done
