#!/usr/bin/env bash
# Demonstration of the original script's example configuration.
# NOT the fixed-LHS three-seed protocol of the published tables.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPOCHS="${EPOCHS:-500}"
BATCHES="${BATCHES:-256}"
LR="${LR:-0.001}"
python3 "$ROOT/src/time_dependent_vortex_reference.py" \
 --traj parabolic --x0 0 --y0 0 --vx 20 --vy 0 --ax 0 --ay 30 \
 --a -25 --b 25 --c -25 --ddd 25 --e 0 --f 7 \
 --t0 -1 --t1 1 --epsilon 0.02 \
 --eval_times -1 -0.8 -0.5 -0.2 0 0.2 0.5 0.8 1 \
 --width 80 --depth 5 \
 --nEpoch "$EPOCHS" --nBatch "$BATCHES" --lr "$LR" \
 --outdir "$ROOT/outputs/reference_run" --tag demo
