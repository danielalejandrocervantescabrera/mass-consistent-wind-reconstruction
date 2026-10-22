#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python3 "$ROOT/src/time_dependent_vortex_reference.py" \
 --traj parabolic --x0 0 --y0 0 --vx 20 --vy 0 --ax 0 --ay 30 \
 --a -25 --b 25 --c -25 --ddd 25 --e 0 --f 7 \
 --t0 -1 --t1 1 --epsilon 0.02 \
 --width 8 --depth 1 \
 --nEpoch 1 --nBatch 1 --innerBatchSize 16 --bdyBatchSize 8 \
 --numQuad 3 --eval_times -1 0 1 \
 --outdir "$ROOT/outputs/smoke" --tag smoke
