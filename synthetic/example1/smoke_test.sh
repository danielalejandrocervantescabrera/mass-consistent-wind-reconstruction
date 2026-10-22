#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
"$PYTHON" "$HERE/deepWind3D_vortex_autograd_fixed_observations_lhs.py" \
 --width 10 --depth 1 --nEpoch 1 --nBatch 1 \
 --innerBatchSize 16 --bdyBatchSize 8 --nObs 32 \
 --nVal 16 --nTest 16 --valEvery 1 --seed 0 --device cpu \
 --outdir "$HERE/outputs/smoke" --tag smoke --noPlots
