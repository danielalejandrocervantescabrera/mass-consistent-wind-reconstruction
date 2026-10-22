#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
"$PYTHON" "$HERE/sasaki_example2_corrected.py" --nCenters 1000 --ep 1.0 \
  --nTest 10000 --nBottomEval 5000 --evalSeed 54321 \
  --outdir "$HERE/outputs/example2"
