#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
mkdir -p "$HERE/outputs"
"$PYTHON" "$HERE/test_sasaki_svd_conditioning.py" --code-dir "$HERE" \
  --example 2 --nCenters2 1000 --ep2 1.0 \
  --rconds 1e-10 1e-12 1e-14 \
  --output "$HERE/outputs/sasaki_svd_conditioning.csv"
