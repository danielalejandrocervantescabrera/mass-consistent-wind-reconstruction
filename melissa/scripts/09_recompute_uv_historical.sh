#!/usr/bin/env bash
# Independently recompute in-sample horizontal fit for all 61 assimilated hours.
# Requires the processed data prepared by 07_recompute_historical.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
"${PYTHON:-python3}" melissa/scripts/evaluate_melissa_uv_historico.py --repo-root "$ROOT" "$@"
