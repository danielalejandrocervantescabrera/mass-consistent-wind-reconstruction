#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
"${PYTHON_BIN:-python3}" "$DIR/scripts/check_historical_results.py"
