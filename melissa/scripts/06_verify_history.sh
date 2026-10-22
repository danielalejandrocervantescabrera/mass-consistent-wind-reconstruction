#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
"$PYTHON" "$ROOT/scripts/validate_raw.py" --raw-dir "$RAW"
"$PYTHON" "$ROOT/scripts/verify_historical_checkpoints.py"
