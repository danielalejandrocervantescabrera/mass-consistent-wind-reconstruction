#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
"$PYTHON" "$ROOT/scripts/verify_outputs.py" "$@"
