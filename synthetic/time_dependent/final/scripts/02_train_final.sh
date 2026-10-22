#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
# Fresh training is isolated; historical original weights and CSVs are read-only references.
export OUTDIR="$DIR/outputs/new_training/out_time_final_16384"
export SWEEP_DIR="$DIR/outputs/new_training/no_previous_sweep"
export LOGDIR="$DIR/outputs/new_training/logs"
export CODE="$DIR/deepWind3D_vortex_time_final_protocol.py"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "$DIR"
bash "$DIR/run_time_final_16384.sh"
