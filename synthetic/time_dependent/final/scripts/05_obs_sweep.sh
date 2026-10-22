#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"
export CODE="$DIR/deepWind3D_vortex_time_final_protocol.py"
export OUTDIR="$DIR/out_time_obs_sweep"
export LOGDIR="$DIR/time_obs_sweep_logs"
export BASELINE_DIR="$DIR/out_time_final_protocol"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
bash "$DIR/run_time_obs_sweep.sh"
