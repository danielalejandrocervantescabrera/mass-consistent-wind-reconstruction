#!/usr/bin/env bash
# Generate paper figures from the historical models and recomputed diagnostics.
source "$(dirname "$0")/common.sh"
export MODEL_ROOT="$ROOT/reference/checkpoints"
export ANALYSIS_ROOT="$ROOT/outputs/historical_analysis"
export FIGROOT="$ROOT/outputs/historical_figures"
[[ -f "$ANALYSIS_ROOT/seed_stability/w_fields_all_real_times.npz" ]] || {
 echo 'First run: bash melissa/scripts/07_recompute_historical.sh' >&2;exit 1;
}
bash "$ROOT/scripts/04_figures.sh"
