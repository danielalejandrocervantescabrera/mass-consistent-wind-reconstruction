#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${TEMPORAL_CKPT_SET:-historical}"
WORK="$DIR/outputs/evaluation_${MODE}"
# Prepare the isolated checkpoint links if needed without evaluating all points.
if [[ ! -L "$WORK/deepWind3D_vortex_time_final_protocol.py" ]]; then
    echo "First run scripts/03_evaluate_virgin.sh (or scripts/09_stage_checkpoints.sh)." >&2
    exit 2
fi
mkdir -p "$WORK/figures" "$WORK/gnuplot"
cd "$WORK"
"${PYTHON_BIN:-python3}" "$DIR/plot_time_final_16384.py" --outbase "$WORK/figures/temporal_parabolic"
"${PYTHON_BIN:-python3}" "$DIR/export_time_final_16384_for_gnuplot.py" --outdir "$WORK/gnuplot"
echo "Figures: $WORK/figures"
