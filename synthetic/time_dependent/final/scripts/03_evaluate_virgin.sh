#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${TEMPORAL_CKPT_SET:-historical}"
[[ "$MODE" == historical || "$MODE" == new ]] || { echo 'Use TEMPORAL_CKPT_SET=historical|new' >&2; exit 2; }
WORK="$DIR/outputs/evaluation_${MODE}"
mkdir -p "$WORK/out_time_final_16384/checkpoints" "$WORK/out_time_obs_sweep/checkpoints"
ln -sfn "$DIR/deepWind3D_vortex_time_final_protocol.py" "$WORK/deepWind3D_vortex_time_final_protocol.py"
if [[ "$MODE" == historical ]]; then
    declare -a SRC=(
      "$DIR/out_time_obs_sweep/checkpoints/timeobs_n16384_s0_best.pt"
      "$DIR/out_time_final_16384/checkpoints/timefinal16384_s1_best.pt"
      "$DIR/out_time_final_16384/checkpoints/timefinal16384_s2_best.pt"
    )
    declare -a DST=(
      "$WORK/out_time_obs_sweep/checkpoints/timeobs_n16384_s0_best.pt"
      "$WORK/out_time_final_16384/checkpoints/timefinal16384_s1_best.pt"
      "$WORK/out_time_final_16384/checkpoints/timefinal16384_s2_best.pt"
    )
else
    declare -a SRC=(
      "$DIR/outputs/new_training/out_time_final_16384/checkpoints/timefinal16384_s0_best.pt"
      "$DIR/outputs/new_training/out_time_final_16384/checkpoints/timefinal16384_s1_best.pt"
      "$DIR/outputs/new_training/out_time_final_16384/checkpoints/timefinal16384_s2_best.pt"
    )
    declare -a DST=(
      "$WORK/out_time_final_16384/checkpoints/timefinal16384_s0_best.pt"
      "$WORK/out_time_final_16384/checkpoints/timefinal16384_s1_best.pt"
      "$WORK/out_time_final_16384/checkpoints/timefinal16384_s2_best.pt"
    )
fi
for i in 0 1 2; do
    [[ -s "${SRC[$i]}" ]] || { echo "Missing checkpoint ${SRC[$i]}" >&2; exit 2; }
    ln -sfn "${SRC[$i]}" "${DST[$i]}"
done
cd "$WORK"
"${PYTHON_BIN:-python3}" "$DIR/eval_time_final_virgin.py"
echo "Evaluation outputs: $WORK"
