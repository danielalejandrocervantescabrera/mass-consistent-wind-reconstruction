#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
if [[ -f "$PREPARED/manifest.json" && -f "$PREPARED/train_X.npy" && -f "$PREPARED/train_Yuv.npy" && "${FORCE:-0}" != 1 ]]; then
  echo 'Prepared data already exist; use FORCE=1 to regenerate.'; exit 0
fi
"$PYTHON" "$ROOT/src/prepare_melissa_hourly.py" \
 --input_dir "$RAW" --centers "$RAW/hourly_eye_centers.csv" \
 --outdir "$PREPARED" --dev_key '' --val_key '' --test_key '' \
 --nx 40 --ny 40 --dz 20 --dt_minutes 5 --holdout_gap_minutes 0 \
 --wahba_lambda 0.001 --max_height 300
"$PYTHON" "$ROOT/scripts/verify_outputs.py" --prepared-only
