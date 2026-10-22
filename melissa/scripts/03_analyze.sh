#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
for SEED in 1234 2345 3456; do
  for name in final_model.pt summary.json; do
    if [[ ! -f "$MODEL_ROOT/seed_${SEED}/$name" ]]; then
      echo "Missing $MODEL_ROOT/seed_${SEED}/$name" >&2
      echo 'Run scripts/02_train.sh or copy matched historical weights and summary.' >&2; exit 1
    fi
  done
done
"$PYTHON" "$ROOT/src/analyze_melissa_vertical_velocity.py" \
  --data_dir "$PREPARED" --model_dir "$MODEL_ROOT/seed_2345" \
  --outdir "$ANALYSIS_ROOT/vertical_seed_2345" --model_file final_model.pt
"$PYTHON" "$ROOT/src/compare_melissa_w_across_seeds.py" \
  --data_dir "$PREPARED" --model_root "$MODEL_ROOT" \
  --outdir "$ANALYSIS_ROOT/seed_stability" \
  --seeds 1234,2345,3456 --width 80 --depth 5
"$PYTHON" "$ROOT/scripts/verify_outputs.py" --model-root "$MODEL_ROOT" --analysis-root "$ANALYSIS_ROOT"
