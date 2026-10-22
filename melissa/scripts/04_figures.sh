#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
STAB="$ANALYSIS_ROOT/seed_stability/w_fields_all_real_times.npz"
[[ -f "$STAB" ]] || { echo "Missing $STAB: run scripts/03_analyze.sh first" >&2; exit 1; }
[[ -f "$MODEL_ROOT/seed_2345/final_model.pt" ]] || { echo "Missing trained seed 2345 model" >&2; exit 1; }
FIGROOT="${FIGROOT:-$ROOT/outputs/figures}"
mkdir -p "$FIGROOT"
MPLBACKEND=Agg "$PYTHON" "$ROOT/src/make_melissa_paper_figures.py" \
  --data_dir "$PREPARED" \
  --model_root "$MODEL_ROOT" \
  --stability_npz "$STAB" \
  --representative_seed 2345 \
  --outdir "$FIGROOT" \
  --context_offset_hours 12 --batch_size 65536 --dpi 300 \
  2>&1 | tee "$FIGROOT/figures.log"
echo "Publication figures in $FIGROOT"
