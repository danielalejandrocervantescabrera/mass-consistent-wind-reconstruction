#!/usr/bin/env bash
set -euo pipefail

# Run from melissa/

mkdir -p melissa_paper_figures

python3 make_melissa_paper_figures.py \
  --data_dir melissa_reconstruction_allobs_prepared_5min \
  --model_root melissa_reconstruction_final \
  --stability_npz melissa_w_seed_stability/w_fields_all_real_times.npz \
  --representative_seed 2345 \
  --outdir melissa_paper_figures \
  --context_offset_hours 12 \
  --batch_size 65536 \
  --dpi 300 \
  2>&1 | tee melissa_paper_figures/figures.log
