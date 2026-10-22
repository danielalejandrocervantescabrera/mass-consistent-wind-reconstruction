#!/usr/bin/env bash
# Recompute derived vertical diagnostics from frozen historical models; no training.
source "$(dirname "$0")/common.sh"
export MODEL_ROOT="$ROOT/reference/checkpoints"
export ANALYSIS_ROOT="$ROOT/outputs/historical_analysis"
# Preprocessing can use considerable memory/disk: 721 x 24000 examples.
if [[ ! -f "$PREPARED/manifest.json" || ! -f "$PREPARED/train_X.npy" || ! -f "$PREPARED/train_Yuv.npy" || ! -f "$PREPARED/train_hourly_processed.npz" ]]; then
  echo 'Processed fields absent: rebuilding from the archived 61 hourly inputs.'
  bash "$ROOT/scripts/01_prepare.sh"
fi
bash "$ROOT/scripts/06_verify_history.sh"
bash "$ROOT/scripts/03_analyze.sh"
echo "Historical-model derived diagnostics: $ANALYSIS_ROOT"
