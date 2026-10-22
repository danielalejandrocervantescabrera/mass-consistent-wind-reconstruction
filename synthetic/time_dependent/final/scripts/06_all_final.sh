#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
bash "$DIR/scripts/00_check.sh"
bash "$DIR/scripts/02_train_final.sh"
TEMPORAL_CKPT_SET=new bash "$DIR/scripts/03_evaluate_virgin.sh"
TEMPORAL_CKPT_SET=new bash "$DIR/scripts/04_figures.sh"
