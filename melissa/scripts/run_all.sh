#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
echo "Full 61-hour reconstruction; no future-time forecasting and no temporal holdouts."
bash "$ROOT/scripts/00_preflight.sh"
bash "$ROOT/scripts/01_prepare.sh"
bash "$ROOT/scripts/02_train.sh"
bash "$ROOT/scripts/03_analyze.sh"
bash "$ROOT/scripts/04_figures.sh"
bash "$ROOT/scripts/04_report.sh"
