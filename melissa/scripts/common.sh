#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
RAW="$ROOT/data/raw"
PREPARED="${PREPARED:-$ROOT/data/processed}"
MODEL_ROOT="${MODEL_ROOT:-$ROOT/outputs/models}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$ROOT/outputs/analysis}"
mkdir -p "$MODEL_ROOT" "$ANALYSIS_ROOT"
