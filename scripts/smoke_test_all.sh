#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
bash "$ROOT/scripts/check_sources.sh"
bash "$ROOT/synthetic/example1/smoke_test.sh"
bash "$ROOT/melissa/scripts/smoke_test.sh"
bash "$ROOT/synthetic/time_dependent/final/scripts/01_smoke.sh"
