#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/outputs"
{
  date -u +"UTC: %Y-%m-%dT%H:%M:%SZ"
  echo '--- OS ---'
  uname -a
  echo '--- Python ---'
  python3 --version
  echo '--- pip packages ---'
  python3 -m pip freeze
  echo '--- GPU ---'
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; else echo 'nvidia-smi unavailable'; fi
} > "$ROOT/outputs/environment.txt"
echo "Recorded environment at $ROOT/outputs/environment.txt"
