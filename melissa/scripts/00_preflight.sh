#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
"$PYTHON" - <<'PYIN'
import importlib
for name in ['numpy','scipy','requests','torch','matplotlib']:
 mod=importlib.import_module(name)
 print(name, getattr(mod,'__version__','available'))
import torch
print('CUDA available:',torch.cuda.is_available())
PYIN
"$PYTHON" "$ROOT/scripts/validate_raw.py" --raw-dir "$RAW"
