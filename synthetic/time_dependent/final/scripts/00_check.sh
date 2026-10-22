#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"
"${PYTHON_BIN:-python3}" -m py_compile   deepWind3D_vortex_time_final_protocol.py   eval_time_final_virgin.py   export_time_final_16384_for_gnuplot.py   plot_time_final_16384.py
for f in run_time_final_16384.sh run_time_obs_sweep.sh run_time_final_protocol.sh; do bash -n "$f"; done
"${PYTHON_BIN:-python3}" - <<'PY_CHECK'
import importlib
for x in ('numpy','scipy','torch','matplotlib'):
 m=importlib.import_module(x)
 print(f'{x}: {m.__version__}')
PY_CHECK
echo 'Fuentes temporales finales verificadas (sintaxis y dependencias).'
