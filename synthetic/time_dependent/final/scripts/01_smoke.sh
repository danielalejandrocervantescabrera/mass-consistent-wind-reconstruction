#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"
PY="${PYTHON_BIN:-python3}"
"$PY" deepWind3D_vortex_time_final_protocol.py  --traj parabolic --a -25 --b 25 --c -25 --ddd 25 --e 0 --f 7 --t0 -1 --t1 1  --epsilon 0.02 --width 16 --depth 1 --nEpoch 1 --nBatch 1  --innerBatchSize 8 --bdyBatchSize 8 --nObs 16 --obsSampling lhs --obsSeed 12345  --nVal 32 --nTest 48 --nValBottom 24 --nTestBottom 36 --evalSeed 54321  --valEvery 1 --logEvery 1 --numQuad 3 --eval_times -1 0 1  --seed 0 --device cpu --outdir outputs/smoke --tag smoke_final --noPlots
"$PY" - <<'PY_CHECK'
import csv, pathlib
p=pathlib.Path('outputs/smoke/summary/smoke_final_eval_summary.csv')
assert p.is_file(),p
row=next(csv.DictReader(p.open()))
assert int(float(row['nObs']))==16
assert int(float(row['seed']))==0
print('OK: entrenamiento, checkpoint, métricas y archivos de prueba.')
PY_CHECK
