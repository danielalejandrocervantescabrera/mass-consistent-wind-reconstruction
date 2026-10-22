#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python3 - "$ROOT" <<'PYCODE'
from pathlib import Path
import ast,sys,json
r=Path(sys.argv[1]); sources=list(r.rglob('*.py'))
for p in sources:
 if 'outputs' in p.parts:continue
 ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
print('Python syntax verified:',len(sources),'files')
raw=list((r/'melissa/data/raw').glob('*.dat'))
assert len(raw)==61,(len(raw),'hourly files')
print('Original Melissa hourly files:',len(raw))
print('Synthetic architecture summaries:',len(list((r/'synthetic/example1/reference/architecture_sensitivity').glob('*.csv'))))
assert (r/'melissa/src/make_melissa_paper_figures.py').exists()
PYCODE
test -f "$ROOT/synthetic/time_dependent/final/deepWind3D_vortex_time_final_protocol.py"
test -f "$ROOT/synthetic/time_dependent/final/eval_time_final_virgin.py"
echo 'Temporal final protocol and virgin evaluation present.'
