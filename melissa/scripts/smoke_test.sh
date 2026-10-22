#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
TMP="$ROOT/outputs/smoke_test"
mkdir -p "$TMP/raw"
# Create a tiny 6-hour subset without touching archived raw data.
"$PYTHON" - "$RAW" "$TMP/raw" <<'PYIN'
from pathlib import Path
import csv,sys,shutil
raw=Path(sys.argv[1]);dest=Path(sys.argv[2]);dest.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader(open(raw/'hourly_eye_centers.csv',newline='')))
rows=rows[:6]
with open(dest/'hourly_eye_centers.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
for r in rows:
 key=r['key'];date,hh=key.split('T')
 matches=list(raw.glob(f'*date{date}_hour{hh}.dat'))
 assert len(matches)==1,key
 shutil.copy2(matches[0],dest/matches[0].name)
print('Smoke input: 6 archived hourly fields')
PYIN
# Functional test only; 6 hours, unlike the full 61-hour published experiment.
"$PYTHON" "$ROOT/src/prepare_melissa_hourly.py" \
 --input_dir "$TMP/raw" --centers "$TMP/raw/hourly_eye_centers.csv" \
 --outdir "$TMP/prepared" --dev_key '' --val_key '' --test_key '' \
 --nx 6 --ny 6 --dz 40 --dt_minutes 30 \
 --holdout_gap_minutes 0 --wahba_lambda 0.001
"$PYTHON" "$ROOT/src/train_melissa_reconstruction_final.py" \
 --data_dir "$TMP/prepared" --outdir "$TMP/model_seed_1234" \
 --seed 1234 --width 12 --depth 1 --nEpoch 1 --nBatch 1 \
 --dataBatchSize 32 --divBatchSize 32 --bottomBatchSize 16 \
 --evalDataSize 200 --evalDivSize 32 --evalBdySize 32 \
 --eval_every 1 --log_every 1
"$PYTHON" - "$TMP" <<'PYIN'
from pathlib import Path
import json,sys
p=Path(sys.argv[1]); assert (p/'prepared/manifest.json').is_file()
assert (p/'model_seed_1234/final_model.pt').is_file()
assert json.load(open(p/'model_seed_1234/summary.json'))['seed']==1234
print('SMOKE TEST PASSED (reduced-size, not publication results)')
PYIN
