#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CODE="${CODE:-deepWind3D_vortex_time_final_protocol.py}"
OUTDIR="${OUTDIR:-out_time_final_protocol}"
LOGDIR="${LOGDIR:-time_final_protocol_logs}"
DEVICE="${DEVICE:-cuda:0}"

SEEDS=(0 1 2)
mkdir -p "$OUTDIR" "$LOGDIR"

if [[ ! -f "$CODE" ]]; then
    echo "[ERROR] Training code not found: $CODE"
    echo "Run this script from the directory containing $CODE or set CODE=/full/path/to/file.py"
    exit 2
fi

for SEED in "${SEEDS[@]}"; do
    TAG="timefinal_parabolic_s${SEED}"
    SUMMARY="${OUTDIR}/summary/${TAG}_eval_summary.csv"
    LOGFILE="${LOGDIR}/${TAG}.log"

    if [[ -s "$SUMMARY" ]]; then
        echo "[SKIP] $TAG"
        continue
    fi

    echo "[RUN] $TAG"
    "$PYTHON_BIN" "$CODE" \
        --traj parabolic \
        --x0 0 --y0 0 \
        --vx 20 --vy 0 \
        --ax 0 --ay 30 \
        --a -25 --b 25 \
        --c -25 --ddd 25 \
        --e 0 --f 7 \
        --t0 -1 --t1 1 \
        --epsilon 0.02 \
        --width 60 --depth 2 \
        --nObs 1024 --obsSampling lhs --obsSeed 12345 \
        --innerBatchSize 512 --bdyBatchSize 64 \
        --penalty 1 --penalty2 1 \
        --nEpoch 2000 --nBatch 25 \
        --lr 1e-2 --decay 1e-6 --step_size 200 --gamma 0.5 \
        --nVal 5000 --nTest 20000 --nValBottom 5000 --nTestBottom 10000 \
        --evalSeed 54321 \
        --eval_times -1 -0.8 -0.5 -0.2 0 0.2 0.5 0.8 1 \
        --valEvery 10 --logEvery 30 \
        --seed "$SEED" --device "$DEVICE" \
        --outdir "$OUTDIR" --tag "$TAG" --noPlots \
        > "$LOGFILE" 2>&1

    echo "[DONE] $TAG"
    tail -n 20 "$LOGFILE" || true
done

# Aggregate run-level and time-slice results.
"$PYTHON_BIN" - "$OUTDIR" <<'PY'
import csv, glob, os, statistics, sys

outdir = sys.argv[1]
summary_files = sorted(glob.glob(os.path.join(outdir, "summary", "timefinal_parabolic_s*_eval_summary.csv")))
if len(summary_files) != 3:
    raise SystemExit(f"Expected 3 summary files, found {len(summary_files)}")

runs = []
for fn in summary_files:
    with open(fn, newline="") as f:
        runs.append(next(csv.DictReader(f)))

run_fields_num = [
    "best_val_mse_uv", "best_val_mse_uvw_diag", "best_val_relL2_diag",
    "test_mse_uv", "test_mse_uvw", "test_relL2", "test_rms_div",
    "test_rms_bottom", "test_rel_uv", "test_rel_uvw", "best_epoch",
    "train_seconds", "n_params"
]

with open("time_final_protocol_runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=runs[0].keys())
    w.writeheader(); w.writerows(runs)

agg = {}
for field in run_fields_num:
    vals = [float(r[field]) for r in runs]
    agg[f"mean_{field}"] = statistics.mean(vals)
    agg[f"std_{field}"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
agg["n_seeds"] = len(runs)

with open("time_final_protocol_mean.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=agg.keys())
    w.writeheader(); w.writerow(agg)

# Per-time aggregation.
time_files = sorted(glob.glob(os.path.join(outdir, "summary", "timefinal_parabolic_s*_time_metrics.csv")))
if len(time_files) != 3:
    raise SystemExit(f"Expected 3 time-metric files, found {len(time_files)}")

by_t = {}
for fn in time_files:
    with open(fn, newline="") as f:
        for r in csv.DictReader(f):
            t = float(r["t"])
            by_t.setdefault(t, []).append(r)

metrics = ["mse_uv", "mse_uvw", "relL2_uv", "relL2_uvw", "rms_div", "rms_bottom"]
rows = []
for t in sorted(by_t):
    grp = by_t[t]
    row = {"t": t, "n_seeds": len(grp)}
    for m in metrics:
        vals = [float(r[m]) for r in grp]
        row[f"mean_{m}"] = statistics.mean(vals)
        row[f"std_{m}"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    rows.append(row)

with open("time_final_protocol_by_time_mean.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)

print("Wrote:")
print("  time_final_protocol_runs.csv")
print("  time_final_protocol_mean.csv")
print("  time_final_protocol_by_time_mean.csv")
print()
print("Run-level means:")
for k, v in agg.items():
    if k.startswith("mean_"):
        print(f"  {k:32s} {v:.8e}")
PY

echo "All final time-dependent runs completed."
