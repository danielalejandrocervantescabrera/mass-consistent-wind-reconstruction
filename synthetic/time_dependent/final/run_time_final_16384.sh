#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CODE="${CODE:-deepWind3D_vortex_time_final_protocol.py}"

OUTDIR="${OUTDIR:-out_time_final_16384}"
SWEEP_DIR="${SWEEP_DIR:-out_time_obs_sweep}"
LOGDIR="${LOGDIR:-time_final_16384_logs}"

mkdir -p "${OUTDIR}" "${LOGDIR}"

if [[ ! -f "${CODE}" ]]; then
    echo "[ERROR] No encuentro ${CODE}"
    exit 1
fi

# ============================================================
# FROZEN TEMPORAL CONFIGURATION
# ============================================================

NOBS=16384

WIDTH=60
DEPTH=2

INNER=512
BDY=64

EPOCHS=2000
NBATCH=25

BETA1=1.0
BETA2=1.0

LR=1e-2
DECAY=1e-6
STEP_SIZE=200
GAMMA=0.5

OBS_SEED=12345
EVAL_SEED=54321

SEEDS=(0 1 2)

# ------------------------------------------------------------
# Temporal benchmark
# ------------------------------------------------------------

A=-25
B=25
C=-25
DDD=25
E=0
F=7

T0=-1
T1=1

X0=0
Y0=0

VX=20
VY=0

AX=0
AY=30

EPS=0.02


echo "============================================================"
echo "FINAL TEMPORAL TRAINING"
echo "============================================================"
echo "Nobs         = ${NOBS}"
echo "width        = ${WIDTH}"
echo "depth        = ${DEPTH}"
echo "Ninner       = ${INNER}"
echo "Nbdy         = ${BDY}"
echo "beta         = (${BETA1}, ${BETA2})"
echo "epochs       = ${EPOCHS}"
echo "seeds        = ${SEEDS[*]}"
echo "obsSeed      = ${OBS_SEED}"
echo "evalSeed     = ${EVAL_SEED}"
echo "============================================================"


run_one () {

    local SEED="$1"

    local TAG="timefinal16384_s${SEED}"

    local SUMMARY="${OUTDIR}/summary/${TAG}_eval_summary.csv"

    local LOGFILE="${LOGDIR}/${TAG}.log"


    if [[ -s "${SUMMARY}" ]]; then

        echo "[SKIP] seed=${SEED} already completed"

        return
    fi


    # --------------------------------------------------------
    # Reuse seed=0 from the previous Nobs sweep
    # --------------------------------------------------------

    if [[ "${SEED}" == "0" ]]; then

        local OLD_SUM="${SWEEP_DIR}/summary/timeobs_n16384_s0_eval_summary.csv"

        local OLD_TIME="${SWEEP_DIR}/summary/timeobs_n16384_s0_time_metrics.csv"

        if [[ -s "${OLD_SUM}" && -s "${OLD_TIME}" ]]; then

            echo "[REUSE] seed=0 from ${SWEEP_DIR}"

            return
        fi

    fi


    echo
    echo "------------------------------------------------------------"
    echo "[RUN] seed=${SEED}"
    echo "------------------------------------------------------------"


    "${PYTHON_BIN}" "${CODE}" \
        --traj parabolic \
        --x0 "${X0}" \
        --y0 "${Y0}" \
        --vx "${VX}" \
        --vy "${VY}" \
        --ax "${AX}" \
        --ay "${AY}" \
        --a "${A}" \
        --b "${B}" \
        --c "${C}" \
        --ddd "${DDD}" \
        --e "${E}" \
        --f "${F}" \
        --t0 "${T0}" \
        --t1 "${T1}" \
        --epsilon "${EPS}" \
        --width "${WIDTH}" \
        --depth "${DEPTH}" \
        --nEpoch "${EPOCHS}" \
        --nBatch "${NBATCH}" \
        --innerBatchSize "${INNER}" \
        --bdyBatchSize "${BDY}" \
        --nObs "${NOBS}" \
        --obsSampling lhs \
        --obsSeed "${OBS_SEED}" \
        --evalSeed "${EVAL_SEED}" \
        --penalty "${BETA1}" \
        --penalty2 "${BETA2}" \
        --lr "${LR}" \
        --decay "${DECAY}" \
        --step_size "${STEP_SIZE}" \
        --gamma "${GAMMA}" \
        --seed "${SEED}" \
        --outdir "${OUTDIR}" \
        --tag "${TAG}" \
        --noPlots \
        2>&1 | tee "${LOGFILE}"
}


for SEED in "${SEEDS[@]}"
do
    run_one "${SEED}"
done


export OUTDIR SWEEP_DIR


"${PYTHON_BIN}" - <<'PY'

import csv
import os
from pathlib import Path
import numpy as np


outdir = Path(os.environ["OUTDIR"])
sweep_dir = Path(os.environ["SWEEP_DIR"])

seeds = [0, 1, 2]


def files_for_seed(seed):

    if seed == 0:

        old_sum = (
            sweep_dir
            / "summary"
            / "timeobs_n16384_s0_eval_summary.csv"
        )

        old_time = (
            sweep_dir
            / "summary"
            / "timeobs_n16384_s0_time_metrics.csv"
        )

        if old_sum.exists() and old_time.exists():

            return old_sum, old_time


    tag = f"timefinal16384_s{seed}"

    return (

        outdir
        / "summary"
        / f"{tag}_eval_summary.csv",

        outdir
        / "summary"
        / f"{tag}_time_metrics.csv",
    )


# ============================================================
# Per-run summaries
# ============================================================

runs = []

time_rows = []


for seed in seeds:

    summary_file, time_file = files_for_seed(seed)


    if not summary_file.exists():

        raise FileNotFoundError(
            f"Missing summary for seed={seed}: "
            f"{summary_file}"
        )


    if not time_file.exists():

        raise FileNotFoundError(
            f"Missing time metrics for seed={seed}: "
            f"{time_file}"
        )


    with summary_file.open(newline="") as f:

        r = next(csv.DictReader(f))


    row = {

        "seed":
            int(float(r["seed"])),

        "nObs":
            int(float(r["nObs"])),

        "best_epoch":
            int(float(r["best_epoch"])),

        "best_val_mse_uv":
            float(r["best_val_mse_uv"]),

        "best_val_mse_uvw_diag":
            float(r["best_val_mse_uvw_diag"]),

        "test_mse_uv":
            float(r["test_mse_uv"]),

        "test_mse_uvw":
            float(r["test_mse_uvw"]),

        "test_relL2":
            float(r["test_relL2"]),

        "test_rms_div":
            float(r["test_rms_div"]),

        "test_rms_bottom":
            float(r["test_rms_bottom"]),

        "test_rel_uv":
            float(r["test_rel_uv"]),

        "test_rel_uvw":
            float(r["test_rel_uvw"]),

        "train_seconds":
            float(r["train_seconds"]),

        "summary_file":
            str(summary_file),
    }


    runs.append(row)


    with time_file.open(newline="") as f:

        for tr in csv.DictReader(f):

            time_rows.append({

                "seed":
                    seed,

                "t":
                    float(tr["t"]),

                "mse_uv":
                    float(tr["mse_uv"]),

                "mse_uvw":
                    float(tr["mse_uvw"]),

                "relL2_uv":
                    float(tr["relL2_uv"]),

                "relL2_uvw":
                    float(tr["relL2_uvw"]),

                "rms_div":
                    float(tr["rms_div"]),

                "rms_bottom":
                    float(tr["rms_bottom"]),
            })


# ============================================================
# Write runs CSV
# ============================================================

runs_file = Path(
    "time_final_16384_runs.csv"
)


with runs_file.open(
    "w",
    newline="",
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=list(
            runs[0].keys()
        ),
    )

    w.writeheader()

    w.writerows(
        runs
    )


# ============================================================
# Mean/std summary
# ============================================================

metrics = [

    "best_val_mse_uv",

    "best_val_mse_uvw_diag",

    "test_mse_uv",

    "test_mse_uvw",

    "test_relL2",

    "test_rms_div",

    "test_rms_bottom",

    "test_rel_uv",

    "test_rel_uvw",

    "train_seconds",
]


mean_row = {
    "nObs": 16384,
    "nSeeds": len(runs),
}


for key in metrics:

    vals = np.array(
        [r[key] for r in runs],
        dtype=float,
    )

    mean_row[f"{key}_mean"] = vals.mean()

    mean_row[f"{key}_std"] = vals.std(
        ddof=1
    )


epochs = np.array(
    [r["best_epoch"] for r in runs],
    dtype=float,
)

mean_row["best_epoch_mean"] = epochs.mean()

mean_row["best_epoch_std"] = epochs.std(
    ddof=1
)


mean_file = Path(
    "time_final_16384_mean.csv"
)


with mean_file.open(
    "w",
    newline="",
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=list(
            mean_row.keys()
        ),
    )

    w.writeheader()

    w.writerow(
        mean_row
    )


# ============================================================
# Per-time mean/std
# ============================================================

times = sorted(
    set(
        row["t"]
        for row in time_rows
    )
)


time_metrics = [

    "mse_uv",

    "mse_uvw",

    "relL2_uv",

    "relL2_uvw",

    "rms_div",

    "rms_bottom",
]


by_time = []


for t in times:

    subset = [
        r
        for r in time_rows
        if abs(r["t"] - t) < 1e-12
    ]


    row = {
        "t": t,
        "nSeeds": len(subset),
    }


    for key in time_metrics:

        vals = np.array(
            [r[key] for r in subset],
            dtype=float,
        )

        row[f"{key}_mean"] = vals.mean()

        row[f"{key}_std"] = vals.std(
            ddof=1
        )


    by_time.append(row)


time_mean_file = Path(
    "time_final_16384_by_time_mean.csv"
)


with time_mean_file.open(
    "w",
    newline="",
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=list(
            by_time[0].keys()
        ),
    )

    w.writeheader()

    w.writerows(
        by_time
    )


# ============================================================
# Pretty console summary
# ============================================================

print()

print("=" * 112)

print(
    "FINAL TEMPORAL CONFIGURATION — Nobs=16384"
)

print("=" * 112)


print(
    f"{'seed':>4}  "
    f"{'best epoch':>10}  "
    f"{'val MSEuv':>14}  "
    f"{'test MSEuv':>14}  "
    f"{'test MSEuvw':>14}  "
    f"{'relL2':>10}  "
    f"{'RMSdiv':>11}  "
    f"{'RMSb':>11}  "
    f"{'time(s)':>9}"
)


for r in runs:

    print(

        f"{r['seed']:4d}  "

        f"{r['best_epoch']:10d}  "

        f"{r['best_val_mse_uv']:14.6e}  "

        f"{r['test_mse_uv']:14.6e}  "

        f"{r['test_mse_uvw']:14.6e}  "

        f"{r['test_relL2']:10.4e}  "

        f"{r['test_rms_div']:11.4e}  "

        f"{r['test_rms_bottom']:11.4e}  "

        f"{r['train_seconds']:9.1f}"
    )


print()

print("MEAN ± STD")

for key in [

    "best_val_mse_uv",

    "test_mse_uv",

    "test_mse_uvw",

    "test_relL2",

    "test_rms_div",

    "test_rms_bottom",

]:

    print(

        f"{key:24s} = "

        f"{mean_row[key + '_mean']:.6e} "

        f"± "

        f"{mean_row[key + '_std']:.6e}"
    )


print()

print("Generated:")

print(
    f"  {runs_file}"
)

print(
    f"  {mean_file}"
)

print(
    f"  {time_mean_file}"
)

print("=" * 112)

PY


echo
echo "============================================================"
echo "FINAL 3-SEED TEMPORAL RUN FINISHED"
echo "============================================================"
