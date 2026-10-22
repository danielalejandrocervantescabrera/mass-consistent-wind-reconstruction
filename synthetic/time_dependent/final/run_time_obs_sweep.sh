#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CODE="${CODE:-deepWind3D_vortex_time_final_protocol.py}"

OUTDIR="${OUTDIR:-out_time_obs_sweep}"
BASELINE_DIR="${BASELINE_DIR:-out_time_final_protocol}"
LOGDIR="${LOGDIR:-time_obs_sweep_logs}"

mkdir -p "${OUTDIR}" "${LOGDIR}"

if [[ ! -f "${CODE}" ]]; then
    echo "[ERROR] No encuentro ${CODE}"
    echo "Ejecuta este script desde la carpeta donde está el .py"
    exit 1
fi

# ------------------------------------------------------------
# Frozen protocol
# ------------------------------------------------------------

SEED=0
OBS_SEED=12345
EVAL_SEED=54321

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

# Temporal benchmark
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

NOBS_LIST=(1024 4096 8192 16384)

echo "============================================================"
echo "TEMPORAL OBSERVATION-DENSITY SWEEP"
echo "============================================================"
echo "Architecture : width=${WIDTH}, depth=${DEPTH}"
echo "Physics      : Ninner=${INNER}, Nbdy=${BDY}, beta=(1,1)"
echo "Training     : ${EPOCHS} epochs, ${NBATCH} batches/epoch"
echo "Seed         : ${SEED}"
echo "Nobs         : ${NOBS_LIST[*]}"
echo "============================================================"

run_one () {

    local NOBS="$1"

    local TAG="timeobs_n${NOBS}_s${SEED}"

    local SUMMARY="${OUTDIR}/summary/${TAG}_eval_summary.csv"

    local LOGFILE="${LOGDIR}/${TAG}.log"


    if [[ -s "${SUMMARY}" ]]; then
        echo "[SKIP] ${TAG}"
        return
    fi


    # Reuse the previous Nobs=1024 seed-0 run if available
    if [[ "${NOBS}" == "1024" ]]; then

        local BASE_SUM="${BASELINE_DIR}/summary/timefinal_parabolic_s0_eval_summary.csv"
        local BASE_TIME="${BASELINE_DIR}/summary/timefinal_parabolic_s0_time_metrics.csv"

        if [[ -s "${BASE_SUM}" && -s "${BASE_TIME}" ]]; then

            echo "[REUSE] Nobs=1024 from ${BASELINE_DIR}"

            return

        fi
    fi


    echo
    echo "------------------------------------------------------------"
    echo "[RUN] Nobs=${NOBS}"
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


for NOBS in "${NOBS_LIST[@]}"
do
    run_one "${NOBS}"
done


export OUTDIR BASELINE_DIR SEED


"${PYTHON_BIN}" - <<'PY'

import csv
import os
from pathlib import Path


outdir = Path(os.environ["OUTDIR"])

baseline_dir = Path(os.environ["BASELINE_DIR"])

seed = int(os.environ["SEED"])


nobs_list = [
    1024,
    4096,
    8192,
    16384,
]


def paths_for(nobs):

    if nobs == 1024:

        base_sum = (
            baseline_dir
            / "summary"
            / "timefinal_parabolic_s0_eval_summary.csv"
        )

        base_time = (
            baseline_dir
            / "summary"
            / "timefinal_parabolic_s0_time_metrics.csv"
        )

        if base_sum.exists() and base_time.exists():

            return base_sum, base_time


    tag = f"timeobs_n{nobs}_s{seed}"

    return (

        outdir
        / "summary"
        / f"{tag}_eval_summary.csv",

        outdir
        / "summary"
        / f"{tag}_time_metrics.csv",
    )


rows = []

time_rows = []


for nobs in nobs_list:

    sf, tf = paths_for(nobs)


    if not sf.exists():

        raise FileNotFoundError(
            f"Missing summary for Nobs={nobs}: {sf}"
        )


    if not tf.exists():

        raise FileNotFoundError(
            f"Missing time metrics for Nobs={nobs}: {tf}"
        )


    with sf.open(newline="") as f:

        r = next(
            csv.DictReader(f)
        )


    row = {

        "nObs":
            int(float(r["nObs"])),

        "seed":
            int(float(r["seed"])),

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
            str(sf),
    }


    rows.append(row)


    with tf.open(newline="") as f:

        for tr in csv.DictReader(f):

            time_rows.append({

                "nObs":
                    nobs,

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


summary_out = Path(
    "time_obs_sweep_seed0.csv"
)


with summary_out.open(
    "w",
    newline="",
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=list(
            rows[0].keys()
        ),
    )

    w.writeheader()

    w.writerows(rows)


time_out = Path(
    "time_obs_sweep_by_time_seed0.csv"
)


with time_out.open(
    "w",
    newline="",
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=list(
            time_rows[0].keys()
        ),
    )

    w.writeheader()

    w.writerows(
        time_rows
    )


print()

print("=" * 110)

print(
    "TEMPORAL Nobs SWEEP — SEED 0"
)

print("=" * 110)


print(

    f"{'Nobs':>7}  "

    f"{'best val MSEuv':>16}  "

    f"{'test MSEuv':>14}  "

    f"{'test MSEuvw':>14}  "

    f"{'relL2':>10}  "

    f"{'RMSdiv':>11}  "

    f"{'best ep':>8}  "

    f"{'time(s)':>9}"
)


for r in rows:

    print(

        f"{r['nObs']:7d}  "

        f"{r['best_val_mse_uv']:16.6e}  "

        f"{r['test_mse_uv']:14.6e}  "

        f"{r['test_mse_uvw']:14.6e}  "

        f"{r['test_relL2']:10.4e}  "

        f"{r['test_rms_div']:11.4e}  "

        f"{r['best_epoch']:8d}  "

        f"{r['train_seconds']:9.1f}"
    )


print()

print("Generated:")

print(
    f"  {summary_out}"
)

print(
    f"  {time_out}"
)

PY


echo
echo "============================================================"
echo "SWEEP FINISHED"
echo "============================================================"
