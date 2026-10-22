#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# FINAL SYNTHETIC EXPERIMENT
#
# A) Observation-density sweep with FROZEN architecture
# B) One-time VIRGIN TEST of the already-frozen Nobs=1024 model
#
# IMPORTANT:
# The density sweep is descriptive/sensitivity analysis.
# It MUST NOT be used to change the final Nobs=1024 configuration.
# ============================================================

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda:0}"

OUTDIR="${OUTDIR:-out_density_final_arch}"

# Completely new test seed:
VIRGIN_SEED="${VIRGIN_SEED:-260926}"

# Virgin test size:
VIRGIN_N="${VIRGIN_N:-50000}"
VIRGIN_BOTTOM_N="${VIRGIN_BOTTOM_N:-10000}"
VIRGIN_BATCH="${VIRGIN_BATCH:-5000}"

# ------------------------------------------------------------
# Find the Example-I code
# ------------------------------------------------------------

if [[ -f "deepwind_example1_fixed_lhs.py" ]]; then

    TRAIN_PY="deepwind_example1_fixed_lhs.py"

elif [[ -f "deepWind3D_vortex_autograd_fixed_observations_lhs.py" ]]; then

    TRAIN_PY="deepWind3D_vortex_autograd_fixed_observations_lhs.py"

else

    echo "[ERROR] I cannot find either:"
    echo "  deepwind_example1_fixed_lhs.py"
    echo "or"
    echo "  deepWind3D_vortex_autograd_fixed_observations_lhs.py"
    exit 1

fi


echo "============================================================"
echo "FINAL DENSITY SWEEP + VIRGIN TEST"
echo "============================================================"
echo "Training code : ${TRAIN_PY}"
echo "Device        : ${DEVICE}"
echo "Output dir    : ${OUTDIR}"
echo
echo "Frozen architecture:"
echo "  width       = 60"
echo "  depth       = 2"
echo "  innerBatch  = 512"
echo "  bottomBatch = 64"
echo "  beta1       = 1"
echo "  beta2       = 1"
echo "  epochs      = 2000"
echo
echo "Density sweep:"
echo "  Nobs = 256 512 1024 2048"
echo "  seeds = 0 1 2"
echo
echo "Virgin test:"
echo "  FINAL Nobs = 1024"
echo "  Ntest       = ${VIRGIN_N}"
echo "  seed        = ${VIRGIN_SEED}"
echo "============================================================"


mkdir -p "${OUTDIR}/console"


# ============================================================
# PART A
# DENSITY SWEEP WITH FINAL/FROZEN ARCHITECTURE
# ============================================================

for NOBS in 256 512 1024 2048
do

    for SEED in 0 1 2
    do

        TAG="density_n${NOBS}_seed${SEED}"

        SUMMARY="${OUTDIR}/summary/${TAG}_eval_summary.csv"

        echo
        echo "============================================================"
        echo "[RUN] Nobs=${NOBS}, seed=${SEED}"
        echo "============================================================"

        if [[ -s "${SUMMARY}" ]]; then

            echo "[SKIP] already exists:"
            echo "       ${SUMMARY}"

        else

            ${PYTHON_BIN} "${TRAIN_PY}" \
                --width 60 \
                --depth 2 \
                --nEpoch 2000 \
                --nBatch 25 \
                --innerBatchSize 512 \
                --bdyBatchSize 64 \
                --nObs "${NOBS}" \
                --obsSampling lhs \
                --obsSeed 12345 \
                --nVal 5000 \
                --nTest 10000 \
                --evalSeed 54321 \
                --penalty 1.0 \
                --penalty2 1.0 \
                --epsilon 0.1 \
                --lr 1e-2 \
                --decay 1e-6 \
                --step_size 200 \
                --gamma 0.5 \
                --valEvery 10 \
                --logEvery 30 \
                --seed "${SEED}" \
                --device "${DEVICE}" \
                --outdir "${OUTDIR}" \
                --tag "${TAG}" \
                --noPlots \
                2>&1 | tee "${OUTDIR}/console/${TAG}.log"

        fi

    done

done


# ============================================================
# AGGREGATE DENSITY RESULTS
# ============================================================

echo
echo "============================================================"
echo "AGGREGATING DENSITY RESULTS"
echo "============================================================"


export OUTDIR


${PYTHON_BIN} - <<'PY'

import csv
import glob
import os

import numpy as np


outdir = os.environ["OUTDIR"]

pattern = os.path.join(
    outdir,
    "summary",
    "density_n*_seed*_eval_summary.csv",
)

files = sorted(glob.glob(pattern))

if len(files) != 12:

    raise RuntimeError(
        f"Expected 12 density summaries, found {len(files)}"
    )


def get_value(row, *names):

    for name in names:

        if name in row and row[name] not in ("", None):

            return float(row[name])

    return float("nan")


runs = []

for filename in files:

    with open(filename, newline="") as f:

        row = next(csv.DictReader(f))

    runs.append({

        "nObs":
            int(row["nObs"]),

        "seed":
            int(row["seed"]),

        "best_epoch":
            int(float(row["best_epoch"])),

        "best_val_mse_uv":
            get_value(
                row,
                "best_val_mse_uv",
            ),

        "dev_test_mse_uv":
            get_value(
                row,
                "test_mse_uv",
            ),

        "dev_test_mse_uvw":
            get_value(
                row,
                "test_mse_uvw",
            ),

        "dev_test_relL2":
            get_value(
                row,
                "test_relL2_field",
            ),

        "dev_test_rms_div":
            get_value(
                row,
                "test_rms_div",
            ),

        "dev_test_rms_bottom":
            get_value(
                row,
                "test_rms_u3_bottom",
                "test_rms_bottom_bc",
            ),

        "train_seconds":
            get_value(
                row,
                "train_seconds",
            ),

        "checkpoint_path":
            row.get(
                "checkpoint_path",
                "",
            ),
    })


runs.sort(
    key=lambda r: (
        r["nObs"],
        r["seed"],
    )
)


# ------------------------------------------------------------
# Save individual runs
# ------------------------------------------------------------

run_fields = list(runs[0].keys())

runs_csv = os.path.join(
    outdir,
    "density_final_arch_runs.csv",
)

with open(
    runs_csv,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=run_fields,
    )

    writer.writeheader()
    writer.writerows(runs)


# ------------------------------------------------------------
# Mean over seeds
# ------------------------------------------------------------

mean_rows = []

metric_names = [

    "best_val_mse_uv",

    "dev_test_mse_uv",

    "dev_test_mse_uvw",

    "dev_test_relL2",

    "dev_test_rms_div",

    "dev_test_rms_bottom",

    "best_epoch",

    "train_seconds",
]


for nobs in [256, 512, 1024, 2048]:

    rr = [
        r
        for r in runs
        if r["nObs"] == nobs
    ]

    out = {

        "nObs":
            nobs,

        "nSeeds":
            len(rr),
    }

    for metric in metric_names:

        vals = np.asarray(
            [
                float(r[metric])
                for r in rr
            ],
            dtype=float,
        )

        out[f"{metric}_mean"] = float(
            np.mean(vals)
        )

        # Sample standard deviation (n-1), consistent with manuscript.
        out[f"{metric}_std"] = float(
            np.std(
                vals,
                ddof=1,
            )
        )

    mean_rows.append(out)


mean_csv = os.path.join(
    outdir,
    "density_final_arch_mean.csv",
)

with open(
    mean_csv,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            mean_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(mean_rows)


print()
print("DENSITY SWEEP SUMMARY")
print("=" * 100)

for r in mean_rows:

    print(
        f"Nobs={r['nObs']:4d}  "
        f"val MSEuv="
        f"{r['best_val_mse_uv_mean']:.6e} "
        f"+/- "
        f"{r['best_val_mse_uv_std']:.2e}  "
        f"dev MSEuvw="
        f"{r['dev_test_mse_uvw_mean']:.6e}  "
        f"RMSdiv="
        f"{r['dev_test_rms_div_mean']:.6e}"
    )

print()
print(f"Wrote: {runs_csv}")
print(f"Wrote: {mean_csv}")

PY


# ============================================================
# PART B
# ONE-TIME VIRGIN TEST
#
# IMPORTANT:
#
# Nobs=1024 is hard-coded below.
# We DO NOT select Nobs using the density results above.
#
# We evaluate all 3 validation-selected checkpoints exactly once
# on the SAME new 50,000-point test set.
# ============================================================


echo
echo "============================================================"
echo "STARTING VIRGIN TEST"
echo "============================================================"
echo "This is the ONE-TIME final test."
echo
echo "Nobs        = 1024 (already frozen)"
echo "test points = ${VIRGIN_N}"
echo "test seed   = ${VIRGIN_SEED}"
echo "============================================================"


export TRAIN_PY
export DEVICE
export VIRGIN_SEED
export VIRGIN_N
export VIRGIN_BOTTOM_N
export VIRGIN_BATCH


${PYTHON_BIN} - <<'PY'

import csv
import importlib.util
import os
from pathlib import Path

import numpy as np
import torch


# ============================================================
# Configuration
# ============================================================

train_py = os.environ["TRAIN_PY"]

outdir = os.environ["OUTDIR"]

device_name = os.environ["DEVICE"]

virgin_seed = int(
    os.environ["VIRGIN_SEED"]
)

n_test = int(
    os.environ["VIRGIN_N"]
)

n_bottom = int(
    os.environ["VIRGIN_BOTTOM_N"]
)

batch_size = int(
    os.environ["VIRGIN_BATCH"]
)


# ============================================================
# Import EXACTLY the same model code used for training
# ============================================================

spec = importlib.util.spec_from_file_location(
    "deepwind_final_module",
    train_py,
)

if spec is None or spec.loader is None:

    raise RuntimeError(
        f"Could not import {train_py}"
    )

mod = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(mod)


torch.set_default_dtype(
    torch.float64
)


device = torch.device(
    device_name
    if torch.cuda.is_available()
       or not device_name.startswith("cuda")
    else "cpu"
)


print()
print(
    f"[VIRGIN] device = {device}"
)


# ============================================================
# Fixed virgin evaluation set
#
# This seed has never been used for:
#   - observations
#   - validation
#   - development test
#   - network initialization
# ============================================================

rng = np.random.RandomState(
    virgin_seed
)


X = rng.uniform(
    [-7.0, -7.0, 0.0],
    [ 7.0,  7.0, 7.0],
    size=(
        n_test,
        3,
    ),
).astype(np.float64)


# Exact Example-I solution

try:

    U_exact = mod.uexact_np(
        X,
        0.1,
    )

except TypeError:

    U_exact = mod.uexact_np(
        X
    )


# ============================================================
# Independent bottom set
# ============================================================

rng_bottom = np.random.RandomState(
    virgin_seed + 1
)


X_bottom = np.column_stack(
    [
        rng_bottom.uniform(
            -7.0,
            7.0,
            n_bottom,
        ),

        rng_bottom.uniform(
            -7.0,
            7.0,
            n_bottom,
        ),

        np.zeros(
            n_bottom,
            dtype=np.float64,
        ),
    ]
).astype(np.float64)


# ============================================================
# Output
# ============================================================

virgin_dir = os.path.join(
    outdir,
    "virgin_test",
)

os.makedirs(
    virgin_dir,
    exist_ok=True,
)


run_rows = []


# ============================================================
# Evaluate the three FINAL Nobs=1024 checkpoints
# ============================================================

for seed in [0, 1, 2]:

    tag = (
        f"density_n1024_seed{seed}"
    )

    checkpoint_path = os.path.join(
        outdir,
        "checkpoints",
        f"{tag}_best.pt",
    )


    if not os.path.isfile(
        checkpoint_path
    ):

        raise FileNotFoundError(
            checkpoint_path
        )


    print()
    print(
        "=" * 70
    )
    print(
        f"VIRGIN TEST: seed={seed}"
    )
    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Recreate exact frozen architecture
    # --------------------------------------------------------

    model = mod.RitzNet(
        in_dim=3,
        out_dim=3,
        width=60,
        depth=2,
        bounds=(
            -7.0,
             7.0,
            -7.0,
             7.0,
             0.0,
             7.0,
        ),
    ).to(device)


    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()


    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    U_pred = np.empty(
        (
            n_test,
            3,
        ),
        dtype=np.float64,
    )


    with torch.no_grad():

        for i0 in range(
            0,
            n_test,
            batch_size,
        ):

            i1 = min(
                i0 + batch_size,
                n_test,
            )

            xb = torch.from_numpy(
                X[i0:i1]
            ).to(device)

            U_pred[i0:i1] = (
                model(xb)
                .detach()
                .cpu()
                .numpy()
            )


    # --------------------------------------------------------
    # Error metrics
    # --------------------------------------------------------

    diff = (
        U_pred
        -
        U_exact
    )


    mse_uv = float(
        np.mean(
            diff[:, :2] ** 2
        )
    )


    mse_uvw = float(
        np.mean(
            diff ** 2
        )
    )


    rel_l2 = float(
        np.linalg.norm(
            diff
        )
        /
        (
            np.linalg.norm(
                U_exact
            )
            +
            1.0e-14
        )
    )


    rms_uv_exact = float(
        np.sqrt(
            np.mean(
                U_exact[:, :2] ** 2
            )
        )
    )


    rms_uvw_exact = float(
        np.sqrt(
            np.mean(
                U_exact ** 2
            )
        )
    )


    rel_uv = float(
        np.sqrt(
            mse_uv
            /
            (
                rms_uv_exact ** 2
                +
                1.0e-14
            )
        )
    )


    rel_uvw = float(
        np.sqrt(
            mse_uvw
            /
            (
                rms_uvw_exact ** 2
                +
                1.0e-14
            )
        )
    )


    # --------------------------------------------------------
    # Physical divergence:
    #
    # Do this in chunks because gradients w.r.t. coordinates
    # are needed.
    # --------------------------------------------------------

    div_sq_sum = 0.0
    div_count = 0


    for i0 in range(
        0,
        n_test,
        batch_size,
    ):

        i1 = min(
            i0 + batch_size,
            n_test,
        )

        xb = torch.from_numpy(
            X[i0:i1]
        ).to(device)


        with torch.enable_grad():

            _u, div = (
                mod.physical_divergence_autograd(
                    model=model,
                    x_phys=xb,
                    create_graph=False,
                )
            )


        div_np = (
            div.detach()
            .cpu()
            .numpy()
        )


        div_sq_sum += float(
            np.sum(
                div_np ** 2
            )
        )


        div_count += (
            div_np.size
        )


    rms_div = float(
        np.sqrt(
            div_sq_sum
            /
            div_count
        )
    )


    # --------------------------------------------------------
    # Bottom boundary: exact w=0 at z=0
    # --------------------------------------------------------

    bottom_sq_sum = 0.0
    bottom_count = 0


    with torch.no_grad():

        for i0 in range(
            0,
            n_bottom,
            batch_size,
        ):

            i1 = min(
                i0 + batch_size,
                n_bottom,
            )

            xb = torch.from_numpy(
                X_bottom[i0:i1]
            ).to(device)


            wb = (
                model(xb)[:, 2]
                .detach()
                .cpu()
                .numpy()
            )


            bottom_sq_sum += float(
                np.sum(
                    wb ** 2
                )
            )


            bottom_count += (
                wb.size
            )


    rms_bottom = float(
        np.sqrt(
            bottom_sq_sum
            /
            bottom_count
        )
    )


    # --------------------------------------------------------
    # Checkpoint selection information
    # --------------------------------------------------------

    best_epoch = int(
        checkpoint.get(
            "epoch",
            -1,
        )
    )


    best_val_mse_uv = float(
        checkpoint.get(
            "val_mse_uv",
            np.nan,
        )
    )


    # --------------------------------------------------------
    # Save predictions for reproducibility / future figures
    # --------------------------------------------------------

    pred_file = os.path.join(
        virgin_dir,
        f"virgin_predictions_seed{seed}.csv",
    )


    table = np.column_stack(
        [
            X,
            U_exact,
            U_pred,
        ]
    )


    np.savetxt(
        pred_file,
        table,
        delimiter=",",
        header=(
            "x,y,z,"
            "u_exact,v_exact,w_exact,"
            "u_pred,v_pred,w_pred"
        ),
        comments="",
    )


    row = {

        "seed":
            seed,

        "nObs":
            1024,

        "width":
            60,

        "depth":
            2,

        "nParameters":
            n_params,

        "virgin_seed":
            virgin_seed,

        "virgin_nTest":
            n_test,

        "virgin_nBottom":
            n_bottom,

        "best_epoch":
            best_epoch,

        "best_val_mse_uv":
            best_val_mse_uv,

        "virgin_mse_uv":
            mse_uv,

        "virgin_mse_uvw":
            mse_uvw,

        "virgin_relL2_field":
            rel_l2,

        "virgin_rel_uv":
            rel_uv,

        "virgin_rel_uvw":
            rel_uvw,

        "virgin_rms_div":
            rms_div,

        "virgin_rms_bottom":
            rms_bottom,

        "checkpoint":
            checkpoint_path,

        "prediction_file":
            pred_file,
    }


    run_rows.append(
        row
    )


    print(
        f"parameters               = {n_params}"
    )

    print(
        f"best epoch               = {best_epoch}"
    )

    print(
        f"best val MSE(u,v)        = "
        f"{best_val_mse_uv:.6e}"
    )

    print(
        f"VIRGIN MSE(u,v)          = "
        f"{mse_uv:.6e}"
    )

    print(
        f"VIRGIN MSE(u,v,w)        = "
        f"{mse_uvw:.6e}"
    )

    print(
        f"VIRGIN relL2(field)      = "
        f"{rel_l2:.6e}"
    )

    print(
        f"VIRGIN RMS(div)          = "
        f"{rms_div:.6e}"
    )

    print(
        f"VIRGIN RMS(bottom)       = "
        f"{rms_bottom:.6e}"
    )


# ============================================================
# Save individual virgin-test results
# ============================================================

runs_csv = os.path.join(
    virgin_dir,
    "virgin_test_final_runs.csv",
)


with open(
    runs_csv,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            run_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        run_rows
    )


# ============================================================
# Mean/std over training seeds
# ============================================================

metrics = [

    "best_val_mse_uv",

    "virgin_mse_uv",

    "virgin_mse_uvw",

    "virgin_relL2_field",

    "virgin_rel_uv",

    "virgin_rel_uvw",

    "virgin_rms_div",

    "virgin_rms_bottom",

    "best_epoch",
]


summary = {

    "nObs":
        1024,

    "width":
        60,

    "depth":
        2,

    "nSeeds":
        3,

    "virgin_seed":
        virgin_seed,

    "virgin_nTest":
        n_test,

    "virgin_nBottom":
        n_bottom,
}


for metric in metrics:

    values = np.asarray(
        [
            float(r[metric])
            for r in run_rows
        ],
        dtype=float,
    )


    summary[
        f"{metric}_mean"
    ] = float(
        np.mean(values)
    )


    summary[
        f"{metric}_std"
    ] = float(
        np.std(
            values,
            ddof=1,
        )
    )


mean_csv = os.path.join(
    virgin_dir,
    "virgin_test_final_mean.csv",
)


with open(
    mean_csv,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            summary.keys()
        ),
    )

    writer.writeheader()
    writer.writerow(
        summary
    )


print()
print(
    "=" * 100
)

print(
    "FINAL VIRGIN TEST — MEAN +/- STD OVER SEEDS"
)

print(
    "=" * 100
)


print(
    "MSE(u,v)      = "
    f"{summary['virgin_mse_uv_mean']:.6e} "
    "+/- "
    f"{summary['virgin_mse_uv_std']:.2e}"
)


print(
    "MSE(u,v,w)    = "
    f"{summary['virgin_mse_uvw_mean']:.6e} "
    "+/- "
    f"{summary['virgin_mse_uvw_std']:.2e}"
)


print(
    "relL2(field)  = "
    f"{summary['virgin_relL2_field_mean']:.6e} "
    "+/- "
    f"{summary['virgin_relL2_field_std']:.2e}"
)


print(
    "RMS(div)      = "
    f"{summary['virgin_rms_div_mean']:.6e} "
    "+/- "
    f"{summary['virgin_rms_div_std']:.2e}"
)


print(
    "RMS(bottom)   = "
    f"{summary['virgin_rms_bottom_mean']:.6e} "
    "+/- "
    f"{summary['virgin_rms_bottom_std']:.2e}"
)


print()
print(
    f"Wrote: {runs_csv}"
)

print(
    f"Wrote: {mean_csv}"
)

print()
print(
    "IMPORTANT: this virgin test has now been opened."
)

print(
    "Do not change model/hyperparameters based on these results."
)

PY


echo
echo "============================================================"
echo "ALL FINAL SYNTHETIC RUNS FINISHED"
echo "============================================================"

echo
echo "Density:"
echo "  ${OUTDIR}/density_final_arch_runs.csv"
echo "  ${OUTDIR}/density_final_arch_mean.csv"

echo
echo "Virgin test:"
echo "  ${OUTDIR}/virgin_test/virgin_test_final_runs.csv"
echo "  ${OUTDIR}/virgin_test/virgin_test_final_mean.csv"

echo
echo "Predictions:"
echo "  ${OUTDIR}/virgin_test/virgin_predictions_seed0.csv"
echo "  ${OUTDIR}/virgin_test/virgin_predictions_seed1.csv"
echo "  ${OUTDIR}/virgin_test/virgin_predictions_seed2.csv"

