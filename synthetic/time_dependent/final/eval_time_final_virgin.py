#!/usr/bin/env python3

import csv
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


# ============================================================
# FINAL VIRGIN EVALUATION
# ============================================================

FINAL_EVAL_SEED = 20260929

N_TEST = 50000
N_TEST_BOTTOM = 20000

N_SLICE = 20000
N_SLICE_BOTTOM = 10000

BATCH = 5000

EVAL_TIMES = [
    -1.0,
    -0.8,
    -0.5,
    -0.2,
     0.0,
     0.2,
     0.5,
     0.8,
     1.0,
]


# ============================================================
# Frozen domain
# ============================================================

A, B = -25.0, 25.0
C, D = -25.0, 25.0
E, F =   0.0,  7.0

T0, T1 = -1.0, 1.0


# ============================================================
# Frozen network
# ============================================================

WIDTH = 60
DEPTH = 2


CODE = Path(
    "deepWind3D_vortex_time_final_protocol.py"
)


# ============================================================
# Frozen checkpoints
# ============================================================

CHECKPOINT_CANDIDATES = {

    0: [
        Path(
            "out_time_obs_sweep/"
            "checkpoints/"
            "timeobs_n16384_s0_best.pt"
        ),

        Path(
            "out_time_final_16384/"
            "checkpoints/"
            "timefinal16384_s0_best.pt"
        ),
    ],

    1: [
        Path(
            "out_time_final_16384/"
            "checkpoints/"
            "timefinal16384_s1_best.pt"
        ),
    ],

    2: [
        Path(
            "out_time_final_16384/"
            "checkpoints/"
            "timefinal16384_s2_best.pt"
        ),
    ],
}


# ============================================================
# Output files
# ============================================================

RUNS_CSV = Path(
    "time_final_16384_virgin_runs.csv"
)

MEAN_CSV = Path(
    "time_final_16384_virgin_mean.csv"
)

TIME_RUNS_CSV = Path(
    "time_final_16384_virgin_by_time_runs.csv"
)

TIME_MEAN_CSV = Path(
    "time_final_16384_virgin_by_time_mean.csv"
)


# ============================================================
# Load original protocol module
# ============================================================

def load_protocol_module(path):

    if not path.exists():

        raise FileNotFoundError(
            f"No encuentro {path}"
        )


    spec = importlib.util.spec_from_file_location(
        "deepwind_time_protocol",
        path,
    )


    mod = importlib.util.module_from_spec(
        spec
    )


    spec.loader.exec_module(
        mod
    )


    return mod


# ============================================================
# Locate checkpoint
# ============================================================

def resolve_checkpoint(seed):

    for path in CHECKPOINT_CANDIDATES[seed]:

        if path.exists():

            return path


    tried = "\n  ".join(
        str(p)
        for p in CHECKPOINT_CANDIDATES[seed]
    )


    raise FileNotFoundError(

        f"No encuentro checkpoint "
        f"para seed={seed}.\n"
        f"Probé:\n  {tried}"
    )


# ============================================================
# Sampling
# ============================================================

def uniform_box(
    rng,
    n,
    fixed_t=None,
):

    x = rng.uniform(
        A,
        B,
        n,
    )

    y = rng.uniform(
        C,
        D,
        n,
    )

    z = rng.uniform(
        E,
        F,
        n,
    )


    if fixed_t is None:

        t = rng.uniform(
            T0,
            T1,
            n,
        )

    else:

        t = np.full(
            n,
            float(fixed_t),
            dtype=np.float64,
        )


    return np.column_stack(
        [
            x,
            y,
            z,
            t,
        ]
    ).astype(
        np.float64,
        copy=False,
    )


def bottom_points(
    rng,
    n,
    fixed_t=None,
):

    x = rng.uniform(
        A,
        B,
        n,
    )

    y = rng.uniform(
        C,
        D,
        n,
    )

    z = np.full(
        n,
        E,
        dtype=np.float64,
    )


    if fixed_t is None:

        t = rng.uniform(
            T0,
            T1,
            n,
        )

    else:

        t = np.full(
            n,
            float(fixed_t),
            dtype=np.float64,
        )


    return np.column_stack(
        [
            x,
            y,
            z,
            t,
        ]
    ).astype(
        np.float64,
        copy=False,
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    mod,
    model,
    X,
    Xb,
    p,
    device,
):

    n = X.shape[0]


    sse_uv = 0.0

    sse_uvw = 0.0

    exact_uv2 = 0.0

    exact_uvw2 = 0.0

    div2 = 0.0


    model.eval()


    for start in range(
        0,
        n,
        BATCH,
    ):

        stop = min(
            start + BATCH,
            n,
        )


        xb_np = X[
            start:stop
        ]


        exact = mod.uexact_time_np(
            xb_np,
            p,
        )


        xt = torch.from_numpy(
            xb_np
        ).to(
            device
        )


        with torch.no_grad():

            pred = model(
                xt
            ).detach().cpu().numpy()


        err = pred - exact


        sse_uv += float(
            np.sum(
                err[:, :2] ** 2
            )
        )


        sse_uvw += float(
            np.sum(
                err ** 2
            )
        )


        exact_uv2 += float(
            np.sum(
                exact[:, :2] ** 2
            )
        )


        exact_uvw2 += float(
            np.sum(
                exact ** 2
            )
        )


        with torch.enable_grad():

            _, div = (
                mod.physical_divergence_autograd(
                    model=model,
                    x_phys=xt,
                    create_graph=False,
                )
            )


        div2 += float(
            torch.sum(
                div.detach() ** 2
            ).cpu()
        )


    bottom2 = 0.0


    nb = Xb.shape[0]


    for start in range(
        0,
        nb,
        BATCH,
    ):

        stop = min(
            start + BATCH,
            nb,
        )


        xbt = torch.from_numpy(
            Xb[start:stop]
        ).to(
            device
        )


        with torch.no_grad():

            w = model(
                xbt
            )[:, 2]


        bottom2 += float(
            torch.sum(
                w * w
            ).cpu()
        )


    mse_uv = (
        sse_uv
        /
        (2.0 * n)
    )


    mse_uvw = (
        sse_uvw
        /
        (3.0 * n)
    )


    rel_uv = math.sqrt(

        sse_uv
        /
        (
            exact_uv2
            +
            1e-30
        )
    )


    rel_uvw = math.sqrt(

        sse_uvw
        /
        (
            exact_uvw2
            +
            1e-30
        )
    )


    rms_div = math.sqrt(
        div2 / n
    )


    rms_bottom = math.sqrt(
        bottom2 / nb
    )


    return {

        "mse_uv":
            mse_uv,

        "mse_uvw":
            mse_uvw,

        "relL2_uv":
            rel_uv,

        "relL2_uvw":
            rel_uvw,

        "rms_div":
            rms_div,

        "rms_bottom":
            rms_bottom,
    }


# ============================================================
# CSV helpers
# ============================================================

def write_csv(
    path,
    rows,
):

    if not rows:

        return


    with path.open(
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

        w.writerows(
            rows
        )


def mean_std_rows(
    rows,
    group_key=None,
):

    metrics = [

        "mse_uv",

        "mse_uvw",

        "relL2_uv",

        "relL2_uvw",

        "rms_div",

        "rms_bottom",
    ]


    if group_key is None:

        groups = [
            None
        ]

    else:

        groups = sorted(
            set(
                r[group_key]
                for r in rows
            )
        )


    out = []


    for group in groups:

        if group_key is None:

            subset = rows

        else:

            subset = [

                r

                for r in rows

                if r[group_key]
                ==
                group
            ]


        row = {

            "nSeeds":
                len(subset),

            "final_eval_seed":
                FINAL_EVAL_SEED,
        }


        if group_key is not None:

            row[group_key] = group


        for metric in metrics:

            vals = np.array(

                [
                    r[metric]
                    for r in subset
                ],

                dtype=float,
            )


            row[
                metric
                +
                "_mean"
            ] = float(
                vals.mean()
            )


            row[
                metric
                +
                "_std"
            ] = float(

                vals.std(
                    ddof=1
                )

                if len(vals) > 1

                else 0.0
            )


        out.append(
            row
        )


    return out


# ============================================================
# Main
# ============================================================

def main():

    torch.set_default_dtype(
        torch.float64
    )


    device = torch.device(

        "cuda:0"

        if torch.cuda.is_available()

        else "cpu"
    )


    mod = load_protocol_module(
        CODE
    )


    # Exact temporal benchmark
    p = SimpleNamespace(

        traj="parabolic",

        x0=0.0,

        y0=0.0,

        vx=20.0,

        vy=0.0,

        ax=0.0,

        ay=30.0,

        R=2.0,

        omega=2.0 * math.pi,

        phi=0.0,

        epsilon=0.02,
    )


    lower = np.array(
        [
            A,
            C,
            E,
            T0,
        ],
        dtype=np.float64,
    )


    upper = np.array(
        [
            B,
            D,
            F,
            T1,
        ],
        dtype=np.float64,
    )


    # ========================================================
    # VIRGIN random 4D test set
    # ========================================================

    rng_test = np.random.RandomState(
        FINAL_EVAL_SEED
    )


    Xtest = uniform_box(
        rng_test,
        N_TEST,
    )


    Xtest_bottom = bottom_points(
        rng_test,
        N_TEST_BOTTOM,
    )


    # ========================================================
    # Independent diagnostic time-slice points
    # ========================================================

    rng_slice = np.random.RandomState(
        FINAL_EVAL_SEED + 1
    )


    xyz_slice = np.column_stack(
        [

            rng_slice.uniform(
                A,
                B,
                N_SLICE,
            ),

            rng_slice.uniform(
                C,
                D,
                N_SLICE,
            ),

            rng_slice.uniform(
                E,
                F,
                N_SLICE,
            ),
        ]
    )


    xy_bottom = np.column_stack(
        [

            rng_slice.uniform(
                A,
                B,
                N_SLICE_BOTTOM,
            ),

            rng_slice.uniform(
                C,
                D,
                N_SLICE_BOTTOM,
            ),
        ]
    )


    runs = []

    time_runs = []


    print(
        "=" * 100
    )


    print(
        "VIRGIN FINAL SPACE-TIME TEST"
    )


    print(
        f"evaluation seed = "
        f"{FINAL_EVAL_SEED}"
    )


    print(
        f"Ntest={N_TEST}, "
        f"Nbottom={N_TEST_BOTTOM}"
    )


    print(
        f"Nslice={N_SLICE}, "
        f"Nslice_bottom={N_SLICE_BOTTOM}"
    )


    print(
        f"device={device}"
    )


    print(
        "=" * 100
    )


    # ========================================================
    # Evaluate frozen checkpoints
    # ========================================================

    for seed in [
        0,
        1,
        2,
    ]:

        checkpoint_path = (
            resolve_checkpoint(
                seed
            )
        )


        model = mod.RitzNet(

            4,

            3,

            WIDTH,

            DEPTH,

            lower,

            upper,

        ).to(
            device
        )


        checkpoint = (
            mod.safe_torch_load(

                str(
                    checkpoint_path
                ),

                device,
            )
        )


        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )


        model.eval()


        metrics = evaluate(

            mod,

            model,

            Xtest,

            Xtest_bottom,

            p,

            device,
        )


        row = {

            "seed":
                seed,

            "checkpoint_epoch":
                int(
                    checkpoint.get(
                        "epoch",
                        -1,
                    )
                ),

            "checkpoint":
                str(
                    checkpoint_path
                ),

            "final_eval_seed":
                FINAL_EVAL_SEED,

            "nTest":
                N_TEST,

            "nTestBottom":
                N_TEST_BOTTOM,

            **metrics,
        }


        runs.append(
            row
        )


        print(

            f"seed={seed} | "

            f"MSEuv="
            f"{metrics['mse_uv']:.6e} | "

            f"MSEuvw="
            f"{metrics['mse_uvw']:.6e} | "

            f"relL2="
            f"{metrics['relL2_uvw']:.6e} | "

            f"RMSdiv="
            f"{metrics['rms_div']:.6e} | "

            f"RMSb="
            f"{metrics['rms_bottom']:.6e}"
        )


        # ====================================================
        # Diagnostic time slices
        # ====================================================

        for t in EVAL_TIMES:

            Xslice = np.column_stack(
                [

                    xyz_slice,

                    np.full(
                        N_SLICE,
                        t,
                        dtype=np.float64,
                    ),
                ]
            )


            Xslice_bottom = np.column_stack(
                [

                    xy_bottom[:, 0],

                    xy_bottom[:, 1],

                    np.full(
                        N_SLICE_BOTTOM,
                        E,
                        dtype=np.float64,
                    ),

                    np.full(
                        N_SLICE_BOTTOM,
                        t,
                        dtype=np.float64,
                    ),
                ]
            )


            tm = evaluate(

                mod,

                model,

                Xslice,

                Xslice_bottom,

                p,

                device,
            )


            time_runs.append({

                "seed":
                    seed,

                "t":
                    t,

                "final_eval_seed":
                    FINAL_EVAL_SEED,

                "nSlice":
                    N_SLICE,

                "nSliceBottom":
                    N_SLICE_BOTTOM,

                **tm,
            })


    # ========================================================
    # Save individual runs
    # ========================================================

    write_csv(
        RUNS_CSV,
        runs,
    )


    write_csv(
        TIME_RUNS_CSV,
        time_runs,
    )


    # ========================================================
    # Overall mean/std
    # ========================================================

    mean_rows = mean_std_rows(
        runs
    )


    mean_rows[0][
        "nTest"
    ] = N_TEST


    mean_rows[0][
        "nTestBottom"
    ] = N_TEST_BOTTOM


    write_csv(
        MEAN_CSV,
        mean_rows,
    )


    # ========================================================
    # Mean/std by time
    # ========================================================

    time_mean = mean_std_rows(

        time_runs,

        group_key="t",
    )


    for row in time_mean:

        row[
            "nSlice"
        ] = N_SLICE

        row[
            "nSliceBottom"
        ] = N_SLICE_BOTTOM


    write_csv(
        TIME_MEAN_CSV,
        time_mean,
    )


    # ========================================================
    # Print final numbers
    # ========================================================

    mean = mean_rows[0]


    print()


    print(
        "=" * 100
    )


    print(
        "FINAL VIRGIN TEST"
    )


    print(
        "MEAN ± STD OVER 3 CHECKPOINTS"
    )


    print(
        "=" * 100
    )


    for key in [

        "mse_uv",

        "mse_uvw",

        "relL2_uvw",

        "rms_div",

        "rms_bottom",
    ]:

        print(

            f"{key:14s} = "

            f"{mean[key + '_mean']:.6e} "

            f"± "

            f"{mean[key + '_std']:.6e}"
        )


    print()


    print(
        "Generated:"
    )


    print(
        f"  {RUNS_CSV}"
    )


    print(
        f"  {MEAN_CSV}"
    )


    print(
        f"  {TIME_RUNS_CSV}"
    )


    print(
        f"  {TIME_MEAN_CSV}"
    )


    print(
        "=" * 100
    )


if __name__ == "__main__":

    main()

