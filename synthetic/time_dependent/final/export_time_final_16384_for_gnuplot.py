#!/usr/bin/env python3

import argparse
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


A, B = -25.0, 25.0
C, D = -25.0, 25.0
E, F = 0.0, 7.0
T0, T1 = -1.0, 1.0

WIDTH = 60
DEPTH = 2
NOBS = 16384

DEFAULT_TIMES = [-0.8, -0.2, 0.5, 0.8]
DEFAULT_Z = 1.56


CHECKPOINT_CANDIDATES = {
    0: [
        Path("out_time_obs_sweep/checkpoints/timeobs_n16384_s0_best.pt"),
        Path("out_time_final_16384/checkpoints/timefinal16384_s0_best.pt"),
    ],
    1: [
        Path("out_time_final_16384/checkpoints/timefinal16384_s1_best.pt")
    ],
    2: [
        Path("out_time_final_16384/checkpoints/timefinal16384_s2_best.pt")
    ],
}


def load_module(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Training module not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        "deepwind_time_final",
        path
    )

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def safe_load(mod, path, device):

    if hasattr(mod, "safe_torch_load"):
        return mod.safe_torch_load(
            str(path),
            device
        )

    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False
        )

    except TypeError:
        return torch.load(
            path,
            map_location=device
        )


def resolve_checkpoint(seed, override=None):

    if override is not None:

        p = Path(override)

        if not p.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {p}"
            )

        return p

    for p in CHECKPOINT_CANDIDATES[seed]:

        if p.exists():
            return p

    tried = "\n  ".join(
        str(p)
        for p in CHECKPOINT_CANDIDATES[seed]
    )

    raise FileNotFoundError(
        f"No checkpoint found for seed={seed}.\n"
        f"Tried:\n  {tried}"
    )


def predict(model, X, device, batch=4096):

    out = np.empty(
        (X.shape[0], 3),
        dtype=np.float64
    )

    model.eval()

    with torch.no_grad():

        for i in range(
            0,
            X.shape[0],
            batch
        ):

            j = min(
                i + batch,
                X.shape[0]
            )

            xt = torch.from_numpy(
                X[i:j]
            ).to(device)

            out[i:j] = (
                model(xt)
                .detach()
                .cpu()
                .numpy()
            )

    return out


def center(t, p):

    t = np.asarray(
        t,
        dtype=float
    )

    cx = (
        p.x0
        + p.vx*t
        + p.ax*t*t
    )

    cy = (
        p.y0
        + p.vy*t
        + p.ay*t*t
    )

    return cx, cy


def time_tag(t):

    sign = "m" if t < 0 else "p"

    n = int(
        round(
            abs(t)*10
        )
    )

    return f"t{sign}{n:02d}"


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--module",
        default="deepWind3D_vortex_time_final_protocol.py"
    )

    ap.add_argument(
        "--outdir",
        default="gnuplot_time_final"
    )

    ap.add_argument(
        "--grid",
        type=int,
        default=15
    )

    ap.add_argument(
        "--zslice",
        type=float,
        default=DEFAULT_Z
    )

    ap.add_argument(
        "--times",
        nargs="+",
        type=float,
        default=DEFAULT_TIMES
    )

    ap.add_argument(
        "--device",
        default=(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    ap.add_argument(
        "--ckpt0",
        default=None
    )

    ap.add_argument(
        "--ckpt1",
        default=None
    )

    ap.add_argument(
        "--ckpt2",
        default=None
    )

    ap.add_argument(
        "--cpercentile",
        type=float,
        default=98.0
    )

    args = ap.parse_args()

    if len(args.times) != 4:
        raise SystemExit(
            "Use exactly four values in --times."
        )

    torch.set_default_dtype(
        torch.float64
    )

    device = torch.device(
        args.device
    )

    mod = load_module(
        args.module
    )


    # --------------------------------------------
    # Final temporal benchmark parameters
    # --------------------------------------------

    p = SimpleNamespace(

        traj="parabolic",

        x0=0.0,
        y0=0.0,

        vx=20.0,
        vy=0.0,

        ax=0.0,
        ay=30.0,

        R=2.0,

        omega=2.0*math.pi,

        phi=0.0,

        epsilon=0.02
    )


    lower = np.array(
        [A, C, E, T0],
        dtype=np.float64
    )

    upper = np.array(
        [B, D, F, T1],
        dtype=np.float64
    )


    overrides = {
        0: args.ckpt0,
        1: args.ckpt1,
        2: args.ckpt2
    }


    models = []
    meta = []


    # --------------------------------------------
    # Load checkpoints
    # --------------------------------------------

    for seed in [0, 1, 2]:

        ckpt_path = resolve_checkpoint(
            seed,
            overrides[seed]
        )

        ckpt = safe_load(
            mod,
            ckpt_path,
            device
        )

        model = mod.RitzNet(
            4,
            3,
            WIDTH,
            DEPTH,
            lower,
            upper
        ).to(device)

        model.load_state_dict(
            ckpt["model_state_dict"]
        )

        model.eval()

        models.append(
            model
        )

        meta.append(
            (
                seed,
                ckpt_path,
                int(
                    ckpt.get(
                        "epoch",
                        -1
                    )
                )
            )
        )


    print(
        "\nFrozen checkpoints:"
    )

    for seed, pth, epoch in meta:

        print(
            f" seed={seed}: "
            f"{pth} "
            f"(epoch={epoch})"
        )


    # --------------------------------------------
    # Output directory
    # --------------------------------------------

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------
    # Vector grid
    # --------------------------------------------

    xv = np.linspace(
        A,
        B,
        args.grid
    )

    yv = np.linspace(
        C,
        D,
        args.grid
    )

    Xg, Yg = np.meshgrid(
        xv,
        yv,
        indexing="xy"
    )

    n = Xg.size


    all_errors = []
    tags = []


    # --------------------------------------------
    # Evaluate four times
    # --------------------------------------------

    for t in args.times:

        tag = time_tag(
            t
        )

        tags.append(
            tag
        )

        XYZT = np.column_stack(
            [
                Xg.ravel(),
                Yg.ravel(),

                np.full(
                    n,
                    args.zslice,
                    dtype=np.float64
                ),

                np.full(
                    n,
                    t,
                    dtype=np.float64
                )
            ]
        )


        # Exact solution
        exact = mod.uexact_time_np(
            XYZT,
            p
        )


        # Prediction from each seed
        pred3 = np.stack(
            [
                predict(
                    m,
                    XYZT,
                    device
                )
                for m in models
            ],
            axis=0
        )


        # Ensemble mean
        pred = pred3.mean(
            axis=0
        )


        # Local full-field error
        err = np.linalg.norm(
            pred - exact,
            axis=1
        )


        all_errors.append(
            err
        )


        # ----------------------------------------
        # Gnuplot data
        #
        # columns:
        # 1 x
        # 2 y
        # 3 u_pred
        # 4 v_pred
        # 5 local error
        # 6 u_exact
        # 7 v_exact
        # 8 w_pred
        # 9 w_exact
        # ----------------------------------------

        data = np.column_stack(
            [
                Xg.ravel(),
                Yg.ravel(),

                pred[:, 0],
                pred[:, 1],

                err,

                exact[:, 0],
                exact[:, 1],

                pred[:, 2],
                exact[:, 2]
            ]
        )


        np.savetxt(
            outdir / f"field_{tag}.dat",
            data,
            fmt="%.10e",
            header=(
                "x y "
                "u_pred v_pred "
                "error "
                "u_exact v_exact "
                "w_pred w_exact"
            )
        )


        # Instantaneous center
        cx, cy = center(
            np.array([t]),
            p
        )


        np.savetxt(
            outdir / f"center_{tag}.dat",

            np.array(
                [
                    [
                        cx[0],
                        cy[0]
                    ]
                ]
            ),

            fmt="%.10e",

            header="cx cy"
        )


    # --------------------------------------------
    # Complete parabolic trajectory
    # --------------------------------------------

    tt = np.linspace(
        T0,
        T1,
        401
    )

    cx, cy = center(
        tt,
        p
    )


    np.savetxt(
        outdir / "trajectory.dat",

        np.column_stack(
            [
                tt,
                cx,
                cy
            ]
        ),

        fmt="%.10e",

        header="t cx cy"
    )


    # --------------------------------------------
    # Common error scale
    # --------------------------------------------

    err_all = np.concatenate(
        all_errors
    )


    cmax_auto = float(
        np.max(
            err_all
        )
    )


    pct = float(
        np.clip(
            args.cpercentile,
            50.0,
            100.0
        )
    )


    cmax_pct = float(
        np.percentile(
            err_all,
            pct
        )
    )


    if cmax_pct <= 0:

        if cmax_auto > 0:
            cmax_pct = cmax_auto
        else:
            cmax_pct = 1.0


    # --------------------------------------------
    # Parameters read by Gnuplot
    # --------------------------------------------

    params = (
        outdir
        / "gnuplot_params.gp"
    )


    with open(
        params,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"zslice = "
            f"{args.zslice:.10g}\n"
        )

        f.write(
            f"cmax_auto = "
            f"{cmax_auto:.12g}\n"
        )

        f.write(
            f"cmax_p98 = "
            f"{cmax_pct:.12g}\n"
        )

        f.write(
            "xmin_data = -25.0\n"
        )

        f.write(
            "xmax_data = 25.0\n"
        )

        f.write(
            "ymin_data = -25.0\n"
        )

        f.write(
            "ymax_data = 25.0\n"
        )


        for i, t in enumerate(
            args.times,
            1
        ):

            f.write(
                f"t{i} = {t:.10g}\n"
            )

            f.write(
                f'tag{i} = '
                f'"{time_tag(t)}"\n'
            )


    # --------------------------------------------
    # Record checkpoint provenance
    # --------------------------------------------

    with open(
        outdir / "checkpoints_used.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"Nobs = {NOBS}\n"
        )

        f.write(
            f"zslice = "
            f"{args.zslice}\n"
        )

        f.write(
            f"times = "
            f"{args.times}\n"
        )

        f.write(
            f"cmax_auto = "
            f"{cmax_auto:.12g}\n"
        )

        f.write(
            f"cmax_p{pct:g} = "
            f"{cmax_pct:.12g}\n"
        )


        for seed, pth, epoch in meta:

            f.write(
                f"seed {seed}: "
                f"{pth}; "
                f"epoch={epoch}\n"
            )


    print(
        "\nWrote:"
    )

    for tag in tags:

        print(
            f"  field_{tag}.dat"
        )

        print(
            f"  center_{tag}.dat"
        )


    print(
        "  trajectory.dat"
    )

    print(
        "  gnuplot_params.gp"
    )

    print(
        "  checkpoints_used.txt"
    )


    print(
        "\nError scale:"
    )

    print(
        f"  max = {cmax_auto:.6g}"
    )

    print(
        f"  p{pct:g} = "
        f"{cmax_pct:.6g}"
    )

    print(
        "\nNow run:"
    )

    print(
        "gnuplot "
        "plot_time_final_16384.gp"
    )


if __name__ == "__main__":
    main()
