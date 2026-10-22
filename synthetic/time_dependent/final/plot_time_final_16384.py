#!/usr/bin/env python3
"""Regenerate the synthetic time-dependent figure (former f05.eps)
from the frozen N_obs=16384 checkpoints.

The figure contains nine diagnostic time slices.  For each time, the
horizontal quiver is the ensemble-mean prediction over the three frozen
checkpoints and the background color is the pointwise full-field error
||u_pred - u_exact||_2 on the horizontal plane z = Z_SLICE.

Expected files (relative to the working directory):
  deepWind3D_vortex_time_final_protocol.py
  out_time_obs_sweep/checkpoints/timeobs_n16384_s0_best.pt
  out_time_final_16384/checkpoints/timefinal16384_s1_best.pt
  out_time_final_16384/checkpoints/timefinal16384_s2_best.pt

For seed 0, a checkpoint in out_time_final_16384 is also accepted.
"""

import argparse
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# -----------------------------------------------------------------------------
# Frozen experiment
# -----------------------------------------------------------------------------
A, B = -25.0, 25.0
C, D = -25.0, 25.0
E, F = 0.0, 7.0
T0, T1 = -1.0, 1.0

WIDTH = 60
DEPTH = 2
NOBS = 16384

EVAL_TIMES = [-1.0, -0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.0]

# Same representative interior plane used by the old figure description.
Z_SLICE = 1.56

CHECKPOINT_CANDIDATES = {
    0: [
        Path("out_time_obs_sweep/checkpoints/timeobs_n16384_s0_best.pt"),
        Path("out_time_final_16384/checkpoints/timefinal16384_s0_best.pt"),
    ],
    1: [
        Path("out_time_final_16384/checkpoints/timefinal16384_s1_best.pt"),
    ],
    2: [
        Path("out_time_final_16384/checkpoints/timefinal16384_s2_best.pt"),
    ],
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def load_protocol_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Training module not found: {path}")
    spec = importlib.util.spec_from_file_location("deepwind_time_protocol", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_checkpoint(seed: int) -> Path:
    for p in CHECKPOINT_CANDIDATES[seed]:
        if p.exists():
            return p
    tried = "\n  ".join(str(p) for p in CHECKPOINT_CANDIDATES[seed])
    raise FileNotFoundError(
        f"No checkpoint found for seed={seed}. Tried:\n  {tried}"
    )


def safe_load(mod, path: Path, device: torch.device):
    if hasattr(mod, "safe_torch_load"):
        return mod.safe_torch_load(str(path), device)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def predict(model, X: np.ndarray, device: torch.device, batch: int = 4096):
    out = np.empty((X.shape[0], 3), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, X.shape[0], batch):
            stop = min(start + batch, X.shape[0])
            xt = torch.from_numpy(X[start:stop]).to(device)
            out[start:stop] = model(xt).detach().cpu().numpy()
    return out


def center_trajectory(t, p):
    t = np.asarray(t, dtype=float)
    cx = p.x0 + p.vx * t + p.ax * t * t
    cy = p.y0 + p.vy * t + p.ay * t * t
    return cx, cy


def build_models(mod, device):
    lower = np.array([A, C, E, T0], dtype=np.float64)
    upper = np.array([B, D, F, T1], dtype=np.float64)

    models = []
    checkpoints = []
    for seed in (0, 1, 2):
        ckpt_path = resolve_checkpoint(seed)
        model = mod.RitzNet(4, 3, WIDTH, DEPTH, lower, upper).to(device)
        ckpt = safe_load(mod, ckpt_path, device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)
        checkpoints.append((seed, ckpt_path, int(ckpt.get("epoch", -1))))
    return models, checkpoints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--module",
        default="deepWind3D_vortex_time_final_protocol.py",
        help="Training/protocol module used to define RitzNet and the exact field.",
    )
    ap.add_argument(
        "--outbase",
        default="f05",
        help="Output basename without extension. Both .eps and .pdf are written.",
    )
    ap.add_argument("--grid", type=int, default=51, help="Grid points per x/y axis.")
    ap.add_argument(
        "--quiver-step", type=int, default=5, help="Subsampling step for quiver arrows."
    )
    ap.add_argument("--zslice", type=float, default=Z_SLICE)
    ap.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument(
        "--show-full-trajectory",
        action="store_true",
        help=(
            "Extend the y-axis to show the complete prescribed center trajectory. "
            "The reconstructed field itself is still evaluated only inside the frozen domain."
        ),
    )
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    mod = load_protocol_module(Path(args.module))

    # Exact benchmark parameters used by run_time_final_16384.sh
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

    models, checkpoints = build_models(mod, device)

    print("Frozen checkpoints:")
    for seed, path, epoch in checkpoints:
        print(f"  seed={seed}: {path} (epoch={epoch})")

    # Common slice grid for every time.
    xv = np.linspace(A, B, args.grid)
    yv = np.linspace(C, D, args.grid)
    Xg, Yg = np.meshgrid(xv, yv, indexing="xy")
    npts = Xg.size

    panel_data = []
    all_errors = []

    for t in EVAL_TIMES:
        X = np.column_stack(
            [
                Xg.ravel(),
                Yg.ravel(),
                np.full(npts, args.zslice, dtype=np.float64),
                np.full(npts, t, dtype=np.float64),
            ]
        )

        exact = mod.uexact_time_np(X, p)
        preds = np.stack([predict(m, X, device) for m in models], axis=0)
        pred_mean = preds.mean(axis=0)

        # Pointwise full-field reconstruction error of the ensemble-mean field.
        err = np.linalg.norm(pred_mean - exact, axis=1).reshape(Xg.shape)
        U = pred_mean[:, 0].reshape(Xg.shape)
        V = pred_mean[:, 1].reshape(Xg.shape)

        panel_data.append((t, U, V, err))
        all_errors.append(err.ravel())

    all_errors = np.concatenate(all_errors)
    err_min = 0.0
    err_max = float(np.max(all_errors))
    if not np.isfinite(err_max) or err_max <= 0.0:
        err_max = 1.0

    # Full prescribed center trajectory for reference.
    tt = np.linspace(T0, T1, 400)
    cx_curve, cy_curve = center_trajectory(tt, p)

    fig, axes = plt.subplots(3, 3, figsize=(10.2, 9.2), constrained_layout=False)
    axes = axes.ravel()

    mappable = None
    qstep = max(1, args.quiver_step)

    for ax, (t, U, V, err) in zip(axes, panel_data):
        mappable = ax.pcolormesh(
            Xg,
            Yg,
            err,
            shading="auto",
            vmin=err_min,
            vmax=err_max,
            rasterized=True,
        )

        # Horizontal reconstructed velocity field.
        ax.quiver(
            Xg[::qstep, ::qstep],
            Yg[::qstep, ::qstep],
            U[::qstep, ::qstep],
            V[::qstep, ::qstep],
            color="k",
            angles="xy",
            scale_units="xy",
            scale=None,
            width=0.0030,
            headwidth=3.0,
            headlength=4.0,
            headaxislength=3.5,
        )

        # Prescribed trajectory and instantaneous center.
        ax.plot(cx_curve, cy_curve, color="green", linewidth=1.4)
        cx, cy = center_trajectory(np.array([t]), p)
        ax.plot(cx[0], cy[0], "o", color="red", markersize=4.8)

        # Frozen spatial domain outline; useful when the full trajectory is shown.
        ax.add_patch(
            Rectangle(
                (A, C),
                B - A,
                D - C,
                fill=False,
                linestyle="--",
                linewidth=0.7,
                edgecolor="0.35",
            )
        )

        ax.set_xlim(A, B)
        if args.show_full_trajectory:
            ax.set_ylim(C, max(D, float(np.max(cy_curve)) + 2.0))
        else:
            ax.set_ylim(C, D)

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(rf"$t={t:g}$", fontsize=9)
        ax.tick_params(labelsize=7, length=2.5)
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)

    # Shared colorbar and compact spacing.
    fig.subplots_adjust(left=0.07, right=0.87, bottom=0.07, top=0.95, wspace=0.28, hspace=0.30)
    cax = fig.add_axes([0.90, 0.18, 0.018, 0.64])
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(r"Pointwise error $\|\mathbf{u}_{\theta}-\mathbf{u}^{\star}\|_2$", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        rf"Frozen temporal reconstruction ($N_{{\rm obs}}={NOBS}$, $z={args.zslice:g}$)",
        fontsize=10,
        y=0.985,
    )

    outbase = Path(args.outbase)
    outbase.parent.mkdir(parents=True, exist_ok=True)
    eps_path = outbase.with_suffix(".eps")
    pdf_path = outbase.with_suffix(".pdf")
    png_path = outbase.with_suffix(".png")

    fig.savefig(eps_path, format="eps", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("Generated:")
    print(f"  {eps_path}")
    print(f"  {pdf_path}")
    print(f"  {png_path}")


if __name__ == "__main__":
    main()
