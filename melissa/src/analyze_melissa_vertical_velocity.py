#!/usr/bin/env python3
"""
Analyze the reconstructed vertical velocity for the final Hurricane Melissa run.

Default model: representative seed 2345.

This script performs TWO complementary kinematic checks.

1) Internal mass-consistency check
   --------------------------------
   From the neural horizontal field (u_theta, v_theta), compute

       w_kin,model(z) = - integral_0^z (du_theta/dxi + dv_theta/deta) dzeta.

   Since the network is only approximately divergence-free, compare both

       w_theta(z)                         vs w_kin,model(z)

   and the boundary-corrected increment

       w_theta(z) - w_theta(0)            vs w_kin,model(z).

   The latter isolates the continuity consistency from the small residual
   lower-boundary error.

2) Observation-based kinematic check
   ----------------------------------
   At each REAL hourly anchor, use the processed observed horizontal fields
   U_obs(xi,eta,z), V_obs(xi,eta,z), compute horizontal derivatives by centered
   finite differences on the common metric grid, and form

       Delta w_kin,obs(z)
         = - integral_{z_min}^z (dU_obs/dxi + dV_obs/deta) dzeta,

   where z_min is the lowest observed height (10 m for the current dataset).
   Compare this with the neural vertical increment

       Delta w_theta(z) = w_theta(z) - w_theta(z_min).

   This comparison avoids inventing an unobserved absolute value of w at 10 m.
   The absolute lower-boundary condition w_theta(z=0) ~ 0 is reported separately.

Important scientific interpretation
-----------------------------------
The observation-based comparison is a kinematic compatibility check, NOT an
independent validation of atmospheric vertical velocity: Melissa has no observed
w truth in this dataset.

Outputs
-------
  vertical_velocity_metrics_by_time.csv
  vertical_velocity_summary.json
  peak_time_fields.npz
  peak_wtheta_xz.png
  peak_wkin_model_xz.png
  peak_wtheta_increment_obslevels_xz.png
  peak_wkin_obs_increment_xz.png
  peak_scatter_internal_consistency.png
  peak_scatter_observation_kinematic.png
  time_series_vertical_metrics.png
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

from train_melissa_reconstruction_final import (
    ResidualNet,
    StormFrame,
    Scaling,
    TemporalVelocityScale,
)


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)))


def rel_l2(a, b, eps=1e-30):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.sum((a - b) ** 2) / (np.sum(b ** 2) + eps)))


def corrcoef_safe(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    sa = np.std(a)
    sb = np.std(b)
    if sa < 1e-15 or sb < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load_state_dict_compat(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_storm_grid(xi_m, eta_m, z_m, t_norm, frame):
    """Return normalized storm-relative NN coordinates and shape (nz,ny,nx)."""
    Z, ETA, XI = np.meshgrid(z_m, eta_m, xi_m, indexing="ij")

    xihat = (XI - frame.xi_min) / frame.Lxi
    etahat = (ETA - frame.eta_min) / frame.Leta
    zhat = Z / frame.Lz
    that = np.full_like(zhat, float(t_norm))

    X = np.column_stack([
        xihat.ravel(),
        etahat.ravel(),
        zhat.ravel(),
        that.ravel(),
    ]).astype(np.float32)

    return X, Z.shape


def evaluate_model_with_derivatives(
    model, X_np, shape, t_norm, frame, scaling, tscale, device, batch_size
):
    """
    Evaluate physical u,v,w, horizontal divergence, and full divergence.

    Network outputs are u_bar, v_bar, w_bar. Physical quantities are

        u = S(t) u_bar
        v = S(t) v_bar
        w = S(t) Lz/Lh w_bar.
    """
    S = float(tscale.S_phys(np.asarray([t_norm]))[0])
    Sw = S * scaling.Lz / scaling.Lh

    n = len(X_np)
    u_all = np.empty(n, dtype=np.float64)
    v_all = np.empty(n, dtype=np.float64)
    w_all = np.empty(n, dtype=np.float64)
    hdiv_all = np.empty(n, dtype=np.float64)
    div_all = np.empty(n, dtype=np.float64)

    model.eval()

    for a in range(0, n, batch_size):
        b = min(a + batch_size, n)
        X = torch.from_numpy(X_np[a:b]).to(device)
        X = X.detach().clone().requires_grad_(True)

        pred = model(X)

        gu = torch.autograd.grad(
            pred[:, 0].sum(), X, create_graph=False, retain_graph=True
        )[0]
        gv = torch.autograd.grad(
            pred[:, 1].sum(), X, create_graph=False, retain_graph=True
        )[0]
        gw = torch.autograd.grad(
            pred[:, 2].sum(), X, create_graph=False, retain_graph=False
        )[0]

        p = pred.detach().cpu().numpy()
        gu = gu.detach().cpu().numpy()
        gv = gv.detach().cpu().numpy()
        gw = gw.detach().cpu().numpy()

        u = S * p[:, 0]
        v = S * p[:, 1]
        w = Sw * p[:, 2]

        du_dxi = (S / scaling.Lxi) * gu[:, 0]
        dv_deta = (S / scaling.Leta) * gv[:, 1]
        dw_dz = (S / scaling.Lh) * gw[:, 2]

        hdiv = du_dxi + dv_deta
        div = hdiv + dw_dz

        u_all[a:b] = u
        v_all[a:b] = v
        w_all[a:b] = w
        hdiv_all[a:b] = hdiv
        div_all[a:b] = div

    return {
        "u": u_all.reshape(shape),
        "v": v_all.reshape(shape),
        "w": w_all.reshape(shape),
        "hdiv": hdiv_all.reshape(shape),
        "div": div_all.reshape(shape),
    }


def observation_kinematic_increment(U, V, xi_m, eta_m, z_m):
    """
    Compute -int_{z_min}^z (dU/dxi + dV/deta) dz using the processed REAL field.

    U,V shape: (nz, ny, nx)
    """
    dU_dxi = np.gradient(U.astype(np.float64), xi_m, axis=2, edge_order=2)
    dV_deta = np.gradient(V.astype(np.float64), eta_m, axis=1, edge_order=2)
    hdiv = dU_dxi + dV_deta

    w_inc = -cumulative_trapezoid(hdiv, z_m, axis=0, initial=0.0)
    return hdiv, w_inc


def save_xz_heatmap(path, xi_m, z_m, F_zx, title, cbar_label):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    im = ax.pcolormesh(
        xi_m / 1000.0,
        z_m,
        F_zx,
        shading="auto",
        cmap="coolwarm",
    )
    ax.set_xlabel(r"Storm-relative $\xi$ (km)")
    ax.set_ylabel("Height z (m)")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_scatter(path, x, y, xlabel, ylabel, title):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    # Deterministic thinning for plotting only.
    if len(x) > 60000:
        idx = np.linspace(0, len(x) - 1, 60000, dtype=int)
        x = x[idx]
        y = y[idx]

    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.scatter(x, y, s=4, alpha=0.3)
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        default="melissa_reconstruction_allobs_prepared_5min",
    )
    ap.add_argument(
        "--model_dir",
        default="melissa_reconstruction_final/seed_2345",
    )
    ap.add_argument(
        "--outdir",
        default="melissa_vertical_analysis/seed_2345",
    )
    ap.add_argument(
        "--model_file",
        default="final_model.pt",
        help="Checkpoint inside model_dir. Final protocol uses final_model.pt.",
    )
    ap.add_argument("--batch_size", type=int, default=65536)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    manifest_path = os.path.join(args.data_dir, "manifest.json")
    hourly_path = os.path.join(args.data_dir, "train_hourly_processed.npz")
    summary_path = os.path.join(args.model_dir, "summary.json")
    model_path = os.path.join(args.model_dir, args.model_file)

    for p in [manifest_path, hourly_path, summary_path, model_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    with open(summary_path, "r") as f:
        model_summary = json.load(f)

    # Safety: this analysis is intended for the final all-observed protocol.
    if int(manifest.get("n_power_snapshots_available", -1)) != int(
        manifest.get("n_train_snapshots", -2)
    ):
        raise RuntimeError("Prepared dataset is not the all-observed reconstruction dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    if device.type == "cuda":
        print(f"GPU = {torch.cuda.get_device_name(0)}")

    frame = StormFrame(args.data_dir, manifest)
    scaling = Scaling(manifest, frame)
    tscale = TemporalVelocityScale(args.data_dir, manifest)

    width = int(model_summary["architecture"]["width"])
    depth = int(model_summary["architecture"]["depth"])

    model = ResidualNet(width=width, depth=depth).to(device)
    model.load_state_dict(load_state_dict_compat(model_path, device))
    model.eval()

    d = np.load(hourly_path, allow_pickle=False)
    hours = np.asarray(d["hours"], dtype=np.float64)
    keys = [str(x) for x in d["keys"].tolist()]
    xi_m = np.asarray(d["xi_grid_m"], dtype=np.float64)
    eta_m = np.asarray(d["eta_grid_m"], dtype=np.float64)
    z_obs = np.asarray(d["z_grid_m"], dtype=np.float64)
    Uobs = np.asarray(d["U_phys_mps"], dtype=np.float32)
    Vobs = np.asarray(d["V_phys_mps"], dtype=np.float32)

    total_hours = float(manifest["domain"]["total_hours"])
    t_real = hours / total_hours

    if len(hours) != 61:
        print(f"WARNING: expected 61 REAL hourly anchors, found {len(hours)}")

    # Include the physical lower boundary z=0 for the internal continuity check.
    if abs(z_obs[0]) < 1e-12:
        z_model = z_obs.copy()
        obs_offset = 0
    else:
        z_model = np.concatenate([[0.0], z_obs])
        obs_offset = 1

    S_real = tscale.S_phys(t_real)
    peak_i = int(np.argmax(S_real))

    print("=" * 88)
    print("MELISSA VERTICAL-VELOCITY ANALYSIS")
    print("=" * 88)
    print(f"model       = {model_path}")
    print(f"seed        = {model_summary['seed']}")
    print(f"REAL times  = {len(hours)}")
    print(f"grid        = {len(xi_m)} x {len(eta_m)} x {len(z_obs)} observed z levels")
    print(f"peak S(t)   = {S_real[peak_i]:.6f} m/s at {keys[peak_i]}")
    print("=" * 88)

    rows = []
    peak_payload = None

    for it, (key, h, tn) in enumerate(zip(keys, hours, t_real)):
        X_np, shape = build_storm_grid(xi_m, eta_m, z_model, tn, frame)
        mf = evaluate_model_with_derivatives(
            model,
            X_np,
            shape,
            tn,
            frame,
            scaling,
            tscale,
            device,
            args.batch_size,
        )

        # ------------------------------------------------------------
        # 1) Internal neural-field kinematic reconstruction from z=0.
        # ------------------------------------------------------------
        wkin_model = -cumulative_trapezoid(
            mf["hdiv"], z_model, axis=0, initial=0.0
        )
        wtheta_bc = mf["w"] - mf["w"][0:1, :, :]

        # ------------------------------------------------------------
        # 2) Observation-based kinematic increment from z_min=10 m.
        # ------------------------------------------------------------
        obs_hdiv, wkin_obs_inc = observation_kinematic_increment(
            Uobs[it], Vobs[it], xi_m, eta_m, z_obs
        )

        wtheta_obs = mf["w"][obs_offset:obs_offset + len(z_obs)]
        wtheta_obs_inc = wtheta_obs - wtheta_obs[0:1, :, :]

        # Also compare the model horizontal divergence on the observed z levels.
        model_hdiv_obs = mf["hdiv"][obs_offset:obs_offset + len(z_obs)]
        wkin_model_inc_obs = -cumulative_trapezoid(
            model_hdiv_obs, z_obs, axis=0, initial=0.0
        )

        row = {
            "index": it,
            "key": key,
            "hours_from_start": float(h),
            "t_norm": float(tn),
            "S_t_mps": float(S_real[it]),

            "rms_wtheta_mps": rms(mf["w"]),
            "maxabs_wtheta_mps": float(np.max(np.abs(mf["w"]))),
            "rms_w_bottom_mps": rms(mf["w"][0]),
            "rms_div_model_sinv": rms(mf["div"]),

            # Internal continuity check, absolute and boundary-corrected.
            "rms_wkin_model_mps": rms(wkin_model),
            "rmse_wtheta_vs_wkin_model_mps": rms(mf["w"] - wkin_model),
            "rmse_wtheta_bc_vs_wkin_model_mps": rms(wtheta_bc - wkin_model),
            "relL2_wtheta_bc_vs_wkin_model": rel_l2(wtheta_bc, wkin_model),
            "corr_wtheta_bc_vs_wkin_model": corrcoef_safe(wtheta_bc, wkin_model),

            # Observation-based horizontal-divergence check above z_min.
            "rms_wkin_obs_increment_mps": rms(wkin_obs_inc),
            "rmse_wtheta_increment_vs_wkin_obs_mps": rms(
                wtheta_obs_inc - wkin_obs_inc
            ),
            "relL2_wtheta_increment_vs_wkin_obs": rel_l2(
                wtheta_obs_inc, wkin_obs_inc
            ),
            "corr_wtheta_increment_vs_wkin_obs": corrcoef_safe(
                wtheta_obs_inc, wkin_obs_inc
            ),

            # Useful diagnostic: same increment comparison using model u,v.
            "rmse_wtheta_increment_vs_wkin_model_mps": rms(
                wtheta_obs_inc - wkin_model_inc_obs
            ),
            "corr_wtheta_increment_vs_wkin_model": corrcoef_safe(
                wtheta_obs_inc, wkin_model_inc_obs
            ),
        }
        rows.append(row)

        print(
            f"[{it+1:02d}/{len(hours):02d}] {key} | "
            f"RMS(w)={row['rms_wtheta_mps']:.4e} m/s | "
            f"div={row['rms_div_model_sinv']:.3e} 1/s | "
            f"internal corr={row['corr_wtheta_bc_vs_wkin_model']:.4f} | "
            f"obs-kin corr={row['corr_wtheta_increment_vs_wkin_obs']:.4f}"
        )

        if it == peak_i:
            peak_payload = {
                "key": key,
                "hours": float(h),
                "t_norm": float(tn),
                "S_t_mps": float(S_real[it]),
                "fields": mf,
                "wkin_model": wkin_model,
                "wtheta_bc": wtheta_bc,
                "obs_hdiv": obs_hdiv,
                "wkin_obs_inc": wkin_obs_inc,
                "wtheta_obs_inc": wtheta_obs_inc,
                "wkin_model_inc_obs": wkin_model_inc_obs,
                "Uobs": Uobs[it].astype(np.float64),
                "Vobs": Vobs[it].astype(np.float64),
            }

    # ------------------------------------------------------------------
    # Save metrics table.
    # ------------------------------------------------------------------
    csv_path = os.path.join(args.outdir, "vertical_velocity_metrics_by_time.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate mean/std over the 61 REAL hourly anchors.
    metric_names = [
        k for k in rows[0].keys()
        if k not in {"index", "key", "hours_from_start", "t_norm"}
    ]

    stats = {}
    for name in metric_names:
        x = np.asarray([r[name] for r in rows], dtype=np.float64)
        finite = np.isfinite(x)
        stats[name] = {
            "mean": float(np.mean(x[finite])) if np.any(finite) else float("nan"),
            "std": float(np.std(x[finite], ddof=1)) if np.sum(finite) > 1 else 0.0,
            "min": float(np.min(x[finite])) if np.any(finite) else float("nan"),
            "max": float(np.max(x[finite])) if np.any(finite) else float("nan"),
        }

    summary = {
        "scientific_scope": (
            "kinematic consistency analysis of reconstructed w; not independent "
            "validation of atmospheric vertical velocity"
        ),
        "model_seed": int(model_summary["seed"]),
        "checkpoint": args.model_file,
        "n_real_hourly_times": len(rows),
        "observed_vertical_range_m": [float(z_obs[0]), float(z_obs[-1])],
        "bottom_boundary_m": 0.0,
        "internal_kinematic_definition": (
            "w_kin_model(z)=-int_0^z (du_theta/dxi+dv_theta/deta) dz"
        ),
        "observation_kinematic_definition": (
            "Delta w_kin_obs(z)=-int_zmin^z (dU_obs/dxi+dV_obs/deta) dz; "
            "compared with w_theta(z)-w_theta(zmin)"
        ),
        "peak_intensity_time": {
            "index": peak_i,
            "key": keys[peak_i],
            "hours_from_start": float(hours[peak_i]),
            "S_t_mps": float(S_real[peak_i]),
        },
        "statistics_over_real_hourly_times": stats,
    }

    json_path = os.path.join(args.outdir, "vertical_velocity_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Detailed peak-intensity output + paper-oriented figures.
    # ------------------------------------------------------------------
    if peak_payload is None:
        raise RuntimeError("Peak-time payload was not generated")

    P = peak_payload
    np.savez_compressed(
        os.path.join(args.outdir, "peak_time_fields.npz"),
        key=np.asarray(P["key"]),
        hours_from_start=np.asarray(P["hours"]),
        t_norm=np.asarray(P["t_norm"]),
        S_t_mps=np.asarray(P["S_t_mps"]),
        xi_m=xi_m,
        eta_m=eta_m,
        z_model_m=z_model,
        z_obs_m=z_obs,
        u_theta_mps=P["fields"]["u"],
        v_theta_mps=P["fields"]["v"],
        w_theta_mps=P["fields"]["w"],
        horizontal_divergence_theta_sinv=P["fields"]["hdiv"],
        divergence_theta_sinv=P["fields"]["div"],
        w_kin_model_mps=P["wkin_model"],
        w_theta_boundary_corrected_mps=P["wtheta_bc"],
        U_obs_mps=P["Uobs"],
        V_obs_mps=P["Vobs"],
        horizontal_divergence_obs_sinv=P["obs_hdiv"],
        w_kin_obs_increment_mps=P["wkin_obs_inc"],
        w_theta_increment_obslevels_mps=P["wtheta_obs_inc"],
        w_kin_model_increment_obslevels_mps=P["wkin_model_inc_obs"],
    )

    # eta=0 cut (or nearest available grid line).
    j0 = int(np.argmin(np.abs(eta_m)))
    eta0_km = eta_m[j0] / 1000.0
    key = P["key"]

    save_xz_heatmap(
        os.path.join(args.outdir, "peak_wtheta_xz.png"),
        xi_m,
        z_model,
        P["fields"]["w"][:, j0, :],
        f"Reconstructed vertical velocity $w_\\theta$ — {key}, eta={eta0_km:.1f} km",
        "w (m/s)",
    )

    save_xz_heatmap(
        os.path.join(args.outdir, "peak_wkin_model_xz.png"),
        xi_m,
        z_model,
        P["wkin_model"][:, j0, :],
        f"Kinematic $w$ from neural $u,v$ — {key}, eta={eta0_km:.1f} km",
        "w (m/s)",
    )

    save_xz_heatmap(
        os.path.join(args.outdir, "peak_wtheta_increment_obslevels_xz.png"),
        xi_m,
        z_obs,
        P["wtheta_obs_inc"][:, j0, :],
        f"Neural vertical increment above {z_obs[0]:.0f} m — {key}",
        "Delta w (m/s)",
    )

    save_xz_heatmap(
        os.path.join(args.outdir, "peak_wkin_obs_increment_xz.png"),
        xi_m,
        z_obs,
        P["wkin_obs_inc"][:, j0, :],
        f"Kinematic increment from observed $u,v$ — {key}",
        "Delta w (m/s)",
    )

    save_scatter(
        os.path.join(args.outdir, "peak_scatter_internal_consistency.png"),
        P["wkin_model"],
        P["wtheta_bc"],
        r"$w_{kin,model}$ (m/s)",
        r"$w_\theta-w_\theta(z=0)$ (m/s)",
        f"Internal mass-consistency check — {key}",
    )

    save_scatter(
        os.path.join(args.outdir, "peak_scatter_observation_kinematic.png"),
        P["wkin_obs_inc"],
        P["wtheta_obs_inc"],
        r"$\Delta w_{kin,obs}$ (m/s)",
        r"$\Delta w_\theta$ (m/s)",
        f"Observation-based kinematic check — {key}",
    )

    # Time series of the key quantities.
    hh = np.asarray([r["hours_from_start"] for r in rows])
    w_rms = np.asarray([r["rms_wtheta_mps"] for r in rows])
    rmse_obs = np.asarray([
        r["rmse_wtheta_increment_vs_wkin_obs_mps"] for r in rows
    ])
    rmse_internal = np.asarray([
        r["rmse_wtheta_bc_vs_wkin_model_mps"] for r in rows
    ])

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(hh, w_rms, label=r"RMS($w_\theta$)")
    ax.plot(hh, rmse_internal, label="Internal kinematic RMSE")
    ax.plot(hh, rmse_obs, label="Observation-kinematic increment RMSE")
    ax.set_xlabel("Hours from first observed snapshot")
    ax.set_ylabel("Velocity scale (m/s)")
    ax.set_title("Vertical-velocity diagnostics across observed hourly fields")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "time_series_vertical_metrics.png"), dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Console summary.
    # ------------------------------------------------------------------
    S = summary["statistics_over_real_hourly_times"]

    print()
    print("=" * 88)
    print("VERTICAL-VELOCITY SUMMARY OVER REAL HOURLY ANCHORS")
    print("=" * 88)
    print(
        "RMS(w_theta)                    = "
        f"{S['rms_wtheta_mps']['mean']:.6e} +/- {S['rms_wtheta_mps']['std']:.3e} m/s"
    )
    print(
        "RMS(div_theta)                  = "
        f"{S['rms_div_model_sinv']['mean']:.6e} +/- {S['rms_div_model_sinv']['std']:.3e} 1/s"
    )
    print(
        "RMS(w_theta at z=0)             = "
        f"{S['rms_w_bottom_mps']['mean']:.6e} +/- {S['rms_w_bottom_mps']['std']:.3e} m/s"
    )
    print()
    print("Internal neural-field continuity check:")
    print(
        "  RMSE[ w_theta-w_theta(0), w_kin_model ] = "
        f"{S['rmse_wtheta_bc_vs_wkin_model_mps']['mean']:.6e} +/- "
        f"{S['rmse_wtheta_bc_vs_wkin_model_mps']['std']:.3e} m/s"
    )
    print(
        "  correlation                              = "
        f"{S['corr_wtheta_bc_vs_wkin_model']['mean']:.6f} +/- "
        f"{S['corr_wtheta_bc_vs_wkin_model']['std']:.3e}"
    )
    print()
    print("Observation-based kinematic increment check (from lowest observed z):")
    print(
        "  RMSE[ Delta w_theta, Delta w_kin_obs ]   = "
        f"{S['rmse_wtheta_increment_vs_wkin_obs_mps']['mean']:.6e} +/- "
        f"{S['rmse_wtheta_increment_vs_wkin_obs_mps']['std']:.3e} m/s"
    )
    print(
        "  correlation                              = "
        f"{S['corr_wtheta_increment_vs_wkin_obs']['mean']:.6f} +/- "
        f"{S['corr_wtheta_increment_vs_wkin_obs']['std']:.3e}"
    )
    print()
    print("IMPORTANT: the observation-based result is a kinematic compatibility check,")
    print("not an independent validation of true atmospheric vertical velocity.")
    print()
    print(f"Peak-intensity snapshot = {keys[peak_i]}")
    print(f"CSV  = {csv_path}")
    print(f"JSON = {json_path}")
    print(f"Plots/data saved in {args.outdir}")
    print("=" * 88)


if __name__ == "__main__":
    main()
