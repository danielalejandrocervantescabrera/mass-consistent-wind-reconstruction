#!/usr/bin/env python3
"""
Generate publication-ready figures for the Hurricane Melissa section.

Run from the melissa/ directory.

Figures produced
----------------
1) fig_melissa_horizontal_context.png
   Processed horizontal wind observations at three real hourly anchors:
   12 h before peak, peak, and 12 h after peak, at the lowest observed level.

2) fig_melissa_vertical_kinematic.png
   Peak-intensity x-z section through eta ~= 0:
      (a) Delta w_theta from representative seed 2345
      (b) Delta w_kin,obs from processed observed u,v
      (c) Difference Delta w_theta - Delta w_kin,obs

3) fig_melissa_seed_stability.png
   Peak-intensity x-z section through eta ~= 0:
      (a) three-seed ensemble mean w
      (b) pointwise standard deviation sigma_w

4) fig_melissa_peak_scatter.png
   Scatter comparison of Delta w_theta and Delta w_kin,obs at the peak snapshot.

The observation-based kinematic field is NOT ground truth for atmospheric w.
It is a continuity-based compatibility diagnostic derived from processed u,v.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ======================================================================
# Network
# ======================================================================

class RitzNet(nn.Module):
    def __init__(self, width=80, depth=5):
        super().__init__()
        self.linear_in = nn.Linear(4, width)
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(width, width),
                nn.Linear(width, width),
            ])
            for _ in range(depth)
        ])
        self.linear_out = nn.Linear(width, 3)
        self.act = nn.Tanh()

    def forward(self, x):
        h = self.act(self.linear_in(x))
        for l1, l2 in self.blocks:
            r = self.act(l1(h))
            r = self.act(l2(r))
            h = h + r
        return self.linear_out(h)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def decode_key(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def load_state(path, device):
    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location=device)

    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    elif isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]

    if any(k.startswith("module.") for k in obj.keys()):
        obj = {k.replace("module.", "", 1): v for k, v in obj.items()}

    return obj


def correlation(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    da = a - np.mean(a)
    db = b - np.mean(b)
    den = np.sqrt(np.sum(da * da) * np.sum(db * db))
    if den <= 1e-30:
        return float("nan")
    return float(np.sum(da * db) / den)


def rmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ======================================================================
# Coordinates and prediction
# ======================================================================

def build_input(xi, eta, z, t_norm, xi_min, xi_max, eta_min, eta_max, Lz):
    Z, ETA, XI = np.meshgrid(z, eta, xi, indexing="ij")

    xihat = (XI.ravel() - xi_min) / (xi_max - xi_min)
    etahat = (ETA.ravel() - eta_min) / (eta_max - eta_min)
    zhat = Z.ravel() / Lz
    that = np.full(XI.size, float(t_norm), dtype=np.float64)

    return np.column_stack([xihat, etahat, zhat, that]).astype(np.float32)


@torch.no_grad()
def predict_w(model, X, Sw, device, batch_size=65536):
    out = np.empty(len(X), dtype=np.float32)
    for a in range(0, len(X), batch_size):
        b = min(a + batch_size, len(X))
        xb = torch.from_numpy(X[a:b]).to(device)
        out[a:b] = model(xb)[:, 2].detach().cpu().numpy()
    return Sw * out


# ======================================================================
# Kinematic vertical increment from observed u,v
# ======================================================================

def horizontal_divergence(U, V, xi, eta):
    """
    U,V shape = (nz, ny, nx)
    xi axis = 2, eta axis = 1.
    """
    du_dxi = np.gradient(U, xi, axis=2, edge_order=2)
    dv_deta = np.gradient(V, eta, axis=1, edge_order=2)
    return du_dxi + dv_deta


def integrate_w_increment(hdiv, z):
    """
    Delta w(z) = - int_{z_min}^z hdiv dz.
    First observed level has Delta w = 0.
    """
    out = np.zeros_like(hdiv, dtype=np.float64)
    for k in range(1, len(z)):
        dz = float(z[k] - z[k - 1])
        out[k] = out[k - 1] - 0.5 * dz * (hdiv[k] + hdiv[k - 1])
    return out


# ======================================================================
# Plot helpers
# ======================================================================

def add_panel_label(ax, label):
    ax.text(
        0.02, 0.97, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11, fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2.0),
    )


def symmetric_limit(*arrays, percentile=99.0):
    vals = np.concatenate([np.abs(np.asarray(a).ravel()) for a in arrays])
    vmax = float(np.percentile(vals, percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(vals)) if vals.size else 1.0
    return max(vmax, 1e-12)


def select_nearest_hour(hours, target):
    return int(np.argmin(np.abs(hours - target)))


# ======================================================================
# Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data_dir",
        default="data/processed",
    )
    ap.add_argument(
        "--model_root",
        default="outputs/split_protocol/models",
    )
    ap.add_argument(
        "--stability_npz",
        default="outputs/split_protocol/analysis/seed_stability/w_fields_all_real_times.npz",
    )
    ap.add_argument(
        "--model_file", default="best_model.pt",
        help="Validation-selected checkpoint for the split protocol",
    )
    ap.add_argument(
        "--representative_seed",
        type=int,
        default=2345,
    )
    ap.add_argument(
        "--outdir",
        default="outputs/split_protocol/figures",
    )
    ap.add_argument(
        "--context_offset_hours",
        type=float,
        default=12.0,
        help="Horizontal-context panels use peak +/- this many hours.",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=65536,
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    manifest = read_json(os.path.join(args.data_dir, "manifest.json"))
    d = np.load(
        os.path.join(args.data_dir, "train_hourly_processed.npz"),
        allow_pickle=False,
    )

    keys = [decode_key(x) for x in d["keys"]]
    hours = np.asarray(d["hours"], dtype=np.float64)
    U = np.asarray(d["U_phys_mps"], dtype=np.float64)
    V = np.asarray(d["V_phys_mps"], dtype=np.float64)
    xi = np.asarray(d["xi_grid_m"], dtype=np.float64)
    eta = np.asarray(d["eta_grid_m"], dtype=np.float64)
    z = np.asarray(d["z_grid_m"], dtype=np.float64)

    nt, nz, ny, nx = U.shape

    common = manifest["storm_relative_common_support"]
    domain = manifest["domain"]

    xi_min = float(common["xi_min_m"])
    xi_max = float(common["xi_max_m"])
    eta_min = float(common["eta_min_m"])
    eta_max = float(common["eta_max_m"])

    Lxi = xi_max - xi_min
    Leta = eta_max - eta_min
    Lz = float(domain["Lz_m"])
    Lh = float(np.sqrt(Lxi * Leta))
    total_hours = float(domain["total_hours"])

    axes = tuple(range(1, U.ndim))
    S_real = np.maximum(
        np.max(np.abs(U), axis=axes),
        np.max(np.abs(V), axis=axes),
    )

    peak_idx = int(np.argmax(S_real))
    peak_key = keys[peak_idx]
    peak_hour = float(hours[peak_idx])
    peak_t = peak_hour / total_hours

    print(f"Peak snapshot = {peak_key}, hour={peak_hour:.2f}, S={S_real[peak_idx]:.6f} m/s")

    # ------------------------------------------------------------------
    # Representative model and w_theta at peak
    # ------------------------------------------------------------------
    seed_dir = os.path.join(args.model_root, f"seed_{args.representative_seed}")
    summary = read_json(os.path.join(seed_dir, "summary.json"))
    width = int(summary["architecture"]["width"])
    depth = int(summary["architecture"]["depth"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RitzNet(width=width, depth=depth).to(device)
    model.load_state_dict(
        load_state(os.path.join(seed_dir, args.model_file), device)
    )
    model.eval()

    Xpeak = build_input(
        xi, eta, z, peak_t,
        xi_min, xi_max, eta_min, eta_max, Lz,
    )

    Sw_peak = float(S_real[peak_idx] * Lz / Lh)
    wtheta = predict_w(
        model,
        Xpeak,
        Sw_peak,
        device,
        args.batch_size,
    ).reshape(nz, ny, nx)

    # Increment from lowest observed level, z_min = 10 m.
    delta_wtheta = wtheta - wtheta[0:1, :, :]

    # ------------------------------------------------------------------
    # Observation-based kinematic increment at peak
    # ------------------------------------------------------------------
    hdiv_obs = horizontal_divergence(U[peak_idx], V[peak_idx], xi, eta)
    delta_wkin_obs = integrate_w_increment(hdiv_obs, z)
    delta_error = delta_wtheta - delta_wkin_obs

    rho_peak = correlation(delta_wtheta, delta_wkin_obs)
    rmse_peak = rmse(delta_wtheta, delta_wkin_obs)

    print(f"Peak kinematic correlation = {rho_peak:.6f}")
    print(f"Peak kinematic RMSE        = {rmse_peak:.6e} m/s")

    # eta slice closest to storm center.
    j0 = int(np.argmin(np.abs(eta)))
    eta0_km = eta[j0] / 1000.0

    dwt_xz = delta_wtheta[:, j0, :]
    dwk_xz = delta_wkin_obs[:, j0, :]
    err_xz = delta_error[:, j0, :]

    # ------------------------------------------------------------------
    # FIGURE 1: horizontal context, peak +/- offset
    # ------------------------------------------------------------------
    offset = float(args.context_offset_hours)
    idx_before = select_nearest_hour(hours, peak_hour - offset)
    idx_after = select_nearest_hour(hours, peak_hour + offset)
    context_indices = [idx_before, peak_idx, idx_after]

    zidx = 0  # lowest observed level
    speed_context = [
        np.sqrt(U[i, zidx]**2 + V[i, zidx]**2)
        for i in context_indices
    ]
    vmax_speed = max(float(np.max(s)) for s in speed_context)

    fig, axes_plot = plt.subplots(1, 3, figsize=(14.0, 4.3), constrained_layout=True)

    for pidx, (ax, it) in enumerate(zip(axes_plot, context_indices)):
        speed = np.sqrt(U[it, zidx]**2 + V[it, zidx]**2)
        im = ax.pcolormesh(
            xi / 1000.0,
            eta / 1000.0,
            speed,
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax_speed,
        )

        step = max(1, nx // 10)
        ax.quiver(
            xi[::step] / 1000.0,
            eta[::step] / 1000.0,
            U[it, zidx, ::step, ::step],
            V[it, zidx, ::step, ::step],
            angles="xy",
            scale_units="xy",
            scale=None,
            width=0.003,
        )

        ax.plot(0.0, 0.0, marker="+", markersize=10, markeredgewidth=2)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\xi$ [km]")
        if pidx == 0:
            ax.set_ylabel(r"$\eta$ [km]")
        ax.set_title(f"{keys[it]}\n$z={z[zidx]:.0f}$ m")
        add_panel_label(ax, f"({chr(97+pidx)})")

    cbar = fig.colorbar(im, ax=axes_plot, shrink=0.92, pad=0.02)
    cbar.set_label(r"horizontal speed [m s$^{-1}$]")

    fig.savefig(
        os.path.join(args.outdir, "fig_melissa_horizontal_context.png"),
        dpi=args.dpi,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(args.outdir, "fig_melissa_horizontal_context.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # FIGURE 2: vertical kinematic comparison
    # ------------------------------------------------------------------
    main_lim = symmetric_limit(dwt_xz, dwk_xz, percentile=99.0)
    err_lim = symmetric_limit(err_xz, percentile=99.0)

    fig, axes_plot = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)

    im0 = axes_plot[0].pcolormesh(
        xi / 1000.0,
        z,
        dwt_xz,
        shading="auto",
        cmap="RdBu_r",
        vmin=-main_lim,
        vmax=main_lim,
    )
    axes_plot[0].set_title(r"$\Delta w_\theta$")
    axes_plot[0].set_ylabel(r"$z$ [m]")
    add_panel_label(axes_plot[0], "(a)")

    im1 = axes_plot[1].pcolormesh(
        xi / 1000.0,
        z,
        dwk_xz,
        shading="auto",
        cmap="RdBu_r",
        vmin=-main_lim,
        vmax=main_lim,
    )
    axes_plot[1].set_title(r"$\Delta w_{\mathrm{kin,obs}}$")
    add_panel_label(axes_plot[1], "(b)")

    im2 = axes_plot[2].pcolormesh(
        xi / 1000.0,
        z,
        err_xz,
        shading="auto",
        cmap="RdBu_r",
        vmin=-err_lim,
        vmax=err_lim,
    )
    axes_plot[2].set_title(r"$\Delta w_\theta-\Delta w_{\mathrm{kin,obs}}$")
    add_panel_label(axes_plot[2], "(c)")

    for ax in axes_plot:
        ax.set_xlabel(r"$\xi$ [km]")

    cb1 = fig.colorbar(im1, ax=axes_plot[:2], shrink=0.92, pad=0.02)
    cb1.set_label(r"vertical-velocity increment [m s$^{-1}$]")
    cb2 = fig.colorbar(im2, ax=axes_plot[2], shrink=0.92, pad=0.02)
    cb2.set_label(r"difference [m s$^{-1}$]")

    fig.suptitle(
        rf"Melissa {peak_key}, $\eta\approx{eta0_km:.1f}$ km; "
        rf"$\rho={rho_peak:.3f}$, RMSE={rmse_peak:.3e} m/s",
        fontsize=11,
    )

    fig.savefig(
        os.path.join(args.outdir, "fig_melissa_vertical_kinematic.png"),
        dpi=args.dpi,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(args.outdir, "fig_melissa_vertical_kinematic.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # FIGURE 3: seed stability at peak
    # ------------------------------------------------------------------
    if not os.path.exists(args.stability_npz):
        raise FileNotFoundError(
            f"Missing {args.stability_npz}. Run compare_melissa_w_across_seeds.py first."
        )

    stab = np.load(args.stability_npz, allow_pickle=False)
    mean_all = np.asarray(stab["w_ensemble_mean_mps"], dtype=np.float64)
    std_all = np.asarray(stab["w_ensemble_std_mps"], dtype=np.float64)

    if mean_all.shape[0] != nt:
        raise RuntimeError(
            f"Stability NPZ has {mean_all.shape[0]} times; prepared data has {nt}."
        )

    mean_xz = mean_all[peak_idx, :, j0, :]
    std_xz = std_all[peak_idx, :, j0, :]

    mean_lim = symmetric_limit(mean_xz, percentile=99.0)
    std_lim = float(np.percentile(std_xz, 99.0))
    std_lim = max(std_lim, 1e-12)

    fig, axes_plot = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True)

    im0 = axes_plot[0].pcolormesh(
        xi / 1000.0,
        z,
        mean_xz,
        shading="auto",
        cmap="RdBu_r",
        vmin=-mean_lim,
        vmax=mean_lim,
    )
    axes_plot[0].set_title(r"ensemble mean $\overline{w}$")
    axes_plot[0].set_xlabel(r"$\xi$ [km]")
    axes_plot[0].set_ylabel(r"$z$ [m]")
    add_panel_label(axes_plot[0], "(a)")

    im1 = axes_plot[1].pcolormesh(
        xi / 1000.0,
        z,
        std_xz,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=std_lim,
    )
    axes_plot[1].set_title(r"seed spread $\sigma_w$")
    axes_plot[1].set_xlabel(r"$\xi$ [km]")
    add_panel_label(axes_plot[1], "(b)")

    cb0 = fig.colorbar(im0, ax=axes_plot[0], shrink=0.92, pad=0.02)
    cb0.set_label(r"$\overline{w}$ [m s$^{-1}$]")
    cb1 = fig.colorbar(im1, ax=axes_plot[1], shrink=0.92, pad=0.02)
    cb1.set_label(r"$\sigma_w$ [m s$^{-1}$]")

    fig.suptitle(f"Melissa {peak_key}, three-seed stability", fontsize=11)

    fig.savefig(
        os.path.join(args.outdir, "fig_melissa_seed_stability.png"),
        dpi=args.dpi,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(args.outdir, "fig_melissa_seed_stability.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # FIGURE 4: peak scatter, using all grid points at observed levels
    # ------------------------------------------------------------------
    x = delta_wkin_obs.ravel()
    y = delta_wtheta.ravel()

    lim = symmetric_limit(x, y, percentile=99.5)

    fig, ax = plt.subplots(figsize=(5.3, 5.0), constrained_layout=True)
    ax.scatter(x, y, s=3, alpha=0.20)
    ax.plot([-lim, lim], [-lim, lim], linewidth=1.2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\Delta w_{\mathrm{kin,obs}}$ [m s$^{-1}$]")
    ax.set_ylabel(r"$\Delta w_\theta$ [m s$^{-1}$]")
    ax.set_title(
        f"Melissa {peak_key}: kinematic compatibility\n"
        rf"$\rho={rho_peak:.3f}$, RMSE={rmse_peak:.3e} m/s"
    )

    fig.savefig(
        os.path.join(args.outdir, "fig_melissa_peak_scatter.png"),
        dpi=args.dpi,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(args.outdir, "fig_melissa_peak_scatter.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # Save exact peak fields used in figures
    # ------------------------------------------------------------------
    np.savez_compressed(
        os.path.join(args.outdir, "melissa_peak_figure_data.npz"),
        key=np.asarray(peak_key),
        peak_index=np.asarray(peak_idx),
        xi_grid_m=xi,
        eta_grid_m=eta,
        z_grid_m=z,
        eta_slice_index=np.asarray(j0),
        eta_slice_m=np.asarray(eta[j0]),
        wtheta_mps=wtheta,
        delta_wtheta_mps=delta_wtheta,
        horizontal_divergence_obs_sinv=hdiv_obs,
        delta_wkin_obs_mps=delta_wkin_obs,
        delta_difference_mps=delta_error,
        correlation=np.asarray(rho_peak),
        rmse_mps=np.asarray(rmse_peak),
    )

    print()
    print("=" * 78)
    print("PAPER FIGURES GENERATED")
    print("=" * 78)
    print(os.path.join(args.outdir, "fig_melissa_horizontal_context.png"))
    print(os.path.join(args.outdir, "fig_melissa_vertical_kinematic.png"))
    print(os.path.join(args.outdir, "fig_melissa_seed_stability.png"))
    print(os.path.join(args.outdir, "fig_melissa_peak_scatter.png"))
    print(os.path.join(args.outdir, "melissa_peak_figure_data.npz"))
    print("PDF versions were also written for the manuscript.")
    print("=" * 78)


if __name__ == "__main__":
    main()
