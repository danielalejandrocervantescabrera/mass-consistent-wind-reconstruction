#!/usr/bin/env python3
"""
Final Hurricane Melissa space-time reconstruction experiment.

Scientific objective
--------------------
This is a reconstruction experiment, not a temporal-forecasting experiment.
All available hourly horizontal-wind observations are assimilated. Time t is an
input coordinate of a single continuous neural field

    (xi_hat, eta_hat, z_hat, t_hat) -> (u_bar, v_bar, w_bar),

where only u and v are supervised. The unobserved vertical component is inferred
through approximate mass consistency and the lower-boundary condition.

The prepared dataset is assumed to have been built with ALL observed hourly
snapshots (no DEV/VAL/TEST snapshot removal), using

    fixed metric frame -> storm-relative registration ->
    Wahba/thin-plate horizontal smoothing -> linear vertical interpolation ->
    PCHIP temporal densification.

Velocity normalization follows the useful scale convention of the original
Melissa implementation. For each observed hourly anchor,

    S_k = max(max |u_k|, max |v_k|),

and log(S_k) is interpolated with PCHIP to obtain a positive S(t). The network
therefore learns

    u_bar = u / S(t),  v_bar = v / S(t),

and physical reconstruction is

    u = S(t) u_bar,
    v = S(t) v_bar,
    w = S(t) (Lz/Lh) w_bar.

Because S(t) depends only on time, the spatial mass residual is

    r = (Lh/Lxi) d(u_bar)/d(xi_hat)
      + (Lh/Leta) d(v_bar)/d(eta_hat)
      + d(w_bar)/d(z_hat).

The training objective deliberately retains the simple structure of the original
method:

    J = J_uv + beta_div J_div + beta_bdy J_bdy,

with

    J_uv = J_real + alpha_interp J_pchip,
    J_div = mean(r^2),
    J_bdy = mean(w_bar^2 at z_hat=0).

There is no hybrid J_old/J_phys term and no lambda_phys parameter.

The PCHIP samples are auxiliary dense supervision derived from the observed
hourly fields; they are not treated as independent observations or as a temporal
test set. The default minibatch balances REAL hourly anchors and PCHIP samples
so temporal densification does not numerically overwhelm the observations.

No snapshot is withheld and no checkpoint is selected by a held-out temporal
field. Training uses a fixed epoch budget; the final-epoch model is the reported
model. Multiple runs should be launched from independent random seeds.
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import PchipInterpolator
from torch.optim.lr_scheduler import StepLR


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Neural network
# ============================================================

class ResidualNet(nn.Module):
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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.act(self.linear_in(x))
        for l1, l2 in self.blocks:
            r = self.act(l1(h))
            r = self.act(l2(r))
            h = h + r
        return self.linear_out(h)


# ============================================================
# Storm-relative coordinate transformation
# ============================================================

class StormFrame:
    def __init__(self, data_dir, manifest):
        domain = manifest["domain"]
        common = manifest["storm_relative_common_support"]

        self.xmin = float(domain["xmin_m"])
        self.ymin = float(domain["ymin_m"])
        self.Lx_earth = float(domain["Lx_m"])
        self.Ly_earth = float(domain["Ly_m"])
        self.Lz = float(domain["Lz_m"])

        self.xi_min = float(common["xi_min_m"])
        self.xi_max = float(common["xi_max_m"])
        self.eta_min = float(common["eta_min_m"])
        self.eta_max = float(common["eta_max_m"])

        self.Lxi = self.xi_max - self.xi_min
        self.Leta = self.eta_max - self.eta_min

        if self.Lxi <= 0 or self.Leta <= 0 or self.Lz <= 0:
            raise ValueError("Invalid physical dimensions in manifest.json")

        self.Lh = float(np.sqrt(self.Lxi * self.Leta))

        eye = np.load(os.path.join(data_dir, "eye_path.npz"))
        self.eye_t = np.asarray(eye["t_norm"], dtype=float)
        self.eye_x = np.asarray(eye["x_m"], dtype=float)
        self.eye_y = np.asarray(eye["y_m"], dtype=float)

        self.eye_x_of_t = PchipInterpolator(
            self.eye_t, self.eye_x, extrapolate=False
        )
        self.eye_y_of_t = PchipInterpolator(
            self.eye_t, self.eye_y, extrapolate=False
        )

    def earth_to_storm(self, X):
        """
        Input:
            X[:,0:4] = xhat_earth, yhat_earth, zhat, that
        Output:
            xihat, etahat, zhat, that
        """
        X = np.asarray(X, dtype=np.float32)

        x = self.xmin + X[:, 0].astype(np.float64) * self.Lx_earth
        y = self.ymin + X[:, 1].astype(np.float64) * self.Ly_earth
        t = X[:, 3].astype(np.float64)

        xc = np.asarray(self.eye_x_of_t(t), dtype=np.float64)
        yc = np.asarray(self.eye_y_of_t(t), dtype=np.float64)

        if not np.all(np.isfinite(xc)) or not np.all(np.isfinite(yc)):
            raise RuntimeError("Eye-path interpolation failed for some t values")

        xi = x - xc
        eta = y - yc

        xihat = (xi - self.xi_min) / self.Lxi
        etahat = (eta - self.eta_min) / self.Leta

        out = np.column_stack([
            xihat,
            etahat,
            X[:, 2],
            X[:, 3],
        ]).astype(np.float32)

        if (
            out[:, 0].min() < -2e-4
            or out[:, 0].max() > 1.0002
            or out[:, 1].min() < -2e-4
            or out[:, 1].max() > 1.0002
        ):
            raise RuntimeError(
                "Storm-relative coordinates left [0,1]: "
                f"xi=[{out[:,0].min():.6e},{out[:,0].max():.6e}], "
                f"eta=[{out[:,1].min():.6e},{out[:,1].max():.6e}]"
            )

        out[:, 0] = np.clip(out[:, 0], 0.0, 1.0)
        out[:, 1] = np.clip(out[:, 1], 0.0, 1.0)
        return out


# ============================================================
# Physical/geometric scaling
# ============================================================

class Scaling:
    def __init__(self, manifest, frame):
        # This is the preprocessing constant used to store train_Yuv.npy.
        self.Sglobal = float(
            manifest["velocity_normalization"]["S_global_mps"]
        )
        self.Lxi = frame.Lxi
        self.Leta = frame.Leta
        self.Lz = frame.Lz
        self.Lh = frame.Lh

        # Reference w scale corresponding to Sglobal.  Actual physical w uses
        # S(t)*Lz/Lh and is handled by TemporalVelocityScale.
        self.Sw_global = self.Sglobal * self.Lz / self.Lh

        self.cx = self.Lh / self.Lxi
        self.cy = self.Lh / self.Leta
        self.cz = 1.0


class TemporalVelocityScale:
    """
    S(t) constructed from all processed observed hourly anchors.

    S_k is computed in physical m/s from train_hourly_processed.npz and
    log(S_k) is interpolated with PCHIP.  The dimensionless ratio

        shat(t) = S(t) / S_global

    is what converts the existing globally normalized train_Yuv targets into
    temporally normalized targets.
    """

    def __init__(self, data_dir, manifest):
        path = os.path.join(data_dir, "train_hourly_processed.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Missing train_hourly_processed.npz; required to build TRAIN-only S(t)"
            )

        d = np.load(path)
        hours = np.asarray(d["hours"], dtype=np.float64)
        U = np.asarray(d["U_phys_mps"], dtype=np.float64)
        V = np.asarray(d["V_phys_mps"], dtype=np.float64)

        if U.shape != V.shape or U.shape[0] != len(hours):
            raise RuntimeError(
                f"Bad hourly processed shapes: hours={hours.shape}, U={U.shape}, V={V.shape}"
            )

        total_hours = float(manifest["domain"]["total_hours"])
        self.Sglobal = float(
            manifest["velocity_normalization"]["S_global_mps"]
        )

        self.hours_real = hours
        self.t_real = hours / total_hours

        # One positive scale per observed processed hourly anchor.
        axes = tuple(range(1, U.ndim))
        umax = np.max(np.abs(U), axis=axes)
        vmax = np.max(np.abs(V), axis=axes)
        self.S_real_phys = np.maximum(np.maximum(umax, vmax), 1e-8)
        self.shat_real = self.S_real_phys / self.Sglobal

        if not np.all(np.isfinite(self.S_real_phys)):
            raise RuntimeError("Non-finite values found in hourly velocity scales")
        if np.any(self.S_real_phys <= 0):
            raise RuntimeError("Temporal velocity scales must be positive")

        # log-PCHIP guarantees S(t) > 0.
        self._logS_of_t = PchipInterpolator(
            self.t_real,
            np.log(self.S_real_phys),
            extrapolate=False,
        )

        self.tmin = float(np.min(self.t_real))
        self.tmax = float(np.max(self.t_real))

    def S_phys(self, t):
        t = np.asarray(t, dtype=np.float64)
        vals = np.exp(self._logS_of_t(t))
        if not np.all(np.isfinite(vals)):
            raise RuntimeError(
                "S(t) requested outside the observed-anchor interval or interpolation failed"
            )
        return vals

    def shat(self, t):
        return self.S_phys(t) / self.Sglobal

    def Sw_phys(self, t, Lz, Lh):
        return self.S_phys(t) * (Lz / Lh)


# ============================================================
# Physics
# ============================================================

def divergence_residual(model, X, coeffs, create_graph=True):
    Xg = X.detach().clone().requires_grad_(True)
    pred = model(Xg)

    gu = torch.autograd.grad(
        pred[:, 0].sum(),
        Xg,
        create_graph=create_graph,
        retain_graph=True,
    )[0]

    gv = torch.autograd.grad(
        pred[:, 1].sum(),
        Xg,
        create_graph=create_graph,
        retain_graph=True,
    )[0]

    gw = torch.autograd.grad(
        pred[:, 2].sum(),
        Xg,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]

    r = (
        coeffs[0] * gu[:, 0]
        + coeffs[1] * gv[:, 1]
        + coeffs[2] * gw[:, 2]
    )
    return pred, r


def sample_volume(rng, n, tmin=0.0, tmax=1.0):
    X = rng.random((n, 4)).astype(np.float32)
    X[:, 3] = (tmin + (tmax - tmin) * X[:, 3]).astype(np.float32)
    return X


def sample_bottom(rng, n, tmin=0.0, tmax=1.0):
    X = sample_volume(rng, n, tmin=tmin, tmax=tmax)
    X[:, 2] = 0.0
    return X


# ============================================================
# Time partition: REAL hourly anchors vs PCHIP generated times
# ============================================================

def load_real_times(data_dir, manifest):
    path = os.path.join(data_dir, "train_hourly_processed.npz")
    d = np.load(path)
    hours = np.asarray(d["hours"], dtype=np.float64)
    total_hours = float(manifest["domain"]["total_hours"])
    return hours / total_hours


def load_generated_times(data_dir):
    path = os.path.join(data_dir, "train_times.csv")
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.asarray(arr[:, 1], dtype=np.float64)


def classify_time_slices(generated_t, real_t, tol=2e-6):
    real_idx = []
    interp_idx = []

    for k, t in enumerate(generated_t):
        if np.min(np.abs(real_t - t)) < tol:
            real_idx.append(k)
        else:
            interp_idx.append(k)

    return (
        np.asarray(real_idx, dtype=np.int64),
        np.asarray(interp_idx, dtype=np.int64),
    )


def sample_rows_from_time_group(rng, time_indices, npts_per_time, n_samples):
    if n_samples <= 0:
        return np.empty((0,), dtype=np.int64)
    if len(time_indices) == 0:
        raise RuntimeError("Requested samples from an empty time group")

    tids = rng.choice(time_indices, size=n_samples, replace=True)
    local = rng.integers(0, npts_per_time, size=n_samples)
    return tids * npts_per_time + local


# ============================================================
# Target/prediction transforms
# ============================================================

def temporal_normalize_targets(Yglobal, tnorm, tscale):
    """Convert stored Y=(u,v)/Sglobal to Ybar=(u,v)/S(t)."""
    sh = tscale.shat(tnorm).astype(np.float32)
    return Yglobal / sh[:, None], sh


def temporal_denormalize_prediction(pred_bar_uv, tnorm, tscale):
    """Convert predicted (u_bar,v_bar) to the stored global-normalized scale."""
    sh = tscale.shat(tnorm).astype(np.float32)
    return pred_bar_uv * sh[:, None], sh


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def uv_metrics_on_rows(
    model,
    frame,
    Xtrain,
    Ytrain,
    rows,
    scaling,
    tscale,
    device,
    batch=65536,
):
    # Physical/global-normalized accumulators.
    err2_global = 0.0
    ref2_global = 0.0
    ncomp = 0

    # Old-style temporally normalized MSE accumulators.
    err2_bar = 0.0

    for a in range(0, len(rows), batch):
        ridx = rows[a:a + batch]

        Xe = np.array(Xtrain[ridx], dtype=np.float32, copy=True)
        Yg = np.array(Ytrain[ridx], dtype=np.float32, copy=True)
        Xs = frame.earth_to_storm(Xe)

        t = Xs[:, 3].astype(np.float64)
        Ybar, sh = temporal_normalize_targets(Yg, t, tscale)

        pred_bar = model(torch.from_numpy(Xs).to(device))[:, :2].cpu().numpy()
        pred_global = pred_bar * sh[:, None]

        e_global = pred_global - Yg
        e_bar = pred_bar - Ybar

        err2_global += float(np.sum(e_global ** 2))
        ref2_global += float(np.sum(Yg ** 2))
        err2_bar += float(np.sum(e_bar ** 2))
        ncomp += e_global.size

    rel = np.sqrt(err2_global / (ref2_global + 1e-30))
    rmse_phys = scaling.Sglobal * np.sqrt(err2_global / max(ncomp, 1))

    mse_bar_component = err2_bar / max(ncomp, 1)
    mse_bar_pair = err2_bar / max(ncomp / 2.0, 1.0)

    return {
        "relL2_uv": float(rel),
        "RMSE_uv_mps": float(rmse_phys),
        "mse_bar_uv": float(mse_bar_component),
        "mse_bar_pair": float(mse_bar_pair),
    }


def physics_metrics(model, Xdiv, Xbdy, scaling, tscale, coeffs):
    model.eval()

    _, r = divergence_residual(
        model,
        Xdiv,
        coeffs,
        create_graph=False,
    )

    r_np = r.detach().cpu().numpy()
    t_div = Xdiv[:, 3].detach().cpu().numpy().astype(np.float64)
    S_div = tscale.S_phys(t_div)

    # Physical divergence pointwise:
    # div = S(t)/Lh * r.
    div_phys = (S_div / scaling.Lh) * r_np
    rms_r = float(np.sqrt(np.mean(r_np ** 2)))
    rms_div = float(np.sqrt(np.mean(div_phys ** 2)))

    with torch.no_grad():
        wb_bar = model(Xbdy)[:, 2].cpu().numpy()
        wi_bar = model(Xdiv)[:, 2].cpu().numpy()

    t_bdy = Xbdy[:, 3].detach().cpu().numpy().astype(np.float64)
    Sw_bdy = tscale.Sw_phys(t_bdy, scaling.Lz, scaling.Lh)
    Sw_div = tscale.Sw_phys(t_div, scaling.Lz, scaling.Lh)

    wb_phys = Sw_bdy * wb_bar
    wi_phys = Sw_div * wi_bar

    return {
        "rms_div_dimless": rms_r,
        "rms_div_phys_sinv": rms_div,
        "rms_bdy_bar": float(np.sqrt(np.mean(wb_bar ** 2))),
        "rms_bdy_phys_mps": float(np.sqrt(np.mean(wb_phys ** 2))),
        "rms_w_phys_mps": float(np.sqrt(np.mean(wi_phys ** 2))),
    }


@torch.no_grad()
def full_data_metrics(
    model,
    frame,
    Xtrain,
    Ytrain,
    scaling,
    tscale,
    device,
    generated_t,
    real_t,
):
    nt = len(generated_t)
    if len(Xtrain) % nt != 0:
        raise RuntimeError(
            "Training rows are not divisible by number of generated time slices"
        )

    npts = len(Xtrain) // nt

    groups = {
        "all": {"e2": 0.0, "r2": 0.0, "ebar2": 0.0, "n": 0},
        "real": {"e2": 0.0, "r2": 0.0, "ebar2": 0.0, "n": 0},
        "interp": {"e2": 0.0, "r2": 0.0, "ebar2": 0.0, "n": 0},
    }

    for k, tn in enumerate(generated_t):
        a = k * npts
        b = a + npts

        Xe = np.array(Xtrain[a:b], dtype=np.float32, copy=True)
        Yg = np.array(Ytrain[a:b], dtype=np.float32, copy=True)
        Xs = frame.earth_to_storm(Xe)

        sh = float(tscale.shat(np.asarray([tn], dtype=np.float64))[0])
        Ybar = Yg / sh

        pred_bar = model(torch.from_numpy(Xs).to(device))[:, :2].cpu().numpy()
        pred_global = pred_bar * sh

        e_global = pred_global - Yg
        e_bar = pred_bar - Ybar

        e2 = float(np.sum(e_global ** 2))
        r2 = float(np.sum(Yg ** 2))
        ebar2 = float(np.sum(e_bar ** 2))
        n = e_global.size

        is_real = np.min(np.abs(real_t - tn)) < 2e-6
        kind = "real" if is_real else "interp"

        for g in ("all", kind):
            groups[g]["e2"] += e2
            groups[g]["r2"] += r2
            groups[g]["ebar2"] += ebar2
            groups[g]["n"] += n

    out = {}
    for name, q in groups.items():
        rel = np.sqrt(q["e2"] / (q["r2"] + 1e-30))
        rmse = scaling.Sglobal * np.sqrt(q["e2"] / max(q["n"], 1))
        mse_bar = q["ebar2"] / max(q["n"], 1)
        out[name] = {
            "relL2_uv": float(rel),
            "RMSE_uv_mps": float(rmse),
            "mse_bar_uv": float(mse_bar),
        }

    return out


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data_dir",
        default="melissa_reconstruction_allobs_prepared_5min",
    )
    ap.add_argument(
        "--outdir",
        default="melissa_reconstruction_final/seed_1234",
    )

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--width", type=int, default=80)
    ap.add_argument("--depth", type=int, default=5)

    ap.add_argument("--nEpoch", type=int, default=2500)
    ap.add_argument("--nBatch", type=int, default=25)
    ap.add_argument("--dataBatchSize", type=int, default=512)
    ap.add_argument("--divBatchSize", type=int, default=2048)
    ap.add_argument("--bottomBatchSize", type=int, default=1024)

    ap.add_argument("--evalDataSize", type=int, default=100000)
    ap.add_argument("--evalDivSize", type=int, default=8192)
    ap.add_argument("--evalBdySize", type=int, default=8192)

    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--step_size", type=int, default=200)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--beta_div", type=float, default=0.03)
    ap.add_argument("--beta_bdy", type=float, default=1.0)

    ap.add_argument(
        "--real_fraction",
        type=float,
        default=0.5,
        help="Fraction of each UV minibatch sampled from real hourly anchors.",
    )
    ap.add_argument(
        "--alpha_interp",
        type=float,
        default=0.5,
        help="J_uv = J_real + alpha_interp * J_pchip in temporally normalized variables.",
    )

    ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--log_every", type=int, default=20)

    args = ap.parse_args()

    if not (0.0 < args.real_fraction < 1.0):
        raise ValueError("--real_fraction must lie strictly between 0 and 1")
    if args.alpha_interp < 0:
        raise ValueError("--alpha_interp must be non-negative")

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("MELISSA FINAL RECONSTRUCTION -- ALL OBSERVED SNAPSHOTS")
    print("=" * 78)
    print(f"device = {device}")
    if device.type == "cuda":
        print(f"GPU = {torch.cuda.get_device_name(0)}")

    with open(os.path.join(args.data_dir, "manifest.json"), "r") as f:
        manifest = json.load(f)

    # Final Melissa protocol: all available wind snapshots must be assimilated.
    n_available = int(manifest.get("n_power_snapshots_available", -1))
    n_assimilated = int(manifest.get("n_train_snapshots", -1))
    active_holdouts = {
        "dev_key": manifest.get("dev_key", ""),
        "val_key": manifest.get("val_key", ""),
        "test_key_reserved": manifest.get("test_key_reserved", ""),
    }
    active_holdouts = {k: v for k, v in active_holdouts.items() if v}

    if n_available <= 0 or n_assimilated <= 0:
        raise RuntimeError("manifest.json does not contain valid snapshot counts")
    if n_available != n_assimilated:
        raise RuntimeError(
            "This final reconstruction requires ALL available snapshots to be assimilated: "
            f"available={n_available}, assimilated={n_assimilated}"
        )
    if active_holdouts:
        raise RuntimeError(
            "This final reconstruction must not reserve temporal snapshots. "
            f"Non-empty holdout keys found: {active_holdouts}"
        )

    frame = StormFrame(args.data_dir, manifest)
    scaling = Scaling(manifest, frame)
    tscale = TemporalVelocityScale(args.data_dir, manifest)

    real_t = load_real_times(args.data_dir, manifest)
    generated_t = load_generated_times(args.data_dir)
    real_time_idx, interp_time_idx = classify_time_slices(generated_t, real_t)

    if generated_t.min() < tscale.tmin - 2e-6 or generated_t.max() > tscale.tmax + 2e-6:
        raise RuntimeError(
            "Generated training times extend outside the REAL-anchor interval used for S(t): "
            f"generated=[{generated_t.min():.8f},{generated_t.max():.8f}], "
            f"S-anchor=[{tscale.tmin:.8f},{tscale.tmax:.8f}]"
        )

    Xtrain = np.load(
        os.path.join(args.data_dir, "train_X.npy"),
        mmap_mode="r",
    )
    Ytrain = np.load(
        os.path.join(args.data_dir, "train_Yuv.npy"),
        mmap_mode="r",
    )

    if Xtrain.shape[1] != 4 or Ytrain.shape[1] != 2 or len(Xtrain) != len(Ytrain):
        raise RuntimeError(f"Bad shapes: X={Xtrain.shape}, Y={Ytrain.shape}")

    nt = len(generated_t)
    if len(Xtrain) % nt != 0:
        raise RuntimeError(
            f"len(Xtrain)={len(Xtrain)} is not divisible by nt={nt}"
        )
    npts_per_time = len(Xtrain) // nt

    n_real_batch = int(round(args.dataBatchSize * args.real_fraction))
    n_real_batch = min(max(n_real_batch, 1), args.dataBatchSize - 1)
    n_interp_batch = args.dataBatchSize - n_real_batch

    S_dense = tscale.S_phys(generated_t)
    sh_dense = S_dense / scaling.Sglobal

    # Save the exact S(t) used by this experiment.
    np.savez(
        os.path.join(args.outdir, "temporal_velocity_scale.npz"),
        t_real=tscale.t_real,
        hours_real=tscale.hours_real,
        S_real_phys_mps=tscale.S_real_phys,
        shat_real=tscale.shat_real,
        t_dense=generated_t,
        S_dense_phys_mps=S_dense,
        shat_dense=sh_dense,
        S_global_mps=np.asarray(scaling.Sglobal, dtype=np.float64),
    )

    print()
    print("Storm-relative physical domain:")
    print(f"  Lxi  = {frame.Lxi/1000:.3f} km")
    print(f"  Leta = {frame.Leta/1000:.3f} km")
    print(f"  Lz   = {frame.Lz:.3f} m")
    print(f"  Lh   = {frame.Lh/1000:.3f} km")

    print()
    print("Velocity scaling:")
    print(f"  stored S_global          = {scaling.Sglobal:.6f} m/s")
    print(
        "  REAL S_k range          = "
        f"[{tscale.S_real_phys.min():.6f}, {tscale.S_real_phys.max():.6f}] m/s"
    )
    print(f"  REAL S_k mean           = {tscale.S_real_phys.mean():.6f} m/s")
    print(
        "  dense S(t) range        = "
        f"[{S_dense.min():.6f}, {S_dense.max():.6f}] m/s"
    )
    print(
        "  dense S(t)/S_global     = "
        f"[{sh_dense.min():.6f}, {sh_dense.max():.6f}]"
    )
    print("  interpolation            = PCHIP(log S_k), all observed hourly anchors")
    print(f"  reference Sw_global      = {scaling.Sw_global:.6f} m/s")

    print()
    print("Divergence coefficients:")
    print(
        f"  ({scaling.cx:.6e}, {scaling.cy:.6e}, {scaling.cz:.6e})"
    )
    print("  physical div = S(t)/Lh * r")

    print()
    print(f"Observed snapshots  = {n_assimilated} / {n_available} available")
    print(f"Training rows       = {len(Xtrain)}")
    print(f"Generated times     = {nt}")
    print(f"Real hourly times   = {len(real_time_idx)}")
    print(f"Interpolated times  = {len(interp_time_idx)}")
    print(f"Points/time         = {npts_per_time}")

    print()
    print("Balanced UV minibatch:")
    print(f"  dataBatchSize     = {args.dataBatchSize}")
    print(f"  real samples      = {n_real_batch}")
    print(f"  PCHIP samples     = {n_interp_batch}")
    print(f"  real_fraction     = {args.real_fraction:.6f}")
    print(f"  alpha_interp      = {args.alpha_interp:.6f}")
    print("  targets           = (u/S(t), v/S(t))")
    print("  J_uv              = J_real + alpha_interp * J_pchip")

    model = ResidualNet(args.width, args.depth).to(device)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Network parameters  = {nparams}")

    print()
    print(f"beta_div = {args.beta_div}")
    print(f"beta_bdy = {args.beta_bdy}")
    print("=" * 78)

    coeffs = torch.tensor(
        [scaling.cx, scaling.cy, scaling.cz],
        dtype=torch.float32,
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma,
    )

    rng = np.random.default_rng(args.seed)

    # --------------------------------------------------------
    # Fixed balanced evaluation sets
    # --------------------------------------------------------
    rng_eval = np.random.default_rng(260926)

    eval_real_n = max(1, args.evalDataSize // 2)
    eval_interp_n = max(1, args.evalDataSize - eval_real_n)

    eval_real_rows = sample_rows_from_time_group(
        rng_eval,
        real_time_idx,
        npts_per_time,
        eval_real_n,
    )
    eval_interp_rows = sample_rows_from_time_group(
        rng_eval,
        interp_time_idx,
        npts_per_time,
        eval_interp_n,
    )

    t_phys_min = float(generated_t.min())
    t_phys_max = float(generated_t.max())

    Xdiv_eval = torch.from_numpy(
        sample_volume(rng_eval, args.evalDivSize, t_phys_min, t_phys_max)
    ).to(device)
    Xbdy_eval = torch.from_numpy(
        sample_bottom(rng_eval, args.evalBdySize, t_phys_min, t_phys_max)
    ).to(device)

    history = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_t0 = time.perf_counter()

    # ========================================================
    # Training
    # ========================================================
    for epoch in range(args.nEpoch):
        model.train()

        acc_total = 0.0
        acc_real = 0.0
        acc_interp = 0.0
        acc_uv = 0.0
        acc_div = 0.0
        acc_bdy = 0.0

        for _ in range(args.nBatch):
            ridx = sample_rows_from_time_group(
                rng,
                real_time_idx,
                npts_per_time,
                n_real_batch,
            )
            iidx = sample_rows_from_time_group(
                rng,
                interp_time_idx,
                npts_per_time,
                n_interp_batch,
            )

            Xe_r = np.array(Xtrain[ridx], dtype=np.float32, copy=True)
            Yg_r = np.array(Ytrain[ridx], dtype=np.float32, copy=True)
            Xe_i = np.array(Xtrain[iidx], dtype=np.float32, copy=True)
            Yg_i = np.array(Ytrain[iidx], dtype=np.float32, copy=True)

            Xs_r = frame.earth_to_storm(Xe_r)
            Xs_i = frame.earth_to_storm(Xe_i)

            Ybar_r, _ = temporal_normalize_targets(
                Yg_r, Xs_r[:, 3].astype(np.float64), tscale
            )
            Ybar_i, _ = temporal_normalize_targets(
                Yg_i, Xs_i[:, 3].astype(np.float64), tscale
            )

            Xb_np = np.concatenate([Xs_r, Xs_i], axis=0)
            Yb_np = np.concatenate([Ybar_r, Ybar_i], axis=0)

            Xb = torch.from_numpy(Xb_np).to(device)
            Yb = torch.from_numpy(Yb_np).to(device)

            pred = model(Xb)

            pr = pred[:n_real_batch, :2]
            pi = pred[n_real_batch:, :2]
            yr = Yb[:n_real_batch]
            yi = Yb[n_real_batch:]

            # Same pairwise data-loss convention used by the balanced trainer.
            loss_real = torch.mean(
                (pr[:, 0] - yr[:, 0]) ** 2
                + (pr[:, 1] - yr[:, 1]) ** 2
            )

            loss_interp = torch.mean(
                (pi[:, 0] - yi[:, 0]) ** 2
                + (pi[:, 1] - yi[:, 1]) ** 2
            )

            loss_uv = loss_real + args.alpha_interp * loss_interp

            # --------------------------
            # Divergence collocation
            # --------------------------
            Xp = torch.from_numpy(
                sample_volume(
                    rng,
                    args.divBatchSize,
                    t_phys_min,
                    t_phys_max,
                )
            ).to(device)

            _, rdiv = divergence_residual(
                model,
                Xp,
                coeffs,
                create_graph=True,
            )
            loss_div = torch.mean(rdiv ** 2)

            # --------------------------
            # Bottom boundary condition
            # --------------------------
            Xbdy = torch.from_numpy(
                sample_bottom(
                    rng,
                    args.bottomBatchSize,
                    t_phys_min,
                    t_phys_max,
                )
            ).to(device)

            wb = model(Xbdy)[:, 2]
            loss_bdy = torch.mean(wb ** 2)

            # --------------------------
            # Total objective
            # --------------------------
            loss = (
                loss_uv
                + args.beta_div * loss_div
                + args.beta_bdy * loss_bdy
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )

            optimizer.step()

            acc_total += float(loss.item())
            acc_real += float(loss_real.item())
            acc_interp += float(loss_interp.item())
            acc_uv += float(loss_uv.item())
            acc_div += float(loss_div.item())
            acc_bdy += float(loss_bdy.item())

        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": acc_total / args.nBatch,
            "train_real_pair": acc_real / args.nBatch,
            "train_interp_pair": acc_interp / args.nBatch,
            "train_uv": acc_uv / args.nBatch,
            "train_div": acc_div / args.nBatch,
            "train_bdy": acc_bdy / args.nBatch,
            "lr": optimizer.param_groups[0]["lr"],
        }

        do_eval = (
            epoch % args.eval_every == 0
            or epoch == args.nEpoch - 1
        )

        if do_eval:
            model.eval()

            mr = uv_metrics_on_rows(
                model,
                frame,
                Xtrain,
                Ytrain,
                eval_real_rows,
                scaling,
                tscale,
                device,
            )

            mi = uv_metrics_on_rows(
                model,
                frame,
                Xtrain,
                Ytrain,
                eval_interp_rows,
                scaling,
                tscale,
                device,
            )

            phys = physics_metrics(
                model,
                Xdiv_eval,
                Xbdy_eval,
                scaling,
                tscale,
                coeffs,
            )

            # Monitoring objective has the same mathematical structure as training.
            # It is NOT used for checkpoint/model selection.
            # mse_bar_pair = mean[(du_bar)^2 + (dv_bar)^2].
            monitor_obj = (
                mr["mse_bar_pair"]
                + args.alpha_interp * mi["mse_bar_pair"]
                + args.beta_div * phys["rms_div_dimless"] ** 2
                + args.beta_bdy * phys["rms_bdy_bar"] ** 2
            )

            row.update({
                "mse_bar_real": mr["mse_bar_uv"],
                "mse_bar_interp": mi["mse_bar_uv"],
                "error_uv_real": mr["relL2_uv"],
                "RMSE_uv_real_mps": mr["RMSE_uv_mps"],
                "error_uv_interp": mi["relL2_uv"],
                "RMSE_uv_interp_mps": mi["RMSE_uv_mps"],
                "div_sinv": phys["rms_div_phys_sinv"],
                "bdy_mps": phys["rms_bdy_phys_mps"],
                "w_rms_mps": phys["rms_w_phys_mps"],
                "monitor_objective": float(monitor_obj),
            })

        history.append(row)

        if (
            epoch % args.log_every == 0
            or epoch == args.nEpoch - 1
        ):
            msg = (
                f"Epoch {epoch:5d} | "
                f"loss={row['train_loss']:.6e} "
                f"(real_pair={row['train_real_pair']:.3e}, "
                f"pchip_pair={row['train_interp_pair']:.3e}, "
                f"uv={row['train_uv']:.3e}, "
                f"div={row['train_div']:.3e}, "
                f"bdy={row['train_bdy']:.3e})"
            )

            if "error_uv_real" in row:
                msg += (
                    " | "
                    f"mse_bar_real={row['mse_bar_real']:.3e} "
                    f"mse_bar_pchip={row['mse_bar_interp']:.3e} "
                    f"error_uv_real={row['error_uv_real']:.6e} "
                    f"error_uv_pchip={row['error_uv_interp']:.6e} "
                    f"RMSE_real={row['RMSE_uv_real_mps']:.4f} m/s "
                    f"div={row['div_sinv']:.3e} 1/s "
                    f"bdy={row['bdy_mps']:.3e} m/s "
                    f"w_rms={row['w_rms_mps']:.3e} m/s"
                )

            print(msg, flush=True)

    train_seconds = time.perf_counter() - train_t0
    peak_gpu_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda" else 0
    )

    # ========================================================
    # Fixed-budget final model
    # ========================================================
    torch.save(
        model.state_dict(),
        os.path.join(args.outdir, "final_model.pt"),
    )

    model.eval()

    print()
    print("Evaluating FINAL-EPOCH model over the complete reconstruction dataset...")

    all_uv = full_data_metrics(
        model,
        frame,
        Xtrain,
        Ytrain,
        scaling,
        tscale,
        device,
        generated_t,
        real_t,
    )

    rng_final = np.random.default_rng(260927)
    Xdiv_final = torch.from_numpy(
        sample_volume(rng_final, 50000, t_phys_min, t_phys_max)
    ).to(device)
    Xbdy_final = torch.from_numpy(
        sample_bottom(rng_final, 50000, t_phys_min, t_phys_max)
    ).to(device)

    phys_final = physics_metrics(
        model,
        Xdiv_final,
        Xbdy_final,
        scaling,
        tscale,
        coeffs,
    )

    summary = {
        "experiment": (
            "full reconstruction using all observed hourly snapshots, balanced REAL/PCHIP minibatches, "
            "time-dependent velocity scaling S(t), physical divergence scaling, and no Fourier features"
        ),
        "coordinates": "storm-relative (xi,eta,z,t)",
        "scientific_scope": "reconstruction, not forecasting; all available hourly u,v observations assimilated",
        "seed": args.seed,
        "architecture": {
            "width": args.width,
            "depth": args.depth,
            "n_parameters": nparams,
            "fourier_features": False,
        },
        "sampling": {
            "generated_times": int(nt),
            "real_hourly_times": int(len(real_time_idx)),
            "interpolated_times": int(len(interp_time_idx)),
            "points_per_time": int(npts_per_time),
            "data_batch_size": int(args.dataBatchSize),
            "real_batch_size": int(n_real_batch),
            "interp_batch_size": int(n_interp_batch),
            "real_fraction": float(args.real_fraction),
            "alpha_interp": float(args.alpha_interp),
        },
        "scaling": {
            "definition": "S_k=max(max|u_k|,max|v_k|) on all processed observed hourly anchors; log-PCHIP in time",
            "S_global_storage_mps": scaling.Sglobal,
            "S_real_min_mps": float(np.min(tscale.S_real_phys)),
            "S_real_mean_mps": float(np.mean(tscale.S_real_phys)),
            "S_real_max_mps": float(np.max(tscale.S_real_phys)),
            "S_dense_min_mps": float(np.min(S_dense)),
            "S_dense_max_mps": float(np.max(S_dense)),
            "Lxi_m": frame.Lxi,
            "Leta_m": frame.Leta,
            "Lz_m": frame.Lz,
            "Lh_m": frame.Lh,
            "div_coeff_x": scaling.cx,
            "div_coeff_y": scaling.cy,
            "div_coeff_z": scaling.cz,
        },
        "loss": {
            "definition_uv": (
                "J_real + alpha_interp * J_pchip on temporally normalized "
                "targets (u/S(t),v/S(t))"
            ),
            "beta_div": args.beta_div,
            "beta_bdy": args.beta_bdy,
        },
        "training": {
            "epochs": args.nEpoch,
            "batches_per_epoch": args.nBatch,
            "div_batch": args.divBatchSize,
            "bdy_batch": args.bottomBatchSize,
            "initial_lr": args.lr,
            "weight_decay": args.weight_decay,
            "step_size": args.step_size,
            "gamma": args.gamma,
            "checkpoint_selection": "none; fixed final epoch is reported",
            "wall_time_seconds": float(train_seconds),
            "peak_gpu_memory_bytes": int(peak_gpu_bytes),
        },
        "reported_checkpoint": "final epoch; fixed training budget; no temporal holdout selection",
        "final_epoch": int(args.nEpoch - 1),
        "full_data_uv_metrics": all_uv,
        "physics_metrics": {
            "rms_div_phys_sinv": phys_final["rms_div_phys_sinv"],
            "rms_bdy_phys_mps": phys_final["rms_bdy_phys_mps"],
            "rms_w_phys_mps": phys_final["rms_w_phys_mps"],
        },
    }

    with open(
        os.path.join(args.outdir, "summary.json"),
        "w",
    ) as f:
        json.dump(summary, f, indent=2)

    fieldnames = [
        "epoch",
        "train_loss",
        "train_real_pair",
        "train_interp_pair",
        "train_uv",
        "train_div",
        "train_bdy",
        "lr",
        "mse_bar_real",
        "mse_bar_interp",
        "error_uv_real",
        "RMSE_uv_real_mps",
        "error_uv_interp",
        "RMSE_uv_interp_mps",
        "div_sinv",
        "bdy_mps",
        "w_rms_mps",
        "monitor_objective",
    ]

    with open(
        os.path.join(args.outdir, "history.csv"),
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print()
    print("=" * 78)
    print("FINAL RECONSTRUCTION METRICS -- ALL OBSERVED SNAPSHOTS")
    print("=" * 78)
    print(f"reported epoch  = {args.nEpoch - 1} (fixed final epoch)")
    print(f"training time   = {train_seconds:.2f} s")
    if device.type == "cuda":
        print(f"peak GPU memory = {peak_gpu_bytes / 1024**3:.3f} GiB")

    print()
    print("ALL dense targets:")
    print(f"  mse_bar_uv    = {all_uv['all']['mse_bar_uv']:.6e}  <-- old-style normalized MSE")
    print(f"  error_uv      = {all_uv['all']['relL2_uv']:.6e}")
    print(f"  RMSE_uv       = {all_uv['all']['RMSE_uv_mps']:.6f} m/s")

    print()
    print("REAL hourly anchors:")
    print(f"  mse_bar_real  = {all_uv['real']['mse_bar_uv']:.6e}")
    print(f"  error_uv_real = {all_uv['real']['relL2_uv']:.6e}")
    print(f"  RMSE_uv_real  = {all_uv['real']['RMSE_uv_mps']:.6f} m/s")

    print()
    print("PCHIP intermediate targets:")
    print(f"  mse_bar_pchip = {all_uv['interp']['mse_bar_uv']:.6e}")
    print(f"  error_uv_pchip= {all_uv['interp']['relL2_uv']:.6e}")
    print(f"  RMSE_uv_pchip = {all_uv['interp']['RMSE_uv_mps']:.6f} m/s")

    print()
    print(
        f"RMS(div)        = {phys_final['rms_div_phys_sinv']:.6e} 1/s"
    )
    print(
        f"RMS(w bottom)   = {phys_final['rms_bdy_phys_mps']:.6e} m/s"
    )
    print(
        f"RMS(w interior) = {phys_final['rms_w_phys_mps']:.6e} m/s"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
