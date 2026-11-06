#!/usr/bin/env python3

import argparse
import csv
import json
import os
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Network: same architecture as the final Melissa runs
# ============================================================

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


# ============================================================
# Utilities
# ============================================================

def decode_key(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x**2)))


def rmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    return float(
        np.sqrt(np.mean((a - b)**2))
    )


def correlation(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    a = a - np.mean(a)
    b = b - np.mean(b)

    den = np.sqrt(
        np.sum(a*a) *
        np.sum(b*b)
    )

    if den < 1e-30:
        return np.nan

    return float(
        np.sum(a*b) / den
    )


def load_state(path, device):
    try:
        state = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state = torch.load(
            path,
            map_location=device,
        )

    if "state_dict" in state:
        state = state["state_dict"]

    if "model_state_dict" in state:
        state = state["model_state_dict"]

    if any(k.startswith("module.") for k in state.keys()):
        state = {
            k.replace("module.", "", 1): v
            for k, v in state.items()
        }

    return state


# ============================================================
# Coordinates
# ============================================================

def build_input(
    xi_grid,
    eta_grid,
    z_grid,
    t_norm,
    xi_min,
    xi_max,
    eta_min,
    eta_max,
    Lz,
):
    """
    Build normalized storm-relative coordinates:

        (xi_hat, eta_hat, z_hat, t_hat)

    with the same convention used during training.
    """

    Z, ETA, XI = np.meshgrid(
        z_grid,
        eta_grid,
        xi_grid,
        indexing="ij",
    )

    xi_hat = (
        (XI.ravel() - xi_min)
        / (xi_max - xi_min)
    )

    eta_hat = (
        (ETA.ravel() - eta_min)
        / (eta_max - eta_min)
    )

    z_hat = Z.ravel() / Lz

    t_hat = np.full(
        XI.size,
        t_norm,
        dtype=np.float64,
    )

    X = np.column_stack([
        xi_hat,
        eta_hat,
        z_hat,
        t_hat,
    ])

    return X.astype(np.float32)


# ============================================================
# Prediction
# ============================================================

@torch.no_grad()
def predict_w(
    model,
    X,
    Sw,
    device,
    batch_size,
):
    """
    Network output:
        w_bar

    Physical vertical velocity:
        w = S(t) * Lz/Lh * w_bar
    """

    out = np.empty(
        len(X),
        dtype=np.float32,
    )

    for i in range(0, len(X), batch_size):
        j = min(
            i + batch_size,
            len(X),
        )

        xb = torch.from_numpy(
            X[i:j]
        ).to(device)

        out[i:j] = (
            model(xb)[:, 2]
            .detach()
            .cpu()
            .numpy()
        )

    return Sw * out


# ============================================================
# Main
# ============================================================

def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--data_dir",
        default="data/processed",
    )

    p.add_argument(
        "--model_root",
        default="outputs/split_protocol/models",
    )

    p.add_argument(
        "--model_file", default="best_model.pt",
        help="Validation-selected checkpoint for the split protocol",
    )

    p.add_argument(
        "--outdir",
        default="outputs/split_protocol/analysis/seed_stability",
    )

    p.add_argument(
        "--seeds",
        default="1234,2345,3456",
    )

    p.add_argument(
        "--width",
        type=int,
        default=80,
    )

    p.add_argument(
        "--depth",
        type=int,
        default=5,
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=65536,
    )

    args = p.parse_args()

    seeds = [
        int(s.strip())
        for s in args.seeds.split(",")
    ]

    pairs = list(
        combinations(seeds, 2)
    )

    os.makedirs(
        args.outdir,
        exist_ok=True,
    )

    # ========================================================
    # Prepared data
    # ========================================================

    manifest_path = os.path.join(
        args.data_dir,
        "manifest.json",
    )

    hourly_path = os.path.join(
        args.data_dir,
        "train_hourly_processed.npz",
    )

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    dat = np.load(
        hourly_path,
        allow_pickle=False,
    )

    keys = [
        decode_key(x)
        for x in dat["keys"]
    ]

    hours = np.asarray(
        dat["hours"],
        dtype=np.float64,
    )

    U = np.asarray(
        dat["U_phys_mps"],
        dtype=np.float32,
    )

    V = np.asarray(
        dat["V_phys_mps"],
        dtype=np.float32,
    )

    xi_grid = np.asarray(
        dat["xi_grid_m"],
        dtype=np.float64,
    )

    eta_grid = np.asarray(
        dat["eta_grid_m"],
        dtype=np.float64,
    )

    z_grid = np.asarray(
        dat["z_grid_m"],
        dtype=np.float64,
    )

    if U.shape != V.shape:
        raise RuntimeError(
            "U and V arrays have different shapes"
        )

    nt, nz, ny, nx = U.shape

    # ========================================================
    # Physical geometry
    # ========================================================

    common = manifest[
        "storm_relative_common_support"
    ]

    domain = manifest["domain"]

    xi_min = float(
        common["xi_min_m"]
    )

    xi_max = float(
        common["xi_max_m"]
    )

    eta_min = float(
        common["eta_min_m"]
    )

    eta_max = float(
        common["eta_max_m"]
    )

    Lxi = xi_max - xi_min
    Leta = eta_max - eta_min

    Lz = float(
        domain["Lz_m"]
    )

    Lh = float(
        np.sqrt(Lxi * Leta)
    )

    total_hours = float(
        domain["total_hours"]
    )

    # ========================================================
    # S(t) at original hourly fields
    #
    # Same definition used by final training:
    #
    # S_k = max(max|u_k|, max|v_k|)
    #
    # At an hourly anchor we do NOT need temporal interpolation.
    # ========================================================

    axes = tuple(
        range(1, U.ndim)
    )

    umax = np.max(
        np.abs(U),
        axis=axes,
    )

    vmax = np.max(
        np.abs(V),
        axis=axes,
    )

    S_real = np.maximum(
        umax,
        vmax,
    )

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 88)
    print(
        "MELISSA: STABILITY OF w ACROSS RANDOM INITIALIZATIONS"
    )
    print("=" * 88)

    print(f"device          = {device}")

    if device.type == "cuda":
        print(
            f"GPU             = "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"hourly anchors  = {nt}"
    )

    print(
        f"grid per time   = "
        f"{nz} x {ny} x {nx} "
        f"= {nz*ny*nx} points"
    )

    print(
        f"seeds           = {seeds}"
    )

    print(
        f"Lxi             = "
        f"{Lxi/1000:.3f} km"
    )

    print(
        f"Leta            = "
        f"{Leta/1000:.3f} km"
    )

    print(
        f"Lz              = "
        f"{Lz:.3f} m"
    )

    print(
        f"Lh              = "
        f"{Lh/1000:.3f} km"
    )

    print("=" * 88)

    # ========================================================
    # Models
    # ========================================================

    models = {}

    for seed in seeds:

        model_path = os.path.join(
            args.model_root,
            f"seed_{seed}",
            args.model_file,
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                model_path
            )

        model = RitzNet(
            width=args.width,
            depth=args.depth,
        ).to(device)

        state = load_state(
            model_path,
            device,
        )

        model.load_state_dict(
            state
        )

        model.eval()

        models[seed] = model

        print(
            f"Loaded seed {seed}: "
            f"{model_path}"
        )

    # ========================================================
    # Storage
    #
    # Shape:
    #   time x z x eta x xi
    #
    # This is small enough:
    #   61 x 15 x 40 x 40
    # ========================================================

    all_w = {
        seed: np.empty(
            U.shape,
            dtype=np.float32,
        )
        for seed in seeds
    }

    rows = []

    # ========================================================
    # Evaluate all 61 REAL times
    # ========================================================

    for it in range(nt):

        key = keys[it]

        hour = float(
            hours[it]
        )

        t_norm = (
            hour / total_hours
        )

        S = float(
            S_real[it]
        )

        # Physical vertical-velocity scale
        #
        # w = S(t) Lz/Lh w_bar

        Sw = (
            S * Lz / Lh
        )

        X = build_input(
            xi_grid,
            eta_grid,
            z_grid,
            t_norm,
            xi_min,
            xi_max,
            eta_min,
            eta_max,
            Lz,
        )

        fields = {}

        for seed in seeds:

            w = predict_w(
                models[seed],
                X,
                Sw,
                device,
                args.batch_size,
            )

            w = w.reshape(
                nz,
                ny,
                nx,
            )

            fields[seed] = w

            all_w[seed][it] = w

        # ----------------------------------------------------
        # Ensemble statistics
        # ----------------------------------------------------

        stack = np.stack(
            [
                fields[s]
                for s in seeds
            ],
            axis=0,
        )

        ensemble_mean = np.mean(
            stack,
            axis=0,
        )

        ensemble_std = np.std(
            stack,
            axis=0,
            ddof=1,
        )

        row = {
            "key": key,
            "hours_from_start": hour,
            "S_mps": S,
            "ensemble_rms_w_mps":
                rms(ensemble_mean),
            "ensemble_mean_pointwise_std_mps":
                float(
                    np.mean(
                        ensemble_std
                    )
                ),
            "ensemble_rms_pointwise_std_mps":
                rms(ensemble_std),
        }

        # Individual RMS(w)

        for seed in seeds:

            row[
                f"rms_w_{seed}_mps"
            ] = rms(
                fields[seed]
            )

        # ----------------------------------------------------
        # Pairwise comparisons
        # ----------------------------------------------------

        for a, b in pairs:

            wa = fields[a]
            wb = fields[b]

            rho = correlation(
                wa,
                wb,
            )

            err = rmse(
                wa,
                wb,
            )

            reference_scale = (
                0.5 *
                (
                    rms(wa)
                    +
                    rms(wb)
                )
            )

            relative_err = (
                err
                /
                (
                    reference_scale
                    + 1e-30
                )
            )

            row[
                f"corr_{a}_{b}"
            ] = rho

            row[
                f"rmse_{a}_{b}_mps"
            ] = err

            row[
                f"relative_rmse_{a}_{b}"
            ] = relative_err

        rows.append(row)

        print(
            f"[{it+1:02d}/{nt:02d}] "
            f"{key} | "
            +
            " | ".join(
                [
                    (
                        f"rho({a},{b})="
                        f"{row[f'corr_{a}_{b}']:.6f}"
                    )
                    for a, b in pairs
                ]
            )
            +
            (
                " | spread="
                f"{row['ensemble_rms_pointwise_std_mps']:.3e}"
                " m/s"
            ),
            flush=True,
        )

    # ========================================================
    # CSV by time
    # ========================================================

    csv_path = os.path.join(
        args.outdir,
        "w_seed_stability_by_time.csv",
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "purpose": (
            "stability of reconstructed w "
            "across independent random initializations; "
            "this is not validation against true atmospheric w"
        ),
        "seeds": seeds,
        "n_real_hourly_times": nt,
        "grid_shape_nz_ny_nx": [
            nz,
            ny,
            nx,
        ],
        "pairwise": {},
    }

    print()
    print("=" * 88)
    print(
        "w STABILITY SUMMARY OVER REAL HOURLY ANCHORS"
    )
    print("=" * 88)

    # ========================================================
    # Pairwise summaries
    # ========================================================

    for a, b in pairs:

        c = np.asarray(
            [
                r[f"corr_{a}_{b}"]
                for r in rows
            ],
            dtype=np.float64,
        )

        e = np.asarray(
            [
                r[f"rmse_{a}_{b}_mps"]
                for r in rows
            ],
            dtype=np.float64,
        )

        er = np.asarray(
            [
                r[f"relative_rmse_{a}_{b}"]
                for r in rows
            ],
            dtype=np.float64,
        )

        # Global metrics using every physical point
        # from every real hourly snapshot.

        wa = all_w[a].ravel()
        wb = all_w[b].ravel()

        pair_summary = {

            "correlation_by_time_mean":
                float(
                    np.nanmean(c)
                ),

            "correlation_by_time_std":
                float(
                    np.nanstd(
                        c,
                        ddof=1,
                    )
                ),

            "correlation_by_time_min":
                float(
                    np.nanmin(c)
                ),

            "correlation_by_time_max":
                float(
                    np.nanmax(c)
                ),

            "rmse_by_time_mps_mean":
                float(
                    np.mean(e)
                ),

            "rmse_by_time_mps_std":
                float(
                    np.std(
                        e,
                        ddof=1,
                    )
                ),

            "relative_rmse_by_time_mean":
                float(
                    np.mean(er)
                ),

            "relative_rmse_by_time_std":
                float(
                    np.std(
                        er,
                        ddof=1,
                    )
                ),

            "global_correlation":
                correlation(
                    wa,
                    wb,
                ),

            "global_rmse_mps":
                rmse(
                    wa,
                    wb,
                ),
        }

        summary[
            "pairwise"
        ][
            f"{a}_{b}"
        ] = pair_summary

        print()
        print(
            f"Seeds {a} vs {b}"
        )

        print(
            "  correlation by time = "
            f"{pair_summary['correlation_by_time_mean']:.6f}"
            " +/- "
            f"{pair_summary['correlation_by_time_std']:.3e}"
        )

        print(
            "  RMSE by time        = "
            f"{pair_summary['rmse_by_time_mps_mean']:.6e}"
            " +/- "
            f"{pair_summary['rmse_by_time_mps_std']:.3e}"
            " m/s"
        )

        print(
            "  relative RMSE       = "
            f"{100*pair_summary['relative_rmse_by_time_mean']:.3f}"
            " +/- "
            f"{100*pair_summary['relative_rmse_by_time_std']:.3f}"
            " %"
        )

        print(
            "  GLOBAL correlation  = "
            f"{pair_summary['global_correlation']:.6f}"
        )

        print(
            "  GLOBAL RMSE         = "
            f"{pair_summary['global_rmse_mps']:.6e}"
            " m/s"
        )

    # ========================================================
    # Three-seed ensemble spread
    # ========================================================

    stack_all = np.stack(
        [
            all_w[s]
            for s in seeds
        ],
        axis=0,
    )

    ensemble_mean_all = np.mean(
        stack_all,
        axis=0,
    )

    ensemble_std_all = np.std(
        stack_all,
        axis=0,
        ddof=1,
    )

    rms_mean = rms(
        ensemble_mean_all
    )

    mean_std = float(
        np.mean(
            ensemble_std_all
        )
    )

    rms_std = rms(
        ensemble_std_all
    )

    relative_spread = (
        rms_std
        /
        (
            rms_mean
            + 1e-30
        )
    )

    summary["ensemble"] = {

        "rms_ensemble_mean_w_mps":
            rms_mean,

        "mean_pointwise_std_w_mps":
            mean_std,

        "rms_pointwise_std_w_mps":
            rms_std,

        "relative_rms_spread":
            relative_spread,
    }

    print()
    print(
        "Three-seed ensemble"
    )

    print(
        "  RMS ensemble-mean w   = "
        f"{rms_mean:.6e} m/s"
    )

    print(
        "  mean pointwise std(w) = "
        f"{mean_std:.6e} m/s"
    )

    print(
        "  RMS pointwise std(w)  = "
        f"{rms_std:.6e} m/s"
    )

    print(
        "  relative RMS spread   = "
        f"{100*relative_spread:.3f} %"
    )

    # ========================================================
    # Save fields for later figures
    # ========================================================

    peak_index = int(
        np.argmax(
            S_real
        )
    )

    peak_key = keys[
        peak_index
    ]

    summary[
        "peak_intensity_snapshot"
    ] = {
        "index": peak_index,
        "key": peak_key,
        "S_mps": float(
            S_real[
                peak_index
            ]
        ),
    }

    npz_data = {
        "keys":
            np.asarray(keys),

        "hours":
            hours,

        "xi_grid_m":
            xi_grid,

        "eta_grid_m":
            eta_grid,

        "z_grid_m":
            z_grid,

        "S_real_mps":
            S_real,

        "peak_index":
            np.asarray(
                peak_index
            ),

        "peak_key":
            np.asarray(
                peak_key
            ),

        "w_ensemble_mean_mps":
            ensemble_mean_all,

        "w_ensemble_std_mps":
            ensemble_std_all,
    }

    for seed in seeds:

        npz_data[
            f"w_seed_{seed}_mps"
        ] = all_w[seed]

    npz_path = os.path.join(
        args.outdir,
        "w_fields_all_real_times.npz",
    )

    np.savez_compressed(
        npz_path,
        **npz_data,
    )

    # ========================================================
    # JSON summary
    # ========================================================

    json_path = os.path.join(
        args.outdir,
        "w_seed_stability_summary.json",
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print(
        f"Peak-intensity snapshot = "
        f"{peak_key}"
    )

    print(
        f"CSV  = {csv_path}"
    )

    print(
        f"JSON = {json_path}"
    )

    print(
        f"NPZ  = {npz_path}"
    )

    print("=" * 88)


if __name__ == "__main__":
    main()
