#!/usr/bin/env python3
"""
prepare_melissa_hourly.py

Leakage-aware preprocessing for the hourly Hurricane Melissa NASA POWER dataset.

Pipeline
--------
1) Read raw POWER snapshots (lat, lon, z, ..., WSC, WD10M).
2) Convert WSC + meteorological FROM direction to earth-relative u_east, v_north.
3) Convert lon/lat to ONE fixed local Cartesian frame (x,y in metres).
4) Register every snapshot with respect to the externally supplied eye trajectory:
       xi  = x - x_c(t)
       eta = y - y_c(t)
   Registration changes coordinates only, not velocity values.
5) On each original vertical level, fit a 2-D Wahba/thin-plate smoothing spline
   independently for u and v and evaluate it on one common storm-relative grid.
6) Refine vertically by LINEAR interpolation between levels.
7) Fit TRAIN-only PCHIP interpolants in time at fixed (xi,eta,z).
8) At each generated time, put the field back into the fixed physical frame:
       x = xi + x_c(t),  y = eta + y_c(t).
9) Normalize network coordinates to [0,1]^4 using physical TRAIN geometry/time.
10) Normalize velocity with ONE global TRAIN-only constant S_global.

Important
---------
- DEV/VAL/TEST field snapshots are excluded before spatial/temporal fitting.
- The final TEST may be absent; its eye centre may still exist in the external
  trajectory CSV because that metadata is independent of the POWER wind field.
- Held-out sets are saved at their ORIGINAL raw POWER points (no Wahba, no
  vertical densification, no temporal interpolation in their target values).
- No RBF is used to reconstruct a velocity scale.  The same S_global is used
  everywhere.

SciPy implementation of Wahba smoothing
---------------------------------------
The horizontal smoother uses scipy.interpolate.RBFInterpolator with the
"thin_plate_spline" kernel, degree=1 and a non-negative smoothing parameter.
In 2-D this is the standard thin-plate spline representation underlying the
Wahba smoothing-spline variational formulation.  Horizontal coordinates are
scaled to [-1,1]^2 before fitting so the smoothing parameter is numerically
meaningful and independent of metres.
"""

import argparse
import csv
import glob
import json
import os
import re
from datetime import datetime

import numpy as np
from numpy.lib.format import open_memmap
from scipy.interpolate import PchipInterpolator, RBFInterpolator, RegularGridInterpolator


EARTH_RADIUS_M = 6_371_000.0
KEY_RE = re.compile(r"date(\d{8})_hour(\d{2})")


def parse_key_from_path(path):
    m = KEY_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse date/hour from filename: {path}")
    date, hour = m.group(1), m.group(2)
    return f"{date}T{hour}", datetime.strptime(date + hour, "%Y%m%d%H")


def parse_key_datetime(key):
    return datetime.strptime(key, "%Y%m%dT%H")


def load_raw(path, max_height=300.0):
    """Read downloader format: lat lon z u_old v_old w_old WSC WD10M."""
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 8:
        raise ValueError(
            f"{path} must have >=8 columns: lat lon z u_old v_old w_old WSC WD10M"
        )

    lat = data[:, 0].astype(float)
    lon = data[:, 1].astype(float)
    z = data[:, 2].astype(float)
    speed = data[:, 6].astype(float)
    direction = data[:, 7].astype(float)

    good = (
        np.isfinite(lat) & np.isfinite(lon) & np.isfinite(z)
        & np.isfinite(speed) & np.isfinite(direction)
        & (z >= 0.0) & (z <= max_height)
        & (speed >= 0.0) & (speed < 200.0)
        & (direction >= 0.0) & (direction <= 360.0)
    )
    lat, lon, z = lat[good], lon[good], z[good]
    speed, direction = speed[good], direction[good]

    theta = np.deg2rad(direction)
    # Meteorological WD10M is direction FROM which wind comes.
    u = -speed * np.sin(theta)  # eastward
    v = -speed * np.cos(theta)  # northward

    return {
        "lat": lat,
        "lon": lon,
        "z": z,
        "speed": speed,
        "direction_deg": direction,
        "u": u,
        "v": v,
    }


def lonlat_to_xy(lon, lat, lon0_deg, lat0_deg):
    """Fixed local equirectangular projection. Output in metres."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon0 = np.deg2rad(lon0_deg)
    lat0 = np.deg2rad(lat0_deg)
    x = EARTH_RADIUS_M * np.cos(lat0) * (np.deg2rad(lon) - lon0)
    y = EARTH_RADIUS_M * (np.deg2rad(lat) - lat0)
    return x, y


def _pick_field(fieldnames, candidates):
    lut = {f.strip().lower(): f for f in fieldnames if f is not None}
    for c in candidates:
        if c.lower() in lut:
            return lut[c.lower()]
    return None


def load_eye_centers(path):
    """
    Read hourly_eye_centers.csv robustly.

    Accepted key-like fields: key, snapshot_key, time_key
    Accepted latitude fields: eye_lat, center_lat, lat, latitude
    Accepted longitude fields: eye_lon, center_lon, lon, longitude

    If no key is present, date+hour or datetime_utc/datetime/time are accepted.
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        fields = reader.fieldnames
        kf = _pick_field(fields, ["key", "snapshot_key", "time_key"])
        latf = _pick_field(fields, ["eye_lat_deg", "eye_lat", "center_lat", "lat", "latitude"])
        lonf = _pick_field(fields, ["eye_lon_deg", "eye_lon", "center_lon", "lon", "longitude"])
        datef = _pick_field(fields, ["date", "yyyymmdd"])
        hourf = _pick_field(fields, ["hour", "hr"])
        dtf = _pick_field(fields, ["datetime_utc", "datetime", "time", "timestamp"])

        if latf is None or lonf is None:
            raise ValueError(
                f"Could not find eye latitude/longitude columns in {path}. Fields={fields}"
            )

        out = []
        for row in reader:
            if kf and row.get(kf, "").strip():
                key = row[kf].strip()
            elif datef and hourf:
                key = f"{row[datef].strip()}T{int(float(row[hourf])):02d}"
            elif dtf:
                txt = row[dtf].strip().replace("Z", "+00:00")
                dt = datetime.fromisoformat(txt)
                key = dt.strftime("%Y%m%dT%H")
            else:
                raise ValueError(
                    "Eye-centre CSV needs key, date+hour, or datetime column. "
                    f"Fields={fields}"
                )

            dt = parse_key_datetime(key)
            out.append({
                "key": key,
                "datetime": dt,
                "lat": float(row[latf]),
                "lon": float(row[lonf]),
            })

    # Deduplicate and sort.
    by_key = {}
    for r in out:
        if r["key"] in by_key:
            raise ValueError(f"Duplicate eye-centre key {r['key']} in {path}")
        by_key[r["key"]] = r
    out = sorted(by_key.values(), key=lambda q: q["datetime"])
    if len(out) < 2:
        raise ValueError("Need at least two eye-centre records.")
    return out


def make_vertical_targets(z_original, dz_m):
    z_original = np.sort(np.unique(np.asarray(z_original, dtype=float)))
    if dz_m <= 0:
        return z_original

    z0, z1 = float(z_original[0]), float(z_original[-1])
    # Build a regular target grid while forcing original levels to remain present.
    dense = np.arange(z0, z1 + 0.5 * dz_m, dz_m, dtype=float)
    dense = dense[(dense >= z0 - 1e-10) & (dense <= z1 + 1e-10)]
    zt = np.unique(np.concatenate([dense, z_original, [z1]]))
    zt.sort()
    return zt


def normalize_horizontal(xi, eta, xi_min, xi_max, eta_min, eta_max):
    sx = 2.0 * (np.asarray(xi) - xi_min) / (xi_max - xi_min) - 1.0
    sy = 2.0 * (np.asarray(eta) - eta_min) / (eta_max - eta_min) - 1.0
    return sx, sy


def wahba_surface(
    xi, eta, values, XI, ETA,
    xi_min, xi_max, eta_min, eta_max,
    smoothing,
):
    """2-D thin-plate smoothing spline on one vertical layer."""
    sx, sy = normalize_horizontal(xi, eta, xi_min, xi_max, eta_min, eta_max)
    gx, gy = normalize_horizontal(XI.ravel(), ETA.ravel(), xi_min, xi_max, eta_min, eta_max)

    pts = np.column_stack([sx, sy])
    qpts = np.column_stack([gx, gy])

    fit = RBFInterpolator(
        pts,
        np.asarray(values, dtype=float),
        kernel="thin_plate_spline",
        smoothing=float(smoothing),
        degree=1,
    )
    out = fit(qpts).reshape(XI.shape)
    if not np.all(np.isfinite(out)):
        raise RuntimeError("Wahba/thin-plate interpolation produced non-finite values.")
    return out


def process_snapshot_wahba(
    rec, xi_grid, eta_grid, z_targets,
    xi_min, xi_max, eta_min, eta_max,
    smoothing,
):
    """Wahba horizontal per original level, then linear vertical interpolation."""
    raw = rec["raw"]
    xi, eta = rec["xi"], rec["eta"]
    z = raw["z"]
    u, v = raw["u"], raw["v"]

    XI, ETA = np.meshgrid(xi_grid, eta_grid)
    z_orig = np.sort(np.unique(z))

    U_levels = []
    V_levels = []
    used = []

    for zl in z_orig:
        m = np.isclose(z, zl, atol=1e-8)
        if np.count_nonzero(m) < 6:
            continue
        U = wahba_surface(
            xi[m], eta[m], u[m], XI, ETA,
            xi_min, xi_max, eta_min, eta_max, smoothing,
        )
        V = wahba_surface(
            xi[m], eta[m], v[m], XI, ETA,
            xi_min, xi_max, eta_min, eta_max, smoothing,
        )
        U_levels.append(U)
        V_levels.append(V)
        used.append(float(zl))

    if len(used) < 2:
        raise RuntimeError(f"{rec['key']}: fewer than two valid vertical levels.")

    used = np.asarray(used, dtype=float)
    U_levels = np.stack(U_levels, axis=0)  # [nz0, ny, nx]
    V_levels = np.stack(V_levels, axis=0)

    # Linear interpolation in z, vectorized by flattening the horizontal grid.
    # np.interp is 1-D, so interpolate each horizontal column.
    nxy = U_levels.shape[1] * U_levels.shape[2]
    Uf = U_levels.reshape(len(used), nxy)
    Vf = V_levels.reshape(len(used), nxy)
    Uout = np.empty((len(z_targets), nxy), dtype=np.float64)
    Vout = np.empty_like(Uout)
    for j in range(nxy):
        Uout[:, j] = np.interp(z_targets, used, Uf[:, j])
        Vout[:, j] = np.interp(z_targets, used, Vf[:, j])

    Uout = Uout.reshape(len(z_targets), len(eta_grid), len(xi_grid))
    Vout = Vout.reshape(len(z_targets), len(eta_grid), len(xi_grid))
    return Uout, Vout


def rel_l2(pred, truth, eps=1e-12):
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return float(np.linalg.norm(pred - truth) / (np.linalg.norm(truth) + eps))


def rmse_uv(pred_uv, truth_uv):
    return float(np.sqrt(np.mean((np.asarray(pred_uv) - np.asarray(truth_uv)) ** 2)))


def evaluate_holdout_interpolant(
    rec, pchip_u, pchip_v, z_targets, eta_grid, xi_grid, tol_m=100.0
):
    """Evaluate TRAIN-only temporal interpolant against raw held-out POWER points."""
    h = float(rec["hours_from_start"])
    Ug = np.asarray(pchip_u(h), dtype=float)
    Vg = np.asarray(pchip_v(h), dtype=float)

    iu = RegularGridInterpolator(
        (z_targets, eta_grid, xi_grid), Ug,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    iv = RegularGridInterpolator(
        (z_targets, eta_grid, xi_grid), Vg,
        method="linear", bounds_error=False, fill_value=np.nan,
    )

    raw = rec["raw"]
    truth_all = np.column_stack([raw["u"], raw["v"]])

    # Permit tiny metric/projection differences at the common-domain boundary,
    # then clip only those sub-tolerance coordinate differences for evaluation.
    support = (
        (rec["xi"] >= xi_grid[0] - tol_m)
        & (rec["xi"] <= xi_grid[-1] + tol_m)
        & (rec["eta"] >= eta_grid[0] - tol_m)
        & (rec["eta"] <= eta_grid[-1] + tol_m)
        & (raw["z"] >= z_targets[0] - 1e-8)
        & (raw["z"] <= z_targets[-1] + 1e-8)
    )
    q = np.column_stack([
        np.clip(raw["z"][support], z_targets[0], z_targets[-1]),
        np.clip(rec["eta"][support], eta_grid[0], eta_grid[-1]),
        np.clip(rec["xi"][support], xi_grid[0], xi_grid[-1]),
    ])
    pred = np.column_stack([iu(q), iv(q)])
    truth = truth_all[support]
    finite = np.all(np.isfinite(pred), axis=1)
    pred = pred[finite]
    truth = truth[finite]
    if len(pred) == 0:
        raise RuntimeError(f"No supported raw points for holdout {rec['key']}")

    return {
        "n": int(len(pred)),
        "relL2_uv": rel_l2(pred, truth),
        "relL2_u": rel_l2(pred[:, 0], truth[:, 0]),
        "relL2_v": rel_l2(pred[:, 1], truth[:, 1]),
        "rmse_mps": rmse_uv(pred, truth),
    }


def build_raw_holdout(rec, domain, common, t_norm, tol_m=100.0):
    raw = rec["raw"]
    support = (
        (rec["xi"] >= common["xi_min_m"] - tol_m)
        & (rec["xi"] <= common["xi_max_m"] + tol_m)
        & (rec["eta"] >= common["eta_min_m"] - tol_m)
        & (rec["eta"] <= common["eta_max_m"] + tol_m)
        & (raw["z"] >= common["z_min_obs_m"] - 1e-8)
        & (raw["z"] <= common["z_max_obs_m"] + 1e-8)
    )
    x = rec["x"][support]
    y = rec["y"][support]
    z = raw["z"][support]
    u = raw["u"][support]
    v = raw["v"][support]

    xn = (x - domain["xmin_m"]) / domain["Lx_m"]
    yn = (y - domain["ymin_m"]) / domain["Ly_m"]
    zn = (z - domain["zmin_m"]) / domain["Lz_m"]
    tn = np.full_like(xn, float(t_norm))

    X = np.column_stack([xn, yn, zn, tn]).astype(np.float32)
    Y = np.column_stack([u, v]).astype(np.float32)
    xyz = np.column_stack([x, y, z]).astype(np.float32)
    return X, Y, xyz


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="melissa_data_nhc")
    p.add_argument("--centers", default="melissa_data_nhc/hourly_eye_centers.csv")
    p.add_argument("--outdir", default="melissa_hourly_prepared")
    p.add_argument("--dev_key", default="20251026T21")
    p.add_argument("--val_key", default="20251027T06")
    p.add_argument("--test_key", default="20251028T12")
    p.add_argument("--nx", type=int, default=40)
    p.add_argument("--ny", type=int, default=40)
    p.add_argument(
        "--dz", type=float, default=20.0,
        help="Vertical target spacing in m. <=0 preserves the 8 original POWER levels.",
    )
    p.add_argument(
        "--dt_minutes", type=float, default=15.0,
        help="Temporal spacing of generated training snapshots.",
    )
    p.add_argument(
        "--holdout_gap_minutes", type=float, default=30.0,
        help="Do not generate pseudo-data within +/- this gap around DEV/VAL/TEST.",
    )
    p.add_argument(
        "--wahba_lambda", type=float, default=1e-2,
        help="Thin-plate smoothing parameter after mapping horizontal coordinates to [-1,1]^2.",
    )
    p.add_argument("--max_height", type=float, default=300.0)
    p.add_argument("--support_tol_m", type=float, default=100.0)
    p.add_argument(
        "--overshoot_warn_factor", type=float, default=1.20,
        help="Warn if processed max(|u|,|v|) exceeds raw max by this factor.",
    )
    p.add_argument(
        "--diagnostics_only", action="store_true",
        help="Run final-pipeline DEV/VAL diagnostics but do not write dense train_X/train_Yuv.",
    )
    args = p.parse_args()

    if args.wahba_lambda < 0:
        raise ValueError("--wahba_lambda must be >=0")
    if args.nx < 3 or args.ny < 3:
        raise ValueError("--nx and --ny must be >=3")
    if args.dt_minutes <= 0:
        raise ValueError("--dt_minutes must be >0")

    os.makedirs(args.outdir, exist_ok=True)

    centers = load_eye_centers(args.centers)
    center_by_key = {r["key"]: r for r in centers}
    t0 = centers[0]["datetime"]
    tf = centers[-1]["datetime"]
    total_hours = (tf - t0).total_seconds() / 3600.0
    if total_hours <= 0:
        raise ValueError("Invalid centre trajectory time span.")

    # Fixed projection origin = first eye centre.  This frame is never moved.
    lat0_deg = float(centers[0]["lat"])
    lon0_deg = float(centers[0]["lon"])

    center_hours = []
    center_x = []
    center_y = []
    for c in centers:
        h = (c["datetime"] - t0).total_seconds() / 3600.0
        xc, yc = lonlat_to_xy(c["lon"], c["lat"], lon0_deg, lat0_deg)
        c["hours_from_start"] = float(h)
        c["x_m"] = float(np.asarray(xc))
        c["y_m"] = float(np.asarray(yc))
        center_hours.append(h)
        center_x.append(c["x_m"])
        center_y.append(c["y_m"])

    center_hours = np.asarray(center_hours, dtype=float)
    center_x = np.asarray(center_x, dtype=float)
    center_y = np.asarray(center_y, dtype=float)
    eye_x_of_h = PchipInterpolator(center_hours, center_x, extrapolate=False)
    eye_y_of_h = PchipInterpolator(center_hours, center_y, extrapolate=False)

    # Load all available POWER .dat files, one per date/hour.
    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.dat")))
    if not paths:
        raise FileNotFoundError(f"No .dat files found in {args.input_dir}")

    records = []
    seen = set()
    for path in paths:
        key, dt = parse_key_from_path(path)
        if key in seen:
            raise ValueError(f"Duplicate POWER snapshot key {key}")
        seen.add(key)
        if key not in center_by_key:
            raise KeyError(f"No eye centre metadata for POWER snapshot {key}")

        raw = load_raw(path, args.max_height)
        h = (dt - t0).total_seconds() / 3600.0
        t_norm = h / total_hours
        x, y = lonlat_to_xy(raw["lon"], raw["lat"], lon0_deg, lat0_deg)
        xc = float(eye_x_of_h(h))
        yc = float(eye_y_of_h(h))
        records.append({
            "key": key,
            "datetime": dt,
            "hours_from_start": float(h),
            "t_norm": float(t_norm),
            "path": path,
            "raw": raw,
            "x": x,
            "y": y,
            "xc_m": xc,
            "yc_m": yc,
            "xi": x - xc,
            "eta": y - yc,
        })

    records.sort(key=lambda r: r["datetime"])
    by_key = {r["key"]: r for r in records}

    holdout_keys = {k for k in [args.dev_key, args.val_key, args.test_key] if k}
    train = [r for r in records if r["key"] not in holdout_keys]
    if len(train) < 4:
        raise RuntimeError("Need at least four TRAIN snapshots.")

    # DEV and VAL must exist for development diagnostics.  Final TEST may be absent.
    for k, label in [(args.dev_key, "DEV"), (args.val_key, "VAL")]:
        if k and k not in by_key:
            raise FileNotFoundError(f"{label} snapshot {k} not found in {args.input_dir}")

    # Common storm-relative support from TRAIN geometry only.
    xi_min = max(float(np.min(r["xi"])) for r in train)
    xi_max = min(float(np.max(r["xi"])) for r in train)
    eta_min = max(float(np.min(r["eta"])) for r in train)
    eta_max = min(float(np.max(r["eta"])) for r in train)
    if xi_min >= xi_max or eta_min >= eta_max:
        raise RuntimeError("TRAIN snapshots have no common storm-relative horizontal support.")

    xi_grid = np.linspace(xi_min, xi_max, args.nx)
    eta_grid = np.linspace(eta_min, eta_max, args.ny)

    # Require a common set/range of observed heights; current downloader uses the same 8 levels.
    z_sets = [np.sort(np.unique(r["raw"]["z"])) for r in train]
    z_common = z_sets[0]
    for zs in z_sets[1:]:
        if len(zs) != len(z_common) or not np.allclose(zs, z_common, atol=1e-8, rtol=0):
            raise RuntimeError("TRAIN snapshots do not share the same original vertical levels.")
    z_targets = make_vertical_targets(z_common, args.dz)

    # Global TRAIN-only velocity scale from original raw observations.
    S_global = max(
        max(float(np.max(np.abs(r["raw"]["u"]))) for r in train),
        max(float(np.max(np.abs(r["raw"]["v"]))) for r in train),
        1e-6,
    )

    print("=" * 78)
    print("MELISSA HOURLY PREPROCESSING: WAHBA + LINEAR-z + PCHIP-t")
    print("=" * 78)
    print(f"POWER snapshots available : {len(records)}")
    print(f"TRAIN snapshots           : {len(train)}")
    print(f"DEV                       : {args.dev_key}")
    print(f"VAL                       : {args.val_key}")
    print(f"TEST final                : {args.test_key} "
          f"({'available but held out' if args.test_key in by_key else 'not downloaded / not used'})")
    print(f"Projection origin         : lat0={lat0_deg:.6f}, lon0={lon0_deg:.6f}")
    print("Common storm-relative TRAIN support:")
    print(f"  xi  = [{xi_min/1000:.3f}, {xi_max/1000:.3f}] km")
    print(f"  eta = [{eta_min/1000:.3f}, {eta_max/1000:.3f}] km")
    print(f"Original z levels         : {z_common.tolist()}")
    print(f"Refined z levels ({len(z_targets)}): {z_targets.tolist()}")
    print(f"Wahba lambda              : {args.wahba_lambda:g}")
    print(f"Global TRAIN scale S      : {S_global:.6f} m/s")
    print("=" * 78)

    # Spatial + vertical preprocessing for each TRAIN snapshot.
    U_hourly = np.empty((len(train), len(z_targets), args.ny, args.nx), dtype=np.float32)
    V_hourly = np.empty_like(U_hourly)
    train_hours = np.array([r["hours_from_start"] for r in train], dtype=float)
    train_keys = [r["key"] for r in train]

    for i, r in enumerate(train):
        U, V = process_snapshot_wahba(
            r, xi_grid, eta_grid, z_targets,
            xi_min, xi_max, eta_min, eta_max,
            args.wahba_lambda,
        )
        U_hourly[i] = U.astype(np.float32)
        V_hourly[i] = V.astype(np.float32)

        raw_max = max(
            float(np.max(np.abs(r["raw"]["u"]))),
            float(np.max(np.abs(r["raw"]["v"]))),
        )
        proc_max = max(float(np.max(np.abs(U))), float(np.max(np.abs(V))))
        tag = ""
        if proc_max > args.overshoot_warn_factor * max(raw_max, 1e-12):
            tag = "  [WARN: smoothing overshoot]"
        if (i + 1) % 10 == 0 or i == 0 or i + 1 == len(train):
            print(
                f"Processed TRAIN {i+1:2d}/{len(train)}: {r['key']}  "
                f"raw_max={raw_max:.3f}  proc_max={proc_max:.3f}{tag}"
            )

    # Save the processed REAL hourly anchors. This is small and lets us change dt later
    # without recomputing Wahba.
    np.savez_compressed(
        os.path.join(args.outdir, "train_hourly_processed.npz"),
        keys=np.asarray(train_keys),
        hours=train_hours,
        xi_grid_m=xi_grid,
        eta_grid_m=eta_grid,
        z_grid_m=z_targets,
        U_phys_mps=U_hourly,
        V_phys_mps=V_hourly,
        S_global_mps=np.asarray(S_global, dtype=np.float64),
    )

    # TRAIN-only temporal interpolation in PHYSICAL m/s.
    pchip_u = PchipInterpolator(train_hours, U_hourly.astype(np.float64), axis=0, extrapolate=False)
    pchip_v = PchipInterpolator(train_hours, V_hourly.astype(np.float64), axis=0, extrapolate=False)

    diagnostics = {}
    for label, key in [("DEV", args.dev_key), ("VAL", args.val_key)]:
        if key and key in by_key:
            m = evaluate_holdout_interpolant(
                by_key[key], pchip_u, pchip_v,
                z_targets, eta_grid, xi_grid,
                tol_m=args.support_tol_m,
            )
            diagnostics[label.lower()] = {"key": key, **m}
            print(
                f"{label} interpolation check {key}: "
                f"n={m['n']} relL2_uv={m['relL2_uv']:.6e} "
                f"RMSE={m['rmse_mps']:.6f} m/s "
                f"(u={m['relL2_u']:.6e}, v={m['relL2_v']:.6e})"
            )

    if args.test_key and args.test_key in by_key:
        print("Final TEST field is available but its interpolation diagnostic is intentionally NOT printed.")

    # Dense temporal targets in hours, with holdout exclusion zones.
    h_dense = np.arange(0.0, total_hours + 0.5 * args.dt_minutes / 60.0, args.dt_minutes / 60.0)
    h_dense = h_dense[h_dense <= total_hours + 1e-10]
    gap_h = max(0.0, args.holdout_gap_minutes / 60.0)
    keep = np.ones(len(h_dense), dtype=bool)
    for key in holdout_keys:
        if key in center_by_key:
            hh = (center_by_key[key]["datetime"] - t0).total_seconds() / 3600.0
            keep &= np.abs(h_dense - hh) > gap_h + 1e-12
    h_out = h_dense[keep]

    # PCHIP requires interpolation, not extrapolation. Endpoints should be TRAIN.
    hmin_train, hmax_train = float(train_hours[0]), float(train_hours[-1])
    inside = (h_out >= hmin_train - 1e-10) & (h_out <= hmax_train + 1e-10)
    h_out = h_out[inside]
    t_out = h_out / total_hours

    # Eye trajectory is external metadata; it can be evaluated at generated times.
    xc_out = np.asarray(eye_x_of_h(h_out), dtype=float)
    yc_out = np.asarray(eye_y_of_h(h_out), dtype=float)
    if not np.all(np.isfinite(xc_out)) or not np.all(np.isfinite(yc_out)):
        raise RuntimeError("Eye trajectory interpolation failed at generated times.")

    # Physical NN domain is the union of the moving common storm-relative support.
    xmin_m = float(np.min(xc_out + xi_min))
    xmax_m = float(np.max(xc_out + xi_max))
    ymin_m = float(np.min(yc_out + eta_min))
    ymax_m = float(np.max(yc_out + eta_max))
    zmin_m = 0.0
    zmax_m = float(np.max(z_targets))
    domain = {
        "projection": "fixed_local_equirectangular",
        "earth_radius_m": EARTH_RADIUS_M,
        "lon0_deg": lon0_deg,
        "lat0_deg": lat0_deg,
        "xmin_m": xmin_m,
        "xmax_m": xmax_m,
        "ymin_m": ymin_m,
        "ymax_m": ymax_m,
        "zmin_m": zmin_m,
        "zmax_m": zmax_m,
        "Lx_m": xmax_m - xmin_m,
        "Ly_m": ymax_m - ymin_m,
        "Lz_m": zmax_m - zmin_m,
        "time_origin_utc": t0.isoformat() + "Z",
        "time_end_utc": tf.isoformat() + "Z",
        "total_hours": total_hours,
    }
    if min(domain["Lx_m"], domain["Ly_m"], domain["Lz_m"]) <= 0:
        raise RuntimeError("Degenerate physical network domain.")

    common = {
        "xi_min_m": xi_min,
        "xi_max_m": xi_max,
        "eta_min_m": eta_min,
        "eta_max_m": eta_max,
        "z_min_obs_m": float(np.min(z_common)),
        "z_max_obs_m": float(np.max(z_common)),
    }

    # Save raw DEV/VAL and, only when present, TEST. Targets remain raw POWER values.
    holdout_files = {}
    for label, key, filename in [
        ("dev", args.dev_key, "dev_raw.npz"),
        ("val", args.val_key, "val_raw.npz"),
        ("test", args.test_key, "test_raw.npz"),
    ]:
        if not key or key not in by_key:
            continue
        rec = by_key[key]
        Xh, Yh, xyz = build_raw_holdout(
            rec, domain, common, rec["t_norm"], tol_m=args.support_tol_m
        )
        np.savez(
            os.path.join(args.outdir, filename),
            X=Xh,
            Yuv_phys=Yh,
            xyz_phys=xyz,
            S_global_mps=np.asarray(S_global, dtype=np.float32),
            key=np.asarray(key),
        )
        holdout_files[label] = filename
        print(f"Saved {label.upper()} raw set: {filename}, n={len(Xh)}")

    npts = len(z_targets) * args.ny * args.nx
    nrows = len(h_out) * npts

    if not args.diagnostics_only:
        Xpath = os.path.join(args.outdir, "train_X.npy")
        Ypath = os.path.join(args.outdir, "train_Yuv.npy")
        Xmm = open_memmap(Xpath, mode="w+", dtype=np.float32, shape=(nrows, 4))
        Ymm = open_memmap(Ypath, mode="w+", dtype=np.float32, shape=(nrows, 2))

        # Fixed storm-relative coordinates, z-major ordering matching U/V arrays.
        Z3, ETA3, XI3 = np.meshgrid(z_targets, eta_grid, xi_grid, indexing="ij")
        xi_flat = XI3.ravel()
        eta_flat = ETA3.ravel()
        z_flat = Z3.ravel()
        zn_flat = (z_flat - domain["zmin_m"]) / domain["Lz_m"]

        for k, (h, tn, xc, yc) in enumerate(zip(h_out, t_out, xc_out, yc_out)):
            U = np.asarray(pchip_u(h), dtype=np.float64).ravel()
            V = np.asarray(pchip_v(h), dtype=np.float64).ravel()
            x = xi_flat + xc
            y = eta_flat + yc
            xn = (x - domain["xmin_m"]) / domain["Lx_m"]
            yn = (y - domain["ymin_m"]) / domain["Ly_m"]

            sl = slice(k * npts, (k + 1) * npts)
            Xmm[sl, 0] = xn.astype(np.float32)
            Xmm[sl, 1] = yn.astype(np.float32)
            Xmm[sl, 2] = zn_flat.astype(np.float32)
            Xmm[sl, 3] = np.float32(tn)
            Ymm[sl, 0] = (U / S_global).astype(np.float32)
            Ymm[sl, 1] = (V / S_global).astype(np.float32)

            if (k + 1) % 25 == 0 or k == 0 or k + 1 == len(h_out):
                print(f"Generated dense time {k+1:3d}/{len(h_out)}: h={h:6.2f}, t={tn:.6f}")

        Xmm.flush()
        Ymm.flush()
        del Xmm, Ymm

        np.savetxt(
            os.path.join(args.outdir, "train_times.csv"),
            np.column_stack([h_out, t_out, xc_out, yc_out]),
            delimiter=",",
            header="hours_from_start,t_norm,eye_x_m,eye_y_m",
            comments="",
            fmt="%.12e",
        )

    # Track data saved for moving-support physics collocation in the trainer.
    np.savez(
        os.path.join(args.outdir, "eye_path.npz"),
        hours=center_hours,
        t_norm=center_hours / total_hours,
        x_m=center_x,
        y_m=center_y,
    )

    manifest = {
        "pipeline": "fixed metric frame -> eye registration -> Wahba horizontal -> linear vertical -> TRAIN-only PCHIP temporal -> physical frame -> [0,1]^4",
        "train_keys": train_keys,
        "dev_key": args.dev_key,
        "val_key": args.val_key,
        "test_key_reserved": args.test_key,
        "test_field_available": bool(args.test_key in by_key) if args.test_key else False,
        "n_power_snapshots_available": len(records),
        "n_train_snapshots": len(train),
        "external_eye_metadata": {
            "source_file": args.centers,
            "used_for_registration_not_as_wind_target": True,
            "may_include_reserved_test_time": True,
        },
        "domain": domain,
        "storm_relative_common_support": common,
        "velocity_normalization": {
            "type": "single global TRAIN-only constant",
            "S_global_mps": S_global,
            "definition": "max over TRAIN raw observations of max(|u|,|v|)",
            "rbf_scale_interpolation": False,
        },
        "wind_reconstruction": {
            "speed_source": "NASA POWER WSC",
            "direction_source": "NASA POWER WD10M meteorological FROM direction",
            "u_east": "-WSC*sin(WD10M*pi/180)",
            "v_north": "-WSC*cos(WD10M*pi/180)",
            "WD10M_note": "common 10-m direction combined with requested-height WSC",
        },
        "preprocessing": {
            "horizontal_method": "Wahba thin-plate smoothing spline per original z level",
            "scipy_implementation": "RBFInterpolator(kernel='thin_plate_spline', degree=1)",
            "wahba_lambda": args.wahba_lambda,
            "horizontal_fit_coordinates": "storm-relative xi,eta rescaled to [-1,1]^2",
            "nx": args.nx,
            "ny": args.ny,
            "vertical_method": "linear interpolation between horizontally smoothed layers",
            "dz_requested_m": args.dz,
            "z_targets_m": z_targets.tolist(),
            "temporal_method": "PCHIP at fixed storm-relative (xi,eta,z), TRAIN wind fields only",
            "dt_minutes": args.dt_minutes,
            "holdout_gap_minutes": args.holdout_gap_minutes,
            "n_generated_times": int(len(h_out)),
            "n_points_per_generated_time": int(npts),
            "n_train_rows": int(nrows) if not args.diagnostics_only else 0,
            "diagnostics_only": bool(args.diagnostics_only),
            "raw_holdouts_not_smoothed": True,
        },
        "holdout_files": holdout_files,
        "interpolation_diagnostics": diagnostics,
    }

    with open(os.path.join(args.outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 78)
    print("[OK] Melissa hourly preprocessing finished")
    print(f"Processed hourly anchors : {len(train)}")
    print(f"Generated times          : {len(h_out)}")
    print(f"Points/time              : {npts}")
    if not args.diagnostics_only:
        print(f"Training rows            : {nrows}")
        print(f"train_X                  : {os.path.join(args.outdir, 'train_X.npy')}")
        print(f"train_Yuv                : {os.path.join(args.outdir, 'train_Yuv.npy')}")
    else:
        print("Dense training arrays    : skipped (--diagnostics_only)")
    print(f"Global velocity scale S  : {S_global:.6f} m/s")
    print(
        f"Physical domain          : Lx={domain['Lx_m']/1000:.3f} km, "
        f"Ly={domain['Ly_m']/1000:.3f} km, Lz={domain['Lz_m']:.3f} m"
    )
    print(f"Output                   : {args.outdir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
