#!/usr/bin/env python3

"""
download_melissa_power_hourly.py

Descarga snapshots horarios de NASA POWER para Hurricane Melissa.

Intervalo por defecto:
    20251026T09 -- 20251028T21

Características
---------------
- Una instantánea cada hora.
- Ventana móvil +/-2 grados alrededor del ojo.
- Centro del ojo interpolado a partir de posiciones NHC conocidas.
- Malla horizontal 10x10.
- Niveles:
      10, 50, 90, 130, 170, 210, 250, 290 m
- Variables:
      WSC + WD10M
- Convención meteorológica:
      u = -WSC sin(theta)
      v = -WSC cos(theta)
- Detecta y salta snapshots completos ya existentes.
- Reintentos automáticos.
- Escritura atómica: no deja .dat incompletos.
- Si un snapshot falla, continúa con los siguientes.
- Se puede volver a ejecutar para completar únicamente lo faltante.

IMPORTANTE
----------
Por defecto se excluye:

    20251028T12

para reservarlo como TEST final sin inspeccionar.

Ejemplo:
    python3 download_melissa_power_hourly.py \
        --outdir melissa_data_nhc \
        --workers 2

Dry run:
    python3 download_melissa_power_hourly.py \
        --outdir melissa_data_nhc \
        --workers 2 \
        --dry-run
"""

import argparse
import csv
import glob
import io
import os
import re
import sys
import threading
import time

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

from datetime import (
    datetime,
    timedelta,
    timezone,
)

import numpy as np
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from scipy.interpolate import PchipInterpolator


# ============================================================
# NASA POWER
# ============================================================

POWER_URL = (
    "https://power.larc.nasa.gov/api/temporal/hourly/point"
)


# ============================================================
# NHC CENTERS
#
# (time, latitude, longitude)
#
# Estas son las posiciones que ya estamos utilizando
# en el experimento.
# ============================================================

CENTER_ANCHORS = [

    ("20251026T09", 16.3, -76.3),

    ("20251026T15", 16.4, -76.6),

    ("20251026T21", 16.4, -77.2),

    ("20251027T03", 16.3, -77.5),

    ("20251027T06", 16.3, -77.7),

    ("20251027T09", 16.4, -77.8),

    ("20251027T15", 16.4, -78.2),

    ("20251027T21", 16.7, -78.4),

    ("20251028T03", 16.6, -78.5),

    ("20251028T09", 17.2, -78.3),

    ("20251028T15", 17.9, -77.9),

    ("20251028T21", 18.5, -77.7),
]


DEFAULT_START = "20251026T09"
DEFAULT_END   = "20251028T21"

# Lo dejamos completamente fuera por ahora.
DEFAULT_EXCLUDE = "20251028T12"


_thread_local = threading.local()


# ============================================================
# TIME UTILITIES
# ============================================================

def parse_time(key):

    return datetime.strptime(
        key,
        "%Y%m%dT%H"
    ).replace(
        tzinfo=timezone.utc
    )


def format_time(dt):

    return dt.strftime(
        "%Y%m%dT%H"
    )


def hourly_times(
    start_key,
    end_key
):

    t0 = parse_time(
        start_key
    )

    tf = parse_time(
        end_key
    )

    if tf < t0:

        raise ValueError(
            "end debe ser posterior a start."
        )

    out = []

    t = t0

    while t <= tf:

        out.append(
            format_time(t)
        )

        t += timedelta(
            hours=1
        )

    return out


def parse_key_list(text):

    if text is None:
        return []

    vals = [

        x.strip()

        for x in text.split(",")

        if x.strip()
    ]

    for x in vals:
        parse_time(x)

    return sorted(
        set(vals),
        key=parse_time
    )


# ============================================================
# CONTINUOUS EYE TRAJECTORY
# ============================================================

def build_eye_interpolators():

    t0 = parse_time(
        CENTER_ANCHORS[0][0]
    )

    times_h = np.array(
        [
            (
                parse_time(k)
                - t0
            ).total_seconds()
            / 3600.0

            for k, _, _
            in CENTER_ANCHORS
        ],
        dtype=float
    )

    lats = np.array(
        [
            lat

            for _, lat, _
            in CENTER_ANCHORS
        ],
        dtype=float
    )

    lons = np.array(
        [
            lon

            for _, _, lon
            in CENTER_ANCHORS
        ],
        dtype=float
    )

    lat_interp = PchipInterpolator(
        times_h,
        lats,
        extrapolate=False
    )

    lon_interp = PchipInterpolator(
        times_h,
        lons,
        extrapolate=False
    )

    return (
        t0,
        lat_interp,
        lon_interp,
    )


TRACK_T0, TRACK_LAT, TRACK_LON = (
    build_eye_interpolators()
)


def interpolate_center(key):

    dt = parse_time(
        key
    )

    h = (
        dt - TRACK_T0
    ).total_seconds() / 3600.0

    lat = TRACK_LAT(h)
    lon = TRACK_LON(h)

    if (
        not np.isfinite(lat)
        or
        not np.isfinite(lon)
    ):

        raise ValueError(
            f"{key} fuera del intervalo "
            "de la trayectoria NHC."
        )

    return (
        float(lat),
        float(lon)
    )


# ============================================================
# HTTP SESSION
# ============================================================

def get_session():

    if not hasattr(
        _thread_local,
        "session"
    ):

        s = requests.Session()

        retry = Retry(

            total=8,

            connect=8,

            read=8,

            status=8,

            backoff_factor=1.5,

            status_forcelist=(
                429,
                500,
                502,
                503,
                504,
            ),

            allowed_methods=frozenset(
                ["GET"]
            ),

            raise_on_status=False,

            respect_retry_after_header=True,
        )

        adapter = HTTPAdapter(

            max_retries=retry,

            pool_connections=8,

            pool_maxsize=8,
        )

        s.mount(
            "https://",
            adapter
        )

        s.headers.update(
            {
                "User-Agent":
                    "Melissa-DeepWind-research/2.0",

                "Accept":
                    "text/csv,*/*",
            }
        )

        _thread_local.session = s

    return _thread_local.session


# ============================================================
# PARSE POWER CSV
# ============================================================

def parse_power_csv(
    text,
    target_hour
):

    marker = "-END HEADER-"

    if marker not in text:

        raise RuntimeError(
            "Respuesta POWER sin '-END HEADER-'. "
            f"Inicio: {text[:300]!r}"
        )

    body = text.split(
        marker,
        1
    )[1].strip()

    reader = csv.DictReader(
        io.StringIO(body)
    )

    if reader.fieldnames is None:

        raise RuntimeError(
            "POWER CSV sin encabezados."
        )

    fieldnames = [

        f.strip()

        for f in reader.fieldnames
    ]

    required = {
        "HR",
        "WSC",
        "WD10M",
    }

    missing = (
        required
        -
        set(fieldnames)
    )

    if missing:

        raise RuntimeError(
            f"Faltan columnas {missing}. "
            f"Columnas POWER: {fieldnames}"
        )

    for raw_row in reader:

        row = {

            (
                k.strip()
                if k is not None
                else k
            ):
            (
                v.strip()
                if isinstance(v, str)
                else v
            )

            for k, v
            in raw_row.items()
        }

        try:

            hr = int(
                float(
                    row["HR"]
                )
            )

        except Exception:

            continue

        if hr != target_hour:
            continue

        try:

            speed = float(
                row["WSC"]
            )

            direction = float(
                row["WD10M"]
            )

        except Exception as exc:

            raise RuntimeError(
                f"WSC/WD10M invalidos: {row}"
            ) from exc

        if (
            not np.isfinite(speed)
            or
            not np.isfinite(direction)
        ):

            raise RuntimeError(
                f"WSC/WD10M no finitos: {row}"
            )

        if (
            speed < 0.0
            or speed > 150.0
        ):

            raise RuntimeError(
                f"WSC fuera de rango: "
                f"{speed} m/s"
            )

        if (
            direction < 0.0
            or direction > 360.0
        ):

            raise RuntimeError(
                f"WD10M fuera de rango: "
                f"{direction}"
            )

        return (
            speed,
            direction
        )

    raise RuntimeError(
        f"No se encontro HR={target_hour} "
        "en la respuesta POWER."
    )


# ============================================================
# DOWNLOAD ONE SPACE/HEIGHT POINT
# ============================================================

def fetch_one(
    date,
    hour,
    lat,
    lon,
    elevation,
    timeout
):

    params = {

        "start":
            date,

        "end":
            date,

        "latitude":
            f"{lat:.8f}",

        "longitude":
            f"{lon:.8f}",

        "community":
            "AG",

        # WSC aparece al solicitar wind-elevation.
        "parameters":
            "WS10M,WD10M,PS",

        "format":
            "CSV",

        "theme":
            "light",

        "user":
            "DAVE",

        "time-standard":
            "UTC",

        "wind-surface":
            "openwater",

        "wind-elevation":
            str(
                int(elevation)
            ),
    }

    session = get_session()

    response = session.get(

        POWER_URL,

        params=params,

        timeout=timeout
    )

    if response.status_code != 200:

        raise RuntimeError(
            f"POWER HTTP "
            f"{response.status_code}: "
            f"{response.text[:300]!r}"
        )

    speed, direction = parse_power_csv(

        response.text,

        hour
    )

    theta = np.deg2rad(
        direction
    )

    # WD10M = FROM direction.
    u = (
        -speed
        * np.sin(theta)
    )

    v = (
        -speed
        * np.cos(theta)
    )

    w = 0.0

    return (
        u,
        v,
        w,
        speed,
        direction,
    )


# ============================================================
# FILE HANDLING
# ============================================================

def filename_for_snapshot(
    outdir,
    clat,
    clon,
    date,
    hour
):

    lat_tag = (
        f"{clat:.4f}"
        .replace(".", "_")
    )

    lon_tag = (
        f"{clon:.4f}"
        .replace(".", "_")
    )

    name = (

        f"lat{lat_tag}_"
        f"lon{lon_tag}_"
        f"date{date}_"
        f"hour{hour:02d}.dat"
    )

    return os.path.join(
        outdir,
        name
    )


def existing_candidates(
    outdir,
    key
):

    dt = parse_time(key)

    date = dt.strftime(
        "%Y%m%d"
    )

    hour = dt.hour

    pattern = os.path.join(

        outdir,

        f"*date{date}_hour{hour:02d}.dat"
    )

    return glob.glob(
        pattern
    )


def validate_existing_snapshot(
    path,
    n,
    heights
):

    try:

        data = np.loadtxt(
            path
        )

    except Exception:

        return False

    if data.ndim == 1:

        data = data.reshape(
            1,
            -1
        )

    expected_rows = (
        n
        * n
        * len(heights)
    )

    if data.shape[0] != expected_rows:

        return False

    if data.shape[1] < 8:

        return False

    if not np.all(
        np.isfinite(data)
    ):

        return False

    z = np.unique(
        data[:, 2]
    )

    z = np.sort(z)

    expected_z = np.asarray(
        heights,
        dtype=float
    )

    if (
        len(z) != len(expected_z)
        or
        not np.allclose(
            z,
            expected_z
        )
    ):

        return False

    return True


def find_complete_snapshot(
    outdir,
    key,
    n,
    heights
):

    files = existing_candidates(
        outdir,
        key
    )

    for path in files:

        if validate_existing_snapshot(
            path,
            n,
            heights
        ):

            return path

    return None


# ============================================================
# DOWNLOAD ONE SNAPSHOT
# ============================================================

def download_snapshot(
    key,
    outdir,
    n,
    half_span_lat,
    half_span_lon,
    heights,
    workers,
    timeout
):

    dt = parse_time(
        key
    )

    date = dt.strftime(
        "%Y%m%d"
    )

    hour = dt.hour

    clat, clon = interpolate_center(
        key
    )

    outpath = filename_for_snapshot(

        outdir,

        clat,

        clon,

        date,

        hour
    )

    lat0 = (
        clat
        - half_span_lat
    )

    lat1 = (
        clat
        + half_span_lat
    )

    lon0 = (
        clon
        - half_span_lon
    )

    lon1 = (
        clon
        + half_span_lon
    )

    lats = np.linspace(
        lat0,
        lat1,
        n
    )

    lons = np.linspace(
        lon0,
        lon1,
        n
    )

    jobs = []

    idx = 0

    for z in heights:

        for lat in lats:

            for lon in lons:

                jobs.append(
                    (
                        idx,
                        float(lat),
                        float(lon),
                        int(z),
                    )
                )

                idx += 1

    total = len(
        jobs
    )

    rows = [
        None
    ] * total

    print()

    print("=" * 78)

    print(
        f"SNAPSHOT {key}"
    )

    print(
        f"center = "
        f"({clat:.5f}, {clon:.5f})"
    )

    print(
        f"lat = "
        f"[{lat0:.5f}, {lat1:.5f}]"
    )

    print(
        f"lon = "
        f"[{lon0:.5f}, {lon1:.5f}]"
    )

    print(
        f"requests = {total}"
    )

    print("=" * 78)

    def task(job):

        j, lat, lon, z = job

        u, v, w, speed, direction = (
            fetch_one(

                date,

                hour,

                lat,

                lon,

                z,

                timeout
            )
        )

        return (
            j,
            (
                lat,
                lon,
                z,
                u,
                v,
                w,
                speed,
                direction,
            )
        )

    completed = 0

    t_start = time.time()

    with ThreadPoolExecutor(
        max_workers=workers
    ) as pool:

        futures = {

            pool.submit(
                task,
                job
            ):
            job

            for job
            in jobs
        }

        for fut in as_completed(
            futures
        ):

            job = futures[fut]

            try:

                j, row = fut.result()

            except Exception as exc:

                for other in futures:
                    other.cancel()

                _, lat, lon, z = job

                raise RuntimeError(
                    f"Fallo en {key}: "
                    f"lat={lat:.6f}, "
                    f"lon={lon:.6f}, "
                    f"z={z}"
                ) from exc

            rows[j] = row

            completed += 1

            if (
                completed % 50 == 0
                or
                completed == total
            ):

                elapsed = (
                    time.time()
                    - t_start
                )

                rate = (
                    completed
                    /
                    max(
                        elapsed,
                        1e-9
                    )
                )

                remaining = (
                    total - completed
                ) / max(
                    rate,
                    1e-9
                )

                print(
                    f"  {completed:4d}/{total} "
                    f"({100.0*completed/total:5.1f}%) "
                    f"rate={rate:5.2f} req/s "
                    f"ETA~{remaining/60.0:5.1f} min"
                )

    if any(
        r is None
        for r in rows
    ):

        raise RuntimeError(
            f"{key}: snapshot incompleto."
        )

    arr = np.asarray(
        rows,
        dtype=float
    )

    expected_shape = (
        n * n * len(heights),
        8
    )

    if arr.shape != expected_shape:

        raise RuntimeError(
            f"{key}: forma {arr.shape}, "
            f"esperada {expected_shape}"
        )

    if not np.all(
        np.isfinite(arr)
    ):

        raise RuntimeError(
            f"{key}: NaN/Inf."
        )

    os.makedirs(
        outdir,
        exist_ok=True
    )

    tmp = (
        outpath
        + ".part"
    )

    np.savetxt(

        tmp,

        arr,

        header=(
            "lat lon z "
            "u v w "
            "speed direction"
        ),

        fmt="%.6f"
    )

    os.replace(
        tmp,
        outpath
    )

    print(
        f"[OK] {key}: "
        f"{outpath}"
    )

    return outpath


# ============================================================
# MAIN
# ============================================================

def main():

    p = argparse.ArgumentParser(
        description=(
            "NASA POWER hourly downloader "
            "for Hurricane Melissa."
        )
    )

    p.add_argument(
        "--outdir",
        default="melissa_data_nhc"
    )

    p.add_argument(
        "--start",
        default=DEFAULT_START
    )

    p.add_argument(
        "--end",
        default=DEFAULT_END
    )

    p.add_argument(
        "--exclude",
        default=DEFAULT_EXCLUDE,
        help=(
            "Horas separadas por coma que NO se descargaran. "
            "Default: 20251028T12 (TEST final)."
        )
    )

    p.add_argument(
        "--n",
        type=int,
        default=10
    )

    p.add_argument(
        "--half-span-lat",
        type=float,
        default=2.0
    )

    p.add_argument(
        "--half-span-lon",
        type=float,
        default=2.0
    )

    p.add_argument(
        "--min-elev",
        type=int,
        default=10
    )

    p.add_argument(
        "--max-elev",
        type=int,
        default=290
    )

    p.add_argument(
        "--step-elev",
        type=int,
        default=40
    )

    p.add_argument(
        "--workers",
        type=int,
        default=2,
        help=(
            "Recomiendo 2 para una descarga larga. "
            "Puede probarse 4 si POWER responde bien."
        )
    )

    p.add_argument(
        "--timeout",
        type=float,
        default=90.0
    )

    p.add_argument(
        "--snapshot-retries",
        type=int,
        default=3
    )

    p.add_argument(
        "--snapshot-pause",
        type=float,
        default=5.0
    )

    p.add_argument(
        "--dry-run",
        action="store_true"
    )

    args = p.parse_args()

    if args.n < 2:

        raise ValueError(
            "--n debe ser >= 2."
        )

    if args.workers < 1:

        raise ValueError(
            "--workers debe ser >= 1."
        )

    if args.step_elev <= 0:

        raise ValueError(
            "--step-elev debe ser > 0."
        )

    heights = list(
        range(
            args.min_elev,
            args.max_elev + 1,
            args.step_elev
        )
    )

    expected_heights = [
        10,
        50,
        90,
        130,
        170,
        210,
        250,
        290,
    ]

    if heights != expected_heights:

        print(
            "[WARNING] Heights are not "
            "the standard Melissa levels:",
            heights
        )

    times = hourly_times(
        args.start,
        args.end
    )

    excluded = set(
        parse_key_list(
            args.exclude
        )
    )

    times = [
        key
        for key in times
        if key not in excluded
    ]

    os.makedirs(
        args.outdir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Guardar trayectoria horaria que usamos para centrar
    # las ventanas.
    # --------------------------------------------------------

    centers_path = os.path.join(
        args.outdir,
        "hourly_eye_centers.csv"
    )

    with open(
        centers_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "key",
                "eye_lat_deg",
                "eye_lon_deg",
                "excluded",
            ]
        )

        all_times = hourly_times(
            args.start,
            args.end
        )

        for key in all_times:

            lat, lon = interpolate_center(
                key
            )

            writer.writerow(
                [
                    key,
                    f"{lat:.8f}",
                    f"{lon:.8f}",
                    int(
                        key in excluded
                    ),
                ]
            )

    # --------------------------------------------------------
    # Detectar qué ya existe.
    # --------------------------------------------------------

    existing = []
    missing = []

    for key in times:

        found = find_complete_snapshot(
            args.outdir,
            key,
            args.n,
            heights
        )

        if found is not None:

            existing.append(
                (
                    key,
                    found
                )
            )

        else:

            missing.append(
                key
            )

    requests_per_snapshot = (
        args.n
        *
        args.n
        *
        len(heights)
    )

    print()

    print("=" * 78)

    print(
        "NASA POWER MELISSA HOURLY DOWNLOAD"
    )

    print("=" * 78)

    print(
        f"Interval : "
        f"{args.start} -> {args.end}"
    )

    print(
        f"Hours    : "
        f"{len(hourly_times(args.start, args.end))}"
    )

    print(
        f"Excluded : "
        f"{sorted(excluded)}"
    )

    print(
        f"Requested: "
        f"{len(times)} snapshots"
    )

    print(
        f"Existing : "
        f"{len(existing)}"
    )

    print(
        f"Missing  : "
        f"{len(missing)}"
    )

    print(
        f"Grid     : "
        f"{args.n} x {args.n}"
    )

    print(
        f"Heights  : "
        f"{heights}"
    )

    print(
        f"Requests/snapshot: "
        f"{requests_per_snapshot}"
    )

    print(
        f"Maximum new requests: "
        f"{len(missing) * requests_per_snapshot}"
    )

    print(
        f"Workers  : "
        f"{args.workers}"
    )

    print(
        f"Centers  : "
        f"{centers_path}"
    )

    print("=" * 78)

    print()

    if existing:

        print(
            "Already complete:"
        )

        for key, path in existing:

            print(
                f"  [SKIP] {key}  {path}"
            )

    print()

    if missing:

        print(
            "Snapshots to download:"
        )

        for key in missing:

            lat, lon = interpolate_center(
                key
            )

            print(
                f"  {key}  "
                f"eye~({lat:.4f}, {lon:.4f})"
            )

    else:

        print(
            "No missing snapshots."
        )

    if args.dry_run:

        print()

        print(
            "[DRY RUN] No data downloaded."
        )

        return

    # --------------------------------------------------------
    # Download missing snapshots.
    # --------------------------------------------------------

    failed = []

    completed_files = []

    total_missing = len(
        missing
    )

    for i, key in enumerate(
        missing,
        start=1
    ):

        print()

        print(
            f"######## SNAPSHOT "
            f"{i}/{total_missing}: "
            f"{key} ########"
        )

        success = False

        last_error = None

        for attempt in range(
            1,
            args.snapshot_retries + 1
        ):

            try:

                path = download_snapshot(

                    key=key,

                    outdir=args.outdir,

                    n=args.n,

                    half_span_lat=
                        args.half_span_lat,

                    half_span_lon=
                        args.half_span_lon,

                    heights=heights,

                    workers=args.workers,

                    timeout=args.timeout,
                )

                completed_files.append(
                    path
                )

                success = True

                break

            except KeyboardInterrupt:

                print(
                    "\nInterrupted by user."
                )

                raise

            except Exception as exc:

                last_error = exc

                print()

                print(
                    f"[ERROR] {key}, "
                    f"attempt "
                    f"{attempt}/"
                    f"{args.snapshot_retries}"
                )

                print(
                    f"        {exc}"
                )

                if (
                    attempt
                    < args.snapshot_retries
                ):

                    wait_s = (
                        30.0
                        * attempt
                    )

                    print(
                        f"Retry in "
                        f"{wait_s:.0f} s..."
                    )

                    time.sleep(
                        wait_s
                    )

        if not success:

            failed.append(
                (
                    key,
                    str(last_error)
                )
            )

            print(
                f"[FAILED] {key}. "
                "Continuing with next snapshot."
            )

        if (
            i < total_missing
            and args.snapshot_pause > 0
        ):

            time.sleep(
                args.snapshot_pause
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()

    print("=" * 78)

    print(
        "HOURLY DOWNLOAD FINISHED"
    )

    print("=" * 78)

    print(
        f"Already existed : "
        f"{len(existing)}"
    )

    print(
        f"Downloaded now  : "
        f"{len(completed_files)}"
    )

    print(
        f"Failed          : "
        f"{len(failed)}"
    )

    print(
        f"Excluded        : "
        f"{sorted(excluded)}"
    )

    if failed:

        print()

        print(
            "FAILED SNAPSHOTS:"
        )

        for key, err in failed:

            print(
                f"  {key}: {err}"
            )

        print()

        print(
            "Rerun the same command. "
            "Complete snapshots will be skipped."
        )

        sys.exit(1)

    print()

    print(
        "All requested hourly snapshots "
        "are complete."
    )

    print("=" * 78)


if __name__ == "__main__":

    main()
