#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib import patheffects as pe

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


# ============================================================
# CONFIGURACION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "figs"

PC = ccrs.PlateCarree()

# Posiciones de referencia utilizadas en el experimento

NHC = [
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


# ============================================================
# CARGA DE DATOS
# ============================================================

def read_data():

    centers_file = RAW / "hourly_eye_centers.csv"

    if not centers_file.exists():
        raise FileNotFoundError(
            f"No existe: {centers_file}"
        )

    centers = pd.read_csv(centers_file)

    choices = list(
        RAW.glob("*date20251028_hour12.dat")
    )

    if len(choices) != 1:
        raise FileNotFoundError(
            "No se encontró el archivo .dat "
            "del 28 de octubre a las 12 UTC."
        )

    cols = [
        "lat", "lon", "z", "u", "v",
        "w", "speed", "direction"
    ]

    winds = pd.read_csv(
        choices[0],
        sep=r"\s+",
        comment="#",
        header=None,
        names=cols,
    )

    winds = winds.loc[
        np.isclose(winds.z, 10)
    ].copy()

    if len(centers) != 61 or len(winds) != 100:
        raise ValueError(
            f"Esperados 61 centros y 100 vectores: "
            f"{len(centers)} y {len(winds)}"
        )

    for key, lat, lon in NHC:

        row = centers.loc[
            centers.key.eq(key)
        ]

        if (
            len(row) != 1
            or not np.isclose(row.iloc[0].eye_lat_deg, lat)
            or not np.isclose(row.iloc[0].eye_lon_deg, lon)
        ):
            raise ValueError(
                f"El ancla {key} no coincide con el CSV"
            )

    return centers, winds


# ============================================================
# MAPA DE ALTA RESOLUCION
# ============================================================

def base_map(ax, bounds):

    # Fondo del oceano
    ax.set_facecolor("#e8f3f9")

    # Tierra: Natural Earth 1:10m
    land = cfeature.NaturalEarthFeature(
        "physical",
        "land",
        "10m",
        facecolor="#eff0e9",
        edgecolor="none",
    )

    ax.add_feature(
        land,
        zorder=1,
    )

    # Costas de alta resolucion
    ax.coastlines(
        resolution="10m",
        color="#556a73",
        linewidth=0.72,
        zorder=3,
    )

    # Region geografica
    ax.set_extent(
        bounds,
        crs=PC,
    )

    # Cuadricula
    gl = ax.gridlines(
        crs=PC,
        draw_labels=False,
        linewidth=0.55,
        color="#92b5c7",
        alpha=0.5,
        linestyle="--",
        zorder=2,
    )

    gl.xlocator = plt.MultipleLocator(1)
    gl.ylocator = plt.MultipleLocator(1)

    xticks = np.arange(
        np.ceil(bounds[0]),
        np.floor(bounds[1]) + 1,
        1,
    )

    yticks = np.arange(
        np.ceil(bounds[2]),
        np.floor(bounds[3]) + 1,
        1,
    )

    ax.set_xticks(
        xticks,
        crs=PC,
    )

    ax.set_yticks(
        yticks,
        crs=PC,
    )

    ax.xaxis.set_major_formatter(
        LongitudeFormatter()
    )

    ax.yaxis.set_major_formatter(
        LatitudeFormatter()
    )

    ax.tick_params(
        labelsize=8,
        length=2,
        color="#506578",
    )

    ax.spines["geo"].set_edgecolor("#627888")
    ax.spines["geo"].set_linewidth(0.8)


# ============================================================
# FIGURA
# ============================================================

def main():

    centers, winds = read_data()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    })

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.8, 5.8),
        subplot_kw={
            "projection": PC
        },
        constrained_layout=True,
    )

    track_color = "#c02c39"

    # ========================================================
    # PANEL (a): TRAYECTORIA
    # ========================================================

    ax = axes[0]

    # Encuadre cercano a Melissa
    base_map(
        ax,
        (-83.0, -72.0, 12.0, 23.5)
    )

    # Trayectoria de 61 horas
    ax.plot(
        centers.eye_lon_deg,
        centers.eye_lat_deg,
        color=track_color,
        linewidth=2.0,
        transform=PC,
        zorder=5,
        path_effects=[
            pe.Stroke(
                linewidth=3.1,
                foreground="white"
            ),
            pe.Normal(),
        ],
        label="Hourly PCHIP track (61 positions)",
    )

    # Las 12 posiciones NHC
    ax.scatter(
        [item[2] for item in NHC],
        [item[1] for item in NHC],
        s=35,
        facecolor=track_color,
        edgecolor="white",
        linewidth=0.75,
        transform=PC,
        zorder=7,
        label="NHC reference positions (12)",
    )

    ax.set_title(
        "(a) NHC references and hourly storm track",
        loc="left",
        pad=9,
    )

    ax.legend(
        loc="upper left",
        fontsize=7.3,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#becbd0",
    )

    # ========================================================
    # PANEL (b): VIENTO HORIZONTAL
    # ========================================================

    ax = axes[1]

    current = centers.loc[
        centers.key.eq("20251028T12")
    ].iloc[0]

    lon0 = float(current.eye_lon_deg)
    lat0 = float(current.eye_lat_deg)

    # Encuadre alrededor del ojo
    base_map(
        ax,
        (
            lon0 - 2.5,
            lon0 + 2.5,
            lat0 - 2.5,
            lat0 + 2.5,
        ),
    )

    # Trayectoria
    ax.plot(
        centers.eye_lon_deg,
        centers.eye_lat_deg,
        linewidth=1.4,
        color=track_color,
        transform=PC,
        zorder=4,
        label="Hourly center track",
    )

    # Magnitud del viento
    mag = np.hypot(
        winds.u.to_numpy(),
        winds.v.to_numpy(),
    )

    # Flechas de viento
    q = ax.quiver(
        winds.lon.to_numpy(),
        winds.lat.to_numpy(),
        winds.u.to_numpy(),
        winds.v.to_numpy(),
        mag,
        transform=PC,
        angles="uv",
        scale_units="width",
        scale=290,
        width=0.0037,
        headwidth=3.6,
        headlength=4.5,
        pivot="mid",
        cmap="turbo",
        norm=Normalize(
            vmin=10,
            vmax=32,
        ),
        zorder=6,
    )

    # Ventana de adquisicion
    ax.add_patch(
        Rectangle(
            (lon0 - 2, lat0 - 2),
            4,
            4,
            transform=PC,
            fill=False,
            linestyle="--",
            linewidth=1.2,
            edgecolor="#485e7a",
            zorder=5,
            label=r"Acquisition window ($\pm 2^\circ$)",
        )
    )

    # Centro de la tormenta
    ax.scatter(
        lon0,
        lat0,
        s=140,
        marker="*",
        color="#f04a4c",
        edgecolor="white",
        linewidth=0.9,
        transform=PC,
        zorder=8,
        label="Storm center",
    )

    ax.set_title(
        "(b) NASA POWER horizontal winds at 10 m\n"
        "28 October 2025, 12:00 UTC",
        loc="left",
        pad=9,
    )

    ax.legend(
        loc="upper left",
        fontsize=7.2,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#becbd0",
    )

    # Barra de colores
    cb = fig.colorbar(
        q,
        ax=ax,
        shrink=0.78,
        fraction=0.046,
        pad=0.028,
    )

    cb.set_label(
        r"Horizontal wind speed $\sqrt{u^2+v^2}$ (m/s)"
    )

    cb.ax.tick_params(
        labelsize=8,
    )

    # ========================================================
    # GUARDAR PDF VECTORIAL Y PNG DE 600 DPI
    # ========================================================

    for ext in ("pdf", "png"):

        output = (
            OUT / f"melissa_campos_12_nhc.{ext}"
        )

        fig.savefig(
            output,
            bbox_inches="tight",
            facecolor="white",
            dpi=600,
        )

        print("Generado:", output)

    plt.close(fig)


if __name__ == "__main__":
    main()
