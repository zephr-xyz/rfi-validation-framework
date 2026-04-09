#!/usr/bin/env python3
"""
Iran Jammer Scan Visualization
================================
Creates a map showing all detected GPS jammers across Iran from the
CYGNSS blind search scan.

Usage:
    python3 visualize_iran_scan.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

RESULTS_JSON = Path("output/iran_scan/iran_jammers_2026-03-15.json")
OUTPUT_DIR = Path("output/iran_scan")

BG = "#0a0a0a"
RED = "#ff2a2a"
GREEN = "#00ff88"
WHITE = "#f0f0f0"
WHITE_DIM = "#ffffff60"
AMBER = "#ffaa00"

# Known Shiraz jammer for reference
GT_LAT, GT_LON = 27.3182, 52.8703

# Iran approximate bounds for the map
MAP_LAT_MIN, MAP_LAT_MAX = 24.0, 40.5
MAP_LON_MIN, MAP_LON_MAX = 43.0, 64.0


def load_data():
    with open(RESULTS_JSON) as f:
        return json.load(f)


def add_basemap(ax, extent):
    try:
        import contextily as ctx
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        xmin, ymin = t.transform(extent[0], extent[2])
        xmax, ymax = t.transform(extent[1], extent[3])
        img, ext_m = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=6,
                                     source=ctx.providers.Esri.WorldImagery)
        inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        e0, e2 = inv.transform(ext_m[0], ext_m[2])
        e1, e3 = inv.transform(ext_m[1], ext_m[3])
        img_dark = (img.astype(float) * 0.55).astype(np.uint8)
        ax.imshow(img_dark, extent=[e0, e1, e2, e3],
                  aspect="auto", zorder=0, interpolation="bilinear")
        return True
    except Exception as e:
        print(f"Basemap failed: {e}")
        return False


def main():
    data = load_data()
    jammers = data["jammers"]

    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.edgecolor": "none",
        "font.family": "Helvetica, Arial, sans-serif",
    })

    fig, ax = plt.subplots(figsize=(18, 14))

    extent = [MAP_LON_MIN, MAP_LON_MAX, MAP_LAT_MIN, MAP_LAT_MAX]
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1.0 / np.cos(np.radians(32.0)))  # mid-Iran latitude

    # Basemap
    add_basemap(ax, extent)

    text_fx = [pe.withStroke(linewidth=3, foreground=BG)]

    # ── Plot jammers ─────────────────────────────────────────────────────
    lats = np.array([j["estimated_lat"] for j in jammers])
    lons = np.array([j["estimated_lon"] for j in jammers])
    amps = np.array([j["amplitude"] for j in jammers])
    ceps = np.array([j["bootstrap_cep_km"] if j.get("bootstrap_cep_km") else 20.0
                     for j in jammers])

    # Size by amplitude, color by z-score
    zscores = np.array([j["cluster_mean_zscore"] for j in jammers])

    # Normalize sizes: amplitude range maps to marker size 30-300
    amp_norm = (amps - amps.min()) / (amps.max() - amps.min() + 1)
    sizes = 40 + amp_norm * 260

    # Color by z-score
    norm = Normalize(vmin=2.5, vmax=6.0)
    cmap = plt.cm.YlOrRd

    scatter = ax.scatter(lons, lats, s=sizes, c=zscores, cmap=cmap, norm=norm,
                         alpha=0.8, edgecolors=WHITE, linewidths=0.5,
                         zorder=10)

    # ── CEP circles for top jammers with good CEP ────────────────────────
    for j in jammers[:20]:
        cep = j.get("bootstrap_cep_km")
        if cep and cep < 15:
            cep_deg = cep / 111.0
            ax.add_patch(Circle((j["estimated_lon"], j["estimated_lat"]),
                                cep_deg, fill=False, edgecolor=WHITE_DIM,
                                linewidth=0.6, linestyle="--", zorder=9))

    # ── Label top 10 jammers ─────────────────────────────────────────────
    labeled_positions = []
    for i, j in enumerate(jammers[:15]):
        lat, lon = j["estimated_lat"], j["estimated_lon"]
        amp = j["amplitude"]
        cep = j.get("bootstrap_cep_km")

        # Avoid overlapping labels
        too_close = False
        for plat, plon in labeled_positions:
            if abs(lat - plat) < 0.8 and abs(lon - plon) < 1.0:
                too_close = True
                break
        if too_close:
            continue

        cep_str = f"  CEP {cep:.1f} km" if cep else ""
        label = f"#{i+1}{cep_str}"

        # Offset direction based on position
        dx = 0.3 if lon < 55 else -0.3
        dy = 0.25

        ax.annotate(label, (lon, lat), (lon + dx, lat + dy),
                    fontsize=7, color=WHITE, fontweight="bold",
                    path_effects=text_fx, zorder=25,
                    arrowprops=dict(arrowstyle="-", color=WHITE_DIM,
                                    lw=0.5))
        labeled_positions.append((lat, lon))

    # ── Known Shiraz jammer — green marker ───────────────────────────────
    ax.plot(GT_LON, GT_LAT, marker="*", color=GREEN, markersize=14,
            markeredgecolor=GREEN, markeredgewidth=1, zorder=15)
    ax.text(GT_LON + 0.4, GT_LAT - 0.3,
            "Validated Jammer\n(4.33 km accuracy)",
            fontsize=8, color=GREEN, fontweight="bold",
            path_effects=text_fx, zorder=25)

    # ── Major cities for reference ───────────────────────────────────────
    cities = [
        ("Tehran", 35.69, 51.39),
        ("Isfahan", 32.65, 51.68),
        ("Shiraz", 29.59, 52.58),
        ("Tabriz", 38.08, 46.29),
        ("Mashhad", 36.30, 59.60),
        ("Bandar Abbas", 27.19, 56.27),
        ("Bushehr", 28.97, 50.84),
        ("Kerman", 30.28, 57.08),
        ("Zahedan", 29.50, 60.86),
    ]
    for name, clat, clon in cities:
        if MAP_LAT_MIN < clat < MAP_LAT_MAX and MAP_LON_MIN < clon < MAP_LON_MAX:
            ax.plot(clon, clat, marker="s", color=WHITE_DIM, markersize=4, zorder=8)
            ax.text(clon + 0.2, clat + 0.15, name, fontsize=7,
                    color=WHITE_DIM, path_effects=text_fx, zorder=8)

    # ── Colorbar ─────────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.4])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Noise Elevation Z-Score", color=WHITE, fontsize=9)
    cbar.ax.tick_params(colors=WHITE, labelsize=7)

    # ── Title ────────────────────────────────────────────────────────────
    ax.text(0.5, 0.97, "109 GPS Jammers Detected Across Iran",
            transform=ax.transAxes, fontsize=20, color=WHITE,
            fontweight="bold", ha="center", va="top",
            path_effects=text_fx, zorder=30)

    ax.text(0.5, 0.935,
            "CYGNSS GNSS-R Blind Search  |  March 15, 2026 vs December 27, 2025 Baseline",
            transform=ax.transAxes, fontsize=10, color=WHITE_DIM,
            ha="center", va="top", path_effects=text_fx, zorder=30)

    # ── Stats box — lower right ──────────────────────────────────────────
    stats_text = (
        f"Conflict measurements: {data['n_conflict_measurements']:,}\n"
        f"Baseline measurements: {data['n_baseline_measurements']:,}\n"
        f"Clusters found: {data['n_clusters']}\n"
        f"Jammers localized: {data['n_jammers']}\n"
        f"Circle size = signal amplitude\n"
        f"Color = noise elevation z-score"
    )
    ax.text(0.98, 0.03, stats_text, transform=ax.transAxes,
            fontsize=8, color=WHITE, va="bottom", ha="right",
            path_effects=text_fx, zorder=25,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#00000080",
                      edgecolor=WHITE_DIM, linewidth=0.5))

    # ── Clean axes ───────────────────────────────────────────────────────
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Save ─────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "iran_jammers_map.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    main()
