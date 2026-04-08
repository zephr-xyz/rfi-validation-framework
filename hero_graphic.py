#!/usr/bin/env python3
"""
Hero Graphic — Clean Targeting Overlay
=======================================
Minimal satellite-imagery visualization: CYGNSS heatmap + targeting
reticle over the jammer. Zoomed in, clean typography, few colors.

Usage:
    python3 hero_graphic.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.ndimage import gaussian_filter

RESULTS_JSON = Path("output/comparison_results.json")
OUTPUT_DIR = Path("output/article_figures")

GT_LAT, GT_LON = 27.3182, 52.8703
BG = "#0a0a0a"
RED = "#ff2a2a"
RED_DIM = "#ff2a2a"
WHITE = "#f0f0f0"
WHITE_DIM = "#ffffff60"


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
        img, ext_m = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=10,
                                     source=ctx.providers.Esri.WorldImagery)
        inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        e0, e2 = inv.transform(ext_m[0], ext_m[2])
        e1, e3 = inv.transform(ext_m[1], ext_m[3])
        img_dark = (img.astype(float) * 0.4).astype(np.uint8)
        ax.imshow(img_dark, extent=[e0, e1, e2, e3],
                  aspect="auto", zorder=0, interpolation="bilinear")
        return True
    except Exception as e:
        print(f"Basemap failed: {e}")
        return False


def draw_reticle(ax, lon, lat, r):
    """Clean targeting reticle — two circles + crosshair."""
    for radius, lw, alpha in [(r, 1.8, 0.9), (r * 0.5, 1.2, 0.6)]:
        ax.add_patch(Circle((lon, lat), radius, fill=False,
                            edgecolor=RED, linewidth=lw, alpha=alpha, zorder=20))
    # Crosshair — four lines with center gap
    gap = r * 0.2
    arm = r * 1.4
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        ax.plot([lon + dx * gap, lon + dx * arm],
                [lat + dy * gap, lat + dy * arm],
                color=RED, linewidth=1.2, alpha=0.7, zorder=20)


def main():
    data = load_data()

    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.edgecolor": "none",
        "font.family": "Helvetica, Arial, sans-serif",
    })

    fig, ax = plt.subplots(figsize=(14, 11))

    # Tighter zoom — ~120 km across, centered on jammer
    pad_lon = 0.7
    pad_lat = 0.55
    extent = [GT_LON - pad_lon, GT_LON + pad_lon,
              GT_LAT - pad_lat, GT_LAT + pad_lat]
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1.0 / np.cos(np.radians(GT_LAT)))

    # Basemap
    add_basemap(ax, extent)

    # ── CYGNSS heatmap ───────────────────────────────────────────────────
    cygnss_dets = data["cygnss"]["detections"]
    c_lats = np.array([d["lat"] for d in cygnss_dets])
    c_lons = np.array([d["lon"] for d in cygnss_dets])
    c_ints = np.array([d["intensity"] for d in cygnss_dets])

    # Filter to view extent with margin
    in_view = ((c_lons > extent[0] - 0.3) & (c_lons < extent[1] + 0.3) &
               (c_lats > extent[2] - 0.3) & (c_lats < extent[3] + 0.3))
    c_lats, c_lons, c_ints = c_lats[in_view], c_lons[in_view], c_ints[in_view]

    # Intensity-weighted heatmap
    nx, ny = 250, 200
    x_edges = np.linspace(extent[0] - 0.3, extent[1] + 0.3, nx + 1)
    y_edges = np.linspace(extent[2] - 0.3, extent[3] + 0.3, ny + 1)
    H, _, _ = np.histogram2d(c_lons, c_lats, bins=[x_edges, y_edges],
                              weights=np.log1p(c_ints))
    H = H.T
    H_smooth = gaussian_filter(H, sigma=6)

    # Single-color heatmap: transparent → red glow
    heatmap_cmap = LinearSegmentedColormap.from_list("heat", [
        (0.0, (1, 0.1, 0.1, 0)),
        (0.5, (1, 0.1, 0.1, 0)),
        (0.65, (1, 0.15, 0.1, 0.08)),
        (0.75, (1, 0.15, 0.05, 0.2)),
        (0.85, (1, 0.2, 0.1, 0.4)),
        (0.95, (1, 0.5, 0.3, 0.6)),
        (1.0, (1, 0.85, 0.7, 0.75)),
    ])

    h_extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    norm = Normalize(vmin=0, vmax=np.percentile(H_smooth[H_smooth > 0], 95)
                     if (H_smooth > 0).any() else 1)
    ax.imshow(H_smooth, extent=h_extent, origin="lower", cmap=heatmap_cmap,
              norm=norm, aspect="auto", zorder=2, interpolation="bilinear")

    # Scatter points — subtle
    ax.scatter(c_lons, c_lats, s=3, c=RED, alpha=0.2,
              edgecolors="none", rasterized=True, zorder=3)

    # ── Targeting reticle on ground truth ─────────────────────────────────
    draw_reticle(ax, GT_LON, GT_LAT, 0.06)

    # Ground truth center dot
    ax.plot(GT_LON, GT_LAT, marker="+", color=RED, markersize=14,
            markeredgewidth=2, zorder=22)

    # ── CYGNSS estimate ──────────────────────────────────────────────────
    est_lat = data["cygnss"]["estimated_lat"]
    est_lon = data["cygnss"]["estimated_lon"]
    err = data["cygnss"]["euclidean_error_km"]
    cep = data["cygnss"]["cep_km"]

    ax.plot(est_lon, est_lat, marker="^", color=WHITE, markersize=12,
            markeredgecolor=WHITE, markeredgewidth=1.5, zorder=12,
            fillstyle="none")

    # Distance line
    ax.plot([GT_LON, est_lon], [GT_LAT, est_lat],
            color=WHITE_DIM, linewidth=1, linestyle="-", zorder=11)

    # CEP circle
    cep_deg = cep / 111.0
    ax.add_patch(Circle((est_lon, est_lat), cep_deg, fill=False,
                        edgecolor=WHITE_DIM, linewidth=1,
                        linestyle="--", zorder=9))

    # ── Labels — clean, no monospace ─────────────────────────────────────
    text_fx = [pe.withStroke(linewidth=3, foreground=BG)]

    # Ground truth label
    ax.text(GT_LON + 0.08, GT_LAT - 0.04,
            f"Ground Truth\n{GT_LAT:.4f}°N, {GT_LON:.4f}°E",
            fontsize=10, color=RED, fontweight="bold",
            path_effects=text_fx, zorder=25)

    # CYGNSS estimate label
    ax.text(est_lon + 0.06, est_lat + 0.03,
            f"CYGNSS Estimate\n{err:.2f} km error  |  {cep:.1f} km CEP",
            fontsize=10, color=WHITE, fontweight="bold",
            path_effects=text_fx, zorder=25)

    # ── Range ring ───────────────────────────────────────────────────────
    for r_km in [25, 50]:
        r_deg = r_km / 111.0
        ax.add_patch(Circle((GT_LON, GT_LAT), r_deg, fill=False,
                            edgecolor=WHITE, linewidth=0.4,
                            linestyle=":", alpha=0.2, zorder=5))
        ax.text(GT_LON, GT_LAT + r_deg + 0.01,
                f"{r_km} km", fontsize=7, color=WHITE, alpha=0.3,
                ha="center", path_effects=text_fx, zorder=6)

    # ── Title — top center ───────────────────────────────────────────────
    ax.text(0.5, 0.97, "GPS Jammer Localized from Space",
            transform=ax.transAxes, fontsize=18, color=WHITE,
            fontweight="bold", ha="center", va="top",
            path_effects=text_fx, zorder=30)

    ax.text(0.5, 0.93, "CYGNSS GNSS-R  |  4.33 km accuracy  |  Shiraz, Iran  |  January 2026",
            transform=ax.transAxes, fontsize=10, color=WHITE_DIM,
            ha="center", va="top", path_effects=text_fx, zorder=30)

    # ── Clean axes ───────────────────────────────────────────────────────
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Save ─────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "hero_targeting_overlay.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    main()
