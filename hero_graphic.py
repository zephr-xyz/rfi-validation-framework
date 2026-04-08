#!/usr/bin/env python3
"""
Hero Graphic — "Targeting Overlay" Style
=========================================
Cinematic satellite-imagery visualization with CYGNSS heatmap,
NISAR streak detections, and targeting reticle over the jammer location.
Styled like a military targeting HUD / Jason Bourne movie overlay.

Usage:
    python3 hero_graphic.py
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, FancyArrowPatch, Arc
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter

# ── Config ───────────────────────────────────────────────────────────────
RESULTS_JSON = Path("output/comparison_results.json")
OUTPUT_DIR = Path("output/article_figures")

GT_LAT, GT_LON = 27.3182, 52.8703

# HUD palette
BG = "#0a0a0a"
HUD_GREEN = "#00ff88"
HUD_GREEN_DIM = "#00ff8840"
HUD_RED = "#ff2020"
HUD_AMBER = "#ffaa00"
HUD_CYAN = "#00ddff"
HUD_WHITE = "#e0e0e0"
RETICLE_COLOR = "#ff3030"
CROSSHAIR = "#ff303080"

MONO_FONT = {"fontfamily": "monospace"}


def load_data():
    with open(RESULTS_JSON) as f:
        return json.load(f)


def add_basemap(ax, extent):
    """Try to add a dark satellite basemap via contextily."""
    try:
        import contextily as ctx
        # Convert extent from lon/lat to Web Mercator for contextily
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        xmin, ymin = transformer.transform(extent[0], extent[2])
        xmax, ymax = transformer.transform(extent[1], extent[3])

        # Fetch tile
        img, ext_merc = ctx.bounds2img(xmin, ymin, xmax, ymax,
                                        zoom=9,
                                        source=ctx.providers.Esri.WorldImagery)
        # Transform back to lat/lon extent
        inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        e_lon_min, e_lat_min = inv.transform(ext_merc[0], ext_merc[2])
        e_lon_max, e_lat_max = inv.transform(ext_merc[1], ext_merc[3])

        # Darken the image — keep enough detail to see terrain
        img_dark = (img.astype(float) * 0.5).astype(np.uint8)

        ax.imshow(img_dark, extent=[e_lon_min, e_lon_max, e_lat_min, e_lat_max],
                  aspect="auto", zorder=0, interpolation="bilinear")
        return True
    except Exception as e:
        print(f"Basemap failed: {e}, using plain dark background")
        return False


def draw_reticle(ax, lon, lat, radius_deg, color=RETICLE_COLOR, label=None):
    """Draw a targeting reticle (concentric circles + crosshairs)."""
    # Concentric circles
    for r, alpha, lw in [(radius_deg, 0.9, 2.0),
                          (radius_deg * 0.6, 0.6, 1.5),
                          (radius_deg * 0.3, 0.4, 1.0)]:
        circle = Circle((lon, lat), r, fill=False, edgecolor=color,
                        linewidth=lw, alpha=alpha, zorder=20,
                        linestyle="-")
        ax.add_patch(circle)

    # Crosshair lines (with gaps in center)
    gap = radius_deg * 0.15
    ext = radius_deg * 1.3
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        ax.plot([lon + dx * gap, lon + dx * ext],
                [lat + dy * gap, lat + dy * ext],
                color=color, linewidth=1.5, alpha=0.8, zorder=20)

    # Corner tick marks on outer circle
    for angle in [45, 135, 225, 315]:
        rad = np.radians(angle)
        r1 = radius_deg * 0.9
        r2 = radius_deg * 1.1
        ax.plot([lon + r1 * np.cos(rad), lon + r2 * np.cos(rad)],
                [lat + r1 * np.sin(rad), lat + r2 * np.sin(rad)],
                color=color, linewidth=2, alpha=0.7, zorder=20)

    if label:
        text_fx = [pe.withStroke(linewidth=3, foreground=BG)]
        ax.text(lon + radius_deg * 1.2, lat - radius_deg * 0.3, label,
                fontsize=9, color=color, fontweight="bold",
                path_effects=text_fx, zorder=21, **MONO_FONT)


def draw_scanlines(ax, extent, n_lines=80, alpha=0.03):
    """Draw horizontal scan line effect."""
    y_vals = np.linspace(extent[2], extent[3], n_lines)
    for y in y_vals:
        ax.axhline(y=y, color=HUD_GREEN, linewidth=0.3, alpha=alpha, zorder=15)


def draw_hud_border(ax, extent):
    """Draw HUD-style border with corner brackets."""
    x0, x1, y0, y1 = extent
    pad = (x1 - x0) * 0.02
    corner_len = (x1 - x0) * 0.06

    for cx, cy, dx, dy in [(x0 + pad, y0 + pad, 1, 1),
                            (x1 - pad, y0 + pad, -1, 1),
                            (x0 + pad, y1 - pad, 1, -1),
                            (x1 - pad, y1 - pad, -1, -1)]:
        ax.plot([cx, cx + dx * corner_len], [cy, cy],
                color=HUD_GREEN, linewidth=1.5, alpha=0.6, zorder=25)
        ax.plot([cx, cx], [cy, cy + dy * corner_len],
                color=HUD_GREEN, linewidth=1.5, alpha=0.6, zorder=25)


def draw_hud_text(fig, ax, data):
    """Draw HUD overlay text panels."""
    text_fx = [pe.withStroke(linewidth=2, foreground=BG)]
    kw = {**MONO_FONT, "path_effects": text_fx, "zorder": 30}

    # Top-left: mission info
    lines_tl = [
        ("SYS: CYGNSS/NISAR FUSION", HUD_GREEN, 11),
        ("MODE: GPS JAMMER LOCALIZATION", HUD_GREEN, 9),
        ("AOR: STRAIT OF HORMUZ", HUD_AMBER, 9),
        (f"TGT: {GT_LAT:.4f}°N  {GT_LON:.4f}°E", HUD_WHITE, 9),
    ]
    for i, (text, color, size) in enumerate(lines_tl):
        ax.text(0.02, 0.97 - i * 0.04, text, transform=ax.transAxes,
                fontsize=size, color=color, fontweight="bold", va="top", **kw)

    # Top-right: status
    lines_tr = [
        (datetime.now().strftime("UTC %Y-%m-%d %H:%M"), HUD_GREEN_DIM, 8),
        ("STATUS: TARGET ACQUIRED", HUD_RED, 10),
        (f"CYGNSS: {data['cygnss']['euclidean_error_km']:.2f} km  CEP {data['cygnss']['cep_km']:.1f} km", HUD_AMBER, 9),
        (f"NISAR:  {data['nisar']['euclidean_error_km']:.2f} km  CEP {data['nisar']['cep_km']:.1f} km", HUD_CYAN, 9),
        (f"FUSED:  {data['fused']['euclidean_error_km']:.2f} km", HUD_GREEN, 9),
    ]
    for i, (text, color, size) in enumerate(lines_tr):
        ax.text(0.98, 0.97 - i * 0.04, text, transform=ax.transAxes,
                fontsize=size, color=color, fontweight="bold", va="top",
                ha="right", **kw)

    # Bottom-left: detection stats
    lines_bl = [
        (f"DETECTIONS: {data['cygnss']['num_detections']} CYGNSS + {data['nisar']['num_detections']} NISAR", HUD_WHITE, 9),
        ("BASELINE: 0 FALSE POSITIVES", HUD_GREEN, 8),
        ("CONFLICT STATUS: JAMMER ACTIVE +79% POWER", HUD_RED, 8),
    ]
    for i, (text, color, size) in enumerate(lines_bl):
        ax.text(0.02, 0.08 - i * 0.035, text, transform=ax.transAxes,
                fontsize=size, color=color, fontweight="bold", va="top", **kw)

    # Bottom-right: classification
    ax.text(0.98, 0.05, "UNCLASSIFIED // OPEN SOURCE",
            transform=ax.transAxes, fontsize=8, color=HUD_GREEN_DIM,
            fontweight="bold", ha="right", va="bottom", **kw)


def main():
    data = load_data()

    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.edgecolor": "none",
    })

    fig, ax = plt.subplots(figsize=(16, 12))

    # Map extent — zoom to show CYGNSS spread + detail around jammer
    extent = [51.5, 54.5, 26.0, 28.8]  # [lon_min, lon_max, lat_min, lat_max]
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    aspect = 1.0 / np.cos(np.radians(GT_LAT))
    ax.set_aspect(aspect)

    # ── Basemap ──────────────────────────────────────────────────────────
    has_basemap = add_basemap(ax, extent)

    # ── CYGNSS heatmap ───────────────────────────────────────────────────
    cygnss_dets = data["cygnss"]["detections"]
    c_lats = np.array([d["lat"] for d in cygnss_dets])
    c_lons = np.array([d["lon"] for d in cygnss_dets])
    c_ints = np.array([d["intensity"] for d in cygnss_dets])

    # Create 2D histogram / heatmap
    # Grid resolution
    nx, ny = 300, 220
    x_edges = np.linspace(extent[0], extent[1], nx + 1)
    y_edges = np.linspace(extent[2], extent[3], ny + 1)

    # Intensity-weighted 2D histogram
    H, _, _ = np.histogram2d(c_lons, c_lats, bins=[x_edges, y_edges],
                              weights=np.log1p(c_ints))
    H = H.T  # transpose for imshow

    # Smooth to create a diffuse radial glow centered on jammer
    H_smooth = gaussian_filter(H, sigma=8)

    # Custom heatmap colormap: wide transparent zone so only
    # the hot core near the jammer glows through
    heatmap_cmap = LinearSegmentedColormap.from_list("jammer_heat", [
        (0.0, (0, 0, 0, 0)),
        (0.5, (0, 0, 0, 0)),
        (0.6, (1.0, 0.5, 0, 0.05)),
        (0.7, (1.0, 0.3, 0, 0.15)),
        (0.8, (1.0, 0.1, 0, 0.35)),
        (0.9, (1.0, 0.15, 0.05, 0.55)),
        (1.0, (1.0, 0.7, 0.4, 0.8)),
    ])

    norm = Normalize(vmin=0, vmax=np.percentile(H_smooth[H_smooth > 0], 95)
                     if (H_smooth > 0).any() else 1)
    ax.imshow(H_smooth, extent=[extent[0], extent[1], extent[2], extent[3]],
              origin="lower", cmap=heatmap_cmap, norm=norm, aspect="auto",
              zorder=2, interpolation="bilinear")

    # Individual CYGNSS points — subtle but visible track structure
    ax.scatter(c_lons, c_lats, s=4, c=HUD_AMBER, alpha=0.25,
              edgecolors="none", rasterized=True, zorder=3)

    # ── NISAR detections ─────────────────────────────────────────────────
    nisar_dets = data["nisar"]["detections"]
    n_lats = np.array([d["lat"] for d in nisar_dets])
    n_lons = np.array([d["lon"] for d in nisar_dets])

    # Bearing lines from NISAR clusters
    jan8_mask = np.array(["20260108" in d["timestamp"] for d in nisar_dets])
    jan20_mask = np.array(["20260120" in d["timestamp"] for d in nisar_dets])
    nisar_est = (data["nisar"]["estimated_lat"], data["nisar"]["estimated_lon"])

    for mask, alpha in [(jan8_mask, 0.5), (jan20_mask, 0.5)]:
        if mask.any():
            clat, clon = np.mean(n_lats[mask]), np.mean(n_lons[mask])
            dlat = nisar_est[0] - clat
            dlon = nisar_est[1] - clon
            length = np.sqrt(dlat**2 + dlon**2)
            if length > 0:
                ext_far = 0.3
                ext_near = 0.15
                ax.plot([clon - dlon/length*ext_near, clon + dlon/length*(ext_far+length)],
                       [clat - dlat/length*ext_near, clat + dlat/length*(ext_far+length)],
                       color=HUD_CYAN, linewidth=1.5, linestyle="--",
                       alpha=alpha, zorder=8)

    # NISAR detection points
    ax.scatter(n_lons, n_lats, s=50, c=HUD_CYAN, alpha=0.9,
              edgecolors="white", linewidths=0.8, marker="D", zorder=10)

    # ── Targeting reticle on ground truth ─────────────────────────────────
    reticle_r = 0.08
    draw_reticle(ax, GT_LON, GT_LAT, reticle_r, color=RETICLE_COLOR)

    # Ground truth marker
    ax.plot(GT_LON, GT_LAT, marker="+", color=RETICLE_COLOR, markersize=16,
            markeredgewidth=2.5, zorder=22)

    # ── Estimated positions ──────────────────────────────────────────────
    text_fx = [pe.withStroke(linewidth=3, foreground=BG)]
    cygnss_est = (data["cygnss"]["estimated_lat"], data["cygnss"]["estimated_lon"])
    fused_est = (data["fused"]["estimated_lat"], data["fused"]["estimated_lon"])

    # CYGNSS estimate
    ax.plot(cygnss_est[1], cygnss_est[0], marker="^", color=HUD_AMBER,
            markersize=14, markeredgecolor="white", markeredgewidth=1, zorder=12)
    ax.annotate(f"CYGNSS EST\n{cygnss_est[0]:.4f}°N {cygnss_est[1]:.4f}°E\n"
                f"ERR: {data['cygnss']['euclidean_error_km']:.2f} km",
                (cygnss_est[1], cygnss_est[0]),
                textcoords="offset points", xytext=(18, 12),
                fontsize=8, color=HUD_AMBER, fontweight="bold",
                path_effects=text_fx, zorder=22, **MONO_FONT)

    # NISAR estimate
    ax.plot(nisar_est[1], nisar_est[0], marker="v", color=HUD_CYAN,
            markersize=14, markeredgecolor="white", markeredgewidth=1, zorder=12)
    ax.annotate(f"NISAR EST\n{nisar_est[0]:.4f}°N {nisar_est[1]:.4f}°E\n"
                f"ERR: {data['nisar']['euclidean_error_km']:.2f} km",
                (nisar_est[1], nisar_est[0]),
                textcoords="offset points", xytext=(18, -30),
                fontsize=8, color=HUD_CYAN, fontweight="bold",
                path_effects=text_fx, zorder=22, **MONO_FONT)

    # Distance line from GT to CYGNSS
    ax.plot([GT_LON, cygnss_est[1]], [GT_LAT, cygnss_est[0]],
            color=HUD_AMBER, linewidth=1, linestyle=":", alpha=0.6, zorder=11)
    # Distance line from GT to NISAR
    ax.plot([GT_LON, nisar_est[1]], [GT_LAT, nisar_est[0]],
            color=HUD_CYAN, linewidth=1, linestyle=":", alpha=0.6, zorder=11)

    # ── CEP circles ──────────────────────────────────────────────────────
    cygnss_cep_deg = data["cygnss"]["cep_km"] / 111.0
    nisar_cep_deg = data["nisar"]["cep_km"] / 111.0

    cep_c = Circle((cygnss_est[1], cygnss_est[0]), cygnss_cep_deg,
                    fill=False, edgecolor=HUD_AMBER, linewidth=1,
                    linestyle="--", alpha=0.5, zorder=9)
    ax.add_patch(cep_c)

    cep_n = Circle((nisar_est[1], nisar_est[0]), nisar_cep_deg,
                    fill=False, edgecolor=HUD_CYAN, linewidth=1.5,
                    linestyle="-", alpha=0.6, zorder=9)
    ax.add_patch(cep_n)

    # ── Range rings from ground truth ────────────────────────────────────
    for r_km in [25, 50, 100]:
        r_deg = r_km / 111.0
        ring = Circle((GT_LON, GT_LAT), r_deg, fill=False,
                      edgecolor=HUD_GREEN, linewidth=0.5,
                      linestyle=":", alpha=0.2, zorder=5)
        ax.add_patch(ring)
        # Label
        ax.text(GT_LON + r_deg * 0.71, GT_LAT + r_deg * 0.71,
                f"{r_km}km", fontsize=7, color=HUD_GREEN, alpha=0.35,
                rotation=45, **MONO_FONT, zorder=6)

    # ── Scale bar ────────────────────────────────────────────────────────
    scale_lat = extent[2] + 0.12
    scale_lon = extent[0] + 0.15
    km_per_deg = 111.0 * np.cos(np.radians(GT_LAT))
    scale_50km = 50.0 / km_per_deg
    ax.plot([scale_lon, scale_lon + scale_50km], [scale_lat, scale_lat],
            color=HUD_GREEN, linewidth=2, alpha=0.7, zorder=25)
    ax.text(scale_lon + scale_50km / 2, scale_lat + 0.06, "50 km",
            ha="center", fontsize=8, color=HUD_GREEN, alpha=0.7,
            fontweight="bold", zorder=25, **MONO_FONT)

    # ── Visual effects ───────────────────────────────────────────────────
    draw_scanlines(ax, extent)
    draw_hud_border(ax, extent)

    # Vignette effect (darken edges)
    nx_v, ny_v = 100, 100
    x_v = np.linspace(0, 1, nx_v)
    y_v = np.linspace(0, 1, ny_v)
    X_v, Y_v = np.meshgrid(x_v, y_v)
    vignette = 1.0 - 0.5 * ((X_v - 0.5)**2 + (Y_v - 0.5)**2) / 0.5
    vignette = np.clip(vignette, 0, 1)
    # Apply as overlay
    ax.imshow(1 - vignette, extent=[extent[0], extent[1], extent[2], extent[3]],
              cmap="Greys", alpha=0.25, aspect="auto", zorder=14,
              interpolation="bilinear")

    # ── HUD text overlays ────────────────────────────────────────────────
    draw_hud_text(fig, ax, data)

    # ── Remove axis decorations ──────────────────────────────────────────
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Save ─────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "hero_targeting_overlay.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {out} ({out.stat().st_size / 1e6:.1f} MB)")

    # Also save a version without vignette for print
    return out


if __name__ == "__main__":
    main()
