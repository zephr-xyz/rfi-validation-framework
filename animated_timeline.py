#!/usr/bin/env python3
"""
Animated Jammer Timeline GIF
=============================
Shows CYGNSS jammer detection evolving over time — from baseline silence
through January validation to conflict-period escalation. Each frame is
one date, with the heatmap intensity scaled by noise elevation.

Uses the same satellite basemap as the hero graphic.

Usage:
    python3 animated_timeline.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.ndimage import gaussian_filter
from PIL import Image

RESULTS_JSON = Path("output/comparison_results.json")
OUTPUT_DIR = Path("output/article_figures")
INSET_IMG = Path("assets/inset_jammer_site.png")

GT_LAT, GT_LON = 27.3182, 52.8703
BG = "#0a0a0a"
GREEN = "#00ff88"
RED = "#ff2a2a"
WHITE = "#f0f0f0"
WHITE_DIM = "#ffffff60"
AMBER = "#ffaa00"

# Timeline data from daily checks
TIMELINE = [
    # (date, elevation_pct, detections, near_50km, label)
    ("2025-12-15",  0.0,    0,   0, "BASELINE"),
    ("2025-12-27",  0.0,    0,   0, "BASELINE"),
    ("2026-01-08", 14.5,   89,   2, "JAMMER DETECTED"),
    ("2026-01-09", 16.6,  137,  31, "JAMMER ACTIVE"),
    ("2026-01-20",  9.7,    2,   0, "LOW ACTIVITY"),
    ("2026-01-21", 17.0,  192,  18, "JAMMER ACTIVE"),
    ("2026-02-28", 34.5,  584,  31, "CONFLICT BEGINS"),
    ("2026-03-01", 51.2,  908,  84, "POWER ESCALATION"),
    ("2026-03-03", 62.7, 1085,  18, "ESCALATING"),
    ("2026-03-07", 66.3, 1735, 104, "ESCALATING"),
    ("2026-03-15", 68.9, 2024, 117, "PEAK DETECTIONS"),
    ("2026-03-20", 83.7, 1138,  81, "PEAK POWER"),
    ("2026-03-25", 67.5, 1533, 108, "SUSTAINED"),
    ("2026-03-30", 77.9, 1651, 102, "SUSTAINED"),
    ("2026-04-04", 66.5, 1261, 148, "STILL ACTIVE"),
    ("2026-04-06", 79.1,  725,   0, "STILL ACTIVE"),
]

# Reference: January mean elevation for scaling
JAN_MEAN_ELEV = 14.5  # %


def load_data():
    with open(RESULTS_JSON) as f:
        return json.load(f)


def get_basemap(extent):
    """Fetch and cache the satellite basemap."""
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
        img_dark = (img.astype(float) * 0.65).astype(np.uint8)
        return img_dark, [e0, e1, e2, e3]
    except Exception as e:
        print(f"Basemap failed: {e}")
        return None, None


def build_base_heatmap(data, extent):
    """Build the January heatmap grid (unscaled)."""
    cygnss_dets = data["cygnss"]["detections"]
    c_lats = np.array([d["lat"] for d in cygnss_dets])
    c_lons = np.array([d["lon"] for d in cygnss_dets])
    c_ints = np.array([d["intensity"] for d in cygnss_dets])

    in_view = ((c_lons > extent[0] - 0.3) & (c_lons < extent[1] + 0.3) &
               (c_lats > extent[2] - 0.3) & (c_lats < extent[3] + 0.3))
    c_lats, c_lons, c_ints = c_lats[in_view], c_lons[in_view], c_ints[in_view]

    nx, ny = 250, 200
    x_edges = np.linspace(extent[0] - 0.3, extent[1] + 0.3, nx + 1)
    y_edges = np.linspace(extent[2] - 0.3, extent[3] + 0.3, ny + 1)
    H, _, _ = np.histogram2d(c_lons, c_lats, bins=[x_edges, y_edges],
                              weights=np.log1p(c_ints))
    H = H.T
    H_smooth = gaussian_filter(H, sigma=4)

    h_extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    return H_smooth, h_extent, c_lons, c_lats


def draw_reticle(ax, lon, lat, r):
    """Small targeting reticle."""
    ax.add_patch(Circle((lon, lat), r, fill=False,
                        edgecolor=GREEN, linewidth=1.5, alpha=0.9, zorder=20))
    gap = r * 0.3
    arm = r * 1.3
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        ax.plot([lon + dx * gap, lon + dx * arm],
                [lat + dy * gap, lat + dy * arm],
                color=GREEN, linewidth=1.0, alpha=0.7, zorder=20)


def render_frame(fig, ax, basemap_img, basemap_ext, H_base, h_extent,
                 heatmap_cmap, norm_base, date, elev_pct, detections,
                 near_50, label, inset_img):
    """Render a single animation frame."""
    ax.clear()
    ax.set_facecolor(BG)

    # Extent
    pad_lon, pad_lat = 0.7, 0.55
    extent = [GT_LON - pad_lon, GT_LON + pad_lon,
              GT_LAT - pad_lat, GT_LAT + pad_lat]
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1.0 / np.cos(np.radians(GT_LAT)))

    # Basemap
    if basemap_img is not None:
        ax.imshow(basemap_img, extent=basemap_ext,
                  aspect="auto", zorder=0, interpolation="bilinear")

    # Heatmap — scaled by elevation relative to January
    if elev_pct > 0:
        scale = min(elev_pct / JAN_MEAN_ELEV, 6.0)  # cap at 6x
        H_scaled = H_base * scale
        base_vmax = np.percentile(H_base[H_base > 0], 95) if (H_base > 0).any() else 1
        vmax = base_vmax * max(scale * 0.5, 1.0)
        # Build RGBA manually: red glow with alpha from intensity
        H_norm = np.clip(H_scaled / vmax, 0, 1)
        rgba = np.zeros((*H_norm.shape, 4))
        # RGB: transition from pure red to warm orange-white at peak
        rgba[..., 0] = 1.0  # R always 1
        rgba[..., 1] = H_norm * 0.5  # G ramps up
        rgba[..., 2] = H_norm * 0.3  # B ramps slightly
        # Alpha: zero below threshold, steep ramp on hotspots only
        alpha = np.where(H_norm > 0.35, (H_norm - 0.35) / 0.65, 0)
        rgba[..., 3] = np.clip(alpha ** 2.0 * 0.5, 0, 0.5)  # max 50% opacity on peaks
        ax.imshow(rgba, extent=h_extent, origin="lower",
                  aspect="auto", zorder=2, interpolation="bilinear")

    text_fx = [pe.withStroke(linewidth=3, foreground=BG)]

    # Range rings
    for r_km in [25, 50]:
        r_deg = r_km / 111.0
        ax.add_patch(Circle((GT_LON, GT_LAT), r_deg, fill=False,
                            edgecolor=WHITE, linewidth=0.8,
                            linestyle="--", alpha=0.45, zorder=5))
        ax.text(GT_LON, GT_LAT + r_deg + 0.01,
                f"{r_km} km", fontsize=8, color=WHITE, alpha=0.5,
                ha="center", path_effects=text_fx, zorder=6)

    # Targeting reticle
    draw_reticle(ax, GT_LON, GT_LAT, 0.035)
    ax.plot(GT_LON, GT_LAT, marker="+", color=GREEN, markersize=10,
            markeredgewidth=1.5, zorder=22)

    # Ground truth label
    ax.text(GT_LON + 0.08, GT_LAT - 0.04,
            f"Ground Truth\n{GT_LAT:.4f}°N, {GT_LON:.4f}°E",
            fontsize=9, color=GREEN, fontweight="bold",
            path_effects=text_fx, zorder=25)

    # ── Title ────────────────────────────────────────────────────────────
    ax.text(0.5, 0.97, "GPS Jammer Localized from Space",
            transform=ax.transAxes, fontsize=18, color=WHITE,
            fontweight="bold", ha="center", va="top",
            path_effects=text_fx, zorder=30)

    ax.text(0.5, 0.93, "CYGNSS GNSS-R  |  Shiraz, Iran  |  Daily Monitoring",
            transform=ax.transAxes, fontsize=10, color=WHITE_DIM,
            ha="center", va="top", path_effects=text_fx, zorder=30)

    # ── Date and status — upper left ─────────────────────────────────────
    # Status color
    if elev_pct == 0:
        status_color = WHITE_DIM
    elif elev_pct < 20:
        status_color = AMBER
    else:
        status_color = RED

    ax.text(0.03, 0.88, date,
            transform=ax.transAxes, fontsize=22, color=WHITE,
            fontweight="bold", path_effects=text_fx, zorder=30)

    ax.text(0.03, 0.83, label,
            transform=ax.transAxes, fontsize=12, color=status_color,
            fontweight="bold", path_effects=text_fx, zorder=30)

    # ── Stats panel — upper left below date ──────────────────────────────
    stats_y = 0.77
    line_h = 0.04

    ax.text(0.03, stats_y, f"Noise Elevation:  +{elev_pct:.1f}%",
            transform=ax.transAxes, fontsize=11, color=WHITE,
            path_effects=text_fx, zorder=30)

    ax.text(0.03, stats_y - line_h, f"Detections:  {detections:,}",
            transform=ax.transAxes, fontsize=11, color=WHITE,
            path_effects=text_fx, zorder=30)

    ax.text(0.03, stats_y - 2 * line_h, f"Near Jammer (<50 km):  {near_50}",
            transform=ax.transAxes, fontsize=11, color=WHITE,
            path_effects=text_fx, zorder=30)

    # ── Power bar — right side ───────────────────────────────────────────
    bar_x = 0.92
    bar_w = 0.03
    bar_bottom = 0.15
    bar_height = 0.65

    # Background bar
    ax.add_patch(FancyBboxPatch(
        (bar_x, bar_bottom), bar_w, bar_height,
        transform=ax.transAxes, facecolor="#ffffff10", edgecolor=WHITE_DIM,
        linewidth=0.8, zorder=28, clip_on=False,
        boxstyle="round,pad=0.005"))

    # Fill — proportional to elevation (cap at 85%)
    fill_frac = min(elev_pct / 85.0, 1.0)
    if fill_frac > 0:
        fill_color = RED if elev_pct >= 20 else AMBER if elev_pct > 0 else WHITE_DIM
        ax.add_patch(FancyBboxPatch(
            (bar_x, bar_bottom), bar_w, bar_height * fill_frac,
            transform=ax.transAxes, facecolor=fill_color, alpha=0.7,
            edgecolor="none", zorder=29, clip_on=False,
            boxstyle="round,pad=0.005"))

    # Bar label
    ax.text(bar_x + bar_w / 2, bar_bottom + bar_height + 0.03, "POWER",
            transform=ax.transAxes, fontsize=8, color=WHITE, alpha=0.7,
            ha="center", va="bottom", fontweight="bold",
            path_effects=text_fx, zorder=30)

    ax.text(bar_x + bar_w / 2, bar_bottom - 0.02, f"+{elev_pct:.0f}%",
            transform=ax.transAxes, fontsize=9, color=WHITE,
            ha="center", va="top", fontweight="bold",
            path_effects=text_fx, zorder=30)

    # Clean axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Inset
    if inset_img is not None:
        # Remove old inset axes if any
        for a in fig.get_axes():
            if a is not ax:
                a.remove()
        inset_ax = fig.add_axes([0.02, 0.02, 0.22, 0.22], zorder=40)
        inset_ax.imshow(inset_img)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        for spine in inset_ax.spines.values():
            spine.set_edgecolor(WHITE)
            spine.set_linewidth(1.5)


def main():
    data = load_data()

    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.edgecolor": "none",
        "font.family": "Helvetica, Arial, sans-serif",
    })

    # Setup extent
    pad_lon, pad_lat = 0.7, 0.55
    extent = [GT_LON - pad_lon, GT_LON + pad_lon,
              GT_LAT - pad_lat, GT_LAT + pad_lat]

    # Pre-compute expensive things once
    print("Fetching basemap...")
    basemap_img, basemap_ext = get_basemap(extent)

    print("Building heatmap grid...")
    H_base, h_extent, _, _ = build_base_heatmap(data, extent)

    # Heatmap colormap — lower alphas to keep basemap visible
    heatmap_cmap = LinearSegmentedColormap.from_list("heat", [
        (0.0, (1, 0.1, 0.1, 0)),
        (0.5, (1, 0.1, 0.1, 0)),
        (0.65, (1, 0.15, 0.1, 0.05)),
        (0.75, (1, 0.15, 0.05, 0.10)),
        (0.85, (1, 0.2, 0.1, 0.20)),
        (0.95, (1, 0.5, 0.3, 0.30)),
        (1.0, (1, 0.85, 0.7, 0.40)),
    ])
    norm_base = Normalize(vmin=0, vmax=np.percentile(H_base[H_base > 0], 95)
                          if (H_base > 0).any() else 1)

    # Load inset
    inset_img = None
    if INSET_IMG.exists():
        inset_img = Image.open(INSET_IMG)

    # Render frames
    fig, ax = plt.subplots(figsize=(14, 11))
    frames = []

    for i, (date, elev, dets, near, label) in enumerate(TIMELINE):
        print(f"  Frame {i+1}/{len(TIMELINE)}: {date} — {label}")
        render_frame(fig, ax, basemap_img, basemap_ext, H_base, h_extent,
                     heatmap_cmap, norm_base, date, elev, dets, near, label,
                     inset_img)

        # Render to image
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]  # drop alpha
        frames.append(Image.fromarray(img.copy()))

    plt.close(fig)

    # Build GIF — hold baseline/key frames longer
    durations = []
    for i, (date, elev, dets, near, label) in enumerate(TIMELINE):
        if label == "BASELINE":
            durations.append(1500)  # 1.5s on baseline
        elif label == "CONFLICT BEGINS":
            durations.append(2500)  # 2.5s on conflict start
        elif label == "PEAK POWER":
            durations.append(2500)  # 2.5s on peak
        elif label in ("STILL ACTIVE",) and i == len(TIMELINE) - 1:
            durations.append(3000)  # 3s on final frame
        else:
            durations.append(1200)  # 1.2s default

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "jammer_timeline_animated.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)

    print(f"\nSaved: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    main()
