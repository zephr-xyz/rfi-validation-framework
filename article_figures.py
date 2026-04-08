#!/usr/bin/env python3
"""
GPS World Article Figures
=========================
Publication-quality visualizations for GPS jammer localization validation:
CYGNSS (GNSS-R) vs NISAR (L-band SAR) vs Fused.

Generates Figure 1 (hero/cover): "Converging on the Jammer" — dual-modality
detection map on dark background.

Usage:
    python3 article_figures.py                  # all figures
    python3 article_figures.py --figure 1       # hero only
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.optimize import minimize

# ── Paths ────────────────────────────────────────────────────────────────
RESULTS_JSON = Path("output/comparison_results.json")
OUTPUT_DIR = Path("output/article_figures")

# ── Ground truth ─────────────────────────────────────────────────────────
GT_LAT, GT_LON = 27.3182, 52.8703

# ── Dark theme palette ───────────────────────────────────────────────────
BG_COLOR = "#0d1117"
BG_LIGHT = "#161b22"
TEXT_COLOR = "#e6edf3"
TEXT_DIM = "#8b949e"
GRID_COLOR = "#21262d"

CYGNSS_COLOR = "#f59f00"       # amber
CYGNSS_HOT = "#e03131"         # red for high intensity
NISAR_COLOR = "#339af0"        # electric blue
NISAR_LIGHT = "#74c0fc"
FUSED_COLOR = "#51cf66"        # green
GT_COLOR = "#ffd43b"           # gold
CEP_CYGNSS = "#f59f0040"
CEP_NISAR = "#339af040"


def load_results():
    with open(RESULTS_JSON) as f:
        return json.load(f)


def setup_dark_style():
    """Configure matplotlib for dark publication theme."""
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_LIGHT,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.titlepad": 12,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_DIM,
        "ytick.color": TEXT_DIM,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.4,
        "legend.facecolor": BG_LIGHT,
        "legend.edgecolor": GRID_COLOR,
        "legend.labelcolor": TEXT_COLOR,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "savefig.facecolor": BG_COLOR,
        "savefig.edgecolor": "none",
    })


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Hero / Cover: "Converging on the Jammer"
# ═══════════════════════════════════════════════════════════════════════════

def figure1_hero(data):
    """
    Full-page hero map showing both modalities converging on the jammer.
    - CYGNSS detections as scatter, colored by intensity (amber→red gradient)
    - NISAR detections as electric-blue markers with bearing lines
    - Ground truth gold star
    - Estimated positions with CEP circles
    - Distance annotations
    """
    setup_dark_style()

    fig, ax = plt.subplots(figsize=(12, 14))

    cygnss_dets = data["cygnss"]["detections"]
    nisar_dets = data["nisar"]["detections"]
    cygnss_est = (data["cygnss"]["estimated_lat"], data["cygnss"]["estimated_lon"])
    nisar_est = (data["nisar"]["estimated_lat"], data["nisar"]["estimated_lon"])
    fused_est = (data["fused"]["estimated_lat"], data["fused"]["estimated_lon"])

    # ── CYGNSS detections (amber→red by intensity) ──────────────────────
    c_lats = np.array([d["lat"] for d in cygnss_dets])
    c_lons = np.array([d["lon"] for d in cygnss_dets])
    c_ints = np.array([d["intensity"] for d in cygnss_dets])

    # Custom colormap: amber → orange → red
    cygnss_cmap = LinearSegmentedColormap.from_list(
        "cygnss_heat", ["#f59f00", "#e8590c", "#e03131"], N=256
    )

    # Log-scale intensity for better visual spread
    c_ints_log = np.log1p(c_ints)
    norm = Normalize(vmin=np.percentile(c_ints_log, 5),
                     vmax=np.percentile(c_ints_log, 95))

    # Size by proximity to jammer (closer = larger)
    c_dists = np.array([d["metadata"]["distance_km"] for d in cygnss_dets])
    c_sizes = np.clip(8 + 80 * (1 - c_dists / c_dists.max()), 8, 80)

    sc = ax.scatter(c_lons, c_lats, c=c_ints_log, cmap=cygnss_cmap, norm=norm,
                    s=c_sizes, alpha=0.6, edgecolors="none", zorder=3,
                    rasterized=True)

    # ── NISAR detections (electric blue) ─────────────────────────────────
    n_lats = np.array([d["lat"] for d in nisar_dets])
    n_lons = np.array([d["lon"] for d in nisar_dets])
    n_ints = np.array([d["intensity"] for d in nisar_dets])

    # Get NISAR dates from filenames
    nisar_dates = []
    for d in nisar_dets:
        fn = d["timestamp"]
        if "20260108" in fn:
            nisar_dates.append("Jan 8")
        elif "20260120" in fn:
            nisar_dates.append("Jan 20")
        else:
            nisar_dates.append("?")

    # Plot NISAR by date with different markers
    for date_label, marker, offset in [("Jan 8", "D", 0), ("Jan 20", "s", 0)]:
        mask = [nd == date_label for nd in nisar_dates]
        if any(mask):
            mlats = n_lats[mask]
            mlons = n_lons[mask]
            ax.scatter(mlons, mlats, c=NISAR_COLOR, s=90, alpha=0.9,
                      edgecolors="white", linewidths=0.8, marker=marker,
                      zorder=6, label=f"NISAR {date_label}")

    # ── Bearing lines from NISAR clusters toward intersection ────────────
    # Jan 8 cluster center
    jan8_mask = [nd == "Jan 8" for nd in nisar_dates]
    jan20_mask = [nd == "Jan 20" for nd in nisar_dates]

    if any(jan8_mask):
        j8_center = (np.mean(n_lats[jan8_mask]), np.mean(n_lons[jan8_mask]))
        # Draw bearing line through cluster toward intersection
        bearing_target = (data["nisar"]["estimated_lat"], data["nisar"]["estimated_lon"])
        ax.plot([j8_center[1], bearing_target[1]], [j8_center[0], bearing_target[0]],
                color=NISAR_LIGHT, linewidth=1.5, linestyle="--", alpha=0.7, zorder=5)

    if any(jan20_mask):
        j20_center = (np.mean(n_lats[jan20_mask]), np.mean(n_lons[jan20_mask]))
        bearing_target = (data["nisar"]["estimated_lat"], data["nisar"]["estimated_lon"])
        ax.plot([j20_center[1], bearing_target[1]], [j20_center[0], bearing_target[0]],
                color=NISAR_LIGHT, linewidth=1.5, linestyle="--", alpha=0.7, zorder=5)

    # ── CEP circles ──────────────────────────────────────────────────────
    # NISAR CEP (6.88 km ≈ 0.062°)
    nisar_cep_deg = data["nisar"]["cep_km"] / 111.0
    cep_nisar = Circle((nisar_est[1], nisar_est[0]), nisar_cep_deg,
                        fill=False, edgecolor=NISAR_COLOR, linewidth=2,
                        linestyle="-", alpha=0.8, zorder=7)
    ax.add_patch(cep_nisar)
    # CEP fill
    cep_nisar_fill = Circle((nisar_est[1], nisar_est[0]), nisar_cep_deg,
                             fill=True, facecolor=CEP_NISAR, edgecolor="none",
                             zorder=2)
    ax.add_patch(cep_nisar_fill)

    # CYGNSS CEP would be enormous (127 km), just note it in text
    # Show a small portion to suggest the scale
    cygnss_cep_deg = min(data["cygnss"]["cep_km"] / 111.0, 2.0)  # cap at 2° for display

    # ── Estimated positions ──────────────────────────────────────────────
    text_effects = [pe.withStroke(linewidth=3, foreground=BG_COLOR)]

    # CYGNSS estimate
    ax.plot(cygnss_est[1], cygnss_est[0], marker="^", color=CYGNSS_COLOR,
            markersize=18, markeredgecolor="white", markeredgewidth=1.5, zorder=9)
    ax.annotate(f"CYGNSS\n4.33 km", (cygnss_est[1], cygnss_est[0]),
                textcoords="offset points", xytext=(15, 15),
                fontsize=10, fontweight="bold", color=CYGNSS_COLOR,
                path_effects=text_effects, zorder=10)

    # NISAR estimate
    ax.plot(nisar_est[1], nisar_est[0], marker="v", color=NISAR_COLOR,
            markersize=18, markeredgecolor="white", markeredgewidth=1.5, zorder=9)
    ax.annotate(f"NISAR\n6.26 km", (nisar_est[1], nisar_est[0]),
                textcoords="offset points", xytext=(15, -25),
                fontsize=10, fontweight="bold", color=NISAR_COLOR,
                path_effects=text_effects, zorder=10)

    # Fused estimate
    ax.plot(fused_est[1], fused_est[0], marker="*", color=FUSED_COLOR,
            markersize=22, markeredgecolor="white", markeredgewidth=1, zorder=9)
    ax.annotate(f"Fused\n4.91 km", (fused_est[1], fused_est[0]),
                textcoords="offset points", xytext=(-55, 15),
                fontsize=10, fontweight="bold", color=FUSED_COLOR,
                path_effects=text_effects, zorder=10)

    # ── Ground truth ─────────────────────────────────────────────────────
    ax.plot(GT_LON, GT_LAT, marker="*", color=GT_COLOR, markersize=28,
            markeredgecolor="white", markeredgewidth=2, zorder=11)
    ax.annotate("GPS JAMMER\n(Ground Truth)", (GT_LON, GT_LAT),
                textcoords="offset points", xytext=(-80, -35),
                fontsize=11, fontweight="bold", color=GT_COLOR,
                path_effects=text_effects, zorder=12,
                arrowprops=dict(arrowstyle="->", color=GT_COLOR, lw=1.5))

    # ── Distance scale bar ───────────────────────────────────────────────
    # 50 km scale bar in bottom-left
    scale_lat = 26.0
    scale_lon = 51.2
    km_per_deg = 111.0 * np.cos(np.radians(scale_lat))
    scale_deg = 50.0 / km_per_deg
    ax.plot([scale_lon, scale_lon + scale_deg], [scale_lat, scale_lat],
            color=TEXT_COLOR, linewidth=3, zorder=10)
    ax.plot([scale_lon, scale_lon], [scale_lat - 0.03, scale_lat + 0.03],
            color=TEXT_COLOR, linewidth=2, zorder=10)
    ax.plot([scale_lon + scale_deg, scale_lon + scale_deg],
            [scale_lat - 0.03, scale_lat + 0.03],
            color=TEXT_COLOR, linewidth=2, zorder=10)
    ax.text(scale_lon + scale_deg / 2, scale_lat + 0.08, "50 km",
            ha="center", fontsize=10, color=TEXT_COLOR, fontweight="bold",
            path_effects=text_effects, zorder=10)

    # ── Map extent & grid ────────────────────────────────────────────────
    ax.set_xlim(50.5, 55.2)
    ax.set_ylim(25.3, 29.3)
    ax.set_xlabel("Longitude (°E)", fontsize=13)
    ax.set_ylabel("Latitude (°N)", fontsize=13)
    ax.grid(True, alpha=0.3, color=GRID_COLOR)
    ax.set_aspect(1.0 / np.cos(np.radians(27.3)))

    # ── Title ────────────────────────────────────────────────────────────
    ax.set_title(
        "Converging on the Jammer: Dual-Satellite GPS Interference Localization",
        fontsize=16, fontweight="bold", color=TEXT_COLOR, pad=16
    )

    # ── Subtitle / context ───────────────────────────────────────────────
    fig.text(0.5, 0.01,
             "785 CYGNSS GNSS-R reflections (amber) + 17 NISAR L-band SAR detections (blue)  |  "
             "Shiraz, Iran  |  January 2026",
             ha="center", fontsize=10, color=TEXT_DIM, style="italic")

    # ── Colorbar for CYGNSS intensity ────────────────────────────────────
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.35])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("CYGNSS Signal Anomaly (log)", fontsize=10, color=TEXT_DIM)
    cbar.ax.yaxis.set_tick_params(color=TEXT_DIM)
    plt.setp(plt.getp(cbar.ax, "yticklabels"), color=TEXT_DIM, fontsize=9)

    # ── Legend ────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor=GT_COLOR,
               markeredgecolor="white", markersize=14, label="Ground Truth"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=CYGNSS_COLOR,
               markeredgecolor="white", markersize=11,
               label=f"CYGNSS Est. (4.33 km, CEP 127 km)"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor=NISAR_COLOR,
               markeredgecolor="white", markersize=11,
               label=f"NISAR Est. (6.26 km, CEP 6.9 km)"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=FUSED_COLOR,
               markeredgecolor="white", markersize=13,
               label=f"Fused Est. (4.91 km)"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=NISAR_COLOR,
               markeredgecolor="white", markersize=8, label="NISAR Jan 8"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=NISAR_COLOR,
               markeredgecolor="white", markersize=8, label="NISAR Jan 20"),
        Line2D([0], [0], color=NISAR_LIGHT, linestyle="--", linewidth=1.5,
               label="NISAR Bearing Lines"),
        Circle((0, 0), 0.1, fill=False, edgecolor=NISAR_COLOR, linewidth=2,
               label="NISAR CEP (6.9 km)"),
    ]
    leg = ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
                    framealpha=0.9, fancybox=True)
    leg.get_frame().set_facecolor(BG_LIGHT)
    leg.get_frame().set_edgecolor(GRID_COLOR)

    # ── Inset: zoomed view of convergence zone ───────────────────────────
    inset = fig.add_axes([0.14, 0.13, 0.35, 0.28])
    inset.set_facecolor(BG_LIGHT)

    # Zoom to ~0.15° around ground truth
    zoom = 0.12
    inset.set_xlim(GT_LON - zoom, GT_LON + 0.15)
    inset.set_ylim(GT_LAT - zoom, GT_LAT + 0.12)

    # NISAR CEP circle in inset
    cep_inset = Circle((nisar_est[1], nisar_est[0]), nisar_cep_deg,
                        fill=True, facecolor=CEP_NISAR, edgecolor=NISAR_COLOR,
                        linewidth=1.5, alpha=0.5, zorder=2)
    inset.add_patch(cep_inset)

    # Nearby CYGNSS detections
    close_mask = c_dists < 30
    if close_mask.any():
        inset.scatter(c_lons[close_mask], c_lats[close_mask],
                     c=c_ints_log[close_mask], cmap=cygnss_cmap, norm=norm,
                     s=40, alpha=0.7, edgecolors="none", zorder=3)

    # NISAR detections
    inset.scatter(n_lons, n_lats, c=NISAR_COLOR, s=50, alpha=0.9,
                 edgecolors="white", linewidths=0.6, marker="D", zorder=6)

    # Estimates
    inset.plot(GT_LON, GT_LAT, marker="*", color=GT_COLOR, markersize=18,
              markeredgecolor="white", markeredgewidth=1.5, zorder=11)
    inset.plot(cygnss_est[1], cygnss_est[0], marker="^", color=CYGNSS_COLOR,
              markersize=12, markeredgecolor="white", markeredgewidth=1, zorder=9)
    inset.plot(nisar_est[1], nisar_est[0], marker="v", color=NISAR_COLOR,
              markersize=12, markeredgecolor="white", markeredgewidth=1, zorder=9)
    inset.plot(fused_est[1], fused_est[0], marker="*", color=FUSED_COLOR,
              markersize=14, markeredgecolor="white", markeredgewidth=1, zorder=9)

    inset.set_aspect(1.0 / np.cos(np.radians(GT_LAT)))
    inset.grid(True, alpha=0.2, color=GRID_COLOR)
    inset.set_title("Convergence Zone", fontsize=10, color=TEXT_COLOR, pad=6)
    for spine in inset.spines.values():
        spine.set_edgecolor(TEXT_DIM)
        spine.set_linewidth(1.5)

    # ── Save ─────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "fig1_hero_converging_on_jammer.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

FIGURES = {
    1: ("Hero: Converging on the Jammer", figure1_hero),
}


def main():
    parser = argparse.ArgumentParser(description="GPS World article figures")
    parser.add_argument("--figure", type=int, help="Generate specific figure (1-7)")
    args = parser.parse_args()

    data = load_results()

    if args.figure:
        name, func = FIGURES[args.figure]
        print(f"Generating Figure {args.figure}: {name}")
        func(data)
    else:
        for num, (name, func) in sorted(FIGURES.items()):
            print(f"Generating Figure {num}: {name}")
            func(data)

    print("Done.")


if __name__ == "__main__":
    main()
