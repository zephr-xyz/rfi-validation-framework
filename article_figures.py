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
    inset = fig.add_axes([0.15, 0.22, 0.25, 0.20])
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
    inset.set_title("Convergence Zone", fontsize=9, color=TEXT_COLOR, pad=4)
    inset.set_xticklabels([])
    inset.set_yticklabels([])
    inset.tick_params(length=0)
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
# FIGURE 2 — Jammer ON vs OFF: The Smoking Gun
# ═══════════════════════════════════════════════════════════════════════════

def figure2_on_vs_off(data):
    """Side-by-side: 785 detections (ON) vs 0 detections (OFF)."""
    setup_dark_style()
    fig, (ax_on, ax_off) = plt.subplots(1, 2, figsize=(16, 8))

    cygnss_dets = data["cygnss"]["detections"]
    c_lats = np.array([d["lat"] for d in cygnss_dets])
    c_lons = np.array([d["lon"] for d in cygnss_dets])
    c_ints = np.array([d["intensity"] for d in cygnss_dets])

    cygnss_cmap = LinearSegmentedColormap.from_list(
        "cygnss_heat", ["#f59f00", "#e8590c", "#e03131"], N=256
    )
    c_ints_log = np.log1p(c_ints)
    norm = Normalize(vmin=np.percentile(c_ints_log, 5),
                     vmax=np.percentile(c_ints_log, 95))

    extent = [50.5, 55.2, 25.3, 29.3]
    aspect = 1.0 / np.cos(np.radians(27.3))

    # ON panel
    ax_on.set_title("Jammer ON  —  January 8 & 20, 2026", fontsize=13,
                    fontweight="bold", color="#e03131")
    sc = ax_on.scatter(c_lons, c_lats, c=c_ints_log, cmap=cygnss_cmap, norm=norm,
                       s=25, alpha=0.6, edgecolors="none", rasterized=True, zorder=3)
    ax_on.plot(GT_LON, GT_LAT, marker="*", color=GT_COLOR, markersize=22,
              markeredgecolor="white", markeredgewidth=1.5, zorder=10)

    text_fx = [pe.withStroke(linewidth=3, foreground=BG_COLOR)]
    ax_on.text(0.03, 0.97, f"785 detections", transform=ax_on.transAxes,
              fontsize=16, fontweight="bold", color="#e03131", va="top",
              path_effects=text_fx)
    ax_on.set_xlim(extent[0], extent[1])
    ax_on.set_ylim(extent[2], extent[3])
    ax_on.set_aspect(aspect)
    ax_on.set_xlabel("Longitude (°E)")
    ax_on.set_ylabel("Latitude (°N)")
    ax_on.grid(True, alpha=0.3)

    # OFF panel
    ax_off.set_title("Jammer OFF  —  December 15 & 27, 2025", fontsize=13,
                     fontweight="bold", color=FUSED_COLOR)
    ax_off.plot(GT_LON, GT_LAT, marker="*", color=GT_COLOR, markersize=22,
               markeredgecolor="white", markeredgewidth=1.5, zorder=10)
    ax_off.text(0.03, 0.97, "0 detections", transform=ax_off.transAxes,
               fontsize=16, fontweight="bold", color=FUSED_COLOR, va="top",
               path_effects=text_fx)
    ax_off.text(0.5, 0.5, "CLEAN", transform=ax_off.transAxes,
               fontsize=48, fontweight="bold", color="#2ea04330",
               ha="center", va="center", zorder=1)
    ax_off.set_xlim(extent[0], extent[1])
    ax_off.set_ylim(extent[2], extent[3])
    ax_off.set_aspect(aspect)
    ax_off.set_xlabel("Longitude (°E)")
    ax_off.set_ylabel("Latitude (°N)")
    ax_off.grid(True, alpha=0.3)

    # Shared colorbar
    cbar = fig.colorbar(sc, ax=[ax_on, ax_off], shrink=0.7, pad=0.02)
    cbar.set_label("Signal Anomaly (log)", color=TEXT_DIM)
    cbar.ax.yaxis.set_tick_params(color=TEXT_DIM)
    plt.setp(plt.getp(cbar.ax, "yticklabels"), color=TEXT_DIM)

    fig.suptitle("The Smoking Gun: CYGNSS Jammer Detection Baseline Test",
                 fontsize=15, fontweight="bold", y=0.98)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "fig2_on_vs_off.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — CYGNSS 1/r² Gradient
# ═══════════════════════════════════════════════════════════════════════════

def figure3_inverse_distance(data):
    """Intensity vs distance with fitted 1/r² curve."""
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    cygnss_dets = data["cygnss"]["detections"]
    dists = np.array([d["metadata"]["distance_km"] for d in cygnss_dets])
    ints = np.array([d["intensity"] for d in cygnss_dets])

    # Color by detection method
    method_colors = {
        "precomp_noise": ("#f59f00", "DDM Noise Floor"),
        "nbrcs_drop": ("#e03131", "NBRCS Drop"),
        "spatial_noise_grid": ("#ff922b", "Spatial Noise Grid"),
        "spatial_hole": ("#ffd43b", "SNR Hole"),
    }

    for method, (color, label) in method_colors.items():
        mask = np.array([d["metadata"].get("method") == method for d in cygnss_dets])
        if mask.any():
            ax.scatter(dists[mask], ints[mask], c=color, s=20, alpha=0.5,
                      label=f"{label} ({mask.sum()})", edgecolors="none",
                      rasterized=True, zorder=3)

    # Fitted 1/r² curve
    fit = data.get("cygnss_inv_dist_fit", {})
    amp = fit.get("amplitude", 248.68)
    r_fit = np.linspace(1, 200, 500)
    i_fit = amp / (r_fit ** 2)

    ax.plot(r_fit, i_fit, color="white", linewidth=2.5, zorder=5,
            label=f"1/r² fit (A={amp:.0f})")
    ax.plot(r_fit, i_fit, color=CYGNSS_COLOR, linewidth=1.5, linestyle="--",
            zorder=6)

    # Mark estimated jammer distance = 0
    ax.axvline(x=4.33, color=GT_COLOR, linewidth=1.5, linestyle=":",
              alpha=0.8, zorder=4, label="Est. jammer (4.33 km)")

    ax.set_xlabel("Distance from Estimated Jammer (km)", fontsize=13)
    ax.set_ylabel("Signal Anomaly Intensity", fontsize=13)
    ax.set_title("CYGNSS 1/r² Inverse-Distance Jammer Model",
                fontsize=15, fontweight="bold")
    ax.set_xlim(0, 210)
    ax.set_ylim(0, np.percentile(ints, 99.5) * 1.1)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Annotation
    text_fx = [pe.withStroke(linewidth=3, foreground=BG_COLOR)]
    ax.annotate("Jammer signature:\nintensity ∝ 1/r²",
               xy=(15, amp / 225), xytext=(60, amp / 4),
               fontsize=11, color=TEXT_COLOR, fontweight="bold",
               path_effects=text_fx,
               arrowprops=dict(arrowstyle="->", color=CYGNSS_COLOR, lw=2),
               zorder=7)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "fig3_inverse_distance.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — NISAR Streak Anatomy
# ═══════════════════════════════════════════════════════════════════════════

def figure4_nisar_streaks(data):
    """NISAR detections with streak bearings and intersection geometry."""
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(10, 10))

    nisar_dets = data["nisar"]["detections"]
    nisar_est = (data["nisar"]["estimated_lat"], data["nisar"]["estimated_lon"])

    n_lats = np.array([d["lat"] for d in nisar_dets])
    n_lons = np.array([d["lon"] for d in nisar_dets])
    n_ints = np.array([d["intensity"] for d in nisar_dets])

    # Separate by date
    jan8_mask = np.array(["20260108" in d["timestamp"] for d in nisar_dets])
    jan20_mask = np.array(["20260120" in d["timestamp"] for d in nisar_dets])

    # Plot detections sized by intensity
    for mask, color, marker, label in [
        (jan8_mask, "#339af0", "D", "Jan 8 Pass (Track 157)"),
        (jan20_mask, "#74c0fc", "s", "Jan 20 Pass (Track 157)"),
    ]:
        if mask.any():
            sizes = 60 + 400 * (n_ints[mask] / n_ints.max())
            ax.scatter(n_lons[mask], n_lats[mask], c=color, s=sizes, alpha=0.85,
                      edgecolors="white", linewidths=1, marker=marker, zorder=6,
                      label=label)

    # Bearing lines — extend beyond cluster to show intersection
    for mask, color in [(jan8_mask, "#339af080"), (jan20_mask, "#74c0fc80")]:
        if mask.any():
            clat, clon = np.mean(n_lats[mask]), np.mean(n_lons[mask])
            # Direction from cluster center to intersection
            dlat = nisar_est[0] - clat
            dlon = nisar_est[1] - clon
            length = np.sqrt(dlat**2 + dlon**2)
            if length > 0:
                # Extend line in both directions
                ext = 0.15  # degrees
                ax.plot([clon - dlon/length*ext, clon + dlon/length*(ext+length)],
                       [clat - dlat/length*ext, clat + dlat/length*(ext+length)],
                       color=color, linewidth=2.5, linestyle="--", zorder=4)

    # Intersection point
    ax.plot(nisar_est[1], nisar_est[0], marker="X", color=NISAR_COLOR,
            markersize=18, markeredgecolor="white", markeredgewidth=2, zorder=9)

    # CEP circle
    nisar_cep_deg = data["nisar"]["cep_km"] / 111.0
    cep = Circle((nisar_est[1], nisar_est[0]), nisar_cep_deg,
                  fill=True, facecolor="#339af015", edgecolor=NISAR_COLOR,
                  linewidth=2, linestyle="-", zorder=2)
    ax.add_patch(cep)

    # Ground truth
    ax.plot(GT_LON, GT_LAT, marker="*", color=GT_COLOR, markersize=24,
            markeredgecolor="white", markeredgewidth=2, zorder=11)

    text_fx = [pe.withStroke(linewidth=3, foreground=BG_COLOR)]
    ax.annotate("Ground Truth", (GT_LON, GT_LAT),
               textcoords="offset points", xytext=(-70, -25),
               fontsize=11, fontweight="bold", color=GT_COLOR,
               path_effects=text_fx, zorder=12,
               arrowprops=dict(arrowstyle="->", color=GT_COLOR, lw=1.5))

    ax.annotate(f"Bearing Intersection\n6.26 km error\nCEP 6.88 km",
               (nisar_est[1], nisar_est[0]),
               textcoords="offset points", xytext=(20, 25),
               fontsize=10, fontweight="bold", color=NISAR_COLOR,
               path_effects=text_fx, zorder=12,
               arrowprops=dict(arrowstyle="->", color=NISAR_COLOR, lw=1.5))

    # Distance annotation line between GT and estimate
    ax.plot([GT_LON, nisar_est[1]], [GT_LAT, nisar_est[0]],
            color=TEXT_DIM, linewidth=1, linestyle=":", alpha=0.8, zorder=5)
    mid_lon = (GT_LON + nisar_est[1]) / 2
    mid_lat = (GT_LAT + nisar_est[0]) / 2
    ax.text(mid_lon - 0.02, mid_lat, "6.26 km", fontsize=9, color=TEXT_DIM,
            rotation=25, path_effects=text_fx, zorder=10)

    # Zoom to NISAR region
    pad = 0.12
    all_lons = list(n_lons) + [GT_LON, nisar_est[1]]
    all_lats = list(n_lats) + [GT_LAT, nisar_est[0]]
    ax.set_xlim(min(all_lons) - pad, max(all_lons) + pad)
    ax.set_ylim(min(all_lats) - pad, max(all_lats) + pad)
    ax.set_aspect(1.0 / np.cos(np.radians(GT_LAT)))

    ax.set_xlabel("Longitude (°E)", fontsize=13)
    ax.set_ylabel("Latitude (°N)", fontsize=13)
    ax.set_title("NISAR L-band SAR: RFI Streak Bearing Intersection",
                fontsize=15, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Info box
    info = (f"Sensor: NISAR L-band GCOV (1.257 GHz)\n"
            f"Method: λ₁ eigenvalue decomposition\n"
            f"Bearings: 308.1° + 316.2° → intersection\n"
            f"17 detections across 2 ascending passes")
    ax.text(0.97, 0.03, info, transform=ax.transAxes, fontsize=9,
            color=TEXT_DIM, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=BG_LIGHT,
                      edgecolor=GRID_COLOR, alpha=0.9))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "fig4_nisar_streaks.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Three-Column Method Comparison
# ═══════════════════════════════════════════════════════════════════════════

def figure5_method_comparison(data):
    """Three-column comparison: CYGNSS vs NISAR vs Fused with metrics."""
    setup_dark_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))

    methods = [
        ("CYGNSS\n(GNSS-R)", data["cygnss"], CYGNSS_COLOR, "^"),
        ("NISAR\n(L-band SAR)", data["nisar"], NISAR_COLOR, "v"),
        ("Fused\n(Bayesian)", data["fused"], FUSED_COLOR, "*"),
    ]

    for ax, (label, result, color, marker) in zip(axes, methods):
        est_lat = result["estimated_lat"]
        est_lon = result["estimated_lon"]
        err = result["euclidean_error_km"]
        cep = result["cep_km"]
        n_det = result["num_detections"]

        # Plot detections if available
        if result.get("detections"):
            dets = result["detections"]
            d_lats = [d["lat"] for d in dets]
            d_lons = [d["lon"] for d in dets]
            d_ints = [d["intensity"] for d in dets]
            ax.scatter(d_lons, d_lats, c=color, s=15, alpha=0.4,
                      edgecolors="none", rasterized=True, zorder=3)

        # Ground truth
        ax.plot(GT_LON, GT_LAT, marker="*", color=GT_COLOR, markersize=18,
               markeredgecolor="white", markeredgewidth=1.5, zorder=10)

        # Estimate
        ax.plot(est_lon, est_lat, marker=marker, color=color, markersize=16,
               markeredgecolor="white", markeredgewidth=1.5, zorder=9)

        # CEP circle (cap at reasonable display size)
        cep_deg = min(cep, 20) / 111.0
        cep_circle = Circle((est_lon, est_lat), cep_deg,
                            fill=True, facecolor=color + "15",
                            edgecolor=color, linewidth=1.5, zorder=2)
        ax.add_patch(cep_circle)

        # Error line
        ax.plot([GT_LON, est_lon], [GT_LAT, est_lat],
               color=color, linewidth=1.5, linestyle=":", alpha=0.8, zorder=5)

        # Zoom to relevant area
        pad = 0.15
        ax.set_xlim(GT_LON - pad, max(est_lon, GT_LON) + pad)
        ax.set_ylim(GT_LAT - pad, max(est_lat, GT_LAT) + pad)
        ax.set_aspect(1.0 / np.cos(np.radians(GT_LAT)))
        ax.grid(True, alpha=0.3)

        # Title with metrics
        ax.set_title(label, fontsize=14, fontweight="bold", color=color)

        # Metrics box
        cep_str = f"{cep:.1f}" if cep < 200 else f"{cep:.0f}"
        metrics = (f"Error: {err:.2f} km\n"
                   f"CEP: {cep_str} km\n"
                   f"Detections: {n_det}")
        ax.text(0.97, 0.03, metrics, transform=ax.transAxes, fontsize=11,
               color=color, ha="right", va="bottom", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_LIGHT,
                         edgecolor=color, alpha=0.9, linewidth=1.5))

    axes[0].set_ylabel("Latitude (°N)")
    for ax in axes:
        ax.set_xlabel("Longitude (°E)")

    fig.suptitle("Method Comparison: Which Satellite Wins?",
                fontsize=16, fontweight="bold", y=0.98)
    fig.text(0.5, 0.01,
             "Gold star = ground truth  |  Colored marker = estimate  |  "
             "Circle = CEP (50% confidence radius)",
             ha="center", fontsize=10, color=TEXT_DIM, style="italic")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "fig5_method_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Bayesian Fusion Mechanics
# ═══════════════════════════════════════════════════════════════════════════

def figure6_bayesian_fusion(data):
    """Overlapping 2D Gaussians showing precision-weighted fusion."""
    setup_dark_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    cygnss_est = np.array([data["cygnss"]["estimated_lon"], data["cygnss"]["estimated_lat"]])
    nisar_est = np.array([data["nisar"]["estimated_lon"], data["nisar"]["estimated_lat"]])
    fused_est = np.array([data["fused"]["estimated_lon"], data["fused"]["estimated_lat"]])

    cygnss_sigma = data["cygnss"]["cep_km"] / 1.1774 / 111.0  # degrees
    nisar_sigma = data["nisar"]["cep_km"] / 1.1774 / 111.0

    # Grid for Gaussian contours — centered on NISAR (tighter)
    center = nisar_est
    extent = 0.25  # degrees
    x = np.linspace(center[0] - extent, center[0] + extent, 300)
    y = np.linspace(center[1] - extent, center[1] + extent, 300)
    X, Y = np.meshgrid(x, y)

    def gauss2d(X, Y, cx, cy, sigma):
        return np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

    Z_cygnss = gauss2d(X, Y, cygnss_est[0], cygnss_est[1], cygnss_sigma)
    Z_nisar = gauss2d(X, Y, nisar_est[0], nisar_est[1], nisar_sigma)

    # Bayesian product (multiply, renormalize)
    Z_fused = Z_cygnss * Z_nisar
    Z_fused_max = Z_fused.max()
    if Z_fused_max > 0:
        Z_fused = Z_fused / Z_fused_max

    panels = [
        (axes[0], Z_cygnss, "CYGNSS Likelihood\nσ = 108 km", CYGNSS_COLOR, cygnss_est),
        (axes[1], Z_nisar, "NISAR Likelihood\nσ = 5.8 km", NISAR_COLOR, nisar_est),
        (axes[2], Z_fused, "Bayesian Posterior\n(Product)", FUSED_COLOR, fused_est),
    ]

    for ax, Z, title, color, est in panels:
        # Contour plot
        levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        cs = ax.contourf(X, Y, Z, levels=20, cmap=LinearSegmentedColormap.from_list(
            "custom", [BG_LIGHT, color], N=256), zorder=2)
        ax.contour(X, Y, Z, levels=levels, colors="white", linewidths=0.5,
                  alpha=0.5, zorder=3)

        # Ground truth
        ax.plot(GT_LON, GT_LAT, marker="*", color=GT_COLOR, markersize=16,
               markeredgecolor="white", markeredgewidth=1.5, zorder=10)

        # Estimate
        ax.plot(est[0], est[1], marker="+", color="white",
               markersize=14, markeredgewidth=2, zorder=9)

        ax.set_title(title, fontsize=12, fontweight="bold", color=color)
        ax.set_xlim(center[0] - extent, center[0] + extent)
        ax.set_ylim(center[1] - extent, center[1] + extent)
        ax.set_aspect(1.0 / np.cos(np.radians(GT_LAT)))
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Latitude (°N)")
    for ax in axes:
        ax.set_xlabel("Longitude (°E)")

    fig.suptitle("Bayesian Sensor Fusion: Precision-Weighted Gaussian Product",
                fontsize=15, fontweight="bold", y=1.02)

    # Equation annotation
    cygnss_cep = data["cygnss"]["cep_km"]
    nisar_cep = data["nisar"]["cep_km"]
    ratio = nisar_cep / max(cygnss_cep, 0.1)
    fig.text(0.5, -0.02,
             "P(jammer | CYGNSS, NISAR) ∝ P(CYGNSS | jammer) × P(NISAR | jammer)    |    "
             f"σ_CYGNSS/σ_NISAR ≈ 1:{ratio:.1f} → CYGNSS dominates posterior",
             ha="center", fontsize=10, color=TEXT_DIM, style="italic")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "fig6_bayesian_fusion.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Accuracy & Confidence Summary Dashboard
# ═══════════════════════════════════════════════════════════════════════════

def figure7_dashboard(data):
    """Combined metrics dashboard: bar chart + radar + key takeaways."""
    setup_dark_style()
    fig = plt.figure(figsize=(14, 8))

    # Layout: left = bar chart, right = summary table
    ax_bar = fig.add_axes([0.08, 0.15, 0.42, 0.72])
    ax_table = fig.add_axes([0.56, 0.15, 0.40, 0.72])
    ax_table.axis("off")

    modalities = ["CYGNSS", "NISAR", "Fused"]
    errors = [data["cygnss"]["euclidean_error_km"],
              data["nisar"]["euclidean_error_km"],
              data["fused"]["euclidean_error_km"]]
    ceps = [data["cygnss"]["cep_km"],
            data["nisar"]["cep_km"],
            data["fused"]["cep_km"]]
    n_dets = [data["cygnss"]["num_detections"],
              data["nisar"]["num_detections"],
              data["fused"]["num_detections"]]
    colors = [CYGNSS_COLOR, NISAR_COLOR, FUSED_COLOR]

    text_fx = [pe.withStroke(linewidth=2, foreground=BG_COLOR)]

    # Bar chart — error and CEP side by side (CEP capped for display)
    x = np.arange(len(modalities))
    width = 0.35
    ceps_display = [min(c, 15) for c in ceps]  # cap for visual

    bars_err = ax_bar.bar(x - width/2, errors, width, color=colors, alpha=0.9,
                          edgecolor="white", linewidth=0.5, label="Euclidean Error")
    bars_cep = ax_bar.bar(x + width/2, ceps_display, width, color=colors, alpha=0.4,
                          edgecolor="white", linewidth=0.5, hatch="//",
                          label="CEP (50%)")

    # Value labels
    for bar, val in zip(bars_err, errors):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   f"{val:.1f}", ha="center", fontsize=11, fontweight="bold",
                   color=TEXT_COLOR, path_effects=text_fx)
    for bar, val, orig in zip(bars_cep, ceps_display, ceps):
        label = f"{orig:.1f}" if orig < 20 else f"{orig:.0f}"
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   label, ha="center", fontsize=10, color=TEXT_DIM)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(modalities, fontsize=12, fontweight="bold")
    ax_bar.set_ylabel("Distance (km)", fontsize=12)
    ax_bar.set_title("Localization Accuracy", fontsize=14, fontweight="bold")
    ax_bar.legend(fontsize=10, loc="upper right")
    ax_bar.grid(True, alpha=0.3, axis="y")
    ax_bar.set_ylim(0, 18)

    # Note about bootstrap CEP
    boot_cep = data.get("cygnss_inv_dist_fit", {}).get("bootstrap_cep_km")
    if boot_cep and boot_cep < 10:
        ax_bar.annotate(f"Bootstrap CEP\n{boot_cep:.1f} km", xy=(0 + width/2, ceps_display[0] + 0.5),
                       fontsize=9, color=CYGNSS_COLOR, ha="center", va="bottom",
                       fontweight="bold")

    # Summary table
    rows = [
        ["Metric", "CYGNSS", "NISAR", "Fused"],
        ["Error (km)", f"{errors[0]:.2f}", f"{errors[1]:.2f}", f"{errors[2]:.2f}"],
        ["CEP (km)", f"{ceps[0]:.1f}", f"{ceps[1]:.2f}", f"{ceps[2]:.2f}"],
        ["Detections", str(n_dets[0]), str(n_dets[1]), str(n_dets[2])],
        ["Sensor", "GNSS-R", "L-band SAR", "Both"],
        ["Frequency", "L1/L2 GPS", "1.257 GHz", "—"],
        ["Method", "1/r² fit", "Bearing △", "Bayesian"],
        ["Best use", "Detect+locate", "Confirm", "Cross-validate"],
    ]

    header_colors = [TEXT_COLOR, CYGNSS_COLOR, NISAR_COLOR, FUSED_COLOR]

    y_start = 0.95
    for i, row in enumerate(rows):
        y = y_start - i * 0.11
        for j, cell in enumerate(row):
            x_pos = 0.0 + j * 0.25
            weight = "bold" if i == 0 else "normal"
            color = header_colors[j] if i == 0 else TEXT_COLOR
            fontsize = 11 if i == 0 else 10
            ax_table.text(x_pos, y, cell, fontsize=fontsize, fontweight=weight,
                         color=color, transform=ax_table.transAxes, va="center")

        # Separator line after header
        if i == 0:
            ax_table.axhline(y=y - 0.04, xmin=0, xmax=1, color=GRID_COLOR,
                            linewidth=1)

    # Key insight box
    insight = ("KEY INSIGHT: CYGNSS wins on both accuracy (4.33 km) and confidence\n"
               "(3.48 km bootstrap CEP). The 1/r² fit converges stably across 500\n"
               "bootstrap resamples. With comparable CEPs, Bayesian fusion now\n"
               "genuinely blends both sensors (4.69 km fused estimate).")
    ax_table.text(0.0, -0.05, insight, transform=ax_table.transAxes,
                 fontsize=9, color=TEXT_DIM, va="top",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=BG_LIGHT,
                          edgecolor=GT_COLOR, alpha=0.9, linewidth=1))

    fig.suptitle("GPS Jammer Localization: Results Summary",
                fontsize=16, fontweight="bold", y=0.97)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "fig7_dashboard.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Jammer Persistence Timeline
# ═══════════════════════════════════════════════════════════════════════════

def figure8_timeline(data):
    """Jammer activity from Jan through Apr 2026 showing escalation during conflict."""
    setup_dark_style()
    fig, (ax_elev, ax_det) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                           gridspec_kw={"height_ratios": [2, 1]})

    from datetime import datetime

    # All data points collected (date, elevation_pct, detections, near_50km)
    timeline = [
        # Baseline (jammer OFF)
        ("2025-12-15", 0.0, 0, 0),
        ("2025-12-27", 0.0, 0, 0),
        # January (jammer ON — original validation)
        ("2026-01-08", 14.5, 89, 2),
        ("2026-01-09", 16.6, 137, 31),
        ("2026-01-20", 9.7, 2, 0),
        ("2026-01-21", 17.0, 192, 18),
        # Feb 28 — conflict starts
        ("2026-02-28", 34.5, 584, 31),
        # March escalation
        ("2026-03-01", 51.2, 908, 84),
        ("2026-03-02", 51.6, 856, 47),
        ("2026-03-03", 62.7, 1085, 18),
        ("2026-03-07", 66.3, 1735, 104),
        ("2026-03-15", 68.9, 2024, 117),
        ("2026-03-20", 83.7, 1138, 81),
        ("2026-03-21", 73.7, 1177, 50),
        ("2026-03-25", 67.5, 1533, 108),
        ("2026-03-30", 77.9, 1651, 102),
        ("2026-03-31", 74.8, 1455, 11),
        # April — still active
        ("2026-04-04", 66.5, 1261, 148),
        ("2026-04-06", 79.1, 725, 0),
    ]

    dates = [datetime.strptime(d[0], "%Y-%m-%d") for d in timeline]
    elevations = [d[1] for d in timeline]
    detections = [d[2] for d in timeline]

    # Color by phase
    phase_colors = []
    for d in timeline:
        dt = datetime.strptime(d[0], "%Y-%m-%d")
        if dt < datetime(2026, 1, 1):
            phase_colors.append(FUSED_COLOR)  # baseline
        elif dt < datetime(2026, 2, 28):
            phase_colors.append(CYGNSS_COLOR)  # pre-conflict ON
        else:
            phase_colors.append("#e03131")  # conflict period

    text_fx = [pe.withStroke(linewidth=3, foreground=BG_COLOR)]

    # ── Top: Noise elevation ─────────────────────────────────────────────
    ax_elev.fill_between(dates, elevations, alpha=0.15, color="#e03131")
    ax_elev.plot(dates, elevations, color="#e03131", linewidth=2, zorder=3)
    ax_elev.scatter(dates, elevations, c=phase_colors, s=60, edgecolors="white",
                   linewidths=1, zorder=5)

    # Baseline reference line
    ax_elev.axhline(y=0, color=FUSED_COLOR, linewidth=1.5, linestyle="--",
                   alpha=0.7, label="Baseline (jammer OFF)")

    # Conflict start line
    conflict_date = datetime(2026, 2, 28)
    ax_elev.axvline(x=conflict_date, color=GT_COLOR, linewidth=2, linestyle=":",
                   alpha=0.9, zorder=4)
    ax_elev.text(conflict_date, max(elevations) * 0.95, "  Conflict\n  begins",
                fontsize=10, fontweight="bold", color=GT_COLOR,
                path_effects=text_fx, va="top")

    # January ON label
    jan_date = datetime(2026, 1, 14)
    ax_elev.annotate("January\nvalidation\nperiod", xy=(jan_date, 17),
                    fontsize=9, color=CYGNSS_COLOR, ha="center",
                    path_effects=text_fx)

    # Escalation annotation
    mar_date = datetime(2026, 3, 10)
    ax_elev.annotate("Signal intensity\n~5x January levels",
                    xy=(datetime(2026, 3, 15), 68.9),
                    xytext=(datetime(2026, 3, 8), 45),
                    fontsize=10, fontweight="bold", color="#e03131",
                    path_effects=text_fx,
                    arrowprops=dict(arrowstyle="->", color="#e03131", lw=2))

    ax_elev.set_ylabel("Noise Floor Elevation (%)", fontsize=12)
    ax_elev.set_ylim(-5, 95)
    ax_elev.grid(True, alpha=0.3)
    ax_elev.set_title("GPS Jammer Persistence: Shiraz, Iran — December 2025 to April 2026",
                     fontsize=14, fontweight="bold")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FUSED_COLOR,
               markeredgecolor="white", markersize=8, label="Baseline (OFF)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CYGNSS_COLOR,
               markeredgecolor="white", markersize=8, label="January (ON)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#e03131",
               markeredgecolor="white", markersize=8, label="Conflict period (ON)"),
        Line2D([0], [0], color=GT_COLOR, linewidth=2, linestyle=":",
               label="Conflict start (Feb 28)"),
    ]
    leg = ax_elev.legend(handles=legend_elements, loc="upper left", fontsize=9)
    leg.get_frame().set_facecolor(BG_LIGHT)
    leg.get_frame().set_edgecolor(GRID_COLOR)

    # ── Bottom: Detection count ──────────────────────────────────────────
    ax_det.bar(dates, detections, width=1.5, color=phase_colors, alpha=0.8,
              edgecolor="none")
    ax_det.axvline(x=conflict_date, color=GT_COLOR, linewidth=2, linestyle=":",
                  alpha=0.9, zorder=4)
    ax_det.set_ylabel("CYGNSS Detections", fontsize=12)
    ax_det.set_xlabel("Date (2025–2026)", fontsize=12)
    ax_det.grid(True, alpha=0.3, axis="y")

    # Rotate x labels
    fig.autofmt_xdate(rotation=45)

    fig.text(0.5, -0.02,
             "CYGNSS ddm_noise_floor within 200 km of 27.32°N, 52.87°E  |  "
             "Baseline: Dec 15 & 27, 2025 (mean=9,858)  |  "
             "Threshold: baseline + 2.5σ",
             ha="center", fontsize=9, color=TEXT_DIM, style="italic")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "fig8_jammer_timeline.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

FIGURES = {
    1: ("Hero: Converging on the Jammer", figure1_hero),
    2: ("ON vs OFF: The Smoking Gun", figure2_on_vs_off),
    3: ("CYGNSS 1/r² Inverse-Distance Model", figure3_inverse_distance),
    4: ("NISAR Streak Bearing Intersection", figure4_nisar_streaks),
    5: ("Three-Column Method Comparison", figure5_method_comparison),
    6: ("Bayesian Fusion Mechanics", figure6_bayesian_fusion),
    7: ("Results Dashboard", figure7_dashboard),
    8: ("Jammer Persistence Timeline", figure8_timeline),
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
