#!/usr/bin/env python3
"""Generate publication-quality map of Hormuz jammer analysis."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from pathlib import Path

# Dark theme
BG_COLOR = "#0d1117"
BG_LIGHT = "#161b22"
TEXT_COLOR = "#e6edf3"
TEXT_DIM = "#8b949e"
GRID_COLOR = "#21262d"
WATER_COLOR = "#0a1628"
LAND_COLOR = "#1a1e24"

JAMMER_THREAT = "#e03131"
JAMMER_OTHER = "#f59f00"
LANE_COLOR = "#339af0"
LANE_DENIED = "#e03131"
DENIAL_ZONE = "#e0313120"

OUTPUT_DIR = Path("output/hormuz_scan")


def setup_dark_style():
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_DIM,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_DIM,
        "ytick.color": TEXT_DIM,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.3,
    })


def main():
    data = json.load(open(OUTPUT_DIR / "hormuz_jammers.json"))
    jammers = data["jammers"]
    lane = [(p["lat"], p["lon"]) for p in data["shipping_lane"]]

    setup_dark_style()
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    # Approximate coastlines using a simple polygon approach
    # Just shade the background as "water" and we'll mark jammers on land
    ax.set_facecolor(WATER_COLOR)

    # Draw shipping lane
    lane_lats = [p[0] for p in lane]
    lane_lons = [p[1] for p in lane]
    ax.plot(lane_lons, lane_lats, color=LANE_COLOR, linewidth=2.5,
            alpha=0.8, zorder=5, linestyle="-")

    # Mark denied vs safe lane segments
    denied_points = set()
    for j in jammers:
        for imp in j.get("shipping_lane_impacts", []):
            denied_points.add((imp["lane_lat"], imp["lane_lon"]))

    for lat, lon in lane:
        if (lat, lon) in denied_points:
            ax.plot(lon, lat, "o", color=LANE_DENIED, markersize=6,
                    zorder=7, markeredgecolor="white", markeredgewidth=0.5)
        else:
            ax.plot(lon, lat, "o", color=LANE_COLOR, markersize=4,
                    zorder=7, markeredgecolor="white", markeredgewidth=0.5)

    # Draw jammer denial zones and positions
    for j in jammers:
        lat = j["estimated_lat"]
        lon = j["estimated_lon"]
        amp = j["amplitude"]
        denial_km = j.get("denial_range_km", 0)
        threats = j.get("impacts_shipping", False)

        color = JAMMER_THREAT if threats else JAMMER_OTHER
        fill_alpha = 0.12 if threats else 0.06

        # Denial radius circle
        if denial_km > 0:
            denial_deg = denial_km / 111.0
            circle = Circle((lon, lat), denial_deg,
                             fill=True, facecolor=color,
                             alpha=fill_alpha,
                             edgecolor=color, linewidth=1.0,
                             linestyle="--", zorder=2)
            ax.add_patch(circle)

        # Jammer marker
        marker_size = max(8, min(20, amp / 1500))
        ax.plot(lon, lat, "^", color=color, markersize=marker_size,
                markeredgecolor="white", markeredgewidth=1.2, zorder=8)

        # CEP circle if available
        cep = j.get("bootstrap_cep_km")
        if cep and cep > 0:
            cep_deg = cep / 111.0
            cep_circle = Circle((lon, lat), cep_deg,
                                fill=False, edgecolor=color,
                                linewidth=1.5, linestyle="-", alpha=0.7,
                                zorder=3)
            ax.add_patch(cep_circle)

        # Label threatening jammers
        if threats:
            label = f"amp={amp:.0f}\n{denial_km:.0f}km range"
            ax.annotate(label, (lon, lat),
                        textcoords="offset points", xytext=(12, 8),
                        fontsize=8, color=color, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor=BG_LIGHT, edgecolor=color,
                                  alpha=0.9, linewidth=1))

    # Geographic labels
    labels = [
        (26.3, 56.3, "Strait of\nHormuz", 10),
        (27.0, 52.5, "Persian Gulf", 11),
        (25.5, 57.5, "Gulf of\nOman", 10),
        (29.0, 49.0, "Iraq/Kuwait", 9),
        (28.0, 55.5, "IRAN", 14),
        (25.5, 54.0, "UAE", 11),
        (25.0, 50.5, "Qatar\n& Bahrain", 9),
        (29.5, 51.0, "IRAN", 11),
    ]
    for lat, lon, text, size in labels:
        ax.text(lon, lat, text, fontsize=size, color=TEXT_DIM,
                ha="center", va="center", style="italic", alpha=0.6)

    # Axis setup
    ax.set_xlim(48.0, 59.0)
    ax.set_ylim(24.5, 30.0)
    ax.set_aspect(1.0 / np.cos(np.radians(27.0)))
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)

    # Title
    n_threats = sum(1 for j in jammers if j.get("impacts_shipping"))
    n_denied = len(denied_points)
    n_total_lane = len(lane)
    fig.suptitle("GPS Jammer Threat to Persian Gulf & Strait of Hormuz Shipping",
                 fontsize=16, fontweight="bold", y=0.97, color=TEXT_COLOR)
    ax.set_title(
        f"{len(jammers)} jammers detected  |  {n_threats} threaten shipping  |  "
        f"{n_denied}/{n_total_lane} lane waypoints denied GPS",
        fontsize=11, color=TEXT_DIM, pad=8)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor=JAMMER_THREAT,
               markeredgecolor="white", markersize=10,
               label="Jammer (threatens shipping)"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=JAMMER_OTHER,
               markeredgecolor="white", markersize=10,
               label="Jammer (inland)"),
        Line2D([0], [0], color=LANE_COLOR, linewidth=2.5,
               label="Shipping lane"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LANE_DENIED,
               markeredgecolor="white", markersize=6,
               label="Lane point — GPS denied"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LANE_COLOR,
               markeredgecolor="white", markersize=4,
               label="Lane point — GPS OK"),
        Line2D([0], [0], color=JAMMER_THREAT, linewidth=1.0, linestyle="--",
               label="Denial radius (1/r²)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=9, facecolor=BG_LIGHT, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR)

    # Stats box
    stats_text = (
        f"Detection: CYGNSS noise floor elevation\n"
        f"Localization: 1/r² inverse-distance fit\n"
        f"Denial model: calibrated on Bushehr GT\n"
        f"  (amp/r² > 3.0 → GPS denied)\n"
        f"Baseline: {data['baseline_date']}"
    )
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=8, color=TEXT_DIM, va="bottom",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_LIGHT,
                      edgecolor=GRID_COLOR, alpha=0.9))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "hormuz_jammer_map.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
