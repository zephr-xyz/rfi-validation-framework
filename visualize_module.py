"""
Visualization Module
====================
Generates matplotlib comparison plots for CYGNSS vs NISAR GPS jammer
localization accuracy.
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

log = logging.getLogger(__name__)


def plot_comparison(results, ground_truth, output_dir):
    """Generate the full comparison visualization.

    Creates a 2×2 figure:
      - Top-left: Map with ground truth + estimated positions + CEP circles
      - Top-right: Detection heatmap for CYGNSS (specular points colored by kurtosis)
      - Bottom-left: Detection heatmap for NISAR (streak centroids colored by λ₁/λ₂)
      - Bottom-right: Accuracy comparison bar chart (CEP + Euclidean error)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_lat = ground_truth["lat"]
    gt_lon = ground_truth["lon"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        "GPS Jammer Localization: CYGNSS (GNSS-R) vs NISAR (L-band SAR)\n"
        f"Ground Truth: {gt_lat:.4f}°N, {gt_lon:.4f}°E",
        fontsize=14, fontweight="bold", y=0.98,
    )

    cygnss = results.get("cygnss")
    nisar = results.get("nisar")
    fused = results.get("fused")

    # ── Top-left: Overview map ───────────────────────────────────────────
    ax = axes[0, 0]
    ax.set_title("Localization Overview", fontsize=12, fontweight="bold")

    # Ground truth
    ax.plot(gt_lon, gt_lat, marker="*", color="gold", markersize=20,
            markeredgecolor="black", markeredgewidth=1.5, zorder=10,
            label="Ground Truth")

    # CYGNSS estimate
    if cygnss and cygnss.num_detections > 0:
        ax.plot(cygnss.estimated_lon, cygnss.estimated_lat,
                marker="^", color="#1f77b4", markersize=14,
                markeredgecolor="black", markeredgewidth=1, zorder=9,
                label=f"CYGNSS ({cygnss.euclidean_error_km:.1f} km)")
        cep_deg = cygnss.cep_km / 111.0
        circle = Circle((cygnss.estimated_lon, cygnss.estimated_lat),
                         cep_deg, fill=False, edgecolor="#1f77b4",
                         linestyle="--", linewidth=1.5, alpha=0.6)
        ax.add_patch(circle)

    # NISAR estimate
    if nisar and nisar.num_detections > 0:
        ax.plot(nisar.estimated_lon, nisar.estimated_lat,
                marker="v", color="#d62728", markersize=14,
                markeredgecolor="black", markeredgewidth=1, zorder=9,
                label=f"NISAR ({nisar.euclidean_error_km:.1f} km)")
        cep_deg = nisar.cep_km / 111.0
        circle = Circle((nisar.estimated_lon, nisar.estimated_lat),
                         cep_deg, fill=False, edgecolor="#d62728",
                         linestyle="--", linewidth=1.5, alpha=0.6)
        ax.add_patch(circle)

    # Fused estimate
    if fused and fused.num_detections > 0:
        ax.plot(fused.estimated_lon, fused.estimated_lat,
                marker="D", color="#2ca02c", markersize=14,
                markeredgecolor="black", markeredgewidth=1.5, zorder=11,
                label=f"Fused ({fused.euclidean_error_km:.1f} km)")

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Auto-scale to show all points with padding
    all_lons = [gt_lon]
    all_lats = [gt_lat]
    for r in [cygnss, nisar, fused]:
        if r and r.num_detections > 0:
            all_lons.append(r.estimated_lon)
            all_lats.append(r.estimated_lat)

    pad = 0.5
    ax.set_xlim(min(all_lons) - pad, max(all_lons) + pad)
    ax.set_ylim(min(all_lats) - pad, max(all_lats) + pad)

    # ── Top-right: CYGNSS detection heatmap ──────────────────────────────
    ax = axes[0, 1]
    ax.set_title("CYGNSS Detections (Forbidden-Zone Kurtosis)", fontsize=12, fontweight="bold")

    if cygnss and cygnss.detections:
        det_lats = [d["lat"] for d in cygnss.detections]
        det_lons = [d["lon"] for d in cygnss.detections]
        det_kurt = [d["intensity"] for d in cygnss.detections]

        sc = ax.scatter(det_lons, det_lats, c=det_kurt, cmap="YlOrRd",
                        s=40, alpha=0.7, edgecolors="black", linewidths=0.3,
                        vmin=KURTOSIS_VMIN, vmax=KURTOSIS_VMAX)
        plt.colorbar(sc, ax=ax, label="Excess Kurtosis", shrink=0.8)
    else:
        ax.text(0.5, 0.5, "No CYGNSS detections", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")

    ax.plot(gt_lon, gt_lat, marker="*", color="gold", markersize=16,
            markeredgecolor="black", markeredgewidth=1.5, zorder=10)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(True, alpha=0.3)

    # ── Bottom-left: NISAR detection heatmap ─────────────────────────────
    ax = axes[1, 0]
    ax.set_title("NISAR Detections (λ₁ Eigenvalue Streaks)", fontsize=12, fontweight="bold")

    if nisar and nisar.detections:
        det_lats = [d["lat"] for d in nisar.detections]
        det_lons = [d["lon"] for d in nisar.detections]
        det_int = [d["intensity"] for d in nisar.detections]
        det_orb = [d.get("orbit_direction", "?") for d in nisar.detections]

        # Color by orbit direction, size by intensity
        asc_mask = [o == "Ascending" for o in det_orb]
        desc_mask = [o == "Descending" for o in det_orb]

        for mask, color, label in [(asc_mask, "#ff7f0e", "Ascending"),
                                    (desc_mask, "#9467bd", "Descending")]:
            if any(mask):
                mlats = [la for la, m in zip(det_lats, mask) if m]
                mlons = [lo for lo, m in zip(det_lons, mask) if m]
                mints = [it for it, m in zip(det_int, mask) if m]
                sizes = [max(20, min(200, i * 10)) for i in mints]
                ax.scatter(mlons, mlats, c=color, s=sizes, alpha=0.7,
                          edgecolors="black", linewidths=0.3, label=label)

        ax.legend(loc="upper left", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No NISAR detections", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")

    ax.plot(gt_lon, gt_lat, marker="*", color="gold", markersize=16,
            markeredgecolor="black", markeredgewidth=1.5, zorder=10)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(True, alpha=0.3)

    # ── Bottom-right: Accuracy comparison ────────────────────────────────
    ax = axes[1, 1]
    ax.set_title("Localization Accuracy Comparison", fontsize=12, fontweight="bold")

    modalities = []
    euclidean_errors = []
    cep_values = []
    colors = []

    if cygnss and cygnss.num_detections > 0:
        modalities.append("CYGNSS\n(GNSS-R)")
        euclidean_errors.append(cygnss.euclidean_error_km)
        cep_values.append(cygnss.cep_km)
        colors.append("#1f77b4")

    if nisar and nisar.num_detections > 0:
        modalities.append("NISAR\n(L-band SAR)")
        euclidean_errors.append(nisar.euclidean_error_km)
        cep_values.append(nisar.cep_km)
        colors.append("#d62728")

    if fused and fused.num_detections > 0:
        modalities.append("Fused\n(CYGNSS+NISAR)")
        euclidean_errors.append(fused.euclidean_error_km)
        cep_values.append(fused.cep_km)
        colors.append("#2ca02c")

    if modalities:
        x = np.arange(len(modalities))
        width = 0.35

        bars1 = ax.bar(x - width / 2, euclidean_errors, width, label="Euclidean Error",
                        color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        bars2 = ax.bar(x + width / 2, cep_values, width, label="CEP (50%)",
                        color=colors, alpha=0.4, edgecolor="black", linewidth=0.5,
                        hatch="//")

        # Value labels
        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        f"{h:.1f} km", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(modalities)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No detections from either modality",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="gray")

    ax.set_ylabel("Distance (km)")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Summary text ─────────────────────────────────────────────────────
    summary_lines = [f"Ground Truth: {gt_lat:.4f}°N, {gt_lon:.4f}°E"]
    for label, r in [("CYGNSS", cygnss), ("NISAR", nisar), ("Fused", fused)]:
        if r and r.num_detections > 0:
            summary_lines.append(
                f"{label}: {r.num_detections} det, "
                f"Error={r.euclidean_error_km:.1f} km, CEP={r.cep_km:.1f} km"
            )
    fig.text(0.5, 0.01, " | ".join(summary_lines),
             ha="center", fontsize=10, style="italic", color="#555")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = output_dir / "rfi_localization_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved comparison plot to %s", out_path)

    # Also save a text summary
    _save_text_summary(results, ground_truth, output_dir)


# Colorbar ranges for kurtosis
KURTOSIS_VMIN = 4.0
KURTOSIS_VMAX = 20.0


def _save_text_summary(results, ground_truth, output_dir):
    """Save a plain-text summary of results."""
    gt_lat = ground_truth["lat"]
    gt_lon = ground_truth["lon"]

    lines = [
        "=" * 60,
        "GPS JAMMER LOCALIZATION VALIDATION RESULTS",
        "=" * 60,
        f"Ground Truth:  {gt_lat:.4f}°N, {gt_lon:.4f}°E",
        f"Assumed Jammer: Civilian GPS jammer (L1/L2, power unknown)",
        "",
    ]

    labels_map = {
        "cygnss": "CYGNSS (GNSS-R 1/r² Fit)",
        "nisar": "NISAR (L-band λ₁ EVD)",
        "fused": "FUSED (CYGNSS + NISAR)",
    }
    for key in ("cygnss", "nisar", "fused"):
        r = results.get(key)
        if r is None:
            continue
        label = labels_map.get(key, key)
        lines.append(f"--- {label} ---")
        lines.append(f"  Detections:       {r.num_detections}")
        if r.num_detections > 0:
            lines.append(f"  Estimated Pos:    {r.estimated_lat:.4f}°N, {r.estimated_lon:.4f}°E")
            lines.append(f"  Euclidean Error:  {r.euclidean_error_km:.2f} km")
            lines.append(f"  CEP (50%):        {r.cep_km:.2f} km")
        else:
            lines.append("  No detections — cannot localize")
        lines.append("")

    # Winner — compare all modalities
    all_results = [(k, results.get(k)) for k in ("cygnss", "nisar", "fused")]
    valid_results = [(k, r) for k, r in all_results if r and r.num_detections > 0]
    if len(valid_results) >= 2:
        valid_results.sort(key=lambda x: x[1].euclidean_error_km)
        winner_key, winner = valid_results[0]
        runner_key, runner = valid_results[1]
        lines.append(f"WINNER: {winner_key.upper()} ({winner.euclidean_error_km:.2f} km) "
                     f"by {runner.euclidean_error_km - winner.euclidean_error_km:.1f} km over {runner_key.upper()}")

    lines.append("=" * 60)
    text = "\n".join(lines)

    out_path = output_dir / "localization_summary.txt"
    out_path.write_text(text)
    log.info("Saved text summary to %s", out_path)
    print(text)
