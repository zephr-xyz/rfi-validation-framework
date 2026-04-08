#!/usr/bin/env python3
"""
Head-to-head CYGNSS vs NISAR comparison for Jan 8 and Jan 20 2026.

CYGNSS: Streamed from S3 (no local disk).
NISAR:  Downloaded to /mnt/rfi-data/nisar/ then processed.
"""

import json
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from rfi_validation import (
    GROUND_TRUTH, NISAR_KNOWN_PASSES, OUTPUT_DIR,
    localize, localize_nisar_triangulated, localize_fused,
    geodesic_distance_km, LocalizationResult,
)
from dataclasses import asdict

OUTPUT_DIR = Path("/mnt/rfi-data/output")
NISAR_DIR = Path("/mnt/rfi-data/nisar")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NISAR_DIR.mkdir(parents=True, exist_ok=True)

JAMMER_ON_DATES = ["2026-01-08", "2026-01-20"]
BASELINE_DATES = ["2025-12-15", "2025-12-27"]


def run_cygnss():
    """Stream CYGNSS L1 and detect RFI on jammer-ON + baseline dates."""
    from cygnss_module import detect_cygnss_rfi_streaming

    log.info("=" * 60)
    log.info("CYGNSS: Streaming L1 DDMs from S3")
    log.info("=" * 60)

    t0 = time.time()
    result = detect_cygnss_rfi_streaming(
        gt_lat=GROUND_TRUTH["lat"],
        gt_lon=GROUND_TRUTH["lon"],
        dates=JAMMER_ON_DATES,
        baseline_dates=BASELINE_DATES,
        search_radius_km=200,
    )
    elapsed = time.time() - t0

    detections = result["detections"]
    baseline = result["baseline"]
    inv_dist_fit = result.get("inv_dist_fit")

    log.info("CYGNSS complete in %.1f seconds", elapsed)
    log.info("  Jammer-ON detections: %d", len(detections))
    log.info("  Baseline detections:  %d", len(baseline))
    if inv_dist_fit:
        log.info("  1/r² fit: (%.4f, %.4f) error=%.1f km",
                 inv_dist_fit["estimated_lat"], inv_dist_fit["estimated_lon"],
                 inv_dist_fit["error_km"])

    if detections:
        for d in detections[:5]:
            log.info("    (%.4f, %.4f) intensity=%.2f dist=%.1fkm method=%s",
                     d.lat, d.lon, d.intensity, d.metadata.get("distance_km", 0),
                     d.metadata.get("method", "?"))

    return detections, baseline, inv_dist_fit


def run_nisar():
    """Download and process NISAR GCOV for jammer-ON dates."""
    from nisar_module import download_nisar_known_passes, detect_nisar_rfi

    log.info("=" * 60)
    log.info("NISAR: Downloading GCOV products")
    log.info("=" * 60)

    # Download all known passes (ON + OFF)
    t0 = time.time()
    files = download_nisar_known_passes(NISAR_KNOWN_PASSES, NISAR_DIR)
    dl_elapsed = time.time() - t0
    log.info("NISAR download complete in %.1f seconds (%d files)", dl_elapsed, len(files))

    # Process
    log.info("NISAR: Running eigenvalue RFI detection...")
    t0 = time.time()
    detections = detect_nisar_rfi(NISAR_DIR, GROUND_TRUTH)
    proc_elapsed = time.time() - t0
    log.info("NISAR processing complete in %.1f seconds", proc_elapsed)
    log.info("  Detections: %d", len(detections))

    if detections:
        for d in detections[:5]:
            log.info("    (%.4f, %.4f) intensity=%.2f orbit=%s dist=%.1fkm",
                     d.lat, d.lon, d.intensity, d.orbit_direction,
                     d.metadata.get("distance_km", 0))

    return detections


def main():
    log.info("GPS Jammer Localization Comparison")
    log.info("Ground Truth: %.4f°N, %.4f°E", GROUND_TRUTH["lat"], GROUND_TRUTH["lon"])
    log.info("Jammer-ON dates: %s", JAMMER_ON_DATES)
    log.info("Baseline dates:  %s", BASELINE_DATES)
    log.info("")

    # Run both modalities
    cygnss_detections, cygnss_baseline, cygnss_inv_dist = run_cygnss()
    nisar_detections = run_nisar()

    # Localize
    log.info("=" * 60)
    log.info("LOCALIZATION RESULTS")
    log.info("=" * 60)

    # CYGNSS: use 1/r² fit if available (much better than simple centroid)
    cygnss_result = localize(cygnss_detections, "CYGNSS")
    if cygnss_inv_dist and cygnss_detections:
        inv_error = cygnss_inv_dist["error_km"]
        centroid_error = cygnss_result.euclidean_error_km
        log.info("CYGNSS localization: centroid=%.1f km, 1/r² fit=%.1f km",
                 centroid_error, inv_error)
        # Override with 1/r² fit if better
        if inv_error < centroid_error:
            log.info("  → Using 1/r² fit (%.1f km better)", centroid_error - inv_error)
            cygnss_result.estimated_lat = cygnss_inv_dist["estimated_lat"]
            cygnss_result.estimated_lon = cygnss_inv_dist["estimated_lon"]
            cygnss_result.euclidean_error_km = inv_error
            # Use bootstrap CEP if available (fit uncertainty, not raw scatter)
            boot_cep = cygnss_inv_dist.get("bootstrap_cep_km")
            if boot_cep is not None:
                log.info("  Bootstrap CEP: %.2f km (n=%d fits)",
                         boot_cep, cygnss_inv_dist.get("bootstrap_n_fits", 0))
                cygnss_result.cep_km = boot_cep
            else:
                # Fallback: raw scatter CEP
                from rfi_validation import circular_error_probable
                det_lats = [d.lat for d in cygnss_detections]
                det_lons = [d.lon for d in cygnss_detections]
                cygnss_result.cep_km = circular_error_probable(
                    det_lats, det_lons,
                    cygnss_result.estimated_lat, cygnss_result.estimated_lon)

    nisar_result = localize_nisar_triangulated(nisar_detections)
    cygnss_baseline_result = localize(cygnss_baseline, "CYGNSS_baseline")

    # Fused localization
    log.info("")
    log.info("=" * 60)
    log.info("FUSED LOCALIZATION (CYGNSS + NISAR)")
    log.info("=" * 60)
    fused_result = localize_fused(
        cygnss_result, nisar_result,
        cygnss_detections, nisar_detections,
        cygnss_inv_dist)

    for label, r in [("CYGNSS (jammer ON)", cygnss_result),
                     ("CYGNSS (baseline)", cygnss_baseline_result),
                     ("NISAR", nisar_result),
                     ("FUSED", fused_result)]:
        log.info("")
        log.info("--- %s ---", label)
        log.info("  Detections: %d", r.num_detections)
        if r.num_detections > 0:
            log.info("  Estimated: %.4f°N, %.4f°E", r.estimated_lat, r.estimated_lon)
            log.info("  Error:     %.2f km", r.euclidean_error_km)
            log.info("  CEP(50%%):  %.2f km", r.cep_km)

    # Save results
    results = {
        "cygnss": asdict(cygnss_result),
        "nisar": asdict(nisar_result),
        "fused": asdict(fused_result),
        "cygnss_baseline": asdict(cygnss_baseline_result),
        "cygnss_inv_dist_fit": cygnss_inv_dist,
        "ground_truth": GROUND_TRUTH,
        "jammer_on_dates": JAMMER_ON_DATES,
        "baseline_dates": BASELINE_DATES,
    }
    results_path = OUTPUT_DIR / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("\nResults saved to %s", results_path)

    # Generate visualization
    log.info("\nGenerating comparison plot...")
    from visualize_module import plot_comparison
    plot_results = {"cygnss": cygnss_result, "nisar": nisar_result, "fused": fused_result}
    plot_comparison(plot_results, GROUND_TRUTH, OUTPUT_DIR)


if __name__ == "__main__":
    main()
