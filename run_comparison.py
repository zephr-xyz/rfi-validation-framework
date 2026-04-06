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
    localize, localize_nisar_triangulated,
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

    log.info("CYGNSS complete in %.1f seconds", elapsed)
    log.info("  Jammer-ON detections: %d", len(detections))
    log.info("  Baseline detections:  %d", len(baseline))

    if detections:
        for d in detections[:5]:
            log.info("    (%.4f, %.4f) kurtosis=%.2f dist=%.1fkm",
                     d.lat, d.lon, d.intensity, d.metadata.get("distance_km", 0))

    return detections, baseline


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
    cygnss_detections, cygnss_baseline = run_cygnss()
    nisar_detections = run_nisar()

    # Localize
    log.info("=" * 60)
    log.info("LOCALIZATION RESULTS")
    log.info("=" * 60)

    cygnss_result = localize(cygnss_detections, "CYGNSS")
    nisar_result = localize_nisar_triangulated(nisar_detections)
    cygnss_baseline_result = localize(cygnss_baseline, "CYGNSS_baseline")

    for label, r in [("CYGNSS (jammer ON)", cygnss_result),
                     ("CYGNSS (baseline)", cygnss_baseline_result),
                     ("NISAR", nisar_result)]:
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
        "cygnss_baseline": asdict(cygnss_baseline_result),
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
    plot_results = {"cygnss": cygnss_result, "nisar": nisar_result}
    plot_comparison(plot_results, GROUND_TRUTH, OUTPUT_DIR)


if __name__ == "__main__":
    main()
