#!/usr/bin/env python3
"""Run NISAR-only processing on already-downloaded GCOV files."""

import json
import logging
import time
from pathlib import Path
from dataclasses import asdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

import sys
sys.path.insert(0, "/mnt/rfi-data")

from rfi_validation import GROUND_TRUTH, localize, localize_nisar_triangulated, LocalizationResult
from nisar_module import detect_nisar_rfi

OUTPUT_DIR = Path("/mnt/rfi-data/output")
NISAR_DIR = Path("/mnt/rfi-data/nisar")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    log.info("NISAR RFI Detection — Cropped + Optimized")
    log.info("Ground Truth: %.4f°N, %.4f°E", GROUND_TRUTH["lat"], GROUND_TRUTH["lon"])
    log.info("NISAR data dir: %s", NISAR_DIR)
    log.info("Files: %d", len(list(NISAR_DIR.glob("*.h5"))))

    t0 = time.time()
    detections = detect_nisar_rfi(NISAR_DIR, GROUND_TRUTH)
    elapsed = time.time() - t0

    log.info("")
    log.info("=" * 60)
    log.info("NISAR RESULTS (%.1f seconds)", elapsed)
    log.info("=" * 60)
    log.info("Total detections: %d", len(detections))

    for d in detections:
        log.info("  (%.4f°N, %.4f°E) intensity=%.2f orbit=%s dist=%.1fkm file=%s",
                 d.lat, d.lon, d.intensity, d.orbit_direction,
                 d.metadata.get("distance_km", 0), d.timestamp[:60])

    result = localize_nisar_triangulated(detections)
    log.info("")
    log.info("LOCALIZATION:")
    log.info("  Estimated: %.4f°N, %.4f°E", result.estimated_lat, result.estimated_lon)
    log.info("  Error:     %.2f km", result.euclidean_error_km)
    log.info("  CEP(50%%):  %.2f km", result.cep_km)

    # Save
    out = asdict(result)
    out["ground_truth"] = GROUND_TRUTH
    out["elapsed_s"] = round(elapsed, 1)
    results_path = OUTPUT_DIR / "nisar_results.json"
    results_path.write_text(json.dumps(out, indent=2, default=str))
    log.info("Saved to %s", results_path)

if __name__ == "__main__":
    main()
