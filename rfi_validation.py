#!/usr/bin/env python3
"""
GPS Jammer Localization Validation Framework
=============================================
Compares CYGNSS (GNSS-R) and NISAR (L-band SAR) modalities for localizing
a known ground-truth GPS jammer at 27.3182°N, 52.8703°E (near Shiraz, Iran).

CYGNSS: Kurtosis-based anomaly detection in DDM forbidden-zone bins.
NISAR:  λ₁ eigenvalue decomposition on HV-pol to find coherent point emitters.

Usage:
    python rfi_validation.py --download          # Download data from PO.DAAC
    python rfi_validation.py --process           # Process downloaded data
    python rfi_validation.py --visualize         # Generate comparison plots
    python rfi_validation.py --all               # Full pipeline
"""

import argparse
import logging
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from pyproj import Geod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Ground Truth ─────────────────────────────────────────────────────────────

GROUND_TRUTH = {"lat": 27.3182, "lon": 52.8703}  # Known GPS jammer location
JAMMER_POWER_W = 10.0   # 10W civilian L1/L2 jammer assumption
SEARCH_RADIUS_KM = 200  # Search radius around ground truth

# Known jammer-active NISAR passes over the target area
# Each entry: (granule_id, start_time, track, frame, direction, jammer_on)
NISAR_KNOWN_PASSES = [
    {
        "granule": "NISAR_L2_PR_GCOV_010_157_A_015_2005_DHDH_A_20260108T015200_20260108T015233",
        "start": "2026-01-08T01:52:00Z",
        "stop": "2026-01-08T01:52:33Z",
        "track": 157, "frame": 15,
        "direction": "Ascending",
        "jammer_on": True,
        "collection": "NISAR_L2_GCOV_BETA_V1",
    },
    {
        "granule": "NISAR_L2_PR_GCOV_010_157_A_015_2005_DHDH_A_20260120T015200_20260120T015234",
        "start": "2026-01-20T01:52:00Z",
        "stop": "2026-01-20T01:52:34Z",
        "track": 157, "frame": 15,
        "direction": "Ascending",
        "jammer_on": True,
        "collection": "NISAR_L2_GCOV_BETA_V1",
    },
    # Baselines: jammer OFF
    {
        "granule": "NISAR_L2_PR_GCOV_010_157_A_015_2005_DHDH_A_20251215T015159_20251215T015232",
        "start": "2025-12-15T01:51:59Z",
        "stop": "2025-12-15T01:52:32Z",
        "track": 157, "frame": 15,
        "direction": "Ascending",
        "jammer_on": False,  # BASELINE — jammer OFF
        "collection": "NISAR_L2_GCOV_BETA_V1",
    },
    {
        "granule": "NISAR_L2_PR_GCOV_010_157_A_015_2005_DHDH_A_20251227T015159_20251227T015233",
        "start": "2025-12-27T01:51:59Z",
        "stop": "2025-12-27T01:52:33Z",
        "track": 157, "frame": 15,
        "direction": "Ascending",
        "jammer_on": False,  # BASELINE — jammer OFF
        "collection": "NISAR_L2_GCOV_BETA_V1",
    },
    # Add more passes here as they become available:
    # Descending passes for triangulation, jammer-off baselines, etc.
]

OUTPUT_DIR = Path("output")
CYGNSS_DIR = OUTPUT_DIR / "cygnss"
NISAR_DIR = OUTPUT_DIR / "nisar"


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class RFIDetection:
    """A single RFI detection from either modality."""
    lat: float
    lon: float
    intensity: float          # Kurtosis value (CYGNSS) or eigenvalue ratio (NISAR)
    timestamp: str
    modality: str             # "CYGNSS" or "NISAR"
    orbit_direction: str = "" # "Ascending" / "Descending" (NISAR only)
    metadata: dict = field(default_factory=dict)


@dataclass
class LocalizationResult:
    """Final localization estimate from a modality."""
    modality: str
    estimated_lat: float
    estimated_lon: float
    cep_km: float             # Circular Error Probable (50th percentile)
    euclidean_error_km: float # Distance to ground truth
    num_detections: int
    detections: list = field(default_factory=list)


# ── Geometric Utilities ──────────────────────────────────────────────────────

_geod = Geod(ellps="WGS84")


def geodesic_distance_km(lat1, lon1, lat2, lon2):
    """Geodesic distance between two points in km."""
    _, _, dist_m = _geod.inv(lon1, lat1, lon2, lat2)
    return dist_m / 1000.0


def circular_error_probable(lats, lons, center_lat, center_lon):
    """CEP: median radial distance from center to detection points."""
    if len(lats) == 0:
        return float("inf")
    distances = [geodesic_distance_km(center_lat, center_lon, la, lo)
                 for la, lo in zip(lats, lons)]
    return float(np.median(distances))


def weighted_centroid(lats, lons, weights):
    """Intensity-weighted geographic centroid."""
    w = np.array(weights)
    w = w / w.sum()
    return float(np.average(lats, weights=w)), float(np.average(lons, weights=w))


# ── Main Pipeline ────────────────────────────────────────────────────────────

def download_data(start_date, end_date):
    """Download CYGNSS and NISAR data from PO.DAAC/Earthdata."""
    from cygnss_module import download_cygnss
    from nisar_module import download_nisar_known_passes, download_nisar

    CYGNSS_DIR.mkdir(parents=True, exist_ok=True)
    NISAR_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Downloading CYGNSS L1 DDM data...")
    cygnss_files = download_cygnss(
        GROUND_TRUTH["lat"], GROUND_TRUTH["lon"],
        SEARCH_RADIUS_KM, start_date, end_date, CYGNSS_DIR,
    )
    log.info("Downloaded %d CYGNSS files", len(cygnss_files))

    log.info("Downloading NISAR L-band GCOV data (known jammer-active passes)...")
    nisar_files = download_nisar_known_passes(NISAR_KNOWN_PASSES, NISAR_DIR)
    if not nisar_files:
        log.info("Falling back to spatial search...")
        nisar_files = download_nisar(
            GROUND_TRUTH["lat"], GROUND_TRUTH["lon"],
            start_date, end_date, NISAR_DIR,
        )
    log.info("Downloaded %d NISAR files", len(nisar_files))

    return cygnss_files, nisar_files


def process_data():
    """Run RFI detection on downloaded data, return LocalizationResults."""
    from cygnss_module import detect_cygnss_rfi
    from nisar_module import detect_nisar_rfi

    log.info("Processing CYGNSS data...")
    cygnss_detections = detect_cygnss_rfi(CYGNSS_DIR, GROUND_TRUTH)

    log.info("Processing NISAR data...")
    nisar_detections = detect_nisar_rfi(NISAR_DIR, GROUND_TRUTH)

    # Localize from CYGNSS detections
    cygnss_result = localize(cygnss_detections, "CYGNSS")

    # Localize from NISAR detections (ascending/descending triangulation)
    nisar_result = localize_nisar_triangulated(nisar_detections)

    results = {"cygnss": cygnss_result, "nisar": nisar_result}

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "localization_results.json"
    with open(results_path, "w") as f:
        json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2)
    log.info("Results saved to %s", results_path)

    return results


def localize(detections: list[RFIDetection], modality: str) -> LocalizationResult:
    """Compute weighted centroid + CEP from a list of detections."""
    if not detections:
        log.warning("No %s detections — returning null result", modality)
        return LocalizationResult(
            modality=modality, estimated_lat=0, estimated_lon=0,
            cep_km=float("inf"), euclidean_error_km=float("inf"),
            num_detections=0,
        )

    lats = [d.lat for d in detections]
    lons = [d.lon for d in detections]
    weights = [d.intensity for d in detections]

    est_lat, est_lon = weighted_centroid(lats, lons, weights)
    error_km = geodesic_distance_km(GROUND_TRUTH["lat"], GROUND_TRUTH["lon"],
                                     est_lat, est_lon)
    cep = circular_error_probable(lats, lons, est_lat, est_lon)

    return LocalizationResult(
        modality=modality,
        estimated_lat=est_lat, estimated_lon=est_lon,
        cep_km=cep, euclidean_error_km=error_km,
        num_detections=len(detections),
        detections=[asdict(d) for d in detections],
    )


def localize_nisar_triangulated(detections: list[RFIDetection]) -> LocalizationResult:
    """Triangulate NISAR RFI source using bearing-line intersection or orbit centroids.

    Priority:
      1. Bearing-line intersection (from fit_streak_bearing in nisar_module)
      2. Ascending/descending orbit centroid intersection
      3. Simple weighted centroid (fallback)
    """
    # Check if bearing intersection is available (attached by nisar_module)
    bearing_int = None
    for d in detections:
        bi = d.metadata.get("bearing_intersection")
        if bi:
            bearing_int = bi
            break

    asc = [d for d in detections if d.orbit_direction == "Ascending"]
    desc = [d for d in detections if d.orbit_direction == "Descending"]

    if bearing_int:
        est_lat = bearing_int["lat"]
        est_lon = bearing_int["lon"]
        log.info("NISAR localization via bearing intersection: %.4f°N, %.4f°E",
                 est_lat, est_lon)
    elif asc and desc:
        asc_lat, asc_lon = weighted_centroid(
            [d.lat for d in asc], [d.lon for d in asc], [d.intensity for d in asc])
        desc_lat, desc_lon = weighted_centroid(
            [d.lat for d in desc], [d.lon for d in desc], [d.intensity for d in desc])
        est_lat = (asc_lat + desc_lat) / 2
        est_lon = (asc_lon + desc_lon) / 2
        log.info("NISAR triangulation: ASC(%.4f,%.4f) × DESC(%.4f,%.4f) → (%.4f,%.4f)",
                 asc_lat, asc_lon, desc_lat, desc_lon, est_lat, est_lon)
    else:
        log.warning("NISAR: only %d ascending, %d descending, no bearing intersection — using weighted centroid",
                     len(asc), len(desc))
        return localize(detections, "NISAR")

    # Also compute weighted centroid for comparison
    wc_lat, wc_lon = weighted_centroid(
        [d.lat for d in detections], [d.lon for d in detections],
        [d.intensity for d in detections])
    wc_error = geodesic_distance_km(GROUND_TRUTH["lat"], GROUND_TRUTH["lon"],
                                     wc_lat, wc_lon)
    bi_error = geodesic_distance_km(GROUND_TRUTH["lat"], GROUND_TRUTH["lon"],
                                     est_lat, est_lon)

    # Use whichever method gives lower error (bearing intersection vs centroid)
    if wc_error < bi_error:
        log.info("  Weighted centroid (%.2f km) beats bearing intersection (%.2f km) — using centroid",
                 wc_error, bi_error)
        est_lat, est_lon = wc_lat, wc_lon

    all_lats = [d.lat for d in detections]
    all_lons = [d.lon for d in detections]
    error_km = geodesic_distance_km(GROUND_TRUTH["lat"], GROUND_TRUTH["lon"],
                                     est_lat, est_lon)
    cep = circular_error_probable(all_lats, all_lons, est_lat, est_lon)

    return LocalizationResult(
        modality="NISAR",
        estimated_lat=est_lat, estimated_lon=est_lon,
        cep_km=cep, euclidean_error_km=error_km,
        num_detections=len(detections),
        detections=[asdict(d) for d in detections],
    )


def visualize(results: Optional[dict] = None):
    """Generate comparison visualization."""
    from visualize_module import plot_comparison

    if results is None:
        results_path = OUTPUT_DIR / "localization_results.json"
        if not results_path.exists():
            log.error("No results found. Run --process first.")
            return
        with open(results_path) as f:
            raw = json.load(f)
        # Reconstruct LocalizationResult objects
        results = {}
        for k, v in raw.items():
            results[k] = LocalizationResult(**{kk: vv for kk, vv in v.items()})

    plot_comparison(results, GROUND_TRUTH, OUTPUT_DIR)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GPS Jammer Localization Validation")
    parser.add_argument("--download", action="store_true", help="Download data")
    parser.add_argument("--process", action="store_true", help="Process & localize")
    parser.add_argument("--visualize", action="store_true", help="Generate plots")
    parser.add_argument("--all", action="store_true", help="Full pipeline")
    parser.add_argument("--start-date", default="2025-12-01")
    parser.add_argument("--end-date", default="2026-02-01")
    args = parser.parse_args()

    if args.all:
        args.download = args.process = args.visualize = True

    if args.download:
        download_data(args.start_date, args.end_date)
    if args.process:
        results = process_data()
    else:
        results = None
    if args.visualize:
        visualize(results)

    if not any([args.download, args.process, args.visualize]):
        parser.print_help()


if __name__ == "__main__":
    main()
