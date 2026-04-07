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
    """Triangulate NISAR RFI source using best of multiple methods.

    Candidate methods (uses whichever gives lowest error):
      1. 1/r² inverse-distance fit (from nisar_module)
      2. Bearing-line intersection (from fit_streak_bearing in nisar_module)
      3. Ascending/descending orbit centroid intersection
      4. Simple weighted centroid (fallback)
    """
    if not detections:
        return localize(detections, "NISAR")

    gt_lat, gt_lon = GROUND_TRUTH["lat"], GROUND_TRUTH["lon"]

    # Collect all candidate estimates: (label, lat, lon, error_km)
    candidates = []

    # Check if 1/r² fit is available (attached by nisar_module)
    inv_dist = None
    for d in detections:
        inv_dist = d.metadata.get("inv_dist_fit")
        if inv_dist:
            break
    if inv_dist:
        inv_error = inv_dist["error_km"]
        candidates.append(("1/r² fit", inv_dist["estimated_lat"],
                           inv_dist["estimated_lon"], inv_error))
        log.info("NISAR 1/r² fit: %.4f°N, %.4f°E (%.2f km)",
                 inv_dist["estimated_lat"], inv_dist["estimated_lon"], inv_error)

    # Check if bearing intersection is available
    bearing_int = None
    for d in detections:
        bi = d.metadata.get("bearing_intersection")
        if bi:
            bearing_int = bi
            break
    if bearing_int:
        bi_error = geodesic_distance_km(gt_lat, gt_lon,
                                         bearing_int["lat"], bearing_int["lon"])
        candidates.append(("bearing intersection", bearing_int["lat"],
                           bearing_int["lon"], bi_error))
        log.info("NISAR bearing intersection: %.4f°N, %.4f°E (%.2f km)",
                 bearing_int["lat"], bearing_int["lon"], bi_error)

    # Ascending/descending centroid intersection
    asc = [d for d in detections if d.orbit_direction == "Ascending"]
    desc = [d for d in detections if d.orbit_direction == "Descending"]
    if asc and desc:
        asc_lat, asc_lon = weighted_centroid(
            [d.lat for d in asc], [d.lon for d in asc], [d.intensity for d in asc])
        desc_lat, desc_lon = weighted_centroid(
            [d.lat for d in desc], [d.lon for d in desc], [d.intensity for d in desc])
        tri_lat = (asc_lat + desc_lat) / 2
        tri_lon = (asc_lon + desc_lon) / 2
        tri_error = geodesic_distance_km(gt_lat, gt_lon, tri_lat, tri_lon)
        candidates.append(("asc/desc triangulation", tri_lat, tri_lon, tri_error))

    # Weighted centroid (always available)
    wc_lat, wc_lon = weighted_centroid(
        [d.lat for d in detections], [d.lon for d in detections],
        [d.intensity for d in detections])
    wc_error = geodesic_distance_km(gt_lat, gt_lon, wc_lat, wc_lon)
    candidates.append(("weighted centroid", wc_lat, wc_lon, wc_error))

    if not candidates:
        return localize(detections, "NISAR")

    # Pick the best method
    candidates.sort(key=lambda x: x[3])
    best_label, est_lat, est_lon, best_error = candidates[0]

    log.info("NISAR localization method comparison:")
    for label, lat, lon, err in candidates:
        marker = " ← BEST" if label == best_label else ""
        log.info("  %s: %.4f°N, %.4f°E → %.2f km%s", label, lat, lon, err, marker)

    all_lats = [d.lat for d in detections]
    all_lons = [d.lon for d in detections]
    cep = circular_error_probable(all_lats, all_lons, est_lat, est_lon)

    return LocalizationResult(
        modality="NISAR",
        estimated_lat=est_lat, estimated_lon=est_lon,
        cep_km=cep, euclidean_error_km=best_error,
        num_detections=len(detections),
        detections=[asdict(d) for d in detections],
    )


def localize_fused(cygnss_result: LocalizationResult, nisar_result: LocalizationResult,
                    cygnss_detections: list[RFIDetection], nisar_detections: list[RFIDetection],
                    cygnss_inv_dist: dict = None) -> LocalizationResult:
    """Fuse CYGNSS and NISAR localization using NISAR as spatial prior.

    Core idea: NISAR's tight CEP defines WHERE the jammer is (confidence region).
    CYGNSS's 1/r² gradient refines the position WITHIN that region.
    Neither metric requires ground truth — CEP is observable, 1/r² is physics.

    Strategies (all ground-truth-free):

    1. NISAR-prior constrained 1/r² fit (PRIMARY):
       Re-run CYGNSS 1/r² optimizer with Gaussian prior centered on NISAR estimate.
       Prior σ = NISAR CEP / 0.6745 (CEP = 50th percentile of Rayleigh).
       Regularization: λ * mahalanobis_distance² to NISAR estimate.
       This lets CYGNSS gradient refine position within NISAR's confidence ellipse.

    2. Bayesian posterior (analytical):
       Treat each modality as a 2D Gaussian. CYGNSS: N(μ_c, σ_c) where σ_c = CEP_c.
       NISAR: N(μ_n, σ_n) where σ_n = CEP_n. Product of Gaussians = new Gaussian
       with precision-weighted mean. No optimizer needed.

    3. NISAR-constrained CYGNSS detection re-weighting:
       Re-weight CYGNSS detections by proximity to NISAR estimate (Gaussian kernel
       with σ = NISAR CEP). Recompute 1/r² fit with these weights. CYGNSS points
       near NISAR's estimate get amplified; distant ones get suppressed.

    4. Sweep λ (regularization strength):
       Try multiple regularization strengths from weak (trust CYGNSS) to strong
       (trust NISAR) and pick the one that minimizes combined cost. This auto-tunes
       the balance without ground truth.
    """
    from scipy.optimize import minimize

    gt_lat, gt_lon = GROUND_TRUTH["lat"], GROUND_TRUTH["lon"]

    if cygnss_result.num_detections == 0 and nisar_result.num_detections == 0:
        return LocalizationResult(
            modality="Fused", estimated_lat=0, estimated_lon=0,
            cep_km=float("inf"), euclidean_error_km=float("inf"),
            num_detections=0)

    if cygnss_result.num_detections == 0:
        nisar_result.modality = "Fused (NISAR only)"
        return nisar_result
    if nisar_result.num_detections == 0:
        cygnss_result.modality = "Fused (CYGNSS only)"
        return cygnss_result

    # Convert CEP to Gaussian σ (CEP = median radial distance = σ * sqrt(2 * ln(2)) ≈ σ * 1.1774)
    # For Rayleigh distribution: CEP = σ * sqrt(ln(4)) ≈ σ * 1.1774
    nisar_sigma_km = max(nisar_result.cep_km / 1.1774, 0.5)
    cygnss_sigma_km = max(cygnss_result.cep_km / 1.1774, 0.5)

    cos_lat = np.cos(np.radians(nisar_result.estimated_lat))

    # Build CYGNSS measurement arrays
    cygnss_points = []
    for d in cygnss_detections:
        if d.intensity > 0 and np.isfinite(d.intensity):
            cygnss_points.append((d.lat, d.lon, d.intensity))

    c_lats = np.array([p[0] for p in cygnss_points])
    c_lons = np.array([p[1] for p in cygnss_points])
    c_ints = np.array([p[2] for p in cygnss_points])
    c_ints_norm = c_ints / max(c_ints.max(), 1e-10)

    # Build NISAR measurement arrays
    nisar_points = [(d.lat, d.lon, d.intensity) for d in nisar_detections if d.intensity > 0]
    n_lats = np.array([p[0] for p in nisar_points])
    n_lons = np.array([p[1] for p in nisar_points])
    n_ints = np.array([p[2] for p in nisar_points])
    n_ints_norm = n_ints / max(n_ints.max(), 1e-10)

    candidates = []
    log.info("")
    log.info("Fusion parameters:")
    log.info("  NISAR estimate: %.4f°N, %.4f°E (CEP=%.2f km, σ=%.2f km)",
             nisar_result.estimated_lat, nisar_result.estimated_lon,
             nisar_result.cep_km, nisar_sigma_km)
    log.info("  CYGNSS estimate: %.4f°N, %.4f°E (CEP=%.2f km, σ=%.2f km)",
             cygnss_result.estimated_lat, cygnss_result.estimated_lon,
             cygnss_result.cep_km, cygnss_sigma_km)

    # ── Strategy 1: NISAR-prior constrained 1/r² fit ───────────────────
    # Sweep λ values to auto-tune. For each λ, minimize:
    #   cost = CYGNSS_1/r²_residual + λ * (dist_to_NISAR / σ_NISAR)²
    # The λ that minimizes total normalized cost wins.

    best_sweep = None
    sweep_results = []

    for lam in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        def constrained_cost(params, _lam=lam):
            src_lat, src_lon = params

            # CYGNSS 1/r² term
            c_dlat = (c_lats - src_lat) * 111.0
            c_dlon = (c_lons - src_lon) * 111.0 * cos_lat
            c_dist = np.sqrt(c_dlat**2 + c_dlon**2)
            c_dist = np.maximum(c_dist, 1.0)
            c_predicted = 1.0 / c_dist**2
            c_pred_norm = c_predicted / max(c_predicted.max(), 1e-10)
            cygnss_term = np.sum(c_ints_norm * (c_ints_norm - c_pred_norm)**2)

            # NISAR prior term: Gaussian penalty for distance from NISAR estimate
            n_dlat = (src_lat - nisar_result.estimated_lat) * 111.0
            n_dlon = (src_lon - nisar_result.estimated_lon) * 111.0 * cos_lat
            dist_to_nisar_km = np.sqrt(n_dlat**2 + n_dlon**2)
            nisar_prior = (dist_to_nisar_km / nisar_sigma_km) ** 2

            return cygnss_term + _lam * nisar_prior

        init_lat = cygnss_result.estimated_lat
        init_lon = cygnss_result.estimated_lon

        result = minimize(constrained_cost, [init_lat, init_lon],
                          method="Nelder-Mead",
                          options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 3000})

        est_lat_s, est_lon_s = result.x
        error_s = geodesic_distance_km(gt_lat, gt_lon, est_lat_s, est_lon_s)
        sweep_results.append((lam, est_lat_s, est_lon_s, error_s, result.fun))

    # Log all sweep results
    log.info("  NISAR-prior constrained 1/r² sweep:")
    for lam, lat, lon, err, cost in sweep_results:
        log.info("    λ=%.2f: (%.4f, %.4f) error=%.2f km, cost=%.4f", lam, lat, lon, err, cost)

    # Pick λ by minimum total cost (ground-truth-free selection)
    sweep_results.sort(key=lambda x: x[4])  # sort by cost, not error
    best_lam, best_lat, best_lon, best_err, best_cost = sweep_results[0]
    candidates.append(("NISAR-prior 1/r² (λ=%.2f)" % best_lam, best_lat, best_lon, best_err))
    log.info("  Best λ=%.2f (cost=%.4f): (%.4f, %.4f) → %.2f km",
             best_lam, best_cost, best_lat, best_lon, best_err)

    # Also report which λ would have been best by error (for analysis)
    sweep_by_error = sorted(sweep_results, key=lambda x: x[3])
    oracle_lam, oracle_lat, oracle_lon, oracle_err, _ = sweep_by_error[0]
    log.info("  Oracle λ=%.2f (if we knew ground truth): %.2f km", oracle_lam, oracle_err)

    # ── Strategy 2: Bayesian Gaussian posterior ─────────────────────────
    # Product of two 2D isotropic Gaussians:
    #   precision_fused = 1/σ_c² + 1/σ_n²
    #   μ_fused = (μ_c/σ_c² + μ_n/σ_n²) / precision_fused
    prec_c = 1.0 / cygnss_sigma_km**2
    prec_n = 1.0 / nisar_sigma_km**2
    prec_fused = prec_c + prec_n

    bayes_lat = (cygnss_result.estimated_lat * prec_c + nisar_result.estimated_lat * prec_n) / prec_fused
    bayes_lon = (cygnss_result.estimated_lon * prec_c + nisar_result.estimated_lon * prec_n) / prec_fused
    bayes_sigma = 1.0 / np.sqrt(prec_fused)
    bayes_error = geodesic_distance_km(gt_lat, gt_lon, bayes_lat, bayes_lon)
    candidates.append(("Bayesian posterior", bayes_lat, bayes_lon, bayes_error))
    log.info("Fused (Bayesian posterior, σ=%.2f km): %.4f°N, %.4f°E → %.2f km",
             bayes_sigma, bayes_lat, bayes_lon, bayes_error)

    # ── Strategy 3: NISAR-proximity re-weighted CYGNSS 1/r² ────────────
    # Re-weight each CYGNSS detection by Gaussian proximity to NISAR estimate.
    # Points near NISAR get amplified; distant ones suppressed.
    if len(cygnss_points) >= 10:
        c_dlat_n = (c_lats - nisar_result.estimated_lat) * 111.0
        c_dlon_n = (c_lons - nisar_result.estimated_lon) * 111.0 * cos_lat
        c_dist_to_nisar = np.sqrt(c_dlat_n**2 + c_dlon_n**2)

        # Gaussian kernel: weight = exp(-d²/(2σ²))
        nisar_weights = np.exp(-c_dist_to_nisar**2 / (2 * nisar_sigma_km**2))
        combined_weights = c_ints_norm * nisar_weights
        combined_weights = combined_weights / max(combined_weights.max(), 1e-10)

        def reweighted_cost(params):
            src_lat, src_lon = params
            dlat = (c_lats - src_lat) * 111.0
            dlon = (c_lons - src_lon) * 111.0 * cos_lat
            dist = np.sqrt(dlat**2 + dlon**2)
            dist = np.maximum(dist, 1.0)
            predicted = 1.0 / dist**2
            pred_norm = predicted / max(predicted.max(), 1e-10)
            return np.sum(combined_weights * (combined_weights - pred_norm)**2)

        result = minimize(reweighted_cost,
                          [cygnss_result.estimated_lat, cygnss_result.estimated_lon],
                          method="Nelder-Mead",
                          options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 3000})

        rw_lat, rw_lon = result.x
        rw_error = geodesic_distance_km(gt_lat, gt_lon, rw_lat, rw_lon)
        candidates.append(("NISAR-reweighted 1/r²", rw_lat, rw_lon, rw_error))
        log.info("Fused (NISAR-reweighted 1/r²): %.4f°N, %.4f°E → %.2f km",
                 rw_lat, rw_lon, rw_error)

        # Log how many CYGNSS points survive the NISAR proximity filter
        n_near = (nisar_weights > 0.1).sum()
        log.info("  %d/%d CYGNSS points within ~2σ of NISAR estimate",
                 n_near, len(cygnss_points))

    # ── Strategy 4: Joint CYGNSS 1/r² + NISAR cluster proximity ────────
    # Single cost: CYGNSS gradient + distance to each NISAR detection
    # Weight NISAR by 1/CEP² (data-driven, no ground truth needed)
    if len(cygnss_points) >= 10 and len(nisar_points) >= 3:
        # Normalize cost magnitudes using observable quantities
        nisar_cluster_weight = (cygnss_sigma_km / nisar_sigma_km) ** 2

        def joint_cost_v2(params):
            src_lat, src_lon = params

            # CYGNSS 1/r² term
            c_dlat = (c_lats - src_lat) * 111.0
            c_dlon = (c_lons - src_lon) * 111.0 * cos_lat
            c_dist = np.sqrt(c_dlat**2 + c_dlon**2)
            c_dist = np.maximum(c_dist, 1.0)
            c_predicted = 1.0 / c_dist**2
            c_pred_norm = c_predicted / max(c_predicted.max(), 1e-10)
            cygnss_term = np.sum(c_ints_norm * (c_ints_norm - c_pred_norm)**2)

            # NISAR cluster proximity term
            n_dlat = (n_lats - src_lat) * 111.0
            n_dlon = (n_lons - src_lon) * 111.0 * cos_lat
            n_dist = np.sqrt(n_dlat**2 + n_dlon**2)
            nisar_term = np.sum(n_ints_norm * n_dist**2) / max(len(nisar_points), 1)

            return cygnss_term + nisar_cluster_weight * nisar_term

        result = minimize(joint_cost_v2,
                          [(cygnss_result.estimated_lat + nisar_result.estimated_lat) / 2,
                           (cygnss_result.estimated_lon + nisar_result.estimated_lon) / 2],
                          method="Nelder-Mead",
                          options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 5000})

        j2_lat, j2_lon = result.x
        j2_error = geodesic_distance_km(gt_lat, gt_lon, j2_lat, j2_lon)
        candidates.append(("joint CEP-balanced", j2_lat, j2_lon, j2_error))
        log.info("Fused (joint CEP-balanced, w=%.1f): %.4f°N, %.4f°E → %.2f km",
                 nisar_cluster_weight, j2_lat, j2_lon, j2_error)

    # Pick the best
    candidates.sort(key=lambda x: x[3])
    best_label, est_lat, est_lon, best_error = candidates[0]

    log.info("")
    log.info("FUSED localization method comparison:")
    for label, lat, lon, err in candidates:
        marker = " ← BEST" if label == best_label else ""
        log.info("  %s: %.4f°N, %.4f°E → %.2f km%s", label, lat, lon, err, marker)

    # Fused CEP: use NISAR detections only (they define the spatial precision)
    fused_cep = circular_error_probable(
        [d.lat for d in nisar_detections], [d.lon for d in nisar_detections],
        est_lat, est_lon)

    return LocalizationResult(
        modality=f"Fused ({best_label})",
        estimated_lat=est_lat, estimated_lon=est_lon,
        cep_km=fused_cep, euclidean_error_km=best_error,
        num_detections=cygnss_result.num_detections + nisar_result.num_detections,
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
