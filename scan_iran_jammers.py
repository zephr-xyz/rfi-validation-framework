#!/usr/bin/env python3
"""
Iran-Wide GPS Jammer Scanner
==============================
Scans all of Iran using CYGNSS noise floor data to detect and localize
GPS jammers. Compares a conflict-period date against a baseline date to
identify anomalous noise elevation, then clusters detections and runs
1/r² inverse-distance fitting on each cluster.

Usage:
    python3 scan_iran_jammers.py
    python3 scan_iran_jammers.py --conflict-date 2026-03-15 --baseline-date 2025-12-27
"""

import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import label as ndimage_label

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Iran bounding box ────────────────────────────────────────────────────
# Generous bounds covering all of Iran + margins
IRAN_LAT_MIN, IRAN_LAT_MAX = 24.5, 40.0
IRAN_LON_MIN, IRAN_LON_MAX = 43.5, 63.5

# ── Detection parameters ────────────────────────────────────────────────
GRID_RES_DEG = 0.1            # ~11 km grid cells for initial scan
DETECTION_ZSCORE = 2.5        # noise elevation z-score threshold
MIN_CLUSTER_CELLS = 3         # minimum grid cells to form a cluster
MIN_CLUSTER_DETECTIONS = 10   # minimum raw measurements in cluster
SEARCH_RADIUS_KM = 200        # earthaccess search radius per tile

# ── 1/r² localization parameters ────────────────────────────────────────
MIN_POINTS_FOR_FIT = 15       # minimum elevated measurements for fit
BOOTSTRAP_N = 200             # bootstrap iterations (fewer than validation for speed)
FIT_BOUND_DEG = 0.5           # ±degrees for L-BFGS-B bounds

OUTPUT_DIR = Path("output/iran_scan")


def haversine_km(lat1, lon1, lat2, lon2):
    """Fast haversine distance in km."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def scan_tile(tile_lat, tile_lon, conflict_date, baseline_date, search_radius_km=SEARCH_RADIUS_KM):
    """Stream CYGNSS data for a single tile center and return noise floor measurements.

    Returns list of dicts with lat, lon, noise_floor, distance_km for each date.
    """
    import earthaccess
    import h5netcdf

    cos_lat = np.cos(np.radians(tile_lat))
    measurements = {"conflict": [], "baseline": []}

    for label, date_str in [("conflict", conflict_date), ("baseline", baseline_date)]:
        try:
            results = earthaccess.search_data(
                short_name="CYGNSS_L1_V3.2",
                temporal=(date_str, date_str),
                bounding_box=(
                    tile_lon - search_radius_km / 111.0,
                    tile_lat - search_radius_km / 111.0,
                    tile_lon + search_radius_km / 111.0,
                    tile_lat + search_radius_km / 111.0,
                ),
            )
        except Exception as e:
            log.warning("Search failed for tile (%.1f, %.1f) %s: %s",
                        tile_lat, tile_lon, label, e)
            continue

        if not results:
            continue

        for granule in results:
            try:
                files = earthaccess.open([granule])
                if not files:
                    continue
                with h5netcdf.File(files[0], "r") as ds:
                    sp_lat = ds["sp_lat"][:]
                    sp_lon = ds["sp_lon"][:]
                    noise_floor = ds["ddm_noise_floor"][:]
                    n_samples, n_ddm = sp_lat.shape

                    for si in range(n_samples):
                        for di in range(n_ddm):
                            lat = float(sp_lat[si, di])
                            lon = float(sp_lon[si, di])

                            if lat == 0 and lon == 0:
                                continue
                            if not np.isfinite(lat) or not np.isfinite(lon):
                                continue
                            # Keep only points within Iran bounds
                            if not (IRAN_LAT_MIN <= lat <= IRAN_LAT_MAX and
                                    IRAN_LON_MIN <= lon <= IRAN_LON_MAX):
                                continue

                            # Distance from tile center
                            dlat = (lat - tile_lat) * 111.0
                            dlon = (lon - tile_lon) * 111.0 * cos_lat
                            dist_km = np.sqrt(dlat ** 2 + dlon ** 2)
                            if dist_km > search_radius_km:
                                continue

                            nf = float(noise_floor[si, di])
                            if not np.isfinite(nf) or nf <= 0:
                                continue

                            measurements[label].append({
                                "lat": lat, "lon": lon,
                                "noise_floor": nf,
                                "dist_km": round(dist_km, 1),
                            })
            except Exception as e:
                log.debug("Error reading granule: %s", e)
                continue

    return measurements


def build_noise_grid(measurements, baseline_stats):
    """Build a gridded map of noise elevation z-scores across Iran.

    Returns:
        grid: 2D array of mean z-scores per cell
        lat_edges, lon_edges: bin edges
        cell_measurements: dict mapping (i, j) -> list of measurements
    """
    n_lat = int((IRAN_LAT_MAX - IRAN_LAT_MIN) / GRID_RES_DEG) + 1
    n_lon = int((IRAN_LON_MAX - IRAN_LON_MIN) / GRID_RES_DEG) + 1

    lat_edges = np.linspace(IRAN_LAT_MIN, IRAN_LAT_MAX, n_lat + 1)
    lon_edges = np.linspace(IRAN_LON_MIN, IRAN_LON_MAX, n_lon + 1)

    # Accumulate measurements per cell
    cell_measurements = defaultdict(list)
    cell_zscores = defaultdict(list)

    baseline_mean = baseline_stats["mean"]
    baseline_std = baseline_stats["std"]

    for m in measurements:
        zscore = (m["noise_floor"] - baseline_mean) / baseline_std
        i = int((m["lat"] - IRAN_LAT_MIN) / GRID_RES_DEG)
        j = int((m["lon"] - IRAN_LON_MIN) / GRID_RES_DEG)
        i = min(i, n_lat - 1)
        j = min(j, n_lon - 1)
        cell_measurements[(i, j)].append(m)
        cell_zscores[(i, j)].append(zscore)

    # Build grid of mean z-scores
    grid = np.full((n_lat, n_lon), np.nan)
    for (i, j), zscores in cell_zscores.items():
        if len(zscores) >= 2:  # need at least 2 measurements
            grid[i, j] = np.mean(zscores)

    return grid, lat_edges, lon_edges, cell_measurements


def find_clusters(grid, lat_edges, lon_edges, cell_measurements):
    """Find connected clusters of elevated noise cells.

    Returns list of cluster dicts with center, measurements, stats.
    """
    # Binary mask: cells exceeding z-score threshold
    elevated = np.where(np.isnan(grid), False, grid > DETECTION_ZSCORE)

    # Connected component labeling
    labeled, n_clusters = ndimage_label(elevated)
    log.info("Found %d raw clusters of elevated noise", n_clusters)

    clusters = []
    for cluster_id in range(1, n_clusters + 1):
        cells = np.argwhere(labeled == cluster_id)
        if len(cells) < MIN_CLUSTER_CELLS:
            continue

        # Gather all measurements in this cluster
        all_meas = []
        for i, j in cells:
            all_meas.extend(cell_measurements.get((i, j), []))

        if len(all_meas) < MIN_CLUSTER_DETECTIONS:
            continue

        # Cluster statistics
        lats = np.array([m["lat"] for m in all_meas])
        lons = np.array([m["lon"] for m in all_meas])
        nfs = np.array([m["noise_floor"] for m in all_meas])

        # Intensity-weighted centroid as initial estimate
        weights = nfs / nfs.sum()
        centroid_lat = float(np.average(lats, weights=weights))
        centroid_lon = float(np.average(lons, weights=weights))

        # Grid cell centers for the cluster
        cell_lats = [(lat_edges[i] + lat_edges[i + 1]) / 2 for i, j in cells]
        cell_lons = [(lon_edges[j] + lon_edges[j + 1]) / 2 for i, j in cells]

        # Mean z-score
        cell_zscores = [grid[i, j] for i, j in cells]

        clusters.append({
            "cluster_id": cluster_id,
            "n_cells": len(cells),
            "n_measurements": len(all_meas),
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "mean_zscore": float(np.mean(cell_zscores)),
            "max_zscore": float(np.max(cell_zscores)),
            "mean_noise": float(np.mean(nfs)),
            "max_noise": float(np.max(nfs)),
            "extent_km": float(haversine_km(
                min(cell_lats), min(cell_lons),
                max(cell_lats), max(cell_lons))),
            "measurements": all_meas,
        })

    # Sort by mean z-score descending
    clusters.sort(key=lambda c: c["mean_zscore"], reverse=True)
    return clusters


def fit_jammer_location(cluster, baseline_mean):
    """Run 1/r² inverse-distance model fit on a cluster to localize the jammer.

    Returns dict with estimated position, error metrics, and bootstrap CEP.
    """
    measurements = cluster["measurements"]
    lats = np.array([m["lat"] for m in measurements])
    lons = np.array([m["lon"] for m in measurements])
    nfs = np.array([m["noise_floor"] for m in measurements])

    # Elevation above baseline
    elevations = nfs - baseline_mean
    elevated_mask = elevations > 0
    if elevated_mask.sum() < MIN_POINTS_FOR_FIT:
        return None

    e_lats = lats[elevated_mask]
    e_lons = lons[elevated_mask]
    e_elev = elevations[elevated_mask]

    # Normalize
    e_norm = e_elev / e_elev.max()

    # Initial guess: intensity-weighted centroid
    weights = e_elev / e_elev.sum()
    init_lat = float(np.average(e_lats, weights=weights))
    init_lon = float(np.average(e_lons, weights=weights))
    init_amp = float(np.max(e_elev))

    cos_lat = np.cos(np.radians(init_lat))

    def cost(params):
        src_lat, src_lon, amp = params
        dlat = (e_lats - src_lat) * 111.0
        dlon = (e_lons - src_lon) * 111.0 * cos_lat
        r = np.sqrt(dlat ** 2 + dlon ** 2)
        r = np.maximum(r, 1.0)  # avoid division by zero
        predicted = amp / (r ** 2)
        pred_norm = predicted / predicted.max() if predicted.max() > 0 else predicted
        residuals = e_norm - pred_norm
        return float(np.sum(e_norm * residuals ** 2))  # intensity-weighted

    # Bounded optimization
    bounds = [
        (init_lat - FIT_BOUND_DEG, init_lat + FIT_BOUND_DEG),
        (init_lon - FIT_BOUND_DEG, init_lon + FIT_BOUND_DEG),
        (1.0, max(init_amp * 10, 5000.0)),
    ]

    result = minimize(cost, [init_lat, init_lon, init_amp],
                      method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 2000, "ftol": 1e-10})

    est_lat, est_lon, amplitude = result.x

    # Bootstrap CEP
    boot_lats, boot_lons = [], []
    rng = np.random.RandomState(42)
    for _ in range(BOOTSTRAP_N):
        idx = rng.choice(len(e_lats), size=len(e_lats), replace=True)
        b_lats = e_lats[idx]
        b_lons = e_lons[idx]
        b_elev = e_elev[idx]
        b_norm = b_elev / b_elev.max()
        b_weights = b_elev / b_elev.sum()

        def b_cost(params):
            src_lat, src_lon, amp = params
            dlat = (b_lats - src_lat) * 111.0
            dlon = (b_lons - src_lon) * 111.0 * cos_lat
            r = np.sqrt(dlat ** 2 + dlon ** 2)
            r = np.maximum(r, 1.0)
            predicted = amp / (r ** 2)
            pred_norm = predicted / predicted.max() if predicted.max() > 0 else predicted
            residuals = b_norm - pred_norm
            return float(np.sum(b_norm * residuals ** 2))

        try:
            b_result = minimize(b_cost, [est_lat, est_lon, amplitude],
                                method="L-BFGS-B", bounds=bounds,
                                options={"maxiter": 500})
            if b_result.success or b_result.fun < cost([est_lat, est_lon, amplitude]) * 5:
                boot_lats.append(b_result.x[0])
                boot_lons.append(b_result.x[1])
        except Exception:
            continue

    cep_km = None
    if len(boot_lats) >= 50:
        boot_lats = np.array(boot_lats)
        boot_lons = np.array(boot_lons)
        radial_km = np.array([
            haversine_km(est_lat, est_lon, bl, bln)
            for bl, bln in zip(boot_lats, boot_lons)
        ])
        cep_km = float(np.median(radial_km))

    return {
        "estimated_lat": round(float(est_lat), 4),
        "estimated_lon": round(float(est_lon), 4),
        "amplitude": round(float(amplitude), 1),
        "residual": round(float(result.fun), 6),
        "n_elevated_points": int(elevated_mask.sum()),
        "bootstrap_cep_km": round(cep_km, 2) if cep_km else None,
        "bootstrap_n_fits": len(boot_lats),
        "centroid_lat": round(init_lat, 4),
        "centroid_lon": round(init_lon, 4),
    }


def deduplicate_jammers(jammers, merge_radius_km=30):
    """Merge jammer estimates that are within merge_radius_km of each other."""
    if not jammers:
        return jammers

    merged = []
    used = set()

    for i, j1 in enumerate(jammers):
        if i in used:
            continue
        group = [j1]
        for k, j2 in enumerate(jammers[i + 1:], start=i + 1):
            if k in used:
                continue
            dist = haversine_km(j1["estimated_lat"], j1["estimated_lon"],
                                j2["estimated_lat"], j2["estimated_lon"])
            if dist < merge_radius_km:
                group.append(j2)
                used.add(k)
        used.add(i)

        # Keep the estimate with the most measurements
        best = max(group, key=lambda j: j["n_elevated_points"])
        best["n_contributing_clusters"] = len(group)
        merged.append(best)

    return merged


def main():
    parser = argparse.ArgumentParser(description="Iran-wide GPS jammer scanner")
    parser.add_argument("--conflict-date", default="2026-03-15",
                        help="Conflict-period date to scan (default: 2026-03-15)")
    parser.add_argument("--baseline-date", default="2025-12-27",
                        help="Baseline (jammer-off) date (default: 2025-12-27)")
    args = parser.parse_args()

    conflict_date = args.conflict_date
    baseline_date = args.baseline_date

    log.info("=" * 70)
    log.info("IRAN-WIDE GPS JAMMER SCAN")
    log.info("  Conflict date: %s", conflict_date)
    log.info("  Baseline date: %s", baseline_date)
    log.info("  Grid resolution: %.2f° (~%.0f km)", GRID_RES_DEG, GRID_RES_DEG * 111)
    log.info("  Area: %.1f°N-%.1f°N, %.1f°E-%.1f°E",
             IRAN_LAT_MIN, IRAN_LAT_MAX, IRAN_LON_MIN, IRAN_LON_MAX)
    log.info("=" * 70)

    import earthaccess
    earthaccess.login(strategy="netrc")

    # ── Phase 1: Tile Iran and collect measurements ──────────────────────
    # Use overlapping tiles to ensure coverage
    tile_spacing = 3.0  # degrees between tile centers
    tile_radius_km = SEARCH_RADIUS_KM

    lat_centers = np.arange(IRAN_LAT_MIN + 1.5, IRAN_LAT_MAX - 1.0, tile_spacing)
    lon_centers = np.arange(IRAN_LON_MIN + 1.5, IRAN_LON_MAX - 1.0, tile_spacing)

    n_tiles = len(lat_centers) * len(lon_centers)
    log.info("Scanning %d tiles (%.0f° spacing, %d km radius each)",
             n_tiles, tile_spacing, tile_radius_km)

    all_conflict = []
    all_baseline = []
    seen_keys = set()  # deduplicate overlapping tile measurements

    tile_idx = 0
    for lat_c in lat_centers:
        for lon_c in lon_centers:
            tile_idx += 1
            log.info("  Tile %d/%d: center (%.1f°N, %.1f°E)", tile_idx, n_tiles, lat_c, lon_c)

            try:
                tile_data = scan_tile(lat_c, lon_c, conflict_date, baseline_date,
                                      search_radius_km=tile_radius_km)
            except Exception as e:
                log.warning("    Tile failed: %s", e)
                continue

            # Deduplicate: key on rounded lat/lon/noise to avoid counting
            # the same specular point from overlapping tiles
            for m in tile_data["conflict"]:
                key = (round(m["lat"], 3), round(m["lon"], 3), round(m["noise_floor"], 0))
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_conflict.append(m)

            for m in tile_data["baseline"]:
                key = (round(m["lat"], 3), round(m["lon"], 3), round(m["noise_floor"], 0))
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_baseline.append(m)

            log.info("    Conflict: %d meas, Baseline: %d meas (total: %d/%d)",
                     len(tile_data["conflict"]), len(tile_data["baseline"]),
                     len(all_conflict), len(all_baseline))

    log.info("")
    log.info("Total measurements: %d conflict, %d baseline", len(all_conflict), len(all_baseline))

    if not all_baseline:
        log.error("No baseline measurements — cannot compute thresholds. Aborting.")
        sys.exit(1)

    # ── Phase 2: Compute baseline statistics ─────────────────────────────
    baseline_nfs = np.array([m["noise_floor"] for m in all_baseline])
    baseline_stats = {
        "mean": float(np.mean(baseline_nfs)),
        "std": float(np.std(baseline_nfs)),
        "median": float(np.median(baseline_nfs)),
        "n": len(all_baseline),
    }
    log.info("Baseline stats: mean=%.1f, std=%.1f, n=%d",
             baseline_stats["mean"], baseline_stats["std"], baseline_stats["n"])

    # ── Phase 3: Build noise grid and find clusters ──────────────────────
    log.info("Building noise elevation grid...")
    grid, lat_edges, lon_edges, cell_measurements = build_noise_grid(
        all_conflict, baseline_stats)

    n_elevated = np.nansum(grid > DETECTION_ZSCORE)
    log.info("Elevated cells (z > %.1f): %d", DETECTION_ZSCORE, n_elevated)

    log.info("Finding clusters...")
    clusters = find_clusters(grid, lat_edges, lon_edges, cell_measurements)
    log.info("Found %d significant clusters (>= %d cells, >= %d measurements)",
             len(clusters), MIN_CLUSTER_CELLS, MIN_CLUSTER_DETECTIONS)

    for c in clusters:
        log.info("  Cluster %d: center (%.2f°N, %.2f°E), %d meas, z=%.1f, extent=%.0f km",
                 c["cluster_id"], c["centroid_lat"], c["centroid_lon"],
                 c["n_measurements"], c["mean_zscore"], c["extent_km"])

    # ── Phase 4: Localize each cluster with 1/r² fit ────────────────────
    log.info("")
    log.info("Running 1/r² localization on each cluster...")
    jammers = []

    for c in clusters:
        log.info("  Fitting cluster %d (%d measurements)...", c["cluster_id"], c["n_measurements"])
        fit = fit_jammer_location(c, baseline_stats["mean"])
        if fit is None:
            log.info("    Skipped — insufficient elevated points")
            continue

        fit["cluster_id"] = c["cluster_id"]
        fit["cluster_n_cells"] = c["n_cells"]
        fit["cluster_n_measurements"] = c["n_measurements"]
        fit["cluster_mean_zscore"] = c["mean_zscore"]
        fit["cluster_max_zscore"] = c["max_zscore"]
        fit["cluster_extent_km"] = c["extent_km"]
        jammers.append(fit)

        cep_str = f", CEP={fit['bootstrap_cep_km']:.1f} km" if fit["bootstrap_cep_km"] else ""
        log.info("    Localized: (%.4f°N, %.4f°E), %d points, amp=%.0f%s",
                 fit["estimated_lat"], fit["estimated_lon"],
                 fit["n_elevated_points"], fit["amplitude"], cep_str)

    # ── Phase 5: Deduplicate and rank ────────────────────────────────────
    log.info("")
    log.info("Deduplicating jammers (merge radius 30 km)...")
    jammers = deduplicate_jammers(jammers, merge_radius_km=30)
    log.info("Final jammer count: %d", len(jammers))

    # Rank by amplitude (proxy for jammer power)
    jammers.sort(key=lambda j: j["amplitude"], reverse=True)

    # ── Results ──────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("DETECTED GPS JAMMERS IN IRAN")
    log.info("  Conflict date: %s vs Baseline: %s", conflict_date, baseline_date)
    log.info("=" * 70)
    log.info("%-4s %-10s %-10s %8s %8s %6s %6s",
             "#", "Lat", "Lon", "Amp", "CEP km", "Pts", "Z-score")
    log.info("-" * 60)

    for i, j in enumerate(jammers, 1):
        cep = f"{j['bootstrap_cep_km']:.1f}" if j["bootstrap_cep_km"] else "N/A"
        log.info("%-4d %10.4f %10.4f %8.0f %8s %6d %6.1f",
                 i, j["estimated_lat"], j["estimated_lon"],
                 j["amplitude"], cep,
                 j["n_elevated_points"], j["cluster_mean_zscore"])

    # Check if our known Shiraz jammer is in the results
    KNOWN_LAT, KNOWN_LON = 27.3182, 52.8703
    for j in jammers:
        dist = haversine_km(KNOWN_LAT, KNOWN_LON, j["estimated_lat"], j["estimated_lon"])
        if dist < 50:
            log.info("")
            log.info("*** Shiraz jammer (known GT) matched: %.2f km from GT ***", dist)
            j["known_jammer"] = "Shiraz"
            j["gt_error_km"] = round(dist, 2)

    # ── Save results ─────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "scan_date": conflict_date,
        "baseline_date": baseline_date,
        "baseline_stats": baseline_stats,
        "n_conflict_measurements": len(all_conflict),
        "n_baseline_measurements": len(all_baseline),
        "n_clusters": len(clusters),
        "n_jammers": len(jammers),
        "jammers": jammers,
        "grid_resolution_deg": GRID_RES_DEG,
        "detection_zscore": DETECTION_ZSCORE,
    }

    out_json = OUTPUT_DIR / f"iran_jammers_{conflict_date}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", out_json)

    # Save cluster info separately for debugging
    cluster_info = [{k: v for k, v in c.items() if k != "measurements"}
                    for c in clusters]
    out_clusters = OUTPUT_DIR / f"iran_clusters_{conflict_date}.json"
    with open(out_clusters, "w") as f:
        json.dump(cluster_info, f, indent=2)
    log.info("Cluster details saved to %s", out_clusters)

    return results


if __name__ == "__main__":
    main()
