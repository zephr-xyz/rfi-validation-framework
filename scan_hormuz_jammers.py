#!/usr/bin/env python3
"""
Strait of Hormuz Multi-Pass Jammer Scanner
============================================
Extends the Iran jammer scanner to focus on GPS jammers threatening maritime
navigation through the Persian Gulf and Strait of Hormuz.

Key improvements over the single-day scanner:
  1. Multi-day CYGNSS stacking — multiple viewing geometries per jammer
     tightens localization (CEP shrinks with sqrt(N) independent passes)
  2. Focused search area — Iranian coastline facing the Gulf/Strait
  3. Higher resolution grid (0.05° ≈ 5.5 km vs 0.1° ≈ 11 km)
  4. Waterway impact modeling — projects 1/r² denial footprints onto
     shipping lanes to identify which jammers deny GPS at sea

Physics: A GPS jammer of power P at distance r produces interference
proportional to P/r². Ships lose GPS lock when jammer signal exceeds
receiver noise floor by ~20 dB (J/S ratio). From our validated Bushehr
jammer (amp=19718 at CEP 0.3km), we can estimate effective jamming range.

Usage:
    python3 scan_hormuz_jammers.py
    python3 scan_hormuz_jammers.py --n-days 7 --conflict-start 2026-04-01
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

# ── Geographic bounds: Persian Gulf + Strait of Hormuz coastal Iran ──────
# Covers Iranian coast from Kuwait border to Pakistan border
REGION_LAT_MIN, REGION_LAT_MAX = 24.5, 29.5
REGION_LON_MIN, REGION_LON_MAX = 48.5, 58.5

# Shipping lane waypoints — dense centerline through Gulf and Strait
# Approximately follows the Traffic Separation Scheme (TSS)
SHIPPING_LANE = [
    # Gulf of Oman approach
    (25.40, 57.00),
    (25.80, 56.80),
    (26.10, 56.60),
    # Strait of Hormuz (narrowest ~50km)
    (26.30, 56.40),
    (26.40, 56.20),
    (26.45, 56.00),
    (26.50, 55.80),
    (26.55, 55.60),
    # Western Strait into Gulf
    (26.55, 55.30),
    (26.60, 55.00),
    (26.65, 54.70),
    (26.70, 54.40),
    (26.80, 54.10),
    # Central Persian Gulf
    (26.90, 53.80),
    (27.00, 53.50),
    (27.10, 53.20),
    (27.25, 53.00),
    (27.40, 52.70),
    (27.55, 52.40),
    # Northern Gulf
    (27.70, 52.10),
    (27.90, 51.80),
    (28.10, 51.50),
    (28.30, 51.20),
    (28.50, 50.90),
    (28.60, 50.60),
    # Upper Gulf (Kuwait/Iraq approach)
    (28.80, 50.30),
    (29.00, 50.00),
    (29.20, 49.70),
    (29.40, 49.40),
]

# ── Detection parameters — higher resolution for coastal focus ───────────
GRID_RES_DEG = 0.05           # ~5.5 km grid cells
DETECTION_ZSCORE = 2.0        # lower threshold for sensitivity
MIN_CLUSTER_CELLS = 2         # allow smaller coastal clusters
MIN_CLUSTER_DETECTIONS = 8    # fewer required (tight area)
SEARCH_RADIUS_KM = 200

# ── Localization parameters ──────────────────────────────────────────────
MIN_POINTS_FOR_FIT = 10
BOOTSTRAP_N = 300             # more bootstraps for tighter CEP
FIT_BOUND_DEG = 0.3           # tighter bounds for coastal jammers

# ── Impact modeling ──────────────────────────────────────────────────────
# GPS denial threshold: calibrated against Bushehr ground truth jammer.
# Bushehr (amp=19718) denies GPS ~80km away per Gulf shipping reports.
# At 80km: 19718/80² = 3.08. Use threshold=3.0 for denial boundary.
DENIAL_AMP_THRESHOLD = 3.0    # calibrated 1/r² signal level for GPS denial
IMPACT_GRID_RES_DEG = 0.02   # ~2.2 km for impact heatmap

OUTPUT_DIR = Path("output/hormuz_scan")


def haversine_km(lat1, lon1, lat2, lon2):
    """Fast haversine distance in km."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


def scan_tile(tile_lat, tile_lon, date_str, search_radius_km=SEARCH_RADIUS_KM):
    """Stream CYGNSS data for a single tile/date, return noise floor measurements."""
    import earthaccess
    import h5netcdf

    cos_lat = np.cos(np.radians(tile_lat))
    measurements = []

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
                    tile_lat, tile_lon, date_str, e)
        return measurements

    if not results:
        return measurements

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
                        if not (REGION_LAT_MIN <= lat <= REGION_LAT_MAX and
                                REGION_LON_MIN <= lon <= REGION_LON_MAX):
                            continue

                        dlat = (lat - tile_lat) * 111.0
                        dlon = (lon - tile_lon) * 111.0 * cos_lat
                        dist_km = np.sqrt(dlat ** 2 + dlon ** 2)
                        if dist_km > search_radius_km:
                            continue

                        nf = float(noise_floor[si, di])
                        if not np.isfinite(nf) or nf <= 0:
                            continue

                        measurements.append({
                            "lat": lat, "lon": lon,
                            "noise_floor": nf,
                            "date": date_str,
                        })
        except Exception as e:
            log.debug("Error reading granule: %s", e)
            continue

    return measurements


def collect_multipass(conflict_dates, baseline_date):
    """Collect CYGNSS measurements across multiple conflict dates + one baseline.

    Returns (conflict_measurements, baseline_measurements).
    """
    import earthaccess
    earthaccess.login(strategy="netrc")

    # Tile the region with overlapping tiles
    tile_spacing = 2.0  # tighter spacing for focused area
    lat_centers = np.arange(REGION_LAT_MIN + 1.0, REGION_LAT_MAX, tile_spacing)
    lon_centers = np.arange(REGION_LON_MIN + 1.0, REGION_LON_MAX, tile_spacing)
    n_tiles = len(lat_centers) * len(lon_centers)

    log.info("Region: %.1f-%.1f°N, %.1f-%.1f°E",
             REGION_LAT_MIN, REGION_LAT_MAX, REGION_LON_MIN, REGION_LON_MAX)
    log.info("Tiles: %d (%.0f° spacing)", n_tiles, tile_spacing)
    log.info("Conflict dates: %s", ", ".join(conflict_dates))
    log.info("Baseline date: %s", baseline_date)

    all_conflict = []
    all_baseline = []
    seen_keys = set()

    all_dates = conflict_dates + [baseline_date]
    total_ops = n_tiles * len(all_dates)
    op_idx = 0

    for lat_c in lat_centers:
        for lon_c in lon_centers:
            # Baseline pass
            op_idx += 1
            log.info("  [%d/%d] Tile (%.1f, %.1f) baseline %s",
                     op_idx, total_ops, lat_c, lon_c, baseline_date)
            try:
                meas = scan_tile(lat_c, lon_c, baseline_date)
                for m in meas:
                    key = (round(m["lat"], 3), round(m["lon"], 3),
                           round(m["noise_floor"], 0), "baseline")
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_baseline.append(m)
            except Exception as e:
                log.warning("    Failed: %s", e)

            # Conflict passes
            for date_str in conflict_dates:
                op_idx += 1
                log.info("  [%d/%d] Tile (%.1f, %.1f) conflict %s",
                         op_idx, total_ops, lat_c, lon_c, date_str)
                try:
                    meas = scan_tile(lat_c, lon_c, date_str)
                    for m in meas:
                        key = (round(m["lat"], 3), round(m["lon"], 3),
                               round(m["noise_floor"], 0), date_str)
                        if key not in seen_keys:
                            seen_keys.add(key)
                            all_conflict.append(m)
                except Exception as e:
                    log.warning("    Failed: %s", e)

            log.info("    Running totals: %d conflict, %d baseline",
                     len(all_conflict), len(all_baseline))

    return all_conflict, all_baseline


def build_noise_grid(measurements, baseline_stats):
    """Build high-resolution gridded noise elevation map."""
    n_lat = int((REGION_LAT_MAX - REGION_LAT_MIN) / GRID_RES_DEG) + 1
    n_lon = int((REGION_LON_MAX - REGION_LON_MIN) / GRID_RES_DEG) + 1

    lat_edges = np.linspace(REGION_LAT_MIN, REGION_LAT_MAX, n_lat + 1)
    lon_edges = np.linspace(REGION_LON_MIN, REGION_LON_MAX, n_lon + 1)

    cell_measurements = defaultdict(list)
    cell_zscores = defaultdict(list)

    baseline_mean = baseline_stats["mean"]
    baseline_std = baseline_stats["std"]

    for m in measurements:
        zscore = (m["noise_floor"] - baseline_mean) / baseline_std
        i = int((m["lat"] - REGION_LAT_MIN) / GRID_RES_DEG)
        j = int((m["lon"] - REGION_LON_MIN) / GRID_RES_DEG)
        i = min(i, n_lat - 1)
        j = min(j, n_lon - 1)
        cell_measurements[(i, j)].append(m)
        cell_zscores[(i, j)].append(zscore)

    grid = np.full((n_lat, n_lon), np.nan)
    for (i, j), zscores in cell_zscores.items():
        if len(zscores) >= 2:
            grid[i, j] = np.mean(zscores)

    return grid, lat_edges, lon_edges, cell_measurements


def find_clusters(grid, lat_edges, lon_edges, cell_measurements):
    """Find connected clusters of elevated noise."""
    elevated = np.where(np.isnan(grid), False, grid > DETECTION_ZSCORE)
    labeled, n_clusters = ndimage_label(elevated)
    log.info("Found %d raw clusters", n_clusters)

    clusters = []
    for cluster_id in range(1, n_clusters + 1):
        cells = np.argwhere(labeled == cluster_id)
        if len(cells) < MIN_CLUSTER_CELLS:
            continue

        all_meas = []
        for i, j in cells:
            all_meas.extend(cell_measurements.get((i, j), []))

        if len(all_meas) < MIN_CLUSTER_DETECTIONS:
            continue

        lats = np.array([m["lat"] for m in all_meas])
        lons = np.array([m["lon"] for m in all_meas])
        nfs = np.array([m["noise_floor"] for m in all_meas])

        weights = nfs / nfs.sum()
        centroid_lat = float(np.average(lats, weights=weights))
        centroid_lon = float(np.average(lons, weights=weights))

        cell_lats = [(lat_edges[i] + lat_edges[i + 1]) / 2 for i, j in cells]
        cell_lons = [(lon_edges[j] + lon_edges[j + 1]) / 2 for i, j in cells]
        cell_zs = [grid[i, j] for i, j in cells]

        # Count unique dates contributing to this cluster
        dates = set(m.get("date", "") for m in all_meas)

        clusters.append({
            "cluster_id": cluster_id,
            "n_cells": len(cells),
            "n_measurements": len(all_meas),
            "n_dates": len(dates),
            "dates": sorted(dates),
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "mean_zscore": float(np.mean(cell_zs)),
            "max_zscore": float(np.max(cell_zs)),
            "mean_noise": float(np.mean(nfs)),
            "max_noise": float(np.max(nfs)),
            "extent_km": float(haversine_km(
                min(cell_lats), min(cell_lons),
                max(cell_lats), max(cell_lons))),
            "measurements": all_meas,
        })

    clusters.sort(key=lambda c: c["mean_zscore"], reverse=True)
    return clusters


def fit_jammer_location(cluster, baseline_mean):
    """1/r² inverse-distance localization with bootstrap CEP."""
    measurements = cluster["measurements"]
    lats = np.array([m["lat"] for m in measurements])
    lons = np.array([m["lon"] for m in measurements])
    nfs = np.array([m["noise_floor"] for m in measurements])

    elevations = nfs - baseline_mean
    elevated_mask = elevations > 0
    if elevated_mask.sum() < MIN_POINTS_FOR_FIT:
        return None

    e_lats = lats[elevated_mask]
    e_lons = lons[elevated_mask]
    e_elev = elevations[elevated_mask]
    e_norm = e_elev / e_elev.max()

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
        r = np.maximum(r, 1.0)
        predicted = amp / (r ** 2)
        pred_norm = predicted / predicted.max() if predicted.max() > 0 else predicted
        residuals = e_norm - pred_norm
        return float(np.sum(e_norm * residuals ** 2))

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
        b_lats, b_lons, b_elev = e_lats[idx], e_lons[idx], e_elev[idx]
        b_norm = b_elev / b_elev.max()

        def b_cost(params, _bl=b_lats, _bln=b_lons, _bn=b_norm):
            src_lat, src_lon, amp = params
            dlat = (_bl - src_lat) * 111.0
            dlon = (_bln - src_lon) * 111.0 * cos_lat
            r = np.sqrt(dlat ** 2 + dlon ** 2)
            r = np.maximum(r, 1.0)
            predicted = amp / (r ** 2)
            pn = predicted / predicted.max() if predicted.max() > 0 else predicted
            return float(np.sum(_bn * (_bn - pn) ** 2))

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
        "n_dates": cluster.get("n_dates", 1),
        "dates": cluster.get("dates", []),
        "bootstrap_cep_km": round(cep_km, 2) if cep_km else None,
        "bootstrap_n_fits": len(boot_lats),
        "centroid_lat": round(init_lat, 4),
        "centroid_lon": round(init_lon, 4),
    }


def deduplicate_jammers(jammers, merge_radius_km=20):
    """Merge nearby jammers (tighter radius for coastal region)."""
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
        best = max(group, key=lambda j: j["n_elevated_points"])
        best["n_contributing_clusters"] = len(group)
        merged.append(best)

    return merged


def compute_waterway_impact(jammers):
    """For each jammer, compute GPS denial range and impact on shipping lanes.

    Uses the 1/r² model: signal at distance r = amplitude / r²
    GPS denied when signal > DENIAL_AMP_THRESHOLD
    Therefore denial_range = sqrt(amplitude / threshold)
    """
    for j in jammers:
        amp = j["amplitude"]
        if amp > 0:
            # Denial range in km where jammer signal exceeds threshold
            denial_range_km = np.sqrt(amp / DENIAL_AMP_THRESHOLD)
            j["denial_range_km"] = round(denial_range_km, 1)

            # Check impact on each shipping lane segment
            impacts = []
            for wp_lat, wp_lon in SHIPPING_LANE:
                dist = haversine_km(j["estimated_lat"], j["estimated_lon"],
                                    wp_lat, wp_lon)
                if dist < denial_range_km:
                    signal_at_lane = amp / max(dist, 1.0) ** 2
                    impacts.append({
                        "lane_lat": wp_lat,
                        "lane_lon": wp_lon,
                        "distance_km": round(dist, 1),
                        "signal_strength": round(signal_at_lane, 1),
                    })
            j["shipping_lane_impacts"] = impacts
            j["impacts_shipping"] = len(impacts) > 0
            j["n_lane_points_denied"] = len(impacts)
        else:
            j["denial_range_km"] = 0
            j["shipping_lane_impacts"] = []
            j["impacts_shipping"] = False
            j["n_lane_points_denied"] = 0

    return jammers


def main():
    parser = argparse.ArgumentParser(
        description="Multi-pass Strait of Hormuz GPS jammer scanner")
    parser.add_argument("--conflict-start", default="2026-04-01",
                        help="First conflict date (default: 2026-04-01)")
    parser.add_argument("--n-days", type=int, default=5,
                        help="Number of conflict days to stack (default: 5)")
    parser.add_argument("--baseline-date", default="2025-12-27",
                        help="Baseline date (default: 2025-12-27)")
    parser.add_argument("--use-existing-scan", action="store_true",
                        help="Re-analyze existing Iran scan data instead of streaming new data")
    args = parser.parse_args()

    # Generate date range
    start = datetime.strptime(args.conflict_start, "%Y-%m-%d")
    conflict_dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d")
                      for i in range(args.n_days)]

    log.info("=" * 70)
    log.info("STRAIT OF HORMUZ MULTI-PASS JAMMER SCANNER")
    log.info("  Conflict dates: %s to %s (%d days)",
             conflict_dates[0], conflict_dates[-1], len(conflict_dates))
    log.info("  Baseline date: %s", args.baseline_date)
    log.info("  Grid resolution: %.3f° (~%.1f km)", GRID_RES_DEG, GRID_RES_DEG * 111)
    log.info("  Region: %.1f-%.1f°N, %.1f-%.1f°E",
             REGION_LAT_MIN, REGION_LAT_MAX, REGION_LON_MIN, REGION_LON_MAX)
    log.info("=" * 70)

    if args.use_existing_scan:
        # Re-analyze from the existing Iran-wide scan
        log.info("Loading existing Iran scan data...")
        existing = json.load(open("output/iran_scan/iran_jammers_2026-04-06.json"))
        jammers_raw = existing["jammers"]

        # Filter to our region
        region_jammers = []
        for j in jammers_raw:
            lat, lon = j["estimated_lat"], j["estimated_lon"]
            if (REGION_LAT_MIN <= lat <= REGION_LAT_MAX and
                    REGION_LON_MIN <= lon <= REGION_LON_MAX):
                # Propagate fields for compatibility
                j.setdefault("n_dates", 1)
                j.setdefault("dates", [existing["scan_date"]])
                j.setdefault("denial_range_km", 0)
                region_jammers.append(j)

        log.info("Found %d jammers in Gulf/Strait region from existing scan",
                 len(region_jammers))

        baseline_stats = existing["baseline_stats"]
        all_conflict = []  # not available in re-analysis mode
        all_baseline = []
    else:
        # Stream new multi-pass data
        all_conflict, all_baseline = collect_multipass(conflict_dates, args.baseline_date)

        log.info("Total measurements: %d conflict (%d days), %d baseline",
                 len(all_conflict), len(conflict_dates), len(all_baseline))

        if not all_baseline:
            log.error("No baseline measurements. Aborting.")
            sys.exit(1)

        # Baseline statistics
        baseline_nfs = np.array([m["noise_floor"] for m in all_baseline])
        baseline_stats = {
            "mean": float(np.mean(baseline_nfs)),
            "std": float(np.std(baseline_nfs)),
            "median": float(np.median(baseline_nfs)),
            "n": len(all_baseline),
        }
        log.info("Baseline: mean=%.1f, std=%.1f, n=%d",
                 baseline_stats["mean"], baseline_stats["std"], baseline_stats["n"])

        # Grid, cluster, localize
        log.info("Building noise grid...")
        grid, lat_edges, lon_edges, cell_meas = build_noise_grid(
            all_conflict, baseline_stats)

        n_elevated = np.nansum(grid > DETECTION_ZSCORE)
        log.info("Elevated cells: %d", n_elevated)

        clusters = find_clusters(grid, lat_edges, lon_edges, cell_meas)
        log.info("Significant clusters: %d", len(clusters))

        region_jammers = []
        for c in clusters:
            log.info("  Fitting cluster %d (%d meas, %d dates)...",
                     c["cluster_id"], c["n_measurements"], c["n_dates"])
            fit = fit_jammer_location(c, baseline_stats["mean"])
            if fit is None:
                continue
            fit["cluster_id"] = c["cluster_id"]
            fit["cluster_n_cells"] = c["n_cells"]
            fit["cluster_n_measurements"] = c["n_measurements"]
            fit["cluster_mean_zscore"] = c["mean_zscore"]
            fit["cluster_max_zscore"] = c["max_zscore"]
            fit["cluster_extent_km"] = c["extent_km"]
            region_jammers.append(fit)

        region_jammers = deduplicate_jammers(region_jammers, merge_radius_km=20)

    # ── Compute waterway impact ──────────────────────────────────────────
    log.info("Computing waterway denial footprints...")
    region_jammers = compute_waterway_impact(region_jammers)

    # Rank by shipping impact, then amplitude
    region_jammers.sort(
        key=lambda j: (j["impacts_shipping"], j["amplitude"]),
        reverse=True)

    # ── Results ──────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 80)
    log.info("GULF/STRAIT JAMMERS — MARITIME IMPACT ASSESSMENT")
    log.info("=" * 80)
    log.info("%-3s %-9s %-9s %7s %6s %5s %8s %6s %s",
             "#", "Lat", "Lon", "Amp", "CEP", "Pts",
             "Denial", "Lanes", "Impact")
    log.info("-" * 80)

    shipping_threats = 0
    for i, j in enumerate(region_jammers, 1):
        cep = f"{j['bootstrap_cep_km']:.1f}" if j.get("bootstrap_cep_km") else "N/A"
        denial = f"{j['denial_range_km']:.0f}km"
        impact = "YES" if j["impacts_shipping"] else "no"
        if j["impacts_shipping"]:
            shipping_threats += 1
        log.info("%-3d %9.4f %9.4f %7.0f %6s %5d %8s %6d %s",
                 i, j["estimated_lat"], j["estimated_lon"],
                 j["amplitude"], cep, j["n_elevated_points"],
                 denial, j["n_lane_points_denied"], impact)

    log.info("")
    log.info("SUMMARY: %d jammers detected, %d threaten shipping lanes",
             len(region_jammers), shipping_threats)

    # Estimate total denial zone along the Strait
    total_lane_km = 0
    for i in range(len(SHIPPING_LANE) - 1):
        total_lane_km += haversine_km(*SHIPPING_LANE[i], *SHIPPING_LANE[i + 1])
    denied_waypoints = set()
    for j in region_jammers:
        for imp in j.get("shipping_lane_impacts", []):
            denied_waypoints.add((imp["lane_lat"], imp["lane_lon"]))
    log.info("Shipping lane length: ~%.0f km", total_lane_km)
    log.info("Lane waypoints denied: %d / %d",
             len(denied_waypoints), len(SHIPPING_LANE))

    # ── Save ─────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "scan_type": "hormuz_multipass",
        "conflict_dates": conflict_dates,
        "baseline_date": args.baseline_date,
        "n_days_stacked": len(conflict_dates),
        "baseline_stats": baseline_stats,
        "n_conflict_measurements": len(all_conflict),
        "n_baseline_measurements": len(all_baseline),
        "region": {
            "lat_min": REGION_LAT_MIN, "lat_max": REGION_LAT_MAX,
            "lon_min": REGION_LON_MIN, "lon_max": REGION_LON_MAX,
        },
        "grid_resolution_deg": GRID_RES_DEG,
        "detection_zscore": DETECTION_ZSCORE,
        "denial_threshold": DENIAL_AMP_THRESHOLD,
        "n_jammers": len(region_jammers),
        "n_shipping_threats": shipping_threats,
        "shipping_lane": [{"lat": lat, "lon": lon} for lat, lon in SHIPPING_LANE],
        "jammers": region_jammers,
    }

    out_json = OUTPUT_DIR / "hormuz_jammers.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", out_json)

    return results


if __name__ == "__main__":
    main()
