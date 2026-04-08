#!/usr/bin/env python3
"""
Daily Jammer Activity Check
============================
Streams CYGNSS data for a single date (or range of dates) and reports
whether the GPS jammer near Shiraz is active.

Uses the pre-computed noise floor (ddm_noise_floor) as the primary
discriminator — the same metric that detected the jammer in January.

Usage:
    python3 check_jammer_daily.py 2026-02-28
    python3 check_jammer_daily.py 2026-02-28 2026-04-07
"""

import sys
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────
GT_LAT, GT_LON = 27.3182, 52.8703
SEARCH_RADIUS_KM = 200
# Baseline stats from Dec 15 + Dec 27 2025 (jammer OFF)
# These are the known-clean reference values
BASELINE_NOISE_MEAN = 9858.0
BASELINE_NOISE_STD = 2091.0
DETECTION_THRESHOLD_SIGMA = 2.5  # same as main pipeline

OUTPUT_DIR = Path("output/daily_checks")


def check_date(date_str):
    """Stream CYGNSS data for one date and check for jammer activity."""
    import earthaccess
    import h5netcdf

    log.info("=" * 60)
    log.info("Checking %s for jammer activity near %.4f°N, %.4f°E",
             date_str, GT_LAT, GT_LON)
    log.info("=" * 60)

    date = datetime.strptime(date_str, "%Y-%m-%d")
    # Search for CYGNSS granules on this date
    results = earthaccess.search_data(
        short_name="CYGNSS_L1_V3.2",
        temporal=(date_str, date_str),
        bounding_box=(
            GT_LON - SEARCH_RADIUS_KM / 111.0,
            GT_LAT - SEARCH_RADIUS_KM / 111.0,
            GT_LON + SEARCH_RADIUS_KM / 111.0,
            GT_LAT + SEARCH_RADIUS_KM / 111.0,
        ),
    )

    if not results:
        log.warning("No CYGNSS granules found for %s", date_str)
        return {"date": date_str, "status": "no_data", "granules": 0,
                "detections": 0, "max_noise": None, "mean_noise": None}

    log.info("Found %d granules for %s", len(results), date_str)

    measurements = []
    cos_lat = np.cos(np.radians(GT_LAT))

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

                for sample_idx in range(n_samples):
                    for ddm_idx in range(n_ddm):
                        lat = float(sp_lat[sample_idx, ddm_idx])
                        lon = float(sp_lon[sample_idx, ddm_idx])

                        if lat == 0 and lon == 0:
                            continue
                        if not np.isfinite(lat) or not np.isfinite(lon):
                            continue

                        # Quick distance check
                        dlat = (lat - GT_LAT) * 111.0
                        dlon = (lon - GT_LON) * 111.0 * cos_lat
                        dist_km = np.sqrt(dlat**2 + dlon**2)
                        if dist_km > SEARCH_RADIUS_KM:
                            continue

                        nf = float(noise_floor[sample_idx, ddm_idx])
                        if not np.isfinite(nf) or nf <= 0:
                            continue

                        measurements.append({
                            "lat": lat, "lon": lon,
                            "noise_floor": nf,
                            "distance_km": round(dist_km, 1),
                        })
        except Exception as e:
            log.warning("Error reading granule: %s", e)
            continue

    log.info("Total measurements within %d km: %d", SEARCH_RADIUS_KM, len(measurements))

    if not measurements:
        return {"date": date_str, "status": "no_measurements", "granules": len(results),
                "detections": 0, "max_noise": None, "mean_noise": None}

    # Compute statistics
    noise_vals = np.array([m["noise_floor"] for m in measurements])
    mean_noise = float(np.mean(noise_vals))
    max_noise = float(np.max(noise_vals))
    std_noise = float(np.std(noise_vals))

    # Count detections above baseline threshold
    threshold = BASELINE_NOISE_MEAN + DETECTION_THRESHOLD_SIGMA * BASELINE_NOISE_STD
    elevated = noise_vals > threshold
    n_detections = int(elevated.sum())

    # Near-jammer stats (within 50 km)
    dists = np.array([m["distance_km"] for m in measurements])
    near_mask = dists < 50
    near_noise = noise_vals[near_mask] if near_mask.any() else np.array([])
    near_mean = float(np.mean(near_noise)) if len(near_noise) > 0 else None
    near_detections = int((near_noise > threshold).sum()) if len(near_noise) > 0 else 0

    # Elevation ratio (mean noise / baseline mean)
    elevation_pct = (mean_noise - BASELINE_NOISE_MEAN) / BASELINE_NOISE_MEAN * 100

    # Determine status
    if n_detections >= 20:
        status = "ACTIVE"
    elif n_detections >= 5:
        status = "POSSIBLE"
    else:
        status = "QUIET"

    result = {
        "date": date_str,
        "status": status,
        "granules": len(results),
        "total_measurements": len(measurements),
        "detections": n_detections,
        "near_detections_50km": near_detections,
        "mean_noise": round(mean_noise, 1),
        "max_noise": round(max_noise, 1),
        "baseline_mean": BASELINE_NOISE_MEAN,
        "elevation_pct": round(elevation_pct, 1),
        "near_mean_50km": round(near_mean, 1) if near_mean else None,
        "threshold": round(threshold, 1),
    }

    emoji = {"ACTIVE": "🔴", "POSSIBLE": "🟡", "QUIET": "🟢"}.get(status, "?")
    log.info("")
    log.info("  %s %s: %s", emoji, date_str, status)
    log.info("  Detections: %d (near: %d within 50 km)", n_detections, near_detections)
    log.info("  Mean noise: %.1f (baseline: %.1f, elevation: %+.1f%%)",
             mean_noise, BASELINE_NOISE_MEAN, elevation_pct)
    log.info("  Max noise: %.1f (threshold: %.1f)", max_noise, threshold)
    if near_mean:
        log.info("  Near-jammer mean (< 50 km): %.1f", near_mean)
    log.info("")

    return result


def main():
    import earthaccess
    earthaccess.login(strategy="netrc")

    if len(sys.argv) < 2:
        print("Usage: python3 check_jammer_daily.py START_DATE [END_DATE]")
        sys.exit(1)

    start_date = sys.argv[1]
    end_date = sys.argv[2] if len(sys.argv) > 2 else start_date

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        try:
            result = check_date(date_str)
            all_results.append(result)
        except Exception as e:
            log.error("Failed to check %s: %s", date_str, e)
            all_results.append({"date": date_str, "status": "ERROR", "error": str(e)})
        current += timedelta(days=1)

    # Summary
    log.info("=" * 60)
    log.info("JAMMER ACTIVITY SUMMARY")
    log.info("=" * 60)
    log.info("%-12s %-10s %6s %6s %8s %8s", "Date", "Status", "Det", "Near", "Mean", "Elev%")
    log.info("-" * 60)
    for r in all_results:
        log.info("%-12s %-10s %6s %6s %8s %8s",
                 r["date"], r["status"],
                 r.get("detections", "-"), r.get("near_detections_50km", "-"),
                 r.get("mean_noise", "-"), r.get("elevation_pct", "-"))

    # Save results
    out_path = OUTPUT_DIR / f"jammer_check_{start_date}_to_{end_date}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results saved to %s", out_path)

    # Also save a CSV for easy viewing
    csv_path = OUTPUT_DIR / f"jammer_check_{start_date}_to_{end_date}.csv"
    with open(csv_path, "w") as f:
        f.write("date,status,detections,near_50km,mean_noise,max_noise,elevation_pct\n")
        for r in all_results:
            f.write(f"{r['date']},{r['status']},{r.get('detections','')},{r.get('near_detections_50km','')},{r.get('mean_noise','')},{r.get('max_noise','')},{r.get('elevation_pct','')}\n")
    log.info("CSV saved to %s", csv_path)


if __name__ == "__main__":
    main()
