"""
CYGNSS DDM RFI Detection Module
================================
Parses NASA PO.DAAC CYGNSS Level 1 NetCDF files and detects GPS jamming
via kurtosis-based anomaly detection in the DDM forbidden-zone bins.

The "forbidden zone" is the region of the Delay-Doppler Map at delays shorter
than the specular point — no legitimate surface reflections can arrive earlier
than the shortest geometric path. Elevated power here indicates RFI.

Detection method:
  1. For each DDM, extract bins in the forbidden zone (delay < specular delay).
  2. Compute kurtosis of power values in those bins.
  3. Gaussian noise → kurtosis ≈ 3.0; RFI → kurtosis >> 4.0 (heavy tails).
  4. Apply 10-second temporal persistence filter to reject transients.
  5. Return specular point coordinates of high-kurtosis DDMs as detections.
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
from scipy import stats

log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

KURTOSIS_THRESHOLD = 4.0   # Excess kurtosis threshold for RFI flag
PERSISTENCE_WINDOW_S = 10  # Temporal persistence filter (seconds)
MIN_FORBIDDEN_BINS = 3     # Minimum bins in forbidden zone for valid measurement
SEARCH_RADIUS_KM = 200     # Max distance from ground truth to consider


# ── Data Access ──────────────────────────────────────────────────────────────

def download_cygnss(gt_lat, gt_lon, radius_km, start_date, end_date, output_dir):
    """Download CYGNSS L1 DDM files from PO.DAAC via earthaccess."""
    import earthaccess

    earthaccess.login()

    # CYGNSS L1 v3.2 collection (PODAAC)
    results = earthaccess.search_data(
        short_name="CYGNSS_L1_V3.2",
        temporal=(start_date, end_date),
        bounding_box=(
            gt_lon - 3.0, gt_lat - 3.0,
            gt_lon + 3.0, gt_lat + 3.0,
        ),
    )
    log.info("Found %d CYGNSS granules", len(results))

    if not results:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = earthaccess.download(results, str(output_dir))
    return [Path(f) for f in files]


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_cygnss_l1(filepath):
    """Parse a CYGNSS L1 NetCDF file. Returns dict of arrays.

    Key variables extracted:
      - sp_lat, sp_lon: specular point coordinates
      - ddm_timestamp_utc: timestamp per DDM sample
      - power_analog: calibrated DDM power (n_samples × 17_delay × 11_doppler)
      - sp_delay_row: delay row index of the specular point within the DDM
      - quality_flags: bitfield quality/RFI flags
      - prn_code: GPS PRN of transmitting satellite
      - sc_num: CYGNSS spacecraft number (1-8)
    """
    import xarray as xr

    ds = xr.open_dataset(filepath, engine="netcdf4")

    data = {
        "sp_lat": ds["sp_lat"].values,
        "sp_lon": ds["sp_lon"].values,
    }

    # Timestamps — CYGNSS uses seconds since a reference epoch
    if "ddm_timestamp_utc" in ds:
        data["timestamp"] = ds["ddm_timestamp_utc"].values
    elif "sample_time" in ds:
        data["timestamp"] = ds["sample_time"].values

    # DDM power array: (samples, 17 delay, 11 doppler) or (samples, 4, 17, 11)
    for var_name in ("power_analog", "raw_counts", "brcs"):
        if var_name in ds:
            data["ddm_power"] = ds[var_name].values
            data["ddm_var"] = var_name
            break

    # Specular point bin indices within the DDM
    if "brcs_ddm_sp_bin_delay_row" in ds:
        data["sp_delay_row"] = ds["brcs_ddm_sp_bin_delay_row"].values
    if "brcs_ddm_sp_bin_dopp_col" in ds:
        data["sp_dopp_col"] = ds["brcs_ddm_sp_bin_dopp_col"].values

    # Quality flags
    if "quality_flags" in ds:
        data["quality_flags"] = ds["quality_flags"].values

    # Metadata
    if "prn_code" in ds:
        data["prn_code"] = ds["prn_code"].values
    if "sc_num" in ds:
        data["sc_num"] = ds["sc_num"].values

    ds.close()
    return data


# ── Forbidden Zone Kurtosis ─────────────────────────────────────────────────

def compute_forbidden_zone_kurtosis(ddm_power, sp_delay_row):
    """Compute excess kurtosis of DDM power in the forbidden zone.

    The forbidden zone is all delay bins with index < sp_delay_row (shorter
    path than specular reflection — physically impossible for surface returns).

    Args:
        ddm_power: 2D array (17 delay × 11 doppler) for a single DDM
        sp_delay_row: integer index of the specular point delay bin

    Returns:
        Excess kurtosis (float), or NaN if insufficient forbidden-zone bins.
    """
    if sp_delay_row is None or np.isnan(sp_delay_row):
        return np.nan

    sp_row = int(sp_delay_row)
    if sp_row < MIN_FORBIDDEN_BINS:
        return np.nan  # Not enough bins before specular point

    # Extract forbidden zone: all delay bins before the specular point
    forbidden = ddm_power[:sp_row, :]  # shape (sp_row, 11)
    values = forbidden.flatten()
    values = values[np.isfinite(values)]

    if len(values) < MIN_FORBIDDEN_BINS:
        return np.nan

    return float(stats.kurtosis(values, fisher=True))  # excess kurtosis


# ── Detection Pipeline ───────────────────────────────────────────────────────

def detect_cygnss_rfi(data_dir, ground_truth):
    """Run CYGNSS forbidden-zone kurtosis RFI detection on all files in data_dir.

    Returns list of RFIDetection objects (imported from rfi_validation).
    """
    from rfi_validation import RFIDetection, geodesic_distance_km

    data_dir = Path(data_dir)
    nc_files = sorted(data_dir.glob("*.nc"))
    if not nc_files:
        log.warning("No CYGNSS .nc files found in %s", data_dir)
        return []

    gt_lat, gt_lon = ground_truth["lat"], ground_truth["lon"]
    raw_detections = []

    for filepath in nc_files:
        log.info("Processing CYGNSS: %s", filepath.name)
        try:
            data = parse_cygnss_l1(filepath)
        except Exception as e:
            log.warning("Failed to parse %s: %s", filepath.name, e)
            continue

        if "ddm_power" not in data:
            log.warning("No DDM power variable in %s", filepath.name)
            continue

        ddm_power = data["ddm_power"]
        sp_lat = data["sp_lat"]
        sp_lon = data["sp_lon"]
        timestamps = data.get("timestamp")
        sp_delay_rows = data.get("sp_delay_row")

        # Handle multi-channel DDMs: (samples, channels, 17, 11) → take channel 0
        if ddm_power.ndim == 4:
            ddm_power = ddm_power[:, 0, :, :]
            if sp_delay_rows is not None and sp_delay_rows.ndim == 2:
                sp_delay_rows = sp_delay_rows[:, 0]
            if sp_lat.ndim == 2:
                sp_lat = sp_lat[:, 0]
                sp_lon = sp_lon[:, 0]

        n_samples = len(sp_lat)
        for i in range(n_samples):
            # Skip if outside search radius
            dist = geodesic_distance_km(gt_lat, gt_lon,
                                         float(sp_lat[i]), float(sp_lon[i]))
            if dist > SEARCH_RADIUS_KM:
                continue

            # Get specular point delay row
            sp_row = None
            if sp_delay_rows is not None and i < len(sp_delay_rows):
                sp_row = sp_delay_rows[i]

            if sp_row is None or np.isnan(sp_row):
                # Estimate: assume specular point is near center of 17-bin delay axis
                sp_row = 8

            # Compute forbidden-zone kurtosis
            kurt = compute_forbidden_zone_kurtosis(ddm_power[i], sp_row)

            if np.isnan(kurt) or kurt < KURTOSIS_THRESHOLD:
                continue

            ts_str = ""
            if timestamps is not None and i < len(timestamps):
                ts_val = timestamps[i]
                if hasattr(ts_val, "isoformat"):
                    ts_str = str(ts_val)
                else:
                    ts_str = str(ts_val)

            raw_detections.append(RFIDetection(
                lat=float(sp_lat[i]),
                lon=float(sp_lon[i]),
                intensity=float(kurt),
                timestamp=ts_str,
                modality="CYGNSS",
                metadata={"distance_km": round(dist, 2)},
            ))

    log.info("CYGNSS raw detections (kurtosis > %.1f): %d",
             KURTOSIS_THRESHOLD, len(raw_detections))

    # Apply 10-second temporal persistence filter
    filtered = temporal_persistence_filter(raw_detections, PERSISTENCE_WINDOW_S)
    log.info("CYGNSS after persistence filter: %d", len(filtered))

    return filtered


def temporal_persistence_filter(detections, window_seconds):
    """Keep only detections that have at least one other detection within
    `window_seconds`. Rejects isolated transient spikes."""
    if len(detections) <= 1:
        return detections

    # Sort by timestamp
    detections.sort(key=lambda d: d.timestamp)

    # Parse timestamps to floats for comparison
    ts_values = []
    for d in detections:
        try:
            ts_values.append(float(d.timestamp))
        except (ValueError, TypeError):
            ts_values.append(0.0)

    kept = []
    for i, det in enumerate(detections):
        has_neighbor = False
        for j, other in enumerate(detections):
            if i == j:
                continue
            if abs(ts_values[i] - ts_values[j]) <= window_seconds:
                has_neighbor = True
                break
        if has_neighbor:
            kept.append(det)

    return kept
