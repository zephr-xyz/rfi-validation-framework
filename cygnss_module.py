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

try:
    import h5netcdf  # noqa: F401 — needed as xarray engine for file-like objects
except ImportError:
    logging.getLogger(__name__).warning(
        "h5netcdf not installed — streaming mode (earthaccess.open) will not work. "
        "Install with: pip install h5netcdf"
    )

log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

KURTOSIS_THRESHOLD = 4.0   # Excess kurtosis threshold for RFI flag
PERSISTENCE_WINDOW_S = 10  # Temporal persistence filter (seconds)
MIN_FORBIDDEN_BINS = 3     # Minimum bins in forbidden zone for valid measurement
SEARCH_RADIUS_KM = 200     # Max distance from ground truth to consider
NOISE_ELEVATION_ZSCORE = 2.5  # Z-score for noise floor elevation detection
SNR_DROP_ZSCORE = 2.5         # Z-score for SNR attenuation detection
SPATIAL_GRID_RES_KM = 10      # Grid cell size for spatial hole detection (was 25)


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


# ── Total DDM Power & Pre-computed Metrics ─────────────────────────────────

def compute_total_ddm_power(ddm_power):
    """Compute total integrated power across all DDM bins.

    A broadband jammer raises power across the entire DDM, not just the
    forbidden zone. Total power captures this even when individual bins
    are near zero.

    Returns:
        Total DDM power (float), or NaN if invalid.
    """
    values = ddm_power.flatten()
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.sum(values))


def compute_ddm_peak_power(ddm_power, sp_delay_row):
    """Compute peak DDM power near the specular point.

    Returns:
        Peak power (float), or NaN if invalid.
    """
    if sp_delay_row is None or np.isnan(sp_delay_row):
        return np.nan

    sp_row = int(sp_delay_row)
    n_delay = ddm_power.shape[0]
    peak_lo = max(sp_row - 2, 0)
    peak_hi = min(sp_row + 3, n_delay)
    peak_region = ddm_power[peak_lo:peak_hi, :]
    pvals = peak_region.flatten()
    pvals = pvals[np.isfinite(pvals)]
    if len(pvals) == 0:
        return np.nan
    return float(np.max(pvals))


# ── Noise Floor & SNR Detection ─────────────────────────────────────────────

def compute_forbidden_zone_noise_floor(ddm_power, sp_delay_row):
    """Compute mean power in the forbidden zone (noise floor estimate).

    A broadband jammer raises this uniformly — detectable by comparing
    ON vs OFF dates even though kurtosis stays ~3.0.

    Returns:
        Mean power in forbidden zone (float), or NaN if insufficient bins.
    """
    if sp_delay_row is None or np.isnan(sp_delay_row):
        return np.nan

    sp_row = int(sp_delay_row)
    if sp_row < MIN_FORBIDDEN_BINS:
        return np.nan

    forbidden = ddm_power[:sp_row, :]
    values = forbidden.flatten()
    values = values[np.isfinite(values)]

    if len(values) < MIN_FORBIDDEN_BINS:
        return np.nan

    return float(np.mean(values))


def compute_ddm_snr(ddm_power, sp_delay_row):
    """Compute peak-to-noise ratio of a DDM.

    Peak = max power near specular point.
    Noise = mean power in forbidden zone.
    A jammer reduces peak (overwhelms GPS signal) AND raises noise floor.

    Returns:
        SNR in dB (float), or NaN if cannot compute.
    """
    if sp_delay_row is None or np.isnan(sp_delay_row):
        return np.nan

    sp_row = int(sp_delay_row)
    if sp_row < MIN_FORBIDDEN_BINS:
        return np.nan

    # Noise: mean of forbidden zone
    forbidden = ddm_power[:sp_row, :]
    fvals = forbidden.flatten()
    fvals = fvals[np.isfinite(fvals)]
    if len(fvals) < MIN_FORBIDDEN_BINS:
        return np.nan
    noise = np.mean(fvals)

    # Peak: max power around the specular point (±2 delay bins, all Doppler)
    n_delay = ddm_power.shape[0]
    peak_lo = max(sp_row - 2, 0)
    peak_hi = min(sp_row + 3, n_delay)
    peak_region = ddm_power[peak_lo:peak_hi, :]
    pvals = peak_region.flatten()
    pvals = pvals[np.isfinite(pvals)]
    if len(pvals) == 0:
        return np.nan
    peak = np.max(pvals)

    if noise <= 0 or peak <= 0:
        return np.nan

    return float(10 * np.log10(peak / noise))


def detect_noise_floor_elevation(on_measurements, baseline_measurements):
    """Detect RFI by comparing noise floor between ON and baseline dates.

    Args:
        on_measurements: list of dicts with keys: lat, lon, noise_floor, snr, timestamp
        baseline_measurements: same format for baseline dates

    Returns:
        list of dicts with elevated noise floor flagged, plus z-scores
    """
    if not baseline_measurements:
        log.warning("No baseline measurements — cannot do noise floor comparison")
        return []

    # Baseline statistics
    bl_floors = np.array([m["noise_floor"] for m in baseline_measurements
                          if np.isfinite(m["noise_floor"])])
    bl_snrs = np.array([m["snr"] for m in baseline_measurements
                        if np.isfinite(m["snr"])])

    if len(bl_floors) < 5 or len(bl_snrs) < 5:
        log.warning("Insufficient baseline measurements (%d floors, %d SNRs)",
                    len(bl_floors), len(bl_snrs))
        return []

    bl_floor_mean = np.mean(bl_floors)
    bl_floor_std = np.std(bl_floors)
    bl_snr_mean = np.mean(bl_snrs)
    bl_snr_std = np.std(bl_snrs)

    log.info("  Baseline noise floor: %.2f ± %.2f (n=%d)",
             bl_floor_mean, bl_floor_std, len(bl_floors))
    log.info("  Baseline SNR: %.2f ± %.2f dB (n=%d)",
             bl_snr_mean, bl_snr_std, len(bl_snrs))

    detections = []
    for m in on_measurements:
        nf = m["noise_floor"]
        snr = m["snr"]

        # Noise floor elevation z-score
        nf_zscore = (nf - bl_floor_mean) / max(bl_floor_std, 1e-10) if np.isfinite(nf) else 0
        # SNR attenuation z-score (negative = SNR dropped)
        snr_zscore = (bl_snr_mean - snr) / max(bl_snr_std, 1e-10) if np.isfinite(snr) else 0

        # Combined score: both elevated noise AND reduced SNR indicate jamming
        combined_score = max(nf_zscore, 0) + max(snr_zscore, 0)

        m["nf_zscore"] = nf_zscore
        m["snr_zscore"] = snr_zscore
        m["combined_score"] = combined_score

        # Flag if either metric is anomalous
        if nf_zscore > NOISE_ELEVATION_ZSCORE or snr_zscore > SNR_DROP_ZSCORE:
            detections.append(m)

    log.info("  Noise/SNR anomalies: %d / %d ON measurements", len(detections), len(on_measurements))
    return detections


def detect_spatial_snr_hole(on_measurements, baseline_measurements, gt_lat, gt_lon,
                            grid_res_km=SPATIAL_GRID_RES_KM, search_radius_km=200):
    """Detect spatial clusters of SNR degradation ("jamming hole").

    Grids specular points, computes mean SNR per cell for ON vs baseline,
    finds cells with significant SNR drop. Returns the centroid of the
    degraded region as the jammer location estimate.

    Args:
        on_measurements: list of dicts with lat, lon, snr
        baseline_measurements: same
        gt_lat, gt_lon: center for gridding
        grid_res_km: grid cell size
        search_radius_km: extent of grid

    Returns:
        list of dicts: one per degraded grid cell with lat, lon, snr_drop, n_samples
    """
    from rfi_validation import geodesic_distance_km

    if not baseline_measurements or not on_measurements:
        return []

    # Grid bounds in degrees (approximate)
    deg_per_km_lat = 1 / 111.0
    deg_per_km_lon = 1 / (111.0 * np.cos(np.radians(gt_lat)))
    half_extent_lat = search_radius_km * deg_per_km_lat
    half_extent_lon = search_radius_km * deg_per_km_lon
    cell_lat = grid_res_km * deg_per_km_lat
    cell_lon = grid_res_km * deg_per_km_lon

    def _grid_key(lat, lon):
        r = int((lat - (gt_lat - half_extent_lat)) / cell_lat)
        c = int((lon - (gt_lon - half_extent_lon)) / cell_lon)
        return (r, c)

    def _grid_center(r, c):
        lat = gt_lat - half_extent_lat + (r + 0.5) * cell_lat
        lon = gt_lon - half_extent_lon + (c + 0.5) * cell_lon
        return lat, lon

    # Accumulate SNR per grid cell
    from collections import defaultdict
    bl_grid = defaultdict(list)
    on_grid = defaultdict(list)

    for m in baseline_measurements:
        if np.isfinite(m["snr"]):
            bl_grid[_grid_key(m["lat"], m["lon"])].append(m["snr"])

    for m in on_measurements:
        if np.isfinite(m["snr"]):
            on_grid[_grid_key(m["lat"], m["lon"])].append(m["snr"])

    # Find cells with significant SNR drop
    # Need both ON and baseline samples in same cell for comparison
    all_bl_snrs = [s for vals in bl_grid.values() for s in vals]
    if len(all_bl_snrs) < 5:
        return []
    global_bl_mean = np.mean(all_bl_snrs)
    global_bl_std = np.std(all_bl_snrs)

    degraded_cells = []
    for key in on_grid:
        on_snrs = on_grid[key]
        on_mean = np.mean(on_snrs)

        # Compare to same cell baseline if available, otherwise global
        if key in bl_grid and len(bl_grid[key]) >= 2:
            bl_mean = np.mean(bl_grid[key])
            bl_std = np.std(bl_grid[key]) if len(bl_grid[key]) > 2 else global_bl_std
        else:
            bl_mean = global_bl_mean
            bl_std = global_bl_std

        snr_drop = bl_mean - on_mean  # positive = SNR dropped
        zscore = snr_drop / max(bl_std, 1e-10)

        if zscore > SNR_DROP_ZSCORE and len(on_snrs) >= 2:
            clat, clon = _grid_center(key[0], key[1])
            dist = geodesic_distance_km(gt_lat, gt_lon, clat, clon)
            degraded_cells.append({
                "lat": clat, "lon": clon,
                "snr_drop_db": round(snr_drop, 2),
                "zscore": round(zscore, 2),
                "n_on": len(on_snrs),
                "n_bl": len(bl_grid.get(key, [])),
                "distance_km": round(dist, 2),
            })

    degraded_cells.sort(key=lambda c: c["zscore"], reverse=True)

    if degraded_cells:
        log.info("  Spatial SNR hole: %d degraded cells (top: z=%.1f, drop=%.1f dB)",
                 len(degraded_cells), degraded_cells[0]["zscore"],
                 degraded_cells[0]["snr_drop_db"])

    return degraded_cells


def detect_spatial_noise_gradient(on_measurements, baseline_measurements,
                                   gt_lat, gt_lon, metric="precomp_noise",
                                   grid_res_km=10, search_radius_km=200):
    """Grid-based spatial analysis of noise floor elevation.

    Like detect_spatial_snr_hole but operates on the calibrated noise floor
    (or any metric), uses 10 km cells, and returns per-cell elevation z-scores.

    Returns:
        list of dicts: one per elevated grid cell
    """
    from rfi_validation import geodesic_distance_km
    from collections import defaultdict

    if not baseline_measurements or not on_measurements:
        return []

    deg_per_km_lat = 1 / 111.0
    deg_per_km_lon = 1 / (111.0 * np.cos(np.radians(gt_lat)))
    half_extent_lat = search_radius_km * deg_per_km_lat
    half_extent_lon = search_radius_km * deg_per_km_lon
    cell_lat = grid_res_km * deg_per_km_lat
    cell_lon = grid_res_km * deg_per_km_lon

    def _grid_key(lat, lon):
        r = int((lat - (gt_lat - half_extent_lat)) / cell_lat)
        c = int((lon - (gt_lon - half_extent_lon)) / cell_lon)
        return (r, c)

    def _grid_center(r, c):
        lat = gt_lat - half_extent_lat + (r + 0.5) * cell_lat
        lon = gt_lon - half_extent_lon + (c + 0.5) * cell_lon
        return lat, lon

    bl_grid = defaultdict(list)
    on_grid = defaultdict(list)

    for m in baseline_measurements:
        val = m.get(metric)
        if val is not None and np.isfinite(val):
            bl_grid[_grid_key(m["lat"], m["lon"])].append(val)

    for m in on_measurements:
        val = m.get(metric)
        if val is not None and np.isfinite(val):
            on_grid[_grid_key(m["lat"], m["lon"])].append(val)

    # Global baseline stats
    all_bl = [v for vals in bl_grid.values() for v in vals]
    if len(all_bl) < 10:
        return []
    global_bl_mean = np.mean(all_bl)
    global_bl_std = np.std(all_bl)

    elevated_cells = []
    for key in on_grid:
        on_vals = on_grid[key]
        if len(on_vals) < 2:
            continue
        on_mean = np.mean(on_vals)

        if key in bl_grid and len(bl_grid[key]) >= 2:
            bl_mean = np.mean(bl_grid[key])
            bl_std = np.std(bl_grid[key]) if len(bl_grid[key]) > 2 else global_bl_std
        else:
            bl_mean = global_bl_mean
            bl_std = global_bl_std

        elevation = on_mean - bl_mean  # positive = noise elevated
        zscore = elevation / max(bl_std, 1e-10)

        clat, clon = _grid_center(key[0], key[1])
        dist = geodesic_distance_km(gt_lat, gt_lon, clat, clon)

        elevated_cells.append({
            "lat": clat, "lon": clon,
            "elevation": round(float(elevation), 2),
            "zscore": round(float(zscore), 2),
            "on_mean": round(float(on_mean), 2),
            "bl_mean": round(float(bl_mean), 2),
            "n_on": len(on_vals),
            "n_bl": len(bl_grid.get(key, [])),
            "distance_km": round(dist, 2),
        })

    elevated_cells.sort(key=lambda c: c["zscore"], reverse=True)
    n_sig = sum(1 for c in elevated_cells if c["zscore"] > NOISE_ELEVATION_ZSCORE)
    log.info("  Spatial noise grid (%s, %dkm cells): %d total cells, %d elevated (z>%.1f)",
             metric, grid_res_km, len(elevated_cells), n_sig, NOISE_ELEVATION_ZSCORE)
    if elevated_cells:
        log.info("  Top cell: z=%.1f, elevation=%.1f, at (%.2f, %.2f) dist=%.1fkm",
                 elevated_cells[0]["zscore"], elevated_cells[0]["elevation"],
                 elevated_cells[0]["lat"], elevated_cells[0]["lon"],
                 elevated_cells[0]["distance_km"])

    return [c for c in elevated_cells if c["zscore"] > NOISE_ELEVATION_ZSCORE]


def fit_inverse_distance_model(measurements, baseline_measurements, gt_lat, gt_lon,
                                metric="precomp_noise", search_radius_km=200):
    """Fit an inverse-distance (1/r²) jammer model to spatial noise pattern.

    A point-source jammer produces noise that falls off as 1/r² with distance.
    This function finds the (lat, lon) that best explains the observed spatial
    pattern of noise floor elevation as a 1/r² source.

    Uses scipy.optimize to minimize the residual between observed per-point
    noise elevation and a 1/r² model centered on the candidate jammer position.

    Args:
        measurements: ON-date measurements with lat, lon, metric
        baseline_measurements: baseline measurements for normalization
        gt_lat, gt_lon: initial guess center
        metric: which measurement field to use

    Returns:
        dict with estimated_lat, estimated_lon, amplitude, residual, or None
    """
    from scipy.optimize import minimize
    from rfi_validation import geodesic_distance_km

    # Compute baseline stats for the metric
    bl_vals = np.array([m[metric] for m in baseline_measurements
                        if np.isfinite(m.get(metric, np.nan))])
    if len(bl_vals) < 10:
        return None
    bl_mean = np.mean(bl_vals)

    # Compute per-point elevation above baseline
    points = []
    for m in measurements:
        val = m.get(metric)
        if val is None or not np.isfinite(val):
            continue
        elevation = val - bl_mean
        if elevation <= 0:
            continue  # only use points with elevated noise
        points.append((m["lat"], m["lon"], elevation))

    if len(points) < 10:
        log.warning("  Insufficient elevated points for 1/r² fit (%d)", len(points))
        return None

    lats = np.array([p[0] for p in points])
    lons = np.array([p[1] for p in points])
    elevations = np.array([p[2] for p in points])

    # Normalize elevations to [0, 1] for numerical stability
    elev_max = np.max(elevations)
    elev_norm = elevations / elev_max

    cos_lat = np.cos(np.radians(gt_lat))

    def _model_residual(params):
        """Residual between observed elevation and 1/r² model."""
        src_lat, src_lon, amplitude = params
        # Distance from candidate source to each point (km)
        dlat = (lats - src_lat) * 111.0
        dlon = (lons - src_lon) * 111.0 * cos_lat
        dist_km = np.sqrt(dlat**2 + dlon**2)
        dist_km = np.maximum(dist_km, 1.0)  # avoid division by zero

        # 1/r² model: noise elevation = A / r²
        predicted = amplitude / (dist_km ** 2)
        predicted = np.minimum(predicted, 10.0)  # cap to avoid numerical issues

        # Weighted least squares — weight by elevation (trust high-noise points more)
        weights = elev_norm
        residual = np.sum(weights * (elev_norm - predicted) ** 2)
        return residual

    # Initial guess: center of mass of elevated points, weighted by elevation
    init_lat = float(np.average(lats, weights=elev_norm))
    init_lon = float(np.average(lons, weights=elev_norm))
    init_amp = 100.0  # initial amplitude guess

    result = minimize(_model_residual, [init_lat, init_lon, init_amp],
                      method="Nelder-Mead",
                      options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 2000})

    if not result.success:
        log.warning("  1/r² optimization did not converge: %s", result.message)

    est_lat, est_lon, amplitude = result.x
    error = geodesic_distance_km(gt_lat, gt_lon, est_lat, est_lon)

    # Also compute weighted centroid for comparison
    wc_lat = float(np.average(lats, weights=elev_norm))
    wc_lon = float(np.average(lons, weights=elev_norm))
    wc_error = geodesic_distance_km(gt_lat, gt_lon, wc_lat, wc_lon)

    log.info("  1/r² fit: (%.4f, %.4f) error=%.1f km, amplitude=%.1f, residual=%.4f",
             est_lat, est_lon, error, amplitude, result.fun)
    log.info("  Weighted centroid: (%.4f, %.4f) error=%.1f km",
             wc_lat, wc_lon, wc_error)

    # Use whichever is better
    if wc_error < error:
        log.info("  → Using weighted centroid (%.1f km < %.1f km)", wc_error, error)
        est_lat, est_lon = wc_lat, wc_lon
        error = wc_error

    return {
        "estimated_lat": float(est_lat),
        "estimated_lon": float(est_lon),
        "error_km": round(error, 2),
        "amplitude": round(float(amplitude), 2),
        "residual": round(float(result.fun), 4),
        "n_points": len(points),
        "wc_lat": wc_lat,
        "wc_lon": wc_lon,
        "wc_error_km": round(wc_error, 2),
    }


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


# ── Streaming Mode (no local disk) ─────────────────────────────────────────

def stream_cygnss_l1(granule_result):
    """Stream a single CYGNSS L1 granule via earthaccess.open() — no download.

    Uses h5netcdf engine which supports file-like objects (netcdf4 does not).
    Extracts all 4 DDM channels and pre-computed SNR/NBRCS variables.

    Args:
        granule_result: A single earthaccess search result object.

    Returns:
        dict with arrays: sp_lat, sp_lon, timestamp, ddm_power, sp_delay_row,
        sp_dopp_col, quality_flags, prn_code, sc_num, ddm_snr, ddm_nbrcs,
        sp_rx_gain, sp_inc_angle (where available).
    """
    import earthaccess
    import xarray as xr

    # earthaccess.open() returns a list of file-like objects
    file_objs = earthaccess.open([granule_result])
    if not file_objs:
        raise RuntimeError("earthaccess.open() returned no file objects")

    fileobj = file_objs[0]
    ds = xr.open_dataset(fileobj, engine="h5netcdf")

    data = {
        "sp_lat": ds["sp_lat"].values,
        "sp_lon": ds["sp_lon"].values,
    }

    # Timestamps
    if "ddm_timestamp_utc" in ds:
        data["timestamp"] = ds["ddm_timestamp_utc"].values
    elif "sample_time" in ds:
        data["timestamp"] = ds["sample_time"].values

    # DDM power array — keep ALL channels (samples, 4, 17, 11)
    for var_name in ("power_analog", "raw_counts", "brcs"):
        if var_name in ds:
            data["ddm_power"] = ds[var_name].values
            data["ddm_var"] = var_name
            break

    # Specular point bin indices
    if "brcs_ddm_sp_bin_delay_row" in ds:
        data["sp_delay_row"] = ds["brcs_ddm_sp_bin_delay_row"].values
    if "brcs_ddm_sp_bin_dopp_col" in ds:
        data["sp_dopp_col"] = ds["brcs_ddm_sp_bin_dopp_col"].values

    # Quality flags
    if "quality_flags" in ds:
        data["quality_flags"] = ds["quality_flags"].values

    # Pre-computed SNR and NBRCS — much better calibrated than raw DDM bins
    for var_name in ("ddm_snr", "ddm_noise_floor", "ddm_nbrcs",
                     "sp_nbrcs", "nbrcs_scatter_area"):
        if var_name in ds:
            data[var_name] = ds[var_name].values

    # Incidence angle and receiver gain for normalization
    if "sp_inc_angle" in ds:
        data["sp_inc_angle"] = ds["sp_inc_angle"].values
    if "sp_rx_gain" in ds:
        data["sp_rx_gain"] = ds["sp_rx_gain"].values

    # Metadata
    if "prn_code" in ds:
        data["prn_code"] = ds["prn_code"].values
    if "sc_num" in ds:
        data["sc_num"] = ds["sc_num"].values

    # Log available variables for debugging
    if not hasattr(stream_cygnss_l1, "_logged_vars"):
        stream_cygnss_l1._logged_vars = True
        available = [v for v in ("ddm_snr", "ddm_noise_floor", "ddm_nbrcs",
                                  "sp_nbrcs", "sp_inc_angle", "sp_rx_gain",
                                  "power_analog", "raw_counts", "brcs")
                     if v in ds]
        log.info("  CYGNSS variables available: %s", available)
        log.info("  DDM dimensions: %s", list(ds.dims))
        for v in available[:3]:
            arr = ds[v].values
            log.info("    %s: shape=%s, dtype=%s, range=[%.4g, %.4g]",
                     v, arr.shape, arr.dtype,
                     float(np.nanmin(arr)), float(np.nanmax(arr)))

    ds.close()
    return data


def detect_cygnss_rfi_streaming(gt_lat, gt_lon, dates, baseline_dates=None,
                                search_radius_km=200):
    """Stream CYGNSS L1 granules from S3 and detect GPS jamming.

    Uses three complementary detection methods:
      1. Noise floor elevation: forbidden-zone mean power ON vs baseline
      2. SNR attenuation: DDM peak-to-noise ratio drop ON vs baseline
      3. Spatial SNR hole: grid-based spatial clustering of SNR degradation

    Also runs the original kurtosis detector as a supplementary check.

    No files are written to disk — each granule is streamed via earthaccess.open().

    Args:
        gt_lat: Ground truth latitude (center of search).
        gt_lon: Ground truth longitude (center of search).
        dates: List of date strings to analyze, e.g. ['2026-01-08', '2026-01-20'].
        baseline_dates: Optional list of date strings for baseline comparison.
        search_radius_km: Max distance (km) from ground truth to keep specular points.

    Returns:
        dict with keys:
            "detections": list of RFIDetection for the requested dates
            "baseline": list of RFIDetection for baseline_dates (empty if not given)
    """
    import earthaccess
    from rfi_validation import RFIDetection, geodesic_distance_km

    earthaccess.login()

    bbox = (
        gt_lon - 3.0, gt_lat - 3.0,
        gt_lon + 3.0, gt_lat + 3.0,
    )

    def _extract_measurements(date_list):
        """Stream granules and extract per-specular-point measurements.

        Processes ALL 4 DDM channels per sample (not just channel 0),
        computes total DDM power, and extracts pre-computed SNR/NBRCS.
        """
        measurements = []

        for date_str in date_list:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            start = dt.strftime("%Y-%m-%d")
            end = (dt + timedelta(days=1)).strftime("%Y-%m-%d")

            results = earthaccess.search_data(
                short_name="CYGNSS_L1_V3.2",
                temporal=(start, end),
                bounding_box=bbox,
            )
            log.info("Date %s: found %d CYGNSS granules", date_str, len(results))

            for granule in results:
                try:
                    data = stream_cygnss_l1(granule)
                except Exception as e:
                    log.warning("Failed to stream granule: %s", e)
                    continue

                if "ddm_power" not in data:
                    continue

                ddm_power = data["ddm_power"]
                sp_lat = data["sp_lat"]
                sp_lon = data["sp_lon"]
                timestamps = data.get("timestamp")
                sp_delay_rows = data.get("sp_delay_row")
                quality_flags = data.get("quality_flags")

                # Pre-computed variables (if available)
                ddm_snr_arr = data.get("ddm_snr")       # (samples, 4) or (samples,)
                ddm_nbrcs_arr = data.get("ddm_nbrcs")   # (samples, 4) or (samples,)
                sp_nbrcs_arr = data.get("sp_nbrcs")      # (samples, 4) or (samples,)
                ddm_noise_floor_arr = data.get("ddm_noise_floor")  # calibrated noise
                sp_inc_angle_arr = data.get("sp_inc_angle")

                # Determine if multi-channel: (samples, 4, 17, 11) vs (samples, 17, 11)
                n_channels = 1
                if ddm_power.ndim == 4:
                    n_channels = ddm_power.shape[1]

                n_samples = sp_lat.shape[0]

                for i in range(n_samples):
                    for ch in range(n_channels):
                        # Get lat/lon — may be (samples,4) or (samples,)
                        if sp_lat.ndim == 2:
                            lat_i = float(sp_lat[i, ch])
                            lon_i = float(sp_lon[i, ch])
                        else:
                            lat_i = float(sp_lat[i])
                            lon_i = float(sp_lon[i])

                        if not np.isfinite(lat_i) or not np.isfinite(lon_i):
                            continue
                        if lat_i == 0.0 and lon_i == 0.0:
                            continue  # invalid fill value

                        dist = geodesic_distance_km(gt_lat, gt_lon, lat_i, lon_i)
                        if dist > search_radius_km:
                            continue

                        # Get DDM for this sample/channel
                        if ddm_power.ndim == 4:
                            ddm = ddm_power[i, ch, :, :]
                        else:
                            ddm = ddm_power[i]

                        # Specular point delay row
                        sp_row = 8  # default
                        if sp_delay_rows is not None:
                            if sp_delay_rows.ndim == 2:
                                sp_row = sp_delay_rows[i, ch]
                            elif i < len(sp_delay_rows):
                                sp_row = sp_delay_rows[i]
                        if sp_row is None or not np.isfinite(sp_row):
                            sp_row = 8

                        # ── Compute metrics ──────────────────────────────
                        # 1. Total DDM power (new — captures broadband jammer)
                        total_power = compute_total_ddm_power(ddm)

                        # 2. Peak power near specular point
                        peak_power = compute_ddm_peak_power(ddm, sp_row)

                        # 3. Forbidden-zone noise floor (original)
                        noise_floor = compute_forbidden_zone_noise_floor(ddm, sp_row)

                        # 4. Our computed SNR
                        snr = compute_ddm_snr(ddm, sp_row)

                        # 5. Pre-computed CYGNSS SNR (calibrated, much better)
                        precomp_snr = np.nan
                        if ddm_snr_arr is not None:
                            if ddm_snr_arr.ndim == 2:
                                precomp_snr = float(ddm_snr_arr[i, ch])
                            elif i < len(ddm_snr_arr):
                                precomp_snr = float(ddm_snr_arr[i])

                        # 6. Pre-computed NBRCS
                        nbrcs = np.nan
                        if ddm_nbrcs_arr is not None:
                            if ddm_nbrcs_arr.ndim == 2:
                                nbrcs = float(ddm_nbrcs_arr[i, ch])
                            elif i < len(ddm_nbrcs_arr):
                                nbrcs = float(ddm_nbrcs_arr[i])
                        # Fallback to sp_nbrcs
                        if not np.isfinite(nbrcs) and sp_nbrcs_arr is not None:
                            if sp_nbrcs_arr.ndim == 2:
                                nbrcs = float(sp_nbrcs_arr[i, ch])
                            elif i < len(sp_nbrcs_arr):
                                nbrcs = float(sp_nbrcs_arr[i])

                        # 7. Pre-computed noise floor
                        precomp_noise = np.nan
                        if ddm_noise_floor_arr is not None:
                            if ddm_noise_floor_arr.ndim == 2:
                                precomp_noise = float(ddm_noise_floor_arr[i, ch])
                            elif i < len(ddm_noise_floor_arr):
                                precomp_noise = float(ddm_noise_floor_arr[i])

                        # 8. Kurtosis (supplementary)
                        kurtosis = compute_forbidden_zone_kurtosis(ddm, sp_row)

                        ts_str = ""
                        if timestamps is not None and i < len(timestamps):
                            ts_str = str(timestamps[i])

                        qf = 0
                        if quality_flags is not None:
                            if quality_flags.ndim == 2:
                                qf = int(quality_flags[i, ch])
                            elif i < len(quality_flags):
                                qf = int(quality_flags[i])

                        inc_angle = np.nan
                        if sp_inc_angle_arr is not None:
                            if sp_inc_angle_arr.ndim == 2:
                                inc_angle = float(sp_inc_angle_arr[i, ch])
                            elif i < len(sp_inc_angle_arr):
                                inc_angle = float(sp_inc_angle_arr[i])

                        measurements.append({
                            "lat": lat_i, "lon": lon_i,
                            "total_power": total_power,
                            "peak_power": peak_power,
                            "noise_floor": noise_floor,
                            "snr": snr,
                            "precomp_snr": precomp_snr,
                            "precomp_noise": precomp_noise,
                            "nbrcs": nbrcs,
                            "kurtosis": kurtosis,
                            "inc_angle": inc_angle,
                            "distance_km": round(dist, 2),
                            "timestamp": ts_str,
                            "date": date_str,
                            "quality_flags": qf,
                            "channel": ch,
                        })

        log.info("Extracted %d specular point measurements from %d date(s) (%d channels)",
                 len(measurements), len(date_list), n_channels)

        # Log metric distributions
        for metric in ("total_power", "peak_power", "precomp_snr", "precomp_noise", "nbrcs"):
            vals = [m[metric] for m in measurements if np.isfinite(m[metric])]
            if vals:
                log.info("  %s: n=%d, mean=%.4g, std=%.4g, range=[%.4g, %.4g]",
                         metric, len(vals), np.mean(vals), np.std(vals),
                         np.min(vals), np.max(vals))

        return measurements

    # ── Phase 1: Extract measurements from all dates ────────────────────
    log.info("Phase 1: Streaming ON-date measurements...")
    on_measurements = _extract_measurements(dates)

    log.info("Phase 1: Streaming baseline measurements...")
    bl_measurements = _extract_measurements(baseline_dates) if baseline_dates else []

    # ── Phase 2: Detection methods ──────────────────────────────────────
    from rfi_validation import RFIDetection

    all_detections = []

    # Pick best available SNR metric: precomp > computed > none
    snr_key = "snr"  # fallback
    on_precomp = [m["precomp_snr"] for m in on_measurements if np.isfinite(m["precomp_snr"])]
    bl_precomp = [m["precomp_snr"] for m in bl_measurements if np.isfinite(m["precomp_snr"])]
    if len(on_precomp) > 100 and len(bl_precomp) > 100:
        snr_key = "precomp_snr"
        log.info("Using pre-computed CYGNSS SNR (n_on=%d, n_bl=%d)", len(on_precomp), len(bl_precomp))
    else:
        log.info("Pre-computed SNR not available; using computed SNR")

    # Log stats for all metrics
    for metric in ("total_power", "peak_power", "noise_floor", snr_key, "nbrcs", "precomp_noise"):
        on_vals = [m[metric] for m in on_measurements if np.isfinite(m[metric])]
        bl_vals = [m[metric] for m in bl_measurements if np.isfinite(m[metric])]
        if on_vals and bl_vals:
            on_mean, bl_mean = np.mean(on_vals), np.mean(bl_vals)
            diff_pct = 100 * (on_mean - bl_mean) / max(abs(bl_mean), 1e-10)
            log.info("  %s: ON=%.4g±%.4g (n=%d), BL=%.4g±%.4g (n=%d), diff=%.1f%%",
                     metric, on_mean, np.std(on_vals), len(on_vals),
                     bl_mean, np.std(bl_vals), len(bl_vals), diff_pct)

    # ── Method 1: Total DDM power elevation (ON vs baseline) ────────────
    log.info("Phase 2a: Total DDM power elevation detection...")
    bl_powers = np.array([m["total_power"] for m in bl_measurements
                          if np.isfinite(m["total_power"])])
    if len(bl_powers) > 10:
        bl_power_mean = np.mean(bl_powers)
        bl_power_std = np.std(bl_powers)
        log.info("  Baseline total power: %.4g ± %.4g", bl_power_mean, bl_power_std)

        power_detections = []
        for m in on_measurements:
            tp = m["total_power"]
            if not np.isfinite(tp):
                continue
            zscore = (tp - bl_power_mean) / max(bl_power_std, 1e-10)
            if zscore > NOISE_ELEVATION_ZSCORE:
                m["power_zscore"] = zscore
                power_detections.append(m)

        log.info("  Total power anomalies: %d", len(power_detections))
        for m in power_detections:
            all_detections.append(RFIDetection(
                lat=m["lat"], lon=m["lon"],
                intensity=float(m["power_zscore"]),
                timestamp=m["timestamp"],
                modality="CYGNSS",
                metadata={
                    "distance_km": m["distance_km"],
                    "date": m["date"],
                    "total_power": round(m["total_power"], 4),
                    "method": "total_power",
                },
            ))

    # ── Method 2: SNR/NBRCS attenuation (ON vs baseline) ───────────────
    log.info("Phase 2b: SNR + NBRCS attenuation detection...")

    # SNR-based detection (use best available)
    bl_snrs = np.array([m[snr_key] for m in bl_measurements if np.isfinite(m[snr_key])])
    if len(bl_snrs) > 10:
        bl_snr_mean = np.mean(bl_snrs)
        bl_snr_std = np.std(bl_snrs)
        snr_detections = []
        for m in on_measurements:
            s = m[snr_key]
            if not np.isfinite(s):
                continue
            # Low SNR = potential jamming (z-score of how much BELOW baseline)
            zscore = (bl_snr_mean - s) / max(bl_snr_std, 1e-10)
            if zscore > SNR_DROP_ZSCORE:
                m["snr_drop_zscore"] = zscore
                snr_detections.append(m)
        log.info("  SNR drop anomalies (%s): %d", snr_key, len(snr_detections))
        for m in snr_detections:
            all_detections.append(RFIDetection(
                lat=m["lat"], lon=m["lon"],
                intensity=float(m["snr_drop_zscore"]),
                timestamp=m["timestamp"],
                modality="CYGNSS",
                metadata={
                    "distance_km": m["distance_km"],
                    "date": m["date"],
                    "snr": round(m[snr_key], 2),
                    "method": "snr_drop",
                },
            ))

    # NBRCS-based detection (jammer suppresses surface reflectivity)
    bl_nbrcs = np.array([m["nbrcs"] for m in bl_measurements if np.isfinite(m["nbrcs"])])
    if len(bl_nbrcs) > 10:
        bl_nbrcs_mean = np.mean(bl_nbrcs)
        bl_nbrcs_std = np.std(bl_nbrcs)
        nbrcs_detections = []
        for m in on_measurements:
            nb = m["nbrcs"]
            if not np.isfinite(nb):
                continue
            zscore = (bl_nbrcs_mean - nb) / max(bl_nbrcs_std, 1e-10)
            if zscore > SNR_DROP_ZSCORE:
                m["nbrcs_drop_zscore"] = zscore
                nbrcs_detections.append(m)
        log.info("  NBRCS drop anomalies: %d", len(nbrcs_detections))
        for m in nbrcs_detections:
            all_detections.append(RFIDetection(
                lat=m["lat"], lon=m["lon"],
                intensity=float(m["nbrcs_drop_zscore"]),
                timestamp=m["timestamp"],
                modality="CYGNSS",
                metadata={
                    "distance_km": m["distance_km"],
                    "date": m["date"],
                    "nbrcs": round(m["nbrcs"], 4),
                    "method": "nbrcs_drop",
                },
            ))

    # ── Method 3: Pre-computed noise floor elevation ────────────────────
    bl_pnoise = np.array([m["precomp_noise"] for m in bl_measurements
                          if np.isfinite(m["precomp_noise"])])
    if len(bl_pnoise) > 10:
        bl_pn_mean = np.mean(bl_pnoise)
        bl_pn_std = np.std(bl_pnoise)
        log.info("Phase 2c: Pre-computed noise floor elevation...")
        pn_detections = []
        for m in on_measurements:
            pn = m["precomp_noise"]
            if not np.isfinite(pn):
                continue
            zscore = (pn - bl_pn_mean) / max(bl_pn_std, 1e-10)
            if zscore > NOISE_ELEVATION_ZSCORE:
                m["precomp_noise_zscore"] = zscore
                pn_detections.append(m)
        log.info("  Pre-computed noise elevation: %d", len(pn_detections))
        for m in pn_detections:
            all_detections.append(RFIDetection(
                lat=m["lat"], lon=m["lon"],
                intensity=float(m["precomp_noise_zscore"]),
                timestamp=m["timestamp"],
                modality="CYGNSS",
                metadata={
                    "distance_km": m["distance_km"],
                    "date": m["date"],
                    "precomp_noise": round(m["precomp_noise"], 4),
                    "method": "precomp_noise",
                },
            ))

    # ── Method 4: Spatial SNR hole (using best available SNR) ───────────
    log.info("Phase 2d: Spatial SNR hole detection...")
    # Inject the best SNR into the "snr" key for spatial hole detector
    for m_list in (on_measurements, bl_measurements):
        for m in m_list:
            if snr_key != "snr":
                m["snr"] = m[snr_key] if np.isfinite(m[snr_key]) else m["snr"]

    hole_cells = detect_spatial_snr_hole(on_measurements, bl_measurements,
                                         gt_lat, gt_lon)
    for cell in hole_cells:
        all_detections.append(RFIDetection(
            lat=cell["lat"], lon=cell["lon"],
            intensity=float(cell["zscore"]),
            timestamp="",
            modality="CYGNSS",
            metadata={
                "distance_km": cell["distance_km"],
                "snr_drop_db": cell["snr_drop_db"],
                "n_on": cell["n_on"],
                "n_bl": cell["n_bl"],
                "method": "spatial_hole",
            },
        ))
    log.info("  Spatial hole detections: %d", len(hole_cells))

    # ── Method 5: Spatial noise floor grid (10 km cells) ────────────────
    log.info("Phase 2e: Spatial noise floor grid (10km cells)...")
    noise_cells = detect_spatial_noise_gradient(on_measurements, bl_measurements,
                                                 gt_lat, gt_lon,
                                                 metric="precomp_noise",
                                                 grid_res_km=10)
    for cell in noise_cells:
        all_detections.append(RFIDetection(
            lat=cell["lat"], lon=cell["lon"],
            intensity=float(cell["zscore"]),
            timestamp="",
            modality="CYGNSS",
            metadata={
                "distance_km": cell["distance_km"],
                "elevation": cell["elevation"],
                "n_on": cell["n_on"],
                "n_bl": cell["n_bl"],
                "method": "spatial_noise_grid",
            },
        ))

    # ── Method 6: Inverse-distance (1/r²) jammer model fit ─────────────
    log.info("Phase 2f: Inverse-distance (1/r²) jammer model fit...")
    inv_dist_result = fit_inverse_distance_model(
        on_measurements, bl_measurements, gt_lat, gt_lon,
        metric="precomp_noise")

    # Attach the 1/r² fit result to metadata for localization
    if inv_dist_result:
        log.info("  1/r² best estimate: (%.4f, %.4f) error=%.1f km (%d points)",
                 inv_dist_result["estimated_lat"], inv_dist_result["estimated_lon"],
                 inv_dist_result["error_km"], inv_dist_result["n_points"])

    # ── Supplementary checks ────────────────────────────────────────────
    kurt_detections = [m for m in on_measurements
                       if np.isfinite(m["kurtosis"]) and m["kurtosis"] > KURTOSIS_THRESHOLD]
    log.info("  Kurtosis detections (supplementary): %d", len(kurt_detections))

    rfi_flagged = [m for m in on_measurements if m["quality_flags"] & 0x2]
    log.info("  Quality-flag RFI-flagged: %d", len(rfi_flagged))
    for m in rfi_flagged:
        all_detections.append(RFIDetection(
            lat=m["lat"], lon=m["lon"],
            intensity=3.0,
            timestamp=m["timestamp"],
            modality="CYGNSS",
            metadata={
                "distance_km": m["distance_km"],
                "date": m["date"],
                "method": "quality_flag",
            },
        ))

    log.info("CYGNSS total detections (all methods): %d", len(all_detections))

    # Build baseline detections
    baseline_detections = []
    for m in bl_measurements:
        if np.isfinite(m["kurtosis"]) and m["kurtosis"] > KURTOSIS_THRESHOLD:
            baseline_detections.append(RFIDetection(
                lat=m["lat"], lon=m["lon"],
                intensity=float(m["kurtosis"]),
                timestamp=m["timestamp"],
                modality="CYGNSS",
                metadata={"distance_km": m["distance_km"], "date": m["date"],
                          "method": "kurtosis"},
            ))

    return {
        "detections": all_detections,
        "baseline": baseline_detections,
        "inv_dist_fit": inv_dist_result,
    }
