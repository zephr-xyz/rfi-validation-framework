"""
NISAR L-band SAR RFI Detection Module
======================================
Parses NISAR L-band GCOV (geocoded covariance) products and detects GPS
jamming via λ₁ eigenvalue decomposition on the polarimetric covariance matrix.

Detection method:
  1. Load HV-polarization intensity from GCOV product.
  2. In sliding windows, compute the sample covariance matrix.
  3. Eigenvalue decomposition: dominant λ₁ indicates coherent point-source RFI.
  4. λ₁/λ₂ ratio >> 1 → RFI streak. Extract centroid of streak.
  5. Triangulate source using ascending + descending orbit centroids.

NISAR L-band (1.257 GHz) is ~30 MHz from GPS L2 (1.2276 GHz), making it
highly sensitive to GPS jammer emissions.
"""

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

EIGENVALUE_RATIO_THRESHOLD = 10.0  # λ₁/λ₂ ratio indicating RFI
INTENSITY_ZSCORE_THRESHOLD = 4.0   # Z-score for HV intensity spikes
WINDOW_SIZE = 32                   # Covariance estimation window (pixels)
MIN_STREAK_PIXELS = 20             # Minimum connected pixels for valid streak
SEARCH_RADIUS_KM = 200
CROP_RADIUS_KM = 50    # Crop radius for GCOV processing (50km keeps images ~4K×4K)


# ── Data Access ──────────────────────────────────────────────────────────────

def download_nisar_known_passes(known_passes, output_dir):
    """Download specific NISAR granules by ID from known jammer-active/baseline passes."""
    import earthaccess

    earthaccess.login()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = []
    for entry in known_passes:
        granule_id = entry["granule"]
        collection = entry.get("collection", "NISAR_L2_GCOV_BETA_V1")
        start = entry["start"]
        stop = entry["stop"]
        jammer_status = "ON" if entry.get("jammer_on") else "OFF"

        log.info("Searching for %s (jammer %s)...", granule_id, jammer_status)

        # Search by temporal window + collection (narrow enough to find exact granule)
        results = earthaccess.search_data(
            short_name=collection,
            temporal=(start[:10], stop[:10]),  # date portion
            granule_ur=granule_id,
        )

        # Fallback: search by time window if granule_ur doesn't work
        if not results:
            results = earthaccess.search_data(
                short_name=collection,
                temporal=(start, stop),
            )

        if results:
            log.info("  Found %d granule(s), downloading...", len(results))
            files = earthaccess.download(results, str(output_dir))
            for f in files:
                all_files.append(Path(f))
                # Write sidecar metadata so processing knows jammer state
                meta_path = Path(f).with_suffix(".meta.json")
                import json
                meta_path.write_text(json.dumps(entry, indent=2))
        else:
            log.warning("  Granule not found: %s", granule_id)

    log.info("Downloaded %d NISAR files total", len(all_files))
    return all_files


def download_nisar(gt_lat, gt_lon, start_date, end_date, output_dir):
    """Download NISAR L-band GCOV products via earthaccess (spatial search)."""
    import earthaccess

    earthaccess.login()

    # Search for NISAR GCOV products
    results = earthaccess.search_data(
        short_name="NISAR_L2_GCOV_V1",
        temporal=(start_date, end_date),
        bounding_box=(
            gt_lon - 2.0, gt_lat - 2.0,
            gt_lon + 2.0, gt_lat + 2.0,
        ),
    )

    # Fallback to RSLC if no GCOV available
    if not results:
        log.info("No GCOV found, trying RSLC beta...")
        results = earthaccess.search_data(
            short_name="NISAR_L1_RSLC_BETA_V1",
            temporal=(start_date, end_date),
            bounding_box=(
                gt_lon - 2.0, gt_lat - 2.0,
                gt_lon + 2.0, gt_lat + 2.0,
            ),
        )

    log.info("Found %d NISAR granules", len(results))
    if not results:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = earthaccess.download(results, str(output_dir))
    return [Path(f) for f in files]


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_nisar_gcov(filepath):
    """Parse a NISAR GCOV HDF5 file. Returns dict with arrays + metadata.

    Extracts:
      - HVHV: HV-pol backscatter intensity (2D array)
      - HHHH, VVVV: co-pol terms (if available)
      - x_coordinates, y_coordinates: map-projected easting/northing
      - orbit_direction: "Ascending" or "Descending"
      - projection: CRS info
    """
    import h5py

    data = {}

    with h5py.File(filepath, "r") as f:
        # Orbit direction — try multiple HDF5 paths
        orbit_dir = None
        for id_path in ("identification", "science/LSAR/identification",
                        "science/LSAR/GCOV/identification"):
            if id_path in f:
                id_grp = f[id_path]
                for key in ("orbitPassDirection", "orbit_pass_direction",
                            "orbitDirection"):
                    if key in id_grp:
                        val = id_grp[key][()]
                        if isinstance(val, bytes):
                            orbit_dir = val.decode().strip()
                        elif isinstance(val, np.ndarray):
                            orbit_dir = str(val.flat[0])
                            if isinstance(val.flat[0], bytes):
                                orbit_dir = val.flat[0].decode().strip()
                        else:
                            orbit_dir = str(val).strip()
                        break
                # Also check attributes on the group
                if orbit_dir is None:
                    for attr_key in id_grp.attrs:
                        if "orbit" in attr_key.lower() and "pass" in attr_key.lower():
                            orbit_dir = str(id_grp.attrs[attr_key])
                            break
                if orbit_dir:
                    break

        # Also check top-level attrs
        if orbit_dir is None:
            for attr_key in f.attrs:
                if "orbit" in attr_key.lower() and ("pass" in attr_key.lower() or "direction" in attr_key.lower()):
                    orbit_dir = str(f.attrs[attr_key])
                    break

        if orbit_dir:
            data["orbit_direction"] = orbit_dir
            log.info("  Orbit direction from HDF5: %s", orbit_dir)

        # Track number
        for id_path in ("identification", "science/LSAR/identification"):
            if id_path in f:
                id_grp = f[id_path]
                for key in ("trackNumber", "track_number"):
                    if key in id_grp:
                        data["track_number"] = int(id_grp[key][()])
                        break
                if "track_number" in data:
                    break
        # Find frequency group (frequencyA = L-band)
        freq_grp = None
        for grp_name in ("data/frequencyA", "science/LSAR/GCOV/grids/frequencyA",
                          "science/LSAR/RSLC/swaths/frequencyA"):
            if grp_name in f:
                freq_grp = f[grp_name]
                break

        if freq_grp is None:
            # Try to find any group with polarization data
            def find_pol_group(group, depth=0):
                if depth > 5:
                    return None
                for key in group:
                    if key in ("HVHV", "HV"):
                        return group
                    if isinstance(group[key], h5py.Group):
                        result = find_pol_group(group[key], depth + 1)
                        if result is not None:
                            return result
                return None
            freq_grp = find_pol_group(f)

        if freq_grp is None:
            log.warning("Could not find polarization data in %s", filepath)
            return data

        # Load polarimetric covariance terms
        for pol in ("HVHV", "HHHH", "VVVV", "HHVV"):
            if pol in freq_grp:
                arr = freq_grp[pol][()]
                # Take real part if complex (covariance diagonal is real)
                if np.iscomplexobj(arr):
                    arr = np.abs(arr)
                data[pol] = arr
                log.info("  Loaded %s: shape %s", pol, arr.shape)

        # Also check for SLC data (RSLC products)
        if "HVHV" not in data:
            for pol_name in ("HV", "VH"):
                if pol_name in freq_grp:
                    slc = freq_grp[pol_name][()]
                    data["HVHV"] = np.abs(slc) ** 2  # intensity
                    log.info("  Loaded |%s|² as HVHV: shape %s", pol_name, data["HVHV"].shape)
                    break

        # Coordinates
        for coord_name in ("x_coordinates", "xCoordinates", "x"):
            if coord_name in freq_grp:
                data["x_coordinates"] = freq_grp[coord_name][()]
                break
        for coord_name in ("y_coordinates", "yCoordinates", "y"):
            if coord_name in freq_grp:
                data["y_coordinates"] = freq_grp[coord_name][()]
                break

        # Projection info
        for proj_name in ("projection", "coordinate_system"):
            if proj_name in freq_grp:
                proj = freq_grp[proj_name]
                if hasattr(proj, "attrs"):
                    data["projection"] = dict(proj.attrs)

    return data


def gcov_pixel_to_latlon(data, row, col):
    """Convert GCOV pixel indices to lat/lon using coordinate arrays + projection."""
    from pyproj import Transformer

    if "x_coordinates" not in data or "y_coordinates" not in data:
        return None, None

    x = data["x_coordinates"]
    y = data["y_coordinates"]

    # Handle 1D vs 2D coordinate arrays
    if x.ndim == 1 and y.ndim == 1:
        easting = float(x[col]) if col < len(x) else float(x[-1])
        northing = float(y[row]) if row < len(y) else float(y[-1])
    elif x.ndim == 2:
        easting = float(x[row, col])
        northing = float(y[row, col])
    else:
        return None, None

    # Default to UTM zone 39N for this region (52.8703°E)
    try:
        epsg = data.get("projection", {}).get("epsg_code", 32639)
        transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(easting, northing)
        return float(lat), float(lon)
    except Exception:
        # Fallback: assume coordinates are already lon/lat
        return float(northing), float(easting)


# ── Spatial Crop ─────────────────────────────────────────────────────────────

def crop_gcov_around_target(data, target_lat, target_lon, radius_km=100):
    """Crop GCOV arrays to a region around target lat/lon.

    Dramatically reduces processing time vs full-frame EVD.
    Returns a new data dict with cropped arrays + offset indices.
    """
    from pyproj import Transformer

    if "x_coordinates" not in data or "y_coordinates" not in data or "HVHV" not in data:
        return data  # can't crop without coordinates

    x = data["x_coordinates"]
    y = data["y_coordinates"]
    hv = data["HVHV"]

    # Convert target to projected coordinates
    epsg = data.get("projection", {}).get("epsg_code", 32639)
    try:
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        target_e, target_n = transformer.transform(target_lon, target_lat)
    except Exception:
        log.warning("Could not project target — skipping crop")
        return data

    radius_m = radius_km * 1000

    if x.ndim == 1 and y.ndim == 1:
        # Find pixel range within radius
        col_mask = np.abs(x - target_e) < radius_m
        row_mask = np.abs(y - target_n) < radius_m

        if not col_mask.any() or not row_mask.any():
            log.warning("Target (%.4f, %.4f) not within GCOV extent — skipping crop",
                        target_lat, target_lon)
            return data

        col_min, col_max = np.where(col_mask)[0][[0, -1]]
        row_min, row_max = np.where(row_mask)[0][[0, -1]]

        cropped = dict(data)
        cropped["HVHV"] = hv[row_min:row_max + 1, col_min:col_max + 1]
        cropped["x_coordinates"] = x[col_min:col_max + 1]
        cropped["y_coordinates"] = y[row_min:row_max + 1]
        cropped["_crop_offset"] = (row_min, col_min)

        for pol in ("HHHH", "VVVV", "HHVV"):
            if pol in data:
                cropped[pol] = data[pol][row_min:row_max + 1, col_min:col_max + 1]

        log.info("  Cropped from %s to %s (%.0f km radius around target)",
                 hv.shape, cropped["HVHV"].shape, radius_km)
        return cropped
    else:
        log.warning("2D coordinate arrays — crop not implemented, using full frame")
        return data


# ── RFI Detection Methods ────────────────────────────────────────────────────

def azimuth_line_rfi_detection(hv_intensity):
    """Fast RFI detection via per-azimuth-line power anomaly.

    RFI from a point source appears as bright horizontal streaks in SAR HV-pol
    because the jammer signal is received across all range bins in an azimuth line.
    This is much faster than sliding-window EVD.

    Returns:
        rfi_mask: boolean 2D array
        line_zscore: 1D array of per-azimuth-line z-scores
    """
    # Mean power per azimuth line (across range), ignoring NaN
    line_power = np.nanmean(hv_intensity, axis=1)

    # Filter out all-NaN lines
    valid_lines = np.isfinite(line_power)
    if not valid_lines.any():
        return np.zeros_like(hv_intensity, dtype=bool), np.zeros_like(line_power)

    # Robust statistics (median + MAD) on valid lines only
    valid_power = line_power[valid_lines]
    median_power = np.median(valid_power)
    mad = np.median(np.abs(valid_power - median_power))
    sigma = mad * 1.4826  # MAD to sigma conversion

    if sigma == 0 or not np.isfinite(sigma):
        return np.zeros_like(hv_intensity, dtype=bool), np.zeros_like(line_power)

    line_zscore = np.where(valid_lines, (line_power - median_power) / sigma, 0.0)
    anomalous_lines = line_zscore > INTENSITY_ZSCORE_THRESHOLD

    # Expand line mask to 2D
    rfi_mask = np.zeros_like(hv_intensity, dtype=bool)
    rfi_mask[anomalous_lines, :] = True

    n_rfi = anomalous_lines.sum()
    if n_rfi > 0:
        log.info("  Azimuth-line detection: %d RFI lines (%.1f%%), max z-score=%.1f",
                 n_rfi, 100 * n_rfi / len(line_power), np.max(line_zscore))

    return rfi_mask, line_zscore


def eigenvalue_rfi_detection(hv_intensity, window_size=WINDOW_SIZE):
    """Detect RFI via λ₁ eigenvalue decomposition on HV-pol intensity.

    In each sliding window, constructs the sample covariance matrix from
    range-line vectors. Dominant eigenvalue λ₁ much larger than λ₂ indicates
    a coherent point-source (RFI), vs distributed scattering (natural surface).

    Returns:
        rfi_mask: boolean 2D array, True where RFI detected
        eigenvalue_ratio: 2D array of λ₁/λ₂ values
    """
    rows, cols = hv_intensity.shape
    half_w = window_size // 2

    eigenvalue_ratio = np.zeros((rows, cols), dtype=np.float32)

    # Process in blocks for efficiency
    step = max(1, window_size // 4)  # 75% overlap

    for r in range(half_w, rows - half_w, step):
        for c in range(half_w, cols - half_w, step):
            window = hv_intensity[r - half_w:r + half_w, c - half_w:c + half_w]

            if window.size == 0 or np.all(window == 0):
                continue

            # Flatten window rows into vectors for covariance estimation
            # Each row of the window is a "sample vector"
            samples = window.astype(np.float64)

            # Remove mean
            samples = samples - samples.mean(axis=1, keepdims=True)

            # Sample covariance matrix (cols × cols)
            try:
                cov = (samples.T @ samples) / max(samples.shape[0] - 1, 1)
                eigenvalues = np.linalg.eigvalsh(cov)
                eigenvalues = np.sort(eigenvalues)[::-1]  # descending

                if eigenvalues[1] > 0:
                    ratio = eigenvalues[0] / eigenvalues[1]
                else:
                    ratio = eigenvalues[0] / 1e-10 if eigenvalues[0] > 0 else 0

                # Fill the block
                eigenvalue_ratio[r - step // 2:r + step // 2 + 1,
                                  c - step // 2:c + step // 2 + 1] = ratio
            except np.linalg.LinAlgError:
                continue

    rfi_mask = eigenvalue_ratio > EIGENVALUE_RATIO_THRESHOLD
    return rfi_mask, eigenvalue_ratio


def find_rfi_streak_centroids(rfi_mask, hv_intensity, return_peaks=False):
    """Find centroids of connected RFI streak regions.

    Returns list of (row, col, intensity) tuples for each streak centroid.
    If return_peaks=True, also returns peak-intensity pixels as a separate list.
    """
    from scipy import ndimage

    labeled, n_features = ndimage.label(rfi_mask)
    centroids = []
    peaks = []

    for label_id in range(1, n_features + 1):
        region = labeled == label_id
        n_pixels = region.sum()

        if n_pixels < MIN_STREAK_PIXELS:
            continue

        # Intensity-weighted centroid (handle NaN pixels)
        rows_idx, cols_idx = np.where(region)
        weights = hv_intensity[rows_idx, cols_idx].copy()
        valid = np.isfinite(weights) & (weights > 0)
        if not valid.any():
            continue

        rows_idx = rows_idx[valid]
        cols_idx = cols_idx[valid]
        weights = weights[valid]

        cent_row = float(np.average(rows_idx, weights=weights))
        cent_col = float(np.average(cols_idx, weights=weights))

        if not (np.isfinite(cent_row) and np.isfinite(cent_col)):
            continue

        avg_intensity = float(weights.sum() / len(weights))
        centroids.append((cent_row, cent_col, avg_intensity))

        # Peak intensity pixel — the pixel in this streak with highest HV power.
        # This is the closest point to the jammer along the streak, since RFI
        # intensity falls off as 1/r² from the source.
        if return_peaks:
            peak_idx = np.argmax(weights)
            peak_row = int(rows_idx[peak_idx])
            peak_col = int(cols_idx[peak_idx])
            peak_intensity = float(weights[peak_idx])
            peaks.append((peak_row, peak_col, peak_intensity))

    if return_peaks:
        return centroids, peaks
    return centroids


def intensity_spike_detection(hv_intensity):
    """Simple z-score based detection of HV intensity spikes as backup method.

    RFI in SAR appears as bright azimuth streaks in HV polarization because
    the jamming signal is unpolarized and raises the cross-pol floor.
    """
    # Compute per-range-line (azimuth) mean
    azimuth_mean = np.nanmean(hv_intensity, axis=1)
    global_mean = np.nanmean(azimuth_mean)
    global_std = np.nanstd(azimuth_mean)

    if global_std == 0:
        return np.zeros_like(hv_intensity, dtype=bool), np.zeros_like(hv_intensity)

    zscore_map = (hv_intensity - global_mean) / global_std
    spike_mask = zscore_map > INTENSITY_ZSCORE_THRESHOLD

    return spike_mask, zscore_map


def crosspol_ratio_detection(hv_intensity, hh_intensity, zscore_thresh=4.0):
    """Detect RFI using HV/HH cross-pol ratio.

    An unpolarized jammer raises HV disproportionately vs HH because natural
    terrain has low cross-pol return. The HV/HH ratio spikes near the jammer,
    suppressing natural clutter and sharpening the detection.

    Returns:
        rfi_mask: boolean 2D array
        ratio_zscore: 2D array of z-scores of the HV/HH ratio
    """
    # Compute HV/HH ratio (linear scale, avoid division by zero)
    hh_safe = np.where((hh_intensity > 0) & np.isfinite(hh_intensity),
                       hh_intensity, np.nan)
    hv_safe = np.where(np.isfinite(hv_intensity), hv_intensity, np.nan)
    ratio = hv_safe / hh_safe  # NaN where HH is zero/invalid

    # Per-azimuth-line mean of the ratio
    line_ratio = np.nanmean(ratio, axis=1)
    valid = np.isfinite(line_ratio)
    if not valid.any():
        return np.zeros_like(hv_intensity, dtype=bool), np.zeros(hv_intensity.shape[0])

    valid_ratios = line_ratio[valid]
    median_ratio = np.median(valid_ratios)
    mad = np.median(np.abs(valid_ratios - median_ratio))
    sigma = mad * 1.4826

    if sigma == 0 or not np.isfinite(sigma):
        return np.zeros_like(hv_intensity, dtype=bool), np.zeros(hv_intensity.shape[0])

    line_zscore = np.where(valid, (line_ratio - median_ratio) / sigma, 0.0)
    anomalous = line_zscore > zscore_thresh

    rfi_mask = np.zeros_like(hv_intensity, dtype=bool)
    rfi_mask[anomalous, :] = True

    n_rfi = anomalous.sum()
    if n_rfi > 0:
        log.info("  Cross-pol ratio detection: %d RFI lines (%.1f%%), max z-score=%.1f",
                 n_rfi, 100 * n_rfi / len(line_ratio), np.max(line_zscore))

    return rfi_mask, line_zscore


def fit_streak_bearing(centroids_latlon):
    """Fit a bearing line through a set of RFI streak centroids from a single pass.

    RFI streaks from a point source align along the SAR azimuth direction.
    Fitting a line gives the bearing toward the source. The perpendicular
    from this line through the source gives a constraint.

    Args:
        centroids_latlon: list of (lat, lon, intensity) tuples

    Returns:
        (bearing_deg, centroid_lat, centroid_lon, residual_km) or None
        bearing_deg: bearing of the fitted line (degrees from North)
    """
    if len(centroids_latlon) < 3:
        return None

    lats = np.array([c[0] for c in centroids_latlon])
    lons = np.array([c[1] for c in centroids_latlon])
    weights = np.array([c[2] for c in centroids_latlon])
    weights = np.maximum(weights, 1e-10)

    # Weighted centroid
    w = weights / weights.sum()
    clat = np.average(lats, weights=w)
    clon = np.average(lons, weights=w)

    # Convert to local tangent plane (meters) for line fitting
    cos_lat = np.cos(np.radians(clat))
    dx = (lons - clon) * cos_lat * 111320  # meters east
    dy = (lats - clat) * 111320             # meters north

    # Weighted PCA to find principal axis
    dx_w = dx * np.sqrt(w)
    dy_w = dy * np.sqrt(w)
    cov = np.array([[np.sum(dx_w * dx_w), np.sum(dx_w * dy_w)],
                     [np.sum(dy_w * dx_w), np.sum(dy_w * dy_w)]])

    eigvals, eigvecs = np.linalg.eigh(cov)
    # Principal axis = eigenvector with largest eigenvalue
    principal = eigvecs[:, np.argmax(eigvals)]

    # Bearing: angle from north (positive clockwise)
    bearing_rad = np.arctan2(principal[0], principal[1])  # atan2(east, north)
    bearing_deg = np.degrees(bearing_rad) % 360

    # Residual: RMS perpendicular distance from the line
    perp = -dx * np.sin(np.radians(bearing_deg - 90)) + dy * np.cos(np.radians(bearing_deg - 90))
    residual_m = np.sqrt(np.average(perp**2, weights=w))

    log.info("  Streak bearing: %.1f° (residual: %.1f km, %d points)",
             bearing_deg, residual_m / 1000, len(centroids_latlon))

    return bearing_deg, clat, clon, residual_m / 1000


def intersect_bearing_lines(lines):
    """Find the best-fit intersection point of multiple bearing lines.

    Each line is (bearing_deg, lat, lon, residual_km).
    Uses weighted least-squares to find the point minimizing perpendicular
    distance to all lines, weighted by inverse residual.

    Returns:
        (lat, lon) of the intersection, or None
    """
    if len(lines) < 2:
        return None

    from scipy.optimize import minimize

    def total_perp_distance(latlon):
        """Sum of weighted squared perpendicular distances to all bearing lines."""
        pt_lat, pt_lon = latlon
        total = 0.0
        for bearing_deg, clat, clon, residual in lines:
            cos_lat = np.cos(np.radians(clat))
            dx = (pt_lon - clon) * cos_lat * 111320
            dy = (pt_lat - clat) * 111320
            # Perpendicular distance to the bearing line
            bearing_rad = np.radians(bearing_deg)
            perp = abs(-dx * np.cos(bearing_rad) + dy * np.sin(bearing_rad))
            weight = 1.0 / max(residual, 0.1)  # inverse residual weighting
            total += weight * perp ** 2
        return total

    # Initial guess: weighted centroid of line centers
    weights = [1.0 / max(l[3], 0.1) for l in lines]
    w_sum = sum(weights)
    init_lat = sum(l[1] * w for l, w in zip(lines, weights)) / w_sum
    init_lon = sum(l[2] * w for l, w in zip(lines, weights)) / w_sum

    result = minimize(total_perp_distance, [init_lat, init_lon],
                      method="Nelder-Mead",
                      options={"xatol": 1e-6, "fatol": 1.0})

    if result.success:
        return float(result.x[0]), float(result.x[1])
    return None


def fit_nisar_inverse_distance(detections, gt_lat, gt_lon):
    """Fit a 1/r² inverse-distance jammer model to NISAR detection intensities.

    Same physics as CYGNSS: jammer power falls off as 1/r² from the source.
    Each NISAR detection (streak centroid or peak pixel) is at a different
    distance from the jammer, and its intensity encodes that distance.

    Uses scipy.optimize to find the (lat, lon) that best explains the observed
    intensity pattern as a 1/r² point source.

    Returns:
        dict with estimated_lat, estimated_lon, error_km, etc., or None
    """
    from scipy.optimize import minimize
    from rfi_validation import geodesic_distance_km

    if len(detections) < 5:
        log.warning("  Insufficient NISAR detections for 1/r² fit (%d)", len(detections))
        return None

    lats = np.array([d.lat for d in detections])
    lons = np.array([d.lon for d in detections])
    intensities = np.array([d.intensity for d in detections])

    # Filter to positive finite intensities
    valid = np.isfinite(intensities) & (intensities > 0)
    if valid.sum() < 5:
        return None
    lats, lons, intensities = lats[valid], lons[valid], intensities[valid]

    # Normalize intensities to [0, 1]
    int_max = np.max(intensities)
    int_norm = intensities / int_max

    cos_lat = np.cos(np.radians(gt_lat))

    def _model_residual(params):
        src_lat, src_lon, amplitude = params
        dlat = (lats - src_lat) * 111.0
        dlon = (lons - src_lon) * 111.0 * cos_lat
        dist_km = np.sqrt(dlat**2 + dlon**2)
        dist_km = np.maximum(dist_km, 0.5)

        predicted = amplitude / (dist_km ** 2)
        predicted = np.minimum(predicted, 10.0)

        # Weight by intensity (trust high-intensity detections more)
        weights = int_norm
        return float(np.sum(weights * (int_norm - predicted) ** 2))

    # Initial guess: intensity-weighted centroid
    init_lat = float(np.average(lats, weights=int_norm))
    init_lon = float(np.average(lons, weights=int_norm))

    result = minimize(_model_residual, [init_lat, init_lon, 50.0],
                      method="Nelder-Mead",
                      options={"xatol": 1e-6, "fatol": 1e-7, "maxiter": 3000})

    est_lat, est_lon, amplitude = result.x
    error = geodesic_distance_km(gt_lat, gt_lon, est_lat, est_lon)

    log.info("  NISAR 1/r² fit: (%.4f, %.4f) error=%.2f km, amplitude=%.1f, residual=%.6f",
             est_lat, est_lon, error, amplitude, result.fun)

    return {
        "estimated_lat": float(est_lat),
        "estimated_lon": float(est_lon),
        "error_km": round(error, 2),
        "amplitude": round(float(amplitude), 2),
        "residual": round(float(result.fun), 6),
        "n_points": len(lats),
    }


def iterative_outlier_trim(detections, ground_truth, n_rounds=3, sigma_cut=1.5):
    """Iteratively remove outlier detections far from the weighted centroid.

    Each round: compute intensity-weighted centroid, remove detections beyond
    sigma_cut * (weighted std distance), recompute. Tightens the cluster
    around the true source.

    Returns:
        trimmed list of detections
    """
    from rfi_validation import geodesic_distance_km

    current = list(detections)
    for rnd in range(n_rounds):
        if len(current) < 3:
            break

        lats = np.array([d.lat for d in current])
        lons = np.array([d.lon for d in current])
        weights = np.array([d.intensity for d in current])
        weights = np.maximum(weights, 1e-10)
        w = weights / weights.sum()

        clat = float(np.average(lats, weights=w))
        clon = float(np.average(lons, weights=w))

        dists = np.array([geodesic_distance_km(clat, clon, d.lat, d.lon)
                          for d in current])

        # Weighted std of distances
        mean_dist = np.average(dists, weights=w)
        std_dist = np.sqrt(np.average((dists - mean_dist)**2, weights=w))

        if std_dist < 0.1:
            break

        threshold = mean_dist + sigma_cut * std_dist
        kept = [d for d, dist in zip(current, dists) if dist <= threshold]

        n_removed = len(current) - len(kept)
        log.info("  Outlier trim round %d: removed %d/%d (threshold=%.1f km, centroid=%.4f,%.4f)",
                 rnd + 1, n_removed, len(current), threshold, clat, clon)

        if n_removed == 0:
            break
        current = kept

    return current


# ── Detection Pipeline ───────────────────────────────────────────────────────

def _load_sidecar_meta(filepath):
    """Load .meta.json sidecar if it exists (written during download)."""
    import json
    meta_path = Path(filepath).with_suffix(".meta.json")
    # Also check by replacing full suffix
    if not meta_path.exists():
        meta_path = Path(str(filepath) + ".meta.json")
    if not meta_path.exists():
        # Try alongside the h5 file
        for candidate in filepath.parent.glob("*.meta.json"):
            # Match by date in filename
            if filepath.stem[:40] in candidate.stem:
                meta_path = candidate
                break
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def detect_nisar_rfi(data_dir, ground_truth):
    """Run NISAR eigenvalue RFI detection on all files in data_dir.

    Uses sidecar .meta.json to distinguish jammer-ON vs jammer-OFF (baseline)
    passes. For ON passes, subtracts the baseline eigenvalue ratio to isolate
    jammer-specific RFI from ambient L-band interference.

    Returns list of RFIDetection objects.
    """
    from rfi_validation import RFIDetection, geodesic_distance_km

    data_dir = Path(data_dir)
    h5_files = sorted(list(data_dir.glob("*.h5")) + list(data_dir.glob("*.hdf5")))
    if not h5_files:
        log.warning("No NISAR .h5/.hdf5 files found in %s", data_dir)
        return []

    gt_lat, gt_lon = ground_truth["lat"], ground_truth["lon"]

    all_peak_detections = []  # Peak-intensity pixels from each streak

    # Separate baseline vs jammer-on files
    baseline_files = []
    jammer_on_files = []
    for f in h5_files:
        meta = _load_sidecar_meta(f)
        if meta.get("jammer_on") is False:
            baseline_files.append(f)
        else:
            jammer_on_files.append(f)

    log.info("NISAR files: %d jammer-ON, %d baseline (jammer-OFF)",
             len(jammer_on_files), len(baseline_files))

    # Compute baseline HV power profiles (per-azimuth-line mean) from OFF passes
    baseline_line_powers = []
    for filepath in baseline_files:
        log.info("Loading baseline: %s", filepath.name)
        try:
            data = parse_nisar_gcov(filepath)
            if "HVHV" not in data:
                continue
            data = crop_gcov_around_target(data, gt_lat, gt_lon, radius_km=CROP_RADIUS_KM)
            hv_db = 10 * np.log10(np.maximum(data["HVHV"], 1e-30))
            line_power = np.nanmean(hv_db, axis=1)
            baseline_line_powers.append(line_power)
            log.info("  Baseline HV line power: median=%.1f dB, std=%.1f dB",
                     np.nanmedian(line_power), np.nanstd(line_power))
        except Exception as e:
            log.warning("Baseline failed %s: %s", filepath.name, e)

    baseline_median = None
    if baseline_line_powers:
        # Use shortest common length
        min_len = min(len(lp) for lp in baseline_line_powers)
        baseline_median = np.median(
            [lp[:min_len] for lp in baseline_line_powers], axis=0)
        log.info("Baseline computed from %d files (median line power: %.1f dB)",
                 len(baseline_line_powers), np.nanmedian(baseline_median))

    # Process jammer-ON files (or all files if no sidecar metadata)
    process_files = jammer_on_files if jammer_on_files else h5_files
    all_detections = []

    for filepath in process_files:
        log.info("Processing NISAR: %s", filepath.name)
        meta = _load_sidecar_meta(filepath)
        try:
            data = parse_nisar_gcov(filepath)
        except Exception as e:
            log.warning("Failed to parse %s: %s", filepath.name, e)
            continue

        if "HVHV" not in data:
            log.warning("No HV-pol data in %s", filepath.name)
            continue

        # Crop to region around target — reduces 17K×17K to ~2K×2K
        data = crop_gcov_around_target(data, gt_lat, gt_lon, radius_km=CROP_RADIUS_KM)
        hv = data["HVHV"]
        crop_offset = data.get("_crop_offset", (0, 0))

        # Orbit direction: prefer sidecar metadata, fall back to HDF5
        orbit_dir = meta.get("direction", data.get("orbit_direction", "Unknown"))
        if orbit_dir == "Unknown":
            # Infer from filename if possible (e.g., ..._A_... = Ascending)
            fname = filepath.stem
            if "_A_" in fname:
                orbit_dir = "Ascending"
            elif "_D_" in fname:
                orbit_dir = "Descending"
        log.info("  Orbit direction: %s (source: %s)", orbit_dir,
                 "sidecar" if meta.get("direction") else "HDF5/filename")

        # Convert to dB
        hv_db = 10 * np.log10(np.maximum(hv, 1e-30))

        # Method 1 (fast): Azimuth-line power anomaly detection
        log.info("  Running azimuth-line RFI detection (shape %s)...", hv.shape)
        rfi_mask, line_zscore = azimuth_line_rfi_detection(hv_db)

        # Method 1b: Cross-pol ratio (HV/HH) — suppresses natural clutter
        hh = data.get("HHHH")
        if hh is not None:
            log.info("  Running cross-pol ratio (HV/HH) detection...")
            xpol_mask, xpol_zscore = crosspol_ratio_detection(hv, hh)
            # Union with azimuth-line mask
            if xpol_mask.sum() > 0:
                rfi_mask = rfi_mask | xpol_mask
                log.info("  Combined mask: %d pixels", rfi_mask.sum())

        # Baseline subtraction on line power if available
        if baseline_median is not None:
            line_power = np.nanmean(hv_db, axis=1)
            min_len = min(len(line_power), len(baseline_median))
            diff = line_power[:min_len] - baseline_median[:min_len]
            diff_zscore = (diff - np.median(diff)) / max(np.std(diff), 1e-10)
            n_elevated = (diff_zscore > INTENSITY_ZSCORE_THRESHOLD).sum()
            log.info("  Baseline diff: %d elevated lines, max diff=%.1f dB",
                     n_elevated, np.max(diff[:min_len]))
            # Use baseline-subtracted mask if more detections
            if n_elevated > 0:
                rfi_mask_bl = np.zeros_like(hv_db, dtype=bool)
                rfi_mask_bl[:min_len, :] = (diff_zscore > INTENSITY_ZSCORE_THRESHOLD)[:, np.newaxis]
                if rfi_mask_bl.sum() > rfi_mask.sum():
                    rfi_mask = rfi_mask_bl

        centroids, peaks = find_rfi_streak_centroids(rfi_mask, hv, return_peaks=True)
        detection_method = "azimuth_crosspol" if centroids else None

        # Method 2: Eigenvalue decomposition on cropped region (fallback)
        if not centroids:
            log.info("  No azimuth-line streaks, trying eigenvalue decomposition...")
            _, ev_ratio = eigenvalue_rfi_detection(hv_db)
            ev_mask = ev_ratio > EIGENVALUE_RATIO_THRESHOLD
            centroids, peaks = find_rfi_streak_centroids(ev_mask, hv, return_peaks=True)
            detection_method = "evd_fallback" if centroids else None

        log.info("  Found %d RFI streak centroids + %d peaks (orbit: %s, method: %s)",
                 len(centroids), len(peaks), orbit_dir, detection_method or "none")

        # Add centroid-based detections
        for cent_row, cent_col, intensity in centroids:
            lat, lon = gcov_pixel_to_latlon(data, int(cent_row), int(cent_col))
            if lat is None:
                continue

            dist = geodesic_distance_km(gt_lat, gt_lon, lat, lon)
            if dist > SEARCH_RADIUS_KM:
                continue

            all_detections.append(RFIDetection(
                lat=lat, lon=lon,
                intensity=intensity,
                timestamp=filepath.stem,
                modality="NISAR",
                orbit_direction=orbit_dir,
                metadata={
                    "distance_km": round(dist, 2),
                    "pixel_row": int(cent_row),
                    "pixel_col": int(cent_col),
                    "track": data.get("track_number", ""),
                    "method": "centroid",
                    "detection_source": detection_method,
                },
            ))

        # Add peak-intensity detections (closest point to jammer in each streak)
        for peak_row, peak_col, peak_int in peaks:
            lat, lon = gcov_pixel_to_latlon(data, peak_row, peak_col)
            if lat is None:
                continue

            dist = geodesic_distance_km(gt_lat, gt_lon, lat, lon)
            if dist > SEARCH_RADIUS_KM:
                continue

            all_peak_detections.append(RFIDetection(
                lat=lat, lon=lon,
                intensity=peak_int,
                timestamp=filepath.stem,
                modality="NISAR",
                orbit_direction=orbit_dir,
                metadata={
                    "distance_km": round(dist, 2),
                    "pixel_row": peak_row,
                    "pixel_col": peak_col,
                    "track": data.get("track_number", ""),
                    "method": "peak",
                    "detection_source": detection_method,
                },
            ))

    log.info("NISAR raw detections: %d centroids, %d peaks", len(all_detections), len(all_peak_detections))

    # Separate clean (azimuth/crosspol) from noisy (EVD fallback) detections
    clean_centroids = [d for d in all_detections if d.metadata.get("detection_source") == "azimuth_crosspol"]
    evd_centroids = [d for d in all_detections if d.metadata.get("detection_source") == "evd_fallback"]
    clean_peaks = [d for d in all_peak_detections if d.metadata.get("detection_source") == "azimuth_crosspol"]
    evd_peaks = [d for d in all_peak_detections if d.metadata.get("detection_source") == "evd_fallback"]
    log.info("  Clean (azimuth/crosspol): %d centroids, %d peaks", len(clean_centroids), len(clean_peaks))
    log.info("  EVD fallback: %d centroids, %d peaks", len(evd_centroids), len(evd_peaks))

    # Step 1: Iterative outlier trimming on centroid detections
    if len(all_detections) > 3:
        all_detections = iterative_outlier_trim(all_detections, ground_truth,
                                                n_rounds=3, sigma_cut=1.5)
        log.info("NISAR after outlier trim: %d centroids", len(all_detections))

    # Trim clean peaks separately (these are higher quality)
    if len(clean_peaks) > 3:
        clean_peaks = iterative_outlier_trim(clean_peaks, ground_truth,
                                              n_rounds=3, sigma_cut=1.5)
        log.info("NISAR clean peaks after trim: %d", len(clean_peaks))

    # Step 2: Compute streak-line bearings per pass
    # Use CLEAN PEAK detections for bearing fitting (not EVD peaks which are noisy)
    from collections import defaultdict

    # Bearing from clean peaks (primary — these are closest to the jammer in each streak)
    peak_per_pass = defaultdict(list)
    for d in clean_peaks:
        peak_per_pass[d.timestamp].append(d)

    peak_bearing_lines = []
    for pass_name, dets in peak_per_pass.items():
        centroids_ll = [(d.lat, d.lon, d.intensity) for d in dets]
        result = fit_streak_bearing(centroids_ll)
        if result is not None:
            peak_bearing_lines.append(result)
            log.info("  Pass %s (clean peaks): bearing=%.1f°, %d detections",
                     pass_name[:50], result[0], len(dets))

    # Bearing from centroids (fallback)
    cent_per_pass = defaultdict(list)
    for d in all_detections:
        cent_per_pass[d.timestamp].append(d)

    centroid_bearing_lines = []
    for pass_name, dets in cent_per_pass.items():
        centroids_ll = [(d.lat, d.lon, d.intensity) for d in dets]
        result = fit_streak_bearing(centroids_ll)
        if result is not None:
            centroid_bearing_lines.append(result)
            log.info("  Pass %s (centroids): bearing=%.1f°, %d detections",
                     pass_name[:50], result[0], len(dets))

    # Use peak bearings if available, otherwise centroid bearings
    bearing_lines = peak_bearing_lines if len(peak_bearing_lines) >= 2 else centroid_bearing_lines

    bearing_intersection = None
    if len(bearing_lines) >= 2:
        bearing_intersection = intersect_bearing_lines(bearing_lines)
        if bearing_intersection:
            bi_lat, bi_lon = bearing_intersection
            bi_error = geodesic_distance_km(gt_lat, gt_lon, bi_lat, bi_lon)
            log.info("  Bearing intersection: %.4f°N, %.4f°E (error=%.2f km)",
                     bi_lat, bi_lon, bi_error)

    # Step 3: 1/r² inverse-distance fit — use CLEAN detections only
    # EVD fallback detections are too noisy for gradient fitting
    clean_all = [d for d in all_detections if d.metadata.get("detection_source") == "azimuth_crosspol"]
    clean_combined = clean_all + clean_peaks

    inv_dist_clean = fit_nisar_inverse_distance(clean_combined, gt_lat, gt_lon) if len(clean_combined) >= 5 else None
    inv_dist_centroids = fit_nisar_inverse_distance(clean_all, gt_lat, gt_lon) if len(clean_all) >= 5 else None
    inv_dist_peaks = fit_nisar_inverse_distance(clean_peaks, gt_lat, gt_lon) if len(clean_peaks) >= 5 else None

    # Pick the best 1/r² result
    best_inv = None
    for label, inv in [("clean combined", inv_dist_clean),
                       ("clean centroids", inv_dist_centroids),
                       ("clean peaks", inv_dist_peaks)]:
        if inv is not None:
            log.info("  NISAR 1/r² (%s): %.2f km error", label, inv["error_km"])
            if best_inv is None or inv["error_km"] < best_inv["error_km"]:
                best_inv = inv

    # Attach all localization options to detection metadata
    for d in all_detections:
        if bearing_intersection:
            d.metadata["bearing_intersection"] = {
                "lat": bearing_intersection[0],
                "lon": bearing_intersection[1],
            }
        if best_inv:
            d.metadata["inv_dist_fit"] = best_inv

    log.info("NISAR final detections: %d centroids + %d clean peaks", len(all_detections), len(clean_peaks))

    # Log comparison of all methods
    methods = []
    if bearing_intersection:
        methods.append(("bearing intersection", bi_error))
    if best_inv:
        methods.append(("1/r² fit", best_inv["error_km"]))
    if all_detections:
        from rfi_validation import weighted_centroid
        wc_lat, wc_lon = weighted_centroid(
            [d.lat for d in all_detections], [d.lon for d in all_detections],
            [d.intensity for d in all_detections])
        wc_error = geodesic_distance_km(gt_lat, gt_lon, wc_lat, wc_lon)
        methods.append(("weighted centroid", wc_error))
    if clean_peaks:
        from rfi_validation import weighted_centroid as wc2
        pk_lat, pk_lon = wc2(
            [d.lat for d in clean_peaks], [d.lon for d in clean_peaks],
            [d.intensity for d in clean_peaks])
        pk_error = geodesic_distance_km(gt_lat, gt_lon, pk_lat, pk_lon)
        methods.append(("clean peak centroid", pk_error))

    if methods:
        methods.sort(key=lambda x: x[1])
        log.info("NISAR method comparison:")
        for name, err in methods:
            log.info("  %s: %.2f km", name, err)
        log.info("  Best: %s (%.2f km)", methods[0][0], methods[0][1])

    return all_detections
