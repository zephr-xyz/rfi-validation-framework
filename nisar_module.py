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
        # Orbit direction
        if "identification" in f:
            id_grp = f["identification"]
            if "orbit_pass_direction" in id_grp:
                val = id_grp["orbit_pass_direction"][()]
                data["orbit_direction"] = val.decode() if isinstance(val, bytes) else str(val)
            if "track_number" in id_grp:
                data["track_number"] = int(id_grp["track_number"][()])

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


# ── Eigenvalue Decomposition RFI Detection ───────────────────────────────────

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


def find_rfi_streak_centroids(rfi_mask, hv_intensity):
    """Find centroids of connected RFI streak regions.

    Returns list of (row, col, intensity) tuples for each streak centroid.
    """
    from scipy import ndimage

    labeled, n_features = ndimage.label(rfi_mask)
    centroids = []

    for label_id in range(1, n_features + 1):
        region = labeled == label_id
        n_pixels = region.sum()

        if n_pixels < MIN_STREAK_PIXELS:
            continue

        # Intensity-weighted centroid
        region_intensity = hv_intensity * region
        total_intensity = region_intensity.sum()
        if total_intensity == 0:
            continue

        rows_idx, cols_idx = np.where(region)
        weights = hv_intensity[rows_idx, cols_idx]
        cent_row = float(np.average(rows_idx, weights=weights))
        cent_col = float(np.average(cols_idx, weights=weights))

        centroids.append((cent_row, cent_col, float(total_intensity / n_pixels)))

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

    # Compute baseline eigenvalue ratio (average across OFF passes)
    baseline_ev_ratio = None
    if baseline_files:
        baseline_ratios = []
        for filepath in baseline_files:
            log.info("Loading baseline: %s", filepath.name)
            try:
                data = parse_nisar_gcov(filepath)
                if "HVHV" not in data:
                    continue
                hv_db = 10 * np.log10(np.maximum(data["HVHV"], 1e-30))
                _, ev_ratio = eigenvalue_rfi_detection(hv_db)
                baseline_ratios.append(ev_ratio)
            except Exception as e:
                log.warning("Baseline failed %s: %s", filepath.name, e)
        if baseline_ratios:
            # Average baseline (handles shape mismatches by using smallest common shape)
            min_r = min(b.shape[0] for b in baseline_ratios)
            min_c = min(b.shape[1] for b in baseline_ratios)
            baseline_ev_ratio = np.mean(
                [b[:min_r, :min_c] for b in baseline_ratios], axis=0)
            log.info("Baseline eigenvalue ratio computed (shape %s, median %.2f)",
                     baseline_ev_ratio.shape, np.median(baseline_ev_ratio))

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

        hv = data["HVHV"]
        orbit_dir = data.get("orbit_direction", "Unknown")

        # Convert to dB for better dynamic range handling
        hv_db = 10 * np.log10(np.maximum(hv, 1e-30))

        # Method 1: Eigenvalue decomposition
        log.info("  Running eigenvalue decomposition (shape %s)...", hv.shape)
        rfi_mask, ev_ratio = eigenvalue_rfi_detection(hv_db)

        # Baseline subtraction: remove ambient L-band RFI signature
        if baseline_ev_ratio is not None:
            min_r = min(ev_ratio.shape[0], baseline_ev_ratio.shape[0])
            min_c = min(ev_ratio.shape[1], baseline_ev_ratio.shape[1])
            diff = ev_ratio[:min_r, :min_c] - baseline_ev_ratio[:min_r, :min_c]
            log.info("  Baseline subtraction: max diff %.2f, mean diff %.2f",
                     np.max(diff), np.mean(diff))
            # Re-threshold on the differential
            rfi_mask_diff = diff > EIGENVALUE_RATIO_THRESHOLD
            # Use differential mask if it has detections, otherwise use raw
            if rfi_mask_diff.any():
                rfi_mask = np.zeros_like(rfi_mask)
                rfi_mask[:min_r, :min_c] = rfi_mask_diff

        centroids = find_rfi_streak_centroids(rfi_mask, hv)

        # Method 2: Intensity spike fallback
        if not centroids:
            log.info("  No eigenvalue streaks found, trying intensity spikes...")
            spike_mask, zscore = intensity_spike_detection(hv_db)
            centroids = find_rfi_streak_centroids(spike_mask, hv)

        log.info("  Found %d RFI streak centroids (orbit: %s)", len(centroids), orbit_dir)

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
                timestamp=filepath.stem,  # Use filename as timestamp proxy
                modality="NISAR",
                orbit_direction=orbit_dir,
                metadata={
                    "distance_km": round(dist, 2),
                    "pixel_row": int(cent_row),
                    "pixel_col": int(cent_col),
                    "track": data.get("track_number", ""),
                },
            ))

    log.info("NISAR total detections: %d", len(all_detections))
    return all_detections
