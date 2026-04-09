#!/usr/bin/env python3
"""Build a dual-date GPS jammer viewer HTML from March 15 and April 6, 2026 CYGNSS data."""

import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "output", "iran_scan")

def load_json(name):
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)

def main():
    mar = load_json("iran_jammers_2026-03-15.json")
    apr = load_json("iran_jammers_2026-04-06.json")
    mar_clusters = load_json("iran_clusters_2026-03-15.json")
    apr_clusters = load_json("iran_clusters_2026-04-06.json")

    # Ensure n_jammers_filtered exists
    if "n_jammers_filtered" not in mar:
        mar["n_jammers_filtered"] = len(mar.get("jammers_filtered", []))
    if "n_jammers_filtered" not in apr:
        apr["n_jammers_filtered"] = len(apr.get("jammers_filtered", []))

    # Serialize data for embedding
    mar_json = json.dumps(mar, separators=(",", ":"))
    apr_json = json.dumps(apr, separators=(",", ":"))
    mar_clusters_json = json.dumps(mar_clusters, separators=(",", ":"))
    apr_clusters_json = json.dumps(apr_clusters, separators=(",", ":"))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Iran GPS Jammer Detections — Temporal Comparison</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #111; }}
  #map {{ width: 100vw; height: 100vh; }}

  .info-panel {{
    position: absolute; top: 12px; left: 12px; z-index: 1000;
    background: rgba(10,10,10,0.92); color: #f0f0f0;
    padding: 14px 18px; border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.15);
    max-width: 420px; font-size: 13px; line-height: 1.5;
    backdrop-filter: blur(8px);
  }}
  .info-panel h2 {{ font-size: 15px; margin-bottom: 6px; color: #fff; }}
  .info-panel .subtitle {{ color: #999; font-size: 11px; margin-bottom: 10px; }}
  .info-panel .stat {{ display: flex; justify-content: space-between; padding: 2px 0; }}
  .info-panel .stat .label {{ color: #999; }}
  .info-panel .stat .value {{ color: #fff; font-weight: 600; }}
  .info-panel .divider {{ border-top: 1px solid rgba(255,255,255,0.1); margin: 8px 0; }}
  .info-panel .date-header {{ font-weight: 700; font-size: 12px; margin-bottom: 2px; }}
  .info-panel .mar-header {{ color: #ff4444; }}
  .info-panel .apr-header {{ color: #ff8c00; }}
  .stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0 16px; }}
  .stats-col {{ min-width: 0; }}

  .legend {{
    position: absolute; bottom: 30px; left: 12px; z-index: 1000;
    background: rgba(10,10,10,0.92); color: #f0f0f0;
    padding: 12px 16px; border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.15);
    font-size: 12px; backdrop-filter: blur(8px);
  }}
  .legend-item {{ display: flex; align-items: center; gap: 8px; padding: 3px 0; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; border: 1px solid rgba(255,255,255,0.3); }}

  .layer-control {{
    position: absolute; top: 12px; right: 12px; z-index: 1000;
    background: rgba(10,10,10,0.92); color: #f0f0f0;
    padding: 12px 16px; border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.15);
    font-size: 12px; backdrop-filter: blur(8px);
    max-height: 90vh; overflow-y: auto;
  }}
  .layer-control label {{ display: block; padding: 3px 0; cursor: pointer; }}
  .layer-control input {{ margin-right: 6px; }}
  .layer-group-title {{ font-weight: 700; font-size: 11px; color: #999; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 8px; margin-bottom: 2px; }}
  .layer-group-title:first-child {{ margin-top: 0; }}

  .leaflet-popup-content-wrapper {{
    background: rgba(15,15,15,0.95) !important;
    color: #f0f0f0 !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
  }}
  .leaflet-popup-tip {{ background: rgba(15,15,15,0.95) !important; }}
  .leaflet-popup-content {{ font-size: 12px; line-height: 1.6; }}
  .popup-title {{ font-weight: 700; font-size: 14px; margin-bottom: 4px; }}
  .popup-stat {{ display: flex; justify-content: space-between; gap: 12px; }}
  .popup-stat .k {{ color: #999; }}
  .popup-stat .v {{ color: #fff; font-weight: 600; }}
</style>
</head>
<body>

<div id="map"></div>

<div class="info-panel" id="info">
  <h2>GPS Jammers Detected Across Iran &mdash; Temporal Comparison</h2>
  <div class="subtitle">CYGNSS GNSS-R Blind Search | March 15 &amp; April 6, 2026 vs Dec 27, 2025 Baseline</div>
  <div class="divider"></div>
  <div class="stats-grid">
    <div class="stats-col">
      <div class="date-header mar-header">Mar 15, 2026</div>
      <div class="stat"><span class="label">High-conf</span><span class="value" id="mar-filtered">&mdash;</span></div>
      <div class="stat"><span class="label">Raw detections</span><span class="value" id="mar-raw">&mdash;</span></div>
      <div class="stat"><span class="label">Clusters</span><span class="value" id="mar-clusters">&mdash;</span></div>
      <div class="stat"><span class="label">Measurements</span><span class="value" id="mar-conflict">&mdash;</span></div>
    </div>
    <div class="stats-col">
      <div class="date-header apr-header">Apr 6, 2026</div>
      <div class="stat"><span class="label">High-conf</span><span class="value" id="apr-filtered">&mdash;</span></div>
      <div class="stat"><span class="label">Raw detections</span><span class="value" id="apr-raw">&mdash;</span></div>
      <div class="stat"><span class="label">Clusters</span><span class="value" id="apr-clusters">&mdash;</span></div>
      <div class="stat"><span class="label">Measurements</span><span class="value" id="apr-conflict">&mdash;</span></div>
    </div>
  </div>
  <div class="divider"></div>
  <div class="stat"><span class="label">Baseline measurements</span><span class="value" id="n-baseline">&mdash;</span></div>
  <div class="divider"></div>
  <div style="color:#999; font-size:11px;">
    Filters: CEP &lt; 15 km, &ge; 20 pts, z &gt; 2.8<br>
    Click a marker for details
  </div>
</div>

<div class="layer-control">
  <strong style="font-size:13px;">Layers</strong>
  <div class="layer-group-title" style="color:#ff4444;">March 15, 2026</div>
  <label><input type="checkbox" id="toggle-mar-filtered" checked> High-confidence ({mar['n_jammers_filtered']})</label>
  <label><input type="checkbox" id="toggle-mar-all"> All detections ({mar['n_jammers']})</label>
  <label><input type="checkbox" id="toggle-mar-heatmap"> Detection heatmap</label>
  <div class="layer-group-title" style="color:#ff8c00;">April 6, 2026</div>
  <label><input type="checkbox" id="toggle-apr-filtered" checked> High-confidence ({apr['n_jammers_filtered']})</label>
  <label><input type="checkbox" id="toggle-apr-all"> All detections ({apr['n_jammers']})</label>
  <label><input type="checkbox" id="toggle-apr-heatmap"> Detection heatmap</label>
  <div class="layer-group-title">Reference</div>
  <label><input type="checkbox" id="toggle-shiraz" checked> Validated Shiraz jammer</label>
</div>

<div class="legend">
  <strong>Legend</strong>
  <div class="legend-item"><div class="legend-dot" style="background:#ff2a2a; width:16px; height:16px;"></div> Mar 15 high-confidence</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ff8c00; width:16px; height:16px;"></div> Apr 6 high-confidence</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ff6666; width:10px; height:10px;"></div> Mar 15 all detections</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ffb366; width:10px; height:10px;"></div> Apr 6 all detections</div>
  <div class="legend-item"><div class="legend-dot" style="background:#00ff88;"></div> Validated Shiraz jammer</div>
  <div class="legend-item" style="color:#999; font-size:11px; margin-top:4px;">Marker size = signal amplitude</div>
</div>

<script>
// -- Data (embedded) --
const MAR = {mar_json};
const APR = {apr_json};
const MAR_CLUSTERS = {mar_clusters_json};
const APR_CLUSTERS = {apr_clusters_json};

// -- Map setup --
const map = L.map('map', {{
  center: [31.5, 53.0],
  zoom: 8,
  maxZoom: 24,
  zoomControl: false,
}});

L.control.zoom({{ position: 'bottomright' }}).addTo(map);

// Satellite basemap
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
  attribution: 'Esri World Imagery',
  maxZoom: 24,
  maxNativeZoom: 18,
}}).addTo(map);

// Labels overlay
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_only_labels/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  maxZoom: 24,
  maxNativeZoom: 18,
  opacity: 0.7,
}}).addTo(map);

// -- Populate info panel --
document.getElementById('mar-filtered').textContent = MAR.n_jammers_filtered;
document.getElementById('mar-raw').textContent = MAR.n_jammers;
document.getElementById('mar-clusters').textContent = MAR.n_clusters;
document.getElementById('mar-conflict').textContent = MAR.n_conflict_measurements.toLocaleString();
document.getElementById('apr-filtered').textContent = APR.n_jammers_filtered;
document.getElementById('apr-raw').textContent = APR.n_jammers;
document.getElementById('apr-clusters').textContent = APR.n_clusters;
document.getElementById('apr-conflict').textContent = APR.n_conflict_measurements.toLocaleString();
document.getElementById('n-baseline').textContent = MAR.n_baseline_measurements.toLocaleString();

// -- Helper: amplitude to radius --
function ampToRadius(amp, amps) {{
  const ampMin = Math.min(...amps);
  const ampMax = Math.max(...amps);
  const norm = (amp - ampMin) / (ampMax - ampMin + 1);
  return 6 + norm * 14;
}}

const marAmps = MAR.jammers_filtered.map(j => j.amplitude);
const aprAmps = APR.jammers_filtered.map(j => j.amplitude);

// -- Popup content builder --
function jammerPopup(j, rank, dateLabel) {{
  const cep = j.bootstrap_cep_km ? j.bootstrap_cep_km.toFixed(1) + ' km' : 'N/A';
  return '<div class="popup-title">' + dateLabel + ' Jammer #' + rank + '</div>' +
    '<div class="popup-stat"><span class="k">Position</span><span class="v">' + j.estimated_lat.toFixed(4) + '&deg;N, ' + j.estimated_lon.toFixed(4) + '&deg;E</span></div>' +
    '<div class="popup-stat"><span class="k">Amplitude</span><span class="v">' + j.amplitude.toLocaleString() + '</span></div>' +
    '<div class="popup-stat"><span class="k">Bootstrap CEP</span><span class="v">' + cep + '</span></div>' +
    '<div class="popup-stat"><span class="k">Elevated points</span><span class="v">' + j.n_elevated_points + '</span></div>' +
    '<div class="popup-stat"><span class="k">Cluster z-score</span><span class="v">' + j.cluster_mean_zscore.toFixed(1) + '</span></div>' +
    '<div class="popup-stat"><span class="k">Cluster extent</span><span class="v">' + j.cluster_extent_km.toFixed(0) + ' km</span></div>' +
    (j.known_jammer ? '<div style="color:#00ff88; font-weight:700; margin-top:4px;">Validated: ' + j.known_jammer + ' jammer</div>' : '');
}}

function clusterPopup(c) {{
  return '<div class="popup-title">Cluster ' + c.cluster_id + '</div>' +
    '<div class="popup-stat"><span class="k">Measurements</span><span class="v">' + c.n_measurements + '</span></div>' +
    '<div class="popup-stat"><span class="k">Grid cells</span><span class="v">' + c.n_cells + '</span></div>' +
    '<div class="popup-stat"><span class="k">Mean z-score</span><span class="v">' + c.mean_zscore.toFixed(1) + '</span></div>' +
    '<div class="popup-stat"><span class="k">Max z-score</span><span class="v">' + c.max_zscore.toFixed(1) + '</span></div>' +
    '<div class="popup-stat"><span class="k">Mean noise</span><span class="v">' + c.mean_noise.toLocaleString() + '</span></div>' +
    '<div class="popup-stat"><span class="k">Extent</span><span class="v">' + c.extent_km.toFixed(0) + ' km</span></div>';
}}

// -- Layer: Mar 15 High-confidence --
const marFilteredLayer = L.layerGroup();
MAR.jammers_filtered.forEach((j, i) => {{
  const r = ampToRadius(j.amplitude, marAmps);
  const marker = L.circleMarker([j.estimated_lat, j.estimated_lon], {{
    radius: r,
    fillColor: '#ff2a2a',
    fillOpacity: 0.85,
    color: '#fff',
    weight: 1.5,
    opacity: 0.8,
  }}).bindPopup(jammerPopup(j, i + 1, 'Mar 15'), {{ maxWidth: 300 }});

  if (j.bootstrap_cep_km && j.bootstrap_cep_km < 15) {{
    const cepCircle = L.circle([j.estimated_lat, j.estimated_lon], {{
      radius: j.bootstrap_cep_km * 1000,
      fill: false,
      color: '#ff2a2a40',
      weight: 1,
      dashArray: '4 4',
    }});
    marFilteredLayer.addLayer(cepCircle);
  }}
  marFilteredLayer.addLayer(marker);
}});
marFilteredLayer.addTo(map);

// -- Layer: Apr 6 High-confidence --
const aprFilteredLayer = L.layerGroup();
APR.jammers_filtered.forEach((j, i) => {{
  const r = ampToRadius(j.amplitude, aprAmps);
  const marker = L.circleMarker([j.estimated_lat, j.estimated_lon], {{
    radius: r,
    fillColor: '#ff8c00',
    fillOpacity: 0.85,
    color: '#fff',
    weight: 1.5,
    opacity: 0.8,
  }}).bindPopup(jammerPopup(j, i + 1, 'Apr 6'), {{ maxWidth: 300 }});

  if (j.bootstrap_cep_km && j.bootstrap_cep_km < 15) {{
    const cepCircle = L.circle([j.estimated_lat, j.estimated_lon], {{
      radius: j.bootstrap_cep_km * 1000,
      fill: false,
      color: '#ff8c0040',
      weight: 1,
      dashArray: '4 4',
    }});
    aprFilteredLayer.addLayer(cepCircle);
  }}
  aprFilteredLayer.addLayer(marker);
}});
aprFilteredLayer.addTo(map);

// -- Layer: Mar 15 All detections (smaller, lighter) --
const marAllLayer = L.layerGroup();
const marFilteredIds = new Set(MAR.jammers_filtered.map(j => j.cluster_id));
MAR.jammers.forEach((j, i) => {{
  if (marFilteredIds.has(j.cluster_id)) return;
  L.circleMarker([j.estimated_lat, j.estimated_lon], {{
    radius: 4,
    fillColor: '#ff6666',
    fillOpacity: 0.35,
    color: '#ff666680',
    weight: 0.5,
  }}).bindPopup(
    '<div class="popup-title" style="color:#ff6666;">Mar 15 Detection #' + (i+1) + '</div>' +
    '<div class="popup-stat"><span class="k">Amplitude</span><span class="v">' + j.amplitude.toLocaleString() + '</span></div>' +
    '<div class="popup-stat"><span class="k">CEP</span><span class="v">' + (j.bootstrap_cep_km ? j.bootstrap_cep_km.toFixed(1) + ' km' : 'N/A') + '</span></div>' +
    '<div class="popup-stat"><span class="k">Points</span><span class="v">' + j.n_elevated_points + '</span></div>' +
    '<div class="popup-stat"><span class="k">Z-score</span><span class="v">' + j.cluster_mean_zscore.toFixed(1) + '</span></div>',
    {{ maxWidth: 250 }}
  ).addTo(marAllLayer);
}});

// -- Layer: Apr 6 All detections (smaller, lighter) --
const aprAllLayer = L.layerGroup();
const aprFilteredIds = new Set(APR.jammers_filtered.map(j => j.cluster_id));
APR.jammers.forEach((j, i) => {{
  if (aprFilteredIds.has(j.cluster_id)) return;
  L.circleMarker([j.estimated_lat, j.estimated_lon], {{
    radius: 4,
    fillColor: '#ffb366',
    fillOpacity: 0.35,
    color: '#ffb36680',
    weight: 0.5,
  }}).bindPopup(
    '<div class="popup-title" style="color:#ffb366;">Apr 6 Detection #' + (i+1) + '</div>' +
    '<div class="popup-stat"><span class="k">Amplitude</span><span class="v">' + j.amplitude.toLocaleString() + '</span></div>' +
    '<div class="popup-stat"><span class="k">CEP</span><span class="v">' + (j.bootstrap_cep_km ? j.bootstrap_cep_km.toFixed(1) + ' km' : 'N/A') + '</span></div>' +
    '<div class="popup-stat"><span class="k">Points</span><span class="v">' + j.n_elevated_points + '</span></div>' +
    '<div class="popup-stat"><span class="k">Z-score</span><span class="v">' + j.cluster_mean_zscore.toFixed(1) + '</span></div>',
    {{ maxWidth: 250 }}
  ).addTo(aprAllLayer);
}});

// -- Layer: Mar 15 Heatmap --
const marHeatData = MAR_CLUSTERS.map(c => [
  c.centroid_lat, c.centroid_lon,
  Math.min(c.mean_zscore / 8.0, 1.0) * c.n_measurements / 10
]);
const marHeatLayer = L.heatLayer(marHeatData, {{
  radius: 35,
  blur: 25,
  maxZoom: 8,
  max: 5,
  gradient: {{
    0.0: 'transparent',
    0.3: '#ff2a2a20',
    0.5: '#ff2a2a60',
    0.7: '#ff4a2a90',
    0.85: '#ff6a00c0',
    1.0: '#ff0000e0',
  }},
}});

// -- Layer: Apr 6 Heatmap --
const aprHeatData = APR_CLUSTERS.map(c => [
  c.centroid_lat, c.centroid_lon,
  Math.min(c.mean_zscore / 8.0, 1.0) * c.n_measurements / 10
]);
const aprHeatLayer = L.heatLayer(aprHeatData, {{
  radius: 35,
  blur: 25,
  maxZoom: 8,
  max: 5,
  gradient: {{
    0.0: 'transparent',
    0.3: '#ff8c0020',
    0.5: '#ff8c0060',
    0.7: '#ffa50090',
    0.85: '#ffcc00c0',
    1.0: '#ffee00e0',
  }},
}});

// -- Validated Shiraz jammer --
const shirazMarker = L.marker([27.3182, 52.8703], {{
  icon: L.divIcon({{
    className: '',
    html: '<div style="color:#00ff88; font-size:22px; text-shadow:0 0 6px #000; margin-left:-8px; margin-top:-12px;">&#9733;</div>',
    iconSize: [20, 20],
  }}),
  zIndex: 1000,
}}).bindPopup(
  '<div class="popup-title" style="color:#00ff88;">Validated Shiraz Jammer</div>' +
  '<div class="popup-stat"><span class="k">Ground Truth</span><span class="v">27.3182&deg;N, 52.8703&deg;E</span></div>' +
  '<div class="popup-stat"><span class="k">CYGNSS error</span><span class="v">4.33 km</span></div>' +
  '<div class="popup-stat"><span class="k">Bootstrap CEP</span><span class="v">3.48 km</span></div>' +
  '<div class="popup-stat"><span class="k">Detections</span><span class="v">785</span></div>' +
  '<div style="color:#999; font-size:11px; margin-top:4px;">Validated with independent SIGINT ground truth</div>',
  {{ maxWidth: 300 }}
).addTo(map);

// -- Layer toggle controls --
function bindToggle(id, layer) {{
  document.getElementById(id).addEventListener('change', e => {{
    e.target.checked ? map.addLayer(layer) : map.removeLayer(layer);
  }});
}}

bindToggle('toggle-mar-filtered', marFilteredLayer);
bindToggle('toggle-mar-all', marAllLayer);
bindToggle('toggle-mar-heatmap', marHeatLayer);
bindToggle('toggle-apr-filtered', aprFilteredLayer);
bindToggle('toggle-apr-all', aprAllLayer);
bindToggle('toggle-apr-heatmap', aprHeatLayer);

document.getElementById('toggle-shiraz').addEventListener('change', e => {{
  e.target.checked ? map.addLayer(shirazMarker) : map.removeLayer(shirazMarker);
}});
</script>
</body>
</html>"""

    out_path = os.path.join(DATA_DIR, "iran_jammers_viewer.html")
    with open(out_path, "w") as f:
        f.write(html)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Written: {out_path} ({size_mb:.1f} MB)")
    print(f"Mar 15: {mar['n_jammers_filtered']} high-conf / {mar['n_jammers']} total jammers, {mar['n_clusters']} clusters")
    print(f"Apr 6:  {apr['n_jammers_filtered']} high-conf / {apr['n_jammers']} total jammers, {apr['n_clusters']} clusters")

if __name__ == "__main__":
    main()
