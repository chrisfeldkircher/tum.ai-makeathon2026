"""
submission_visualizer.py — Interactive viewer for challenge GeoJSON submissions.

Expected submission format (osapiens Makeathon 2026):
    FeatureCollection where each feature is a Polygon or MultiPolygon.
    Optional  properties.time_step  as an integer in YYMM form (e.g. 2204 = Apr 2022).

Public API:
    validate_submission(path)      -> (ok: bool, issues: list[str])
    submission_stats(path)         -> dict with polygon count, area, year breakdown
    visualize_submission(path, ...) -> folium.Map (optionally saved to HTML)
    make_fake_submission(...)      -> dict (writes GeoJSON if out_path given)
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional, Union

try:
    import folium
    from folium.plugins import MarkerCluster
    _HAS_FOLIUM = True
except ImportError:
    _HAS_FOLIUM = False

try:
    from shapely.geometry import shape
    from shapely.ops import transform as shp_transform
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False

try:
    from pyproj import Transformer
    _HAS_PYPROJ = True
except ImportError:
    _HAS_PYPROJ = False


PathLike = Union[str, Path]


# --- Format validation ---

def _load_geojson(path: PathLike) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _parse_time_step(value: Any) -> Optional[tuple[int, int]]:
    """Return (year, month) or None. Accepts int or str in YYMM form."""
    if value is None:
        return None
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    if v < 0 or v > 9912:
        return None
    year = 2000 + (v // 100)
    month = v % 100
    if not (1 <= month <= 12):
        return None
    return year, month


def validate_submission(path: PathLike) -> tuple[bool, list[str]]:
    """Check FeatureCollection + Polygon/MultiPolygon + optional YYMM time_step."""
    issues: list[str] = []
    try:
        gj = _load_geojson(path)
    except Exception as e:
        return False, [f"cannot parse JSON: {e}"]

    if gj.get("type") != "FeatureCollection":
        issues.append(f"top-level type must be 'FeatureCollection', got {gj.get('type')!r}")

    feats = gj.get("features", [])
    if not isinstance(feats, list):
        issues.append("'features' must be a list")
        return False, issues
    if len(feats) == 0:
        issues.append("feature collection is empty")

    for i, feat in enumerate(feats):
        if feat.get("type") != "Feature":
            issues.append(f"feature[{i}] type must be 'Feature'")
            continue
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        if gtype not in ("Polygon", "MultiPolygon"):
            issues.append(f"feature[{i}] geometry type {gtype!r} not in (Polygon, MultiPolygon)")
        props = feat.get("properties") or {}
        if "time_step" in props and props["time_step"] is not None:
            parsed = _parse_time_step(props["time_step"])
            if parsed is None:
                issues.append(
                    f"feature[{i}] properties.time_step={props['time_step']!r} is not a valid YYMM integer"
                )
    return (len(issues) == 0), issues


# --- Stats ---

def _iter_geoms(gj: dict):
    for feat in gj.get("features", []):
        geom = feat.get("geometry")
        if geom and geom.get("type") in ("Polygon", "MultiPolygon"):
            yield feat, geom


def _polygon_area_m2(geom: dict) -> float:
    """Area in m^2 via an equal-area projection (EPSG:6933)."""
    if not (_HAS_SHAPELY and _HAS_PYPROJ):
        return float("nan")
    shp = shape(geom)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    projected = shp_transform(transformer.transform, shp)
    return float(projected.area)


def submission_stats(path: PathLike) -> dict:
    gj = _load_geojson(path)
    years = Counter()
    months = Counter()
    missing_ts = 0
    total_area_ha = 0.0
    per_year_area_ha = defaultdict(float)
    n_polygon = 0
    n_multipolygon = 0

    for feat, geom in _iter_geoms(gj):
        if geom["type"] == "Polygon":
            n_polygon += 1
        else:
            n_multipolygon += 1
        props = feat.get("properties") or {}
        parsed = _parse_time_step(props.get("time_step"))
        if parsed is None:
            missing_ts += 1
            year = None
        else:
            year, month = parsed
            years[year] += 1
            months[f"{year}-{month:02d}"] += 1

        area_ha = _polygon_area_m2(geom) / 10_000.0
        total_area_ha += area_ha
        if year is not None:
            per_year_area_ha[year] += area_ha

    return {
        "n_features": n_polygon + n_multipolygon,
        "n_polygon": n_polygon,
        "n_multipolygon": n_multipolygon,
        "n_missing_time_step": missing_ts,
        "years": dict(sorted(years.items())),
        "months": dict(sorted(months.items())),
        "total_area_ha": total_area_ha,
        "area_ha_by_year": dict(sorted(per_year_area_ha.items())),
    }


# --- Visualization ---

_YEAR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]


def _color_for_year(year: Optional[int], year_to_color: dict[int, str]) -> str:
    if year is None:
        return "#888888"
    return year_to_color.get(year, "#444444")


def _bounds_of(gj: dict) -> Optional[tuple[float, float, float, float]]:
    """Return (min_lat, min_lon, max_lat, max_lon) over all coords, or None."""
    min_lat = min_lon = float("inf")
    max_lat = max_lon = float("-inf")
    found = False

    def walk(coords):
        nonlocal min_lat, min_lon, max_lat, max_lon, found
        if (
            isinstance(coords, (list, tuple))
            and len(coords) >= 2
            and isinstance(coords[0], (int, float))
            and isinstance(coords[1], (int, float))
        ):
            lon, lat = coords[0], coords[1]
            if lat < min_lat: min_lat = lat
            if lat > max_lat: max_lat = lat
            if lon < min_lon: min_lon = lon
            if lon > max_lon: max_lon = lon
            found = True
            return
        if isinstance(coords, (list, tuple)):
            for c in coords:
                walk(c)

    for _, geom in _iter_geoms(gj):
        walk(geom.get("coordinates"))
    return (min_lat, min_lon, max_lat, max_lon) if found else None


def visualize_submission(
    path: PathLike,
    out_html: Optional[PathLike] = None,
    truth_path: Optional[PathLike] = None,
    tiles: str = "OpenStreetMap",
    show_centroids: bool = True,
) -> "folium.Map":
    """Render a submission GeoJSON on an interactive folium map.

    Args:
        path: path to submission .geojson.
        out_html: if given, save the map to this HTML file.
        truth_path: optional ground-truth .geojson to overlay for comparison.
        tiles: folium tile provider name.
        show_centroids: cluster polygon centroids for quick navigation.
    """
    if not _HAS_FOLIUM:
        raise ImportError("folium is required: pip install folium")

    gj = _load_geojson(path)
    ok, issues = validate_submission(path)
    stats = submission_stats(path)

    bounds = _bounds_of(gj) or (-10, -60, 10, -40)
    min_lat, min_lon, max_lat, max_lon = bounds
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles=tiles)

    years_present = sorted(y for y in stats["years"].keys())
    year_to_color = {y: _YEAR_PALETTE[i % len(_YEAR_PALETTE)] for i, y in enumerate(years_present)}

    pred_group = folium.FeatureGroup(name="Predictions", show=True)
    centroid_cluster = MarkerCluster(name="Centroids", show=False) if show_centroids else None

    for feat, geom in _iter_geoms(gj):
        props = feat.get("properties") or {}
        parsed = _parse_time_step(props.get("time_step"))
        year = parsed[0] if parsed else None
        month = parsed[1] if parsed else None
        color = _color_for_year(year, year_to_color)
        ts_label = f"{year}-{month:02d}" if parsed else "unknown"

        tooltip = f"time_step: {ts_label}"
        popup_html = "<br>".join(
            f"<b>{k}</b>: {v}" for k, v in {"time_step": ts_label, **props}.items() if k != "time_step" or parsed
        )

        folium.GeoJson(
            {"type": "Feature", "geometry": geom, "properties": props},
            style_function=lambda _f, c=color: {
                "fillColor": c,
                "color": c,
                "weight": 1.2,
                "fillOpacity": 0.45,
            },
            tooltip=tooltip,
            popup=folium.Popup(popup_html, max_width=300) if popup_html else None,
        ).add_to(pred_group)

        if centroid_cluster is not None and _HAS_SHAPELY:
            try:
                c = shape(geom).centroid
                folium.CircleMarker(
                    [c.y, c.x], radius=3, color=color, fill=True, fill_opacity=0.9,
                    tooltip=ts_label,
                ).add_to(centroid_cluster)
            except Exception:
                pass

    pred_group.add_to(fmap)
    if centroid_cluster is not None:
        centroid_cluster.add_to(fmap)

    if truth_path is not None:
        try:
            truth_gj = _load_geojson(truth_path)
            truth_group = folium.FeatureGroup(name="Ground truth", show=True)
            folium.GeoJson(
                truth_gj,
                style_function=lambda _f: {
                    "fillColor": "#00000000",
                    "color": "#ff0000",
                    "weight": 2,
                    "dashArray": "4,3",
                    "fillOpacity": 0.0,
                },
                tooltip="ground truth",
            ).add_to(truth_group)
            truth_group.add_to(fmap)
        except Exception as e:
            print(f"[visualize_submission] skipped truth overlay: {e}")

    legend_rows = "".join(
        f"<div><span style='display:inline-block;width:12px;height:12px;background:{c};"
        f"margin-right:6px;border:1px solid #333;'></span>{y}</div>"
        for y, c in year_to_color.items()
    ) or "<div>(no year info)</div>"

    validation_banner = (
        f"<b style='color:{'green' if ok else 'crimson'}'>Validation: {'OK' if ok else 'ISSUES'}</b>"
        + ("" if ok else f"<div style='font-size:11px'>{len(issues)} issue(s) — see stdout</div>")
    )

    area_by_year = "".join(
        f"<div>{y}: {a:,.0f} ha</div>"
        for y, a in stats["area_ha_by_year"].items()
    ) or "<div>(no area info)</div>"

    legend_html = f"""
    <div style="position: fixed; bottom: 16px; left: 16px; z-index: 9999;
                background: white; padding: 10px 12px; border: 1px solid #888;
                border-radius: 6px; font-family: sans-serif; font-size: 12px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.15); max-width: 240px;">
      {validation_banner}
      <hr style="margin: 6px 0;">
      <b>Features:</b> {stats['n_features']}
        ({stats['n_polygon']} Polygon / {stats['n_multipolygon']} MultiPolygon)<br>
      <b>Missing time_step:</b> {stats['n_missing_time_step']}<br>
      <b>Total area:</b> {stats['total_area_ha']:,.0f} ha
      <hr style="margin: 6px 0;">
      <b>Year legend</b>
      {legend_rows}
      <hr style="margin: 6px 0;">
      <b>Area by year</b>
      {area_by_year}
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(fmap)

    try:
        fmap.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    except Exception:
        pass

    if not ok:
        print(f"[validate_submission] {len(issues)} issue(s):")
        for msg in issues:
            print(f"  - {msg}")

    if out_html is not None:
        out_html = Path(out_html)
        out_html.parent.mkdir(parents=True, exist_ok=True)
        fmap.save(str(out_html))
        print(f"saved map to {out_html}")

    return fmap


# --- Fake data for testing ---

# Rough forest-region centers (lon, lat) representing the challenge test distribution.
_FAKE_REGION_CENTERS = {
    "brazil":    (-60.0,  -3.0),   # Amazon
    "colombia":  (-74.0,   3.0),
    "thailand":  ( 99.0,  18.0),
    "drc":       ( 23.0,  -1.0),   # Congo basin
}


def _rand_polygon_ring(rng, cx: float, cy: float, radius_deg: float, n_vertices: int = 10) -> list[list[float]]:
    import math
    ring = []
    for i in range(n_vertices):
        theta = 2 * math.pi * i / n_vertices
        jitter = 0.6 + 0.8 * rng.random()
        r = radius_deg * jitter
        ring.append([cx + r * math.cos(theta), cy + r * math.sin(theta)])
    ring.append(ring[0])
    return ring


def make_fake_submission(
    n_per_region: int = 8,
    regions: Optional[Iterable[str]] = None,
    year_range: tuple[int, int] = (2021, 2024),
    multipolygon_frac: float = 0.15,
    missing_time_step_frac: float = 0.1,
    invalid_frac: float = 0.0,
    radius_km: float = 5.0,
    seed: int = 0,
    out_path: Optional[PathLike] = None,
) -> dict:
    """Generate a synthetic submission GeoJSON for testing the visualizer.

    Args:
        n_per_region: number of features per region center.
        regions: subset of _FAKE_REGION_CENTERS keys (default: all).
        year_range: inclusive (start, end) years for time_step sampling.
        multipolygon_frac: fraction of features emitted as MultiPolygon.
        missing_time_step_frac: fraction with properties.time_step = None.
        invalid_frac: fraction with a deliberately invalid time_step (to exercise validator).
        radius_km: approximate polygon radius.
        seed: RNG seed.
        out_path: if given, write the FeatureCollection to this .geojson path.

    Returns the FeatureCollection dict.
    """
    import random
    rng = random.Random(seed)

    if regions is None:
        regions = list(_FAKE_REGION_CENTERS.keys())

    radius_deg = radius_km / 111.0  # rough degrees per km at equator
    y0, y1 = year_range
    features: list[dict] = []

    for region in regions:
        if region not in _FAKE_REGION_CENTERS:
            continue
        base_lon, base_lat = _FAKE_REGION_CENTERS[region]
        for i in range(n_per_region):
            cx = base_lon + rng.uniform(-0.6, 0.6)
            cy = base_lat + rng.uniform(-0.6, 0.6)

            if rng.random() < multipolygon_frac:
                parts = []
                for _ in range(rng.randint(2, 3)):
                    ox, oy = rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)
                    ring = _rand_polygon_ring(rng, cx + ox, cy + oy, radius_deg * 0.6)
                    parts.append([ring])
                geom = {"type": "MultiPolygon", "coordinates": parts}
            else:
                ring = _rand_polygon_ring(rng, cx, cy, radius_deg)
                geom = {"type": "Polygon", "coordinates": [ring]}

            roll = rng.random()
            if roll < missing_time_step_frac:
                ts = None
            elif roll < missing_time_step_frac + invalid_frac:
                ts = 9999  # invalid YYMM
            else:
                year = rng.randint(y0, y1)
                month = rng.randint(1, 12)
                ts = (year % 100) * 100 + month

            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "region": region,
                    "time_step": ts,
                    "score": round(rng.uniform(0.4, 0.99), 3),
                },
            })

    fc = {"type": "FeatureCollection", "features": features}

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(fc, f)

    return fc


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Visualize an osapiens Makeathon submission GeoJSON.")
    p.add_argument("submission", help="path to submission .geojson")
    p.add_argument("--out", default=None, help="output HTML path (default: <submission>.html)")
    p.add_argument("--truth", default=None, help="optional ground-truth .geojson overlay")
    p.add_argument("--tiles", default="OpenStreetMap", help="folium tile provider")
    args = p.parse_args()

    out = args.out or str(Path(args.submission).with_suffix(".html"))
    visualize_submission(args.submission, out_html=out, truth_path=args.truth, tiles=args.tiles)
    stats = submission_stats(args.submission)
    print(json.dumps(stats, indent=2, default=str))
