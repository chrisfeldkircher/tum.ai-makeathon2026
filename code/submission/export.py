"""Export per-tile binary prediction masks to a single submission GeoJSON.

Usage:
    python code/submission/export.py \\
        --preds_dir ./predictions \\
        --cache_dir ./cache \\
        --data_dir  ./data \\
        --out       submission/submission.geojson

Input — for each tile a `predictions/<tile_id>.npz` containing:
    pred       (H, W) uint8   1 = deforestation, 0 = no deforestation
    time_step  scalar int     YYMM (e.g. 2204 for April 2022)  [optional]

CRS/transform resolution (in order of preference):
  1. `_crs_wkt` + `_transform` keys in `cache/<tile_id>.npz`
     (present if cache was built after the geo-referencing update)
  2. First Sentinel-2 GeoTIFF in `data/sentinel-2/{split}/{tile_id}__s2_l2a/`
     (fallback — works even with old caches)

The forest-probability gate (`forest_prob_pre >= 0.3`) is applied when
`cache_probs/<tile_id>.npz` is present, masking out predictions on pixels
that were never forest pre-2020.  If the sidecar is absent the gate is
skipped with a warning.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import Affine

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from problem.submission_utils import raster_to_geojson  # noqa: E402

NON_FOREST_THRESHOLD = 0.30  # matches LearnedForestMasker.tier_non_forest_max


def _resolve_crs_transform(
    tile_id: str,
    cache_dir: Path,
    data_dir: Optional[Path],
) -> tuple[CRS, Affine] | None:
    """Return (CRS, Affine) from cache if present, else from raw S2 GeoTIFF."""
    cache_path = cache_dir / f"{tile_id}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=True) as z:
            if "_crs_wkt" in z.files and "_transform" in z.files:
                crs = CRS.from_wkt(z["_crs_wkt"].item().decode())
                transform = Affine(*z["_transform"])
                return crs, transform

    # Fallback: read from the first available raw Sentinel-2 tile file
    if data_dir is not None:
        for split in ("train", "test"):
            tile_dir = data_dir / "sentinel-2" / split / f"{tile_id}__s2_l2a"
            tifs = sorted(tile_dir.glob("*.tif")) if tile_dir.exists() else []
            if tifs:
                with rasterio.open(tifs[0]) as src:
                    return src.crs, src.transform

    return None


def export_tile(
    tile_id: str,
    pred: np.ndarray,
    cache_dir: Path,
    data_dir: Optional[Path],
    probs_dir: Optional[Path],
    min_area_ha: float,
    time_step: Optional[int],
) -> list[dict] | None:
    geo = _resolve_crs_transform(tile_id, cache_dir, data_dir)
    if geo is None:
        print(f"  [SKIP] {tile_id}: cannot resolve CRS — provide --data_dir or regenerate cache")
        return None

    crs, transform = geo

    # Apply forest gate if probability sidecar is available
    if probs_dir is not None:
        sidecar = probs_dir / f"{tile_id}.npz"
        if sidecar.exists():
            with np.load(sidecar, allow_pickle=False) as z:
                prob_pre = z["forest_prob_pre"]
            non_forest = prob_pre < NON_FOREST_THRESHOLD
            pred = pred.copy()
            pred[non_forest] = 0
        else:
            print(f"  [WARN] {tile_id}: no prob sidecar in {probs_dir} — forest gate skipped")

    if pred.sum() == 0:
        print(f"  [SKIP] {tile_id}: no deforestation pixels after gating")
        return None

    H, W = pred.shape
    with MemoryFile() as mf:
        with mf.open(driver="GTiff", height=H, width=W, count=1,
                     dtype="uint8", crs=crs, transform=transform) as dst:
            dst.write(pred.astype(np.uint8)[None])
        try:
            geojson = raster_to_geojson(mf.name, output_path=None, min_area_ha=min_area_ha)
        except ValueError as e:
            print(f"  [SKIP] {tile_id}: {e}")
            return None

    features = geojson["features"]
    if time_step is not None:
        for f in features:
            f.setdefault("properties", {})["time_step"] = int(time_step)

    print(f"  {tile_id}: {len(features)} polygon(s)")
    return features


def main() -> None:
    ap = argparse.ArgumentParser(description="Export per-tile predictions to submission GeoJSON")
    ap.add_argument("--preds_dir",  type=Path, required=True,
                    help="Per-tile prediction .npz files (must contain 'pred' (H,W) uint8)")
    ap.add_argument("--cache_dir",  type=Path, default=REPO_ROOT / "cache",
                    help="Tile .npz cache — used for _crs_wkt/_transform if present")
    ap.add_argument("--data_dir",   type=Path, default=REPO_ROOT / "data",
                    help="Raw data root (fallback CRS source from S2 GeoTIFFs)")
    ap.add_argument("--probs_dir",  type=Path, default=REPO_ROOT / "cache_probs",
                    help="Probability sidecar dir for forest gate (omit to skip gate)")
    ap.add_argument("--out",        type=Path, default=REPO_ROOT / "submission" / "submission.geojson")
    ap.add_argument("--min_area_ha", type=float, default=0.5)
    ap.add_argument("--no_gate",    action="store_true",
                    help="Disable the forest-probability gate even if sidecars exist")
    args = ap.parse_args()

    probs_dir = None if args.no_gate else args.probs_dir

    pred_files = sorted(args.preds_dir.glob("*.npz"))
    if not pred_files:
        print(f"No .npz files in {args.preds_dir}"); sys.exit(1)

    print(f"[export] {len(pred_files)} tiles -> {args.out}")
    all_features: list[dict] = []

    for pf in pred_files:
        tile_id = pf.stem
        with np.load(pf, allow_pickle=True) as z:
            pred      = z["pred"]
            time_step = int(z["time_step"]) if "time_step" in z.files else None

        feats = export_tile(tile_id, pred, args.cache_dir, args.data_dir,
                            probs_dir, args.min_area_ha, time_step)
        if feats:
            all_features.extend(feats)

    if not all_features:
        print("[export] ERROR: no features produced"); sys.exit(1)

    submission = {"type": "FeatureCollection", "features": all_features}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(submission, f)
    print(f"[export] {len(all_features)} polygons -> {args.out}")


if __name__ == "__main__":
    main()
