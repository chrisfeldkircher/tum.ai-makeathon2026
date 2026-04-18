"""Export per-tile binary prediction masks to a single submission GeoJSON.

Usage (once you have per-tile uint8 prediction arrays saved as .npz):

    python code/submission/export.py \\
        --preds_dir ./predictions \\
        --cache_dir ./cache \\
        --out     submission/submission.geojson

Expected input: for each tile, `predictions/<tile_id>.npz` containing a
`pred` array of shape (H, W) uint8, where 1 = deforestation detected.

The tile cache (cache/<tile_id>.npz) must contain `_crs_wkt` and `_transform`
(written by code/data/data.py since the forest-integration update). If these
keys are missing, the tile is skipped with a warning — regenerate the cache.

Optional: if your model also predicts `time_step` (YYMM int, e.g. 2204 for
April 2022), include it as `time_step` in the prediction npz.  It will be
propagated to each polygon's properties.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import Affine

# Allow importing from repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from problem.submission_utils import raster_to_geojson  # noqa: E402


def export_tile(
    tile_id: str,
    pred: np.ndarray,
    cache_dir: Path,
    min_area_ha: float,
    time_step: int | None,
) -> list[dict] | None:
    """Vectorise one tile's prediction mask → list of GeoJSON Feature dicts."""
    cache_path = cache_dir / f"{tile_id}.npz"
    if not cache_path.exists():
        print(f"  [SKIP] {tile_id}: no cache file at {cache_path}")
        return None

    with np.load(cache_path, allow_pickle=True) as z:
        if "_crs_wkt" not in z.files or "_transform" not in z.files:
            print(f"  [SKIP] {tile_id}: cache missing _crs_wkt/_transform — regenerate cache")
            return None
        crs = CRS.from_wkt(z["_crs_wkt"].item().decode())
        transform = Affine(*z["_transform"])

    H, W = pred.shape
    if pred.sum() == 0:
        print(f"  [SKIP] {tile_id}: no deforestation pixels predicted")
        return None

    # Write prediction to an in-memory GeoTIFF, then vectorise via submission_utils
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


def main():
    ap = argparse.ArgumentParser(description="Export predictions to submission GeoJSON")
    ap.add_argument("--preds_dir", type=Path, required=True,
                    help="Directory of per-tile prediction .npz files (must contain 'pred' array)")
    ap.add_argument("--cache_dir", type=Path, default=REPO_ROOT / "cache")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "submission" / "submission.geojson")
    ap.add_argument("--min_area_ha", type=float, default=0.5,
                    help="Minimum polygon area in hectares (default 0.5)")
    args = ap.parse_args()

    pred_files = sorted(args.preds_dir.glob("*.npz"))
    if not pred_files:
        print(f"No .npz files found in {args.preds_dir}")
        sys.exit(1)

    print(f"[export] {len(pred_files)} prediction files -> {args.out}")
    all_features = []

    for pf in pred_files:
        tile_id = pf.stem
        with np.load(pf, allow_pickle=True) as z:
            pred = z["pred"]
            time_step = int(z["time_step"]) if "time_step" in z.files else None

        feats = export_tile(tile_id, pred, args.cache_dir, args.min_area_ha, time_step)
        if feats:
            all_features.extend(feats)

    if not all_features:
        print("[export] ERROR: no features produced — check predictions and cache")
        sys.exit(1)

    submission = {"type": "FeatureCollection", "features": all_features}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(submission, f)
    print(f"[export] wrote {len(all_features)} total polygons to {args.out}")


if __name__ == "__main__":
    main()
