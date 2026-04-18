"""
submit.py — Turn a trained DANN_Temporal_UNet into a leaderboard submission.

Pipeline per tile:
    TileInventory
      → preprocess_tile     (dict of rasters + ReferenceGrid size)
      → stack_features(TEMPORAL_FEATURE_KEYS)  (232, H, W)
      → inference_pipeline  (prob, pred, pred_gated)
      → write pred_gated as single-band GeoTIFF (CRS + transform from ref)
      → raster_to_geojson   (per-tile FeatureCollection)

A second step merges per-tile FeatureCollections into one submission .geojson.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import rasterio
import torch
import torch.nn as nn

# The submission utility lives outside `code/`, add it to the path on import.
_PROBLEM_DIR = Path(__file__).resolve().parents[2] / "problem"
if str(_PROBLEM_DIR) not in sys.path:
    sys.path.insert(0, str(_PROBLEM_DIR))
from submission_utils import raster_to_geojson  # noqa: E402

from data.data import (
    DEFAULT_FEATURE_KEYS,
    ReferenceGrid,
    TEMPORAL_FEATURE_KEYS,
    TileInventory,
    build_inventory,
    preprocess_tile,
    stack_features,
)
from model.dann_temporal_unet import inference_pipeline


class _GRLCompatAdapter(nn.Module):
    """Wrap a non-DANN model so `inference_pipeline` can call it with grl_lambda."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, grl_lambda: float = 0.0):
        return self.model(x)


def _maybe_wrap_for_inference(model: nn.Module) -> nn.Module:
    """Wrap the model if its forward() doesn't accept grl_lambda."""
    try:
        sig = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return model
    if "grl_lambda" in sig.parameters:
        return model
    return _GRLCompatAdapter(model)


def _write_binary_geotiff(pred: np.ndarray, ref: ReferenceGrid, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        width=ref.width, height=ref.height,
        count=1, dtype="uint8",
        crs=ref.crs, transform=ref.transform,
        compress="deflate", nodata=0,
    ) as dst:
        dst.write(pred.astype(np.uint8), 1)


def predict_tile(
    model: nn.Module,
    ti: TileInventory,
    device: torch.device,
    patch_size: int = 256,
    overlap: int = 64,
    threshold: float = 0.5,
    use_forest_gate: bool = True,
    feature_keys: tuple[str, ...] = TEMPORAL_FEATURE_KEYS,
) -> tuple[np.ndarray, ReferenceGrid]:
    """Run inference on one tile, returning (binary_pred_HW, reference_grid)."""
    tensors = preprocess_tile(ti)
    stack = stack_features(tensors, feature_keys=feature_keys)
    tile_tensor = torch.from_numpy(stack).float()

    forest_mask = None
    if use_forest_gate and "forest_mask_2020" in tensors:
        forest_mask = torch.from_numpy(tensors["forest_mask_2020"].astype(np.float32))

    out = inference_pipeline(
        _maybe_wrap_for_inference(model), tile_tensor,
        forest_mask=forest_mask,
        patch_size=patch_size, overlap=overlap,
        threshold=threshold, device=device,
    )
    pred = out["pred_gated"].numpy() if use_forest_gate else out["pred"].numpy()

    ref_source = max(ti.s2_paths.values(), key=lambda p: _scene_pixels(p))
    ref = ReferenceGrid.from_s2(ref_source)
    return pred, ref


def _scene_pixels(path: Path) -> int:
    with rasterio.open(path) as src:
        return src.width * src.height


def build_submission(
    model: nn.Module,
    data_root: Path | str,
    out_dir: Path | str,
    split: str = "test",
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
    min_area_ha: float = 0.5,
    tile_ids: Optional[Iterable[str]] = None,
    feature_keys: tuple[str, ...] = TEMPORAL_FEATURE_KEYS,
) -> Path:
    """Run inference on every tile in `split` and write one merged submission.geojson.

    Intermediate per-tile GeoTIFFs and GeoJSONs are written under `out_dir/tiles/`
    so you can inspect them; the final merged file is `out_dir/submission.geojson`.
    """
    out_dir = Path(out_dir)
    tile_dir = out_dir / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = next(model.parameters()).device

    inventory = build_inventory(Path(data_root), split=split)
    if tile_ids is not None:
        inventory = {tid: ti for tid, ti in inventory.items() if tid in set(tile_ids)}

    all_features: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for tid, ti in sorted(inventory.items()):
        try:
            pred, ref = predict_tile(
                model, ti, device=device, threshold=threshold,
                feature_keys=feature_keys,
            )
            if pred.sum() == 0:
                print(f"  {tid}: no positive pixels — skipping")
                continue
            tif_path = tile_dir / f"{tid}.tif"
            gj_path  = tile_dir / f"{tid}.geojson"
            _write_binary_geotiff(pred, ref, tif_path)
            fc = raster_to_geojson(tif_path, output_path=gj_path, min_area_ha=min_area_ha)
            for feat in fc.get("features", []):
                feat.setdefault("properties", {})
                feat["properties"]["tile_id"] = tid
                feat["properties"].setdefault("time_step", None)
            all_features.extend(fc.get("features", []))
            print(f"  {tid}: {len(fc.get('features', []))} polygons")
        except Exception as e:
            skipped.append((tid, str(e)))
            print(f"  {tid}: SKIP — {e}")

    merged = {"type": "FeatureCollection", "features": all_features}
    out_path = out_dir / "submission.geojson"
    with open(out_path, "w") as f:
        json.dump(merged, f)

    print(f"\nWrote {out_path} — {len(all_features)} polygons, {len(skipped)} tiles skipped.")
    return out_path
