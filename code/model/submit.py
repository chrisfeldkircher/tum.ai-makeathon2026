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
    PROB_FEATURE_KEYS,
    ReferenceGrid,
    TEMPORAL_FEATURE_KEYS,
    TileInventory,
    build_inventory,
    cache_tile,
    load_probability_sidecar,
    preprocess_tile,
    stack_features,
)
from model.dann_temporal_unet import inference_pipeline


def _needs_prob_sidecar(feature_keys: Iterable[str]) -> bool:
    return any(k in PROB_FEATURE_KEYS for k in feature_keys)


DEFAULT_FOREST_GATE_THRESHOLD = 0.30  # matches LearnedForestMasker.tier_non_forest_max


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


def _load_tile_tensors(
    ti: TileInventory,
    feature_keys: tuple[str, ...],
    cache_dir: Optional[Path],
    probs_dir: Optional[Path],
) -> dict[str, np.ndarray]:
    """Mirror the training dataset's load path: cached .npz + optional prob sidecar.

    Falls back to a live `preprocess_tile` when `cache_dir` is None. Raises if
    PROB feature keys are requested but no sidecar is available — silently
    zero-filling those channels would produce garbage predictions because the
    baseline learned to rely on them.
    """
    if cache_dir is not None:
        cache_path = cache_tile(ti, cache_dir)
        with np.load(cache_path, allow_pickle=False) as npz:
            tensors = {k: npz[k] for k in npz.files}
    else:
        tensors = preprocess_tile(ti)

    if _needs_prob_sidecar(feature_keys):
        if probs_dir is None:
            raise RuntimeError(
                f"feature_keys include PROB channels {PROB_FEATURE_KEYS} but probs_dir is None. "
                "Pass probs_dir=... and pre-generate sidecars via "
                "`python code/reports/p2_post_period.py --cache_dir <cache> --out_dir <probs>`."
            )
        tensors.update(load_probability_sidecar(ti.tile_id, probs_dir))

    return tensors


def predict_tile_prob(
    model: nn.Module,
    ti: TileInventory,
    device: torch.device,
    patch_size: int = 256,
    overlap: int = 64,
    feature_keys: tuple[str, ...] = TEMPORAL_FEATURE_KEYS,
    cache_dir: Optional[Path] = None,
    probs_dir: Optional[Path] = None,
    forest_gate_threshold: float = DEFAULT_FOREST_GATE_THRESHOLD,
) -> tuple[np.ndarray, Optional[np.ndarray], ReferenceGrid]:
    """Run model inference once; return (prob_HW, forest_mask_HW_or_None, ref).

    Split out from `predict_tile` so threshold / gate / min_area sweeps can reuse
    the same expensive forward pass.
    """
    tensors = _load_tile_tensors(ti, feature_keys, cache_dir, probs_dir)
    if probs_dir is not None and "forest_prob_pre" not in tensors:
        try:
            tensors.update(load_probability_sidecar(ti.tile_id, probs_dir))
        except FileNotFoundError:
            pass
    stack = stack_features(tensors, feature_keys=feature_keys)
    tile_tensor = torch.from_numpy(stack).float()

    forest_mask_np: Optional[np.ndarray] = None
    forest_mask_t: Optional[torch.Tensor] = None
    if "forest_prob_pre" in tensors:
        forest_mask_np = (tensors["forest_prob_pre"] >= forest_gate_threshold).astype(np.float32)
        forest_mask_t = torch.from_numpy(forest_mask_np)
    elif "forest_mask_2020" in tensors:
        forest_mask_np = tensors["forest_mask_2020"].astype(np.float32)
        forest_mask_t = torch.from_numpy(forest_mask_np)

    out = inference_pipeline(
        _maybe_wrap_for_inference(model), tile_tensor,
        forest_mask=forest_mask_t,
        patch_size=patch_size, overlap=overlap,
        threshold=0.5, device=device,
    )
    prob = out["prob"].numpy()

    ref_source = max(ti.s2_paths.values(), key=lambda p: _scene_pixels(p))
    ref = ReferenceGrid.from_s2(ref_source)
    return prob, forest_mask_np, ref


def _threshold_and_gate(
    prob: np.ndarray,
    forest_mask: Optional[np.ndarray],
    threshold: float,
    use_forest_gate: bool,
) -> np.ndarray:
    pred = (prob >= threshold).astype(np.uint8)
    if use_forest_gate and forest_mask is not None:
        pred = pred * (forest_mask > 0.5).astype(np.uint8)
    return pred


def predict_tile(
    model: nn.Module,
    ti: TileInventory,
    device: torch.device,
    patch_size: int = 256,
    overlap: int = 64,
    threshold: float = 0.5,
    use_forest_gate: bool = True,
    feature_keys: tuple[str, ...] = TEMPORAL_FEATURE_KEYS,
    cache_dir: Optional[Path] = None,
    probs_dir: Optional[Path] = None,
    forest_gate_threshold: float = DEFAULT_FOREST_GATE_THRESHOLD,
) -> tuple[np.ndarray, ReferenceGrid]:
    """Run inference on one tile, returning (binary_pred_HW, reference_grid)."""
    prob, forest_mask, ref = predict_tile_prob(
        model, ti, device,
        patch_size=patch_size, overlap=overlap,
        feature_keys=feature_keys,
        cache_dir=cache_dir, probs_dir=probs_dir,
        forest_gate_threshold=forest_gate_threshold,
    )
    pred = _threshold_and_gate(prob, forest_mask, threshold, use_forest_gate)
    return pred, ref


def _scene_pixels(path: Path) -> int:
    with rasterio.open(path) as src:
        return src.width * src.height


def _polygonize_and_collect(
    pred: np.ndarray,
    ref: ReferenceGrid,
    tid: str,
    tile_dir: Path,
    tif_name: str,
    gj_name: str,
    min_area_ha: float,
    default_time_step: Optional[int],
) -> list[dict]:
    tif_path = tile_dir / tif_name
    gj_path = tile_dir / gj_name
    _write_binary_geotiff(pred, ref, tif_path)
    fc = raster_to_geojson(tif_path, output_path=gj_path, min_area_ha=min_area_ha)
    feats: list[dict] = []
    for feat in fc.get("features", []):
        feat.setdefault("properties", {})
        feat["properties"]["tile_id"] = tid
        feat["properties"].setdefault("time_step", default_time_step)
        feats.append(feat)
    return feats


def build_submission(
    model: nn.Module,
    data_root: Path | str,
    out_dir: Path | str,
    split: str = "test",
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
    min_area_ha: float = 0.5,
    use_forest_gate: bool = True,
    tile_ids: Optional[Iterable[str]] = None,
    feature_keys: tuple[str, ...] = TEMPORAL_FEATURE_KEYS,
    cache_dir: Optional[Path | str] = None,
    probs_dir: Optional[Path | str] = None,
    default_time_step: Optional[int] = None,
    forest_gate_threshold: float = DEFAULT_FOREST_GATE_THRESHOLD,
) -> Path:
    """Run inference on every tile in `split` and write one merged submission.geojson.

    Intermediate per-tile GeoTIFFs and GeoJSONs are written under `out_dir/tiles/`
    so you can inspect them; the final merged file is `out_dir/submission.geojson`.

    `default_time_step` (e.g. 2304) is assigned to every polygon's properties so
    the Year-accuracy metric is non-zero without ground truth.
    """
    out_dir = Path(out_dir)
    tile_dir = out_dir / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(cache_dir) if cache_dir is not None else None
    probs_dir = Path(probs_dir) if probs_dir is not None else None

    if _needs_prob_sidecar(feature_keys) and probs_dir is None:
        raise RuntimeError(
            "feature_keys request forest_prob_* channels but probs_dir=None. "
            "Pass probs_dir=Path('../cache_probs') and run the sidecar step first:\n"
            "  python code/reports/p2_post_period.py --cache_dir ../cache --out_dir ../cache_probs"
        )

    if device is None:
        device = next(model.parameters()).device

    inventory = build_inventory(Path(data_root), split=split)
    if tile_ids is not None:
        inventory = {tid: ti for tid, ti in inventory.items() if tid in set(tile_ids)}

    all_features: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for tid, ti in sorted(inventory.items()):
        try:
            prob, forest_mask, ref = predict_tile_prob(
                model, ti, device=device,
                feature_keys=feature_keys,
                cache_dir=cache_dir, probs_dir=probs_dir,
                forest_gate_threshold=forest_gate_threshold,
            )
            pred = _threshold_and_gate(prob, forest_mask, threshold, use_forest_gate)
            if pred.sum() == 0:
                print(f"  {tid}: no positive pixels — skipping")
                continue
            feats = _polygonize_and_collect(
                pred, ref, tid, tile_dir,
                tif_name=f"{tid}.tif", gj_name=f"{tid}.geojson",
                min_area_ha=min_area_ha,
                default_time_step=default_time_step,
            )
            all_features.extend(feats)
            print(f"  {tid}: {len(feats)} polygons")
        except Exception as e:
            skipped.append((tid, str(e)))
            print(f"  {tid}: SKIP — {e}")

    merged = {"type": "FeatureCollection", "features": all_features}
    out_path = out_dir / "submission.geojson"
    with open(out_path, "w") as f:
        json.dump(merged, f)

    print(f"\nWrote {out_path} — {len(all_features)} polygons, {len(skipped)} tiles skipped.")
    return out_path


def sweep_submission(
    model: nn.Module,
    data_root: Path | str,
    out_root: Path | str,
    thresholds: Iterable[float] = (0.5, 0.6, 0.7, 0.8, 0.85),
    min_areas_ha: Iterable[float] = (0.5, 2.0, 5.0, 10.0),
    split: str = "test",
    device: Optional[torch.device] = None,
    use_forest_gate: bool = True,
    tile_ids: Optional[Iterable[str]] = None,
    feature_keys: tuple[str, ...] = TEMPORAL_FEATURE_KEYS,
    cache_dir: Optional[Path | str] = None,
    probs_dir: Optional[Path | str] = None,
    default_time_step: Optional[int] = 2304,
) -> dict[tuple[float, float], Path]:
    """Run model inference once per tile, then emit one submission per (threshold, min_area_ha).

    Output: `out_root/t{t}_m{m}/submission.geojson` per combo.
    Returns a {(threshold, min_area): path} map for downstream comparison.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    probs_dir = Path(probs_dir) if probs_dir is not None else None

    if _needs_prob_sidecar(feature_keys) and probs_dir is None:
        raise RuntimeError(
            "feature_keys request forest_prob_* channels but probs_dir=None."
        )

    if device is None:
        device = next(model.parameters()).device

    inventory = build_inventory(Path(data_root), split=split)
    if tile_ids is not None:
        inventory = {tid: ti for tid, ti in inventory.items() if tid in set(tile_ids)}

    thresholds = list(thresholds)
    min_areas_ha = list(min_areas_ha)

    # prob_cache[tid] = (prob, forest_mask, ref); None means inference failed
    prob_cache: dict[str, Optional[tuple[np.ndarray, Optional[np.ndarray], ReferenceGrid]]] = {}
    for tid, ti in sorted(inventory.items()):
        try:
            prob_cache[tid] = predict_tile_prob(
                model, ti, device=device,
                feature_keys=feature_keys,
                cache_dir=cache_dir, probs_dir=probs_dir,
            )
            print(f"  inference {tid}: prob shape {prob_cache[tid][0].shape}")
        except Exception as e:
            prob_cache[tid] = None
            print(f"  inference {tid}: SKIP — {e}")

    results: dict[tuple[float, float], Path] = {}
    for thr in thresholds:
        for mah in min_areas_ha:
            combo_dir = out_root / f"t{thr:.2f}_m{mah:g}"
            tile_dir = combo_dir / "tiles"
            tile_dir.mkdir(parents=True, exist_ok=True)

            all_features: list[dict] = []
            for tid, cached in prob_cache.items():
                if cached is None:
                    continue
                prob, forest_mask, ref = cached
                pred = _threshold_and_gate(prob, forest_mask, thr, use_forest_gate)
                if pred.sum() == 0:
                    continue
                feats = _polygonize_and_collect(
                    pred, ref, tid, tile_dir,
                    tif_name=f"{tid}.tif", gj_name=f"{tid}.geojson",
                    min_area_ha=mah,
                    default_time_step=default_time_step,
                )
                all_features.extend(feats)

            merged = {"type": "FeatureCollection", "features": all_features}
            out_path = combo_dir / "submission.geojson"
            with open(out_path, "w") as f:
                json.dump(merged, f)
            print(f"[t={thr:.2f} m={mah:g}] {len(all_features)} polygons → {out_path}")
            results[(thr, mah)] = out_path

    return results
