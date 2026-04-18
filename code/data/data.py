"""Data pipeline for osapiens Makeathon 2026 — Deforestation Detection.

Pipeline stages
---------------
1. Tile inventory             — discover tiles from the on-disk folder layout.
2. Raw loaders                — load S1 / S2 / AEF / labels with rasterio.
3. CRS alignment              — reproject every modality onto a common UTM grid.
4. Spectral indices           — NDVI, NBR, NDMI, EVI from S2 bands.
5. Temporal composites        — pre/post-2020 medians & stds per index.
6. Label decoding + fusion    — decode RADD / GLAD-L / GLAD-S2 date/confidence
                                 encodings, filter post-2020, build a consensus
                                 mask + per-pixel confidence.
7. Forest mask (2020)         — NDVI threshold on a pre-2020 composite.
8. On-disk caching            — cache preprocessed tensors as .npz so the
                                 DataLoader doesn't reproject every epoch.
9. PyTorch Dataset/DataLoader — random-crop patches for training, full tile
                                 for inference.

Folder layout expected (relative to `root`):

    sentinel-1/{split}/{tile_id}__s1_rtc/{tile_id}__s1_rtc_{Y}_{M}_{orbit}.tif
    sentinel-2/{split}/{tile_id}__s2_l2a/{tile_id}__s2_l2a_{Y}_{M}.tif
    aef-embeddings/{split}/{tile_id}_{Y}.tiff
    labels/train/gladl/  gladl_{tile_id}_alert{YY}.tif + gladl_{tile_id}_alertDate{YY}.tif
    labels/train/glads2/ glads2_{tile_id}_alert.tif   + glads2_{tile_id}_alertDate.tif
    labels/train/radd/   radd_{tile_id}_labels.tif
    metadata/{train,test}_tiles.geojson
"""

from __future__ import annotations

import logging
import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np


@contextmanager
def _quiet_nan_reductions():
    """Silence expected 'All-NaN slice' / 'Degrees of freedom <= 0' warnings from
    numpy reductions over sparse temporal stacks — handled by np.nan_to_num."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered",
                                category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0",
                                category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="Mean of empty slice",
                                category=RuntimeWarning)
        yield
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from scipy.ndimage import uniform_filter as ndi_uniform

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # torch is optional at import time
    torch = None
    Dataset = object  # type: ignore[assignment,misc]
    DataLoader = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


S2_BANDS = {
    "B01": 1, "B02": 2, "B03": 3, "B04": 4, "B05": 5, "B06": 6,
    "B07": 7, "B08": 8, "B8A": 9, "B09": 10, "B11": 11, "B12": 12,
}
S2_N_BANDS = 12
S2_MAX_REFLECTANCE = 10_000.0

AEF_N_DIMS = 64

# Label date origins (see challenge.ipynb §5).
RADD_EPOCH    = date(2014, 12, 31)
GLADS2_EPOCH  = date(2019, 1, 1)

# 2019 is not shipped by this dataset (verified across S2/S1/AEF). The "pre"
# window is a single year: 2020. Keeping the tuple singleton-form so downstream
# code that iterates over pre_years (e.g. _select_paths_by_year) still works.
DEFAULT_PRE_YEARS  = (2020,)
DEFAULT_POST_YEARS = (2021, 2022, 2023, 2024)

# Tiles where the raw Sentinel-2 provider shipped unusable scenes and no amount
# of reference-grid / partial-scene logic can recover them. Verified via the
# `debug_data.py` scan (see 2026-04 triage).
#   18NYH_9_9: 71/72 scenes are 4x6 or 1004x6 slivers; the single full-size
#     1004x1004 scene (2020-10) is 100% cloud (p50=6084). Nothing to composite.
BROKEN_TILES: frozenset[str] = frozenset({"18NYH_9_9"})


TILE_ID_RE = re.compile(r"^[A-Z0-9]+_\d+_\d+$")

_S2_FILE_RE = re.compile(r".+_s2_l2a_(\d{4})_(\d{1,2})\.tif$")
_S1_FILE_RE = re.compile(r".+_s1_rtc_(\d{4})_(\d{1,2})_(ascending|descending)\.tif$")
_AEF_FILE_RE = re.compile(r"^(.+)_(\d{4})\.tiff?$")


@dataclass
class TileInventory:
    """Paths available for a single tile + split."""
    tile_id: str
    split: str
    s1_paths: dict[tuple[int, int, str], Path] = field(default_factory=dict)
    s2_paths: dict[tuple[int, int], Path] = field(default_factory=dict)
    aef_paths: dict[int, Path] = field(default_factory=dict)
    radd_path: Path | None = None
    gladl_alert_paths: dict[int, Path] = field(default_factory=dict)
    gladl_date_paths: dict[int, Path] = field(default_factory=dict)
    glads2_alert_path: Path | None = None
    glads2_date_path: Path | None = None

    def has_any_labels(self) -> bool:
        return bool(
            self.radd_path or self.glads2_alert_path or self.gladl_alert_paths
        )


def build_inventory(root: Path | str, split: str) -> dict[str, TileInventory]:
    """Scan the on-disk layout and produce one TileInventory per tile_id.

    Missing modalities or months are simply absent from the inventory — the
    dataset handles that downstream.
    """
    root = Path(root)
    inv: dict[str, TileInventory] = {}

    s2_root = root / "sentinel-2" / split
    if s2_root.exists():
        for tile_dir in s2_root.iterdir():
            if not tile_dir.is_dir():
                continue
            tile_id = tile_dir.name.replace("__s2_l2a", "")
            if not TILE_ID_RE.match(tile_id):
                continue
            ti = inv.setdefault(tile_id, TileInventory(tile_id, split))
            for f in tile_dir.glob("*.tif"):
                m = _S2_FILE_RE.match(f.name)
                if m:
                    ti.s2_paths[(int(m.group(1)), int(m.group(2)))] = f

    s1_root = root / "sentinel-1" / split
    if s1_root.exists():
        for tile_dir in s1_root.iterdir():
            if not tile_dir.is_dir():
                continue
            tile_id = tile_dir.name.replace("__s1_rtc", "")
            if not TILE_ID_RE.match(tile_id):
                continue
            ti = inv.setdefault(tile_id, TileInventory(tile_id, split))
            for f in tile_dir.glob("*.tif"):
                m = _S1_FILE_RE.match(f.name)
                if m:
                    ti.s1_paths[(int(m.group(1)), int(m.group(2)), m.group(3))] = f

    aef_root = root / "aef-embeddings" / split
    if aef_root.exists():
        for f in aef_root.glob("*.tif*"):
            m = _AEF_FILE_RE.match(f.name)
            if not m:
                continue
            tile_id, year = m.group(1), int(m.group(2))
            if not TILE_ID_RE.match(tile_id):
                continue
            ti = inv.setdefault(tile_id, TileInventory(tile_id, split))
            ti.aef_paths[year] = f

    if split == "train":
        lbl_root = root / "labels" / "train"

        radd_dir = lbl_root / "radd"
        if radd_dir.exists():
            for f in radd_dir.glob("radd_*_labels.tif"):
                tile_id = f.stem.replace("radd_", "").replace("_labels", "")
                if tile_id in inv:
                    inv[tile_id].radd_path = f

        gladl_dir = lbl_root / "gladl"
        if gladl_dir.exists():
            for f in gladl_dir.glob("gladl_*_alert*.tif"):
                m = re.match(r"^gladl_(.+)_alert(Date)?(\d{2})\.tif$", f.name)
                if not m:
                    continue
                tile_id, is_date, yy = m.group(1), m.group(2) is not None, int(m.group(3))
                if tile_id not in inv:
                    continue
                if is_date:
                    inv[tile_id].gladl_date_paths[yy] = f
                else:
                    inv[tile_id].gladl_alert_paths[yy] = f

        glads2_dir = lbl_root / "glads2"
        if glads2_dir.exists():
            for f in glads2_dir.glob("glads2_*_alert*.tif"):
                m = re.match(r"^glads2_(.+)_alert(Date)?\.tif$", f.name)
                if not m:
                    continue
                tile_id, is_date = m.group(1), m.group(2) is not None
                if tile_id not in inv:
                    continue
                if is_date:
                    inv[tile_id].glads2_date_path = f
                else:
                    inv[tile_id].glads2_alert_path = f

    return inv


@dataclass
class ReferenceGrid:
    """The canonical (CRS, transform, shape) every modality is resampled to."""
    crs: rasterio.crs.CRS
    transform: rasterio.Affine
    height: int
    width: int

    @classmethod
    def from_s2(cls, path: Path) -> "ReferenceGrid":
        with rasterio.open(path) as src:
            return cls(src.crs, src.transform, src.height, src.width)


def _reproject_to(ref: ReferenceGrid, path: Path, bands: list[int] | None = None,
                  resampling: Resampling = Resampling.bilinear,
                  dtype: str = "float32", fill: float = 0.0) -> np.ndarray:
    """Open `path` and reproject requested bands onto the reference grid."""
    with rasterio.open(path) as src:
        if bands is None:
            bands = list(range(1, src.count + 1))
        out = np.full((len(bands), ref.height, ref.width), fill, dtype=dtype)
        for i, b in enumerate(bands):
            reproject(
                source=rasterio.band(src, b),
                destination=out[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                resampling=resampling,
            )
    return out


def _safe_div(num: np.ndarray, den: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return num / (den + eps)


def compute_indices(s2: np.ndarray) -> dict[str, np.ndarray]:
    """Compute NDVI / NBR / NDMI / EVI from a 12-band S2 stack in [0,1].

    Input shape: (12, H, W). Output arrays: (H, W) float32.
    """
    red  = s2[S2_BANDS["B04"] - 1]
    nir  = s2[S2_BANDS["B08"] - 1]
    swir1 = s2[S2_BANDS["B11"] - 1]
    swir2 = s2[S2_BANDS["B12"] - 1]
    blue  = s2[S2_BANDS["B02"] - 1]

    ndvi = _safe_div(nir - red,   nir + red)
    nbr  = _safe_div(nir - swir2, nir + swir2)
    ndmi = _safe_div(nir - swir1, nir + swir1)
    evi  = 2.5 * _safe_div(nir - red, nir + 6 * red - 7.5 * blue + 1.0)
    return {"ndvi": ndvi.astype(np.float32),
            "nbr":  nbr.astype(np.float32),
            "ndmi": ndmi.astype(np.float32),
            "evi":  evi.astype(np.float32)}


def _load_s2_normalised(path: Path, ref: ReferenceGrid) -> np.ndarray:
    """Load S2 tile, reproject, scale to [0,1]."""
    arr = _reproject_to(ref, path, bands=list(range(1, S2_N_BANDS + 1)),
                        resampling=Resampling.bilinear, dtype="float32")
    arr = arr / S2_MAX_REFLECTANCE
    # Headroom above 1.0 for bright surfaces (clouds, snow) that exceed the
    # nominal TOA reflectance scaling factor.
    return np.clip(arr, 0.0, 1.5)


def s2_temporal_composite(
    paths: Iterable[Path],
    ref: ReferenceGrid,
    compute_trajectory: bool = False,
    trajectory_indices: tuple[str, ...] = ("ndvi", "nbr", "ndmi"),
) -> dict[str, np.ndarray]:
    """Median + std composite across a set of S2 scenes.

    Always returns:
        'band_median' (12, H, W), 'band_std' (12, H, W),
        '<idx>_median', '<idx>_std' for idx ∈ {ndvi, nbr, ndmi, evi}.

    When `compute_trajectory=True`, additionally emits for each idx in
    `trajectory_indices`:
        '<idx>_p10', '<idx>_p90', '<idx>_slope'
        'n_scenes_valid'  (shared across all trajectory indices)

    Missing pixels (B08 == 0 from the reproject fill) are treated as NaN for
    every index so medians and especially p10 aren't dragged toward zero by
    black-border padding. Previously indices were computed with `_safe_div`
    which turns nodata into a finite zero, biasing the median of partial-
    coverage tiles — this fixes that.
    """
    stacks, index_stacks = [], {"ndvi": [], "nbr": [], "ndmi": [], "evi": []}
    for p in paths:
        s2 = _load_s2_normalised(p, ref)
        stacks.append(s2)
        band_valid = s2[S2_BANDS["B08"] - 1] > 0  # per-scene nodata mask
        idx = compute_indices(s2)
        for k, v in idx.items():
            index_stacks[k].append(
                np.where(band_valid & np.isfinite(v), v, np.nan))

    if not stacks:
        empty = np.zeros((S2_N_BANDS, ref.height, ref.width), dtype=np.float32)
        empty_idx = np.zeros((ref.height, ref.width), dtype=np.float32)
        out: dict[str, np.ndarray] = {
            "band_median": empty, "band_std": empty,
            **{f"{k}_median": empty_idx.copy() for k in index_stacks},
            **{f"{k}_std":    empty_idx.copy() for k in index_stacks},
        }
        if compute_trajectory:
            for k in trajectory_indices:
                out[f"{k}_p10"]   = empty_idx.copy()
                out[f"{k}_p90"]   = empty_idx.copy()
                out[f"{k}_slope"] = empty_idx.copy()
            out["n_scenes_valid"] = empty_idx.copy()
        return out

    stk = np.stack(stacks, axis=0)
    mask = stk > 0
    stk_ma = np.where(mask, stk, np.nan)
    with _quiet_nan_reductions():
        band_median = np.nanmedian(stk_ma, axis=0).astype(np.float32)
        band_std = np.nanstd(stk_ma, axis=0).astype(np.float32)
    band_median = np.nan_to_num(band_median, nan=0.0)
    band_std    = np.nan_to_num(band_std,    nan=0.0)

    out = {"band_median": band_median, "band_std": band_std}
    for k, lst in index_stacks.items():
        ik = np.stack(lst, axis=0)
        with _quiet_nan_reductions():
            med = np.nanmedian(ik, axis=0)
            std = np.nanstd(ik, axis=0)
        out[f"{k}_median"] = np.nan_to_num(med, nan=0.0).astype(np.float32)
        out[f"{k}_std"]    = np.nan_to_num(std, nan=0.0).astype(np.float32)

        if compute_trajectory and k in trajectory_indices:
            with _quiet_nan_reductions():
                p10 = np.nanpercentile(ik, 10, axis=0)
                p90 = np.nanpercentile(ik, 90, axis=0)
            out[f"{k}_p10"]   = np.nan_to_num(p10, nan=0.0).astype(np.float32)
            out[f"{k}_p90"]   = np.nan_to_num(p90, nan=0.0).astype(np.float32)
            out[f"{k}_slope"] = _pixel_slope(ik)

    if compute_trajectory:
        # Valid-scene count is shared across all indices because they live on
        # the same scene cadence (they're derived from the same B08 mask).
        ik = np.stack(index_stacks["ndvi"], axis=0)
        out["n_scenes_valid"] = np.isfinite(ik).sum(axis=0).astype(np.float32)

    return out


def _pixel_slope(stk_ma: np.ndarray) -> np.ndarray:
    """Per-pixel linear slope across a (T, H, W) stack. NaN observations are
    skipped. Time axis is scene-index (0..T-1), not calendar days — cadence is
    roughly monthly and we want the tree model to see "is this pixel trending
    down over the 2019–2020 window". Pixels with < 3 valid scenes return 0.
    """
    T = stk_ma.shape[0]
    valid = np.isfinite(stk_ma)
    n = valid.sum(axis=0).astype(np.float32)
    t = np.arange(T, dtype=np.float32).reshape(T, 1, 1)

    with _quiet_nan_reductions():
        t_mean = np.nanmean(np.where(valid, t, np.nan), axis=0)
        y_mean = np.nanmean(stk_ma, axis=0)

    y_filled = np.where(valid, stk_ma, 0.0)
    t_dev = np.where(valid, t - t_mean[None], 0.0)
    y_dev = np.where(valid, y_filled - y_mean[None], 0.0)
    num = (t_dev * y_dev).sum(axis=0)
    den = (t_dev * t_dev).sum(axis=0)
    slope = np.where(den > 1e-6, num / den, 0.0)
    slope = np.where(n >= 3, slope, 0.0)
    return np.nan_to_num(slope, nan=0.0).astype(np.float32)


def _neighborhood_mean(a: np.ndarray, size: int) -> np.ndarray:
    """Uniform-kernel mean over (H, W), treating exact zeros as nodata and
    reweighting by the local valid-pixel count. Exact-zero is the composite
    convention for missing coverage, so not filtering lets tile edges pull
    the mean toward zero and destroys the "am I surrounded by forest" signal.
    """
    valid = np.isfinite(a) & (a != 0)
    a_f = np.where(valid, a, 0.0).astype(np.float32)
    num = ndi_uniform(a_f, size, mode="reflect")
    den = ndi_uniform(valid.astype(np.float32), size, mode="reflect")
    return np.where(den > 0.0, num / den, 0.0).astype(np.float32)


def lee_filter(img: np.ndarray, size: int = 5) -> np.ndarray:
    """Classic Lee speckle filter for SAR intensity.

    w = var_local / (var_local + var_noise) ; out = mean_local + w * (img - mean_local)
    Edges are preserved because `w` approaches 1 where local variance is high.
    Expects a 2-D float array that may contain NaNs (masked as uniform fill).
    """
    nan_mask = np.isnan(img)
    filled = np.where(nan_mask, 0.0, img)
    mean_k = ndi_uniform(filled, size)
    sq_k   = ndi_uniform(filled * filled, size)
    var_k  = np.maximum(sq_k - mean_k * mean_k, 0.0)
    # Noise variance = median of local variances — robust against the high-var
    # tail contributed by genuine edges, which we want to preserve.
    noise_var = float(np.median(var_k[~nan_mask])) if (~nan_mask).any() else 0.0
    w = var_k / (var_k + noise_var + 1e-12)
    out = mean_k + w * (filled - mean_k)
    return np.where(nan_mask, np.nan, out).astype(np.float32)


def s1_temporal_composite(paths: Iterable[Path], ref: ReferenceGrid,
                          apply_lee_filter: bool = False,
                          lee_window: int = 5) -> dict[str, np.ndarray]:
    """Median + std of Sentinel-1 VV (dB) across scenes.

    Args:
        apply_lee_filter: if True, each scene is Lee-filtered in dB space
            before temporal reduction. Median compositing already kills most
            speckle, so this mostly helps preserve edges at fresh clearings.
    """
    stacks = []
    for p in paths:
        arr = _reproject_to(ref, p, bands=[1], resampling=Resampling.bilinear,
                            dtype="float32")[0]
        db = np.where(arr > 0, 10.0 * np.log10(arr + 1e-6), np.nan)
        if apply_lee_filter:
            db = lee_filter(db, size=lee_window)
        stacks.append(db)
    if not stacks:
        z = np.zeros((ref.height, ref.width), dtype=np.float32)
        return {"vv_median": z, "vv_std": z.copy()}
    stk = np.stack(stacks, axis=0)
    with _quiet_nan_reductions():
        med = np.nanmedian(stk, axis=0)
        std = np.nanstd(stk, axis=0)
    return {"vv_median": np.nan_to_num(med, nan=0.0).astype(np.float32),
            "vv_std":    np.nan_to_num(std, nan=0.0).astype(np.float32)}


def s2_monthly_stack(
    paths: dict[tuple[int, int], Path],
    ref: ReferenceGrid,
    year: int,
    months: Iterable[int] = range(1, 13),
) -> np.ndarray:
    """Per-month S2 stack for a single year, reprojected onto `ref`.

    Output shape: (len(months) * 12, H, W). Missing months are zero-filled so
    downstream models see a fixed (12, 12, H, W) tensor regardless of coverage.
    The temporal attention head in DANN_Temporal_UNet can learn to ignore all-
    zero months via the ctx mean it pools over.
    """
    months = list(months)
    H, W = ref.height, ref.width
    out = np.zeros((len(months) * S2_N_BANDS, H, W), dtype=np.float32)
    for i, mo in enumerate(months):
        p = paths.get((year, mo))
        if p is None:
            continue
        s2 = _load_s2_normalised(p, ref)  # (12, H, W), already nodata=0
        out[i * S2_N_BANDS:(i + 1) * S2_N_BANDS] = s2.astype(np.float32)
    return out


def s1_monthly_stack(
    paths: dict[tuple[int, int, str], Path],
    ref: ReferenceGrid,
    year: int,
    direction: str,
    months: Iterable[int] = range(1, 13),
    apply_lee_filter: bool = False,
    lee_window: int = 5,
) -> np.ndarray:
    """Per-month S1 VV stack (dB) for one orbit direction.

    Output shape: (len(months), H, W). Missing months are zero-filled — note
    0 dB is a physically plausible VV value, but with the S1 dynamic range
    sitting in [-25, +5] dB for terrestrial targets, a zero-fill will look
    like a bright anomaly. Prefer a small constant well below the typical
    forest VV (~-7 dB) so the monthly attention head can learn 'zero = missing'.
    We fill with 0.0 here and document it; change later if the model starts
    hallucinating edges on zero-filled months.
    """
    months = list(months)
    H, W = ref.height, ref.width
    out = np.zeros((len(months), H, W), dtype=np.float32)
    for i, mo in enumerate(months):
        p = paths.get((year, mo, direction))
        if p is None:
            continue
        arr = _reproject_to(ref, p, bands=[1], resampling=Resampling.bilinear,
                            dtype="float32")[0]
        db = np.where(arr > 0, 10.0 * np.log10(arr + 1e-6), np.nan)
        if apply_lee_filter:
            db = lee_filter(db, size=lee_window)
        out[i] = np.nan_to_num(db, nan=0.0).astype(np.float32)
    return out


def aef_composite(paths_by_year: dict[int, Path], years: Iterable[int],
                  ref: ReferenceGrid) -> np.ndarray:
    """Median across requested AEF years. Output shape: (AEF_N_DIMS, H, W)."""
    stacks = []
    for y in years:
        p = paths_by_year.get(y)
        if p is None:
            continue
        stacks.append(_reproject_to(ref, p, bands=list(range(1, AEF_N_DIMS + 1)),
                                    resampling=Resampling.bilinear, dtype="float32"))
    if not stacks:
        return np.zeros((AEF_N_DIMS, ref.height, ref.width), dtype=np.float32)
    stk = np.stack(stacks, axis=0)
    stk_ma = np.where(np.isfinite(stk), stk, np.nan)
    with _quiet_nan_reductions():
        out = np.nanmedian(stk_ma, axis=0).astype(np.float32)
    return np.nan_to_num(out, nan=0.0)


def _days_to_date(epoch: date, days: np.ndarray) -> np.ndarray:
    """Vectorised: days-since-epoch → ordinal day-count for cheap comparison
    against a cutoff. Avoids building Python ``date`` objects per pixel."""
    return epoch.toordinal() + days


def decode_radd(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode RADD encoding → (alert_mask_bool, alert_ordinal_int).
    arr values: 0 = no alert; 2XXXX = low-conf; 3XXXX = high-conf, XXXX = days since 2014-12-31.
    """
    alert_mask = arr > 0
    leading = arr // 10_000
    days = arr % 10_000
    confidence = np.where(leading == 3, 1.0, np.where(leading == 2, 0.5, 0.0)).astype(np.float32)
    ordinal = np.where(alert_mask, _days_to_date(RADD_EPOCH, days), 0)
    return confidence, ordinal.astype(np.int64)


def decode_glads2(alert: np.ndarray, date_raster: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """GLAD-S2: alert values 0–4 (0=none, 4=high conf); date = days since 2019-01-01."""
    confidence = np.clip(alert.astype(np.float32) / 4.0, 0.0, 1.0)
    ordinal = np.where(alert > 0, _days_to_date(GLADS2_EPOCH, date_raster), 0)
    return confidence, ordinal.astype(np.int64)


def decode_gladl(alert_yy: np.ndarray, date_yy: np.ndarray, year: int) -> tuple[np.ndarray, np.ndarray]:
    """GLAD-L per-year: alert ∈ {0, 2=probable, 3=confirmed}, date = DOY of 20YY."""
    confidence = np.where(alert_yy == 3, 1.0, np.where(alert_yy == 2, 0.5, 0.0)).astype(np.float32)
    ordinal = np.where(alert_yy > 0,
                       date(2000 + year, 1, 1).toordinal() + np.maximum(date_yy - 1, 0),
                       0).astype(np.int64)
    return confidence, ordinal


def fuse_labels(ti: TileInventory, ref: ReferenceGrid,
                cutoff: date = date(2021, 1, 1)) -> tuple[np.ndarray, np.ndarray]:
    """Build a consensus deforestation label + per-pixel confidence.

    A pixel is positive iff ≥2 sources flagged a post-`cutoff` alert.
    Confidence is the mean of contributing sources' confidences.

    Returns:
        label (H, W) uint8  — 0/1 consensus mask
        conf  (H, W) float32 ∈ [0,1]
    """
    cutoff_ord = cutoff.toordinal()
    votes = np.zeros((ref.height, ref.width), dtype=np.uint8)
    conf_sum = np.zeros((ref.height, ref.width), dtype=np.float32)
    contrib = np.zeros((ref.height, ref.width), dtype=np.uint8)

    if ti.radd_path is not None:
        raw = _reproject_to(ref, ti.radd_path, bands=[1],
                            resampling=Resampling.nearest, dtype="int32")[0]
        c, ord_ = decode_radd(raw)
        hit = (c > 0) & (ord_ >= cutoff_ord)
        votes  += hit.astype(np.uint8)
        conf_sum += np.where(hit, c, 0.0)
        contrib  += hit.astype(np.uint8)

    if ti.glads2_alert_path is not None and ti.glads2_date_path is not None:
        alert = _reproject_to(ref, ti.glads2_alert_path, bands=[1],
                              resampling=Resampling.nearest, dtype="int32")[0]
        dt = _reproject_to(ref, ti.glads2_date_path, bands=[1],
                           resampling=Resampling.nearest, dtype="int32")[0]
        c, ord_ = decode_glads2(alert, dt)
        hit = (c > 0) & (ord_ >= cutoff_ord)
        votes  += hit.astype(np.uint8)
        conf_sum += np.where(hit, c, 0.0)
        contrib  += hit.astype(np.uint8)

    for yy in sorted(ti.gladl_alert_paths.keys()):
        a_path, d_path = ti.gladl_alert_paths[yy], ti.gladl_date_paths.get(yy)
        if d_path is None:
            continue
        alert = _reproject_to(ref, a_path, bands=[1],
                              resampling=Resampling.nearest, dtype="int32")[0]
        dt = _reproject_to(ref, d_path, bands=[1],
                           resampling=Resampling.nearest, dtype="int32")[0]
        c, ord_ = decode_gladl(alert, dt, yy)
        hit = (c > 0) & (ord_ >= cutoff_ord)
        votes  += hit.astype(np.uint8)
        conf_sum += np.where(hit, c, 0.0)
        contrib  += hit.astype(np.uint8)

    consensus = (votes >= 2).astype(np.uint8)
    mean_conf = np.where(contrib > 0, conf_sum / np.maximum(contrib, 1), 0.0).astype(np.float32)
    # Single-source pixels are ambiguous — keep a trickle of weight (0.3×) so
    # the loss can still learn from them but treats them as low-confidence.
    mean_conf = np.where(consensus == 1, mean_conf,
                         np.where(votes == 1, 0.3 * mean_conf, 0.0)).astype(np.float32)
    return consensus, mean_conf


def build_forest_ground_truth(
    ti: TileInventory,
    ref: ReferenceGrid,
    cutoff: date = date(2021, 1, 1),
    ndvi_pre_median: np.ndarray | None = None,
    spectral_ndvi_min: float = 0.4,
    strict: bool = True,
    min_confidence: float = 0.75,
) -> np.ndarray:
    """Free pre-2020 forest ground-truth from weak labels.

    Any pixel flagged by a source as a post-`cutoff` deforestation alert must
    have been forest before the alert — use this as evaluation-only positives
    for the pre-2020 forest mask.

    Unlike `fuse_labels` (which uses ≥2-source consensus for training `y`),
    this uses the UNION because a single detector flagging loss is already
    sufficient evidence that forest existed.

    Phase 1 filters
    ---------------
    1. **Strict confidence** (`strict=True`, default). Accept only pixels where
       the source's confidence ≥ `min_confidence` (0.75 catches RADD high-conf,
       GLAD-L confirmed, and GLAD-S2 codes ≥ 3). Low-confidence alerts are
       themselves often FPs; including them as "free GT" for the masker turns
       the recall metric into a distorted evaluation of label noise. Set
       `strict=False` to restore the old permissive union.
    2. **Spectral sanity gate**. If `ndvi_pre_median` is provided, AND the
       union with `ndvi_pre_median >= spectral_ndvi_min`. A pixel that never
       looked spectrally vegetated in 2019–2020 cannot have been forest —
       this drops RADD-on-cropland and cloud-shadow FPs without touching
       legitimate degraded-forest pixels (0.4 is permissive by design).

    Returns:
        (H, W) uint8 — 1 where any qualifying source flagged post-cutoff loss.
    """
    cutoff_ord = cutoff.toordinal()
    thresh = min_confidence if strict else 0.0  # 0.0 lets `c > 0` through
    gt = np.zeros((ref.height, ref.width), dtype=np.uint8)

    def _include(c: np.ndarray, ord_: np.ndarray) -> np.ndarray:
        pass_conf = (c >= thresh) if strict else (c > 0.0)
        return pass_conf & (ord_ >= cutoff_ord)

    if ti.radd_path is not None:
        raw = _reproject_to(ref, ti.radd_path, bands=[1],
                            resampling=Resampling.nearest, dtype="int32")[0]
        c, ord_ = decode_radd(raw)
        gt |= _include(c, ord_).astype(np.uint8)

    if ti.glads2_alert_path is not None and ti.glads2_date_path is not None:
        alert = _reproject_to(ref, ti.glads2_alert_path, bands=[1],
                              resampling=Resampling.nearest, dtype="int32")[0]
        dt = _reproject_to(ref, ti.glads2_date_path, bands=[1],
                           resampling=Resampling.nearest, dtype="int32")[0]
        c, ord_ = decode_glads2(alert, dt)
        gt |= _include(c, ord_).astype(np.uint8)

    for yy in sorted(ti.gladl_alert_paths.keys()):
        d_path = ti.gladl_date_paths.get(yy)
        if d_path is None:
            continue
        alert = _reproject_to(ref, ti.gladl_alert_paths[yy], bands=[1],
                              resampling=Resampling.nearest, dtype="int32")[0]
        dt = _reproject_to(ref, d_path, bands=[1],
                           resampling=Resampling.nearest, dtype="int32")[0]
        c, ord_ = decode_gladl(alert, dt, yy)
        gt |= _include(c, ord_).astype(np.uint8)

    if ndvi_pre_median is not None:
        gt &= (ndvi_pre_median >= spectral_ndvi_min).astype(np.uint8)

    return gt


def forest_mask_2020(ndvi_2020_median: np.ndarray, threshold: float = 0.6) -> np.ndarray:
    """Simple NDVI threshold on a 2020 annual composite → (H, W) uint8."""
    return (ndvi_2020_median >= threshold).astype(np.uint8)


def _select_paths_by_year(paths: dict, years: Iterable[int], key_year_pos: int = 0) -> list[Path]:
    """Filter a {(year, ...): path} dict by year membership."""
    years = set(years)
    out = []
    for k, v in paths.items():
        y = k if isinstance(k, int) else k[key_year_pos]
        if y in years:
            out.append(v)
    return out


def _select_s1_by_dir(paths: dict[tuple[int, int, str], Path],
                      years: Iterable[int], direction: str) -> list[Path]:
    """S1 paths are keyed by (year, month, direction). Previously we flattened
    everything into one stack and took the median — silently averaging asc and
    desc. The two see canopy geometry differently, so we now split them."""
    years = set(years)
    return [v for (y, _m, d), v in paths.items() if y in years and d == direction]


def _scene_pixel_count(p: Path) -> int:
    with rasterio.open(p) as src:
        return int(src.width) * int(src.height)


def _filter_partial_scenes(paths: dict, ref_size: int, coverage: float = 0.8) -> dict:
    """Drop scenes whose pixel count is < `coverage` of the reference size.

    Partial scenes reproject into strips / half-tile patches and leave the rest
    as zeros — those zeros then dominate medians and confuse downstream models
    (seam artefacts, half-tile NON classifications).
    """
    return {k: p for k, p in paths.items()
            if _scene_pixel_count(p) >= coverage * ref_size}


def preprocess_tile(
    ti: TileInventory,
    pre_years: tuple[int, ...] = DEFAULT_PRE_YEARS,
    post_years: tuple[int, ...] = DEFAULT_POST_YEARS,
    forest_ndvi_threshold: float = 0.6,
    apply_lee_filter: bool = False,
    lee_window: int = 5,
    min_reference_pixels: int = 500_000,
    partial_scene_coverage: float = 0.8,
    gt_strict: bool = True,
    gt_min_confidence: float = 0.75,
    gt_spectral_ndvi_min: float = 0.4,
) -> dict[str, np.ndarray]:
    """Run the full preprocessing pipeline for a single tile.

    Output tensors (all float32, shape (C, H, W) unless noted):
        aef_pre, aef_post                  — (64, H, W)
        s2_pre_band_median, s2_post_band_median    — (12, H, W)
        s2_pre_band_std,    s2_post_band_std       — (12, H, W)
        s2_{pre,post}_{ndvi,nbr,ndmi,evi}_median/std — (H, W)
        s2_pre_{ndvi,nbr,ndmi}_{p10,p90,slope}        — (H, W) pre trajectory
        s2_pre_n_scenes_valid                         — (H, W) valid scene count
        s2_pre_{ndvi,nbr}_{nh9,nh21}                  — (H, W) neighborhood mean
        s2_delta_{ndvi,nbr,ndmi,evi}                  — (H, W) post - pre medians
        s1_{pre,post}_vv_median/std                   — (H, W) combined asc+desc
        s1_{pre,post}_vv_{asc,desc}_median/std        — (H, W) per-orbit
        s1_delta_vv, s1_delta_vv_{asc,desc}           — (H, W) post - pre medians
        s1_{pre,post}_vv_orbit_diff                   — (H, W) asc - desc median
        s2_monthly_pre                                — (144, H, W) 12 mo × 12 bands
        s1_monthly_pre_asc                            — (12, H, W)  12 mo × VV asc
        s1_monthly_pre_desc                           — (12, H, W)  12 mo × VV desc
        forest_mask_2020                              — (H, W) uint8
        label, label_confidence                       — (H, W) uint8/float32 [train only]
    """
    if ti.tile_id in BROKEN_TILES:
        raise RuntimeError(
            f"Tile {ti.tile_id}: listed in BROKEN_TILES — irrecoverable upstream data.")
    if not ti.s2_paths:
        raise RuntimeError(f"Tile {ti.tile_id}: no Sentinel-2 scenes available.")

    # Pick the largest S2 scene as the reference grid. The provider ships
    # inconsistent scene sizes per month — first-scene-wins is catastrophic
    # when the first happens to be a 2×1004 sliver or a 4×6 thumbnail.
    ref_source = max(ti.s2_paths.values(), key=_scene_pixel_count)
    ref_size = _scene_pixel_count(ref_source)
    if ref_size < min_reference_pixels:
        raise RuntimeError(
            f"Tile {ti.tile_id}: largest S2 scene is only {ref_size} px — "
            f"tile is fundamentally broken, skip it.")
    ref = ReferenceGrid.from_s2(ref_source)

    # Drop partial-coverage S2 scenes before building composites (prevents the
    # right-half-zero seam artefact seen on tiles like 19NBD_4_4). We do NOT
    # filter S1 with the same threshold: S1 ships on its own native grid with
    # different pixel dimensions than S2, so comparing S1 scene size against
    # the S2 reference size would drop every S1 scene (verified on 47QMB_0_8,
    # Apr 2026 — caused an all-zero S1 composite regression).
    ti = replace(
        ti,
        s2_paths=_filter_partial_scenes(ti.s2_paths, ref_size, partial_scene_coverage),
    )

    out: dict[str, np.ndarray] = {}

    s2_pre = s2_temporal_composite(
        _select_paths_by_year(ti.s2_paths, pre_years, key_year_pos=0), ref,
        compute_trajectory=True)
    s2_post = s2_temporal_composite(
        _select_paths_by_year(ti.s2_paths, post_years, key_year_pos=0), ref)
    for k, v in s2_pre.items():
        out[f"s2_pre_{k}"] = v
    for k, v in s2_post.items():
        out[f"s2_post_{k}"] = v
    for idx in ("ndvi", "nbr", "ndmi", "evi"):
        out[f"s2_delta_{idx}"] = (out[f"s2_post_{idx}_median"]
                                  - out[f"s2_pre_{idx}_median"]).astype(np.float32)

    # Spatial context: a pixel's neighborhood vegetation. Helps the masker
    # suppress isolated bright pixels (roads, cropland edges, building tops)
    # that look forest-like in isolation but sit in non-forest surroundings.
    for idx in ("ndvi", "nbr"):
        base = out[f"s2_pre_{idx}_median"]
        out[f"s2_pre_{idx}_nh9"]  = _neighborhood_mean(base, 9)
        out[f"s2_pre_{idx}_nh21"] = _neighborhood_mean(base, 21)

    # S1: split ascending / descending — canopy geometry looks different under
    # each orbit pass, and their difference is a forest-structure signal. The
    # combined (asc+desc) median is also kept as a backward-compat feature.
    s1_pre = s1_temporal_composite(
        _select_paths_by_year(ti.s1_paths, pre_years, key_year_pos=0), ref,
        apply_lee_filter=apply_lee_filter, lee_window=lee_window)
    s1_post = s1_temporal_composite(
        _select_paths_by_year(ti.s1_paths, post_years, key_year_pos=0), ref,
        apply_lee_filter=apply_lee_filter, lee_window=lee_window)
    for k, v in s1_pre.items():
        out[f"s1_pre_{k}"] = v
    for k, v in s1_post.items():
        out[f"s1_post_{k}"] = v
    out["s1_delta_vv"] = (out["s1_post_vv_median"] - out["s1_pre_vv_median"]).astype(np.float32)

    for direction, short in (("ascending", "asc"), ("descending", "desc")):
        s1_pre_d = s1_temporal_composite(
            _select_s1_by_dir(ti.s1_paths, pre_years, direction), ref,
            apply_lee_filter=apply_lee_filter, lee_window=lee_window)
        s1_post_d = s1_temporal_composite(
            _select_s1_by_dir(ti.s1_paths, post_years, direction), ref,
            apply_lee_filter=apply_lee_filter, lee_window=lee_window)
        out[f"s1_pre_vv_{short}_median"]  = s1_pre_d["vv_median"]
        out[f"s1_pre_vv_{short}_std"]     = s1_pre_d["vv_std"]
        out[f"s1_post_vv_{short}_median"] = s1_post_d["vv_median"]
        out[f"s1_post_vv_{short}_std"]    = s1_post_d["vv_std"]
        out[f"s1_delta_vv_{short}"] = (
            s1_post_d["vv_median"] - s1_pre_d["vv_median"]).astype(np.float32)

    # orbit_diff = asc - desc; zero where one orbit is absent (composite returns
    # zeros), which LightGBM can disambiguate from the companion _median == 0.
    out["s1_pre_vv_orbit_diff"] = (
        out["s1_pre_vv_asc_median"] - out["s1_pre_vv_desc_median"]).astype(np.float32)
    out["s1_post_vv_orbit_diff"] = (
        out["s1_post_vv_asc_median"] - out["s1_post_vv_desc_median"]).astype(np.float32)

    # Monthly stacks for the temporal U-Net. The per-month layout preserves
    # phenology that composite reductions (median/std/p10) erase — a
    # TemporalChannelAttention head can weight informative months and ignore
    # zero-filled ones. Anchored on the first entry of pre_years (2020 by
    # default) because that's the year every tile in this dataset has S2+S1.
    pre_anchor_year = int(next(iter(pre_years)))
    out["s2_monthly_pre"]     = s2_monthly_stack(ti.s2_paths, ref, pre_anchor_year)
    out["s1_monthly_pre_asc"] = s1_monthly_stack(
        ti.s1_paths, ref, pre_anchor_year, "ascending",
        apply_lee_filter=apply_lee_filter, lee_window=lee_window)
    out["s1_monthly_pre_desc"] = s1_monthly_stack(
        ti.s1_paths, ref, pre_anchor_year, "descending",
        apply_lee_filter=apply_lee_filter, lee_window=lee_window)

    out["aef_pre"] = aef_composite(ti.aef_paths, pre_years, ref)
    out["aef_post"] = aef_composite(ti.aef_paths, post_years, ref)
    out["aef_delta"] = (out["aef_post"] - out["aef_pre"]).astype(np.float32)

    ndvi_pre = out["s2_pre_ndvi_median"]
    out["forest_mask_2020"] = forest_mask_2020(ndvi_pre, forest_ndvi_threshold)

    if ti.split == "train" and ti.has_any_labels():
        label, conf = fuse_labels(ti, ref)
        # Deforestation is only defined where forest existed — gate labels by
        # the 2020 forest mask so we don't score positives on pixels that were
        # never forest in the first place.
        label = (label & out["forest_mask_2020"]).astype(np.uint8)
        conf  = (conf * out["forest_mask_2020"]).astype(np.float32)
        out["label"] = label
        out["label_confidence"] = conf

        # EVALUATION ONLY — never feed to training.
        out["forest_gt_pre2020"] = build_forest_ground_truth(
            ti, ref,
            ndvi_pre_median=out["s2_pre_ndvi_median"],
            spectral_ndvi_min=gt_spectral_ndvi_min,
            strict=gt_strict,
            min_confidence=gt_min_confidence,
        )

    out["_shape"] = np.array([ref.height, ref.width], dtype=np.int32)
    return out


def cache_tile(ti: TileInventory, cache_dir: Path, **kwargs) -> Path:
    """Run `preprocess_tile` and dump to `.npz`. Returns the cache file path."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{ti.tile_id}.npz"
    if cache_path.exists():
        return cache_path
    tensors = preprocess_tile(ti, **kwargs)
    np.savez_compressed(cache_path, **tensors)
    return cache_path


# Channel counts annotated per group so the total (274) stays easy to audit
# when tuning the Tier-2 U-Net input stem.
DEFAULT_FEATURE_KEYS: tuple[str, ...] = (
    "aef_pre", "aef_post", "aef_delta",                               # 3*64 = 192
    "s2_pre_band_median", "s2_post_band_median",                      # 2*12 =  24
    "s2_pre_band_std",    "s2_post_band_std",                         # 2*12 =  24
    "s2_delta_ndvi", "s2_delta_nbr", "s2_delta_ndmi", "s2_delta_evi", #         4
    "s2_pre_ndvi_median", "s2_pre_nbr_median",
    "s2_post_ndvi_median", "s2_post_nbr_median",                      #         4
    # Pre-period trajectory stats — p10 catches transient stress that a median
    # hides; slope catches gradual degradation. NDMI adds moisture context.
    "s2_pre_ndvi_p10", "s2_pre_ndvi_p90", "s2_pre_ndvi_slope",
    "s2_pre_nbr_p10",  "s2_pre_nbr_p90",  "s2_pre_nbr_slope",
    "s2_pre_ndmi_p10", "s2_pre_ndmi_p90", "s2_pre_ndmi_slope",        #         9
    "s2_pre_n_scenes_valid",                                          #         1
    # Neighborhood means — spatial context for isolated-pixel suppression.
    "s2_pre_ndvi_nh9", "s2_pre_ndvi_nh21",
    "s2_pre_nbr_nh9",  "s2_pre_nbr_nh21",                             #         4
    "s1_pre_vv_median", "s1_post_vv_median", "s1_delta_vv",           #         3
    "s1_pre_vv_asc_median",  "s1_post_vv_asc_median",  "s1_delta_vv_asc",   #   3
    "s1_pre_vv_desc_median", "s1_post_vv_desc_median", "s1_delta_vv_desc",  #   3
    "s1_pre_vv_orbit_diff", "s1_post_vv_orbit_diff",                  #         2
    "forest_mask_2020",                                               #         1
)


# Feature layout for DANN_Temporal_UNet. The model expects the per-month
# channels up front (so its TemporalChannelAttention can reshape to a
# (B, T=12, C, H, W) tensor), followed by AEF delta at the tail for
# bottleneck fusion. Total = 144 + 12 + 12 + 64 = 232 channels.
TEMPORAL_FEATURE_KEYS: tuple[str, ...] = (
    "s2_monthly_pre",        # 12 months × 12 bands       = 144
    "s1_monthly_pre_asc",    # 12 months × VV (ascending) =  12
    "s1_monthly_pre_desc",   # 12 months × VV (descending)=  12
    "aef_delta",             # 64 AEF embedding dims      =  64
)


def stack_features(tensors: dict[str, np.ndarray],
                   feature_keys: tuple[str, ...] = DEFAULT_FEATURE_KEYS) -> np.ndarray:
    """Concatenate selected (C, H, W) / (H, W) tensors along channel dim → (C_total, H, W)."""
    chans = []
    for k in feature_keys:
        a = tensors[k]
        if a.ndim == 2:
            a = a[None]
        chans.append(a.astype(np.float32))
    return np.concatenate(chans, axis=0)

_MGRS_PREFIX_RE = re.compile(r"^(\d{2})[A-Z]{3}$")


def tile_id_to_region_label(tile_id: str) -> int:
    """Map MGRS tile id to a stable 0-based region index (UTM zone-1)."""
    prefix = tile_id.split("_")[0]
    m = _MGRS_PREFIX_RE.match(prefix)
    if not m:
        return 0
    zone = int(m.group(1))
    if not 1 <= zone <= 60:
        return 0
    return zone - 1


class DeforestationPatchDataset(Dataset):
    """Random-crop patches from cached tile .npz files.

    Each sample returns:
        x     : (C, patch, patch) float32
        y     : (patch, patch) uint8
        w     : (patch, patch) float32  — label confidence weight
        mask  : (patch, patch) uint8    — 2020 forest mask (loss gating)

    For `is_train=False`, returns the full tile and ignores patch_size.
    """

    def __init__(
        self,
        cache_paths: list[Path],
        feature_keys: tuple[str, ...] = DEFAULT_FEATURE_KEYS,
        patch_size: int = 256,
        patches_per_tile: int = 8,
        positive_ratio: float = 0.5,
        is_train: bool = True,
        seed: int | None = None,
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required for DeforestationPatchDataset")
        self.cache_paths = list(cache_paths)
        self.feature_keys = feature_keys
        self.patch_size = patch_size
        self.patches_per_tile = patches_per_tile
        self.positive_ratio = positive_ratio
        self.is_train = is_train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        if not self.is_train:
            return len(self.cache_paths)
        return len(self.cache_paths) * self.patches_per_tile

    def _load(self, idx: int) -> dict[str, np.ndarray]:
        tile_idx = idx // self.patches_per_tile if self.is_train else idx
        with np.load(self.cache_paths[tile_idx], allow_pickle=False) as npz:
            return {k: npz[k] for k in npz.files}

    def _sample_crop(self, label: np.ndarray, forest: np.ndarray) -> tuple[int, int]:
        H, W = label.shape
        ps = self.patch_size
        if ps > H or ps > W:
            return 0, 0
        # Positives are rare — bias sampling toward label-positive centres so
        # batches carry enough gradient signal. Without this the loss is
        # dominated by easy-negative forest patches.
        if self.rng.random() < self.positive_ratio:
            ys, xs = np.where(label > 0)
            if len(ys):
                i = int(self.rng.integers(len(ys)))
                cy, cx = int(ys[i]), int(xs[i])
                y0 = np.clip(cy - ps // 2, 0, H - ps)
                x0 = np.clip(cx - ps // 2, 0, W - ps)
                return int(y0), int(x0)
        ys, xs = np.where(forest > 0)
        if len(ys) == 0:
            y0 = int(self.rng.integers(0, H - ps + 1))
            x0 = int(self.rng.integers(0, W - ps + 1))
        else:
            i = int(self.rng.integers(len(ys)))
            cy, cx = int(ys[i]), int(xs[i])
            y0 = int(np.clip(cy - ps // 2, 0, H - ps))
            x0 = int(np.clip(cx - ps // 2, 0, W - ps))
        return y0, x0

    def _augment(self, arrays: list[np.ndarray]) -> list[np.ndarray]:
        """Apply the same flip/rot to all arrays in the list.
        Channel-first arrays (ndim == 3) are rotated along axes (1, 2);
        2-D arrays are rotated along axes (0, 1)."""
        flip_v = self.rng.random() < 0.5
        flip_h = self.rng.random() < 0.5
        k_rot = int(self.rng.integers(0, 4))
        out = []
        for a in arrays:
            if a.ndim == 3:
                if flip_v: a = a[:, ::-1, :]
                if flip_h: a = a[:, :, ::-1]
                if k_rot:  a = np.rot90(a, k=k_rot, axes=(1, 2))
            else:
                if flip_v: a = a[::-1, :]
                if flip_h: a = a[:, ::-1]
                if k_rot:  a = np.rot90(a, k=k_rot)
            out.append(a.copy())
        return out

    def __getitem__(self, idx: int):
        tile_idx = idx // self.patches_per_tile if self.is_train else idx
        tile_id = self.cache_paths[tile_idx].stem
        region_label = tile_id_to_region_label(tile_id)

        t = self._load(idx)
        x = stack_features(t, self.feature_keys)
        mask = t.get("forest_mask_2020", np.ones(x.shape[-2:], dtype=np.uint8))
        label = t.get("label", np.zeros(x.shape[-2:], dtype=np.uint8))
        conf  = t.get("label_confidence", np.zeros(x.shape[-2:], dtype=np.float32))
        forest_gt = t.get("forest_gt_pre2020", np.zeros(x.shape[-2:], dtype=np.uint8))

        if self.is_train:
            y0, x0 = self._sample_crop(label, mask)
            ps = self.patch_size
            x         = x[:, y0:y0 + ps, x0:x0 + ps]
            label     = label[y0:y0 + ps, x0:x0 + ps]
            conf      = conf[y0:y0 + ps, x0:x0 + ps]
            mask      = mask[y0:y0 + ps, x0:x0 + ps]
            forest_gt = forest_gt[y0:y0 + ps, x0:x0 + ps]
            x, label, conf, mask, forest_gt = self._augment([x, label, conf, mask, forest_gt])

        return {
            "x":         torch.from_numpy(np.ascontiguousarray(x)).float(),
            "y":         torch.from_numpy(np.ascontiguousarray(label)).long(),
            "w":         torch.from_numpy(np.ascontiguousarray(conf)).float(),
            "mask":      torch.from_numpy(np.ascontiguousarray(mask)).float(),
            "forest_gt": torch.from_numpy(np.ascontiguousarray(forest_gt)).long(),
            "tile":      self.cache_paths[tile_idx].stem,
            "region_label": torch.tensor(region_label, dtype=torch.long),
        }


def split_tiles(tile_ids: list[str], val_frac: float = 0.15, seed: int = 0
               ) -> tuple[list[str], list[str]]:
    """Split by MGRS zone prefix to reduce spatial leakage."""
    rng = np.random.default_rng(seed)
    by_zone: dict[str, list[str]] = {}
    for t in tile_ids:
        zone = t.split("_")[0]
        by_zone.setdefault(zone, []).append(t)

    train, val, singletons = [], [], []
    for zone, ts in by_zone.items():
        ts = sorted(ts)
        rng.shuffle(ts)
        if len(ts) == 1:
            # Singleton zones can't be stratified within-zone; pool them
            # and distribute globally after multi-tile zones are split.
            singletons.append(ts[0])
        else:
            k = max(1, int(round(len(ts) * val_frac)))
            val.extend(ts[:k])
            train.extend(ts[k:])

    rng.shuffle(singletons)
    target_val = max(1, int(round(len(tile_ids) * val_frac)))
    slots = max(0, target_val - len(val))
    val.extend(singletons[:slots])
    train.extend(singletons[slots:])

    # With ≥2 tiles, empty splits break DataLoader — swap one tile across if
    # the round-off above happened to empty either side.
    if len(tile_ids) >= 2:
        if not train and val:
            train.append(val.pop())
        if not val and train:
            val.append(train.pop())

    return sorted(train), sorted(val)


def build_dataloaders(
    root: Path | str,
    cache_dir: Path | str = "./cache",
    batch_size: int = 8,
    patch_size: int = 256,
    patches_per_tile: int = 8,
    num_workers: int = 4,
    val_frac: float = 0.15,
    seed: int = 0,
    preprocess_kwargs: dict | None = None,
) -> tuple["DataLoader", "DataLoader", dict[str, TileInventory]]:
    """End-to-end: discover tiles → cache preprocessed .npz → build train/val loaders."""
    if torch is None:
        raise RuntimeError("PyTorch is required for build_dataloaders")
    preprocess_kwargs = preprocess_kwargs or {}

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(
            f"Data root not found: {root_path.resolve()}\n"
            f"Expected layout: <root>/{{sentinel-1,sentinel-2,aef-embeddings,labels,metadata}}/...\n"
            f"If you're running from code/run.ipynb, try root='../data/makeathon-challenge'."
        )

    inv = build_inventory(root_path, "train")
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)

    if not inv:
        raise RuntimeError(
            f"No tiles discovered under {root_path.resolve()}. "
            f"Did you run `make download_data_from_s3`?"
        )

    cache_paths: dict[str, Path] = {}
    skipped_no_labels, skipped_no_s2 = [], []
    for tile_id, ti in inv.items():
        if not ti.s2_paths:
            skipped_no_s2.append(tile_id); continue
        if not ti.has_any_labels():
            skipped_no_labels.append(tile_id); continue
        try:
            cache_paths[tile_id] = cache_tile(ti, cache_dir, **preprocess_kwargs)
        except Exception as e:
            logger.warning(f"Failed to preprocess {tile_id}: {e}")

    if not cache_paths:
        msg = [
            f"No trainable tiles under {root_path.resolve()}.",
            f"  Total tiles discovered: {len(inv)}",
            f"  Skipped (no Sentinel-2): {len(skipped_no_s2)}",
            f"  Skipped (no labels):     {len(skipped_no_labels)}",
        ]
        if skipped_no_s2:
            msg.append("  → Sentinel-2 data is missing. Run `make download_data_from_s3`.")
        if skipped_no_labels:
            msg.append("  → Labels are missing. Run `make download_data_from_s3`.")
        raise RuntimeError("\n".join(msg))

    logger.info(f"Cached {len(cache_paths)} tiles "
                f"(skipped {len(skipped_no_s2)} no-S2, {len(skipped_no_labels)} no-labels).")

    train_ids, val_ids = split_tiles(list(cache_paths.keys()), val_frac=val_frac, seed=seed)
    if not train_ids or not val_ids:
        raise RuntimeError(
            f"Split produced empty train/val: train={len(train_ids)}, val={len(val_ids)}. "
            f"Need at least 2 tiles — got {len(cache_paths)}."
        )

    train_ds = DeforestationPatchDataset(
        [cache_paths[t] for t in train_ids],
        patch_size=patch_size, patches_per_tile=patches_per_tile, is_train=True, seed=seed,
    )
    val_ds = DeforestationPatchDataset(
        [cache_paths[t] for t in val_ids], is_train=False, seed=seed,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(1, num_workers // 2), pin_memory=True,
    )
    return train_loader, val_loader, inv


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data/makeathon-challenge")
    ap.add_argument("--cache_dir", default="./cache")
    ap.add_argument("--split", default="train")
    ap.add_argument("--tile", default=None, help="Preprocess a single tile_id and print tensor shapes.")
    args = ap.parse_args()

    inv = build_inventory(args.root, args.split)
    print(f"Discovered {len(inv)} tiles in split={args.split}:")
    for tid, ti in sorted(inv.items())[:5]:
        print(f"  {tid}: S2={len(ti.s2_paths)} mo | S1={len(ti.s1_paths)} mo | "
              f"AEF={len(ti.aef_paths)} yr | labels={ti.has_any_labels()}")

    if args.tile and args.tile in inv:
        print(f"\nPreprocessing {args.tile} …")
        t = preprocess_tile(inv[args.tile])
        for k, v in t.items():
            if isinstance(v, np.ndarray):
                print(f"  {k:32s} {str(v.shape):20s} {v.dtype}")
