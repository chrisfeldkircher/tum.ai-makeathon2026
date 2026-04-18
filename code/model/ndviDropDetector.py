"""Per-pixel timing of the sharpest NDVI drop after the pre/post cutoff.

Why this exists
---------------
Median composites (the `s2_delta_ndvi` channel) collapse the full monthly
trajectory into a single scalar — they tell you *how much* NDVI fell, not
*when*. Timing is real signal: a sudden drop in a single month looks different
to a gradual phenological decline, and the downstream deforestation model
cannot recover that distinction from a median delta alone.

Output (per pixel)
------------------
    ndvi_drop_magnitude : max over post-cutoff scenes of (baseline - NDVI_t).
                           Clipped at 0 — pixels that never dropped report 0.
    drop_doy            : day-of-year of the peak-drop month (1..366); 0 = no drop.
    drop_year           : year of the peak-drop month; 0 = no drop.
    drop_month_idx      : index into the post-cutoff scene list; -1 = no drop.

Baseline is the nan-median of NDVI over pre-cutoff scenes. If the tile has no
pre-cutoff scenes, we fall back to the post-cutoff median — the magnitude then
becomes relative rather than absolute, which is still useful for ranking.

Cloud robustness
----------------
Taking the single worst month per pixel is catastrophically cloud-biased —
any pixel that catches one cloudy / shadowed scene reports a huge spurious
drop. We instead smooth the post stack with a k-month rolling nan-median
(default k=3) before the argmax, so the "drop" we pick up has to persist
across ≥⌈k/2⌉ months to win. Clouds don't persist; deforestation does.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from data.data import ReferenceGrid, TileInventory

DEFAULT_PRE_CUTOFF = date(2021, 1, 1)


@dataclass
class NdviDropMaps:
    ndvi_drop_magnitude: np.ndarray   # (H, W) float32
    drop_doy: np.ndarray              # (H, W) int16   1..366, 0 if no drop
    drop_year: np.ndarray             # (H, W) int16   year,   0 if no drop
    drop_month_idx: np.ndarray        # (H, W) int16   index into post list, -1 if no drop


def compute_ndvi_drop(
    ti: "TileInventory",
    ref: "ReferenceGrid | None" = None,
    pre_cutoff: date = DEFAULT_PRE_CUTOFF,
    persistence_window: int = 3,
) -> NdviDropMaps:
    """Scan every S2 scene, compute NDVI, smooth post series with a
    `persistence_window`-month rolling nan-median, then per-pixel argmax of
    (baseline - smoothed_NDVI_t). Setting `persistence_window=1` disables
    smoothing (original single-month argmax).
    """
    from data.data import (
        ReferenceGrid,
        _load_s2_normalised,
        _quiet_nan_reductions,
        compute_indices,
    )

    if ref is None:
        ref_source = next(
            (p for (y, _m), p in sorted(ti.s2_paths.items()) if y == 2020),
            next(iter(sorted(ti.s2_paths.values())), None),
        )
        if ref_source is None:
            raise RuntimeError(f"{ti.tile_id}: no S2 scenes — cannot compute NDVI drop.")
        ref = ReferenceGrid.from_s2(ref_source)

    pre_stack: list[np.ndarray] = []
    post_stack: list[np.ndarray] = []
    post_dates: list[date] = []

    for (y, m), p in sorted(ti.s2_paths.items()):
        s2 = _load_s2_normalised(p, ref)
        ndvi = compute_indices(s2)["ndvi"]
        valid = s2.sum(axis=0) > 0
        ndvi = np.where(valid, ndvi, np.nan).astype(np.float32)
        if date(y, m, 1) < pre_cutoff:
            pre_stack.append(ndvi)
        else:
            post_stack.append(ndvi)
            post_dates.append(date(y, m, 15))

    H, W = ref.height, ref.width
    zeros_f = np.zeros((H, W), dtype=np.float32)
    zeros_i = np.zeros((H, W), dtype=np.int16)
    if not post_stack:
        return NdviDropMaps(zeros_f, zeros_i, zeros_i, np.full((H, W), -1, dtype=np.int16))

    with _quiet_nan_reductions():
        baseline = np.nanmedian(np.stack(pre_stack or post_stack, axis=0), axis=0)

    post = np.stack(post_stack, axis=0)           # (T, H, W)
    T = post.shape[0]
    k = max(1, int(persistence_window))
    if k > 1 and T > 1:
        # Centred rolling nan-median over k months (edges use shrunk windows).
        half = k // 2
        post_smooth = np.empty_like(post)
        with _quiet_nan_reductions():
            for t in range(T):
                lo, hi = max(0, t - half), min(T, t + half + 1)
                post_smooth[t] = np.nanmedian(post[lo:hi], axis=0)
    else:
        post_smooth = post
    drop = baseline[None, :, :] - post_smooth     # positive = NDVI fell
    drop_filled = np.where(np.isfinite(drop), drop, -np.inf)

    idx = np.argmax(drop_filled, axis=0).astype(np.int16)
    mag = np.take_along_axis(drop_filled, idx[None, :, :], axis=0)[0]

    no_drop = ~np.isfinite(mag) | (mag <= 0)
    mag = np.where(no_drop, 0.0, mag).astype(np.float32)

    doy = np.zeros((H, W), dtype=np.int16)
    year = np.zeros((H, W), dtype=np.int16)
    for t, d in enumerate(post_dates):
        sel = (idx == t) & ~no_drop
        doy[sel] = d.timetuple().tm_yday
        year[sel] = d.year

    month_idx = np.where(no_drop, -1, idx).astype(np.int16)
    return NdviDropMaps(mag, doy, year, month_idx)


def augment_cache_with_ndvi_drop(
    cache_path: Path | str,
    ti: "TileInventory",
    pre_cutoff: date = DEFAULT_PRE_CUTOFF,
    persistence_window: int = 3,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    """Add ndvi_drop_* keys to an existing `.npz` cache in-place.

    Returns the updated tensor dict. Skips work (and returns the loaded dict)
    if `ndvi_drop_magnitude` is already present and `overwrite=False`.
    """
    cache_path = Path(cache_path)
    with np.load(cache_path, allow_pickle=False) as npz:
        tensors = {key: npz[key] for key in npz.files}
    if not overwrite and "ndvi_drop_magnitude" in tensors:
        return tensors

    maps = compute_ndvi_drop(ti, ref=None, pre_cutoff=pre_cutoff,
                             persistence_window=persistence_window)
    tensors["ndvi_drop_magnitude"] = maps.ndvi_drop_magnitude
    tensors["drop_doy"]            = maps.drop_doy
    tensors["drop_year"]           = maps.drop_year
    tensors["drop_month_idx"]      = maps.drop_month_idx
    np.savez_compressed(cache_path, **tensors)
    return tensors
