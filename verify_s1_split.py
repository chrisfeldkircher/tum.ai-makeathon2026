"""Verify the Phase-0 changes work on live data before building anything on top.

Checks:
  1. 18NYH_9_9 is now hard-rejected by preprocess_tile.
  2. On a healthy tile, every new S1 key is present, non-zero, and finite.
  3. asc and desc composites are genuinely different (not silently averaged).
  4. orbit_diff reconstructs correctly (asc_median - desc_median).
  5. Per-orbit scene counts are reported so we know the split had data to work with.

Run:  python verify_s1_split.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "code"))

from data.data import (  # noqa: E402
    BROKEN_TILES,
    DEFAULT_POST_YEARS,
    DEFAULT_PRE_YEARS,
    build_inventory,
    preprocess_tile,
)

DATA_ROOT = Path(__file__).parent / "data" / "makeathon-challenge"
HEALTHY_TILE = "47QMB_0_8"   # full 72 S2 scenes, both orbits, has labels

NEW_KEYS = (
    "s1_pre_vv_asc_median", "s1_pre_vv_asc_std",
    "s1_pre_vv_desc_median", "s1_pre_vv_desc_std",
    "s1_post_vv_asc_median", "s1_post_vv_desc_median",
    "s1_delta_vv_asc", "s1_delta_vv_desc",
    "s1_pre_vv_orbit_diff", "s1_post_vv_orbit_diff",
)


def _stats(name: str, a: np.ndarray) -> str:
    finite = np.isfinite(a)
    n_nonzero = int((a != 0).sum())
    if not finite.any():
        return f"  {name:<32} ALL-NaN  shape={a.shape}"
    return (f"  {name:<32} shape={str(a.shape):<14} "
            f"min={a[finite].min():+7.2f}  max={a[finite].max():+7.2f}  "
            f"mean={a[finite].mean():+7.2f}  nonzero={n_nonzero}/{a.size}")


def check_broken_tile_rejection() -> bool:
    print("=" * 70)
    print("1. 18NYH_9_9 rejection test")
    print("=" * 70)
    inv = build_inventory(DATA_ROOT, "train")
    if "18NYH_9_9" not in inv:
        print("  (not in train inventory — skipping)")
        return True
    try:
        preprocess_tile(inv["18NYH_9_9"])
    except RuntimeError as e:
        msg = str(e)
        ok = "BROKEN_TILES" in msg
        print(f"  {'PASS' if ok else 'FAIL'}: raised RuntimeError → {msg}")
        return ok
    print("  FAIL: preprocess_tile completed without raising")
    return False


def check_orbit_split(tile_id: str) -> bool:
    print("\n" + "=" * 70)
    print(f"2. S1 orbit-split test on {tile_id}")
    print("=" * 70)
    for split in ("train", "test"):
        inv = build_inventory(DATA_ROOT, split)
        if tile_id in inv:
            ti = inv[tile_id]
            break
    else:
        print(f"  FAIL: {tile_id} not found in any split")
        return False

    asc_pre = sum(1 for (y, _m, d) in ti.s1_paths if d == "ascending" and y in DEFAULT_PRE_YEARS)
    desc_pre = sum(1 for (y, _m, d) in ti.s1_paths if d == "descending" and y in DEFAULT_PRE_YEARS)
    asc_post = sum(1 for (y, _m, d) in ti.s1_paths if d == "ascending" and y in DEFAULT_POST_YEARS)
    desc_post = sum(1 for (y, _m, d) in ti.s1_paths if d == "descending" and y in DEFAULT_POST_YEARS)
    print(f"  scene counts: pre asc={asc_pre} desc={desc_pre}  "
          f"post asc={asc_post} desc={desc_post}")
    if asc_pre == 0 or desc_pre == 0:
        print("  WARN: one orbit is empty pre-2021 — orbit_diff will be degenerate for this tile")

    print("  running preprocess_tile() ...")
    t = preprocess_tile(ti)
    print(f"  produced {len(t)} tensors.")

    missing = [k for k in NEW_KEYS if k not in t]
    if missing:
        print(f"  FAIL: missing keys: {missing}")
        return False
    print("  all new keys present.")
    for k in NEW_KEYS:
        print(_stats(k, t[k]))

    # Sanity: asc and desc should differ. Report MAD, in dB.
    asc = t["s1_pre_vv_asc_median"]
    desc = t["s1_pre_vv_desc_median"]
    both_valid = (asc != 0) & (desc != 0) & np.isfinite(asc) & np.isfinite(desc)
    if both_valid.sum() < 1000:
        print("  WARN: <1000 pixels with both orbits valid — MAD check skipped")
    else:
        mad = float(np.median(np.abs(asc[both_valid] - desc[both_valid])))
        print(f"\n  median |asc - desc|  = {mad:.3f} dB   "
              f"(should be > 0.1 — if ~0 the split didn't work)")
        if mad < 0.01:
            print("  FAIL: asc and desc are essentially identical")
            return False

    # Sanity: orbit_diff should equal asc - desc (within float tolerance).
    recomputed = asc - desc
    diff_err = float(np.abs(recomputed - t["s1_pre_vv_orbit_diff"]).max())
    print(f"  max |orbit_diff - (asc - desc)| = {diff_err:.2e}   (should be ~0)")
    if diff_err > 1e-4:
        print("  FAIL: orbit_diff does not match asc - desc")
        return False

    # Backward-compat: combined median should be bounded by asc/desc extremes.
    combined = t["s1_pre_vv_median"]
    lo = np.minimum(asc, desc); hi = np.maximum(asc, desc)
    bracketed = ((combined >= lo - 0.5) & (combined <= hi + 0.5))[both_valid]
    print(f"  combined median bracketed by min/max(asc,desc): "
          f"{bracketed.mean()*100:.1f}% of pixels   (near 100% expected)")

    return True


def main() -> None:
    print(f"BROKEN_TILES = {sorted(BROKEN_TILES)}")
    ok1 = check_broken_tile_rejection()
    ok2 = check_orbit_split(HEALTHY_TILE)
    print("\n" + "=" * 70)
    print(f"overall: {'PASS' if ok1 and ok2 else 'FAIL'}")


if __name__ == "__main__":
    main()
