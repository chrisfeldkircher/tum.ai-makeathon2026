"""P2: post-period forest probability + delta channel (handoff to deforestation team).

Trains two LearnedForestMaskers:
  1. pre-period (P1-shipped config) — same as live model
  2. post-period — swap all s2_pre_/s1_pre_/aef_pre keys for post-period equivalents

For each tile in the cache, writes a companion .npz with:
  forest_prob_pre   (H, W) float32 in [0, 1]
  forest_prob_post  (H, W) float32 in [0, 1]
  forest_prob_delta (H, W) float32 in [-1, 1]  (pre - post)

Sanity check: report correlation between delta>0.3 and label==1 (expected: deforested
pixels show positive delta).

Output dir: cache_probs/<tile_id>.npz  (does not mutate the shared cache).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from model.vegetationPredictor import (  # noqa: E402
    LearnedForestMasker,
)

# Post-period feature subset aligned to the latest downstream cache.
# We intentionally use only post-period analogues of features that actually
# exist after the 2020-only pre-period refactor.
_POST_FEATURE_KEYS: tuple[str, ...] = (
    "s2_pre_ndvi_median", "s2_pre_ndvi_std",
    "s2_pre_nbr_median",  "s2_pre_nbr_std",
    "s2_pre_ndmi_median", "s2_pre_ndmi_std",
    "s2_pre_evi_median",  "s2_pre_evi_std",
    "s1_pre_vv_median",
    "s1_pre_vv_asc_median",  "s1_pre_vv_asc_std",
    "s1_pre_vv_desc_median", "s1_pre_vv_desc_std",
    "s1_pre_vv_orbit_diff",
)

_POST_REQUIRED_SOURCE_KEYS: tuple[str, ...] = (
    "s2_post_ndvi_median", "s2_post_ndvi_std",
    "s2_post_nbr_median",  "s2_post_nbr_std",
    "s2_post_ndmi_median", "s2_post_ndmi_std",
    "s2_post_evi_median",  "s2_post_evi_std",
    "s1_post_vv_median",
    "s1_post_vv_asc_median",  "s1_post_vv_asc_std",
    "s1_post_vv_desc_median", "s1_post_vv_desc_std",
    "s1_post_vv_orbit_diff",
    "aef_post",
)


def _missing_required_keys(tensors: dict[str, np.ndarray], required_keys: tuple[str, ...]) -> list[str]:
    return [k for k in required_keys if k not in tensors]


def load_tiles(cache_dir: Path, require_gt: bool = True):
    probe = LearnedForestMasker()
    pre_required = tuple(probe.feature_keys) + ("aef_pre",)
    tiles = {}
    for p in sorted(cache_dir.glob("*.npz")):
        with np.load(p, allow_pickle=True) as z:
            t = {k: z[k] for k in z.files}
        if require_gt and "forest_gt_pre2020" not in t:
            continue
        missing_pre = _missing_required_keys(t, pre_required)
        missing_post = _missing_required_keys(t, _POST_REQUIRED_SOURCE_KEYS)
        if missing_pre or missing_post:
            missing = sorted(set(missing_pre + missing_post))
            raise RuntimeError(
                f"Cache tile {p.stem} is stale for the merged downstream+forest pipeline.\n"
                f"Missing keys: {', '.join(missing[:12])}{'...' if len(missing) > 12 else ''}\n"
                "Regenerate the cache with the latest preprocessing pipeline before running P2."
            )
        tiles[p.stem] = t
    return tiles


def _post_view(tensors: dict[str, np.ndarray],
               mask_deforested_from_positives: bool = False) -> dict[str, np.ndarray]:
    """Return a tensors dict where post-period arrays are aliased to the
    pre-period keys the masker expects. Also swaps aef_pre -> aef_post.

    If `mask_deforested_from_positives` is True, the returned view also
    overrides `forest_gt_pre2020` to be the TRULY-STILL-FOREST set:
    (was forest pre-2020) AND (not deforested during study period).
    This is the correct positive set for training a post-period forest
    detector — otherwise the model would see deforested pixels under their
    post-period appearance as "forest" and learn nothing useful.
    """
    out = dict(tensors)  # shallow copy
    mapping = {
        "s2_pre_ndvi_median": "s2_post_ndvi_median",
        "s2_pre_ndvi_std":    "s2_post_ndvi_std",
        "s2_pre_nbr_median":  "s2_post_nbr_median",
        "s2_pre_nbr_std":     "s2_post_nbr_std",
        "s2_pre_ndmi_median": "s2_post_ndmi_median",
        "s2_pre_ndmi_std":    "s2_post_ndmi_std",
        "s2_pre_evi_median":  "s2_post_evi_median",
        "s2_pre_evi_std":     "s2_post_evi_std",
        "s1_pre_vv_median":   "s1_post_vv_median",
        "s1_pre_vv_asc_median":  "s1_post_vv_asc_median",
        "s1_pre_vv_asc_std":     "s1_post_vv_asc_std",
        "s1_pre_vv_desc_median": "s1_post_vv_desc_median",
        "s1_pre_vv_desc_std":    "s1_post_vv_desc_std",
        "s1_pre_vv_orbit_diff":  "s1_post_vv_orbit_diff",
        "aef_pre":            "aef_post",
    }
    for pre, post in mapping.items():
        if post in tensors:
            out[pre] = tensors[post]
    if mask_deforested_from_positives and "forest_gt_pre2020" in tensors and "label" in tensors:
        still_forest = ((tensors["forest_gt_pre2020"] > 0) &
                        (tensors["label"] == 0)).astype(np.uint8)
        out["forest_gt_pre2020"] = still_forest
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir",  type=Path, default=REPO_ROOT / "cache")
    ap.add_argument("--out_dir",    type=Path, default=REPO_ROOT / "cache_probs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load training tiles (need GT for both maskers)
    train_tiles = load_tiles(args.cache_dir, require_gt=True)
    all_tile_paths = sorted(args.cache_dir.glob("*.npz"))
    print(f"[P2] {len(train_tiles)} training tiles (with GT); {len(all_tile_paths)} total tiles")

    # 2) Train pre-period masker (P1-shipped config — tree dims on, default)
    print("[P2] training pre-period masker ...")
    pre_masker = LearnedForestMasker(backend="lightgbm").fit(list(train_tiles.values()),
                                                              seed=args.seed)
    # 3) Train post-period masker: feed post-period arrays under pre-period key names
    print("[P2] training post-period masker ...")
    post_views = [_post_view(t, mask_deforested_from_positives=True)
                  for t in train_tiles.values()]
    post_masker = LearnedForestMasker(
        backend="lightgbm",
        feature_keys=_POST_FEATURE_KEYS,
    ).fit(post_views, seed=args.seed)

    # 4) Predict + cache per tile
    print(f"[P2] predicting + writing probs to {args.out_dir} ...")
    # Track sanity-check stats across all tiles with labels
    all_delta, all_label = [], []
    all_pre, all_label_for_pre = [], []

    for tpath in all_tile_paths:
        tid = tpath.stem
        with np.load(tpath, allow_pickle=True) as z:
            t = {k: z[k] for k in z.files}
        prob_pre  = pre_masker.predict_proba(t)               # (H, W) float32 in [0,1]
        prob_post = post_masker.predict_proba(_post_view(t))  # (H, W) float32 in [0,1]
        prob_delta = (prob_pre - prob_post).astype(np.float32)

        out_path = args.out_dir / f"{tid}.npz"
        np.savez_compressed(
            out_path,
            forest_prob_pre=prob_pre.astype(np.float32),
            forest_prob_post=prob_post.astype(np.float32),
            forest_prob_delta=prob_delta,
        )
        # Aggregate sanity stats on training tiles with labels
        if "label" in t:
            lbl = (t["label"] > 0).astype(np.uint8).reshape(-1)
            all_delta.append(prob_delta.reshape(-1))
            all_label.append(lbl)
            all_pre.append(prob_pre.reshape(-1))
            all_label_for_pre.append(lbl)

    # 5) Sanity: does delta correlate with deforestation labels?
    if all_delta:
        d = np.concatenate(all_delta)
        lbl = np.concatenate(all_label)
        # Restrict to tiles where pre-period predicts forest — delta is only
        # meaningful for pixels that *were* forest
        pre = np.concatenate(all_pre)
        fm = pre >= 0.5  # "was forest pre-period"
        d = d[fm]; lbl = lbl[fm]

        # Class-conditional delta means
        mu_def  = float(d[lbl == 1].mean()) if (lbl == 1).any() else float("nan")
        mu_ndef = float(d[lbl == 0].mean()) if (lbl == 0).any() else float("nan")
        std_ndef = float(d[lbl == 0].std() + 1e-8)
        sep = abs(mu_def - mu_ndef) / std_ndef

        # Predictive power of delta > 0.3
        pred = d > 0.3
        tp = int((pred & (lbl == 1)).sum()); fp = int((pred & (lbl == 0)).sum())
        fn = int((~pred & (lbl == 1)).sum()); tn = int((~pred & (lbl == 0)).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        print(f"\n[P2] SANITY (pixels with prob_pre>=0.5, train tiles):")
        print(f"  mean delta | deforested pixels (label=1): {mu_def:+.4f}")
        print(f"  mean delta | not-deforested (label=0):    {mu_ndef:+.4f}")
        print(f"  separation (|dmu| / sd_neg):              {sep:.3f}")
        print(f"  delta>0.3 as deforestation indicator: precision={prec:.3f} recall={rec:.3f}")

        verdict = "SHIP" if sep > 0.5 else ("WEAK" if sep > 0.2 else "DEAD")
        print(f"[P2] signal verdict: {verdict}")

    # 6) Write handoff report
    report = REPO_ROOT / "code" / "reports" / "P2_POST_PERIOD.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# P2 — Post-period forest probability channels\n\n")
        f.write(f"Wrote {len(all_tile_paths)} per-tile files to `cache_probs/`.\n\n")
        if all_delta:
            f.write(f"**Sanity:** on pre-forest pixels, mean `forest_prob_delta` "
                    f"= {mu_def:+.4f} for deforested (label=1) vs {mu_ndef:+.4f} "
                    f"for non-deforested (label=0). Separation |Δμ|/σ = {sep:.3f}. "
                    f"Verdict: **{verdict}**.\n\n")
            f.write(f"**`delta > 0.3` as deforestation indicator:** "
                    f"precision={prec:.3f}, recall={rec:.3f}.\n\n")
        f.write("See `P2_INTEGRATION.md` for handoff instructions.\n")
    print(f"[P2] wrote {report}")


if __name__ == "__main__":
    main()
