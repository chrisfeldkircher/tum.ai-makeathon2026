"""Pre-2020 forest mask prediction + evaluation.

Produces a high-precision "was forest pre-2020" mask from the cached tile
features, then evaluates it against the free ground-truth derived from weak
labels (`forest_gt_pre2020` in the cache — union of post-2020 alert sources).

Two maskers:
    HeuristicForestMasker  — rule-based consensus on persistent vegetation.
    LearnedForestMasker    — logistic regression trained on alert pixels
                             (positives) vs stable low-vegetation pixels
                             (negatives). Optional sklearn dependency.

Why a dedicated pre-2020 mask matters
-------------------------------------
Every false forest pixel becomes a bogus deforestation candidate downstream,
so the mask should be precision-biased: tighter is better than broader. The
heuristic below is a conjunction (NDVI AND NBR AND low std AND min size)
rather than a disjunction.

Evaluation uses the free ground-truth:
    recall_vs_alerts = fraction of post-2020 alert pixels inside the mask.
                       (≥0.95 is healthy; <0.90 → mask is too tight.)
    coverage         = fraction of the tile classified as forest.
                       (Sanity-check: should be in the right ballpark for
                        the biome; absurd values signal broken features.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage

try:
    from sklearn.linear_model import LogisticRegression
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False


# Tier codes returned by LearnedForestMasker.predict_tiers. Downstream code
# can gate loss weights on these (e.g. drop UNCERTAIN, half-weight SOFT).
TIER_NON_FOREST = 0
TIER_UNCERTAIN  = 1
TIER_SOFT       = 2
TIER_STRONG     = 3

logger = logging.getLogger(__name__)


@dataclass
class HeuristicForestMasker:
    """Rule-based pre-2020 forest mask.

    A pixel passes if ALL of:
      - pre-2020 median NDVI ≥ ndvi_threshold
      - pre-2020 median NBR  ≥ nbr_threshold          (if nbr_threshold not None)
      - pre-2020 NDVI std    ≤ max_ndvi_std           (if enabled — filters
                                                        seasonal crops)
      - the pixel is inside a connected component of size ≥ min_component_pixels
    """

    ndvi_threshold: float = 0.6
    nbr_threshold: float | None = 0.3
    max_ndvi_std: float | None = 0.18
    min_component_pixels: int = 100
    # 0 disables morphological closing; raise to 3 if the mask is peppered
    # with 1–2 pixel holes caused by per-scene cloud gaps.
    morph_closing_size: int = 0

    def predict(self, tensors: dict[str, np.ndarray]) -> np.ndarray:
        """Takes the dict produced by `preprocess_tile` (or a cached .npz) and
        returns a (H, W) uint8 mask."""
        ndvi_med = tensors["s2_pre_ndvi_median"]
        mask = ndvi_med >= self.ndvi_threshold

        if self.nbr_threshold is not None and "s2_pre_nbr_median" in tensors:
            mask &= tensors["s2_pre_nbr_median"] >= self.nbr_threshold

        if self.max_ndvi_std is not None and "s2_pre_ndvi_std" in tensors:
            mask &= tensors["s2_pre_ndvi_std"] <= self.max_ndvi_std

        if self.morph_closing_size > 0:
            struct = np.ones((self.morph_closing_size,) * 2, dtype=bool)
            mask = ndimage.binary_closing(mask, structure=struct)

        if self.min_component_pixels > 0:
            mask = _filter_small_components(mask, self.min_component_pixels)

        return mask.astype(np.uint8)


def _filter_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    if mask.sum() == 0:
        return mask
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labeled]


_LEARNED_FEATURE_KEYS: tuple[str, ...] = (
    "s2_pre_ndvi_median", "s2_pre_ndvi_std",
    "s2_pre_nbr_median",  "s2_pre_nbr_std",
    "s2_pre_ndmi_median", "s2_pre_ndmi_std",
    "s2_pre_evi_median",  "s2_pre_evi_std",
    "s1_pre_vv_median",   "s1_pre_vv_std",
)


def _per_pixel_features(tensors: dict[str, np.ndarray],
                        keys: Iterable[str] = _LEARNED_FEATURE_KEYS,
                        include_aef: bool = True) -> np.ndarray:
    """Build a (N_pixels, n_features) matrix from a cached tile dict."""
    feats = []
    for k in keys:
        if k in tensors:
            feats.append(tensors[k].reshape(-1))
    if include_aef and "aef_pre" in tensors:
        aef = tensors["aef_pre"]
        # Use channel-wise mean/std as a cheap summary of the 64-dim AEF stack
        # — feeding all 64 channels explodes the training-pixel matrix.
        feats.append(aef.mean(axis=0).reshape(-1))
        feats.append(aef.std(axis=0).reshape(-1))
    return np.stack(feats, axis=1).astype(np.float32)


@dataclass
class LearnedForestMasker:
    """Gradient-boosted (LightGBM) or linear (sklearn LR) classifier trained on
    weak-label-derived positives / strong negatives.

    Positives: pixels in `forest_gt_pre2020` (any post-2020 alert → was forest).
    Negatives: pixels that (a) are NOT in `forest_gt_pre2020` AND
                           (b) have persistently low NDVI AND low NBR across
                               pre-2020 years — i.e. almost certainly not forest.

    Only high-confidence positives and high-confidence negatives enter training;
    ambiguous pixels are excluded. The decision boundary is therefore calibrated
    against actual deforestation events rather than hand-tuned thresholds.

    Backends:
        backend="lightgbm"  — default, handles nonlinear interactions better.
        backend="logreg"    — linear baseline; interpretable via .coef_.

    Outputs a calibrated forest-confidence score in [0, 1]. Use `predict_tiers`
    for downstream code that wants soft/strong bands rather than a hard 0/1.
    """

    backend: str = "lightgbm"
    threshold: float = 0.5

    # Tier thresholds for `predict_tiers`. The gap between `tier_non_forest_max`
    # and `tier_soft_min` leaves room for an explicit UNCERTAIN band — pixels
    # the model is not confident enough about should be excluded from training.
    tier_non_forest_max: float = 0.30
    tier_soft_min: float = 0.60
    tier_strong_min: float = 0.85

    negative_ndvi_max: float = 0.25
    negative_nbr_max: float = 0.10
    max_pixels_per_tile: int = 50_000

    lgbm_num_leaves: int = 63
    lgbm_learning_rate: float = 0.05
    lgbm_n_estimators: int = 300
    lgbm_min_data_in_leaf: int = 200

    C: float = 1.0
    class_weight: str | dict | None = "balanced"

    feature_keys: tuple[str, ...] = field(default_factory=lambda: _LEARNED_FEATURE_KEYS)
    include_aef: bool = True

    model: object | None = None

    def _build_xy(self, tensors: dict[str, np.ndarray], rng: np.random.Generator
                 ) -> tuple[np.ndarray, np.ndarray] | None:
        if "forest_gt_pre2020" not in tensors:
            return None
        pos = tensors["forest_gt_pre2020"] > 0
        ndvi = tensors.get("s2_pre_ndvi_median")
        nbr  = tensors.get("s2_pre_nbr_median")
        ndvi_std = tensors.get("s2_pre_ndvi_std")
        if ndvi is None or nbr is None or ndvi_std is None:
            return None

        stable_low = (~pos) & (ndvi <= self.negative_ndvi_max) & (nbr <= self.negative_nbr_max)
        seasonal   = (~pos) & (ndvi_std > 0.15) & (ndvi < 0.55)
        neg = stable_low | seasonal

        feats = _per_pixel_features(tensors, self.feature_keys, self.include_aef)
        pos_flat = pos.reshape(-1); neg_flat = neg.reshape(-1)

        pos_idx = np.where(pos_flat)[0]
        neg_idx = np.where(neg_flat)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            return None

        k = min(len(pos_idx), len(neg_idx), self.max_pixels_per_tile // 2)
        pos_idx = rng.choice(pos_idx, size=k, replace=False)
        neg_idx = rng.choice(neg_idx, size=k, replace=False)

        X = np.concatenate([feats[pos_idx], feats[neg_idx]], axis=0)
        y = np.concatenate([np.ones(k, dtype=np.int8), np.zeros(k, dtype=np.int8)])
        return X, y

    def _fit_backend(self, X: np.ndarray, Y: np.ndarray):
        if self.backend == "lightgbm":
            if not _HAS_LIGHTGBM:
                raise ImportError(
                    "lightgbm not installed. `pip install lightgbm` "
                    "or use backend='logreg' for the linear fallback."
                )
            self.model = lgb.LGBMClassifier(
                num_leaves=self.lgbm_num_leaves,
                learning_rate=self.lgbm_learning_rate,
                n_estimators=self.lgbm_n_estimators,
                min_data_in_leaf=self.lgbm_min_data_in_leaf,
                objective="binary",
                class_weight="balanced" if self.class_weight == "balanced" else None,
                n_jobs=-1,
                verbosity=-1,
            ).fit(X, Y)
            train_acc = float((self.model.predict(X) == Y).mean())
        elif self.backend == "logreg":
            if not _HAS_SKLEARN:
                raise ImportError("scikit-learn is required for backend='logreg'.")
            self.model = LogisticRegression(
                C=self.C, class_weight=self.class_weight, max_iter=1000, n_jobs=-1,
            ).fit(X, Y)
            train_acc = float(self.model.score(X, Y))
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")
        return train_acc

    def fit(self, tensors_list: list[dict[str, np.ndarray]], seed: int = 0
           ) -> "LearnedForestMasker":
        rng = np.random.default_rng(seed)
        Xs, Ys = [], []
        for t in tensors_list:
            pair = self._build_xy(t, rng)
            if pair is None:
                continue
            Xs.append(pair[0]); Ys.append(pair[1])
        if not Xs:
            raise RuntimeError("No training pixels — check that tiles have forest_gt_pre2020.")
        X = np.concatenate(Xs, axis=0)
        Y = np.concatenate(Ys, axis=0)
        X = np.clip(X, -1e3, 1e3)
        train_acc = self._fit_backend(X, Y)
        logger.info(
            f"LearnedForestMasker[{self.backend}] fitted on {len(Y):,} pixels "
            f"from {len(Xs)} tiles. Train accuracy: {train_acc:.3f}"
        )
        return self

    def predict_proba(self, tensors: dict[str, np.ndarray]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call .fit(...) before .predict_proba(...).")
        feats = _per_pixel_features(tensors, self.feature_keys, self.include_aef)
        feats = np.clip(feats, -1e3, 1e3)
        proba = self.model.predict_proba(feats)[:, 1]
        H, W = tensors["s2_pre_ndvi_median"].shape
        return proba.reshape(H, W).astype(np.float32)

    def predict(self, tensors: dict[str, np.ndarray]) -> np.ndarray:
        return (self.predict_proba(tensors) >= self.threshold).astype(np.uint8)

    def predict_tiers(self, tensors: dict[str, np.ndarray]) -> np.ndarray:
        """Discretise the forest confidence into the four tiers.

        Returns a (H, W) uint8 array using the TIER_* constants:
            TIER_NON_FOREST  p <  tier_non_forest_max
            TIER_UNCERTAIN   tier_non_forest_max <= p < tier_soft_min
            TIER_SOFT        tier_soft_min <= p < tier_strong_min
            TIER_STRONG      p >= tier_strong_min
        """
        p = self.predict_proba(tensors)
        tiers = np.full(p.shape, TIER_UNCERTAIN, dtype=np.uint8)
        tiers[p <  self.tier_non_forest_max] = TIER_NON_FOREST
        tiers[p >= self.tier_soft_min]       = TIER_SOFT
        tiers[p >= self.tier_strong_min]     = TIER_STRONG
        return tiers

    def feature_importance(self) -> np.ndarray | None:
        """Return LGBM `feature_importances_` or LR abs(`coef_`). None if unfit."""
        if self.model is None:
            return None
        if self.backend == "lightgbm":
            return np.asarray(self.model.feature_importances_)
        return np.abs(self.model.coef_[0])


@dataclass
class MaskEvaluation:
    recall_vs_alerts: float
    n_alert_pixels: int
    n_alert_in_mask: int
    mask_coverage: float
    mask_pixels: int
    tile_pixels: int

    def as_dict(self) -> dict:
        return {
            "recall_vs_alerts": self.recall_vs_alerts,
            "n_alert_pixels":   self.n_alert_pixels,
            "n_alert_in_mask":  self.n_alert_in_mask,
            "mask_coverage":    self.mask_coverage,
            "mask_pixels":      self.mask_pixels,
            "tile_pixels":      self.tile_pixels,
        }


def evaluate_mask(mask: np.ndarray, forest_gt: np.ndarray) -> MaskEvaluation:
    """Evaluate a predicted forest mask against free ground-truth pixels.

    `forest_gt` pixels were proven to be forest pre-2020 because they were
    later flagged as deforestation alerts. Every such pixel *must* lie inside
    the predicted mask for the downstream deforestation pipeline to see them.
    """
    mask = mask > 0
    gt = forest_gt > 0
    n_gt = int(gt.sum())
    n_tp = int((mask & gt).sum())
    recall = float("nan") if n_gt == 0 else n_tp / n_gt
    return MaskEvaluation(
        recall_vs_alerts=recall,
        n_alert_pixels=n_gt,
        n_alert_in_mask=n_tp,
        mask_coverage=float(mask.mean()),
        mask_pixels=int(mask.sum()),
        tile_pixels=int(mask.size),
    )


def evaluate_masker_on_tiles(masker, cache_paths: list[Path | str]) -> dict[str, dict]:
    """Run `masker.predict` on every cached tile and report per-tile + aggregate metrics."""
    per_tile: dict[str, dict] = {}
    total_gt = total_tp = total_mask = total_pix = 0
    for p in cache_paths:
        p = Path(p)
        with np.load(p, allow_pickle=False) as npz:
            tensors = {k: npz[k] for k in npz.files}
        if "forest_gt_pre2020" not in tensors:
            continue
        mask = masker.predict(tensors)
        ev = evaluate_mask(mask, tensors["forest_gt_pre2020"])
        per_tile[p.stem] = ev.as_dict()
        total_gt   += ev.n_alert_pixels
        total_tp   += ev.n_alert_in_mask
        total_mask += ev.mask_pixels
        total_pix  += ev.tile_pixels

    aggregate = {
        "recall_vs_alerts": (total_tp / total_gt) if total_gt else float("nan"),
        "mask_coverage":    (total_mask / total_pix) if total_pix else float("nan"),
        "n_alert_pixels":   total_gt,
        "n_tiles":          len(per_tile),
    }
    return {"per_tile": per_tile, "aggregate": aggregate}


def sweep_ndvi_threshold(cache_paths: list[Path | str],
                         ndvi_grid: Iterable[float] = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70),
                         base_masker_kwargs: dict | None = None) -> list[dict]:
    """Grid-search the NDVI threshold; report recall-vs-alerts and coverage.

    Use this to find the knee of the recall curve — the tightest threshold
    that still covers the free-ground-truth alert pixels.
    """
    base_masker_kwargs = dict(base_masker_kwargs or {})
    rows = []
    for thr in ndvi_grid:
        masker = HeuristicForestMasker(ndvi_threshold=thr, **base_masker_kwargs)
        result = evaluate_masker_on_tiles(masker, cache_paths)
        agg = result["aggregate"]
        agg["ndvi_threshold"] = thr
        rows.append(agg)
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="./cache")
    ap.add_argument("--ndvi", type=float, default=0.6)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--learned", action="store_true",
                    help="Fit the LearnedForestMasker on cached tiles and evaluate it.")
    ap.add_argument("--backend", default="lightgbm", choices=["lightgbm", "logreg"])
    args = ap.parse_args()

    paths = sorted(Path(args.cache_dir).glob("*.npz"))
    if not paths:
        raise SystemExit(f"No .npz files in {args.cache_dir}.")
    logger.info(f"Evaluating on {len(paths)} tiles.")
    logging.basicConfig(level=logging.INFO)

    if args.sweep:
        rows = sweep_ndvi_threshold(paths)
        print(f"{'ndvi':>6} {'recall':>8} {'coverage':>10} {'n_alerts':>10}")
        for r in rows:
            print(f"{r['ndvi_threshold']:>6.2f} {r['recall_vs_alerts']:>8.3f} "
                  f"{r['mask_coverage']:>10.3f} {r['n_alert_pixels']:>10d}")
    elif args.learned:
        tensors_list = []
        for p in paths:
            with np.load(p, allow_pickle=False) as npz:
                tensors_list.append({k: npz[k] for k in npz.files})
        masker = LearnedForestMasker(backend=args.backend).fit(tensors_list)
        result = evaluate_masker_on_tiles(masker, paths)
        print(f"Learned[{args.backend}] — aggregate: {result['aggregate']}")
        fi = masker.feature_importance()
        if fi is not None:
            print(f"Feature importance (top 5): {np.argsort(fi)[::-1][:5].tolist()}")
    else:
        masker = HeuristicForestMasker(ndvi_threshold=args.ndvi)
        result = evaluate_masker_on_tiles(masker, paths)
        print(f"Heuristic (ndvi≥{args.ndvi}) — aggregate: {result['aggregate']}")
