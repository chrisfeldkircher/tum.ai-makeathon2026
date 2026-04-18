"""Visual QA for forest-mask predictions.

A six-panel overview per tile:
    (0) Pre-2020 RGB composite
    (1) Learned tier map (NON / UNCERTAIN / SOFT / STRONG)
    (2) Free forest ground truth (union of post-2020 alerts, pre-mask)
    (3) RGB + SOFT/STRONG overlay — visual plausibility
    (4) RGB + post-mask deforestation alerts — gating sanity check
    (5) TP/FN/FP map against `forest_gt_pre2020`

Failure modes to watch for
--------------------------
- Panel (5) dominated by red (FN): mask is too tight — alert pixels outside
  the predicted forest. Recall will suffer downstream.
- Panel (5) heavy orange (FP): mask claims forest where no alert ever fired.
  Could be fine (not every forest pixel gets deforested) but if coupled with
  visually bright/bare patches in panel (3), the masker is over-claiming.
- Panel (4) alerts outside any green area: the label-gate-by-mask step in
  `preprocess_tile` is misaligned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid hard matplotlib dep at import time
    from matplotlib.figure import Figure


def _rgb_from_s2(band_median: np.ndarray, stretch: float = 3.0) -> np.ndarray:
    """Build a display RGB from a (12, H, W) S2 median composite."""
    # Sentinel-2 bands are 1-indexed in reference; array is 0-indexed.
    rgb = np.stack([band_median[3], band_median[2], band_median[1]], axis=-1)
    return np.clip(rgb * stretch, 0.0, 1.0)


def plot_forest_mask_qa(
    tensors: dict[str, np.ndarray],
    masker,
    title: str = "",
    figsize: tuple[float, float] = (18, 10),
    rgb_stretch: float = 3.0,
) -> "Figure":
    """Multi-panel QA plot for a single tile's learned forest mask.

    Parameters
    ----------
    tensors : dict loaded from the `.npz` cache. Required keys: `s2_pre_band_median`.
              Uses `forest_gt_pre2020`, `label` if present (train tiles only).
    masker  : any object exposing `.predict_tiers(tensors)` → (H, W) uint8 in {0..3}.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    rgb = _rgb_from_s2(tensors["s2_pre_band_median"], stretch=rgb_stretch)
    tiers = masker.predict_tiers(tensors)
    forest_gt = tensors.get("forest_gt_pre2020")
    labels = tensors.get("label")

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    ax = axes.ravel()

    ax[0].imshow(rgb)
    ax[0].set_title("Pre-2020 RGB")

    tier_cmap = ListedColormap(["#1a1a1a", "#c43a3a", "#e8b005", "#2db84a"])
    ax[1].imshow(tiers, cmap=tier_cmap, vmin=0, vmax=3, interpolation="nearest")
    counts = np.bincount(tiers.ravel(), minlength=4)
    total = int(counts.sum())
    tier_txt = (f"NON {counts[0]/total:.0%}  UNC {counts[1]/total:.0%}  "
                f"SOFT {counts[2]/total:.0%}  STR {counts[3]/total:.0%}")
    ax[1].set_title(f"Tiers\n{tier_txt}", fontsize=9)

    if forest_gt is not None:
        ax[2].imshow(forest_gt > 0, cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
        ax[2].set_title(f"Forest GT (n={int((forest_gt>0).sum())})")
    else:
        ax[2].set_title("Forest GT (n/a — test tile)")

    ax[3].imshow(rgb)
    ax[3].imshow(np.ma.masked_where(tiers < 2, tiers),
                 cmap=ListedColormap(["#e8b005", "#2db84a"]),
                 vmin=2, vmax=3, alpha=0.40, interpolation="nearest")
    ax[3].set_title("RGB + SOFT/STRONG overlay")

    if labels is not None:
        ax[4].imshow(rgb)
        ax[4].imshow(np.ma.masked_where(labels == 0, labels),
                     cmap="autumn", alpha=0.85, interpolation="nearest")
        ax[4].set_title(f"Alerts over RGB (n={int((labels>0).sum())})")
    else:
        ax[4].set_title("Alerts (n/a — test tile)")

    if forest_gt is not None:
        pred = (tiers >= 2).astype(np.uint8)
        gt = (forest_gt > 0).astype(np.uint8)
        conf = np.zeros_like(pred, dtype=np.uint8)
        conf[(pred == 1) & (gt == 1)] = 1  # TP
        conf[(pred == 0) & (gt == 1)] = 2  # FN — missed forest we know existed
        conf[(pred == 1) & (gt == 0)] = 3  # FP — over-claimed
        conf_cmap = ListedColormap(["#1a1a1a", "#2db84a", "#c43a3a", "#e8b005"])
        ax[5].imshow(conf, cmap=conf_cmap, vmin=0, vmax=3, interpolation="nearest")
        n_gt = int(gt.sum())
        n_tp = int(((pred == 1) & (gt == 1)).sum())
        recall = (n_tp / n_gt) if n_gt else float("nan")
        ax[5].set_title(f"TP (grn) / FN (red) / FP (orn) — recall={recall:.3f}")
    else:
        ax[5].set_title("Confusion (n/a — test tile)")

    for a in ax:
        a.set_xticks([]); a.set_yticks([])

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_ndvi_drop(
    tensors: dict[str, np.ndarray],
    title: str = "",
    figsize: tuple[float, float] = (15, 5),
    magnitude_vmax: float = 0.6,
) -> "Figure":
    """Three-panel view of the NDVI-drop feature.

    Requires cache keys produced by `augment_cache_with_ndvi_drop`:
        ndvi_drop_magnitude, drop_doy, drop_year.
    """
    import matplotlib.pyplot as plt

    mag = tensors["ndvi_drop_magnitude"]
    doy = tensors["drop_doy"]
    year = tensors["drop_year"]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    im0 = axes[0].imshow(mag, cmap="magma", vmin=0, vmax=magnitude_vmax)
    axes[0].set_title("NDVI drop magnitude\n(baseline - NDVI_t*)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    masked_doy = np.ma.masked_where(doy == 0, doy)
    im1 = axes[1].imshow(masked_doy, cmap="twilight", vmin=1, vmax=366)
    axes[1].set_title("Drop DOY (black = no drop)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    masked_year = np.ma.masked_where(year == 0, year)
    im2 = axes[2].imshow(masked_year, cmap="viridis")
    axes[2].set_title("Drop year")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig
