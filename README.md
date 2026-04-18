# Osapiens Makeathon 2026 — Deforestation Detection

Challenge: predict post-2020 deforestation from multi-modal satellite data
(Sentinel-1 VV radar, Sentinel-2 L2A optical, AlphaEarth Foundations embeddings)
using weak alert labels from RADD, GLAD-L, and GLAD-S2.

Original challenge material lives under [`problem/`](problem/). Our code lives
under [`code/`](code/).

## What we built

### 1. Data pipeline — [`code/data/data.py`](code/data/data.py)

End-to-end pipeline from raw `.tif` tiles to a PyTorch `DataLoader`. Stages:

1. **Inventory** — [`build_inventory`](code/data/data.py) scans the on-disk
   layout and produces one [`TileInventory`](code/data/data.py) per tile.
2. **Reference grid** — a 2020 S2 scene defines the canonical CRS/transform;
   every other modality is reprojected onto it. Heterogeneous UTM zones /
   resolutions are handled in one place.
3. **Spectral indices** — NDVI, NBR, NDMI, EVI from S2 bands.
4. **Temporal composites** — NaN-aware median + std across pre-2020
   (`DEFAULT_PRE_YEARS = (2019, 2020)`) and post-2020 months per modality.
   Optional [Lee speckle filter](code/data/data.py) for S1.
5. **Label decoding + fusion** — decode RADD / GLAD-L / GLAD-S2 encodings
   ([`decode_radd`](code/data/data.py), [`decode_glads2`](code/data/data.py),
   [`decode_gladl`](code/data/data.py)), filter to post-2020, then fuse with
   ≥2-source consensus → `label` + per-pixel `label_confidence`.
6. **Free forest ground-truth** — any pixel flagged by *any* source is proven
   to have been forest before the alert. [`build_forest_ground_truth`](code/data/data.py)
   emits `forest_gt_pre2020` **for evaluation only** — it is never fed into
   training.
7. **Caching** — [`cache_tile`](code/data/data.py) writes preprocessed tensors
   to `.npz` so epochs don't reproject.
8. **Dataset/DataLoader** — [`DeforestationPatchDataset`](code/data/data.py)
   samples random crops with positive-biased sampling and flip/rot augmentation;
   [`build_dataloaders`](code/data/data.py) splits tiles by MGRS zone
   (singleton zones pooled and redistributed to hit `val_frac`).

### 2. Pre-2020 forest mask — [`code/model/vegetationPredictor.py`](code/model/vegetationPredictor.py)

A precision-biased "was forest pre-2020" mask. Every false-forest pixel becomes
a bogus deforestation candidate downstream, so we prefer tight over broad.

- [`HeuristicForestMasker`](code/model/vegetationPredictor.py): rule-based
  conjunction — NDVI ≥ τ AND NBR ≥ τ AND NDVI-std ≤ τ AND connected-component
  size ≥ N.
- [`LearnedForestMasker`](code/model/vegetationPredictor.py): LightGBM (default)
  or logistic regression trained on weak-label positives
  (`forest_gt_pre2020`) vs persistent low-NDVI∧low-NBR negatives. Ambiguous
  pixels are excluded from training, so the decision boundary is calibrated
  against actual deforestation events.
- [`predict_tiers`](code/model/vegetationPredictor.py) bins probabilities into
  four confidence tiers (`NON_FOREST` / `UNCERTAIN` / `SOFT` / `STRONG`) so
  downstream losses can weight pixels accordingly.
- [`evaluate_mask`](code/model/vegetationPredictor.py) +
  [`sweep_ndvi_threshold`](code/model/vegetationPredictor.py) score a mask
  against the free ground-truth: recall-vs-alerts is the key metric
  (≥0.95 is healthy; below ~0.9 means the mask is too tight).

### 3. Forest mask — validation results

We validated both maskers on 10 cached training tiles against the free
ground-truth derived from post-2020 alert unions.

**Step 1 — heuristic sweep.** Initial run had aggregate recall **0.18** at
coverage 0.08 because the `max_ndvi_std=0.18` gate wiped 5 tiles to zero. Real
forest has NDVI std 0.2–0.3 across wet/dry seasons, so the gate was too
aggressive. Disabling it (and re-running across NDVI thresholds) gave the knee
of the precision-recall curve:

| ndvi_threshold | recall    | coverage  |
|----------------|-----------|-----------|
| 0.45           | 0.985     | 0.788     |
| 0.50           | 0.981     | 0.772     |
| 0.55           | 0.973     | 0.743     |
| **0.60**       | **0.956** | **0.691** |
| 0.65           | 0.922     | 0.622     |
| 0.70           | 0.849     | 0.534     |

`ndvi_threshold=0.60` is the tightest setting that still clears ≥0.95 recall.

**Step 2 — per-tile audit.** At ndvi=0.55 one tile (`48PUT_0_8`) still had
recall 0.82. Probing the missed pixels showed NDVI 0.42 — degraded/secondary
forest that no single NDVI threshold can separate from pasture. This is where
the learned masker has to carry the weight.

**Step 3 — PU-learning fix.** The naive `LearnedForestMasker` blew up to 0.95
coverage (essentially calling everything forest). Root cause: negatives were
only pixels with `NDVI ≤ 0.25 AND NBR ≤ 0.10` (bare/water/urban). Pasture at
NDVI 0.4 was never shown as a negative, so at inference the model put it on
the forest side of the boundary — classic positive-unlabeled pathology. We
added *seasonal* negatives (`NDVI_std > 0.15 AND NDVI < 0.55` → cropland/
pasture signature) so the model actually learns the pasture boundary.

**Final results** (`ndvi_threshold=0.55` heuristic vs LightGBM learned, all
negatives included):

| Model                    | recall    | coverage  |
|--------------------------|-----------|-----------|
| Heuristic (NDVI+NBR)     | 0.973     | 0.743     |
| **Learned (LightGBM)**   | **0.982** | **0.775** |

On the weak tile `48PUT_0_8`: heuristic recall 0.824 → learned recall 0.934,
coverage 0.534 → 0.572. The learned masker rescues 62.7% of the heuristic's
missed degraded-forest pixels for only +4% coverage cost.

Tier distribution on that tile: NON 40.2%, UNCERTAIN 3.6%, SOFT 1.5%, STRONG
54.7% — a meaningful confidence breakdown (previously it was 8/0/0/91, which
was the blow-up signature).

**Decision: the learned masker is the pre-2020 forest prior going forward.**
Downstream models should use `predict_tiers` and weight the loss as:

| Tier         | Weight |
|--------------|--------|
| STRONG       | 1.0    |
| SOFT         | 0.5    |
| UNCERTAIN    | 0.0    |
| NON_FOREST   | 0.0    |

### 4. Attack plan — [`ATTACK_PLAN.md`](ATTACK_PLAN.md)

Three-tier modeling roadmap: LightGBM pixel baseline → U-Net on stacked
multi-modal features → U-TAE + ViT-style cross-attention between S1/S2
temporal tokens and AEF embeddings.

## Repo layout

```text
makeathon-challenge-2026/
├── ATTACK_PLAN.md          ← 3-tier modeling plan
├── README.md
├── requirements.txt
├── code/
│   ├── run.ipynb           ← driver notebook
│   ├── data/
│   │   ├── __init__.py
│   │   └── data.py         ← pipeline
│   └── model/
│       ├── __init__.py
│       └── vegetationPredictor.py
├── data/                   ← raw S1/S2/AEF/labels   (gitignored)
├── cache/                  ← preprocessed .npz      (gitignored)
└── problem/                ← original challenge notebook + data loader
```

## How to run

### Fast path — use the pre-built cache

The full preprocessing pipeline (reprojection + temporal compositing + label
fusion + forest ground-truth) takes a while to run from the raw `.tif` tiles.
We've published the cached output so you can skip straight to modelling.

**Download:** [cache.zip (LRZ Sync+Share)](https://syncandshare.lrz.de/getlink/fiJeygjgk3L8NkvYHuFrNd/cache.zip)

```bash
# From the repo root:
curl -L -o cache.zip \
    "https://syncandshare.lrz.de/getlink/fiJeygjgk3L8NkvYHuFrNd/cache.zip"
unzip cache.zip                      # produces cache/*.npz
```

Then install deps and you're ready to train:

```bash
pip install -r requirements.txt
pip install lightgbm                 # used by LearnedForestMasker
```

### Slow path — rebuild the cache from raw tiles

```bash
# 1. Download the challenge data (see problem/ for the Makefile).
make -C problem download_data_from_s3

# 2. Smoke-test the pipeline on a single tile.
python -m code.data.data \
    --root ../data/makeathon-challenge \
    --tile <tile_id>

# 3. Cache all tiles and validate the forest mask.
python -m code.model.vegetationPredictor --cache_dir ./cache --sweep
python -m code.model.vegetationPredictor --cache_dir ./cache --learned --backend lightgbm
```

## What's inside `cache/`

One `.npz` file per tile (`<tile_id>.npz`, e.g. `48PUT_0_8.npz`). Each file
contains a dict of `np.ndarray`s already reprojected onto the tile's
canonical 10 m UTM grid, with `H, W ≈ 1000`. Load with:

```python
import numpy as np
t = dict(np.load("cache/48PUT_0_8.npz", allow_pickle=False))
t["s2_pre_ndvi_median"].shape     # → (H, W)
t["aef_pre"].shape                # → (64, H, W)
```

### Keys present on every tile (train and test)

**AlphaEarth Foundations embeddings** — 64-dim semantic vectors per pixel,
temporally reduced across years:

| Key         | Shape        | dtype   | Meaning                                         |
|-------------|--------------|---------|-------------------------------------------------|
| `aef_pre`   | `(64, H, W)` | float32 | Median AEF embedding over **2019–2020**         |
| `aef_post`  | `(64, H, W)` | float32 | Median AEF embedding over **2021–2024**         |
| `aef_delta` | `(64, H, W)` | float32 | `aef_post − aef_pre` (change-detection signal)  |

**Sentinel-2 raw bands** — 12 L2A bands (B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12), scaled to `[0, 1]`, temporally reduced. All float32, shape `(12, H, W)`:

- `s2_pre_band_median` — per-band median across pre-2020 scenes (cloud-robust).
- `s2_post_band_median` — per-band median across post-2020 scenes.
- `s2_pre_band_std` — per-band std pre-2020 (seasonality + residual cloud variance).
- `s2_post_band_std` — per-band std post-2020.

**Sentinel-2 spectral indices** — NDVI, NBR, NDMI, EVI, computed per-scene then reduced. All float32, shape `(H, W)`. For each `{idx}` ∈ `{ndvi, nbr, ndmi, evi}`:

- `s2_pre_{idx}_median` / `s2_pre_{idx}_std` — pre-2020 reductions.
- `s2_post_{idx}_median` / `s2_post_{idx}_std` — post-2020 reductions.
- `s2_delta_{idx}` — `post_median − pre_median`. Primary change-detection signal per index.

What each index means:

- **NDVI** `= (NIR − Red) / (NIR + Red)` — green biomass, general vegetation health.
- **NBR** `= (NIR − SWIR2) / (NIR + SWIR2)` — burn scars, canopy moisture. Drops sharply on fire or clear-cut.
- **NDMI** `= (NIR − SWIR1) / (NIR + SWIR1)` — canopy water content.
- **EVI** `= 2.5·(NIR − Red) / (NIR + 6·Red − 7.5·Blue + 1)` — biomass in dense canopy; saturates less than NDVI.

**Sentinel-1 VV radar (dB, already log-scaled).** All float32, shape `(H, W)`:

- `s1_pre_vv_median` / `s1_post_vv_median` — median VV backscatter (dB) in each window.
- `s1_pre_vv_std` / `s1_post_vv_std` — temporal std.
- `s1_delta_vv` — `post − pre`. Drops on canopy removal (smoother surface → lower backscatter).

**Pre-2020 forest prior.** `forest_mask_2020`, uint8 `(H, W)` — simple NDVI ≥ 0.6 threshold on pre-2020 NDVI median. **Prefer [`LearnedForestMasker`](code/model/vegetationPredictor.py) for a calibrated mask** (see §3).

**Tile geometry.** `_shape`, int32 `(2,)` — stored `(H, W)` of the canonical grid.

### Keys present **only on training tiles**

- **`label`** — uint8 `(H, W)`. Consensus deforestation mask: 1 iff ≥2 of {RADD, GLAD-L, GLAD-S2} agreed on a post-2020 alert, gated by `forest_mask_2020`. **This is the training target.**
- **`label_confidence`** — float32 `(H, W)` ∈ `[0, 1]`. Mean confidence of contributing sources; single-source pixels are kept at 0.3× their mean (ambiguous). Use as `sample_weight` / loss weight.
- **`forest_gt_pre2020`** — uint8 `(H, W)`. **EVALUATION ONLY.** Union of any post-2020 alert source — these pixels are proven to have been forest pre-2020. Used to validate the forest mask. **Do NOT feed this into training** — it leaks the target that `label` already derives from.

### Load it in your own code

```python
import numpy as np
from pathlib import Path

for p in sorted(Path("cache").glob("*.npz")):
    t = dict(np.load(p, allow_pickle=False))
    # e.g. build a change-detection feature stack
    X = np.concatenate([
        t["aef_delta"],                          # (64, H, W)
        t["s2_delta_ndvi"][None],                # (1, H, W)
        t["s2_delta_nbr"][None],
        t["s1_delta_vv"][None],
    ], axis=0)
    y = t["label"]                               # train tiles only
    w = t["label_confidence"]
```

Or let [`DeforestationPatchDataset`](code/data/data.py) handle stacking,
random cropping, and augmentation:

```python
from data import DeforestationPatchDataset, DEFAULT_FEATURE_KEYS
from torch.utils.data import DataLoader

ds = DeforestationPatchDataset(
    cache_paths=sorted(Path("cache").glob("*.npz")),
    feature_keys=DEFAULT_FEATURE_KEYS,   # 251-channel stack
    patch_size=256, patches_per_tile=8, is_train=True,
)
loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)
for batch in loader:
    x, y, w, mask = batch["x"], batch["y"], batch["w"], batch["mask"]
```

From Python:

```python
from code.data import build_dataloaders
from code.model import HeuristicForestMasker, LearnedForestMasker, evaluate_masker_on_tiles

train_loader, val_loader, inv = build_dataloaders(
    root="../data/makeathon-challenge",
    cache_dir="./cache",
    preprocess_kwargs=dict(apply_lee_filter=True),
)

# Heuristic mask
masker = HeuristicForestMasker(ndvi_threshold=0.6)

# Or learned
cache_paths = sorted(Path("./cache").glob("*.npz"))
tensors_list = [dict(np.load(p)) for p in cache_paths]
masker = LearnedForestMasker(backend="lightgbm").fit(tensors_list)

report = evaluate_masker_on_tiles(masker, cache_paths)
```

## Dependencies

Core (from [`requirements.txt`](requirements.txt)): `numpy`, `rasterio`,
`scipy`, `torch`, `scikit-learn`.

Optional: `lightgbm` — enables the default `LearnedForestMasker` backend.
Without it, pass `backend="logreg"` for the linear fallback.
