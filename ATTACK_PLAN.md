# Attack Plan — osapiens Makeathon 2026: Detecting Deforestation from Space

## Task Recap

- Binary dense segmentation: predict deforestation polygons on test tiles.
- Deforestation = permanent tree-cover loss **after 2020** on land that was forest in 2020.
- Inputs: Sentinel-2 (12 bands, monthly), Sentinel-1 (VV, monthly, asc/desc), AlphaEarth Foundations (64-dim annual embeddings).
- Weak labels (training only): RADD, GLAD-L, GLAD-S2 — all noisy, each with different biases.
- Submission: binary raster per test tile → GeoJSON polygons in EPSG:4326 via `submission_utils.raster_to_geojson`.

## Cross-Cutting Concerns (do these first, regardless of tier)

### 1. Framing: change detection, not classification
The task definition *requires* pre-2020 forest state. Every feature/model should operate on `post_2020 - pre_2020` (or stacked pre/post) — never on a single timestamp alone.

### 2. CRS alignment
- S1, S2 → local UTM (e.g. EPSG:32618).
- AEF → EPSG:4326 (geodetic).
- Labels → mixed (RADD, GLAD-L, GLAD-S2 each differ).
- **Pick one target CRS per tile (UTM) and reproject everything with `rasterio.warp.reproject` before stacking.** Resampling: `bilinear` for continuous (AEF/S2/S1), `nearest` for labels.

### 3. Weak-label fusion
Never pick a single source. Build one consolidated label per tile:
- Binarise each source (any non-zero confidence = alert).
- Filter to alerts dated **after 2020-01-01** (decode the date encodings — see notebook §5).
- Consensus rule: **majority vote** (alert if ≥2 of 3 sources agree), with a weighted variant (RADD + GLAD-S2 higher weight, GLAD-L lower since it's Landsat-resolution).
- Optionally emit a `label_confidence` channel (0 / 0.33 / 0.66 / 1.0) to use as sample weight in training.

### 4. Forest mask (2020) — **in iterative refinement (Phase 0–4)**

Shipped as [`LearnedForestMasker`](code/model/vegetationPredictor.py) (LightGBM). The static aggregate number in the original README (recall 0.982 / coverage 0.775 on 10 tiles) was measured *train-on-test*; the real number from LOOCV on 14 gold-labeled tiles is considerably more honest. Current state after Phase 2:

- **LOOCV, 14 folds: recall 0.955 ± 0.042 (min 0.875, max 0.999)** — up from the Phase 0 baseline of 0.943 ± 0.052 (min 0.830). Phase 1 added strict confidence + NDVI spectral gate to `build_forest_ground_truth`; Phase 2 added per-pixel trajectory stats and neighborhood means (see below).
- Trained with a PU-learning fix: negatives include both *stable-low* (bare/water/urban) and *seasonal* (high NDVI std) pixels, so pasture isn't silently dragged onto the forest side of the boundary.
- Exposes 4 confidence tiers (NON / UNCERTAIN / SOFT / STRONG) via `predict_tiers`.

**Downstream contract:** gate labels by the mask AND weight the loss by tier.

| Tier       | Weight | Why                                     |
|------------|--------|-----------------------------------------|
| STRONG     | 1.0    | Confident forest — trust the label      |
| SOFT       | 0.5    | Probable forest — half-credit           |
| UNCERTAIN  | 0.0    | Drop — would add label noise            |
| NON_FOREST | 0.0    | Nothing to lose here by definition      |

### 5. Submission pipeline
Get `raster_to_geojson` running end-to-end on the training tiles on **day 1**, before touching any model. Submission-format bugs eat more time than modelling bugs.

---

## Phase Work Log — Forest Masker Refinement

The forest masker is the upstream gate for everything else; every false-forest pixel becomes a bogus deforestation candidate downstream, so pushing recall on this one model compounds into the submission metric. Work is split into four phases, each gated by a leave-one-tile-out CV check against the Phase 0 baseline so we can tell real wins from lucky tile splits.

| Phase | Scope | Status | CV recall (mean ± std, min) |
| --- | --- | --- | --- |
| **0** | Fix broken-tile ingest (18NYH_9_9 hard-skip), audit S1 asc/desc handling, build LOOCV harness | Done | **0.943 ± 0.052, min 0.830** (15 folds initially, later 14 after label gate) |
| **1** | Strict confidence + NDVI spectral gate in `build_forest_ground_truth` | Done | **0.955 ± 0.042, min 0.875** (14 folds) |
| **2** | Per-pixel NDVI/NBR/NDMI trajectory (p10/p90/slope), neighborhood means (9×9 / 21×21), B08 nodata fix for indices | Done — awaiting re-cache + CV | *pending* |
| **3** | MAESTRO cached enrichment: per-tile positive-centroid distance for hard-negative mining in `_build_xy`; feed embeddings to downstream segmenter, not the masker | Pending | — |
| **4** | Threshold sweep + isotonic calibration (sklearn `CalibratedClassifierCV`), refit tier boundaries | Pending | — |

**Why LOOCV and not a single fixed holdout.** Training set is tiny (14 tiles with labels). Any single holdout split is hostage to one tile's biome; a change that looks like +0.02 recall on one split can be −0.02 on the next. LOOCV gives a 14-fold distribution and a concrete `recall_min` we can monitor — improvements on the worst fold are more meaningful than improvements on the mean.

**Noise floor.** At 14 folds with std ~0.04, the 2σ band on the mean is roughly ±0.021. A mean-recall change below that is indistinguishable from fold-shuffling noise. `recall_min` and `recall_std` are the useful signals for small changes — Phase 1 moved the mean by 0.012 (at the noise floor) but the min by 0.045 and collapsed the tail, which is the real win.

**Phase 2 details.**

- `_pixel_slope` — vectorised per-pixel linear regression across the pre-period scene stack, NaN-skipping, scene-index time. Output is 0 where fewer than 3 scenes are valid.
- `_neighborhood_mean` — 9×9 and 21×21 uniform filter with explicit nodata handling (exact-0 excluded, mean reweighted by local valid-pixel count). Prevents tile edges from biasing the neighborhood signal toward zero.
- `s2_temporal_composite(compute_trajectory=True)` — emits `{idx}_p10 / _p90 / _slope` for NDVI/NBR/NDMI plus a shared `n_scenes_valid` alongside the existing medians/stds.
- **Nodata fix.** Previously `_safe_div` returned NDVI ≈ 0 for black-border pixels (B08 = 0 after reproject fill), which dragged medians — and p10 especially — on partial-coverage tiles. Now a per-scene `B08 > 0` mask NaN's those pixels out of every index before the reduction. This is why Phase 2 requires a full cache rebuild, not just the new keys.
- 14 new channels in `DEFAULT_FEATURE_KEYS` (9 trajectory + 1 n_valid + 4 neighborhood) → stack is now 274 channels. `_LEARNED_FEATURE_KEYS` picks them up.

---

## Tier 1 — LightGBM/XGBoost Baseline (day 1, ~4–6h)

**Goal:** end-to-end submission with a defensible leaderboard score, fast.

### Features (per 10m pixel)
- AEF embeddings: 64 dims (annual, reprojected to UTM).
- S2 spectral indices per month, then aggregated pre/post-2020:
  - Indices: NDVI, NBR, NDMI, EVI.
  - Aggregates: median, std, min, max — over months in 2020 (pre) and 2021+ (post).
  - **Deltas**: `post_median − pre_median` per index (the actual change-detection signal).
- S1 VV backscatter (dB): pre/post median, std, delta.
- Optional: DOY of strongest NDVI drop, coords (lat/lon) as leak-safe spatial priors.

Feature vector size ~80–120 per pixel.

### Training
- Subsample pixels (positives + hard negatives near positives) to avoid 99%-background imbalance.
- Weight samples by `label_confidence` from weak-label fusion.
- LightGBM, `binary` objective, `num_leaves=63`, early stopping on a held-out tile.
- Hyperparam search: Optuna, ~30 trials on a small tile subset.

### Post-processing
- Threshold at 0.5 (tune on validation tiles).
- Apply 2020 forest mask.
- Morphological opening (remove <0.5 ha specks) → matches the example submission stats.
- Write raster → `raster_to_geojson`.

**Deliverable:** `submission/` folder with a GeoJSON per test tile.

---

## Tier 2 — U-Net on Aggregated Features (day 1 evening → day 2, ~8–12h)

**Goal:** add spatial/neighborhood reasoning the GBM lacks.

### Architecture
- `segmentation_models_pytorch.Unet` with a `resnet34` or `efficientnet-b2` encoder.
- Input: a single stacked tile tensor, `C × H × W`, channels =
  - 64 AEF dims
  - Pre/post medians of NDVI, NBR, NDMI, EVI (8 channels)
  - Pre/post std of the same (8 channels)
  - Pre/post S1 VV median + std (4 channels)
  - 2020 forest mask (1 channel)
  - → ~85 channels total
- Output: 1-channel logits, sigmoid head.

### Training
- Loss: **BCE + Dice**, weighted by `label_confidence`.
- Augmentations (Albumentations): flips, 90° rotations, small elastic deforms, random brightness on S2 channels only.
- Patches: 256×256 random crops from 1002×1002 tiles.
- Optimizer: AdamW, lr 1e-4 with cosine schedule.
- Validation: hold out ~15% of training tiles (stratified by MGRS zone to avoid spatial leakage).

### Post-processing
Same as Tier 1 — threshold, forest mask, morphological cleanup, raster → GeoJSON.

**Key point:** train on the same features as Tier 1 so the comparison is clean — if U-Net doesn't beat LightGBM here, something is wrong in the training loop, not the data.

---

## Tier 3 — U-TAE + ViT-Style Cross-Attention Fusion (day 2 → day 3)

**Goal:** exploit the monthly temporal signal natively, fuse three modalities at the semantic level.

### Architecture

Three-branch encoder, cross-attention fusion at the bottleneck, shared U-Net decoder.

```
 S2 monthly (12 bands × T)  ──► U-TAE branch  ──►  F_s2 ∈ R^{d × h × w}
 S1 monthly (1  band  × T)  ──► U-TAE branch  ──►  F_s1 ∈ R^{d × h × w}
 AEF annual (64 dims)       ──► 2-conv stem   ──►  F_ae ∈ R^{d × h × w}

                                           ▼
                                   Cross-Attention Block
                                (Q=F_ae, K=V=concat(F_s2, F_s1))
                                           ▼
                                    Fused tokens  F'
                                           ▼
                           U-Net decoder (skips from S2 branch)
                                           ▼
                                  Binary logits (H × W)
```

### Component details

**U-TAE branches** — use the reference implementation [VSainteuf/utae-paps](https://github.com/VSainteuf/utae-paps). Import **only U-TAE**, drop PaPs (PaPs = panoptic head for instance segmentation of agricultural parcels, not needed for binary semantic task).

**Cross-attention block**
- Flatten each bottleneck feature map to tokens: `F ∈ R^{(h·w) × d}` (e.g. 32×32 = 1024 tokens at a 512×512 tile, d=128).
- Single multi-head cross-attention layer (4 heads, d=128).
- `Q = F_ae`, `K = V = concat([F_s2, F_s1], dim=0)` (2048 tokens).
- LayerNorm + residual + 1× MLP → `F' ∈ R^{(h·w) × d}`.
- Reshape back to `d × h × w`.

Why only one layer: most of the fusion gain comes from the first attention block; stacking more eats the training budget and overfits on weak labels. Ablate later if time permits.

**Decoder** — U-Net decoder initialised from the S2 branch (richest spatial detail). Skip connections come from the S2 U-TAE encoder levels.

### Training
- Warm-start: if PASTIS pretrained U-TAE weights exist and fit the encoder shape, load them. Otherwise train from scratch.
- Two-stage curriculum:
  1. **Pretrain** the S2 U-TAE branch alone as a segmenter (Tier 2-style) on NDVI-delta pseudo-labels for 3–5 epochs to get a decent init.
  2. **Fine-tune** the full 3-branch + cross-attention network end-to-end.
- Loss: same BCE + Dice, label-confidence weighted.
- Mixed precision (`torch.cuda.amp`); gradient accumulation if VRAM-bound.

### Why this beats Tier 2
- Monthly temporal resolution used natively, not pre-aggregated.
- Cross-modal fusion is learned, not concat.
- AEF's semantic prior conditions what monthly signal the model attends to.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Weak-label noise caps ceiling | Label fusion (§Cross-cutting 3), sample weighting, don't over-train |
| CRS misalignment → model learns pixel offset noise | Reproject all modalities to UTM before stacking; spot-check alignment visually |
| Tier 3 doesn't converge in hackathon window | Keep Tier 2 submission as fallback; only swap if Tier 3 validation F1 beats it |
| Overfitting to a few tiles | Stratify val split by MGRS zone; augment heavily; early-stop on val Dice |
| Clouds in S2 | Pre/post aggregation uses median (cloud-robust); S1 branch covers cloudy months |
| Submission format bugs eat time | Submit a trivial prediction (all zeros or Tier 1 output) on day 1 |

## Rough Timeline

- **Day 1 AM**: data loading utilities, CRS reprojection, weak-label fusion, forest mask.
- **Day 1 PM**: Tier 1 (LightGBM) end-to-end, first submission.
- **Day 2 AM**: Tier 2 (U-Net) training + submission.
- **Day 2 PM**: Tier 3 scaffolding — U-TAE branches, cross-attention block, forward pass sanity check.
- **Day 3 AM**: Tier 3 training + submission.
- **Day 3 PM**: ensembling (average Tier 1/2/3 probabilities), final post-processing tuning.

## References

- U-TAE / PaPs: https://github.com/VSainteuf/utae-paps
- AlphaEarth Foundations: https://arxiv.org/abs/2507.22291
- `segmentation_models_pytorch`: https://github.com/qubvel/segmentation_models.pytorch
- LightGBM: https://lightgbm.readthedocs.io
