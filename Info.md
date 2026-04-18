# Sensor & Data Source Reference — osapiens Challenge 2026

## Sentinel-1 (S1) — C-band Synthetic Aperture Radar

**What it measures.** Active microwave sensor. Emits C-band pulses at ~5.4 GHz (roughly 5.6 cm wavelength) and measures the strength of the signal that bounces back. Your data is VV polarisation — pulse sent and received both in the vertical orientation. Values are in dB, typically -25 to +5 for land surfaces, with vegetation clustering around -5 to -10.

**What makes it see what it sees.** Microwaves at 5 cm penetrate cloud, rain, and smoke — S1 works day or night, dry or monsoon. They partly penetrate canopy too: backscatter comes from volume scattering inside the crown (branches, leaves, trunks of similar size to the wavelength) rather than just the surface. Flat, smooth water returns near zero (pulses bounce away from the sensor → very negative dB). Rough surfaces scatter diffusely, returning more. Buildings and corner reflectors can return *more* energy than was emitted (>0 dB).

**What makes it hard.**

- *Speckle* — coherent radar returns interfere constructively and destructively, producing a salt-and-pepper noise pattern that never goes away, only gets averaged down by temporal/spatial pooling.
- *Geometric distortion* — side-looking geometry causes foreshortening on slopes facing the sensor and layover on steep terrain. Your data is Radiometrically Terrain Corrected (RTC), which removes most of this, but residuals on very steep ground still exist.
- *Moisture confusion* — wet soil, flooded forest, and freshly irrigated cropland all raise backscatter, creating patterns that can mimic vegetation structure changes. This is a major FP source for RADD deforestation alerts.

**Ascending vs descending.** Same ground, different look direction:

- *Ascending* — satellite moving S→N, illuminates from the west
- *Descending* — satellite moving N→S, illuminates from the east

A slope, a row of plantation trees, or the ridgeline of a building looks different under the two orbits. Primary forest is relatively isotropic (volume scattering from a structurally chaotic canopy → asc ≈ desc). Plantations, row crops, and topographic anisotropy produce measurable asc-desc differences. **This is why `s1_pre_vv_orbit_diff` is signal and not noise.** Coverage isn't guaranteed equal — in tropical tiles you can easily have ~99% ascending coverage and only ~80% descending, as your `47QMB_0_8` test showed -> How much of the tile area is covered by the flying over satellite.

**Cadence.** Monthly composites in this challenge. Native revisit is 6-12 days depending on latitude and mission status (Sentinel-1B failed in December 2021, so post-2021 cadence is halved over many regions until 1C launched).

**Projection.** Local UTM, aligned with S2 at 10m resolution.

---

## Sentinel-2 (S2) — Multispectral Optical

**What it measures.** Passive optical sensor, 12 spectral bands from visible through shortwave infrared. Measures reflected sunlight, so it only works during daytime passes with clear skies. Values are surface reflectance 0–1 after L2A atmospheric correction.

### The bands and what they're actually for

| Band | Wavelength | Resolution | Primary use |
|------|-----------|------------|-------------|
| B01 | 443 nm | 60 m | Aerosol detection |
| B02 | Blue 490 nm | 10 m | Water, haze, soil/vegetation contrast |
| B03 | Green 560 nm | 10 m | Peak vegetation reflectance in visible |
| B04 | Red 665 nm | 10 m | Chlorophyll absorption — low for healthy plants |
| B05–B07 | Red-edge 705–783 nm | 20 m | Vegetation classification, chlorophyll content |
| B08 | NIR 842 nm | 10 m | Biomass — healthy vegetation reflects strongly |
| B8A | Narrow NIR 865 nm | 20 m | Refined vegetation signal, less water-vapour contamination |
| B09 | 945 nm | 60 m | Water vapour detection |
| B10 | 1375 nm | 60 m | Cirrus cloud detection |
| B11 | SWIR 1610 nm | 20 m | Moisture, burnt area, snow/cloud |
| B12 | SWIR 2190 nm | 20 m | Moisture, burnt area, mineralogy |

Your data is upsampled to 10 m across all bands.

### Derived indices in your pipeline

- **NDVI** = (NIR − Red) / (NIR + Red). Healthy vegetation reflects NIR and absorbs red → high NDVI (0.6–0.9 for forest). Bare soil, water, urban → low (< 0.3).
- **NBR** = (NIR − SWIR2) / (NIR + SWIR2). Normalized Burn Ratio. Intact canopy has high NIR and low SWIR2; fire or clearing flips both → NBR drops sharply. More sensitive to deforestation than NDVI because plantation and pasture can maintain high NDVI while having a different SWIR signature from primary forest.
- **NDMI** = (NIR − SWIR1) / (NIR + SWIR1). Vegetation water content. Drops during drought, clearing, or senescence.
- **EVI** = soil/atmosphere-corrected NDVI variant that saturates less in dense canopy.

### The cloud problem — why rejecting scenes is necessary

Clouds and their shadows contaminate S2 in multiple ways that all look like "change" to a naive model:

- Thick cumulus → pixel values jump toward white (bright, high reflectance across visible/NIR) — looks nothing like forest.
- Thin cirrus → partial contamination, subtly lifts reflectance across bands without being obviously cloudy.
- Cloud shadows → reflectance drops sharply — can look like burn scars or clearing to NBR.
- Cloud edges and haze → smooth gradient contamination over many pixels.

In tropical biomes (your data), multi-month stretches with near-total cloud cover are normal. A single cloudy scene can easily create a multi-dB apparent change that looks exactly like deforestation to a per-scene classifier. This is why:

- Providers compose monthly "best scene" products rather than exposing every pass — you get the least-cloudy observation of each month rather than all of them.
- Your pipeline computes **medians over pre-2020 and post-2020 windows** rather than using single scenes — one cloudy month in 36 gets overwhelmed by the rest.
- Your `compute_ndvi_drop` uses a **k-month rolling median** before the argmax rather than picking a single worst month — clouds don't persist across consecutive months, deforestation does.
- Scenes that fail QA, have too much cloud cover, or produce `sum of bands == 0` are **rejected outright** rather than patched — better to have a gap than fabricated data. Valid-pixel masks are tracked alongside data.

The consequence for modeling: S2 is the richest sensor you have when it works, but coverage is unreliable, and any model that can't handle "data missing this month" for arbitrary months will fail in the tropics. This is also why fusion with S1 matters — S1 is always available, even when S2 is blind.

**Cadence.** Native revisit 5 days (with S2A + S2B); monthly composites in the challenge.

**Projection.** Local UTM, 10 m.

---

## AlphaEarth Foundations (AEF) — learned global embeddings

**What it is.** Not raw sensor data. These are 64-dimensional per-pixel embeddings produced by a foundation model (Google, 2025) trained on a mix of Sentinel-1, Sentinel-2, and other global EO sources. Each pixel is a dense feature vector that captures spatio-temporal patterns — "what this location looks like across all modalities over the year" compressed into 64 numbers.

### What makes it different from S1/S2

- *Temporal cadence*: **annual**, not monthly. One embedding per pixel per year — the within-year dynamics have already been captured by the model and baked into the embedding.
- *Projection*: **EPSG:4326 (WGS84 geodetic lat/lon)**, not local UTM. Pixels are not square on the ground — they're shorter E–W at higher latitudes. For your tropical tiles the distortion is modest (~1% at the equator) but non-zero. Must be reprojected to UTM to align with S1/S2 before fusion, which your `_reproject_to` path handles.
- *No physical units*: the 64 dimensions have no individual semantic meaning. They're a learned basis. Looking at any single dim by itself is not informative; the information is in the relative structure of the whole vector.
- *Sparsity of useful signal*: research suggests the land-cover-relevant signal is concentrated in a small subset of dims, with many dims encoding orthogonal things (spectral, temporal, geolocation). A tree model can pick which dims matter; a linear projection via PCA would lose the sparse structure.

### What makes it valuable

- Pretrained on vastly more data than you could ever label, so it encodes "what a tropical forest canopy looks like" globally rather than learning it from a few hundred tiles.
- Already does the heavy lifting of fusing multi-temporal optical + radar — you get a joint representation for free, without needing to design the fusion architecture yourself.
- Robust to missing observations in any single sensor/month because the embedding was trained to summarise the year as a whole.

### What limits it

- Coarser temporal resolution means it's less useful for pinpointing *when* something happened — your NDVI-drop timing feature is finer-grained than anything AEF provides.
- Being a learned representation, it will have biases from its training distribution. A land-cover class that was rare in pretraining may not be well-separated in embedding space. For the tropical-deforestation use case this is mostly fine, but worth keeping in mind when interpreting "distance to forest centroid" maps — a weird measurement might mean your pixel is unusual, not that it's non-forest.

**How it sits in the pipeline.** Reprojected to UTM at 10 m to match S1/S2, stored as `aef_pre` with shape `(64, H, W)`, consumed per-pixel alongside the engineered S1/S2 features.

---

## Quick comparison

| | S1 | S2 | AEF |
|---|---|---|---|
| Type | Active radar | Passive optical | Learned embedding |
| Weather dependence | None | Severe (clouds) | Inherited from sources, mostly abstracted away |
| Native cadence | 6–12 days | 5 days | Annual |
| Challenge cadence | Monthly | Monthly | Annual |
| Spatial resolution | 10 m (upsampled) | 10 m (upsampled) | ~10 m after reprojection |
| Projection | Local UTM | Local UTM | EPSG:4326 → reproject to UTM |
| Interpretable bands | Single (VV) × asc/desc | 12 spectral | None — 64-D learned basis |
| Primary failure mode | Speckle, moisture-induced FPs, terrain on steep slopes | Cloud/shadow contamination, unreliable coverage in tropics | Domain bias from pretraining distribution |
| Role in fusion | Cloud-robust backbone — always present | High-information when valid — best single-sensor forest signal | Global prior — compresses years of multi-sensor signal into a dense feature |

---

## Labels (for completeness)

Not sensor data, but worth stating their nature: **RADD, GLAD-L, and GLAD-S2 are themselves predictions** from other deforestation detectors, not verified ground truth. Your `build_forest_ground_truth` exploits a one-sided property — if any detector claims post-2020 deforestation at a pixel, that pixel must have been forest pre-2020 — to extract clean positives for the pre-2020 mask. FPs in the alert sources (radar moisture artefacts in RADD, cloud shadows in GLAD-S2, etc.) can leak into those "positives" unless filtered by confidence and spectral plausibility, which is what the Phase 1 spectral gate handles.

---

## Cache contents — what `preprocess_tile` produces

Each tile's `.npz` cache holds the tensors below. Train tiles have 49 keys, test tiles 46 (no labels), plus 4 more if `augment_cache_with_ndvi_drop` has been run. Everything is `float32` / `(H, W)` unless noted. `(H, W)` is set by the largest S2 scene in the tile (typically ~1004×1004 ≈ 1 M pixels). Pre-cutoff windows use 2019–2020; post-cutoff uses 2021–2024.

### AlphaEarth embeddings — the global prior

- **`aef_pre`** `(64, H, W)` — median of annual embeddings 2019–2020. A learned summary of "what this pixel looked like to the foundation model" before any deforestation is allowed to happen.
- **`aef_post`** `(64, H, W)` — median across 2021–2024.
- **`aef_delta`** `(64, H, W)` — `aef_post − aef_pre`. The "how did the learned representation shift" channel. In principle carries change signal without hand-engineered differencing.

### Sentinel-2 reflectance composites

- **`s2_{pre,post}_band_median`** `(12, H, W)` — per-band median across monthly composites. Median over a multi-year window is the primary cloud-robustness mechanism: a few cloudy months can't move it if most of the window is clear.
- **`s2_{pre,post}_band_std`** `(12, H, W)` — per-band std over the same window. High std ⇒ unstable land cover (seasonal crops, cycling through harvest states). Low std ⇒ stable (primary forest, persistent bare, water).

### Sentinel-2 vegetation indices

All of these exist in both median and std flavours, pre and post, plus a delta:

- **`s2_{pre,post}_ndvi_{median,std}`** — vegetation density. Forest 0.6–0.9, bare/urban/water < 0.3.
- **`s2_{pre,post}_nbr_{median,std}`** — Normalized Burn Ratio. More sensitive to canopy *type* than NDVI — primary forest and plantation can match on NDVI but separate on NBR.
- **`s2_{pre,post}_ndmi_{median,std}`** — canopy water content. Drops with drought and clearing.
- **`s2_{pre,post}_evi_{median,std}`** — EVI, saturates less than NDVI in dense canopy.
- **`s2_delta_{ndvi,nbr,ndmi,evi}`** — post-median minus pre-median. The cheapest possible "where did greenness fall" feature. A deforested pixel shows a strongly negative NDVI delta.

### Sentinel-1 VV backscatter (dB)

All in dB (`10·log10(linear)`). Missing observations become zeros after the composite reduction, *not* NaNs — LightGBM handles the sentinel via companion keys (the matching `_std` going to zero tells the model "this is a missing-data pixel, not a real zero-backscatter surface").

- **`s1_{pre,post}_vv_{median,std}`** — combined asc+desc. Kept for backward compatibility; existing code paths that refer to these untouched keys still work.
- **`s1_{pre,post}_vv_asc_{median,std}`** — ascending orbit only (satellite S→N, looking W). Added Phase 0 after verifying the old pipeline silently averaged the two directions.
- **`s1_{pre,post}_vv_desc_{median,std}`** — descending orbit only (N→S, looking E). Coverage may be partial — many tiles have ~99% asc coverage and only ~80% desc because the orbit track doesn't pass over the tile's full extent.
- **`s1_delta_vv`, `s1_delta_vv_asc`, `s1_delta_vv_desc`** — post median minus pre median, grouped accordingly. Sharp drop ⇒ canopy volume lost.
- **`s1_{pre,post}_vv_orbit_diff`** — `asc_median − desc_median`. Primary forest is volume-scattering and roughly isotropic, so asc ≈ desc ⇒ `orbit_diff` is near zero. Plantations, row crops, and anisotropic topography scatter directionally, producing a non-trivial `orbit_diff`. **This is the forest-vs-plantation discriminator.** Caveat: where one orbit is absent the diff is just the other orbit's value — LightGBM can disambiguate via the zero in the missing companion.

### Derived forest mask

- **`forest_mask_2020`** `(H, W) uint8` — simple NDVI ≥ 0.6 threshold on the pre-cutoff composite. Coarse starting point, not the final word. `LearnedForestMasker` (and later MAESTRO-based refinement) replaces this as the canonical pre-2020 forest layer for downstream gating.

### Weak labels — train tiles only

- **`label`** `(H, W) uint8` — consensus deforestation positive. A pixel is 1 iff ≥ 2 of {RADD, GLAD-L, GLAD-S2} flagged a post-2021-01-01 alert **AND** `forest_mask_2020[pixel] == 1`. The forest-mask gate is structurally required: without it you score positives on pixels that were never forest, and everything downstream bleeds.
- **`label_confidence`** `(H, W) float32` — mean of contributing sources' per-source confidence (RADD leading digit, GLAD-L flag strength, GLAD-S2 code/4), gated by the same mask. Use as sample weights when training the deforestation model — not every positive is equally trustworthy.
- **`forest_gt_pre2020`** `(H, W) uint8` — **evaluation-only, never feed to training.** Union of *any* post-2020 alert at *any* confidence. If any detector ever flagged the pixel, that pixel must have been forest pre-2020 (one-sided guarantee: FPs in alerts create spurious "forest", but any *real* alert is a real pre-2020 forest witness). This is how the pre-2020 forest mask gets measured without hand-labeled ground truth.

### NDVI-drop timing (only after `augment_cache_with_ndvi_drop`)

Median composites collapse an NDVI trajectory into a scalar — they tell you *how much* NDVI fell, not *when*. These restore the timing:

- **`ndvi_drop_magnitude`** `(H, W) float32` — baseline NDVI (pre-cutoff median) minus the smallest smoothed NDVI in the post-cutoff window. Smoothing is a 3-month rolling nan-median, which kills isolated cloud-shadow minima — a drop has to persist across ≥ 2 months to win. Pixels that never dropped report 0.
- **`drop_doy`** `(H, W) int16` — day-of-year (1–366) of the peak-drop month; 0 = no drop.
- **`drop_year`** `(H, W) int16` — year of peak drop; 0 = no drop.
- **`drop_month_idx`** `(H, W) int16` — index into the tile's post-cutoff scene list; −1 = no drop.

Timing separates abrupt clearing from gradual phenological decline — information a median composite throws away entirely.

### Meta

- **`_shape`** `(2,) int32` — `[H, W]` of the reference grid. Convenience for code that wants the canonical dimensions without loading any heavy tensor.

### Channel budget at feature stacking

`DEFAULT_FEATURE_KEYS` yields **260 channels** before Phase 2/3 additions:

| Group | Channels |
| --- | --- |
| AEF (pre + post + delta) | 192 |
| S2 band median (pre + post) | 24 |
| S2 band std (pre + post) | 24 |
| S2 delta indices (4) | 4 |
| S2 pre/post index medians (ndvi + nbr × 2) | 4 |
| S1 combined (pre/post/delta) | 3 |
| S1 ascending (pre/post/delta) | 3 |
| S1 descending (pre/post/delta) | 3 |
| S1 orbit_diff (pre + post) | 2 |
| `forest_mask_2020` | 1 |
| **Total** | **260** |

Phase 2 will add NDVI temporal trajectory (min/max/p10/p90/slope/n_valid), NBR/NDMI trajectories, and 9×9 / 21×21 neighborhood medians of NDVI and NBR — expect the budget to grow to ~290. Phase 3 MAESTRO adds one `maestro_emb` block of D channels (D = 512 or 768 depending on checkpoint) plus a scalar `forest_centroid_distance`, pushing it past 800 total before any reduction.
