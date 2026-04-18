# Downstream Integration Plan

**Main branch:** `feat/tier-0-foundation` (latest: `3dcb338`)

---

## Status of all integration items

| Item | Status | Where |
|------|--------|-------|
| Forest prob sidecar loading (`cache_probs/`) | ✅ merged (`d9df263`) | `data.py: load_probability_sidecar`, `DeforestationPatchDataset` |
| 277-channel feature stack (`CORE + PROB`) | ✅ merged (`d9df263`) | `data.py: CORE_FEATURE_KEYS + PROB_FEATURE_KEYS` |
| AEF tree dims A18/21/26/28/34 | ✅ merged (`d9df263`) | `vegetationPredictor.py: _AEF_TREE_DIMS` |
| U-Net decoder bug (`*features`) | ✅ fixed (`3dcb338`) | `train.py:112` |
| Stale-cache auto-regen | ✅ merged (`d9df263`) | `data.py: cache_missing_keys` |
| Save `_crs_wkt` + `_transform` to cache | ✅ this branch | `data.py: preprocess_tile()` |
| Forest gate at inference (`forest_prob_pre >= 0.3`) | ✅ this branch | `code/submission/export.py` |
| Submission export script | ✅ this branch | `code/submission/export.py` |

---

## Running the submission export (no cache regen needed)

`export.py` resolves CRS in two ways — whichever is available:
1. `_crs_wkt` / `_transform` from `cache/<tile_id>.npz` *(present after next cache build)*
2. First raw Sentinel-2 GeoTIFF in `data/sentinel-2/{split}/{tile_id}__s2_l2a/` *(fallback, works today)*

```bash
python code/submission/export.py \
    --preds_dir ./predictions \       # per-tile .npz with 'pred' (H,W) uint8
    --cache_dir ./cache \             # for _crs_wkt/_transform if present
    --data_dir  ./data \              # raw S2 fallback for CRS
    --probs_dir ./cache_probs \       # forest gate (omit with --no_gate to skip)
    --out       submission/submission.geojson
```

Input format for `predictions/<tile_id>.npz`:
- `pred`: `(H, W)` uint8, `1` = deforestation
- `time_step`: scalar int YYMM (e.g. `2204` for April 2022) — optional

Forest gate: predictions on pixels with `forest_prob_pre < 0.30` are zeroed before vectorising. Requires `cache_probs/` (built by `code/reports/p2_post_period.py`). Use `--no_gate` to bypass.

---

## Remaining before submission

- [ ] Regenerate `cache/` — picks up `_crs_wkt`/`_transform` (not blocking for export, which falls back to raw S2)
- [ ] Regenerate `cache_probs/` with `python code/reports/p2_post_period.py --cache_dir ./cache`
- [ ] Train model, run inference → save per-tile `predictions/<tile_id>.npz`
- [ ] Run `code/submission/export.py` → upload `submission/submission.geojson`
- [ ] Validate: file extension `.geojson`, top-level `FeatureCollection`, only Polygon/MultiPolygon, `time_step` in YYMM if present

---

## Deferred

- Temporal DANN
- Yearly `forest_prob_post` per year (flag for post-hackathon)
