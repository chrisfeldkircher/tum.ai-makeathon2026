# Downstream Integration Plan

**Branch:** `feat/downstreaming-forest-integration` (on top of `origin/feat/downstreaming @ 9a6c988`)
**Status:** implemented + pushed; **not yet runtime-validated** (cache needs regeneration).

---

## What is integrated

- `cache_probs/<tile_id>.npz` sidecar loading in `code/data/data.py`
- Feature stack extended: `CORE_FEATURE_KEYS` + `PROB_FEATURE_KEYS` → `DEFAULT_FEATURE_KEYS = 277 channels` (was 274)
  - new channels: `forest_prob_pre`, `forest_prob_post`, `forest_prob_delta`
- `code/model/vegetationPredictor.py` merged: downstream richer feature set preserved + AEF tree dims A18/21/26/28/34 reintroduced
- `code/reports/p2_post_period.py` + `P2_INTEGRATION.md` present on branch

---

## Pending fixes (ordered by priority)

### 1. U-Net decoder call — CRITICAL BUG

In `code/model/train.py`, change:
```python
self.unet.decoder(features)   # wrong — unpacks encoder features incorrectly
```
to:
```python
self.unet.decoder(*features)
```

### 2. Regenerate caches

```bash
# Regenerate tile cache with merged downstream preprocessing
python code/data/data.py --cache_dir ./cache

# Regenerate probability sidecars (forest-detection P2)
python code/reports/p2_post_period.py --cache_dir ./cache --out_dir ./cache_probs
```

### 3. Wire forest-probability gate at inference

Replace coarse legacy gate with learned probability:
```python
# preferred (use forest_prob_pre from P2 sidecars)
predictions[forest_prob_pre < 0.30] = 0   # NON_FOREST threshold

# fallback if sidecars unavailable
predictions[forest_mask_2020 == 0] = 0
```

### 4. Save CRS + transform to npz cache — REQUIRED FOR SUBMISSION

`raster_to_geojson()` (in `problem/submission_utils.py`) needs a GeoTIFF with valid CRS.
Currently **the npz cache stores only `_shape`** — no CRS or transform is saved.

**Fix**: in `code/data/data.py`, extend `cache_tile()` / `preprocess_tile()` to save the reference grid info:

```python
# At end of preprocess_tile(), add:
out["_crs_wkt"]       = np.bytes_(ref.crs.to_wkt())
out["_transform"]     = np.array(ref.transform[:6], dtype=np.float64)  # (a,b,c,d,e,f)
```

Then in the export step, reconstruct with `rasterio.transform.Affine(*transform)` and `rasterio.crs.CRS.from_wkt(crs_wkt.decode())`.

### 5. Submission export script

Once CRS/transform are cached, the export pipeline is:

```python
# pseudo-code — wire into code/submission/export.py

from rasterio import MemoryFile
from rasterio.crs import CRS
from rasterio.transform import Affine
from problem.submission_utils import raster_to_geojson

for tile_id, pred_mask in predictions.items():   # pred_mask: (H,W) uint8 0/1
    npz = np.load(f"cache/{tile_id}.npz")
    crs = CRS.from_wkt(npz["_crs_wkt"].decode())
    transform = Affine(*npz["_transform"])
    H, W = pred_mask.shape

    with MemoryFile() as mf:
        with mf.open(driver="GTiff", height=H, width=W, count=1,
                     dtype="uint8", crs=crs, transform=transform) as dst:
            dst.write(pred_mask[None])
        geojson = raster_to_geojson(mf.name, min_area_ha=0.5)

# Merge all tiles into one FeatureCollection and write
all_features = [f for g in geojson_per_tile.values() for f in g["features"]]
submission = {"type": "FeatureCollection", "features": all_features}
with open("submission/submission.geojson", "w") as f:
    json.dump(submission, f)
```

`time_step` (YYMM) can be added per-feature if your model predicts deforestation timing.

---

## Runtime validation checklist

- [ ] `uv sync` / pip install all deps
- [ ] Fix U-Net decoder bug (`*features`)
- [ ] Regenerate `cache/` with merged preprocessing
- [ ] Regenerate `cache_probs/` with p2_post_period.py
- [ ] Add `_crs_wkt` + `_transform` to cache (or read from raw data at export time)
- [ ] Dataloader smoke test: 277-channel forward pass
- [ ] One training step without NaN
- [ ] Wire `forest_prob_pre >= 0.3` gate
- [ ] End-to-end inference → GeoJSON export → validate with `submission_details.md` rules
- [ ] Upload test submission to leaderboard

---

## Deferred (out of scope for baseline)

- Temporal DANN
- `TEMPORAL_FEATURE_KEYS` with probability channels
- Yearly `forest_prob_post` per year (flag for post-hackathon — would need per-year AEF/S2 cache)
