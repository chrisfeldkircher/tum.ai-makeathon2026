# Forest-Detection Workstream — Status

**Owner:** Lino · **Deadline:** 2026-04-19 11:00
**Environment:** system Python (`c:/...python.exe`) — all deps installed system-wide.

---

## Completed work

| Step | Result | Output |
|------|--------|--------|
| **P0** Baseline audit | Reproduced README 0.982/0.775 exactly; 16 tiles, 3 zones, all sanity checks pass | `code/reports/P0_BASELINE_AUDIT.txt`, `code/reports/data_stats.md` |
| **P3** Error analysis | FN concentrated in NDVI 0.3–0.5 (81% miss rate); edges 7–9× worse than core; AEF tree-dims separate FN/TP up to 3.22σ → clear signal to add dims | `code/reports/P3_ERROR_ANALYSIS.md`, `code/reports/figs/p3_error_overlay_*.png` |
| **P1** Paper AEF dims | Added A18/21/26/28/34 as explicit channels (`include_aef_tree_dims=True`, default on). Recall **0.9824 → 0.9840**; worst tile 18NWH_1_4 +2.4%, 48PUT_0_8 +0.8%; no regressions | `code/reports/P1_AEF_DIMS.md` |
| **P2** Post-period masker | Trained with truly-still-forest positives (`forest_gt_pre2020 & label==0`). 3 new rasters per tile in `cache_probs/`. Signal: deforested pixels mean delta +0.151 vs +0.058 stable (0.43σ, 3× ratio) — weak-but-real, use as soft feature | `code/reports/P2_INTEGRATION.md`, `cache_probs/<tile_id>.npz` |

**Current model baseline** (P1-shipped config): aggregate recall **0.9840**, coverage **0.7717**, min tile-recall **0.9422**.

---

## Remaining / optional

| Step | Priority | Notes |
|------|----------|-------|
| **P4** Edge-weighted retraining | MEDIUM | P3 showed edges 7–9× more FN-prone than core — universal. Approach: precompute edge distance, add `sample_weight = 1 + alpha*(dist<=2)` to LightGBM. Ship only if min tile-recall improves AND coverage stays within ±0.01. Est ~1h. |
| README §3 update | LOW | Update recall/coverage numbers; point to integration doc. |

---

## Handoff contract

- **`LearnedForestMasker` API is frozen**: `predict_proba()`, `predict_tiers()`, tier thresholds (STRONG=0.85, SOFT=0.60, NON_FOREST=0.30) — unchanged.
- **P1 change**: `include_aef_tree_dims=True` is the new default. The old behaviour (`include_aef_tree_dims=False`) is still accessible for ablation.
- **P2 channels** (`cache_probs/<tile_id>.npz`):
  - `forest_prob_pre`: `(H, W)` float32 `[0, 1]` — pre-period P(forest)
  - `forest_prob_post`: `(H, W)` float32 `[0, 1]` — post-period P(still-forest)
  - `forest_prob_delta`: `(H, W)` float32 `[-1, 1]` — pre − post (positive = lost forest-like appearance)
  - Integration instructions: [`code/reports/P2_INTEGRATION.md`](code/reports/P2_INTEGRATION.md)
- Cache (`cache/`) is **not mutated** — P2 writes to a separate `cache_probs/` directory.

---

## Status log

- `2026-04-18 16:45` — P0 complete. Baseline 0.9824/0.7749. 16 tiles.
- `2026-04-18 17:30` — P3 complete. NDVI 0.3–0.5 dominant failure; AEF tree-dims show signal. GO for P1.
- `2026-04-18 17:55` — P1 shipped. Recall 0.9824 → 0.9840, no regressions.
- `2026-04-18 18:40` — P2 shipped. Post-period masker + delta channel cached. Handoff doc written.
