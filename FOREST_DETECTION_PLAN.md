# Forest-Detection Workstream — Updated Plan

**Owner:** Lino · **Status:** P0 (baseline audit) ✅ COMPLETE  
**Clock:** 2026-04-18 16:45 → submission deadline 2026-04-19 11:00 (~18 h remaining)  
**Environment:** `.venv` with scipy, scikit-learn, lightgbm, matplotlib installed; Python code via system Python (`c:/...python.exe`)

---

## P0 Baseline Audit — COMPLETE ✅

**Key findings:**
- **16 cached tiles** (up from 10 mentioned in README): 18N zone ×4, 48P zone ×4, 48Q zone ×2
- **Baseline masker reproduction: PERFECT** — recall 0.9824, coverage 0.7749 (README 0.982/0.775)
- **All sanity checks pass:** no missing keys, no empty-GT tiles, no dequantization failures, AEF has no all-zero pixel groups
- **Tile size variance:** mostly ~1000×1000, with 8 unique shapes across 16 tiles (no canonical grid issue)
- **Representative tile for analysis:** 48PWV_7_8 (45.90% forest_gt, strongest signal for visualizations)

**Output:** `code/reports/P0_BASELINE_AUDIT.txt` (this file records the audit console log)

**What this means:** The baseline is robust, well-calibrated, and a solid foundation. No regression risk; any improvement is pure upside.

---

## Why we built P0 first

- **Guardrail against silent regressions:** if we add paper-driven AEF dims or reweight loss and the aggregate recall drops to 0.97, we'd know immediately
- **Data quality confidence:** verified no missing keys, no NaN explosions, no CRS/tile-size surprises
- **Per-tile insight:** we now know which tiles are strongest (high GT%, highest signal) vs weakest (edge cases for error analysis)

---

## Revised work sequence for P1–P3

Given that:
1. Baseline is already **very strong** (recall 0.982 is near-ceiling for weak labels)
2. We have ~18 h left
3. The team is handling Tier 1 deforestation submission in parallel

### Focus: **Understand where the mask fails, then optionally improve**

**High-value but lower-risk order:**

1. **P3: Error analysis** (~1.5–2 h) — identify the 10–20% of `forest_gt_pre2020` pixels the masker misses
   - Characterize failure modes by NDVI band, forest edge vs core, connected-component size, MGRS zone
   - Produces: one actionable summary document + 1–2 diagnostic visualizations (tile overlays showing FN/FP)
   - Decision gate: if failures are random noise (unavoidable alert-label error), P1 won't help. If patterned (e.g., "all degraded forest NDVI 0.4–0.5"), P1 might.

2. **P1: Paper-driven AEF dims** (~1–2 h, **conditional on P3 insights**) — add specialist dims A18/21/26 + shared A28/34
   - Only if P3 shows that AEF should be able to discriminate the failure cases (e.g., "S2 NDVI is ambiguous, but AEF tree-dims should separate real forest from crops")
   - Retrain LearnedForestMasker with new features
   - Compare aggregate recall + per-tile recall on the weak tiles identified in P3
   - Ship only if it beats baseline; revert if flat/worse

3. **P2: Post-period forest probability** (~1 h, **optional**) — feed soft prefilter into deforestation model
   - Only if P1 succeeds and there's time
   - Lower priority because the team is handling the deforestation model; this is a nice-to-have conditioning signal

### P3 implementation notes

**Error analysis script** (quick, no matplotlib PCA)
- Per-tile: recall sorted ascending → identify bottom 3 tiles
- For each bottom tile: sample misclassified pixels (`forest_gt == 1, pred == 0`)
- Slice by:
  - `s2_pre_ndvi_median` bins: 0.3–0.4, 0.4–0.5, 0.5–0.6, 0.6+
  - Distance to forest edge (morphological erosion)
  - Connected-component size (how many pixels are the miss regions?)
  - MGRS zone (geographic clustering?)
- For each slice: count pixels, compute mean AEF tree-dim values (A18/21/26)
- **Key question:** "Can AEF tree-dims separate the false-negatives from the true positives in this slice?"

**P3 output:** markdown table + one tile overlay figure (GT vs prediction, errors in red/yellow)

### P1 implementation notes (if P3 says yes)

**Feature engineering in** [vegetationPredictor.py](code/model/vegetationPredictor.py)
- Modify `_LEARNED_FEATURE_KEYS` tuple: add 5 dims (A18, A21, A26, A28, A34) as individual channels
- Modify `_per_pixel_features()`: extract these dims explicitly from `aef_pre`, keep existing mean/std as high-generalist baseline
- Retrain `LearnedForestMasker` on all 16 cached tiles
- Eval: print per-tile recall sorted by ID, highlight the weak tiles from P3
- Decision: if recall ≥0.983 and per-tile min-recall improves (e.g., weak tile 48PUT_0_8 goes from X% to >90%), ship. Else revert.

### P2 implementation notes (if P1 ships + time allows)

**Post-period masker as soft feature**
- Train `LearnedForestMasker` on `aef_post/s2_post/s1_post` with positives = pre-period STRONG ∧ no-alert (truly-still-forest)
- Cache `forest_prob_post` per tile
- For deforestation team: add `forest_prob_delta = pre − post` as a feature channel
- Expected signal: deforested pixels have high positive delta (was forest, no longer detected as forest by post-period features)

---

## Deliverables by priority

| Priority | Deliverable | Owner | Status | Notes |
|----------|-------------|-------|--------|-------|
| ✅ DONE | P0 audit report + baseline verification | Lino | COMPLETE | `P0_BASELINE_AUDIT.txt` in `code/reports/` |
| HIGH | P3 error analysis report + viz | Lino | PENDING | Decision-gating feature; start now |
| MEDIUM | P1 paper AEF dims (conditional) | Lino | PENDING | Only if P3 shows signal in A18/21/26/28/34 |
| LOW | P2 post-period masker (conditional) | Lino | PENDING | Nice-to-have for deforestation team |
| HANDOFF | README §3 update + checklist | Lino | PENDING | Final 30 min before submission |

---

## Handoff contract (unchanged)

- `LearnedForestMasker` API immutable: `predict_proba()`, `predict_tiers()`, tier thresholds (STRONG=1.0, SOFT=0.5, etc.)
- Any new channels added (e.g., `forest_prob_pre/post/delta`) must be documented with shape/dtype/ranges
- Cache should be extended, not replaced (existing keys stay at same values)

---

## Status log

- `2026-04-18 16:45` — P0 audit complete. Baseline perfect (0.9824/0.7749). 16 tiles, 3 zones. Ready for P3.
