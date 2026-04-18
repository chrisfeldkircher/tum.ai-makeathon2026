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

### P2 implementation notes — NEXT UP

**Post-period masker as soft feature** (direct handoff to deforestation team)
- Train `LearnedForestMasker` on `aef_post/s2_post/s1_post` with positives = pre-period STRONG ∧ no-alert (truly-still-forest)
- Cache `forest_prob_pre` (from P1-upgraded pre-period masker), `forest_prob_post`, `forest_prob_delta = pre − post` per tile
- Sanity: on held-out tile, `delta > 0.3` should correlate strongly with `label == 1`. If signal is flat, the channel is dead — ship only if signal exists.
- Handoff: document channel names, shape `(H, W)` float32, value range `[-1, 1]` for delta.

### P4 implementation notes — edge-weighted retraining

**Why:** P3 showed that edge pixels (dist ≤ 2 from non-forest boundary) have FN rates 7–9× higher than core pixels — universal across all tiles. P1 narrowed the NDVI-band gap but didn't target edge geometry directly.

**Approach (cheap, ~1 h):**
- Precompute per-pixel edge-distance on `forest_gt_pre2020` using morphological distance transform
- Assign `sample_weight = 1.0 + alpha * (dist <= 2)` (try alpha ∈ {1, 2, 3})
- Pass through LightGBM `fit(..., sample_weight=w)`
- A/B vs P1-shipped config. Decision rule: ship only if min per-tile recall strictly improves AND aggregate coverage stays within ±0.01 of baseline (no precision bloat).
- If coverage balloons (model learned "call everything forest near edges"), revert.

---

## Deliverables by priority (post-P1, revised 2026-04-18 18:10)

| Priority | Deliverable | Owner | Status | Notes |
|----------|-------------|-------|--------|-------|
| ✅ DONE | P0 audit report + baseline verification | Lino | COMPLETE | `P0_BASELINE_AUDIT.txt` |
| ✅ DONE | P3 error analysis report + viz | Lino | COMPLETE | `P3_ERROR_ANALYSIS.md`, `figs/p3_error_overlay_*.png` |
| ✅ DONE | P1 paper AEF dims (A18/21/26/28/34) | Lino | COMPLETE | Shipped; min tile-recall +0.8%, best tile +2.4%. `P1_AEF_DIMS.md` |
| **HIGH** | **P2 post-period forest_prob + delta channel** | Lino | **NEXT** | Direct handoff value to deforestation team. Cache `forest_prob_pre/post/delta` per tile. |
| **MEDIUM** | **P4 edge-weighted retraining** | Lino | PENDING | Targets universal 7–9× edge-FN pattern. Cheap A/B: precompute edge distance → LightGBM `sample_weight`. Ship only if min tile-recall improves without coverage bloat. |
| HANDOFF | README §3 update + checklist | Lino | PENDING | Final 30 min before submission |
| (deferred) | P1b water/non-forest precision audit | — | SKIP unless time | P3 showed no water-vs-forest confusion; speculative. |
| (skipped) | Morphological dilation of predictions | — | SKIP | Would inflate FP where coverage is already high. |
| (skipped) | MLP backend ablation (P4 old) | — | SKIP | Diminishing returns post-P1. |

---

## Handoff contract (unchanged)

- `LearnedForestMasker` API immutable: `predict_proba()`, `predict_tiers()`, tier thresholds (STRONG=1.0, SOFT=0.5, etc.)
- Any new channels added (e.g., `forest_prob_pre/post/delta`) must be documented with shape/dtype/ranges
- Cache should be extended, not replaced (existing keys stay at same values)

---

## Status log

- `2026-04-18 16:45` — P0 audit complete. Baseline perfect (0.9824/0.7749). 16 tiles, 3 zones. Ready for P3.
- `2026-04-18 17:30` — P3 error analysis complete. **Decision: proceed to P1.** Key findings:
  - FN concentrated in NDVI 0.3–0.5 (81% miss rate in [0.4,0.5) bin) — degraded/edge forest
  - AEF tree-dims show strong separation on worst tile 18NWG_6_6: A21=3.22σ, A34=2.60σ, A26=2.11σ, A18=1.82σ
  - Edge pixels 7–9× more likely to be FN than core pixels across all tiles
  - Worst tiles: 48PUT_0_8 (0.934), 18NWH_1_4 (0.943), 18NWG_6_6 (0.967)
  - Report: `code/reports/P3_ERROR_ANALYSIS.md`, figures in `code/reports/figs/`
- `2026-04-18 17:55` — P1 SHIPPED. Added A18/21/26/28/34 as explicit channels in `_per_pixel_features` (flag `include_aef_tree_dims`, default True).
  - Aggregate recall **0.9824 → 0.9840** (+0.0016), coverage 0.7749 → 0.7717 (−0.0033, more precise)
  - **Worst tile 48PUT_0_8: 0.9343 → 0.9422** (+0.8%), **18NWH_1_4: 0.9431 → 0.9675** (+2.4%)
  - No regressions beyond noise (worst delta −0.0001). Min tile-recall improved 0.9343 → 0.9422.
  - Report: `code/reports/P1_AEF_DIMS.md`
- `2026-04-18 18:40` — **P2 SHIPPED.** Post-period masker trained with still-forest positives (`forest_gt_pre2020 & label==0`).
  - 3 new per-tile rasters cached in `cache_probs/<tile_id>.npz`: `forest_prob_pre`, `forest_prob_post`, `forest_prob_delta`
  - Sanity: deforested pixels show mean delta +0.151 vs +0.058 for stable pixels (3× ratio, 0.43σ separation). Weak-but-real signal — use as soft feature, not hard classifier.
  - Handoff doc: `code/reports/P2_INTEGRATION.md` — load + stack-as-features instructions for deforestation team.
  - Script: `code/reports/p2_post_period.py` (deterministic, ~3 min on 16 tiles).
- `2026-04-18 18:10` — Re-prioritisation after reviewing P3 + P1 results:
  - **Biggest remaining signal: edges** (7–9× FN rate vs core, every tile). Not touched by P1.
  - **Second: NDVI [0.4,0.5) still at 81% aggregate miss** — P1 helped, not eliminated.
  - **Coverage already high (~0.77)**, so remaining work should lean toward precision, not recall-at-all-costs.
  - **Next steps ordered:** P2 (handoff value) → P4 edge-weighted (targets dominant pattern) → handoff.
  - **Deferred:** P1b water audit (no evidence of confusion), morphological dilation (precision risk), MLP ablation (diminishing returns).
- `2026-04-18 18:05` — Candidate follow-up (now deferred): **P1b easy non-forest audit + strict water mask**.
  - **Why:** current training/eval is asymmetric: positives are weak alert-derived forest labels, and evaluation is mostly recall-vs-alerts + coverage. This does **not** cleanly measure "obvious non-forest predicted as forest."
  - **Use for validation:** add a tiny high-confidence negative audit set (especially **water**, optionally bare/urban/cropland proxies) and report predicted-forest rate on these pixels.
  - **Use for inference:** if water can be identified robustly from S2/S1/AEF, apply a **strict water mask** as a precision guardrail.
  - **Priority rationale:** worthwhile for precision sanity-checking, but lower priority than P3/P1 because current failures are mainly ambiguous **edge / mid-NDVI forest misses**, not clear water-vs-forest confusion.
  - **Recommendation:** treat this primarily as an audit + conservative post-mask, **not** as a major new training objective unless the audit shows meaningful false positives on obvious non-forest.
