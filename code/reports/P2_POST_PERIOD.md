# P2 — Post-period forest probability channels

Wrote 16 per-tile files to `cache_probs/`.

**Sanity:** on pre-forest pixels, mean `forest_prob_delta` = +0.1515 for deforested (label=1) vs +0.0576 for non-deforested (label=0). Separation |Δμ|/σ = 0.434. Verdict: **WEAK**.

**`delta > 0.3` as deforestation indicator:** precision=0.254, recall=0.204.

See `P2_INTEGRATION.md` for handoff instructions.
