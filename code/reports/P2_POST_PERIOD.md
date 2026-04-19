# P2 — Post-period forest probability channels

Wrote 3 per-tile files to `cache_probs/`.

**Sanity:** on pre-forest pixels, mean `forest_prob_delta` = +0.3142 for deforested (label=1) vs +0.7019 for non-deforested (label=0). Separation |Δμ|/σ = 1.027. Verdict: **SHIP**.

**`delta > 0.3` as deforestation indicator:** precision=0.028, recall=0.426.

See `P2_INTEGRATION.md` for handoff instructions.
