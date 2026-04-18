# P3 Error Analysis — where the pre-2020 forest mask fails

**Tiles analyzed:** 10
**Total GT pixels:** 2,062,570 · **Total FN pixels:** 36,392 (1.76%)
**Aggregate recall:** 0.9824

## Per-tile recall (sorted ascending — worst first)

| tile | recall | coverage | GT | FN | FP |
|------|--------|----------|-----|-----|-----|
| `48PUT_0_8` | 0.9343 | 0.572 | 74,413 | 4,890 | 513,083 |
| `18NWH_1_4` | 0.9431 | 0.899 | 35,198 | 2,003 | 869,009 |
| `18NWG_6_6` | 0.9675 | 0.852 | 369,966 | 12,029 | 497,401 |
| `48QVE_3_0` | 0.9707 | 0.621 | 181,511 | 5,322 | 455,035 |
| `48QWD_2_2` | 0.9718 | 0.604 | 175,863 | 4,952 | 437,047 |
| `48PXC_7_7` | 0.9839 | 0.742 | 149,219 | 2,395 | 618,886 |
| `48PYB_3_6` | 0.9859 | 0.810 | 213,847 | 3,019 | 632,172 |
| `48PWV_7_8` | 0.9973 | 0.843 | 464,520 | 1,263 | 390,263 |
| `18NXJ_7_6` | 0.9987 | 0.824 | 29,995 | 40 | 800,627 |
| `18NXH_6_8` | 0.9987 | 0.986 | 368,038 | 479 | 622,734 |

## Bottom-3 tiles — failure mode breakdown

### `48PUT_0_8` — recall 0.934, 4,890 FN pixels

**FN by NDVI bin** (what NDVI values are being missed?):

| NDVI bin | GT pixels | FN pixels | miss rate |
|----------|-----------|-----------|-----------|
| [-1.0,0.3) | 3,221 | 704 | 21.86% |
| [0.3,0.4) | 1,816 | 883 | 48.62% |
| [0.4,0.5) | 3,891 | 1,814 | 46.62% |
| [0.5,0.6) | 8,645 | 1,489 | 17.22% |
| [0.6,1.0) | 56,840 | 0 | 0.00% |

**Edge vs core:** edge FN rate 8.89% (dist≤2), core FN rate 1.18% (dist>5). Edges are weak.

**FN connected-component size:** isolated (≤10px) 53.9% · medium (11-100) 46.1% · large (>100) 0.0%

**AEF tree-dim separation (TP mean vs FN mean, higher ∣Δ/σ∣ = better discriminator):**

| dim | TP mean | FN mean | TP std | separation (\|Δ\|/σ) |
|-----|---------|---------|--------|-----------------------|
| A18 | -0.0031 | 0.0242 | 0.0548 | **0.50** |
| A21 | -0.2031 | -0.2455 | 0.0698 | **0.61** |
| A26 | 0.0885 | 0.1167 | 0.0499 | **0.56** |
| A28 | -0.1250 | -0.1188 | 0.0331 | **0.19** |
| A34 | -0.0354 | 0.0238 | 0.0713 | **0.83** |

→ Best separation 0.83σ — adding these dims may help the learned masker.

### `18NWH_1_4` — recall 0.943, 2,003 FN pixels

**FN by NDVI bin** (what NDVI values are being missed?):

| NDVI bin | GT pixels | FN pixels | miss rate |
|----------|-----------|-----------|-----------|
| [-1.0,0.3) | 11 | 11 | 100.00% |
| [0.3,0.4) | 211 | 144 | 68.25% |
| [0.4,0.5) | 1,585 | 978 | 61.70% |
| [0.5,0.6) | 3,012 | 870 | 28.88% |
| [0.6,1.0) | 30,379 | 0 | 0.00% |

**Edge vs core:** edge FN rate 7.53% (dist≤2), core FN rate 0.00% (dist>5). Edges are weak.

**FN connected-component size:** isolated (≤10px) 54.5% · medium (11-100) 40.1% · large (>100) 5.4%

**AEF tree-dim separation (TP mean vs FN mean, higher ∣Δ/σ∣ = better discriminator):**

| dim | TP mean | FN mean | TP std | separation (Δ/σ) |
|-----|---------|---------|--------|-------------------|
| A18 | -0.0297 | 0.0144 | 0.0772 | **0.57** |
| A21 | -0.0958 | -0.1130 | 0.0770 | **0.22** |
| A26 | 0.0825 | 0.1135 | 0.0749 | **0.41** |
| A28 | -0.1593 | -0.1017 | 0.0472 | **1.22** |
| A34 | -0.0132 | 0.0820 | 0.0692 | **1.38** |

→ Best separation 1.38σ — adding these dims may help the learned masker.

### `18NWG_6_6` — recall 0.967, 12,029 FN pixels

**FN by NDVI bin** (what NDVI values are being missed?):

| NDVI bin | GT pixels | FN pixels | miss rate |
|----------|-----------|-----------|-----------|
| [-1.0,0.3) | 239 | 173 | 72.38% |
| [0.3,0.4) | 2,138 | 1,751 | 81.90% |
| [0.4,0.5) | 4,663 | 4,451 | 95.45% |
| [0.5,0.6) | 16,820 | 5,654 | 33.61% |
| [0.6,1.0) | 346,106 | 0 | 0.00% |

**Edge vs core:** edge FN rate 7.96% (dist≤2), core FN rate 1.05% (dist>5). Edges are weak.

**FN connected-component size:** isolated (≤10px) 28.3% · medium (11-100) 30.6% · large (>100) 41.1%

**AEF tree-dim separation (TP mean vs FN mean, higher ∣Δ/σ∣ = better discriminator):**

| dim | TP mean | FN mean | TP std | separation (Δ/σ) |
|-----|---------|---------|--------|---------------------|
| A18 | -0.0521 | 0.0062 | 0.0320 | **1.82** |
| A21 | 0.0361 | -0.1160 | 0.0472 | **3.22** |
| A26 | 0.0558 | 0.1279 | 0.0342 | **2.11** |
| A28 | -0.1586 | -0.1363 | 0.0221 | **1.01** |
| A34 | 0.0045 | 0.1475 | 0.0549 | **2.60** |

→ Best separation 3.22σ — adding these dims may help the learned masker.

## Aggregate: FN pixels across all tiles, by NDVI

| NDVI bin | total GT | total FN | miss rate |
|----------|----------|----------|-----------|
| [-1.0,0.3) | 3,784 | 1,178 | 31.13% |
| [0.3,0.4) | 6,021 | 4,509 | 74.89% |
| [0.4,0.5) | 17,017 | 13,826 | 81.25% |
| [0.5,0.6) | 56,386 | 16,879 | 29.93% |
| [0.6,1.0) | 1,979,362 | 0 | 0.00% |

## Recommendations for P1

- **Worst NDVI bin:** [0.4,0.5) with 13,826 FN (81.25% miss rate).
- **AEF tree-dims show separation** in at least one weak tile — P1 is worth trying.
