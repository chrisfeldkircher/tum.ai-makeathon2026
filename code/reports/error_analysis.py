"""P3 error analysis — characterize where the pre-2020 forest mask fails.

Goal: understand *why* the masker misses ~2% of forest_gt_pre2020 pixels,
and whether those failures cluster by NDVI level, forest-edge distance,
connected-component size, or MGRS zone. This informs whether P1
(paper-driven AEF dims) is worth the effort.

Usage (from repo root):
    python code/reports/error_analysis.py --cache_dir ./cache

Produces:
    code/reports/P3_ERROR_ANALYSIS.md — per-tile + aggregate failure summary
    code/reports/figs/p3_error_overlay_<tile>.png — GT vs prediction, errors highlighted
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[2]
_CODE_DIR = str(REPO_ROOT / "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

REPORT_DIR = Path(__file__).resolve().parent
FIG_DIR = REPORT_DIR / "figs"
OUT_MD = REPORT_DIR / "P3_ERROR_ANALYSIS.md"

NDVI_BINS = [(-1.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.0)]
# Paper-driven tree dims (1-based → 0-based indices)
TREE_DIM_IDX = {"A18": 17, "A21": 20, "A26": 25, "A28": 27, "A34": 33}


def load_tile(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def edge_distance(mask: np.ndarray) -> np.ndarray:
    """Distance (in pixels) from each True-pixel to the nearest False-pixel.
    Returns 0 for non-mask pixels."""
    return ndimage.distance_transform_edt(mask)


def connected_component_sizes(mask: np.ndarray) -> np.ndarray:
    """For each True pixel, the size of the CC it belongs to (0 outside mask)."""
    labeled, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=np.int32)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return sizes[labeled].astype(np.int32)


def analyze_tile(tile_id: str, tensors: dict[str, np.ndarray],
                 prediction: np.ndarray) -> dict:
    """Compute failure-mode slices for a single tile."""
    gt = tensors["forest_gt_pre2020"].astype(bool)
    pred = prediction.astype(bool)
    ndvi = tensors["s2_pre_ndvi_median"]
    ndvi_std = tensors.get("s2_pre_ndvi_std", np.zeros_like(ndvi))
    aef = tensors.get("aef_pre")  # (64, H, W)

    tp = gt & pred
    fn = gt & ~pred      # what the masker misses (main failure mode)
    fp = ~gt & pred      # false-alarm forest (less critical — alert union is sparse)

    n_gt = int(gt.sum())
    n_fn = int(fn.sum())
    n_fp = int(fp.sum())
    recall = float(tp.sum()) / max(n_gt, 1)
    coverage = float(pred.mean())

    # Slice FN by NDVI
    fn_by_ndvi = []
    for lo, hi in NDVI_BINS:
        in_bin = (ndvi >= lo) & (ndvi < hi)
        n_in_bin_fn = int((fn & in_bin).sum())
        n_in_bin_gt = int((gt & in_bin).sum())
        fn_by_ndvi.append({
            "bin": f"[{lo:.1f},{hi:.1f})",
            "fn": n_in_bin_fn,
            "gt": n_in_bin_gt,
            "fn_rate": (n_in_bin_fn / n_in_bin_gt) if n_in_bin_gt else 0.0,
        })

    # Edge analysis: dist from each GT pixel to non-GT boundary
    edge_dist = edge_distance(gt).astype(np.int32)
    # FN at edge (dist ≤ 2) vs core (dist > 5)
    edge_fn = int((fn & (edge_dist <= 2)).sum())
    edge_gt = int((gt & (edge_dist <= 2)).sum())
    core_fn = int((fn & (edge_dist > 5)).sum())
    core_gt = int((gt & (edge_dist > 5)).sum())

    # CC sizes of FN regions (small = isolated, large = big missed patches)
    fn_cc_sizes = connected_component_sizes(fn)
    fn_isolated = int(((fn) & (fn_cc_sizes <= 10)).sum())  # <=10 pixels
    fn_medium   = int(((fn) & (fn_cc_sizes > 10) & (fn_cc_sizes <= 100)).sum())
    fn_large    = int(((fn) & (fn_cc_sizes > 100)).sum())

    # AEF tree-dim signal comparison: mean AEF value in FN vs TP
    aef_fn_vs_tp = {}
    if aef is not None and aef.shape[0] == 64:
        for name, idx in TREE_DIM_IDX.items():
            ch = aef[idx]
            tp_vals = ch[tp]
            fn_vals = ch[fn]
            if tp_vals.size and fn_vals.size:
                aef_fn_vs_tp[name] = {
                    "tp_mean": float(tp_vals.mean()),
                    "fn_mean": float(fn_vals.mean()),
                    "tp_std":  float(tp_vals.std()),
                    "fn_std":  float(fn_vals.std()),
                    "separation": float(abs(tp_vals.mean() - fn_vals.mean()) / (tp_vals.std() + 1e-6)),
                }

    return {
        "tile_id": tile_id,
        "n_gt": n_gt, "n_fn": n_fn, "n_fp": n_fp,
        "recall": recall, "coverage": coverage,
        "fn_by_ndvi": fn_by_ndvi,
        "edge_fn": edge_fn, "edge_gt": edge_gt,
        "core_fn": core_fn, "core_gt": core_gt,
        "fn_isolated": fn_isolated, "fn_medium": fn_medium, "fn_large": fn_large,
        "aef_fn_vs_tp": aef_fn_vs_tp,
    }


def viz_error_overlay(tile_id: str, tensors: dict, prediction: np.ndarray,
                      out_path: Path) -> None:
    """6-panel overlay: S2 RGB / NDVI / NDVI_std / GT / prediction / errors (FN red, FP yellow)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    ndvi = tensors["s2_pre_ndvi_median"]
    ndvi_std = tensors.get("s2_pre_ndvi_std", np.zeros_like(ndvi))
    gt = tensors["forest_gt_pre2020"].astype(bool)
    pred = prediction.astype(bool)

    # Error map: 0=correct-non, 1=TP (forest-caught), 2=FN (forest-missed, red),
    # 3=FP (non-forest predicted, yellow)
    error = np.zeros_like(gt, dtype=np.uint8)
    error[gt & pred] = 1
    error[gt & ~pred] = 2
    error[~gt & pred] = 3

    err_cmap = ListedColormap(["#202020", "#2b8a3e", "#e03131", "#ffd43b"])

    rgb = None
    bands = tensors.get("s2_pre_band_median")
    if bands is not None and bands.shape[0] >= 4:
        r = bands[3]; g = bands[2]; b = bands[1]
        rgb_stack = np.stack([r, g, b], axis=-1)
        lo, hi = np.nanpercentile(rgb_stack, [2, 98])
        rgb = np.clip((rgb_stack - lo) / max(hi - lo, 1e-6), 0, 1)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    if rgb is not None:
        axs[0, 0].imshow(rgb); axs[0, 0].set_title(f"{tile_id}  S2 RGB")
    else:
        axs[0, 0].text(0.5, 0.5, "no S2 bands", ha="center")
    axs[0, 1].imshow(ndvi, vmin=-0.2, vmax=0.9, cmap="RdYlGn"); axs[0, 1].set_title("NDVI pre")
    axs[0, 2].imshow(ndvi_std, vmin=0, vmax=0.3, cmap="viridis"); axs[0, 2].set_title("NDVI std")
    axs[1, 0].imshow(gt, cmap="gray", vmin=0, vmax=1); axs[1, 0].set_title(f"GT ({gt.mean():.1%})")
    axs[1, 1].imshow(pred, cmap="gray", vmin=0, vmax=1); axs[1, 1].set_title(f"Pred ({pred.mean():.1%})")
    axs[1, 2].imshow(error, cmap=err_cmap, vmin=0, vmax=3)
    fn_rate = (gt & ~pred).sum() / max(gt.sum(), 1)
    axs[1, 2].set_title(f"Errors: green=TP, RED=FN ({fn_rate:.1%}), yellow=FP")
    for ax in axs.flat: ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def write_report(out_path: Path, results: list[dict]) -> None:
    """Aggregate + per-tile analysis to markdown."""
    # Sort by recall ascending — worst tiles first
    results = sorted(results, key=lambda r: r["recall"])

    total_gt = sum(r["n_gt"] for r in results)
    total_fn = sum(r["n_fn"] for r in results)
    agg_recall = 1 - (total_fn / max(total_gt, 1))

    lines = []
    lines.append("# P3 Error Analysis — where the pre-2020 forest mask fails")
    lines.append("")
    lines.append(f"**Tiles analyzed:** {len(results)}")
    lines.append(f"**Total GT pixels:** {total_gt:,} · **Total FN pixels:** {total_fn:,} ({total_fn/max(total_gt,1):.2%})")
    lines.append(f"**Aggregate recall:** {agg_recall:.4f}")
    lines.append("")

    lines.append("## Per-tile recall (sorted ascending — worst first)")
    lines.append("")
    lines.append("| tile | recall | coverage | GT | FN | FP |")
    lines.append("|------|--------|----------|-----|-----|-----|")
    for r in results:
        lines.append(f"| `{r['tile_id']}` | {r['recall']:.4f} | {r['coverage']:.3f} | "
                     f"{r['n_gt']:,} | {r['n_fn']:,} | {r['n_fp']:,} |")
    lines.append("")

    # Bottom-3 deep dive
    bottom = results[:3]
    lines.append("## Bottom-3 tiles — failure mode breakdown")
    lines.append("")
    for r in bottom:
        lines.append(f"### `{r['tile_id']}` — recall {r['recall']:.3f}, {r['n_fn']:,} FN pixels")
        lines.append("")

        # NDVI slicing
        lines.append("**FN by NDVI bin** (what NDVI values are being missed?):")
        lines.append("")
        lines.append("| NDVI bin | GT pixels | FN pixels | miss rate |")
        lines.append("|----------|-----------|-----------|-----------|")
        for b in r["fn_by_ndvi"]:
            lines.append(f"| {b['bin']} | {b['gt']:,} | {b['fn']:,} | {b['fn_rate']:.2%} |")
        lines.append("")

        # Edge analysis
        edge_rate = r["edge_fn"] / max(r["edge_gt"], 1)
        core_rate = r["core_fn"] / max(r["core_gt"], 1)
        lines.append(f"**Edge vs core:** edge FN rate {edge_rate:.2%} (dist≤2), "
                     f"core FN rate {core_rate:.2%} (dist>5). "
                     f"{'Edges are weak.' if edge_rate > core_rate * 1.5 else 'Core and edges similar.'}")
        lines.append("")

        # CC size
        total_fn_cc = r["fn_isolated"] + r["fn_medium"] + r["fn_large"]
        if total_fn_cc > 0:
            lines.append(f"**FN connected-component size:** "
                         f"isolated (≤10px) {r['fn_isolated']/total_fn_cc:.1%} · "
                         f"medium (11-100) {r['fn_medium']/total_fn_cc:.1%} · "
                         f"large (>100) {r['fn_large']/total_fn_cc:.1%}")
            lines.append("")

        # AEF tree-dim signal
        if r["aef_fn_vs_tp"]:
            lines.append("**AEF tree-dim separation (TP mean vs FN mean, higher ∣Δ/σ∣ = better discriminator):**")
            lines.append("")
            lines.append("| dim | TP mean | FN mean | TP std | separation (|Δ|/σ) |")
            lines.append("|-----|---------|---------|--------|---------------------|")
            for name, v in r["aef_fn_vs_tp"].items():
                lines.append(f"| {name} | {v['tp_mean']:.4f} | {v['fn_mean']:.4f} | "
                             f"{v['tp_std']:.4f} | **{v['separation']:.2f}** |")
            lines.append("")
            max_sep = max(v["separation"] for v in r["aef_fn_vs_tp"].values())
            if max_sep > 0.5:
                lines.append(f"→ Best separation {max_sep:.2f}σ — adding these dims may help the learned masker.")
            else:
                lines.append(f"→ Best separation {max_sep:.2f}σ — AEF tree-dims alone can't rescue these misses.")
            lines.append("")

    # Aggregate NDVI distribution of all FN
    lines.append("## Aggregate: FN pixels across all tiles, by NDVI")
    lines.append("")
    lines.append("| NDVI bin | total GT | total FN | miss rate |")
    lines.append("|----------|----------|----------|-----------|")
    agg_bins = {b["bin"]: {"gt": 0, "fn": 0} for b in results[0]["fn_by_ndvi"]}
    for r in results:
        for b in r["fn_by_ndvi"]:
            agg_bins[b["bin"]]["gt"] += b["gt"]
            agg_bins[b["bin"]]["fn"] += b["fn"]
    for name, v in agg_bins.items():
        rate = v["fn"] / v["gt"] if v["gt"] else 0.0
        lines.append(f"| {name} | {v['gt']:,} | {v['fn']:,} | {rate:.2%} |")
    lines.append("")

    # Recommendations
    lines.append("## Recommendations for P1")
    lines.append("")
    # Quick heuristic: which NDVI bin has the worst miss rate aggregate?
    worst_ndvi = max(agg_bins.items(), key=lambda kv: kv[1]["fn"] / max(kv[1]["gt"], 1))
    lines.append(f"- **Worst NDVI bin:** {worst_ndvi[0]} with {worst_ndvi[1]['fn']:,} FN "
                 f"({worst_ndvi[1]['fn']/max(worst_ndvi[1]['gt'],1):.2%} miss rate).")
    # Check if any tile has strong AEF separation
    any_strong_sep = any(
        any(v["separation"] > 0.5 for v in r["aef_fn_vs_tp"].values())
        for r in results if r["aef_fn_vs_tp"]
    )
    if any_strong_sep:
        lines.append("- **AEF tree-dims show separation** in at least one weak tile — P1 is worth trying.")
    else:
        lines.append("- **AEF tree-dims show weak separation** — P1 unlikely to help; failures may be irreducible label noise.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache_dir", default=str(REPO_ROOT / "cache"))
    ap.add_argument("--backend", default="lightgbm", choices=["lightgbm", "logreg"])
    ap.add_argument("--skip_figs", action="store_true")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    paths = sorted(cache_dir.glob("*.npz"))
    if not paths:
        print(f"[FATAL] no *.npz in {cache_dir}", file=sys.stderr)
        return 2

    # Load all tiles, keep only training (with forest_gt_pre2020)
    print(f"[P3] loading {len(paths)} tiles …")
    train_tiles = []
    for p in paths:
        t = load_tile(p)
        if "forest_gt_pre2020" in t:
            train_tiles.append((p.stem, t))
    print(f"[P3] {len(train_tiles)} training tiles have ground truth")

    # Train masker once on all training tiles
    from model.vegetationPredictor import LearnedForestMasker  # type: ignore
    print(f"[P3] training LearnedForestMasker ({args.backend}) on {len(train_tiles)} tiles …")
    masker = LearnedForestMasker(backend=args.backend).fit([t for _, t in train_tiles])
    print("[P3] masker trained. Running per-tile error analysis …")

    results = []
    for tid, t in train_tiles:
        pred = masker.predict(t)
        res = analyze_tile(tid, t, pred)
        results.append(res)
        print(f"  {tid}: recall={res['recall']:.4f}, FN={res['n_fn']:,}")

    # Figures for bottom-2 tiles
    if not args.skip_figs:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        bottom = sorted(results, key=lambda r: r["recall"])[:2]
        for r in bottom:
            tile_dict = dict(train_tiles)[r["tile_id"]]
            pred = masker.predict(tile_dict)
            out = FIG_DIR / f"p3_error_overlay_{r['tile_id']}.png"
            try:
                viz_error_overlay(r["tile_id"], tile_dict, pred, out)
                print(f"  wrote {out.relative_to(REPO_ROOT)}")
            except Exception as e:
                print(f"  [warn] viz failed for {r['tile_id']}: {e}")

    write_report(OUT_MD, results)
    print(f"[P3] wrote report -> {OUT_MD.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
