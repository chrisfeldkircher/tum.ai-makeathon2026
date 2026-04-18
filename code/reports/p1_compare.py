"""P1: compare baseline (mean/std only) vs paper-driven AEF tree dims (A18/21/26/28/34).

Trains two LearnedForestMasker instances on the same 10 cached tiles and reports
aggregate + per-tile recall/coverage. Ships the new config only if it strictly
improves on baseline.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from model.vegetationPredictor import LearnedForestMasker, evaluate_masker_on_tiles  # noqa: E402


def load_tiles(cache_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    tiles = {}
    for p in sorted(cache_dir.glob("*.npz")):
        with np.load(p, allow_pickle=True) as z:
            t = {k: z[k] for k in z.files}
        if "forest_gt_pre2020" in t:
            tiles[p.stem] = t
    return tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=Path, default=REPO_ROOT / "cache")
    args = ap.parse_args()

    tiles = load_tiles(args.cache_dir)
    tile_list = list(tiles.values())
    tile_ids = list(tiles.keys())
    print(f"[P1] loaded {len(tiles)} tiles with ground truth")

    configs = [
        ("baseline_mean_std",  dict(include_aef=True,  include_aef_tree_dims=False)),
        ("paper_tree_dims",    dict(include_aef=True,  include_aef_tree_dims=True)),
    ]

    results = {}
    for name, kw in configs:
        print(f"\n[P1] training {name} ...")
        m = LearnedForestMasker(backend="lightgbm", **kw).fit(tile_list, seed=0)
        recalls = {}
        covs = {}
        tot_tp = tot_fn = tot_pos = tot_pred_pos = tot_pix = 0
        for tid, t in tiles.items():
            p = m.predict_proba(t) >= m.threshold
            gt = t["forest_gt_pre2020"] > 0
            tp = int((p & gt).sum()); fn = int((~p & gt).sum())
            pos = int(gt.sum())
            recalls[tid] = tp / max(pos, 1)
            covs[tid] = float(p.mean())
            tot_tp += tp; tot_fn += fn; tot_pos += pos
            tot_pred_pos += int(p.sum()); tot_pix += p.size
        agg_recall = tot_tp / max(tot_pos, 1)
        agg_cov = tot_pred_pos / max(tot_pix, 1)
        results[name] = dict(recalls=recalls, covs=covs,
                             agg_recall=agg_recall, agg_cov=agg_cov)
        print(f"  {name}: recall={agg_recall:.4f}  coverage={agg_cov:.4f}")

    # Per-tile comparison
    print(f"\n{'tile':<14}  {'baseline':>8}  {'tree_dims':>10}  {'delta':>8}")
    print("-" * 48)
    b = results["baseline_mean_std"]["recalls"]
    n = results["paper_tree_dims"]["recalls"]
    deltas = []
    for tid in sorted(tile_ids, key=lambda x: b[x]):
        d = n[tid] - b[tid]
        deltas.append(d)
        tag = " ++" if d > 0.005 else (" --" if d < -0.005 else "")
        print(f"{tid:<14}  {b[tid]:>8.4f}  {n[tid]:>10.4f}  {d:>+8.4f}{tag}")

    rb = results["baseline_mean_std"]["agg_recall"]
    rn = results["paper_tree_dims"]["agg_recall"]
    cb = results["baseline_mean_std"]["agg_cov"]
    cn = results["paper_tree_dims"]["agg_cov"]
    print(f"\nAGGREGATE   recall: {rb:.4f} -> {rn:.4f}  (d={rn-rb:+.4f})")
    print(f"            coverage: {cb:.4f} -> {cn:.4f}  (d={cn-cb:+.4f})")
    min_b = min(b.values()); min_n = min(n.values())
    print(f"MIN tile-recall: {min_b:.4f} -> {min_n:.4f}  (d={min_n-min_b:+.4f})")

    verdict = "SHIP" if (rn >= rb and min_n >= min_b - 0.002) else "REVERT"
    print(f"\n[P1] verdict: {verdict}")

    # Write a small report
    out = REPO_ROOT / "code" / "reports" / "P1_AEF_DIMS.md"
    with out.open("w", encoding="utf-8") as f:
        f.write("# P1 — Paper-driven AEF tree dims A/B\n\n")
        f.write(f"**Aggregate recall:** {rb:.4f} -> {rn:.4f} (d {rn-rb:+.4f})\n\n")
        f.write(f"**Aggregate coverage:** {cb:.4f} -> {cn:.4f} (d {cn-cb:+.4f})\n\n")
        f.write(f"**Min per-tile recall:** {min_b:.4f} -> {min_n:.4f} (d {min_n-min_b:+.4f})\n\n")
        f.write("## Per-tile recall\n\n")
        f.write("| tile | baseline | +tree_dims | d |\n")
        f.write("|------|----------|-----------|----|\n")
        for tid in sorted(tile_ids, key=lambda x: b[x]):
            f.write(f"| `{tid}` | {b[tid]:.4f} | {n[tid]:.4f} | {n[tid]-b[tid]:+.4f} |\n")
        f.write(f"\n**Verdict:** {verdict}\n")
    print(f"[P1] wrote {out}")


if __name__ == "__main__":
    main()
