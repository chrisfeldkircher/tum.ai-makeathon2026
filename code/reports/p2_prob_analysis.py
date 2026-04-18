"""Analyze cached P2 forest probabilities and generate summary figures/report.

Outputs:
  - code/reports/P2_STATS.md
  - code/reports/figs/p2_delta_insights.png
  - code/reports/figs/p2_high_delta_true_positive_examples.png
  - code/reports/figs/p2_high_delta_false_positive_examples.png
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TileRecord:
    tile_id: str
    label: np.ndarray | None
    prob_pre: np.ndarray
    prob_post: np.ndarray
    prob_delta: np.ndarray


def load_labeled_tiles(cache_dir: Path, probs_dir: Path) -> list[TileRecord]:
    records: list[TileRecord] = []
    for probs_path in sorted(probs_dir.glob("*.npz")):
        tile_id = probs_path.stem
        cache_path = cache_dir / probs_path.name
        if not cache_path.exists():
            continue
        with np.load(probs_path, allow_pickle=True) as z_probs:
            prob_pre = z_probs["forest_prob_pre"].astype(np.float32)
            prob_post = z_probs["forest_prob_post"].astype(np.float32)
            prob_delta = z_probs["forest_prob_delta"].astype(np.float32)
        with np.load(cache_path, allow_pickle=True) as z_cache:
            label = z_cache["label"].astype(np.uint8) if "forest_gt_pre2020" in z_cache.files and "label" in z_cache.files else None
        if label is None:
            continue
        records.append(
            TileRecord(
                tile_id=tile_id,
                label=label,
                prob_pre=prob_pre,
                prob_post=prob_post,
                prob_delta=prob_delta,
            )
        )
    return records


def auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = int(labels.sum())
    neg = int(labels.size - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    uniq, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if uniq.size != scores.size:
        rank_sums = np.bincount(inv, weights=ranks)
        avg_ranks = rank_sums / counts
        ranks = avg_ranks[inv]
    pos_rank_sum = ranks[labels == 1].sum()
    return float((pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def threshold_metrics(scores: np.ndarray, labels: np.ndarray, thresholds: list[float]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for thr in thresholds:
        pred = scores > thr
        tp = int(np.sum(pred & (labels == 1)))
        fp = int(np.sum(pred & (labels == 0)))
        fn = int(np.sum((~pred) & (labels == 1)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        rows.append(
            {
                "threshold": thr,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_share": float(pred.mean()),
            }
        )
    return rows


def pick_examples(
    records: list[TileRecord],
    target_label: int,
    limit: int = 4,
    min_spacing: int = 120,
    window_radius: int = 60,
    max_per_tile: int = 2,
) -> list[dict[str, int | float | str]]:
    candidates: list[tuple[float, str, int, int]] = []
    for rec in records:
        pre_mask = rec.prob_pre >= 0.5
        mask = pre_mask & (rec.label == target_label)
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        vals = rec.prob_delta[ys, xs]
        order = np.argsort(vals)[::-1]
        take = min(250, order.size)
        for idx in order[:take]:
            candidates.append((float(vals[idx]), rec.tile_id, int(ys[idx]), int(xs[idx])))
    candidates.sort(reverse=True)

    selected: list[dict[str, int | float | str]] = []
    occupied: list[tuple[str, int, int]] = []
    tile_counts: dict[str, int] = {}
    for delta, tile_id, y, x in candidates:
        if tile_counts.get(tile_id, 0) >= max_per_tile:
            continue
        if any(tile_id == otile and abs(y - oy) < min_spacing and abs(x - ox) < min_spacing for otile, oy, ox in occupied):
            continue
        selected.append(
            {
                "tile_id": tile_id,
                "y": y,
                "x": x,
                "delta": delta,
                "window_radius": window_radius,
            }
        )
        occupied.append((tile_id, y, x))
        tile_counts[tile_id] = tile_counts.get(tile_id, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def crop(arr: np.ndarray, y: int, x: int, radius: int) -> np.ndarray:
    y0 = max(0, y - radius)
    y1 = min(arr.shape[0], y + radius)
    x0 = max(0, x - radius)
    x1 = min(arr.shape[1], x + radius)
    return arr[y0:y1, x0:x1]


def save_example_grid(records_by_id: dict[str, TileRecord], examples: list[dict[str, int | float | str]], out_path: Path, title: str) -> None:
    if not examples:
        return
    fig, axes = plt.subplots(len(examples), 4, figsize=(12, 3.2 * len(examples)), constrained_layout=True)
    if len(examples) == 1:
        axes = np.array([axes])
    cmaps = ["viridis", "viridis", "coolwarm", "gray_r"]
    names = ["P(forest pre)", "P(forest post)", "Delta", "Label"]
    for row_idx, ex in enumerate(examples):
        rec = records_by_id[str(ex["tile_id"])]
        y = int(ex["y"])
        x = int(ex["x"])
        radius = int(ex["window_radius"])
        arrays = [
            crop(rec.prob_pre, y, x, radius),
            crop(rec.prob_post, y, x, radius),
            crop(rec.prob_delta, y, x, radius),
            crop(rec.label, y, x, radius),
        ]
        for col_idx, (arr, cmap, name) in enumerate(zip(arrays, cmaps, names)):
            ax = axes[row_idx, col_idx]
            if name == "Delta":
                im = ax.imshow(arr, cmap=cmap, vmin=-1, vmax=1)
            elif name == "Label":
                im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
            else:
                im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(name)
            if col_idx == 0:
                ax.set_ylabel(
                    f"{ex['tile_id']}\n(y={y}, x={x})\nΔ={float(ex['delta']):.3f}",
                    rotation=0,
                    labelpad=48,
                    va="center",
                )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(title, fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=Path, default=REPO_ROOT / "cache")
    ap.add_argument("--probs_dir", type=Path, default=REPO_ROOT / "cache_probs")
    ap.add_argument("--report_path", type=Path, default=REPO_ROOT / "code" / "reports" / "P2_STATS.md")
    ap.add_argument("--fig_dir", type=Path, default=REPO_ROOT / "code" / "reports" / "figs")
    args = ap.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    records = load_labeled_tiles(args.cache_dir, args.probs_dir)
    if not records:
        raise SystemExit("No labeled tiles with cached probabilities found.")

    records_by_id = {rec.tile_id: rec for rec in records}

    pre = np.concatenate([rec.prob_pre.reshape(-1) for rec in records])
    post = np.concatenate([rec.prob_post.reshape(-1) for rec in records])
    delta = np.concatenate([rec.prob_delta.reshape(-1) for rec in records])
    labels = np.concatenate([rec.label.reshape(-1) for rec in records]).astype(np.uint8)

    pre_forest = pre >= 0.5
    delta_pf = delta[pre_forest]
    labels_pf = labels[pre_forest]
    pre_pf = pre[pre_forest]
    post_pf = post[pre_forest]

    mu_pos = float(delta_pf[labels_pf == 1].mean())
    mu_neg = float(delta_pf[labels_pf == 0].mean())
    std_neg = float(delta_pf[labels_pf == 0].std() + 1e-8)
    sep = abs(mu_pos - mu_neg) / std_neg
    auroc = auc_roc(delta_pf, labels_pf)

    quantiles = [0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    q_all = np.quantile(delta_pf, quantiles)
    q_pos = np.quantile(delta_pf[labels_pf == 1], quantiles)
    q_neg = np.quantile(delta_pf[labels_pf == 0], quantiles)

    thresholds = [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    thr_rows = threshold_metrics(delta_pf, labels_pf, thresholds)

    sorted_delta = np.sort(delta_pf)
    top_bands = []
    for frac in [0.10, 0.05, 0.01]:
        cutoff = float(sorted_delta[int(np.floor((1.0 - frac) * (sorted_delta.size - 1)))])
        mask = delta_pf >= cutoff
        top_bands.append(
            {
                "band": f"top {int(frac * 100)}%",
                "cutoff": cutoff,
                "count": int(mask.sum()),
                "positive_rate": float(labels_pf[mask].mean()),
            }
        )

    near_zero_share_all = float(np.mean(np.abs(delta_pf) <= 0.02))
    near_zero_share_pos = float(np.mean(np.abs(delta_pf[labels_pf == 1]) <= 0.02))
    near_zero_share_neg = float(np.mean(np.abs(delta_pf[labels_pf == 0]) <= 0.02))
    extreme_share_all = float(np.mean(delta_pf >= 0.95))
    extreme_share_pos = float(np.mean(delta_pf[labels_pf == 1] >= 0.95))
    extreme_share_neg = float(np.mean(delta_pf[labels_pf == 0] >= 0.95))

    fig_path = args.fig_dir / "p2_delta_insights.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    bins = np.linspace(-1, 1, 80)
    axes[0, 0].hist(delta_pf[labels_pf == 0], bins=bins, density=True, alpha=0.65, label="label=0", color="#4C78A8")
    axes[0, 0].hist(delta_pf[labels_pf == 1], bins=bins, density=True, alpha=0.65, label="label=1", color="#E45756")
    axes[0, 0].axvline(mu_neg, color="#4C78A8", linestyle="--", linewidth=1.5)
    axes[0, 0].axvline(mu_pos, color="#E45756", linestyle="--", linewidth=1.5)
    axes[0, 0].set_title("Delta distribution on pre-forest pixels")
    axes[0, 0].set_xlabel("forest_prob_delta")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()

    sample_n = min(25000, delta_pf.size)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(delta_pf.size, size=sample_n, replace=False)
    colors = np.where(labels_pf[sample_idx] == 1, "#E45756", "#4C78A8")
    axes[0, 1].scatter(pre_pf[sample_idx], post_pf[sample_idx], c=colors, s=4, alpha=0.18, linewidths=0)
    axes[0, 1].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Pre vs post forest probability")
    axes[0, 1].set_xlabel("forest_prob_pre")
    axes[0, 1].set_ylabel("forest_prob_post")

    axes[1, 0].plot([r["threshold"] for r in thr_rows], [r["precision"] for r in thr_rows], marker="o", label="Precision")
    axes[1, 0].plot([r["threshold"] for r in thr_rows], [r["recall"] for r in thr_rows], marker="o", label="Recall")
    axes[1, 0].plot([r["threshold"] for r in thr_rows], [r["f1"] for r in thr_rows], marker="o", label="F1")
    axes[1, 0].set_title("Hard-threshold performance")
    axes[1, 0].set_xlabel("delta threshold")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()

    band_labels = [b["band"] for b in top_bands]
    band_rates = [b["positive_rate"] for b in top_bands]
    baseline_rate = float(labels_pf.mean())
    axes[1, 1].bar(band_labels, band_rates, color="#72B7B2")
    axes[1, 1].axhline(baseline_rate, color="black", linestyle="--", label=f"baseline={baseline_rate:.3f}")
    for i, band in enumerate(top_bands):
        axes[1, 1].text(i, band["positive_rate"] + 0.01, f"{band['positive_rate']:.2f}", ha="center", va="bottom", fontsize=9)
    axes[1, 1].set_title("Deforestation rate in highest-delta tail")
    axes[1, 1].set_ylabel("Positive rate")
    axes[1, 1].set_ylim(0, max(max(band_rates) + 0.05, baseline_rate + 0.05))
    axes[1, 1].legend()

    fig.suptitle(
        f"P2 delta summary on labeled tiles | pre>=0.5 | AUROC={auroc:.3f}, separation={sep:.3f}",
        fontsize=14,
    )
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    tp_examples = pick_examples(records, target_label=1, limit=4)
    fp_examples = pick_examples(records, target_label=0, limit=4)
    tp_fig = args.fig_dir / "p2_high_delta_true_positive_examples.png"
    fp_fig = args.fig_dir / "p2_high_delta_false_positive_examples.png"
    save_example_grid(records_by_id, tp_examples, tp_fig, "High-delta true-positive examples")
    save_example_grid(records_by_id, fp_examples, fp_fig, "High-delta false-positive examples")

    args.report_path.write_text(
        build_report(
            records=records,
            pre=pre,
            post=post,
            delta=delta,
            labels=labels,
            pre_forest=pre_forest,
            delta_pf=delta_pf,
            labels_pf=labels_pf,
            q_all=q_all,
            q_pos=q_pos,
            q_neg=q_neg,
            quantiles=quantiles,
            mu_pos=mu_pos,
            mu_neg=mu_neg,
            sep=sep,
            auroc=auroc,
            thresholds=thr_rows,
            top_bands=top_bands,
            near_zero_share_all=near_zero_share_all,
            near_zero_share_pos=near_zero_share_pos,
            near_zero_share_neg=near_zero_share_neg,
            extreme_share_all=extreme_share_all,
            extreme_share_pos=extreme_share_pos,
            extreme_share_neg=extreme_share_neg,
            fig_path=fig_path,
            tp_fig=tp_fig,
            fp_fig=fp_fig,
            tp_examples=tp_examples,
            fp_examples=fp_examples,
        ),
        encoding="utf-8",
    )

    print(f"Wrote report: {args.report_path}")
    print(f"Wrote figures: {fig_path}, {tp_fig}, {fp_fig}")


def build_report(
    *,
    records: list[TileRecord],
    pre: np.ndarray,
    post: np.ndarray,
    delta: np.ndarray,
    labels: np.ndarray,
    pre_forest: np.ndarray,
    delta_pf: np.ndarray,
    labels_pf: np.ndarray,
    q_all: np.ndarray,
    q_pos: np.ndarray,
    q_neg: np.ndarray,
    quantiles: list[float],
    mu_pos: float,
    mu_neg: float,
    sep: float,
    auroc: float,
    thresholds: list[dict[str, float]],
    top_bands: list[dict[str, float | int | str]],
    near_zero_share_all: float,
    near_zero_share_pos: float,
    near_zero_share_neg: float,
    extreme_share_all: float,
    extreme_share_pos: float,
    extreme_share_neg: float,
    fig_path: Path,
    tp_fig: Path,
    fp_fig: Path,
    tp_examples: list[dict[str, int | float | str]],
    fp_examples: list[dict[str, int | float | str]],
) -> str:
    baseline_rate = float(labels_pf.mean())
    delta_corr = float(np.corrcoef(delta_pf, labels_pf.astype(np.float32))[0, 1])
    fig_rel = f"figs/{fig_path.name}"
    tp_rel = f"figs/{tp_fig.name}"
    fp_rel = f"figs/{fp_fig.name}"

    lines: list[str] = []
    lines.append("# P2 Probability Statistics")
    lines.append("")
    lines.append("Focus: cached `forest_prob_pre`, `forest_prob_post`, and especially `forest_prob_delta` on the 10 labeled training tiles.")
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append(f"- Labeled tiles analyzed: **{len(records)}**")
    lines.append(f"- Total labeled pixels: **{labels.size:,}**")
    lines.append(f"- Pre-forest analysis set (`prob_pre >= 0.5`): **{int(pre_forest.sum()):,}** pixels ({pre_forest.mean():.1%} of labeled pixels)")
    lines.append(f"- Deforestation prevalence within pre-forest set: **{baseline_rate:.3f}**")
    lines.append(f"- Mean delta on pre-forest pixels: **{mu_pos:+.4f}** for `label=1` vs **{mu_neg:+.4f}** for `label=0`")
    lines.append(f"- Separation `|Δμ| / σ_neg`: **{sep:.3f}**")
    lines.append(f"- AUROC for raw delta as a scorer: **{auroc:.3f}**")
    lines.append(f"- Pearson corr(delta, label) on pre-forest pixels: **{delta_corr:.3f}**")
    lines.append(f"- Near-zero delta share (`|delta| <= 0.02`): **{near_zero_share_all:.1%}** overall, **{near_zero_share_pos:.1%}** on `label=1`, **{near_zero_share_neg:.1%}** on `label=0`")
    lines.append(f"- Extreme positive tail (`delta >= 0.95`): **{extreme_share_all:.1%}** overall, **{extreme_share_pos:.1%}** on `label=1`, **{extreme_share_neg:.1%}** on `label=0`")
    lines.append("")
    lines.append("## Distribution notes")
    lines.append("")
    lines.append(f"- Overall delta distribution is right-shifted on pre-forest pixels: median **{q_all[4]:+.3f}**, 90th percentile **{q_all[6]:+.3f}**, 99th percentile **{q_all[8]:+.3f}**.")
    lines.append(f"- Deforested pixels have a heavier positive tail: 90th percentile **{q_pos[6]:+.3f}** vs **{q_neg[6]:+.3f}** for stable pixels.")
    lines.append(f"- The negative-class median is still positive (**{q_neg[4]:+.3f}**), which matches the earlier observation that post-period drift is noisy even without true deforestation.")
    lines.append("- The distribution piles up very close to `0`, while a smaller but important tail jumps close to `1`. That is why the top 1% behaves worse than the top 5-10%: the very highest scores are often saturated false alarms.")
    lines.append("")
    lines.append("## Threshold view")
    lines.append("")
    lines.append("| delta threshold | precision | recall | f1 | predicted share |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in thresholds:
        lines.append(
            f"| `{row['threshold']:+.1f}` | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['predicted_share']:.3f} |"
        )
    lines.append("")
    lines.append("## High-delta tail")
    lines.append("")
    lines.append("| slice | cutoff | pixels | positive rate | lift vs baseline |")
    lines.append("|---|---:|---:|---:|---:|")
    for band in top_bands:
        lift = float(band["positive_rate"]) / max(baseline_rate, 1e-8)
        lines.append(
            f"| {band['band']} | {float(band['cutoff']):+.3f} | {int(band['count']):,} | {float(band['positive_rate']):.3f} | {lift:.2f}x |"
        )
    lines.append("")
    lines.append("## Quantiles")
    lines.append("")
    lines.append("| quantile | all pre-forest | label=1 | label=0 |")
    lines.append("|---|---:|---:|---:|")
    for q, qa, qp, qn in zip(quantiles, q_all, q_pos, q_neg):
        lines.append(f"| {q:.2f} | {qa:+.3f} | {qp:+.3f} | {qn:+.3f} |")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append(f"![P2 delta insights]({fig_rel})")
    lines.append("")
    lines.append(f"![High-delta true positives]({tp_rel})")
    lines.append("")
    lines.append(f"![High-delta false positives]({fp_rel})")
    lines.append("")
    lines.append("## Example callouts")
    lines.append("")
    if tp_examples:
        first_tp = tp_examples[0]
        lines.append(
            f"- Strongest true-positive example in this pass: tile `{first_tp['tile_id']}` at `(y={first_tp['y']}, x={first_tp['x']})`, delta **{float(first_tp['delta']):.3f}**."
        )
    if fp_examples:
        first_fp = fp_examples[0]
        lines.append(
            f"- Strongest false-positive example in this pass: tile `{first_fp['tile_id']}` at `(y={first_fp['y']}, x={first_fp['x']})`, delta **{float(first_fp['delta']):.3f}**."
        )
    lines.append("- The verification panels are meant to answer a simple question: when delta spikes, do we see a localized drop from `prob_pre` to `prob_post`, and does that overlap the GT deforestation mask or not?")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
