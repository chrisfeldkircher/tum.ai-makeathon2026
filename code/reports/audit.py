"""P0 baseline audit — stats, sanity assertions, and diagnostic visualizations.

Reads cached `.npz` tiles, prints a data inventory, emits `data_stats.md`,
saves four diagnostic figures under `figs/`, and verifies the current
LearnedForestMasker reproduces the README's aggregate 0.982 / 0.775.

Run once `cache/*.npz` has been unzipped (from repo root):
    python code/reports/audit.py --cache_dir ./cache

NOTE: we cannot use `python -m code.reports.audit` because Python's stdlib
ships a `code` module that shadows this repo's top-level `code/` package
name. We instead push `<repo>/code` onto sys.path and import siblings
(`model.vegetationPredictor`) flat.

Intentional design choices:
 - Fails loudly if cache is missing or a required key is absent.
 - Prints everything it writes to disk, so running from a notebook is
   just as informative as reading the markdown.
 - Does not mutate the cache or model code; read-only audit.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
# Work around stdlib `code` module shadowing the repo's `code/` package.
_CODE_DIR = str(REPO_ROOT / "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
REPORT_DIR = Path(__file__).resolve().parent
FIG_DIR = REPORT_DIR / "figs"
STATS_MD = REPORT_DIR / "data_stats.md"

REQUIRED_KEYS = (
    "s2_pre_ndvi_median", "s2_pre_ndvi_std",
    "s2_pre_nbr_median",
    "aef_pre",
    "forest_gt_pre2020",
    "forest_mask_2020",
)

# Keys we sample for per-key stats on a representative tile.
INSPECT_KEYS = (
    "aef_pre", "aef_post", "aef_delta",
    "s2_pre_ndvi_median", "s2_pre_ndvi_std",
    "s2_pre_nbr_median",  "s2_pre_nbr_std",
    "s2_pre_band_median", "s2_post_band_median",
    "s1_pre_vv_median",   "s1_pre_vv_std",
    "forest_mask_2020", "forest_gt_pre2020",
    "label", "label_confidence",
)

TILE_ID_RE = re.compile(r"^(\d{2}[A-Z])([A-Z]{2})_(\d+)_(\d+)$")


def mgrs_zone(tile_id: str) -> str:
    m = TILE_ID_RE.match(tile_id)
    return m.group(1) if m else "??"  # e.g. "48P" from "48PUT_0_8"


def load_tile(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def _fmt_num(x: float) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "nan"
    if abs(x) >= 1e4 or (0 < abs(x) < 1e-3):
        return f"{x:.3e}"
    return f"{x:.4f}"


# ---------------------------------------------------------------------------
# Stats collection
# ---------------------------------------------------------------------------

def collect_tile_stats(paths: list[Path]) -> dict:
    """Per-tile top-level stats + cross-tile aggregates.

    Note: test tiles (no forest_gt_pre2020) are skipped silently — they have no labels.
    """
    per_tile = []
    shapes = Counter()
    zones = Counter()
    skipped_test_tiles = []
    missing_keys_tiles = []
    aef_zero_group_tiles = []

    for p in paths:
        t = load_tile(p)

        # Skip test tiles (no ground truth)
        if "forest_gt_pre2020" not in t:
            skipped_test_tiles.append(p.stem)
            continue

        missing = [k for k in REQUIRED_KEYS if k not in t]
        if missing:
            missing_keys_tiles.append((p.stem, missing))
            continue

        H, W = t["s2_pre_ndvi_median"].shape
        shapes[(H, W)] += 1
        zones[mgrs_zone(p.stem)] += 1

        gt = t["forest_gt_pre2020"].astype(bool)
        pix = gt.size
        gt_frac = float(gt.mean())

        row = {
            "tile_id": p.stem,
            "mgrs_zone": mgrs_zone(p.stem),
            "H": int(H), "W": int(W), "pixels": int(pix),
            "forest_gt_frac": gt_frac,
            "forest_gt_pixels": int(gt.sum()),
        }

        if "label" in t:
            lbl = t["label"].astype(bool)
            row["label_frac"] = float(lbl.mean())
            row["label_pixels"] = int(lbl.sum())
        if "label_confidence" in t:
            lc = t["label_confidence"]
            row["label_conf_mean"] = float(lc[lc > 0].mean()) if (lc > 0).any() else 0.0
            row["label_conf_max"] = float(lc.max())
        if "forest_mask_2020" in t:
            fm = t["forest_mask_2020"].astype(bool)
            row["forest_mask_2020_frac"] = float(fm.mean())

        # AEF zero-group sanity (workshop cell 7): any (64, H, W) group all-zero
        # indicates dequantization-missing pixels.
        aef = t["aef_pre"]
        if aef.ndim == 3 and aef.shape[0] == 64:
            zero_px = np.all(aef == 0, axis=0).sum()
            row["aef_zero_group_frac"] = float(zero_px) / (aef.shape[1] * aef.shape[2])
            if row["aef_zero_group_frac"] > 0.001:
                aef_zero_group_tiles.append((p.stem, row["aef_zero_group_frac"]))

        per_tile.append(row)

    return {
        "per_tile": per_tile,
        "shapes": dict(shapes),
        "zones": dict(zones),
        "skipped_test_tiles": skipped_test_tiles,
        "missing_keys_tiles": missing_keys_tiles,
        "aef_zero_group_tiles": aef_zero_group_tiles,
    }


def collect_key_inspection(path: Path) -> list[dict]:
    """Shape/dtype/NaN/min/mean/max on a representative tile."""
    t = load_tile(path)
    rows = []
    for k in INSPECT_KEYS:
        if k not in t:
            rows.append({"key": k, "present": False})
            continue
        a = t[k]
        nan_frac = float(np.isnan(a).mean()) if a.dtype.kind == "f" else 0.0
        finite = a[np.isfinite(a)] if a.dtype.kind == "f" else a
        rows.append({
            "key": k, "present": True,
            "shape": tuple(int(s) for s in a.shape),
            "dtype": str(a.dtype),
            "nan_frac": nan_frac,
            "min": float(finite.min()) if finite.size else float("nan"),
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "max": float(finite.max()) if finite.size else float("nan"),
        })
    return rows


def collect_aef_dim_stats(paths: list[Path], max_tiles: int = 10) -> np.ndarray:
    """Aggregate per-dim AEF stats across tiles. Returns (64, 4): mean, std, min, max."""
    means, stds, mins, maxs = [], [], [], []
    for p in paths[:max_tiles]:
        t = load_tile(p)
        aef = t.get("aef_pre")
        if aef is None or aef.ndim != 3 or aef.shape[0] != 64:
            continue
        flat = aef.reshape(64, -1).astype(np.float32)
        means.append(flat.mean(axis=1))
        stds.append(flat.std(axis=1))
        mins.append(flat.min(axis=1))
        maxs.append(flat.max(axis=1))
    if not means:
        return np.zeros((64, 4))
    return np.stack([
        np.mean(means, axis=0),
        np.mean(stds,  axis=0),
        np.min(mins,   axis=0),
        np.max(maxs,   axis=0),
    ], axis=1)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def viz_tile_overview(path: Path, masker, out_path: Path) -> None:
    """4-panel: S2 RGB (from bands 2/3/4) / NDVI_pre / forest_gt_pre2020 / predicted mask."""
    import matplotlib.pyplot as plt
    t = load_tile(path)
    ndvi = t["s2_pre_ndvi_median"]
    gt = t["forest_gt_pre2020"]
    pred = masker.predict(t) if masker is not None else np.zeros_like(gt)

    # B04=Red (index 3), B03=Green (index 2), B02=Blue (index 1). Guard in case keys are absent.
    rgb = None
    if "s2_pre_band_median" in t and t["s2_pre_band_median"].shape[0] >= 4:
        bands = t["s2_pre_band_median"]
        r = bands[3]; g = bands[2]; b = bands[1]
        rgb = np.stack([r, g, b], axis=-1)
        # Contrast stretch to 2–98 percentile for visibility
        lo, hi = np.nanpercentile(rgb, [2, 98])
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    if rgb is not None:
        axs[0].imshow(rgb); axs[0].set_title(f"{path.stem}  S2 RGB (pre median)")
    else:
        axs[0].text(0.5, 0.5, "no S2 band median", ha="center"); axs[0].set_title(path.stem)
    axs[1].imshow(ndvi, vmin=-0.2, vmax=0.9, cmap="RdYlGn"); axs[1].set_title("NDVI pre (median)")
    axs[2].imshow(gt, cmap="gray", vmin=0, vmax=1); axs[2].set_title(f"forest_gt_pre2020 ({gt.mean():.2%})")
    axs[3].imshow(pred, cmap="gray", vmin=0, vmax=1); axs[3].set_title(f"masker.predict ({pred.mean():.2%})")
    for ax in axs: ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def viz_gt_hist(per_tile: list[dict], out_path: Path) -> None:
    """Bar chart of forest_gt_pre2020 pixel count per tile, sorted ascending."""
    import matplotlib.pyplot as plt
    rows = sorted(per_tile, key=lambda r: r["forest_gt_pixels"])
    ids = [r["tile_id"] for r in rows]
    vals = [r["forest_gt_pixels"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, len(ids) * 0.6), 4))
    ax.bar(range(len(ids)), vals, color="#3b7a57")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("forest_gt_pre2020 pixels")
    ax.set_title("GT pixel count per tile (ascending) — imbalance check")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def viz_aef_pca(path: Path, out_path: Path, n_samples: int = 10_000) -> None:
    """2D PCA of per-pixel AEF_pre, coloured by forest_gt_pre2020."""
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    t = load_tile(path)
    aef = t["aef_pre"]  # (64, H, W)
    gt = t["forest_gt_pre2020"].astype(bool)
    X = aef.reshape(64, -1).T  # (H*W, 64)
    y = gt.reshape(-1)
    pos_idx = np.where(y)[0]
    neg_idx = np.where(~y)[0]
    n_pos = min(n_samples // 2, len(pos_idx))
    n_neg = min(n_samples // 2, len(neg_idx))
    rng = np.random.default_rng(0)
    idx = np.concatenate([
        rng.choice(pos_idx, n_pos, replace=False) if n_pos else np.array([], int),
        rng.choice(neg_idx, n_neg, replace=False) if n_neg else np.array([], int),
    ])
    Xs = X[idx]
    ys = y[idx]
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(n_components=2, random_state=0).fit(Xs)
    Z = pca.transform(Xs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(Z[~ys, 0], Z[~ys, 1], s=2, alpha=0.3, c="#888", label="non-forest_gt")
    ax.scatter(Z[ys, 0], Z[ys, 1], s=2, alpha=0.5, c="#2b8a3e", label="forest_gt")
    var = pca.explained_variance_ratio_
    ax.set_title(f"{path.stem} — AEF_pre PCA (var explained: {var[0]:.2f}+{var[1]:.2f}={var.sum():.2f})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(markerscale=4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def viz_prob_hist(path: Path, masker, out_path: Path) -> None:
    """Histogram of masker probability, split by forest_gt_pre2020 — calibration sanity."""
    import matplotlib.pyplot as plt
    t = load_tile(path)
    prob = masker.predict_proba(t)
    gt = t["forest_gt_pre2020"].astype(bool)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(prob[~gt].ravel(), bins=bins, alpha=0.5, label=f"non-GT (n={(~gt).sum():,})", color="#888", density=True)
    ax.hist(prob[gt].ravel(),  bins=bins, alpha=0.5, label=f"GT forest (n={gt.sum():,})",   color="#2b8a3e", density=True)
    ax.axvline(0.30, linestyle="--", color="#666", linewidth=0.8)
    ax.axvline(0.60, linestyle="--", color="#666", linewidth=0.8)
    ax.axvline(0.85, linestyle="--", color="#666", linewidth=0.8)
    ax.set_xlabel("forest probability")
    ax.set_ylabel("density")
    ax.set_title(f"{path.stem} — LearnedForestMasker probability (tier cuts at 0.30/0.60/0.85)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_stats_md(out_path: Path,
                   stats: dict,
                   key_rows: list[dict],
                   aef_dim_stats: np.ndarray,
                   masker_agg: dict | None,
                   masker_per_tile: list[tuple[str, dict]] | None) -> None:
    per_tile = stats["per_tile"]
    n_tiles = len(per_tile)
    total_pix = sum(r["pixels"] for r in per_tile)
    total_gt = sum(r["forest_gt_pixels"] for r in per_tile)
    total_label = sum(r.get("label_pixels", 0) for r in per_tile)
    shapes = stats["shapes"]
    zones = stats["zones"]

    lines: list[str] = []
    lines.append("# P0 Baseline Audit — data_stats.md")
    lines.append("")
    lines.append(f"**Tiles:** {n_tiles} · **Total pixels:** {total_pix:,}")
    lines.append(f"**forest_gt_pre2020 pixels:** {total_gt:,}  ({total_gt/max(total_pix,1):.3%} of total)")
    lines.append(f"**label (deforestation) pixels:** {total_label:,}  ({total_label/max(total_pix,1):.3%} of total)")
    lines.append("")

    lines.append("## Tile shape distribution")
    for sh, n in sorted(shapes.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {sh[0]}×{sh[1]}  ({n} tiles)")
    lines.append("")

    lines.append("## MGRS-zone distribution")
    for z, n in sorted(zones.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{z}`  ({n} tiles)")
    lines.append("")

    lines.append("## Per-tile summary")
    lines.append("")
    lines.append("| tile_id | zone | H×W | forest_gt% | label% | conf(mean) | forest_mask_2020% | aef_zero_group% |")
    lines.append("|---------|------|-----|------------|--------|------------|-------------------|-----------------|")
    for r in sorted(per_tile, key=lambda x: x["tile_id"]):
        lines.append(
            f"| `{r['tile_id']}` | {r['mgrs_zone']} | {r['H']}×{r['W']} "
            f"| {r['forest_gt_frac']:.2%} "
            f"| {r.get('label_frac', float('nan')):.2%} "
            f"| {r.get('label_conf_mean', float('nan')):.3f} "
            f"| {r.get('forest_mask_2020_frac', float('nan')):.2%} "
            f"| {r.get('aef_zero_group_frac', 0.0):.3%} |"
        )
    lines.append("")

    lines.append("## Representative tile — per-key inspection")
    lines.append("")
    lines.append("| key | present | shape | dtype | nan% | min | mean | max |")
    lines.append("|-----|---------|-------|-------|------|-----|------|-----|")
    for row in key_rows:
        if not row["present"]:
            lines.append(f"| `{row['key']}` | ❌ | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| `{row['key']}` | ✓ | {row['shape']} | {row['dtype']} "
            f"| {row['nan_frac']:.2%} | {_fmt_num(row['min'])} "
            f"| {_fmt_num(row['mean'])} | {_fmt_num(row['max'])} |"
        )
    lines.append("")

    lines.append("## AEF dimension aggregate (across tiles)")
    lines.append("Columns: mean, std, min, max aggregated across tiles.")
    lines.append("")
    lines.append("| dim (1-based) | mean | std | min | max |")
    lines.append("|---|------|-----|-----|-----|")
    for i, row in enumerate(aef_dim_stats):
        flag = ""
        if i + 1 in (18, 21, 26, 28, 34):  # paper-identified tree dims
            flag = "  🌲"
        lines.append(f"| A{i+1:02d}{flag} | {_fmt_num(row[0])} | {_fmt_num(row[1])} | {_fmt_num(row[2])} | {_fmt_num(row[3])} |")
    lines.append("")
    lines.append("🌲 = paper-identified tree-cover dims (A18, A21, A26 exclusive; A28, A34 shared).")
    lines.append("")

    lines.append("## Sanity checks")
    lines.append("")
    if stats["skipped_test_tiles"]:
        lines.append(f"ℹ️ **Test tiles (skipped, no ground truth):** {len(stats['skipped_test_tiles'])} tiles")
        for tid in stats["skipped_test_tiles"]:
            lines.append(f"  - `{tid}`")
        lines.append("")
    if stats["missing_keys_tiles"]:
        lines.append("❌ **Tiles missing required keys:**")
        for tile, keys in stats["missing_keys_tiles"]:
            lines.append(f"  - `{tile}`: missing {keys}")
    else:
        lines.append("✓ All training tiles have required keys " + ", ".join(f"`{k}`" for k in REQUIRED_KEYS))
    if stats["aef_zero_group_tiles"]:
        lines.append("⚠️ **AEF all-zero pixel groups > 0.1% in:**")
        for tile, frac in stats["aef_zero_group_tiles"]:
            lines.append(f"  - `{tile}`: {frac:.3%}")
    else:
        lines.append("✓ No AEF dequantization issues detected (no tile has > 0.1% all-zero pixel groups).")
    empty_gt = [r["tile_id"] for r in per_tile if r["forest_gt_pixels"] == 0]
    if empty_gt:
        lines.append(f"❌ **Empty forest_gt_pre2020 in:** {empty_gt}")
    else:
        lines.append("✓ All tiles have non-empty forest_gt_pre2020.")
    lines.append("")

    if masker_agg is not None:
        lines.append("## LearnedForestMasker reproduction check")
        lines.append("")
        lines.append(f"- Aggregate **recall_vs_alerts**: `{masker_agg['recall_vs_alerts']:.4f}` "
                     f"(README baseline: 0.982)")
        lines.append(f"- Aggregate **mask_coverage**:  `{masker_agg['mask_coverage']:.4f}` "
                     f"(README baseline: 0.775)")
        lines.append(f"- N tiles evaluated: {masker_agg['n_tiles']}")
        lines.append("")
        if masker_per_tile is not None:
            lines.append("### Per-tile recall / coverage (sorted ascending by recall)")
            lines.append("")
            lines.append("| tile_id | recall | coverage | GT pixels |")
            lines.append("|---------|--------|----------|-----------|")
            for tile_id, d in sorted(masker_per_tile, key=lambda kv: kv[1]["recall_vs_alerts"]):
                lines.append(
                    f"| `{tile_id}` | {d['recall_vs_alerts']:.3f} "
                    f"| {d['mask_coverage']:.3f} "
                    f"| {d['n_alert_pixels']:,} |"
                )
            lines.append("")

    lines.append("## Figures")
    lines.append("")
    lines.append(f"- `{FIG_DIR.relative_to(REPO_ROOT).as_posix()}/p0_overview_*.png` — 4-panel tile overview (S2 RGB / NDVI / GT / predicted mask)")
    lines.append(f"- `{FIG_DIR.relative_to(REPO_ROOT).as_posix()}/p0_gt_pixel_count.png` — GT pixel count per tile")
    lines.append(f"- `{FIG_DIR.relative_to(REPO_ROOT).as_posix()}/p0_aef_pca_*.png` — AEF_pre PCA coloured by forest_gt_pre2020")
    lines.append(f"- `{FIG_DIR.relative_to(REPO_ROOT).as_posix()}/p0_prob_hist_*.png` — masker probability histogram, split by GT")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache_dir", default=str(REPO_ROOT / "cache"),
                    help="Directory with *.npz cached tiles.")
    ap.add_argument("--viz_tile", default=None,
                    help="Tile id (e.g. 48PUT_0_8) to use for prob-hist viz. "
                         "Defaults to the tile with the highest forest_gt_pre2020 fraction.")
    ap.add_argument("--skip_masker", action="store_true",
                    help="Skip the LearnedForestMasker fit+eval step (faster, stats-only).")
    ap.add_argument("--skip_figs", action="store_true",
                    help="Skip figure generation (matplotlib). Use for speed when just checking stats.")
    ap.add_argument("--backend", default="lightgbm", choices=["lightgbm", "logreg"])
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"[FATAL] cache dir does not exist: {cache_dir}", file=sys.stderr)
        return 2
    paths = sorted(cache_dir.glob("*.npz"))
    if not paths:
        print(f"[FATAL] no *.npz in {cache_dir} — is the cache.zip unpacked?", file=sys.stderr)
        return 2

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[P0] audit on {len(paths)} tiles (includes test tiles without ground truth) from {cache_dir}")
    stats = collect_tile_stats(paths)
    per_tile = stats["per_tile"]
    print(f"[P0] per-tile stats collected · shapes={stats['shapes']} · zones={stats['zones']}")

    # Representative tile: the one with most forest_gt pixels
    rep_row = max(per_tile, key=lambda r: r["forest_gt_pixels"]) if per_tile else None
    if rep_row is None:
        print("[FATAL] no valid tiles (all missing required keys?)", file=sys.stderr)
        return 2
    rep_path = cache_dir / f"{rep_row['tile_id']}.npz"
    print(f"[P0] representative tile: {rep_row['tile_id']}  (GT={rep_row['forest_gt_frac']:.2%})")

    key_rows = collect_key_inspection(rep_path)
    aef_dim_stats = collect_aef_dim_stats(paths)
    print(f"[P0] per-dim AEF stats aggregated across {min(len(paths), 10)} tiles")

    masker = None
    masker_agg = None
    masker_per_tile = None
    if not args.skip_masker:
        print(f"[P0] training LearnedForestMasker ({args.backend}) on all cached tiles …")
        from model.vegetationPredictor import (  # type: ignore
            LearnedForestMasker, evaluate_masker_on_tiles,
        )
        tensors_list = [load_tile(p) for p in paths]
        masker = LearnedForestMasker(backend=args.backend).fit(tensors_list)
        result = evaluate_masker_on_tiles(masker, paths)
        masker_agg = result["aggregate"]
        masker_per_tile = list(result["per_tile"].items())
        print(f"[P0] aggregate: recall={masker_agg['recall_vs_alerts']:.4f} "
              f"coverage={masker_agg['mask_coverage']:.4f}  (README baseline 0.982/0.775)")

    # Pick viz tile: user-provided or the most-GT one
    viz_tile_id = args.viz_tile or rep_row["tile_id"]
    viz_path = cache_dir / f"{viz_tile_id}.npz"
    if not viz_path.exists():
        print(f"[warn] viz_tile {viz_tile_id} not found — falling back to {rep_row['tile_id']}")
        viz_path = rep_path
        viz_tile_id = rep_row["tile_id"]

    # Figures (skip PCA fit for speed in hackathon; focus on actionable viz)
    if not args.skip_figs:
        print(f"[P0] writing figures to {FIG_DIR}")
        # 4-panel overview on 2 tiles: the rep + the tile with smallest GT (edge case)
        overview_targets = [rep_row["tile_id"]]
        small_gt = min(per_tile, key=lambda r: r["forest_gt_pixels"])
        if small_gt["tile_id"] not in overview_targets:
            overview_targets.append(small_gt["tile_id"])
        for tid in overview_targets:
            viz_tile_overview(cache_dir / f"{tid}.npz", masker, FIG_DIR / f"p0_overview_{tid}.png")
        viz_gt_hist(per_tile, FIG_DIR / "p0_gt_pixel_count.png")
        if masker is not None:
            viz_prob_hist(viz_path, masker, FIG_DIR / f"p0_prob_hist_{viz_tile_id}.png")
    else:
        print("[P0] skipping figures (--skip_figs)")

    write_stats_md(STATS_MD, stats, key_rows, aef_dim_stats, masker_agg, masker_per_tile)
    print(f"[P0] wrote report -> {STATS_MD.relative_to(REPO_ROOT)}")

    # Final sanity gates — printed, not fatal, but visible
    red = []
    if stats["missing_keys_tiles"]:
        red.append(f"missing-keys tiles (training only): {[t for t,_ in stats['missing_keys_tiles']]}")
    empty_gt = [r["tile_id"] for r in per_tile if r["forest_gt_pixels"] == 0]
    if empty_gt:
        red.append(f"empty forest_gt tiles (training only): {empty_gt}")
    if masker_agg is not None:
        rdiff = abs(masker_agg["recall_vs_alerts"] - 0.982)
        cdiff = abs(masker_agg["mask_coverage"] - 0.775)
        if rdiff > 0.01 or cdiff > 0.02:
            red.append(
                f"baseline drift: recall Δ={rdiff:.3f} (>0.01) or coverage Δ={cdiff:.3f} (>0.02) "
                f"vs README (0.982/0.775)"
            )
    if red:
        print("\n[P0] RED FLAGS:")
        for r in red:
            print(f"  [FAIL] {r}")
    else:
        print("\n[P0] [PASS] all sanity checks passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
