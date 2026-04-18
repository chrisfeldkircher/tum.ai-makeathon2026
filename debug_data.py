"""Standalone raw-data sanity check.

Walks the on-disk tile layout, counts what's present per modality, measures
S2 scene sizes (since that's what drives the reference grid), and predicts
how `preprocess_tile` will behave on each tile under the new logic.

Run:  python debug_data.py
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio

DATA_ROOT = Path(__file__).parent / "data" / "makeathon-challenge"

MIN_REF_PX = 500_000      # tile is unusable below this
PARTIAL_COVERAGE = 0.8    # scenes below this fraction of ref get filtered

# Mirror `data.py::BROKEN_TILES`. Kept in sync manually.
BROKEN_TILES = frozenset({"18NYH_9_9"})

TILE_ID_RE = re.compile(r"^[A-Z0-9]+_\d+_\d+$")
_S2_FILE_RE  = re.compile(r".+_s2_l2a_(\d{4})_(\d{1,2})\.tif$")
_S1_FILE_RE  = re.compile(r".+_s1_rtc_(\d{4})_(\d{1,2})_(ascending|descending)\.tif$")
_AEF_FILE_RE = re.compile(r"^(.+)_(\d{4})\.tiff?$")


def _size(p: Path) -> int:
    with rasterio.open(p) as src:
        return int(src.width) * int(src.height)


def scan_split(split: str) -> dict:
    tiles: dict[str, dict] = {}

    s2_root = DATA_ROOT / "sentinel-2" / split
    if s2_root.exists():
        for tdir in s2_root.iterdir():
            if not tdir.is_dir():
                continue
            tid = tdir.name.replace("__s2_l2a", "")
            if not TILE_ID_RE.match(tid):
                continue
            scenes = []
            for f in tdir.glob("*.tif"):
                m = _S2_FILE_RE.match(f.name)
                if m:
                    scenes.append(((int(m.group(1)), int(m.group(2))), f))
            tiles.setdefault(tid, {})["s2"] = scenes

    s1_root = DATA_ROOT / "sentinel-1" / split
    if s1_root.exists():
        for tdir in s1_root.iterdir():
            if not tdir.is_dir():
                continue
            tid = tdir.name.replace("__s1_rtc", "")
            if not TILE_ID_RE.match(tid):
                continue
            scenes = []
            for f in tdir.glob("*.tif"):
                m = _S1_FILE_RE.match(f.name)
                if m:
                    scenes.append(((int(m.group(1)), int(m.group(2)), m.group(3)), f))
            tiles.setdefault(tid, {})["s1"] = scenes

    aef_root = DATA_ROOT / "aef-embeddings" / split
    if aef_root.exists():
        for f in aef_root.glob("*.tif*"):
            m = _AEF_FILE_RE.match(f.name)
            if not m:
                continue
            tid, year = m.group(1), int(m.group(2))
            if not TILE_ID_RE.match(tid):
                continue
            tiles.setdefault(tid, {}).setdefault("aef", []).append((year, f))

    if split == "train":
        lbl = DATA_ROOT / "labels" / "train"
        for tid in tiles:
            tiles[tid]["labels"] = {
                "radd":   bool(list((lbl / "radd").glob(f"radd_{tid}_labels.tif"))),
                "gladl":  bool(list((lbl / "gladl").glob(f"gladl_{tid}_alert*.tif"))),
                "glads2": bool(list((lbl / "glads2").glob(f"glads2_{tid}_alert*.tif"))),
            }

    for tid in tiles:
        tiles[tid]["split"] = split
    return tiles


def predict_outcome(info: dict, tile_id: str | None = None) -> tuple[str, str]:
    if tile_id in BROKEN_TILES:
        return "SKIP", "listed in BROKEN_TILES"
    s2 = info.get("s2", [])
    if not s2:
        return "SKIP", "no S2 scenes"
    sizes = [(_size(p), (y, m), p) for ((y, m), p) in s2]
    max_sz = max(s for s, _, _ in sizes)
    if max_sz < MIN_REF_PX:
        return "SKIP", f"largest scene only {max_sz} px"
    kept = sum(1 for s, _, _ in sizes if s >= PARTIAL_COVERAGE * max_sz)
    dropped = len(sizes) - kept
    note = f"ref={max_sz}px, {kept}/{len(sizes)} scenes kept"
    if dropped:
        note += f" ({dropped} partial dropped)"
    return "OK", note


def _read_rgb(p: Path, stretch: float = 3.0) -> tuple[np.ndarray, int, int, dict]:
    """Return (HxWx3 RGB in [0,1], width, height, stats) from an S2 L2A tif."""
    with rasterio.open(p) as src:
        w, h = int(src.width), int(src.height)
        r = src.read(4).astype(np.float32)
        g = src.read(3).astype(np.float32)
        b = src.read(2).astype(np.float32)
    stack = np.stack([r, g, b], axis=-1)
    n_total = stack.size
    n_zero = int((stack == 0).all(axis=-1).sum()) * 3
    n_nan = int(np.isnan(stack).sum())
    stats = {
        "pct_zero": 100.0 * n_zero / n_total,
        "pct_nan":  100.0 * n_nan / n_total,
        "p50":      float(np.nanmedian(stack)),
        "p99":      float(np.nanpercentile(stack, 99)) if n_nan < n_total else float("nan"),
    }
    rgb = np.clip(stack / 10000.0 * stretch, 0.0, 1.0)
    return rgb, w, h, stats


def plot_problem_tiles(tiles: dict, out_dir: Path) -> None:
    """For every tile flagged SKIP or with partial scenes, plot one RGB per
    distinct size class so we can eyeball what the partial scenes actually are."""
    import matplotlib.pyplot as plt

    problem: list[tuple[str, list, int]] = []
    for tid in sorted(tiles):
        s2 = tiles[tid].get("s2", [])
        if not s2:
            continue
        sizes = [(_size(p), ym, p) for (ym, p) in s2]
        max_sz = max(s for s, _, _ in sizes)
        has_partial = any(s < PARTIAL_COVERAGE * max_sz for s in (s for s, _, _ in sizes))
        if max_sz < MIN_REF_PX or has_partial:
            problem.append((tid, sizes, max_sz))

    if not problem:
        print("\nNo problem tiles to plot.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nPlotting {len(problem)} problem tile(s) → {out_dir}/")
    for tid, sizes, max_sz in problem:
        # One representative scene per distinct size class (pick earliest date).
        reps: dict[int, tuple[tuple[int, int], Path]] = {}
        for sz, ym, p in sizes:
            if sz not in reps or ym < reps[sz][0]:
                reps[sz] = (ym, p)
        ordered = sorted(reps.items(), key=lambda kv: -kv[0])  # largest first

        n = len(ordered)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2), squeeze=False)
        for ax, (sz, (ym, p)) in zip(axes[0], ordered):
            rgb, w, h, stats = _read_rgb(p)
            ax.imshow(rgb, interpolation="nearest")
            count = sum(1 for s, _, _ in sizes if s == sz)
            flag = "REF" if sz == max_sz else "partial"
            frac = sz / max_sz
            ax.set_title(
                f"{ym[0]}-{ym[1]:02d}   {w}x{h}\n"
                f"{sz:,} px  ({frac*100:.1f}%)  [{flag}]  ×{count} scene(s)\n"
                f"zero={stats['pct_zero']:.0f}%  nan={stats['pct_nan']:.0f}%  "
                f"p50={stats['p50']:.0f}  p99={stats['p99']:.0f}",
                fontsize=8,
            )
            ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle(f"{tid} — {n} distinct scene sizes ({len(sizes)} total scenes)",
                     fontsize=11, y=1.02)
        fig.tight_layout()
        out_path = out_dir / f"debug_tile_{tid}.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path.name}")


def main() -> None:
    tiles: dict[str, dict] = {}
    for split in ("train", "test"):
        for tid, info in scan_split(split).items():
            tiles[tid] = info

    print(f"{'tile':<14}{'split':>6}{'S2':>5}{'S1':>5}{'AEF':>5}"
          f"{'labels':>9}  outcome")
    print("-" * 90)

    n_ok = n_skip = 0
    for tid in sorted(tiles):
        info = tiles[tid]
        split = info["split"]
        n_s2  = len(info.get("s2", []))
        n_s1  = len(info.get("s1", []))
        n_aef = len(info.get("aef", []))
        lbls  = info.get("labels", {}) if split == "train" else {}
        if split == "train":
            lbl_flags = (("R" if lbls.get("radd") else "-") +
                         ("L" if lbls.get("gladl") else "-") +
                         ("S" if lbls.get("glads2") else "-"))
        else:
            lbl_flags = "   "
        outcome, note = predict_outcome(info, tid)
        print(f"{tid:<14}{split:>6}{n_s2:>5}{n_s1:>5}{n_aef:>5}"
              f"{lbl_flags:>9}  {outcome:<4} {note}")
        if outcome == "OK":
            n_ok += 1
        else:
            n_skip += 1

    print("-" * 90)
    print(f"summary: {n_ok} OK, {n_skip} SKIP, {len(tiles)} total")

    plot_problem_tiles(tiles, out_dir=Path(__file__).parent / "debug_plots")

    # Scene-size distribution for the problem tiles
    print("\nSize histogram for tiles flagged SKIP or with partial scenes:")
    for tid in sorted(tiles):
        info = tiles[tid]
        s2 = info.get("s2", [])
        if not s2:
            continue
        sizes = [_size(p) for (_ym, p) in s2]
        max_sz = max(sizes)
        if max_sz < MIN_REF_PX or any(s < PARTIAL_COVERAGE * max_sz for s in sizes):
            hist = Counter(sizes)
            print(f"  {tid}: max={max_sz}px")
            for sz, cnt in sorted(hist.items(), key=lambda x: -x[1]):
                frac = sz / max_sz
                mark = " (partial)" if frac < PARTIAL_COVERAGE else ""
                print(f"      {sz:>9} px  x {cnt:>2}  ({frac*100:5.1f}%){mark}")


if __name__ == "__main__":
    main()
