from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import rasterio


REPO_ROOT = Path(__file__).resolve().parent
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one or more submissions from a saved checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "benchmark_first_weights.pt")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "makeathon-challenge")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "submission_baseline")
    parser.add_argument("--probs-dir", type=Path, default=REPO_ROOT / "cache_probs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area-ha", type=float, default=0.5)
    parser.add_argument("--thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--min-areas", nargs="+", type=float, default=None)
    parser.add_argument("--tile-ids", nargs="+", default=None)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--feature-set", choices=("default", "temporal"), default="default")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--strict-empty", action="store_true")
    parser.add_argument("--allow-missing-probs", action="store_true")
    parser.add_argument("--time-step", type=int, default=None)
    return parser.parse_args()


def resolve_device(device_name: str):
    import torch

    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_float(value: float) -> str:
    return str(value).replace(".", "p")


def resolve_tile_ids(data_root: Path, split: str, requested_tile_ids: list[str] | None, max_tiles: int | None) -> list[str] | None:
    from data import build_inventory

    if requested_tile_ids is not None:
        tile_ids = sorted(requested_tile_ids)
    elif max_tiles is not None:
        inventory = build_inventory(data_root, split)
        tile_ids = sorted(inventory)[:max_tiles]
    else:
        return None

    if max_tiles is not None:
        tile_ids = tile_ids[:max_tiles]
    return tile_ids


def maybe_prepare_prob_sidecars(
    data_root: Path,
    split: str,
    probs_dir: Path,
    tile_ids: list[str] | None,
    allow_missing_probs: bool,
    out_dir: Path,
) -> Path:
    from data import build_inventory

    if not allow_missing_probs:
        return probs_dir

    inventory = build_inventory(data_root, split)
    selected_ids = sorted(tile_ids or inventory.keys())
    fallback_dir = out_dir / "_probs_fallback"
    fallback_dir.mkdir(parents=True, exist_ok=True)

    for tile_id in selected_ids:
        source_path = probs_dir / f"{tile_id}.npz"
        target_path = fallback_dir / f"{tile_id}.npz"
        if source_path.exists():
            if not target_path.exists():
                target_path.write_bytes(source_path.read_bytes())
            continue

        tile_inventory = inventory.get(tile_id)
        if tile_inventory is None or not tile_inventory.s2_paths:
            continue
        ref_source = max(tile_inventory.s2_paths.values(), key=lambda path: _scene_pixels(path))
        with rasterio.open(ref_source) as src:
            height, width = src.height, src.width
        zeros = np.zeros((height, width), dtype=np.float32)
        np.savez_compressed(
            target_path,
            forest_prob_pre=zeros,
            forest_prob_post=zeros,
            forest_prob_delta=zeros,
        )
    return fallback_dir


def _scene_pixels(path: Path) -> int:
    with rasterio.open(path) as src:
        return src.width * src.height


def write_time_step(out_path: Path, time_step: int | None) -> int:
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    features = payload.get("features", [])
    if time_step is not None:
        for feature in features:
            feature.setdefault("properties", {})["time_step"] = int(time_step)
        out_path.write_text(json.dumps(payload), encoding="utf-8")
    return len(features)


def run_single_submission(
    args: argparse.Namespace,
    model,
    device,
    feature_keys,
    tile_ids: list[str] | None,
    threshold: float,
    min_area_ha: float,
    out_dir: Path,
) -> dict:
    from model.submit import build_submission

    out_dir.mkdir(parents=True, exist_ok=True)
    probs_dir = maybe_prepare_prob_sidecars(
        data_root=args.data_root,
        split=args.split,
        probs_dir=args.probs_dir,
        tile_ids=tile_ids,
        allow_missing_probs=args.allow_missing_probs,
        out_dir=out_dir,
    )

    build_submission_kwargs = {
        "data_root": args.data_root,
        "out_dir": out_dir,
        "split": args.split,
        "device": device,
        "threshold": threshold,
        "min_area_ha": min_area_ha,
        "tile_ids": tile_ids,
        "feature_keys": feature_keys,
    }
    if "probs_dir" in inspect.signature(build_submission).parameters:
        build_submission_kwargs["probs_dir"] = probs_dir

    out_path = build_submission(model, **build_submission_kwargs)
    n_features = write_time_step(out_path, args.time_step)
    tile_geojson_count = len(list((out_dir / "tiles").glob("*.geojson")))
    tile_tif_count = len(list((out_dir / "tiles").glob("*.tif")))

    if args.strict_empty and n_features == 0:
        raise RuntimeError(f"{out_path} was written but contains zero features.")

    return {
        "submission_path": str(out_path),
        "feature_count": n_features,
        "tile_geojson_count": tile_geojson_count,
        "tile_tif_count": tile_tif_count,
        "tile_ids": tile_ids,
        "feature_set": args.feature_set,
        "threshold": threshold,
        "min_area_ha": min_area_ha,
        "time_step": args.time_step,
    }


def main() -> None:
    args = parse_args()

    import torch

    from data.data import DEFAULT_FEATURE_KEYS, TEMPORAL_FEATURE_KEYS
    from model import build_model

    device = resolve_device(args.device)
    print(f"device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(
        model_type=ckpt["model_type"],
        architecture=ckpt["architecture"],
        in_channels=ckpt["in_channels"],
        num_regions=ckpt.get("num_regions"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tile_ids = resolve_tile_ids(args.data_root, args.split, args.tile_ids, args.max_tiles)
    feature_keys = DEFAULT_FEATURE_KEYS if args.feature_set == "default" else TEMPORAL_FEATURE_KEYS
    thresholds = args.thresholds or [args.threshold]
    min_areas = args.min_areas or [args.min_area_ha]

    summaries = []
    multi_run = len(thresholds) > 1 or len(min_areas) > 1
    for threshold in thresholds:
        for min_area_ha in min_areas:
            run_out_dir = args.out_dir
            if multi_run:
                run_name = f"thr_{format_float(threshold)}__area_{format_float(min_area_ha)}"
                if args.time_step is not None:
                    run_name += f"__ts_{args.time_step}"
                run_out_dir = args.out_dir / run_name
            summary = run_single_submission(
                args=args,
                model=model,
                device=device,
                feature_keys=feature_keys,
                tile_ids=tile_ids,
                threshold=threshold,
                min_area_ha=min_area_ha,
                out_dir=run_out_dir,
            )
            summaries.append(summary)

    print(json.dumps(summaries if multi_run else summaries[0], indent=2))


if __name__ == "__main__":
    main()
