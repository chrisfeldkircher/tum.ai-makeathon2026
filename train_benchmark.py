from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


class LimitedLoader:
    """Wrap a DataLoader and expose only the first N batches."""

    def __init__(self, loader, max_batches: int):
        self.loader = loader
        self.max_batches = max(1, int(max_batches))

    def __iter__(self):
        return itertools.islice(iter(self.loader), self.max_batches)

    def __len__(self) -> int:
        return min(len(self.loader), self.max_batches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the benchmark model extracted from code/run.ipynb."
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "makeathon-challenge")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "cache")
    parser.add_argument("--probs-dir", type=Path, default=REPO_ROOT / "cache_probs")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "weights")
    parser.add_argument("--model-type", choices=("baseline", "dann"), default="baseline")
    parser.add_argument("--architecture", choices=("unet", "deeplabv3plus"), default="unet")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patches-per-tile", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--spectral-aug-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--debug-every-n-steps", type=int, default=50)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-train-tiles", type=int, default=None)
    parser.add_argument("--tile-ids", nargs="+", default=None)
    parser.add_argument("--apply-lee-filter", action="store_true")
    parser.add_argument("--weights-name", default="benchmark_first_weights.pt")
    parser.add_argument("--history-out", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _filter_tile_ids(tile_ids: Iterable[str], selected_ids: set[str] | None, max_tiles: int | None) -> list[str]:
    filtered = [tile_id for tile_id in sorted(tile_ids) if selected_ids is None or tile_id in selected_ids]
    if max_tiles is not None:
        filtered = filtered[:max_tiles]
    return filtered


def infer_num_regions_from_tile_ids(tile_ids: Iterable[str]) -> int:
    from data.data import tile_id_to_region_label

    return max(tile_id_to_region_label(tile_id) for tile_id in tile_ids) + 1


def build_subset_dataloaders(args: argparse.Namespace):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from data.data import (
        DeforestationPatchDataset,
        REQUIRED_TRAIN_CACHE_KEYS,
        build_inventory,
        cache_missing_keys,
        cache_tile,
        split_tiles,
    )

    inventory = build_inventory(args.data_root, "train")
    selected_ids = set(args.tile_ids) if args.tile_ids else None
    candidate_ids = [
        tile_id
        for tile_id, tile_inventory in sorted(inventory.items())
        if tile_inventory.s2_paths and tile_inventory.has_any_labels()
    ]
    chosen_ids = _filter_tile_ids(candidate_ids, selected_ids, args.max_train_tiles)
    if len(chosen_ids) < 2:
        raise RuntimeError(
            "Subset training needs at least two labeled train tiles so train/val do not collapse."
        )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = {}
    preprocess_kwargs = {"apply_lee_filter": args.apply_lee_filter}
    for tile_id in chosen_ids:
        cache_path = cache_tile(inventory[tile_id], args.cache_dir, **preprocess_kwargs)
        missing = cache_missing_keys(cache_path, REQUIRED_TRAIN_CACHE_KEYS)
        if missing:
            cache_path.unlink(missing_ok=True)
            cache_path = cache_tile(inventory[tile_id], args.cache_dir, **preprocess_kwargs)
        cache_paths[tile_id] = cache_path

    train_ids, val_ids = split_tiles(chosen_ids, val_frac=args.val_frac, seed=args.seed)
    train_ds = DeforestationPatchDataset(
        [cache_paths[tile_id] for tile_id in train_ids],
        probs_dir=args.probs_dir,
        patch_size=args.patch_size,
        patches_per_tile=args.patches_per_tile,
        is_train=True,
        seed=args.seed,
    )
    val_ds = DeforestationPatchDataset(
        [cache_paths[tile_id] for tile_id in val_ids],
        probs_dir=args.probs_dir,
        is_train=False,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )
    return train_loader, val_loader, inventory, chosen_ids


def build_full_dataloaders(args: argparse.Namespace):
    from data import build_dataloaders

    train_loader, val_loader, inventory = build_dataloaders(
        root=args.data_root,
        cache_dir=args.cache_dir,
        probs_dir=args.probs_dir,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        patches_per_tile=args.patches_per_tile,
        num_workers=args.num_workers,
        val_frac=args.val_frac,
        seed=args.seed,
        preprocess_kwargs={"apply_lee_filter": args.apply_lee_filter},
    )
    active_tile_ids = [
        tile_id
        for tile_id, tile_inventory in sorted(inventory.items())
        if tile_inventory.s2_paths and tile_inventory.has_any_labels()
    ]
    return train_loader, val_loader, inventory, active_tile_ids


def main() -> None:
    args = parse_args()

    import torch

    from model import build_model, fit

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if args.max_train_tiles is not None or args.tile_ids:
        train_loader, val_loader, inventory, active_tile_ids = build_subset_dataloaders(args)
    else:
        train_loader, val_loader, inventory, active_tile_ids = build_full_dataloaders(args)

    if args.max_train_batches is not None:
        train_loader = LimitedLoader(train_loader, args.max_train_batches)
    if args.max_val_batches is not None:
        val_loader = LimitedLoader(val_loader, args.max_val_batches)

    sample = next(iter(train_loader))
    in_channels = int(sample["x"].shape[1])
    num_regions = infer_num_regions_from_tile_ids(active_tile_ids)
    use_dann = args.model_type == "dann"

    print(f"inventory tiles: {len(inventory)}")
    print(f"train batches: {len(train_loader)} | val batches: {len(val_loader)}")
    print(f"batch shape: {tuple(sample['x'].shape)}")
    print(f"in_channels: {in_channels} | num_regions: {num_regions}")

    model = build_model(
        model_type=args.model_type,
        architecture=args.architecture,
        in_channels=in_channels,
        num_regions=num_regions if use_dann else None,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    resumed_metadata = {}
    if args.resume_from is not None:
        resume_payload = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(resume_payload["model_state_dict"])
        if "optimizer_state_dict" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        resumed_metadata = {
            "resume_from": str(args.resume_from),
            "resume_epoch": resume_payload.get("epoch"),
        }
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        use_domain_adaptation=use_dann,
        alpha=args.alpha,
        amp_scaler=scaler,
        spectral_aug_p=args.spectral_aug_p,
        debug=True,
        debug_every_n_steps=args.debug_every_n_steps,
        checkpoint_dir=args.out_dir / "checkpoints",
        checkpoint_every=args.checkpoint_every,
    )

    weights_path = args.out_dir / args.weights_name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": args.model_type,
            "architecture": args.architecture,
            "in_channels": in_channels,
            "num_regions": num_regions if use_dann else None,
            "history": history,
            "seed": args.seed,
        },
        weights_path,
    )

    peak_allocated_mb = None
    peak_reserved_mb = None
    if device.type == "cuda":
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    summary = {
        "device": str(device),
        "epochs": args.epochs,
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "in_channels": in_channels,
        "num_regions": num_regions,
        "weights_path": str(weights_path),
        "history_last": history[-1] if history else None,
        "peak_memory_allocated_mb": peak_allocated_mb,
        "peak_memory_reserved_mb": peak_reserved_mb,
    }
    summary.update(resumed_metadata)
    print(json.dumps(summary, indent=2))

    if args.history_out is not None:
        args.history_out.parent.mkdir(parents=True, exist_ok=True)
        args.history_out.write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
