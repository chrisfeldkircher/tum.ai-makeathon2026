import math
from typing import Any, Dict, Optional, Tuple

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class DeforestationBaseModel(nn.Module):
    """
    Base prediction model for Deforestation Detection.
    Switches between U-Net and DeepLabV3+ to produce pixel-wise logits.
    """
    def __init__(
        self,
        architecture: str = "deeplabv3plus",
        encoder_name: str = "resnet34",
        in_channels: int = 274,
    ):
        super().__init__()
        self.architecture = architecture.lower()

        # High in_channels means we cannot use standard ImageNet pretrained
        # weights. We must initialize with random weights (None) for the encoder.
        encoder_weights = None

        if self.architecture == "deeplabv3plus":
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=1,
                activation=None,
            )
        elif self.architecture == "unet":
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=1,
                activation=None,
            )
        else:
            raise ValueError(f"Architecture '{architecture}' is not supported. Use 'unet' or 'deeplabv3plus'.")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Takes (B, 251, H, W) and outputs (B, 1, H, W)
        seg_logits = self.model(x)
        return {"seg_logits": seg_logits}


class _GRLFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor, lambda_: Optional[float] = None) -> torch.Tensor:
        lam = self.lambda_ if lambda_ is None else float(lambda_)
        return _GRLFunction.apply(x, lam)


class DANN_UNet(nn.Module):
    """
    Segmentation branch: U-Net (ResNet34 encoder by default)
    Domain branch: GRL -> Linear(512) -> ReLU -> Dropout -> Linear(num_regions)
    """
    def __init__(
        self,
        num_regions: int,
        in_channels: int = 274,
        encoder_name: str = "resnet34",
        domain_dropout: float = 0.3,
    ):
        super().__init__()

        # High in_channels means pretrained imagenet weights are not usable.
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )

        self.grl = GradientReversal(lambda_=1.0)

        bottleneck_channels = self.unet.encoder.out_channels[-1]
        self.to_512 = nn.Identity() if bottleneck_channels == 512 else nn.Linear(bottleneck_channels, 512)

        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=domain_dropout),
            nn.Linear(512, num_regions),
        )

    def forward(self, x: torch.Tensor, grl_lambda: float = 1.0) -> Dict[str, torch.Tensor]:
        # Encoder-decoder forward for segmentation
        features = self.unet.encoder(x)
        decoder_out = self.unet.decoder(features)
        seg_logits = self.unet.segmentation_head(decoder_out)  # [B,1,H,W]

        # Domain branch from bottleneck
        bottleneck = features[-1]                           # [B,C,h,w], C=512 for resnet34
        pooled = F.adaptive_avg_pool2d(bottleneck, output_size=1).flatten(1)  # [B,C]
        pooled = self.to_512(pooled)                        # [B,512]
        domain_feat = self.grl(pooled, lambda_=grl_lambda)
        domain_logits = self.domain_classifier(domain_feat) # [B,num_regions]

        return {
            "seg_logits": seg_logits,
            "domain_logits": domain_logits,
        }


def build_model(
    model_type: str,
    in_channels: int = 274,
    encoder_name: str = "resnet34",
    architecture: str = "deeplabv3plus",
    num_regions: Optional[int] = None,
    domain_dropout: float = 0.3,
) -> nn.Module:
    """Factory for baseline and DANN models."""
    mt = model_type.lower()
    if mt in {"baseline", "seg", "segmentation"}:
        return DeforestationBaseModel(
            architecture=architecture,
            encoder_name=encoder_name,
            in_channels=in_channels,
        )
    if mt in {"dann", "generalization", "domain_adaptation"}:
        if num_regions is None:
            raise ValueError("num_regions is required when model_type='dann'.")
        return DANN_UNet(
            num_regions=num_regions,
            in_channels=in_channels,
            encoder_name=encoder_name,
            domain_dropout=domain_dropout,
        )
    raise ValueError("model_type must be one of: baseline, dann")


def random_spectral_scaling(
    x: torch.Tensor,
    s2_slice: Tuple[int, int] = (0, 144),
    s1_slice: Tuple[int, int] = (144, 168),
    scale_min: float = 0.8,
    scale_max: float = 1.2,
    p: float = 0.5,
) -> torch.Tensor:
    """
    Randomly scales S2 and S1 channel groups per sample.
    x shape: [B,C,H,W]
    """
    if p <= 0.0:
        return x

    b = x.shape[0]
    device = x.device
    apply = (torch.rand(b, device=device) < p).view(b, 1, 1, 1)

    if not apply.any():
        return x

    x_aug = x.clone()

    s2_scale = torch.empty((b, 1, 1, 1), device=device).uniform_(scale_min, scale_max)
    s1_scale = torch.empty((b, 1, 1, 1), device=device).uniform_(scale_min, scale_max)

    s2_lo, s2_hi = s2_slice
    s1_lo, s1_hi = s1_slice

    x_aug[:, s2_lo:s2_hi] = torch.where(apply, x_aug[:, s2_lo:s2_hi] * s2_scale, x_aug[:, s2_lo:s2_hi])
    x_aug[:, s1_lo:s1_hi] = torch.where(apply, x_aug[:, s1_lo:s1_hi] * s1_scale, x_aug[:, s1_lo:s1_hi])

    return x_aug


def weighted_bce_dice_loss(
    seg_logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Weighted BCE + weighted soft Dice.
    seg_logits: [B,1,H,W]
    target: [B,H,W] or [B,1,H,W] with 0/1 labels
    weight: [B,H,W] or [B,1,H,W], e.g. batch['w']
    """
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if weight.ndim == 3:
        weight = weight.unsqueeze(1)

    target = target.float()
    weight = weight.float()

    # Weighted BCE
    bce_per_pixel = F.binary_cross_entropy_with_logits(seg_logits, target, reduction="none")
    wbce = (bce_per_pixel * weight).sum() / weight.sum().clamp_min(1.0)

    # Weighted soft Dice
    prob = torch.sigmoid(seg_logits)
    inter = (prob * target * weight).sum(dim=(1, 2, 3))
    denom = ((prob + target) * weight).sum(dim=(1, 2, 3))
    dice_loss = 1.0 - (2.0 * inter + eps) / (denom + eps)
    dice_loss = dice_loss.mean()

    return 0.5 * wbce + 0.5 * dice_loss


def dann_lambda_schedule(progress_0_to_1: float) -> float:
    # Standard DANN schedule
    p = float(max(0.0, min(1.0, progress_0_to_1)))
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


def _prepare_batch(batch: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["x"].to(device, non_blocking=True).float()
    y = batch["y"].to(device, non_blocking=True)
    w = batch["w"].to(device, non_blocking=True).float()

    if "mask" in batch:
        m = batch["mask"].to(device, non_blocking=True).float()
        w = w * m

    return x, y, w


def train_one_epoch_baseline(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    spectral_aug_p: float = 0.5,
    s2_slice: Tuple[int, int] = (0, 144),
    s1_slice: Tuple[int, int] = (144, 168),
) -> Dict[str, float]:
    """Train one epoch for the baseline segmentation model."""
    model.train()

    running_total = 0.0
    n_batches = 0

    for batch in loader:
        x, y, w = _prepare_batch(batch, device)
        x = random_spectral_scaling(
            x,
            s2_slice=s2_slice,
            s1_slice=s1_slice,
            scale_min=0.8,
            scale_max=1.2,
            p=spectral_aug_p,
        )

        optimizer.zero_grad(set_to_none=True)

        if amp_scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(x)
                seg_loss = weighted_bce_dice_loss(out["seg_logits"], y, w)
            amp_scaler.scale(seg_loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            out = model(x)
            seg_loss = weighted_bce_dice_loss(out["seg_logits"], y, w)
            seg_loss.backward()
            optimizer.step()

        running_total += float(seg_loss.detach().item())
        n_batches += 1

    denom = max(1, n_batches)
    return {
        "loss_total": running_total / denom,
        "loss_seg": running_total / denom,
        "loss_domain": 0.0,
    }


def train_one_epoch_dann(
    model: DANN_UNet,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    alpha: float = 0.1,
    amp_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    spectral_aug_p: float = 0.5,
    s2_slice: Tuple[int, int] = (0, 144),
    s1_slice: Tuple[int, int] = (144, 168),
) -> Dict[str, float]:
    """Train one epoch for DANN (segmentation + domain adversarial)."""
    model.train()
    domain_criterion = nn.CrossEntropyLoss()

    running_total = 0.0
    running_seg = 0.0
    running_dom = 0.0
    n_batches = 0

    total_steps = max(1, len(loader))

    for step, batch in enumerate(loader):
        x, y, w = _prepare_batch(batch, device)
        if "region_label" not in batch:
            raise KeyError("DANN training requires batch['region_label'].")
        region_label = batch["region_label"].to(device, non_blocking=True).long()

        x = random_spectral_scaling(
            x,
            s2_slice=s2_slice,
            s1_slice=s1_slice,
            scale_min=0.8,
            scale_max=1.2,
            p=spectral_aug_p,
        )

        global_step = epoch * total_steps + step
        max_steps = max(1, num_epochs * total_steps - 1)
        progress = global_step / max_steps
        grl_lambda = dann_lambda_schedule(progress)

        optimizer.zero_grad(set_to_none=True)

        if amp_scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(x, grl_lambda=grl_lambda)
                seg_loss = weighted_bce_dice_loss(out["seg_logits"], y, w)
                dom_loss = domain_criterion(out["domain_logits"], region_label)
                total_loss = seg_loss + alpha * dom_loss

            amp_scaler.scale(total_loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            out = model(x, grl_lambda=grl_lambda)
            seg_loss = weighted_bce_dice_loss(out["seg_logits"], y, w)
            dom_loss = domain_criterion(out["domain_logits"], region_label)
            total_loss = seg_loss + alpha * dom_loss

            total_loss.backward()
            optimizer.step()

        running_total += float(total_loss.detach().item())
        running_seg += float(seg_loss.detach().item())
        running_dom += float(dom_loss.detach().item())
        n_batches += 1

    denom = max(1, n_batches)
    return {
        "loss_total": running_total / denom,
        "loss_seg": running_seg / denom,
        "loss_domain": running_dom / denom,
    }


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    alpha: float = 0.1,
    amp_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    spectral_aug_p: float = 0.5,
    s2_slice: Tuple[int, int] = (0, 144),
    s1_slice: Tuple[int, int] = (144, 168),
    use_domain_adaptation: Optional[bool] = None,
) -> Dict[str, float]:
    """
    Unified one-epoch trainer.
    - Baseline mode: segmentation only.
    - DANN mode: segmentation + alpha * domain CE loss.
    """
    if use_domain_adaptation is None:
        use_domain_adaptation = isinstance(model, DANN_UNet)

    if use_domain_adaptation:
        if not isinstance(model, DANN_UNet):
            raise TypeError("use_domain_adaptation=True requires model to be an instance of DANN_UNet.")
        return train_one_epoch_dann(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs,
            alpha=alpha,
            amp_scaler=amp_scaler,
            spectral_aug_p=spectral_aug_p,
            s2_slice=s2_slice,
            s1_slice=s1_slice,
        )

    return train_one_epoch_baseline(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=device,
        amp_scaler=amp_scaler,
        spectral_aug_p=spectral_aug_p,
        s2_slice=s2_slice,
        s1_slice=s1_slice,
    )


def fit(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    alpha: float = 0.1,
    amp_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    spectral_aug_p: float = 0.5,
    s2_slice: Tuple[int, int] = (0, 144),
    s1_slice: Tuple[int, int] = (144, 168),
    use_domain_adaptation: Optional[bool] = None,
    scheduler: Optional[Any] = None,
) -> list[Dict[str, float]]:
    """Train for multiple epochs and return per-epoch loss history."""
    history: list[Dict[str, float]] = []
    for epoch in range(num_epochs):
        stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs,
            alpha=alpha,
            amp_scaler=amp_scaler,
            spectral_aug_p=spectral_aug_p,
            s2_slice=s2_slice,
            s1_slice=s1_slice,
            use_domain_adaptation=use_domain_adaptation,
        )
        history.append(stats)
        if scheduler is not None:
            scheduler.step()
    return history


__all__ = [
    "GradientReversal",
    "DeforestationBaseModel",
    "DANN_UNet",
    "build_model",
    "random_spectral_scaling",
    "weighted_bce_dice_loss",
    "dann_lambda_schedule",
    "train_one_epoch_baseline",
    "train_one_epoch_dann",
    "train_one_epoch",
    "fit",
]