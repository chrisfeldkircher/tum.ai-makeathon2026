import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import segmentation_models_pytorch as smp


import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class DeforestationBaseModel(nn.Module):
    """
    Base prediction model for Deforestation Detection.
    Switches between U-Net and DeepLabV3+ to produce pixel-wise logits.
    """
    def __init__(
        self, 
        architecture: str = "deeplabv3plus", 
        encoder_name: str = "resnet34", 
        in_channels: int = 251
    ):
        super().__init__()
        self.architecture = architecture.lower()
        
        # 251 channels means we cannot use standard ImageNet pretrained weights.
        # We must initialize with random weights (None) for the encoder.
        encoder_weights = None 
        
        if self.architecture == "deeplabv3plus":
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=1,            # 1 for binary segmentation (Deforested or Not)
                activation=None       # Outputs raw logits
            )
        elif self.architecture == "unet":
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=1,
                activation=None
            )
        else:
            raise ValueError(f"Architecture '{architecture}' is not supported. Use 'unet' or 'deeplabv3plus'.")

    def forward(self, x):
        # Takes (B, 251, H, W) and outputs (B, 1, H, W)
        seg_logits = self.model(x)
        return {"seg_logits": seg_logits}
    
class RobustLoss(nn.Module):
    """
    Combines Weighted BCE + Dice for segmentation and 
    CrossEntropy for domain adaptation.
    """
    def __init__(self, domain_weight: float = 0.1):
        super().__init__()
        self.domain_weight = domain_weight
        self.seg_criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.dom_criterion = nn.CrossEntropyLoss()

    def forward(self, out: Dict[str, torch.Tensor], y: torch.Tensor, 
                w: torch.Tensor, region_labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # 1. Segmentation Loss (Weighted by label confidence 'w')
        # BCE part handled manually to use per-pixel weights
        bce = F.binary_cross_entropy_with_logits(out["seg_logits"], y, weight=w)
        dice = self.seg_criterion(out["seg_logits"], y)
        seg_loss = bce + dice

        # 2. Domain Loss (from the GRL branch)
        dom_loss = self.dom_criterion(out["domain_logits"], region_labels)

        total_loss = seg_loss + (self.domain_weight * dom_loss)
        
        return total_loss, {
            "seg_loss": seg_loss.item(),
            "dom_loss": dom_loss.item(),
            "total": total_loss.item()
        }
    

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
        in_channels: int = 251,
        encoder_name: str = "resnet34",
        domain_dropout: float = 0.3,
    ):
        super().__init__()

        # For in_channels=251, pretrained imagenet weights are not directly usable.
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
        features = self.unet.encoder(x)                     # list of feature maps
        decoder_out = self.unet.decoder(features)       # highest-res decoder feature
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

    x_aug[:, s2_lo:s2_hi] = torch.where(
        apply, x_aug[:, s2_lo:s2_hi] * s2_scale, x_aug[:, s2_lo:s2_hi]
    )
    x_aug[:, s1_lo:s1_hi] = torch.where(
        apply, x_aug[:, s1_lo:s1_hi] * s1_scale, x_aug[:, s1_lo:s1_hi]
    )

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


def train_one_epoch(
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
    """
    Expects each batch to contain:
      - batch['x']: [B,251,H,W]
      - batch['y']: [B,H,W] binary mask
      - batch['w']: [B,H,W] pixel weights
      - batch['region_label']: [B] integer region IDs
      - optional batch['mask']: [B,H,W] forest mask for gating

    Loss_total = Segmentation_Loss(weighted_BCE_Dice) + alpha * Domain_Loss(CrossEntropy)
    """
    model.train()
    domain_criterion = nn.CrossEntropyLoss()

    running_total = 0.0
    running_seg = 0.0
    running_dom = 0.0
    n_batches = 0

    total_steps = max(1, len(loader))

    for step, batch in enumerate(loader):
        x = batch["x"].to(device, non_blocking=True).float()
        y = batch["y"].to(device, non_blocking=True)
        w = batch["w"].to(device, non_blocking=True).float()
        region_label = batch["region_label"].to(device, non_blocking=True).long()

        # Optional forest gating in addition to confidence weights

        if model.training:
            x = random_spectral_scaling(x)
            
        if "mask" in batch:
            m = batch["mask"].to(device, non_blocking=True).float()
            w = w * m

        # Spectral augmentation (S2 + S1 channel groups)
        x = random_spectral_scaling(
            x,
            s2_slice=s2_slice,
            s1_slice=s1_slice,
            scale_min=0.8,
            scale_max=1.2,
            p=spectral_aug_p,
        )

        # DANN lambda schedule based on global training progress
        global_step = epoch * total_steps + step
        max_steps = max(1, num_epochs * total_steps - 1)
        progress = (epoch * len(loader) + step) / (num_epochs * len(loader))
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