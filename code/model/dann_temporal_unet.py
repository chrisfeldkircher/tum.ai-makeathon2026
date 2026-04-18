"""
dann_temporal_unet.py — Robust Multi-Task Deforestation Detection
=================================================================
DANN_Temporal_UNet  : Temporal attention + U-Net + AEF bottleneck fusion +
                      gradient-reversal domain adaptation. Input layout is
                      defined by `data.TEMPORAL_FEATURE_KEYS` (232 channels).
inference_pipeline  : Full-tile sliding-window prediction with forest gating.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

try:
    import segmentation_models_pytorch as smp
    _HAS_SMP = True
except ImportError:
    _HAS_SMP = False


# --- 1. Gradient-Reversal Layer ---
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


# --- 2. Temporal Attention blocks ---
class TemporalChannelAttention(nn.Module):
    def __init__(self, n_timesteps: int, channels_per_t: int, out_channels: Optional[int] = None, reduction: int = 4):
        super().__init__()
        self.T = n_timesteps
        self.C = channels_per_t
        self.C_out = out_channels or channels_per_t

        mid = max(1, channels_per_t * n_timesteps // reduction)
        self.score_mlp = nn.Sequential(
            nn.Linear(channels_per_t * n_timesteps, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, n_timesteps),
        )

        self.proj = (
            nn.Conv2d(channels_per_t, self.C_out, 1, bias=False)
            if self.C_out != channels_per_t else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, TC, H, W = x.shape
        x5 = x.view(B, self.T, self.C, H, W)
        ctx = x5.mean(dim=(3, 4)).view(B, self.T * self.C)
        scores = self.score_mlp(ctx)
        weights = torch.softmax(scores, dim=1).view(B, self.T, 1, 1, 1)
        attended = (x5 * weights).sum(dim=1)
        return self.proj(attended)


class TemporalMaxPool(nn.Module):
    def __init__(self, n_timesteps: int, channels_per_t: int, out_channels: Optional[int] = None):
        super().__init__()
        self.T = n_timesteps
        self.C = channels_per_t
        self.C_out = out_channels or channels_per_t
        self.proj = (
            nn.Conv2d(channels_per_t, self.C_out, 1, bias=False)
            if self.C_out != channels_per_t else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, TC, H, W = x.shape
        x5 = x.view(B, self.T, self.C, H, W)
        pooled, _ = x5.max(dim=1)
        return self.proj(pooled)


# --- 3. AEF bottleneck fusion ---
class AEFFusion(nn.Module):
    def __init__(self, aef_channels: int = 23, bottleneck_ch: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(aef_channels, bottleneck_ch, 1, bias=False),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, bottleneck: torch.Tensor, aef: torch.Tensor) -> torch.Tensor:
        Bh, w = bottleneck.shape[2], bottleneck.shape[3]
        aef_down = F.adaptive_avg_pool2d(aef, (Bh, w))
        aef_proj = self.proj(aef_down)
        return bottleneck + aef_proj


class DomainClassifier(nn.Module):
    def __init__(self, in_features: int = 512, num_regions: int = 10, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_regions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --- 4. The Full Network ---
class DANN_Temporal_UNet(nn.Module):
    """Temporal DANN U-Net matching `data.TEMPORAL_FEATURE_KEYS` layout.

    Expected channel order (232 total):
        [  0, 144)  s2_monthly_pre      — 12 months × 12 S2 bands
        [144, 156)  s1_monthly_pre_asc  — 12 months VV ascending
        [156, 168)  s1_monthly_pre_desc — 12 months VV descending
        [168, 232)  aef_delta           — 64 AEF embedding dims

    S1 is fed to the temporal block as (B, 12 months, 2 orbits, H, W); we
    reshape from the flat [asc×12, desc×12] layout inside `forward`. No
    "static" block — the original 60-ch slot was undocumented and no key in
    the pipeline produces it, so it's dropped.
    """
    def __init__(
        self,
        num_regions: int = 12,
        s2_monthly_slice: Tuple[int, int] = (0, 144),
        s1_monthly_slice: Tuple[int, int] = (144, 168),
        aef_delta_slice: Tuple[int, int] = (168, 232),
        encoder_name: str = "resnet34",
        temporal_mode: str = "attention",
        s2_out_ch: int = 32,
        s1_out_ch: int = 8,
    ):
        super().__init__()
        if not _HAS_SMP:
            raise ImportError("segmentation_models_pytorch is required.")

        self.s2_monthly_slice = s2_monthly_slice
        self.s1_monthly_slice = s1_monthly_slice
        self.aef_delta_slice = aef_delta_slice

        TemporalBlock = TemporalChannelAttention if temporal_mode == "attention" else TemporalMaxPool

        self.s2_temporal = TemporalBlock(n_timesteps=12, channels_per_t=12, out_channels=s2_out_ch)
        self.s1_temporal = TemporalBlock(n_timesteps=12, channels_per_t=2, out_channels=s1_out_ch)

        unet_in_ch = s2_out_ch + s1_out_ch

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=unet_in_ch,
            classes=1,
            activation=None,
        )

        bottleneck_ch = self.unet.encoder.out_channels[-1]
        aef_ch = aef_delta_slice[1] - aef_delta_slice[0]

        self.aef_fusion = AEFFusion(aef_channels=aef_ch, bottleneck_ch=bottleneck_ch)
        self.grl = GradientReversal(lambda_=1.0)
        self.to_512 = nn.Identity() if bottleneck_ch == 512 else nn.Linear(bottleneck_ch, 512)
        self.domain_classifier = DomainClassifier(in_features=512, num_regions=num_regions)

    def forward(self, x: torch.Tensor, grl_lambda: float = 1.0) -> Dict[str, torch.Tensor]:
        s2_raw = x[:, self.s2_monthly_slice[0]:self.s2_monthly_slice[1]]
        s1_raw = x[:, self.s1_monthly_slice[0]:self.s1_monthly_slice[1]]
        aef_delta = x[:, self.aef_delta_slice[0]:self.aef_delta_slice[1]]

        # S1 arrives as [asc_m1..asc_m12, desc_m1..desc_m12]. The temporal
        # block wants (B, T=12 months, C=2 orbits) grouping, so reshape as
        # (B, 2, 12, H, W) and permute month and orbit axes.
        B, _, H, W = s1_raw.shape
        s1_raw = s1_raw.view(B, 2, 12, H, W).permute(0, 2, 1, 3, 4).reshape(B, 24, H, W)

        s2_feat = self.s2_temporal(s2_raw)
        s1_feat = self.s1_temporal(s1_raw)

        unet_in = torch.cat([s2_feat, s1_feat], dim=1)
        features = self.unet.encoder(unet_in)
        bottleneck = features[-1]

        bottleneck = self.aef_fusion(bottleneck, aef_delta)
        features = list(features)
        features[-1] = bottleneck

        decoder_out = self.unet.decoder(features)
        seg_logits = self.unet.segmentation_head(decoder_out)

        pooled = F.adaptive_avg_pool2d(bottleneck, 1).flatten(1)
        pooled = self.to_512(pooled)
        domain_feat = self.grl(pooled, lambda_=grl_lambda)
        domain_logits = self.domain_classifier(domain_feat)

        return {"seg_logits": seg_logits, "domain_logits": domain_logits}


# --- 5. Inference Pipeline ---
@torch.no_grad()
def inference_pipeline(
    model: DANN_Temporal_UNet,
    tile_tensor: torch.Tensor,
    forest_mask: Optional[torch.Tensor] = None,
    patch_size: int = 256,
    overlap: int = 64,
    batch_size: int = 4,
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    C, H, W = tile_tensor.shape
    stride = patch_size - overlap
    prob_sum = torch.zeros(1, H, W, device=device)
    count_sum = torch.zeros(1, H, W, device=device)

    origins = []
    y0 = 0
    while True:
        x0 = 0
        while True:
            origins.append((y0, x0))
            if x0 + patch_size >= W: break
            x0 = min(x0 + stride, W - patch_size)
        if y0 + patch_size >= H: break
        y0 = min(y0 + stride, H - patch_size)

    for i in range(0, len(origins), batch_size):
        batch_origins = origins[i: i + batch_size]
        patches = [tile_tensor[:, py:py + patch_size, px:px + patch_size] for (py, px) in batch_origins]
        x_batch = torch.stack(patches).to(device).float()
        out = model(x_batch, grl_lambda=0.0)
        probs = torch.sigmoid(out["seg_logits"])

        for j, (py, px) in enumerate(batch_origins):
            prob_sum[:, py:py + patch_size, px:px + patch_size] += probs[j]
            count_sum[:, py:py + patch_size, px:px + patch_size] += 1.0

    prob_map = (prob_sum / count_sum.clamp_min(1.0)).squeeze(0).cpu()

    if forest_mask is not None:
        if isinstance(forest_mask, np.ndarray):
            forest_mask = torch.from_numpy(forest_mask.astype(np.float32))
        prob_gated = prob_map * forest_mask.float().to(prob_map.device)
    else:
        prob_gated = prob_map

    return {
        "prob": prob_map,
        "pred": (prob_map >= threshold).to(torch.uint8),
        "pred_gated": (prob_gated >= threshold).to(torch.uint8),
    }