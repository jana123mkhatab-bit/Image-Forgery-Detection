"""Dual-head forgery model: RGB + SRM-residual input, shared encoder,
U-Net mask head + 3-way classification head. Pretrained encoder weights come
from the segmentation_models_pytorch / timm libraries (downloaded on first
use on Colab, nothing to store in the repo)."""
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

# three classic SRM high-pass kernels (as used in forensics papers)
_K1 = np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]]) / 4
_K2 = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]) / 12
_K3 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) / 2


class SRM(nn.Module):
    """Fixed SRM filters applied per RGB channel -> 3 residual channels
    (mean over colour channels, clipped like Fridrich & Kodovsky)."""

    def __init__(self, clip=3.0):
        super().__init__()
        k = torch.tensor(np.stack([_K1, _K2, _K3]), dtype=torch.float32)[:, None]  # 3x1x5x5
        self.register_buffer("k", k)
        self.clip = clip

    def forward(self, x):
        b, c, h, w = x.shape
        r = nn.functional.conv2d((x * 255).reshape(b * c, 1, h, w), self.k, padding=2)
        r = r.reshape(b, c, 3, h, w).mean(1)
        return r.clamp(-self.clip, self.clip) / self.clip


class ForgeryNet(nn.Module):
    def __init__(self, encoder="resnet34", weights="imagenet", n_classes=3, use_srm=True):
        super().__init__()
        self.use_srm = use_srm
        self.srm = SRM() if use_srm else None
        self.net = smp.Unet(encoder, encoder_weights=weights, in_channels=6 if use_srm else 3,
                            classes=1, aux_params=dict(classes=n_classes, dropout=0.2))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        rgb = (x - self.mean) / self.std
        inp = torch.cat([rgb, self.srm(x)], 1) if self.use_srm else rgb
        mask_logits, cls_logits = self.net(inp)
        return mask_logits, cls_logits
