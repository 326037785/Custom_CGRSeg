"""Standalone helper to assemble the CGRSeg backbone without mmseg/mmcv.

The original demo relied on ``mmseg``'s ``build_segmentor`` and ``mmcv``'s
configuration system, which can be difficult to install with newer CUDA
toolchains. This helper builds a lightweight segmentation model directly with
PyTorch so you can inspect the network structure even when the OpenMMLab
stack is unavailable.

Notes
-----
* The backbone uses the EfficientFormerV2 implementation from ``models``
  (not the ``mmseg`` shim) with ``fork_feat=True`` to expose multi-scale
  features.
* The decode head is a minimal PyTorch implementation that fuses the
  backbone's four outputs and upsamples to full resolution. It is not intended
  to reproduce training-time results; it simply makes the shapes easy to
  inspect.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# Ensure local packages (``models``) are importable without installation.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


@dataclass
class HeadConfig:
    """Simple configuration holder for the tiny decode head."""

    in_channels: List[int]
    num_classes: int = 150  # ADE20K default
    embed_dim: int = 128


class SimpleFusionHead(nn.Module):
    """A minimal decode head that avoids mmcv dependencies.

    The head takes the four feature maps from EfficientFormerV2, reduces each
    to a common channel width, upsamples lower-resolution features, and fuses
    them via concatenation. The result is projected to the desired number of
    segmentation classes.
    """

    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.reducers = nn.ModuleList(
            nn.Conv2d(in_ch, cfg.embed_dim, kernel_size=1) for in_ch in cfg.in_channels
        )
        fusion_in = cfg.embed_dim * len(cfg.in_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in, cfg.embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg.embed_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(cfg.embed_dim, cfg.num_classes, kernel_size=1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # Bring all features to the spatial size of the highest-resolution map.
        target_h, target_w = feats[0].shape[2:]
        resized = []
        for feat, reduce in zip(feats, self.reducers):
            reduced = reduce(feat)
            if reduced.shape[2:] != (target_h, target_w):
                reduced = F.interpolate(
                    reduced, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
            resized.append(reduced)

        fused = torch.cat(resized, dim=1)
        fused = self.fusion(fused)
        return self.classifier(fused)


class LightweightCGRSeg(nn.Module):
    """Tiny segmentation wrapper around EfficientFormerV2.

    Only depends on PyTorch and the local ``models`` package so it runs even
    when ``mmcv``/``mmseg`` are not installed. The forward pass returns logits
    upsampled to the input resolution for quick inspection.
    """

    def __init__(self, num_classes: int = 150):
        super().__init__()
        from models.backbones.efficientformer_v2 import efficientformerv2_s0_feat

        self.backbone = efficientformerv2_s0_feat(fork_feat=True)
        head_cfg = HeadConfig(in_channels=[32, 48, 96, 176], num_classes=num_classes)
        self.decode_head = SimpleFusionHead(head_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.decode_head(feats)
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
        return logits


if __name__ == "__main__":
    model = LightweightCGRSeg(num_classes=150)
    model.eval()

    dummy = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output = model(dummy)

    print(model)
    print("\nOutput type:", type(output))
    print("Output shape:", output.shape)
