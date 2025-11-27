"""Standalone CGRSeg model demo without mmcv/mmsegmentation dependencies.

This script provides a complete implementation of the CGRSeg architecture
(backbone + decode head) using only PyTorch and timm, without requiring
mmcv or mmsegmentation installation. It accurately reproduces the model
structure as specified in the configuration files.

Components included:
- EfficientFormerV2 backbone (S0/S1/S2/L variants)
- CGRSeg decode head with all components:
  - RCA (Region-wise Channel Attention)
  - RCM (Region-wise Context Module) 
  - DPGHead (Dynamic Point-wise Gating)
  - PyramidPoolAgg, FuseBlockMulti, NextLayer, SpatialGatherModule

Usage:
    python testModel.py [--model {t,b,l}] [--input_size SIZE]
    
    --model: Model variant (t=tiny/S1, b=base/S2, l=large/L). Default: t
    --input_size: Input image size. Default: 512

Example:
    python testModel.py --model t --input_size 512
    python testModel.py --model l --input_size 1024
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath, trunc_normal_
from timm.layers.helpers import to_2tuple

# ============================================================================
# EfficientFormerV2 Backbone (Standalone Implementation)
# ============================================================================

EfficientFormer_width = {
    'L': [40, 80, 192, 384],
    'S2': [32, 64, 144, 288],
    'S1': [32, 48, 120, 224],
    'S0': [32, 48, 96, 176],
}

EfficientFormer_depth = {
    'L': [5, 5, 15, 10],
    'S2': [4, 4, 12, 8],
    'S1': [3, 3, 9, 6],
    'S0': [2, 2, 6, 4],
}

expansion_ratios_L = {
    '0': [4, 4, 4, 4, 4],
    '1': [4, 4, 4, 4, 4],
    '2': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '3': [4, 4, 4, 3, 3, 3, 3, 4, 4, 4],
}

expansion_ratios_S2 = {
    '0': [4, 4, 4, 4],
    '1': [4, 4, 4, 4],
    '2': [4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '3': [4, 4, 3, 3, 3, 3, 4, 4],
}

expansion_ratios_S1 = {
    '0': [4, 4, 4],
    '1': [4, 4, 4],
    '2': [4, 4, 3, 3, 3, 3, 4, 4, 4],
    '3': [4, 4, 3, 3, 4, 4],
}

expansion_ratios_S0 = {
    '0': [4, 4],
    '1': [4, 4],
    '2': [4, 3, 3, 3, 4, 4],
    '3': [4, 3, 3, 4],
}


class Attention4D(nn.Module):
    def __init__(self, dim=384, key_dim=32, num_heads=8, attn_ratio=4,
                 resolution=7, act_layer=nn.ReLU, stride=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = key_dim * num_heads

        if stride is not None:
            self.resolution = math.ceil(resolution / stride)
            self.stride_conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim),
                nn.BatchNorm2d(dim),
            )
            self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
        else:
            self.resolution = resolution
            self.stride_conv = None
            self.upsample = None

        self.N = self.resolution ** 2
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads

        self.q = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.key_dim, 1),
            nn.BatchNorm2d(self.num_heads * self.key_dim),
        )
        self.k = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.key_dim, 1),
            nn.BatchNorm2d(self.num_heads * self.key_dim),
        )
        self.v = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.d, 1),
            nn.BatchNorm2d(self.num_heads * self.d),
        )
        self.v_local = nn.Sequential(
            nn.Conv2d(self.num_heads * self.d, self.num_heads * self.d,
                      kernel_size=3, stride=1, padding=1, groups=self.num_heads * self.d),
            nn.BatchNorm2d(self.num_heads * self.d),
        )
        self.talking_head1 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1)
        self.talking_head2 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1)

        self.proj = nn.Sequential(
            act_layer(),
            nn.Conv2d(self.dh, dim, 1),
            nn.BatchNorm2d(dim),
        )

        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        self.register_buffer('attention_biases', torch.zeros(num_heads, 49))
        self.register_buffer('attention_bias_idxs', torch.ones(49, 49).long())
        self.attention_biases_seg = nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs_seg',
                             torch.LongTensor(idxs).view(N, N))

    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases_seg[:, self.attention_bias_idxs_seg]

    def forward(self, x):
        B, C, H, W = x.shape
        if self.stride_conv is not None:
            x = self.stride_conv(x)
            H = H // 2
            W = W // 2

        q = self.q(x).flatten(2).reshape(B, self.num_heads, -1, H * W).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, H * W).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, H * W).permute(0, 1, 3, 2)

        attn = (q @ k) * self.scale
        bias = self.attention_biases_seg[:, self.attention_bias_idxs_seg] if self.training else self.ab
        bias = F.interpolate(bias.unsqueeze(0), size=(attn.size(-2), attn.size(-1)), mode='bicubic')
        attn = attn + bias

        attn = self.talking_head1(attn)
        attn = attn.softmax(dim=-1)
        attn = self.talking_head2(attn)

        x = (attn @ v)
        out = x.transpose(2, 3).reshape(B, self.dh, H, W) + v_local
        if self.upsample is not None:
            out = self.upsample(out)
        out = self.proj(out)
        return out


def stem(in_chs, out_chs, act_layer=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs // 2),
        act_layer(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs),
        act_layer(),
    )


class LGQuery(nn.Module):
    def __init__(self, in_dim, out_dim, resolution1, resolution2):
        super().__init__()
        self.pool = nn.AvgPool2d(1, 2, 0)
        self.local = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        local_q = self.local(x)
        pool_q = self.pool(x)
        q = local_q + pool_q
        q = self.proj(q)
        return q


class Attention4DDownsample(nn.Module):
    def __init__(self, dim=384, key_dim=16, num_heads=8, attn_ratio=4,
                 resolution=7, out_dim=None, act_layer=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = key_dim * num_heads
        self.resolution = resolution
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.out_dim = out_dim if out_dim is not None else dim
        self.resolution2 = math.ceil(self.resolution / 2)
        
        self.q = LGQuery(dim, self.num_heads * self.key_dim, self.resolution, self.resolution2)
        self.N = self.resolution ** 2
        self.N2 = self.resolution2 ** 2

        self.k = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.key_dim, 1),
            nn.BatchNorm2d(self.num_heads * self.key_dim),
        )
        self.v = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.d, 1),
            nn.BatchNorm2d(self.num_heads * self.d),
        )
        self.v_local = nn.Sequential(
            nn.Conv2d(self.num_heads * self.d, self.num_heads * self.d,
                      kernel_size=3, stride=2, padding=1, groups=self.num_heads * self.d),
            nn.BatchNorm2d(self.num_heads * self.d),
        )
        self.proj = nn.Sequential(
            act_layer(),
            nn.Conv2d(self.dh, self.out_dim, 1),
            nn.BatchNorm2d(self.out_dim),
        )

        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        points_ = list(itertools.product(range(self.resolution2), range(self.resolution2)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * math.ceil(self.resolution / self.resolution2) - p2[0] + (size - 1) / 2),
                    abs(p1[1] * math.ceil(self.resolution / self.resolution2) - p2[1] + (size - 1) / 2)
                )
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        self.register_buffer('attention_biases', torch.zeros(num_heads, 196))
        self.register_buffer('attention_bias_idxs', torch.ones(49, 196).long())
        self.attention_biases_seg = nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs_seg',
                             torch.LongTensor(idxs).view(N_, N))

    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases_seg[:, self.attention_bias_idxs_seg]

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).flatten(2).reshape(B, self.num_heads, -1, H * W // 4).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, H * W).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, H * W).permute(0, 1, 3, 2)

        attn = (q @ k) * self.scale
        bias = self.attention_biases_seg[:, self.attention_bias_idxs_seg] if self.training else self.ab
        bias = F.interpolate(bias.unsqueeze(0), size=(attn.size(-2), attn.size(-1)), mode='bicubic')
        attn = attn + bias
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(2, 3)
        out = x.reshape(B, self.dh, H // 2, W // 2) + v_local
        out = self.proj(out)
        return out


class Embedding(nn.Module):
    def __init__(self, patch_size=3, stride=2, padding=1, in_chans=3, embed_dim=768,
                 norm_layer=nn.BatchNorm2d, light=False, asub=False, resolution=None,
                 act_layer=nn.ReLU, attn_block=Attention4DDownsample):
        super().__init__()
        self.light = light
        self.asub = asub

        if self.light:
            self.new_proj = nn.Sequential(
                nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=2, padding=1, groups=in_chans),
                nn.BatchNorm2d(in_chans),
                nn.Hardswish(),
                nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(embed_dim),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm2d(embed_dim)
            )
        elif self.asub:
            self.attn = attn_block(dim=in_chans, out_dim=embed_dim,
                                   resolution=resolution, act_layer=act_layer)
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                                  stride=stride, padding=padding)
            self.bn = norm_layer(embed_dim) if norm_layer else nn.Identity()
        else:
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                                  stride=stride, padding=padding)
            self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if self.light:
            out = self.new_proj(x) + self.skip(x)
        elif self.asub:
            out_conv = self.conv(x)
            out_conv = self.bn(out_conv)
            out = self.attn(x) + out_conv
        else:
            x = self.proj(x)
            out = self.norm(x)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., mid_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        
        if self.mid_conv:
            self.mid = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, 
                                 stride=1, padding=1, groups=hidden_features)
            self.mid_norm = nn.BatchNorm2d(hidden_features)
        
        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.norm2 = nn.BatchNorm2d(out_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        if self.mid_conv:
            x_mid = self.mid(x)
            x_mid = self.mid_norm(x_mid)
            x = self.act(x_mid)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.drop(x)
        return x


class AttnFFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.ReLU, norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5,
                 resolution=7, stride=None):
        super().__init__()
        self.token_mixer = Attention4D(dim, resolution=resolution, act_layer=act_layer, stride=stride)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, mid_conv=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(x))
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x


class FFN(nn.Module):
    def __init__(self, dim, pool_size=3, mlp_ratio=4., act_layer=nn.GELU,
                 drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, mid_conv=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.mlp(x))
        return x


def meta_blocks(dim, index, layers, pool_size=3, mlp_ratio=4., act_layer=nn.GELU,
                norm_layer=nn.LayerNorm, drop_rate=.0, drop_path_rate=0.,
                use_layer_scale=True, layer_scale_init_value=1e-5, vit_num=1,
                resolution=7, e_ratios=None):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        mlp_ratio = e_ratios[str(index)][block_idx]
        if index >= 2 and block_idx > layers[index] - 1 - vit_num:
            stride = 2 if index == 2 else None
            blocks.append(AttnFFN(
                dim, mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer,
                drop=drop_rate, drop_path=block_dpr, use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value, resolution=resolution, stride=stride,
            ))
        else:
            blocks.append(FFN(
                dim, pool_size=pool_size, mlp_ratio=mlp_ratio, act_layer=act_layer,
                drop=drop_rate, drop_path=block_dpr, use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))
    return nn.Sequential(*blocks)


class EfficientFormerV2(nn.Module):
    """Standalone EfficientFormerV2 backbone without mmseg dependencies."""

    def __init__(self, layers, embed_dims=None, mlp_ratios=4, downsamples=None,
                 pool_size=3, norm_layer=nn.BatchNorm2d, act_layer=nn.GELU,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0., use_layer_scale=True,
                 layer_scale_init_value=1e-5, fork_feat=True, vit_num=0,
                 resolution=512, e_ratios=expansion_ratios_L, **kwargs):
        super().__init__()
        self.fork_feat = fork_feat
        self.patch_embed = stem(3, embed_dims[0], act_layer=act_layer)

        network = []
        for i in range(len(layers)):
            stage = meta_blocks(embed_dims[i], i, layers, pool_size=pool_size,
                                mlp_ratio=mlp_ratios, act_layer=act_layer, norm_layer=norm_layer,
                                drop_rate=drop_rate, drop_path_rate=drop_path_rate,
                                use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value,
                                resolution=math.ceil(resolution / (2 ** (i + 2))),
                                vit_num=vit_num, e_ratios=e_ratios)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                asub = i >= 2
                network.append(Embedding(
                    patch_size=down_patch_size, stride=down_stride, padding=down_pad,
                    in_chans=embed_dims[i], embed_dim=embed_dims[i + 1],
                    resolution=math.ceil(resolution / (2 ** (i + 2))),
                    asub=asub, act_layer=act_layer, norm_layer=norm_layer,
                ))

        self.network = nn.ModuleList(network)
        if self.fork_feat:
            self.out_indices = [0, 2, 4, 6]
        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                outs.append(x)
        if self.fork_feat:
            return outs
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        return x


def efficientformerv2_s0(resolution=512, **kwargs):
    return EfficientFormerV2(
        layers=EfficientFormer_depth['S0'], embed_dims=EfficientFormer_width['S0'],
        downsamples=[True, True, True, True], fork_feat=True, drop_path_rate=0.,
        vit_num=2, e_ratios=expansion_ratios_S0, resolution=resolution, **kwargs)


def efficientformerv2_s1(resolution=512, **kwargs):
    return EfficientFormerV2(
        layers=EfficientFormer_depth['S1'], embed_dims=EfficientFormer_width['S1'],
        downsamples=[True, True, True, True], fork_feat=True, drop_path_rate=0.,
        vit_num=2, e_ratios=expansion_ratios_S1, resolution=resolution, **kwargs)


def efficientformerv2_s2(resolution=512, **kwargs):
    return EfficientFormerV2(
        layers=EfficientFormer_depth['S2'], embed_dims=EfficientFormer_width['S2'],
        downsamples=[True, True, True, True], fork_feat=True, drop_path_rate=0.02,
        vit_num=4, e_ratios=expansion_ratios_S2, resolution=resolution, **kwargs)


def efficientformerv2_l(resolution=512, **kwargs):
    return EfficientFormerV2(
        layers=EfficientFormer_depth['L'], embed_dims=EfficientFormer_width['L'],
        downsamples=[True, True, True, True], fork_feat=True, drop_path_rate=0.1,
        vit_num=6, e_ratios=expansion_ratios_L, resolution=resolution, **kwargs)


# ============================================================================
# CGRSeg Decode Head Components (Standalone Implementation)
# ============================================================================

class ConvModule(nn.Module):
    """Pure PyTorch replacement for mmcv.cnn.ConvModule."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, norm_cfg=None, act_cfg=None, bias='auto'):
        super().__init__()
        # Determine bias
        if bias == 'auto':
            bias = norm_cfg is None
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, groups=groups, bias=bias)
        
        # Normalization
        if norm_cfg is not None:
            norm_type = norm_cfg.get('type', 'BN')
            if norm_type in ('BN', 'SyncBN'):
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm_type == 'GN':
                num_groups = norm_cfg.get('num_groups', 32)
                self.norm = nn.GroupNorm(num_groups, out_channels)
            else:
                self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None
        
        # Activation
        if act_cfg is not None:
            act_type = act_cfg.get('type', 'ReLU')
            if act_type == 'ReLU':
                self.act = nn.ReLU(inplace=True)
            elif act_type == 'ReLU6':
                self.act = nn.ReLU6(inplace=True)
            elif act_type == 'GELU':
                self.act = nn.GELU()
            else:
                self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ConvMlp(nn.Module):
    """MLP using 1x1 convs that keeps spatial dims."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.ReLU, norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class RCA(nn.Module):
    """Region-wise Channel Attention module."""
    
    def __init__(self, inp, kernel_size=1, ratio=1, band_kernel_size=11,
                 dw_size=(1, 1), padding=(0, 0), stride=1, square_kernel_size=2, relu=True):
        super().__init__()
        self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size, 
                                   padding=square_kernel_size // 2, groups=inp)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        # Ensure ratio is at least 1 to prevent division by zero
        ratio = max(1, ratio)
        gc = inp // ratio
        # Ensure gc is at least 1 for valid grouped convolution
        gc = max(1, gc)
        self.excite = nn.Sequential(
            nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), 
                      padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), 
                      padding=(band_kernel_size // 2, 0), groups=gc),
            nn.Sigmoid()
        )

    def sge(self, x):
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w
        ge = self.excite(x_gather)
        return ge

    def forward(self, x):
        loc = self.dwconv_hw(x)
        att = self.sge(x)
        out = att * loc
        return out


class RCM(nn.Module):
    """Region-wise Context Module."""
    
    def __init__(self, dim, token_mixer=RCA, norm_layer=nn.BatchNorm2d,
                 mlp_layer=ConvMlp, mlp_ratio=2, act_layer=nn.GELU,
                 ls_init_value=1e-6, drop_path=0., dw_size=11,
                 square_kernel_size=3, ratio=1):
        super().__init__()
        self.token_mixer = token_mixer(dim, band_kernel_size=dw_size, 
                                       square_kernel_size=square_kernel_size, ratio=ratio)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


class DPGHead(nn.Module):
    """Dynamic Point-wise Gating Head."""
    
    def __init__(self, in_ch, mid_ch, pool='att', fusions=None):
        super().__init__()
        if fusions is None:
            fusions = ['channel_mul']
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0
        
        self.inplanes = in_ch
        self.planes = mid_ch
        self.pool = pool
        self.fusions = fusions
        
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(self.inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            nn.init.kaiming_normal_(self.conv_mask.weight, mode='fan_in')
        if self.channel_add_conv is not None:
            nn.init.constant_(self.channel_add_conv[-1].weight, 0)
            nn.init.constant_(self.channel_add_conv[-1].bias, 0)
        if self.channel_mul_conv is not None:
            nn.init.constant_(self.channel_mul_conv[-1].weight, 0)
            nn.init.constant_(self.channel_mul_conv[-1].bias, 0)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x.view(batch, channel, height * width).unsqueeze(1)
            context_mask = self.conv_mask(x).view(batch, 1, height * width)
            context_mask = self.softmax(context_mask).unsqueeze(3)
            context = torch.matmul(input_x, context_mask).view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)
        return context

    def forward(self, x, y):
        context = self.spatial_pool(y)
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class PyramidPoolAgg(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        B, C, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        return torch.cat([F.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)


class FuseBlockMulti(nn.Module):
    def __init__(self, inp, oup, stride=1, norm_cfg=None, activations=None):
        super().__init__()
        if norm_cfg is None:
            norm_cfg = dict(type='BN', requires_grad=True)
        self.stride = stride
        self.norm_cfg = norm_cfg
        assert stride in [1, 2]
        
        if activations is None:
            activations = nn.ReLU
        
        self.fuse1 = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.fuse2 = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()

    def forward(self, x_l, x_h):
        B, C, H, W = x_l.shape
        inp = self.fuse1(x_l)
        sig_act = self.fuse2(x_h)
        sig_act = F.interpolate(self.act(sig_act), size=(H, W), mode='bilinear', align_corners=False)
        out = inp * sig_act
        return out


class NextLayer(nn.Module):
    def __init__(self, block_num, embedding_dim, dw_size, module=RCM, 
                 mlp_ratio=2, token_mixer=RCA, square_kernel_size=3):
        super().__init__()
        self.block_num = block_num
        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(
                module(embedding_dim, token_mixer=token_mixer, dw_size=dw_size, 
                       mlp_ratio=mlp_ratio, square_kernel_size=square_kernel_size)
            )

    def forward(self, x):
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x


class SpatialGatherModule(nn.Module):
    """Aggregate context features according to initial predicted probability distribution."""
    
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, num_classes, height, width = probs.size()
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1).permute(0, 2, 1)
        probs = F.softmax(self.scale * probs, dim=2)
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context


class CGRSegHead(nn.Module):
    """Standalone CGRSeg decode head without mmseg dependencies.
    
    This is a pure PyTorch implementation of the CGRSeg decode head,
    accurately reproducing the architecture from the original mmseg version.
    """

    def __init__(self, in_channels, in_index, channels, num_classes=150,
                 is_dw=True, next_repeat=5, mr=2, dw_size=11, neck_size=11,
                 square_kernel_size=3, module='RCA', ratio=1,
                 dropout_ratio=0.1, align_corners=False):
        super().__init__()
        
        self.in_channels = in_channels
        self.in_index = in_index
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners
        
        norm_cfg = dict(type='SyncBN', requires_grad=True)
        act_cfg = dict(type='ReLU')
        
        self.linear_fuse = ConvModule(
            in_channels=channels, out_channels=channels,
            kernel_size=1, stride=1,
            groups=channels if is_dw else 1,
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        
        self.ppa = PyramidPoolAgg(stride=2)
        act_layer = nn.ReLU6
        module_dict = {'RCA': RCA}
        
        self.trans = NextLayer(next_repeat, sum(self.in_channels), dw_size=neck_size, 
                               mlp_ratio=mr, token_mixer=module_dict[module], 
                               square_kernel_size=square_kernel_size)
        
        self.SIM = nn.ModuleList()
        self.meta = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.SIM.append(FuseBlockMulti(self.in_channels[i], self.channels, 
                                           norm_cfg=norm_cfg, activations=act_layer))
            self.meta.append(RCM(self.in_channels[i], token_mixer=module_dict[module], 
                                 dw_size=dw_size, mlp_ratio=mr, 
                                 square_kernel_size=square_kernel_size, ratio=ratio))
        
        self.conv = nn.ModuleList()
        for i in range(len(self.in_channels) - 1):
            self.conv.append(nn.Conv2d(self.channels, self.in_channels[i], 1))
        
        self.spatial_gather_module = SpatialGatherModule(1)
        self.lgc = DPGHead(channels, channels, pool='att', fusions=['channel_mul'])
        
        # Classification layer
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def cls_seg(self, feat):
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def _transform_inputs(self, inputs):
        """Select features by index."""
        return [inputs[i] for i in self.in_index]

    def forward(self, inputs):
        xx = self._transform_inputs(inputs)
        out = self.ppa(xx)
        out = self.trans(out)
        f_cat = out.split(self.in_channels, dim=1)
        
        results = []
        for i in range(len(self.in_channels) - 1, -1, -1):
            if i == len(self.in_channels) - 1:
                local_tokens = xx[i]
            else:
                local_tokens = xx[i] + self.conv[i](
                    F.interpolate(results[-1], size=xx[i].shape[2:], 
                                  mode='bilinear', align_corners=False)
                )
            global_semantics = f_cat[i]
            local_tokens = self.meta[i](local_tokens)
            flag = self.SIM[i](local_tokens, global_semantics)
            results.append(flag)
        
        x = results[-1]
        _c = self.linear_fuse(x)
        prev_output = self.cls_seg(_c)
        
        context = self.spatial_gather_module(x, prev_output)
        object_context = self.lgc(x, context) + x
        output = self.cls_seg(object_context)
        
        return output


# ============================================================================
# Complete CGRSeg Model
# ============================================================================

@dataclass
class CGRSegConfig:
    """Configuration for CGRSeg model variants."""
    
    # Backbone configuration
    backbone_variant: str = 'S1'  # S0, S1, S2, or L
    
    # Decode head configuration  
    in_channels: List[int] = field(default_factory=lambda: [48, 120, 224])
    in_index: List[int] = field(default_factory=lambda: [1, 2, 3])
    channels: int = 128
    num_classes: int = 150
    is_dw: bool = True
    dw_size: int = 11
    neck_size: int = 11
    next_repeat: int = 5
    square_kernel_size: int = 3
    ratio: int = 1
    module: str = 'RCA'
    dropout_ratio: float = 0.1
    align_corners: bool = False
    
    # Input configuration
    input_size: int = 512


# Pre-defined configurations matching the local_configs
CGRSEG_CONFIGS = {
    't': CGRSegConfig(  # cgrseg-t (S1 backbone)
        backbone_variant='S1',
        in_channels=[48, 120, 224],
        in_index=[1, 2, 3],
        channels=128,
        dw_size=11,
        neck_size=11,
        next_repeat=5,
        square_kernel_size=3,
        ratio=1,
    ),
    'b': CGRSegConfig(  # cgrseg-b (S2 backbone)
        backbone_variant='S2',
        in_channels=[64, 144, 288],
        in_index=[1, 2, 3],
        channels=256,
        dw_size=11,
        neck_size=11,
        next_repeat=6,
        square_kernel_size=3,
        ratio=1,
    ),
    'l': CGRSegConfig(  # cgrseg-l (L backbone)
        backbone_variant='L',
        in_channels=[80, 192, 384],
        in_index=[1, 2, 3],
        channels=256,
        dw_size=9,
        neck_size=9,
        next_repeat=5,
        square_kernel_size=3,
        ratio=1,
    ),
}


class CGRSeg(nn.Module):
    """Complete CGRSeg model combining backbone and decode head.
    
    This is a standalone implementation that does not require mmcv or mmsegmentation.
    It accurately reproduces the CGRSeg architecture as specified in the configuration files.
    
    Args:
        config: CGRSegConfig or str ('t', 'b', 'l') for predefined configs
        input_size: Input image resolution (default: 512)
    """

    def __init__(self, config='t', input_size=512):
        super().__init__()
        
        # Handle string config
        if isinstance(config, str):
            if config not in CGRSEG_CONFIGS:
                raise ValueError(f"Unknown config '{config}'. Choose from {list(CGRSEG_CONFIGS.keys())}")
            config = CGRSEG_CONFIGS[config]
        
        self.config = config
        self.input_size = input_size
        
        # Build backbone
        backbone_builders = {
            'S0': efficientformerv2_s0,
            'S1': efficientformerv2_s1,
            'S2': efficientformerv2_s2,
            'L': efficientformerv2_l,
        }
        
        if config.backbone_variant not in backbone_builders:
            raise ValueError(f"Unknown backbone variant '{config.backbone_variant}'")
        
        self.backbone = backbone_builders[config.backbone_variant](resolution=input_size)
        
        # Build decode head
        self.decode_head = CGRSegHead(
            in_channels=config.in_channels,
            in_index=config.in_index,
            channels=config.channels,
            num_classes=config.num_classes,
            is_dw=config.is_dw,
            next_repeat=config.next_repeat,
            mr=2,
            dw_size=config.dw_size,
            neck_size=config.neck_size,
            square_kernel_size=config.square_kernel_size,
            module=config.module,
            ratio=config.ratio,
            dropout_ratio=config.dropout_ratio,
            align_corners=config.align_corners,
        )
        
        self.align_corners = config.align_corners

    def forward(self, x):
        """Forward pass returning logits at input resolution."""
        # Get multi-scale features from backbone
        features = self.backbone(x)
        
        # Decode head prediction
        logits = self.decode_head(features)
        
        # Upsample to input resolution if needed
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', 
                                   align_corners=self.align_corners)
        
        return logits


# ============================================================================
# Main Demo
# ============================================================================

def count_parameters(model):
    """Count trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def main():
    parser = argparse.ArgumentParser(description='CGRSeg Standalone Model Demo')
    parser.add_argument('--model', type=str, default='t', choices=['t', 'b', 'l'],
                        help='Model variant: t(iny), b(ase), l(arge)')
    parser.add_argument('--input_size', type=int, default=512,
                        help='Input image size')
    args = parser.parse_args()
    
    print("=" * 70)
    print("CGRSeg Standalone Model Demo")
    print("=" * 70)
    print(f"\nModel variant: {args.model}")
    print(f"Input size: {args.input_size}x{args.input_size}")
    print("\nNote: This implementation does not require mmcv or mmsegmentation.\n")
    
    # Build model
    model = CGRSeg(config=args.model, input_size=args.input_size)
    model.eval()
    
    # Create dummy input
    dummy = torch.randn(1, 3, args.input_size, args.input_size)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy)
    
    # Print model structure
    print("=" * 70)
    print("Model Structure:")
    print("=" * 70)
    print(model)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Parameters:   {count_parameters(model):.2f}M")
    
    # Print backbone feature shapes
    print("\nBackbone feature shapes:")
    with torch.no_grad():
        features = model.backbone(dummy)
        for i, feat in enumerate(features):
            print(f"  Stage {i}: {tuple(feat.shape)}")


if __name__ == "__main__":
    main()
