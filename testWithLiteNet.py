import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from thop import profile
except ImportError:
    print("未安装 thop，请运行: pip install thop")
    profile = None

# 依赖 timm 的基础层，如果不想安装 timm，可以手动实现这两个简单的 helper
from timm.layers import DropPath, trunc_normal_, to_2tuple

# ============================================================================
# MobileNetV2 Backbone -> 输出 F2 / F3 / F4 三层特征
#   F2: 1/8  分辨率，投到 48 通道
#   F3: 1/16 分辨率，投到 120 通道
#   F4: 1/32 分辨率，投到 224 通道
# ============================================================================
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_ch, out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, stride=1))
        else:
            hidden_dim = inp

        layers.append(
            ConvBNReLU(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                groups=hidden_dim,
            )
        )
        layers.extend([
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2Backbone(nn.Module):
    """
    MobileNetV2 轻量 backbone
    - 输入:  B, 3, H, W
    - 输出:  [F2, F3, F4]
        F2: 1/8,  C=32
        F3: 1/16, C=96
        F4: 1/32, C=160
    """
    def __init__(self):
        super().__init__()

        # Stem: 1/2
        self.stem = ConvBNReLU(3, 64, kernel_size=3, stride=2)

        # (t, c, n, s) 展开的几个 stage
        self.stage1 = self._make_stage(64, 32, n=1, stride=1, expand_ratio=1)  # 1/2
        self.stage2 = self._make_stage(32, 48, n=2, stride=2, expand_ratio=6)  # 1/4

        # F2: 1/8, C=32
        self.stage3 = self._make_stage(48, 64, n=3, stride=2, expand_ratio=6)  # 1/8

        # F3: 1/16, C=96
        self.stage4 = self._make_stage(64, 128, n=4, stride=2, expand_ratio=6)  # 1/16
        self.stage5 = self._make_stage(128, 192, n=3, stride=1, expand_ratio=6)  # 1/16

        # F4: 1/32, C=160
        self.stage6 = self._make_stage(192, 256, n=3, stride=2, expand_ratio=6) # 1/32

        self.F2 = 64
        self.F3 = 192
        self.F4 = 256

        self._init_weights()

    def _make_stage(self, inp, oup, n, stride, expand_ratio):
        layers = []
        for i in range(n):
            s = stride if i == 0 else 1
            layers.append(InvertedResidual(inp, oup, stride=s, expand_ratio=expand_ratio))
            inp = oup
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)      # 1/2
        x = self.stage1(x)    # 1/2
        x = self.stage2(x)    # 1/4

        x = self.stage3(x)    # 1/8
        f2 = x                # C = 32

        x = self.stage4(x)    # 1/16
        x = self.stage5(x)    # 1/16
        f3 = x                # C = 96

        x = self.stage6(x)    # 1/32
        f4 = x                # C = 160

        return [f2, f3, f4]
    
# ============================================================================
# Unet backbone for tiny model test
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    你当前在用的 double full conv，可以直接沿用已有实现；
    这里写一份参考版，注意避免和你原来的重复定义。
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DSConv(nn.Module):
    """
    Depthwise-Separable Conv:
      - depthwise 3x3 (groups=in_ch)
      - pointwise 1x1
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.dw_act = nn.ReLU(inplace=True)

        self.pw = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.pw_act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw_act(self.dw_bn(self.dw(x)))
        x = self.pw_act(self.pw_bn(self.pw(x)))
        return x


class DSConvBlock(nn.Module):
    """
    double DSConv，对应 Unet 里的双卷积结构，但全部是分离卷积。
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            DSConv(in_ch, out_ch),
            DSConv(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class UNetBackbone(nn.Module):
    """
    Hybrid 版 U-Net Encoder：
      - enc1, enc2: full ConvBlock（高分辨率，保表达力）
      - enc3, enc4, enc5: DSConvBlock（低分辨率，省 FLOPs）
    尺度仍然对齐 CGRSegHead:
      F2: 1/8,  C=64
      F3: 1/16, C=96
      F4: 1/32, C=128
    """
    def __init__(self, in_channels=3):
        super().__init__()

        # 通道配置，和你 9.14G 那版保持一致
        c1 = 32   # 1/2
        c2 = 48   # 1/4
        c3 = 64   # 1/8
        c4 = 96   # 1/16
        c5 = 128  # 1/32

        # 1/2: full conv
        self.enc1 = ConvBlock(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2)  # -> 1/2

        # 1/4: full conv
        self.enc2 = ConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(2)  # -> 1/4

        # 1/8: depthwise-separable
        self.enc3 = ConvBlock(c2, c3)
        self.pool3 = nn.MaxPool2d(2)  # -> 1/8

        # 1/16: depthwise-separable
        self.enc4 = DSConvBlock(c3, c4)
        self.pool4 = nn.MaxPool2d(2)  # -> 1/16

        # 1/32: depthwise-separable
        self.enc5 = DSConvBlock(c4, c5)

        self.F2 = c3  # 64
        self.F3 = c4  # 96
        self.F4 = c5  # 128

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: B, 3, H, W
        x1 = self.enc1(x)    # 1/2
        p1 = self.pool1(x1)  # 1/2 -> 1/4

        x2 = self.enc2(p1)   # 1/4
        p2 = self.pool2(x2)  # 1/4 -> 1/8

        x3 = self.enc3(p2)   # 1/8 -> F2
        p3 = self.pool3(x3)  # 1/8 -> 1/16

        x4 = self.enc4(p3)   # 1/16 -> F3
        p4 = self.pool4(x4)  # 1/16 -> 1/32

        x5 = self.enc5(p4)   # 1/32 -> F4

        return [x3, x4, x5]

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# STDC Block: Short-Term Dense Concatenation
#    - in_ch: 输入通道
#    - out_ch: 输出通道
#    - stride: 1 / 2 (2 时下采样)
# ============================================================

class STDCBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, num_splits=4):
        super().__init__()
        assert num_splits in [2, 3, 4]
        self.stride = stride
        self.in_ch = in_ch
        self.out_ch = out_ch

        # 把 out_ch 按 1/2, 1/4, 1/8, ... 的方式分段
        if num_splits == 4:
            c1 = out_ch // 2
            c2 = out_ch // 4
            c3 = out_ch // 8
            c4 = out_ch - (c1 + c2 + c3)
            ch_list = [c1, c2, c3, c4]
        elif num_splits == 3:
            c1 = out_ch // 2
            c2 = out_ch // 4
            c3 = out_ch - (c1 + c2)
            ch_list = [c1, c2, c3]
        else:  # num_splits == 2
            c1 = out_ch // 2
            c2 = out_ch - c1
            ch_list = [c1, c2]

        self.num_splits = num_splits
        self.ch_list = ch_list

        convs = []

        # 第一个分支负责下采样（stride=2 时）
        convs.append(
            ConvBNReLU(
                in_ch,
                ch_list[0],
                kernel_size=3,
                stride=stride
            )
        )

        # 后续分支串行堆叠
        for i in range(1, num_splits):
            convs.append(
                ConvBNReLU(
                    ch_list[i-1],
                    ch_list[i],
                    kernel_size=3,
                    stride=1
                )
            )

        self.convs = nn.ModuleList(convs)

        # 残差：如果 stride=2 或者通道不一致，用 1x1 conv 对齐
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = None

    def forward(self, x):
        out_list = []
        x_i = x
        for conv in self.convs:
            x_i = conv(x_i)
            out_list.append(x_i)

        # dense concat
        out = torch.cat(out_list, dim=1)  # (B, out_ch, H', W')

        if self.downsample is not None:
            res = self.downsample(x)
        else:
            res = x

        return F.relu(out + res, inplace=True)


# ============================================================
# STDC Backbone (Tiny-ish)
#  输出:
#   F2: 1/8   分辨率
#   F3: 1/16  分辨率
#   F4: 1/32  分辨率
# ============================================================

class STDCBackbone(nn.Module):
    def __init__(self, in_channels=3, base_ch=32, num_splits=4):
        """
        base_ch 可以调节整体规模：
          - 32: 比较轻
          - 48/64: 更强一点，FLOPs 会上去
        """
        super().__init__()

        # Stem: 1/2
        self.stem = ConvBNReLU(in_channels, base_ch, kernel_size=3, stride=2)  # 1/2

        # Stage1: 1/4
        self.stage1 = nn.Sequential(
            STDCBlock(base_ch, base_ch * 2, stride=2, num_splits=num_splits),   # 1/4
            STDCBlock(base_ch * 2, base_ch * 2, stride=1, num_splits=num_splits),
        )

        # Stage2: 1/8  -> F2
        self.stage2 = nn.Sequential(
            STDCBlock(base_ch * 2, base_ch * 3, stride=2, num_splits=num_splits),  # 1/8
            STDCBlock(base_ch * 3, base_ch * 3, stride=1, num_splits=num_splits),
        )

        # Stage3: 1/16 -> F3
        self.stage3 = nn.Sequential(
            STDCBlock(base_ch * 3, base_ch * 4, stride=2, num_splits=num_splits),  # 1/16
            STDCBlock(base_ch * 4, base_ch * 4, stride=1, num_splits=num_splits),
        )

        # Stage4: 1/32 -> F4
        self.stage4 = nn.Sequential(
            STDCBlock(base_ch * 4, base_ch * 5, stride=2, num_splits=num_splits),  # 1/32
            STDCBlock(base_ch * 5, base_ch * 5, stride=1, num_splits=num_splits),
        )

        # 暴露给 decode head
        self.F2 = base_ch * 3   # stage2 输出通道
        self.F3 = base_ch * 4   # stage3
        self.F4 = base_ch * 5   # stage4

    def forward(self, x):
        x = self.stem(x)            # 1/2
        x = self.stage1(x)          # 1/4

        x2 = self.stage2(x)         # 1/8
        x3 = self.stage3(x2)        # 1/16
        x4 = self.stage4(x3)        # 1/32

        F2 = x2
        F3 = x3
        F4 = x4

        return [F2, F3, F4]
    
# ============================================================================
# EfficientNet
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    """
    EfficientNet 风格的 SE 模块:
      - 全局池化 CxHxW -> Cx1x1
      - 两层全连接 (用 1x1 conv 实现)
      - Sigmoid 得到通道权重
    """
    def __init__(self, in_ch: int, se_ratio: float = 0.25):
        super().__init__()
        hidden = max(1, int(in_ch * se_ratio))
        self.fc1 = nn.Conv2d(in_ch, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, in_ch, kernel_size=1)

    def forward(self, x):
        # 全局平均池化
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.silu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s
        

class MBConv(nn.Module):
    """
    EfficientNet 风格的 MBConv block:
      - expand (1x1)
      - depthwise conv (3x3 or 5x5)
      - SE
      - projection (1x1)
      - 可选残差 + Stochastic Depth (这里只留残差)
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        se_ratio: float = 0.25,
    ):
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.use_res = (stride == 1 and in_ch == out_ch)
        mid = in_ch * expand_ratio

        layers = []

        # 1) Expand
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_ch, mid, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(mid))
            layers.append(nn.SiLU(inplace=True))
        else:
            mid = in_ch

        # 2) Depthwise
        layers.append(
            nn.Conv2d(
                mid,
                mid,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=mid,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(mid))
        layers.append(nn.SiLU(inplace=True))

        self.pre_se = nn.Sequential(*layers)

        # 3) Squeeze-and-Excitation
        self.se = SqueezeExcite(mid, se_ratio=se_ratio)

        # 4) Projection
        self.project = nn.Sequential(
            nn.Conv2d(mid, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        out = self.pre_se(x)
        out = self.se(out)
        out = self.project(out)
        if self.use_res:
            out = out + x
        return out

class EfficientNetBackbone(nn.Module):
    """
    EfficientNet 风格 Tiny Backbone：
      输出三个尺度:
        F2: 1/8   分辨率
        F3: 1/16  分辨率
        F4: 1/32  分辨率
      用于接你现有的 CGRSegHead_Tiny。
    """
    def __init__(self, in_channels: int = 3, width_mult: float = 1.0):
        super().__init__()

        def ch(c):
            # 通道缩放，至少 8
            return max(8, int(c * width_mult))

        # 这一套配置是 B0 的一个缩小版
        c1 = ch(24)   # stem & stage1
        c2 = ch(48)   # 1/4
        c3 = ch(64)   # 1/8  -> F2
        c4 = ch(96)   # 1/16 -> F3
        c5 = ch(128)  # 1/32 -> F4
        # Stem 1/2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )  # -> 1/2

        # Stage1: 1/2
        self.stage1 = nn.Sequential(
            MBConv(c1, c1, kernel_size=3, stride=1, expand_ratio=1, se_ratio=0.25),
        )

        # Stage2: 1/4
        self.stage2 = nn.Sequential(
            MBConv(c1, c2, kernel_size=3, stride=2, expand_ratio=6, se_ratio=0.25),
            MBConv(c2, c2, kernel_size=3, stride=1, expand_ratio=6, se_ratio=0.25),
        )

        # Stage3: 1/8  -> F2
        self.stage3 = nn.Sequential(
            MBConv(c2, c3, kernel_size=5, stride=2, expand_ratio=6, se_ratio=0.25),
            MBConv(c3, c3, kernel_size=5, stride=1, expand_ratio=6, se_ratio=0.25),
        )

        # Stage4: 1/16 -> F3
        self.stage4 = nn.Sequential(
            MBConv(c3, c4, kernel_size=3, stride=2, expand_ratio=6, se_ratio=0.25),
            MBConv(c4, c4, kernel_size=3, stride=1, expand_ratio=6, se_ratio=0.25),
            MBConv(c4, c4, kernel_size=3, stride=1, expand_ratio=6, se_ratio=0.25),
        )

        # Stage5: 1/32 -> F4
        self.stage5 = nn.Sequential(
            MBConv(c4, c5, kernel_size=5, stride=2, expand_ratio=6, se_ratio=0.25),
            MBConv(c5, c5, kernel_size=5, stride=1, expand_ratio=6, se_ratio=0.25),
        )

        self.F2 = c3
        self.F3 = c4
        self.F4 = c5

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)      # 1/2
        x = self.stage1(x)    # 1/2
        x = self.stage2(x)    # 1/4

        x2 = self.stage3(x)   # 1/8
        x3 = self.stage4(x2)  # 1/16
        x4 = self.stage5(x3)  # 1/32

        # 对齐 decode head 的接口
        return [x2, x3, x4]



# ============================================================================
# Part 2: CGRSeg Decode Head (Specific to Tiny config)
# ============================================================================

class RCA(nn.Module):
    """
    Region-wise Channel Attention (论文中的矩形自校准注意力)
    核心功能：通过十字形池化和带状卷积捕获矩形上下文。
    """
    def __init__(self, inp, kernel_size=1, ratio=1, band_kernel_size=11, square_kernel_size=2):
        super().__init__()
        # Step 1: Local Feature Extraction (局部特征提取)
        # 使用方形卷积提取局部细节
        self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size, padding=square_kernel_size // 2, groups=inp)
        
        # Step 2: Cross-shaped Pooling (十字形池化)
        # 分别沿水平和垂直方向进行全局池化，保留长距离依赖
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1)) # 保持 H，压缩 W -> [B, C, H, 1]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None)) # 保持 W，压缩 H -> [B, C, 1, W]
        
        gc = max(1, inp // ratio)
        
        # Step 4: Band Convolution Excitation (带状卷积激励)
        # 使用非对称卷积 (1xK, Kx1) 模拟矩形感受野，生成注意力权重
        self.excite = nn.Sequential(
            nn.Conv2d(inp, gc, (1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc), nn.ReLU(inplace=True),
            nn.Conv2d(gc, inp, (band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.Sigmoid() # 生成 0~1 的注意力图
        )

    def forward(self, x):
        loc = self.dwconv_hw(x)       # 局部特征
        x_h = self.pool_h(x)          # 水平上下文
        x_w = self.pool_w(x)          # 垂直上下文
        
        # Step 3: Cross-shaped Aggregation & Step 5: Attention
        # 广播相加后通过 excite 生成注意力，最后调制局部特征
        att = self.excite(x_h + x_w)  
        return att * loc

class RCM(nn.Module):
    """
    Region-wise Context Module (RCM 模块)
    包含 RCA 注意力和 MLP，用于空间特征重建。
    """
    def __init__(self, dim, dw_size=11):
        super().__init__()
        # 使用 RCA 进行 token mixing
        self.token_mixer = RCA(dim, band_kernel_size=dw_size, square_kernel_size=3, ratio=1)
        self.norm = nn.BatchNorm2d(dim)
        # MLP 用于 channel mixing
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim*2, 1), nn.Identity(), nn.GELU(), nn.Dropout(0), nn.Conv2d(dim*2, dim, 1)
        )
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.mlp(self.norm(x))
        return shortcut + x.mul(self.gamma.view(1, -1, 1, 1))

class DPGHead(nn.Module):
    """
    Dynamic Prototype Guided Head (DPG Head)
    功能：Stage A (Global Context Pooling) + Stage B (Class -> Pixel Modulation)
    """
    def __init__(self, channels):
        super().__init__()
        self.conv_mask = nn.Conv2d(channels, 1, 1)
        self.softmax = nn.Softmax(dim=2)
        
        # Channel Modulation MLP (用于生成门控权重)
        self.channel_mul_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1), nn.LayerNorm([channels, 1, 1]),
            nn.ReLU(inplace=True), nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x, y):
        # Step A: Attention-based Global Context Pooling
        # 将输入的类别原型 y (或特征) 聚合成全局上下文向量
        B, C, H, W = y.shape
        input_x = y.view(B, C, H * W).unsqueeze(1)
        context_mask = self.softmax(self.conv_mask(y).view(B, 1, H * W)).unsqueeze(3) # 空间注意力
        context = torch.matmul(input_x, context_mask).view(B, C, 1, 1) # 加权聚合 -> [B, C, 1, 1]
        
        # Step B: Channel Gating (Channel Multiplication)
        # 利用全局上下文对输入特征 x 进行通道级调制
        return x * torch.sigmoid(self.channel_mul_conv(context))

class CGRSegHead_Tiny(nn.Module):
    """
    Tiny 版解码头
    - 默认配合 MobileNetV2:
        F2: C=32, 1/8
        F3: C=96, 1/16
        F4: C=160, 1/32
    - 统一投到 embedding 通道: 128
    """
    def __init__(self, num_classes=150, in_channels=[64, 192, 256]):
        super().__init__()
        # 用 MobileNetV2 原生特征维度
        self.in_channels = in_channels
        channels = 128 # increase the channel can improve the performance, but only slight improvement(original 128)
        
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        # Global Context Modeling: RCM 堆叠，处理 PPA 聚合后的特征
        self.trans_blocks = nn.ModuleList([
            RCM(sum(self.in_channels), dw_size=11) for _ in range(5)
        ])

        self.SIM = nn.ModuleList()
        self.meta = nn.ModuleList()
        self.conv = nn.ModuleList()
        
        for i in range(len(in_channels)):
            # SIM (Sigmoid-gated Fusion): 门控融合模块
            self.SIM.append(nn.Sequential(
                nn.Conv2d(in_channels[i], channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.Conv2d(in_channels[i], channels, 1, bias=False),
                nn.BatchNorm2d(channels),
            ))
            # Meta: 每个尺度自己的 RCM，用于 refine local tokens
            self.meta.append(RCM(in_channels[i], dw_size=11))
            
            if i < len(in_channels) - 1:
                # 用于把上一级 embedding 映射回该层的原生 in_channels
                self.conv.append(nn.Conv2d(channels, in_channels[i], 1))

        self.lgc = DPGHead(channels)
        self.cls_seg = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(channels, num_classes, 1)
        )

    def forward(self, features):
        # features: [C32, C96, C160] (Scales 1/8, 1/16, 1/32)
        
        # 1. PPA (Pyramid Pooling Aggregation)
        # 将不同尺度的特征下采样到最小分辨率并拼接
        H, W = (features[-1].shape[2] - 1) // 2 + 1, (features[-1].shape[3] - 1) // 2 + 1
        ppa_in = torch.cat(
            [F.adaptive_avg_pool2d(f, (H, W)) for f in features],
            dim=1
        )  # 通道数 = 32+96+160 = 288
        
        # 2. Global Context Modeling (RCM 堆叠)
        # 在低分辨率下利用堆叠的 RCM 提取强全局语义
        out = ppa_in
        for block in self.trans_blocks:
            out = block(out)
        
        # 3. Split: 将全局特征拆回对应通道
        f_cat = out.split([channel for channel in self.in_channels], dim=1)
        
        # 4. Top-Down Hierarchical Fusion (自顶向下层级融合)
        results = []
        for i in range(2, -1, -1):  # 2,1,0 (从最深层/最粗糙层开始)
            if i == 2:
                local_tokens = features[i]
            else:
                # 上采样上一级结果并与当前级特征融合
                upsampled = F.interpolate(
                    results[-1],
                    size=features[i].shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                local_tokens = features[i] + self.conv[i](upsampled)
            
            # 当前尺度的 RCM Refinement
            global_sem = f_cat[i]
            local_tokens = self.meta[i](local_tokens)

            # SIM Fuse: 使用 Sigmoid 门控机制融合 Local 和 Global 特征
            sim_block = self.SIM[i]
            inp = sim_block[0:2](local_tokens)
            sig_act = sim_block[2:4](global_sem)
            sig_act = F.interpolate(
                F.relu6(sig_act + 3) / 6,
                size=inp.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            results.append(inp * sig_act)
            
        x = results[-1] # 最精细层特征

        # 5. DPG Head: Prototype Guided Refinement
        _c = self.linear_fuse(x)
        prev_pred = self.cls_seg(_c) # Initial Prediction

        # Step 5.1: Prototype Extraction (Pixel -> Class)
        # 使用初始预测将像素特征聚合为类别原型
        B, N, H, W = prev_pred.shape
        probs = F.softmax(prev_pred.view(B, N, -1), dim=2)
        feats = x.view(B, x.shape[1], -1).permute(0, 2, 1)
        context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3) # [B, C, K, 1]

        # Step 5.2: Dynamic Guidance (Class -> Pixel)
        # DPG Head 内部再次汇聚原型并对 x 进行调制
        final_out = self.lgc(x, context) + x
        
        return self.cls_seg(final_out)

# ============================================================================
# Part 3: Full Standalone Model
# ============================================================================

class SGTinyNet(nn.Module):
    def __init__(self, in_channels = 3, num_classes=150, input_size=512):
        super().__init__()
        #self.backbone = MobileNetV2Backbone()
        #self.backbone = UNetBackbone(in_channels=in_channels)
        #self.backbone = STDCBackbone(in_channels=in_channels, base_ch=32, num_splits=4)
        self.backbone = EfficientNetBackbone(in_channels=in_channels)
        
        in_channels = [self.backbone.F2, self.backbone.F3, self.backbone.F4]

        self.decode_head = CGRSegHead_Tiny(num_classes=num_classes, in_channels=in_channels)
        
    def forward(self, x):
        # 1. Backbone
        features = self.backbone(x)
        
        # 2. Decode Head
        logits = self.decode_head(features)
        
        # 3. Resize to Input Size
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
            
        return logits

# ============================================================================
# Usage Demo
# ============================================================================

if __name__ == "__main__":
    # 1. initualize model
    input_size = 512
    num_classes = 2
    model = SGTinyNet(num_classes=num_classes, input_size=input_size)
    model.eval()

    # 2. create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)

    print(f"Model Initialized. Input size: {dummy_input.shape}")

    # 3. compute Parameters 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_params / 1e6:.2f} M")

    # 4. compute FLOPs
    if profile is not None:
        # profile 会返回总的 (FLOPs, params)，通过 verbose=False 关闭层级打印
        flops, params = profile(model, inputs=(dummy_input, ), verbose=False)
        
        # 转换为 GFLOPs (1 GFLOPs = 10^9 FLOPs)
        print(f"FLOPs : {flops / 1e9:.2f} G  (Resolution: {input_size}x{input_size})")
    else:
        print("Skipping FLOPs calculation (thop library missing).")

    # 5. simple test forward pass to verify output shape
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output Shape: {output.shape}")
    print("Done!")
