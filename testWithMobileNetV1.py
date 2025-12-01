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
        self.stem = ConvBNReLU(3, 32, kernel_size=3, stride=2)

        # (t, c, n, s) 展开的几个 stage
        self.stage1 = self._make_stage(32, 16, n=1, stride=1, expand_ratio=1)  # 1/2
        self.stage2 = self._make_stage(16, 24, n=2, stride=2, expand_ratio=6)  # 1/4

        # F2: 1/8, C=32
        self.stage3 = self._make_stage(24, 32, n=3, stride=2, expand_ratio=6)  # 1/8

        # F3: 1/16, C=96
        self.stage4 = self._make_stage(32, 64, n=4, stride=2, expand_ratio=6)  # 1/16
        self.stage5 = self._make_stage(64, 96, n=3, stride=1, expand_ratio=6)  # 1/16

        # F4: 1/32, C=160
        self.stage6 = self._make_stage(96, 160, n=3, stride=2, expand_ratio=6) # 1/32

        self.F2 = 32
        self.F3 = 96
        self.F4 = 160

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
    def __init__(self, num_classes=150, in_channels=[32, 96, 160]):
        super().__init__()
        # 用 MobileNetV2 原生特征维度
        self.in_channels = in_channels
        channels = 128
        
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
        f_cat = out.split([32, 96, 160], dim=1)
        
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
        self.backbone = MobileNetV2Backbone()
        
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
