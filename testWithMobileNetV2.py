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
        # pw
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, stride=1))
        else:
            hidden_dim = inp

        # dw
        layers.append(
            ConvBNReLU(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                groups=hidden_dim,
            )
        )
        # pw-linear
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
    轻量级 MobileNetV2 backbone
    - 输入: B,3,H,W
    - 输出: [F2, F3, F4]
        F2: 1/8,  通道 48
        F3: 1/16, 通道 120
        F4: 1/32, 通道 224
    这里把原始的通道 [32, 96, 160] 用 1x1 conv 投到 [48, 120, 224]，
    """
    def __init__(self, out_channels=(48, 120, 224)):
        super().__init__()

        # ==== Stem: 1/2 ====
        self.stem = ConvBNReLU(3, 32, kernel_size=3, stride=2)

        # MobileNetV2 配置: (t, c, n, s)
        # 这里只是展开成若干 stage，方便取 F2/F3/F4
        # 32x -> 16x -> 8x -> 4x 等比例按 stride 下采样
        self.stage1 = self._make_stage(32, 16, n=1, stride=1, expand_ratio=1)  # 1/2
        self.stage2 = self._make_stage(16, 24, n=2, stride=2, expand_ratio=6)  # 1/4
        self.stage3 = self._make_stage(24, 32, n=3, stride=2, expand_ratio=6)  # 1/8 -> F2 raw (C=32)
        self.stage4 = self._make_stage(32, 64, n=4, stride=2, expand_ratio=6)  # 1/16
        self.stage5 = self._make_stage(64, 96, n=3, stride=1, expand_ratio=6)  # 1/16 -> F3 raw (C=96)
        self.stage6 = self._make_stage(96, 160, n=3, stride=2, expand_ratio=6) # 1/32 -> F4 raw (C=160)
        # stage7 (320 通道) 不参与 RCHead，就不建了 / 或者你想要可以再加

        # 原始通道
        c2_raw, c3_raw, c4_raw = 32, 96, 160
        c2, c3, c4 = out_channels

        # 适配你锁死的维度 [48, 120, 224]
        self.f2_proj = nn.Conv2d(c2_raw, c2, kernel_size=1, bias=False)
        self.f3_proj = nn.Conv2d(c3_raw, c3, kernel_size=1, bias=False)
        self.f4_proj = nn.Conv2d(c4_raw, c4, kernel_size=1, bias=False)

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
        # x: B,3,H,W
        x = self.stem(x)      # 1/2
        x = self.stage1(x)    # 1/2
        x = self.stage2(x)    # 1/4

        x = self.stage3(x)    # 1/8
        f2 = self.f2_proj(x)  # (B, 48, H/8, W/8)

        x = self.stage4(x)    # 1/16
        x = self.stage5(x)    # 1/16
        f3 = self.f3_proj(x)  # (B, 120, H/16, W/16)

        x = self.stage6(x)    # 1/32
        f4 = self.f4_proj(x)  # (B, 224, H/32, W/32)

        return [f2, f3, f4]




# ============================================================================
# Part 2: CGRSeg Decode Head (Specific to Tiny config)
# ============================================================================

class RCA(nn.Module):
    """Region-wise Channel Attention"""
    def __init__(self, inp, kernel_size=1, ratio=1, band_kernel_size=11, square_kernel_size=2):
        super().__init__()
        self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size, padding=square_kernel_size // 2, groups=inp)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        gc = max(1, inp // ratio)
        self.excite = nn.Sequential(
            nn.Conv2d(inp, gc, (1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc), nn.ReLU(inplace=True),
            nn.Conv2d(gc, inp, (band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.Sigmoid()
        )

    def forward(self, x):
        loc = self.dwconv_hw(x)
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        att = self.excite(x_h + x_w)
        return att * loc

class RCM(nn.Module):
    """Region-wise Context Module"""
    def __init__(self, dim, dw_size=11):
        super().__init__()
        self.token_mixer = RCA(dim, band_kernel_size=dw_size, square_kernel_size=3, ratio=1)
        self.norm = nn.BatchNorm2d(dim)
        # MLP Ratio is fixed to 2 for Tiny
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
    """Dynamic Point-wise Gating Head"""
    def __init__(self, channels):
        super().__init__()
        self.conv_mask = nn.Conv2d(channels, 1, 1)
        self.softmax = nn.Softmax(dim=2)
        self.channel_mul_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1), nn.LayerNorm([channels, 1, 1]),
            nn.ReLU(inplace=True), nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x, y):
        # Spatial Pool (Att)
        B, C, H, W = y.shape
        input_x = y.view(B, C, H * W).unsqueeze(1)
        context_mask = self.softmax(self.conv_mask(y).view(B, 1, H * W)).unsqueeze(3)
        context = torch.matmul(input_x, context_mask).view(B, C, 1, 1)
        
        # Channel Mul Fusion
        return x * torch.sigmoid(self.channel_mul_conv(context))

class CGRSegHead_Tiny(nn.Module):
    """
    Fixed config for Tiny:
    Input Channels: [48, 120, 224]
    Embedding Channel: 128
    """
    def __init__(self, num_classes=150):
        super().__init__()
        in_channels = [48, 120, 224]
        channels = 128
        
        # Projections
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU()
        )
        
        # Transformer Stage (Next Layer)
        # For Tiny: next_repeat=5, mr=2, neck_size=11
        self.trans_blocks = nn.ModuleList([
            RCM(sum(in_channels), dw_size=11) for _ in range(5)
        ])
        
        # Stages
        self.SIM = nn.ModuleList()
        self.meta = nn.ModuleList()
        self.conv = nn.ModuleList()
        
        for i in range(len(in_channels)):
            # Fuse Block Multi (SIM)
            self.SIM.append(nn.Sequential(
                nn.Conv2d(in_channels[i], channels, 1, bias=False), nn.BatchNorm2d(channels), # Fuse1
                nn.Conv2d(in_channels[i], channels, 1, bias=False), nn.BatchNorm2d(channels), # Fuse2
            ))
            # Meta (RCM)
            self.meta.append(RCM(in_channels[i], dw_size=11))
            
            if i < len(in_channels) - 1:
                self.conv.append(nn.Conv2d(channels, in_channels[i], 1))

        self.lgc = DPGHead(channels)
        self.cls_seg = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(channels, num_classes, 1))

    def forward(self, features):
        # features: [C48, C120, C224] (Scales 1/8, 1/16, 1/32)
        
        # 1. PPA (Pyramid Pool Aggregation)
        H, W = (features[-1].shape[2] - 1) // 2 + 1, (features[-1].shape[3] - 1) // 2 + 1
        ppa_in = torch.cat([F.adaptive_avg_pool2d(f, (H, W)) for f in features], dim=1)
        
        # 2. Next Layer Transformer
        out = ppa_in
        for block in self.trans_blocks:
            out = block(out)
        
        # 3. Split
        f_cat = out.split([48, 120, 224], dim=1) # split by in_channels
        
        # 4. Hierarchical Fusion
        results = []
        for i in range(2, -1, -1): # [2, 1, 0]
            if i == 2:
                local_tokens = features[i]
            else:
                # Upsample previous result and add
                upsampled = F.interpolate(results[-1], size=features[i].shape[2:], mode='bilinear', align_corners=False)
                local_tokens = features[i] + self.conv[i](upsampled)
            
            global_sem = f_cat[i]
            local_tokens = self.meta[i](local_tokens)
            
            # SIM Block Logic (Inline for clarity)
            sim_block = self.SIM[i]
            inp = sim_block[0:2](local_tokens) # fuse1
            sig_act = sim_block[2:4](global_sem) # fuse2
            # H-Sigmoid
            sig_act = F.interpolate(F.relu6(sig_act + 3) / 6, size=inp.shape[2:], mode='bilinear', align_corners=False)
            results.append(inp * sig_act)
            
        x = results[-1] # The finest scale result
        
        # 5. Final Head
        _c = self.linear_fuse(x)
        prev_pred = self.cls_seg(_c)
        
        # Spatial Gather
        B, N, H, W = prev_pred.shape
        probs = F.softmax(prev_pred.view(B, N, -1), dim=2)
        feats = x.view(B, x.shape[1], -1).permute(0, 2, 1)
        context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3) # OCR Context
        
        # DPG Head
        final_out = self.lgc(x, context) + x
        return self.cls_seg(final_out)

# ============================================================================
# Part 3: Full Standalone Model
# ============================================================================

class CGRSeg_Tiny_Standalone(nn.Module):
    def __init__(self, num_classes=150, input_size=512):
        super().__init__()
        self.backbone = MobileNetV2Backbone(out_channels=(48, 120, 224))
        self.decode_head = CGRSegHead_Tiny(num_classes=num_classes)
        
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
    model = CGRSeg_Tiny_Standalone(num_classes=num_classes, input_size=input_size)
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
