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
# Part 1: EfficientFormerV2-S1 Backbone (Specific to Tiny config)
# ============================================================================

class Attention4D(nn.Module):
    """4D Attention used in EfficientFormerV2 (Fixed Dimensions)"""
    def __init__(self, dim=384, key_dim=32, num_heads=8, attn_ratio=4, resolution=7, stride=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = self.d * num_heads

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

        self.q = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.key_dim, 1), nn.BatchNorm2d(self.num_heads * self.key_dim))
        self.k = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.key_dim, 1), nn.BatchNorm2d(self.num_heads * self.key_dim))
        self.v = nn.Sequential(nn.Conv2d(dim, self.dh, 1), nn.BatchNorm2d(self.dh))
        
        self.v_local = nn.Sequential(
            nn.Conv2d(self.dh, self.dh, kernel_size=3, stride=1, padding=1, groups=self.dh),
            nn.BatchNorm2d(self.dh),
        )
        self.talking_head1 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1)
        self.talking_head2 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1)

        self.proj = nn.Sequential(nn.ReLU(), nn.Conv2d(self.dh, dim, 1), nn.BatchNorm2d(dim))

        # Attention Bias setup
        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        len_points = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        
        self.attention_biases_seg = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs_seg', torch.LongTensor(idxs).view(len_points, len_points))

    def forward(self, x):
        B, C, H, W = x.shape
        if self.stride_conv is not None:
            x = self.stride_conv(x)
            # Update H, W after stride
            H, W = x.shape[2], x.shape[3]
        
        N = H * W
        
        # [Fix] Reshape to separate heads: (B, Heads, KeyDim, N)
        q = self.q(x).reshape(B, self.num_heads, self.key_dim, N).permute(0, 1, 3, 2) # (B, H, N, K)
        k = self.k(x).reshape(B, self.num_heads, self.key_dim, N)                     # (B, H, K, N)
        v = self.v(x)
        v_local = self.v_local(v) # (B, H*V, H, W)
        v = v.reshape(B, self.num_heads, self.d, N).permute(0, 1, 3, 2)               # (B, H, N, V)

        # Attn: (B, H, N, K) @ (B, H, K, N) -> (B, H, N, N)
        attn = (q @ k) * self.scale
        
        # Bias: (H, N_static, N_static)
        bias = self.attention_biases_seg[:, self.attention_bias_idxs_seg]
        
        # Interpolate Bias if resolution changed
        if bias.shape[-1] != attn.shape[-1]:
             # (H, N, N) -> (1, H, N, N) for interpolate
             bias = F.interpolate(bias.unsqueeze(0), size=(N, N), mode='bilinear', align_corners=False).squeeze(0)
        
        # Broadcasting: (B, H, N, N) + (H, N, N) -> Works!
        attn = attn + bias

        attn = self.talking_head1(attn).softmax(dim=-1)
        attn = self.talking_head2(attn)

        # Output: (B, H, N, V) -> (B, H, V, N) -> (B, H*V, H, W)
        x = (attn @ v).permute(0, 1, 3, 2).reshape(B, self.dh, H, W) + v_local
        
        if self.upsample is not None:
            x = self.upsample(x)
            
        return self.proj(x)

class EfficientFormerV2_S1(nn.Module):
    """
    Hardcoded EfficientFormerV2-S1 (Tiny) Architecture.
    Widths: [32, 48, 120, 224]
    Depths: [3, 3, 9, 6]
    """
    def __init__(self, resolution=512, drop_path_rate=0.):
        super().__init__()
        
        # Config specific to S1
        self.embed_dims = [32, 48, 120, 224]
        self.depths = [3, 3, 9, 6]
        
        # Expansion ratios specific to S1
        self.e_ratios = {
            '0': [4, 4, 4],
            '1': [4, 4, 4],
            '2': [4, 4, 3, 3, 3, 3, 4, 4, 4],
            '3': [4, 4, 3, 3, 4, 4],
        }

        # 1. Stem
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, self.embed_dims[0] // 2, 3, 2, 1), nn.BatchNorm2d(self.embed_dims[0] // 2), nn.ReLU(),
            nn.Conv2d(self.embed_dims[0] // 2, self.embed_dims[0], 3, 2, 1), nn.BatchNorm2d(self.embed_dims[0]), nn.ReLU(),
        )

        # 2. Stages
        self.network = nn.ModuleList()
        total_depth = sum(self.depths)
        dpr_idx = 0
        
        for i in range(4): # 4 Stages
            blocks = []
            dim = self.embed_dims[i]
            
            # Build Blocks for this stage
            for j in range(self.depths[i]):
                dpr = drop_path_rate * dpr_idx / (total_depth - 1)
                dpr_idx += 1
                ratio = self.e_ratios[str(i)][j]
                
                # S1 Strategy: Last 2 blocks of Stage 2 & 3 are Attention, others are FFN
                is_attn = (i >= 2) and (j > self.depths[i] - 1 - 2) 
                
                blocks.append(self._make_block(
                    dim, ratio, dpr, is_attn, 
                    resolution=math.ceil(resolution / (2 ** (i + 2))),
                    stride=(2 if i==2 and j==self.depths[i]-2 else None) # Specific stride logic
                ))
            
            self.network.append(nn.Sequential(*blocks))

            # Downsample between stages (except last)
            if i < 3:
                out_dim = self.embed_dims[i+1]
                # Light downsample for early stages, Embedding for later
                if i >= 2: # Stage 2->3 uses Embedding (simplified here as conv)
                    down = nn.Sequential(
                         nn.Conv2d(dim, out_dim, 3, 2, 1), nn.BatchNorm2d(out_dim)
                    )
                else: 
                    # Standard downsample
                    down = nn.Sequential(
                        nn.Conv2d(dim, dim, 3, 2, 1, groups=dim), nn.BatchNorm2d(dim), nn.Hardswish(),
                        nn.Conv2d(dim, out_dim, 1), nn.BatchNorm2d(out_dim)
                    )
                    # Add skip connection logic inside forward usually, implemented as simple block here
                    down = ResidualDown(down, dim, out_dim)
                self.network.append(down)

        self.apply(self._init_weights)

    def _make_block(self, dim, mlp_ratio, drop_path, is_attn, resolution, stride=None):
        if is_attn:
            # Attention Block + FFN
            return AttnFFN(dim, mlp_ratio, drop_path=drop_path, resolution=resolution, stride=stride)
        else:
            # Pure FFN Block
            return FFN(dim, mlp_ratio, drop_path=drop_path)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.patch_embed(x)
        outs = []
        # Indices to save: [0, 2, 4, 6] in original list correspond to the 4 main stages
        # Here self.network is [Stage0, Down0, Stage1, Down1, Stage2, Down2, Stage3]
        
        # Stage 0
        x = self.network[0](x)
        # S1/Tiny usually doesn't use Stage 0 for Seg, but we keep the flow
        
        # Down 0 + Stage 1
        x = self.network[1](x)
        x = self.network[2](x)
        outs.append(x) # Index 0 (Scale 1/8) - Dim 48
        
        # Down 1 + Stage 2
        x = self.network[3](x)
        x = self.network[4](x)
        outs.append(x) # Index 1 (Scale 1/16) - Dim 120

        # Down 2 + Stage 3
        x = self.network[5](x)
        x = self.network[6](x)
        outs.append(x) # Index 2 (Scale 1/32) - Dim 224

        return outs # Returns features [C48, C120, C224]

# Helper Blocks
class ResidualDown(nn.Module):
    def __init__(self, main_path, in_ch, out_ch):
        super().__init__()
        self.main = main_path
        self.skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, 2), nn.BatchNorm2d(out_ch))
    def forward(self, x): return self.main(x) + self.skip(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.mid = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.mid_norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, in_features, 1)
        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.norm2 = nn.BatchNorm2d(in_features)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(self.norm1(x))
        x = self.act(self.mid_norm(self.mid(x)))
        x = self.fc2(x)
        x = self.norm2(x)
        return x

class FFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls = nn.Parameter(1e-5 * torch.ones(dim).view(1,-1,1,1))

    def forward(self, x):
        return x + self.drop_path(self.ls * self.mlp(x))

class AttnFFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop_path=0., resolution=7, stride=None):
        super().__init__()
        self.token_mixer = Attention4D(dim, resolution=resolution, stride=stride)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls1 = nn.Parameter(1e-5 * torch.ones(dim).view(1,-1,1,1))
        self.ls2 = nn.Parameter(1e-5 * torch.ones(dim).view(1,-1,1,1))

    def forward(self, x):
        x = x + self.drop_path(self.ls1 * self.token_mixer(x))
        x = x + self.drop_path(self.ls2 * self.mlp(x))
        return x

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
        self.backbone = EfficientFormerV2_S1(resolution=input_size)
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
    input_size = 2048
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
