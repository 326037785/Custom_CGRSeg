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
    矩形自校准注意力模块 (Rectangular Self-Calibration Attention)
    
    核心思想 (参见 paper/readme_zh.md 第 2 节):
    1. 十字形池化 (Cross-shaped Pooling):
       - pool_h: 沿宽度方向做全局平均池化，保留高度信息 → X_h ∈ R^{B×C×H×1}。
       - pool_w: 沿高度方向做全局平均池化，保留宽度信息 → X_w ∈ R^{B×C×1×W}。
       - 两者相加 (广播) 得到十字形上下文特征。
    
    2. 带状卷积激励 (Band Convolution Excitation):
       - 先用 1×k_w 水平带状卷积捕获水平结构 (如道路连续性)。
       - 再用 k_h×1 垂直带状卷积捕获垂直结构 (如路灯、交通标志)。
       - 最后通过 Sigmoid 生成注意力权重。
    
    3. 局部特征调制:
       - dwconv_hw: 方形深度卷积提取局部特征 X_local。
       - 最终输出 = 注意力权重 A ⊙ 局部特征 X_local。
    
    公式:
        X_cross = X_h + X_w  (十字形特征聚合)
        A = σ(Conv_{k_h×1}(ReLU(BN(Conv_{1×k_w}(X_cross)))))  (带状卷积激励)
        Y = A ⊙ X_local  (注意力加权输出)
    """
    def __init__(self, inp, kernel_size=1, ratio=1, band_kernel_size=11, square_kernel_size=2):
        super().__init__()
        # 局部特征提取: 方形深度卷积
        self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size, padding=square_kernel_size // 2, groups=inp)
        # 十字形池化: 分别沿宽度和高度方向做全局平均池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 保留 H，压缩 W → [B, C, H, 1]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 保留 W，压缩 H → [B, C, 1, W]
        
        gc = max(1, inp // ratio)
        # 带状卷积激励序列: 1×k 水平卷积 → BN → ReLU → k×1 垂直卷积 → Sigmoid
        self.excite = nn.Sequential(
            nn.Conv2d(inp, gc, (1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc), nn.ReLU(inplace=True),
            nn.Conv2d(gc, inp, (band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Step 1: 局部特征提取 (方形深度卷积)
        loc = self.dwconv_hw(x)  # X_local ∈ R^{B×C×H×W}
        
        # Step 2: 十字形池化
        x_h = self.pool_h(x)  # X_h ∈ R^{B×C×H×1}
        x_w = self.pool_w(x)  # X_w ∈ R^{B×C×1×W}
        
        # Step 3: 带状卷积激励 (十字形上下文 → 注意力权重)
        att = self.excite(x_h + x_w)  # A ∈ R^{B×C×H×W}，通过广播相加
        
        # Step 4: 注意力加权输出
        return att * loc  # A ⊙ X_local

class RCM(nn.Module):
    """
    矩形自校准模块 (Rectangular Self-Calibration Module)
    
    核心思想 (参见 paper/readme_zh.md 第 2 节):
    - 将 RCA (矩形自校准注意力) 与轻量级 FFN 结合
    - 擅长捕获跨长短边的结构 (如道路、车道线、路杆等)
    
    模块结构:
        1. Token Mixing: 通过 RCA 进行空间特征混合
        2. Channel Mixing: 通过 MLP (1×1 Conv) 进行通道特征混合
        3. Layer Scale: 可学习的逐通道缩放参数 γ
        4. 残差连接: Y = X + γ · MLP(BN(RCA(X)))
    
    公式:
        X_mid = RCA(X)  (空间特征重建)
        X_mid = MLP(BN(X_mid))  (通道混合)
        Y = X + γ · X_mid  (残差 + 层缩放)
    """
    def __init__(self, dim, dw_size=11):
        super().__init__()
        # Token Mixer: RCA 进行空间特征混合
        self.token_mixer = RCA(dim, band_kernel_size=dw_size, square_kernel_size=3, ratio=1)
        self.norm = nn.BatchNorm2d(dim)
        # Channel Mixer: MLP，扩展比固定为 2 (dim → dim*2 → dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim*2, 1), nn.Identity(), nn.GELU(), nn.Dropout(0), nn.Conv2d(dim*2, dim, 1)
        )
        # Layer Scale: 可学习的逐通道缩放参数，初始化为 1e-6
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x):
        shortcut = x  # 残差连接
        x = self.token_mixer(x)  # RCA: 空间特征混合
        x = self.mlp(self.norm(x))  # BN + MLP: 通道特征混合
        # 残差 + Layer Scale
        return shortcut + x.mul(self.gamma.view(1, -1, 1, 1))

class DPGHead(nn.Module):
    """
    动态原型引导头 (Dynamic Prototype Guided Head)
    
    核心思想 (参见 paper/readme_zh.md 第 3 节):
    - 通过 "像素→类别→像素" 的往返过程，实现类别感知的特征调制
    - 从特征图中提取全局上下文，然后通过通道调制增强特征
    
    信息流 (参见 paper/readme.md 第 3.2 节):
    1. 像素空间 → 类别空间 (Spatial Pool):
       - 通过 1×1 卷积生成空间注意力权重
       - 使用 Softmax 归一化后，加权聚合特征得到全局上下文向量
       - W_att = Softmax(Conv_{1×1}(Y)) ∈ R^{B×1×HW}
       - context = Y · W_att^T ∈ R^{B×C×1×1}
    
    2. 类别空间 → 像素空间 (Channel Mul):
       - 通过 MLP 将上下文向量转换为通道调制权重
       - α = σ(MLP(context)) ∈ R^{B×C×1×1}
       - X_out = X ⊙ α (通道门控)
    
    应用示例:
    - 小目标修正: 若初始预测对行人/交通灯置信度较低，通道乘法可提升相关特征响应
    - 大区域平滑: 对天空、路面等大区域，提供平滑偏置，减少块状伪影
    """
    def __init__(self, channels):
        super().__init__()
        # 空间注意力: 1×1 卷积生成注意力 mask
        self.conv_mask = nn.Conv2d(channels, 1, 1)
        self.softmax = nn.Softmax(dim=2)
        # 通道调制 MLP: Conv1x1 → LayerNorm → ReLU → Conv1x1
        self.channel_mul_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1), nn.LayerNorm([channels, 1, 1]),
            nn.ReLU(inplace=True), nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x, y):
        """
        Args:
            x: 待调制的特征图 [B, C, H, W]
            y: 用于提取上下文的特征图 (通常是类别原型) [B, C, H, W] 或 [B, C, K, 1]
        
        Returns:
            调制后的特征图 [B, C, H, W]
        """
        # ========== Step 1: 空间注意力池化 (Spatial Pool) ==========
        # 像素空间 → 全局上下文
        B, C, H, W = y.shape
        input_x = y.view(B, C, H * W).unsqueeze(1)  # [B, 1, C, HW]
        # 生成空间注意力权重并归一化
        context_mask = self.softmax(self.conv_mask(y).view(B, 1, H * W)).unsqueeze(3)  # [B, 1, HW, 1]
        # 加权聚合得到全局上下文向量
        context = torch.matmul(input_x, context_mask).view(B, C, 1, 1)  # [B, C, 1, 1]
        
        # ========== Step 2: 通道乘法调制 (Channel Mul) ==========
        # 类别空间 → 像素空间
        # α = σ(MLP(context))，X_out = X ⊙ α
        return x * torch.sigmoid(self.channel_mul_conv(context))

class CGRSegHead_Tiny(nn.Module):
    """
    Tiny 版解码头 (CGRSeg Decode Head)
    
    整体流程 (参见 paper/readme_zh.md 第 1 节):
    ┌────────────────────────────────────────────────────────────────────┐
    │  Step 1: PPA (金字塔池化聚合)                                        │
    │  将多尺度特征统一到最小分辨率并拼接                                     │
    └────────────────────────────────────────────────────────────────────┘
        ↓
    ┌────────────────────────────────────────────────────────────────────┐
    │  Step 2: RCM 堆叠 (全局上下文建模)                                    │
    │  通过 N 个堆叠的 RCM 模块提取全局语义特征                               │
    └────────────────────────────────────────────────────────────────────┘
        ↓
    ┌────────────────────────────────────────────────────────────────────┐
    │  Step 3: 自顶向下层级特征融合 (Top-Down Hierarchical Fusion)          │
    │  从粗到细逐级上采样，通过 SIM 将全局语义与局部特征融合                    │
    └────────────────────────────────────────────────────────────────────┘
        ↓
    ┌────────────────────────────────────────────────────────────────────┐
    │  Step 4: 初始分割预测                                                │
    │  生成粗分割图                                                        │
    └────────────────────────────────────────────────────────────────────┘
        ↓
    ┌────────────────────────────────────────────────────────────────────┐
    │  Step 5: DPG Head (原型引导精修)                                     │
    │  提取类别原型 → 调制特征 → 最终输出                                    │
    └────────────────────────────────────────────────────────────────────┘
    
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
        channels = 128  # 统一的 embedding 维度
        
        # 最终融合层: 分组卷积 + BN + ReLU
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        # ========== RCM 堆叠 (全局上下文建模) ==========
        # Next Layer Transformer (RCM 堆叠)，输入通道 = C2+C3+C4 = 288
        # 堆叠 5 个 RCM 模块进行全局语义提取
        self.trans_blocks = nn.ModuleList([
            RCM(sum(self.in_channels), dw_size=11) for _ in range(5)
        ])

        # ========== 每个尺度的处理模块 ==========
        self.SIM = nn.ModuleList()  # Sigmoid-gated 融合模块
        self.meta = nn.ModuleList()  # 每个尺度的 RCM
        self.conv = nn.ModuleList()  # 上采样后的通道映射
        
        for i in range(len(in_channels)):
            # SIM (Sigmoid-gated Fusion): 4 层序列，分为两对 Conv+BN:
            #   - [0:2] (Conv+BN): 用于 local tokens 投影
            #   - [2:4] (Conv+BN): 用于 global semantics 投影
            self.SIM.append(nn.Sequential(
                nn.Conv2d(in_channels[i], channels, 1, bias=False),  # [0] local projection conv
                nn.BatchNorm2d(channels),                            # [1] local projection bn
                nn.Conv2d(in_channels[i], channels, 1, bias=False),  # [2] global projection conv
                nn.BatchNorm2d(channels),                            # [3] global projection bn
            ))
            # Meta: 每个尺度自己的 RCM，用于局部特征增强
            self.meta.append(RCM(in_channels[i], dw_size=11))
            
            if i < len(in_channels) - 1:
                # 用于把上一级 embedding 映射回该层的原生 in_channels (Top-Down 融合时使用)
                self.conv.append(nn.Conv2d(channels, in_channels[i], 1))

        # DPG Head: 动态原型引导
        self.lgc = DPGHead(channels)
        # 分类头: Dropout + 1×1 卷积
        self.cls_seg = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(channels, num_classes, 1)
        )

    def forward(self, features):
        """
        CGRSeg Tiny 版解码头的前向传播
        
        Args:
            features: 来自 Backbone 的多尺度特征列表
                - features[0]: F2, [B, 32, H/8, W/8]
                - features[1]: F3, [B, 96, H/16, W/16]
                - features[2]: F4, [B, 160, H/32, W/32]
        
        Returns:
            最终分割结果 [B, num_classes, H/8, W/8]
        """
        
        # ========================================================================
        # Step 1: PPA (Pyramid Pooling Aggregation) - 金字塔池化聚合
        # 将所有尺度的特征统一下采样到最小分辨率，然后在通道维度拼接
        # 参见 paper/readme_zh.md 第 1 节: "PPA：将 F₁~F₄ 全部下采样到最小尺度并拼接"
        # ========================================================================
        H, W = (features[-1].shape[2] - 1) // 2 + 1, (features[-1].shape[3] - 1) // 2 + 1
        ppa_in = torch.cat(
            [F.adaptive_avg_pool2d(f, (H, W)) for f in features],
            dim=1
        )  # 通道数 = 32+96+160 = 288, 分辨率统一为 (H, W)
        
        # ========================================================================
        # Step 2: Global Context Modeling - 全局上下文建模 (RCM 堆叠)
        # 通过堆叠的 RCM 模块提取全局语义特征
        # 参见 paper/readme_zh.md 第 1 节: "RCM 堆叠：建模全局语义"
        # ========================================================================
        out = ppa_in
        for block in self.trans_blocks:
            out = block(out)  # 每个 RCM 模块: RCA + FFN + 残差连接
        
        # 按原始通道数拆分回三个尺度的全局语义特征
        f_cat = out.split([32, 96, 160], dim=1)  # [global_F2, global_F3, global_F4]
        
        # ========================================================================
        # Step 3: Top-Down Hierarchical Feature Fusion - 自顶向下层级特征融合
        # 从最粗尺度 (F4) 开始，逐级上采样并与局部特征融合
        # 参见 paper/readme_zh.md 第 1 节: "自顶向下的层级特征融合：逐级上采样并与局部特征融合"
        # ========================================================================
        results = []
        for i in range(2, -1, -1):  # 从粗到细: 2 (F4) → 1 (F3) → 0 (F2)
            # ---------- 局部特征准备 ----------
            if i == 2:
                # 最粗尺度: 直接使用原始特征
                local_tokens = features[i]
            else:
                # 其他尺度: 上一级结果上采样 + 通道映射 + 当前尺度原始特征
                upsampled = F.interpolate(
                    results[-1],
                    size=features[i].shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                local_tokens = features[i] + self.conv[i](upsampled)  # 残差融合
            
            # 全局语义特征 (来自 RCM 堆叠后的拆分)
            global_sem = f_cat[i]
            
            # 局部特征增强: 通过该尺度专属的 RCM 模块
            local_tokens = self.meta[i](local_tokens)

            # ---------- SIM (Sigmoid-gated Fusion) - Sigmoid 门控融合 ----------
            # 将局部特征和全局语义通过 Sigmoid 门控机制融合
            # 使用 HardSigmoid 近似: HardSigmoid(x) = ReLU6(x + 3) / 6
            # 公式: output = local_proj ⊙ HardSigmoid(global_proj)
            # 参见 paper/readme_zh.md 第 1 节: "SIM_fuse(local_tokens, global_features[i])"
            sim_block = self.SIM[i]
            inp = sim_block[0:2](local_tokens)    # 局部特征投影到 embedding 维度 (Conv+BN)
            sig_act = sim_block[2:4](global_sem)  # 全局语义投影到 embedding 维度 (Conv+BN)
            # HardSigmoid 激活并上采样到局部特征分辨率
            sig_act = F.interpolate(
                F.relu6(sig_act + 3) / 6,  # HardSigmoid(x) = clamp(x+3, 0, 6) / 6
                size=inp.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            # 门控融合: 局部特征 ⊙ 全局注意力权重
            results.append(inp * sig_act)
            
        # 获取最细尺度的融合结果 (1/8 分辨率)
        x = results[-1]

        # ========================================================================
        # Step 4 & 5: Prototype Extraction & DPG Head Refinement - 原型提取与精修
        # 参见 paper/readme_zh.md 第 3 节: "DPG Head：利用类别原型对特征进行再次调制"
        # ========================================================================
        
        # ---------- Step 4: 初始分割预测 ----------
        _c = self.linear_fuse(x)
        prev_pred = self.cls_seg(_c)  # 粗分割结果 [B, K, H, W]

        # ---------- Step 5a: 原型提取 (Spatial Gather) ----------
        # 像素空间 → 类别空间: 使用初始预测加权聚合像素特征，得到类别原型
        # 对每个类别 k: context[k] = sum_i(softmax(P)[k,i] * X[i])
        # 参见 paper/readme.md 第 3.2 节: "Z = P̂ · Y^T"
        B, N, H, W = prev_pred.shape
        probs = F.softmax(prev_pred.view(B, N, -1), dim=2)  # [B, K, HW]，沿空间维度归一化
        feats = x.view(B, x.shape[1], -1).permute(0, 2, 1)  # [B, HW, C]
        # 加权聚合: [B, K, HW] @ [B, HW, C] → [B, K, C] → [B, C, K, 1]
        context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)

        # ---------- Step 5b: DPG Head 精修 (Dynamic Prototype Guided) ----------
        # 类别空间 → 像素空间: 使用类别原型调制像素特征
        # 公式: X_refined = DPG(x, context) + x (残差连接)
        # 参见 paper/readme_zh.md 第 3 节: "DPG_head(x, context) + x"
        final_out = self.lgc(x, context) + x  # 残差连接
        
        # 最终分类
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
