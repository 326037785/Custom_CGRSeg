# 📖 技术细节（中文扩展版）

本文档在英文版基础上，使用中文对 CGRSeg 的三个核心贡献进行更细化的说明，并穿插具体示例，便于快速理解：

1. **CGRSeg 全流程：** 从全局上下文提取到基于原型的细粒度修正。
2. **RCM（Rectangular Self-Calibration Module）：** 矩形自校准注意力 + 多尺度特征融合。
3. **DPG Head（Dynamic Prototype Guided Head）：** 基于类别原型的动态引导，实现像素级细化。

> 术语说明：以下示例默认输入分辨率为 `512×512`，批大小 `B=2`，类别数 `K=19`（类似 Cityscapes）。

---

## 1. CGRSeg 整体流水线

CGRSeg 遵循 **“全局上下文提取 → 多尺度特征融合 → 原型引导精修”** 的范式，以较小的计算量获得更强的语义一致性。

### 流程概览

```
输入图像 (B=2, 3×512×512)
    ↓
Backbone（如 EfficientFormer）
    ↓
多尺度特征 {F₁, F₂, F₃, F₄}，分辨率依次为 1/4、1/8、1/16、1/32
    ↓
PPA（Pyramid Pooling Aggregation）：统一到最小尺度并拼接
    ↓
RCM 堆叠：建模全局语义（例如堆叠 N=2~4 个 RCM）
    ↓
自顶向下的层级特征融合：逐级上采样并与局部特征融合
    ↓
初始分割预测：得到粗分割图
    ↓
DPG Head：利用类别原型对特征进行再次调制，输出最终结果
```

### 关键示例
- **PPA**：将 `F₁ (C1=64)`、`F₂ (C2=128)`、`F₃ (C3=320)`、`F₄ (C4=512)` 全部下采样到 `16×16`，通道相加后得到约 `1024×16×16` 的特征。
- **RCM 堆叠**：若堆叠 3 个 RCM，输出保持 `1024×16×16`，但含有更强的全局语义。
- **Top-Down 融合**：从最粗的 `16×16` 开始，每上采样一次与对应尺度相加，最终在 `128×128` 处形成细节更丰富的特征。

### 示例（基于 `cgr_head.py` 思路）

```python
def CGRSeg_forward(inputs):
    """CGRSeg 前向流程"""
    aggregated = pyramid_pool_aggregate(inputs)  # PPA
    global_context = stacked_RCM_blocks(aggregated)  # RCM 堆叠
    global_features = split_by_channels(global_context, channel_dims)

    results = []
    for i in range(num_scales - 1, -1, -1):
        if i == num_scales - 1:
            local_tokens = inputs[i]
        else:
            local_tokens = inputs[i] + upsample(project(results[-1]), size=inputs[i].shape)
        local_tokens = RCM_block(local_tokens)
        fused = SIM_fuse(local_tokens, global_features[i])
        results.append(fused)

    x = linear_fuse(results[-1])
    prev_output = cls_seg(x)  # 初始预测

    context = spatial_gather(x, prev_output)  # 类别原型
    object_context = DPG_head(x, context) + x  # 残差
    output = cls_seg(object_context)  # 最终预测
    return output
```

---

## 2. RCM：Rectangular Self-Calibration Module

RCM 将矩形自校准注意力（RCA）与轻量 FFN 结合，擅长捕获跨长短边的结构。

### 2.1 核心思想
- **矩形池化**：使用水平/垂直方向的不对称卷积（如 `k_h=1, k_w=7` 与 `k_h=7, k_w=1`），模拟“十字形”感受野。
- **自校准注意力**：通过全局上下文引导每个位置的注意力权重，抑制噪声区域。
- **残差 & 归一化**：保持梯度稳定，易于堆叠多层。

### 2.2 输入输出示例
- 输入：`x ∈ R^{B×C×H×W}`，例如 `B=2, C=256, H=W=64`。
- 输出：`y` 形状与 `x` 相同，但包含更强的方向性语义。

### 2.3 核心流程示例代码

```python
def RCM_forward(x):
    # 方向感知卷积
    horiz = conv_1x7(x)  # 捕获水平结构
    vert = conv_7x1(x)   # 捕获垂直结构

    # 矩形注意力
    attn = sigmoid(horiz + vert)
    refined = x * attn  # 位置加权

    # 轻量 FFN
    hidden = conv1x1(refined)
    hidden = layer_norm(hidden)
    hidden = relu(hidden)
    out = conv1x1(hidden)

    return out + x  # 残差输出
```

### 2.4 直观例子
- 对道路场景，水平卷积强化路面连续性；垂直卷积突出路灯、交通标志等细长结构。
- 在交叉路口，矩形注意力可同时关注长路段与垂直人行道，减少“断裂”预测。

---

## 3. DPG Head：Dynamic Prototype Guided Head

DPG Head 通过“像素→类别→像素”的往返过程，实现类别感知的特征调制。

### 3.1 原型提取
- 使用 `spatial_gather` 将每个类别的像素特征加权平均，得到 `context ∈ R^{B×C×K×1}`。
- 例如 `B=2, C=256, K=19` 时，`context` 形状为 `2×256×19×1`。

### 3.2 动态引导
- `DPG_head` 对 `context` 进行注意力池化，得到全局向量 `g ∈ R^{B×C×1×1}`。
- 通过两条支路进行调制：
  - **通道乘法**：`x * σ(MLP(g))`，强调关键通道，抑制噪声。
  - **通道加法**（可选）：`x + MLP(g)`，为特定类别添加偏置。

### 3.3 伪代码

```python
def DPG_head_forward(x, y):
    context = DPG_spatial_pool(y, pool_type='att')  # 全局上下文

    out = x
    if use_channel_mul:
        mul_term = sigmoid(conv1x1_2(relu(layer_norm(conv1x1_1(context)))))
        out = out * mul_term

    if use_channel_add:
        add_term = conv1x1_2(relu(layer_norm(conv1x1_1(context))))
        out = out + add_term

    return out
```

### 3.4 应用示例
- **小目标修正**：若初始预测对行人/交通灯置信度较低，通道乘法可提升相关特征响应，避免遗漏。
- **大区域平滑**：对天空、路面等大区域，通道加法提供平滑偏置，减少块状伪影。

---

## 总结

| 组件 | 功能 | 关键创新 | 实战提示 |
|------|------|----------|----------|
| **CGRSeg Pipeline** | 端到端语义分割 | 金字塔聚合 + 自顶向下融合 + 原型细化 | 根据显存调整 RCM 堆叠数（2~4 层），在 12GB 显存下可用 3 层。 |
| **RCM** | 空间特征重建 | 十字形卷积 + 矩形注意力 | 对窄长结构（车道线、路杆）效果显著，可减少断裂。 |
| **DPG Head** | 类别感知细化 | 像素→类别原型→像素 调制 | 预测阶段仅增加少量计算，显著提升边界与小目标质量。 |

如果需要快速上手，可在 `configs/` 中选择轻量化 backbone（如 EfficientFormer-L1），并将 RCM 堆叠数设为 2，便于在消费级 GPU 上训练与验证。
