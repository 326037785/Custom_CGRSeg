
# 📖 Technical Details: Main Contributions

This section provides an in-depth explanation of the three main contributions of CGRSeg:
1. **CGRSeg Pipeline**: The overall context-guided spatial feature reconstruction workflow
2. **RCM (Rectangular Self-Calibration Module)**: Rectangular self-calibration attention + feature fusion
3. **DPG Head (Dynamic Prototype Guided Head)**: Prototype generation and class embedding for pixel-level refinement


### 1. CGRSeg Overall Pipeline

CGRSeg follows a **"Global Context Extraction → Multi-scale Feature Fusion → Prototype-Guided Refinement"** paradigm for efficient semantic segmentation.

**Pipeline Logic:**

```
Input Image
    ↓
Backbone (EfficientFormer)
    ↓
Multi-scale Features {F₁, F₂, F₃, F₄} at 1/4, 1/8, 1/16, 1/32 scales
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Pyramid Pooling Aggregation (PPA)                      │
│  Aggregate multi-scale features to unified resolution           │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Global Context Modeling via RCM Blocks                 │
│  Apply N stacked RCM blocks for global semantic extraction      │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Hierarchical Feature Fusion (Top-Down)                 │
│  Fuse global semantics with local tokens at each scale          │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: Initial Segmentation Prediction                        │
│  Generate preliminary segmentation map                          │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: DPG Head - Prototype-Guided Refinement                 │
│  Extract class prototypes → Modulate features → Final output    │
└─────────────────────────────────────────────────────────────────┘
    ↓
Final Segmentation Output
```

**Pseudocode (based on `cgr_head.py`):**

```python
def CGRSeg_forward(inputs):
    """
    CGRSeg complete forward pass.
    
    Args:
        inputs: Multi-scale features from backbone
                List of tensors with shapes:
                - F1: [B, C1, H/4, W/4]
                - F2: [B, C2, H/8, W/8]  
                - F3: [B, C3, H/16, W/16]
                - F4: [B, C4, H/32, W/32]
    
    Returns:
        output: Segmentation logits [B, num_classes, H/4, W/4]
    """
    
    # Step 1: Pyramid Pooling Aggregation
    # Downsample all features to smallest resolution and concatenate
    aggregated = pyramid_pool_aggregate(inputs)  # [B, sum(C_i), H', W']
    
    # Step 2: Global Context Modeling with stacked RCM blocks
    global_context = stacked_RCM_blocks(aggregated)  # [B, sum(C_i), H', W']
    
    # Split back to per-scale features
    global_features = split_by_channels(global_context, channel_dims)
    
    # Step 3: Top-Down Hierarchical Feature Fusion
    results = []
    for i in range(num_scales - 1, -1, -1):  # From coarse to fine
        if i == num_scales - 1:
            local_tokens = inputs[i]  # Start with original features
        else:
            # Add upsampled previous result to current local tokens
            local_tokens = inputs[i] + upsample(project(results[-1]), size=inputs[i].shape)
        
        # Refine local tokens with RCM
        local_tokens = RCM_block(local_tokens)
        
        # Fuse local tokens with global semantics via SIM (Sigmoid-gated fusion)
        fused = SIM_fuse(local_tokens, global_features[i])
        results.append(fused)
    
    # Step 4: Initial Segmentation
    x = results[-1]  # Finest scale features
    x = linear_fuse(x)
    prev_output = cls_seg(x)  # [B, num_classes, H', W']
    
    # Step 5: DPG Head - Prototype Guided Refinement
    # Extract class prototypes from features weighted by predictions
    context = spatial_gather(x, prev_output)  # [B, C, num_classes, 1]
    
    # Apply dynamic prototype guidance
    object_context = DPG_head(x, context) + x  # Residual connection
    
    # Final prediction
    output = cls_seg(object_context)
    
    return output
```

---

### 2. RCM: Rectangular Self-Calibration Module

The RCM module combines **Rectangular Self-Calibration Attention (RCA)** with a feed-forward network for efficient spatial feature modeling.

#### 2.1 Variable Definitions

| Variable | Shape | Description |
|----------|-------|-------------|
| $X$ | $[B, C, H, W]$ | Input feature map |
| $X_{local}$ | $[B, C, H, W]$ | Local feature from depthwise convolution |
| $X_h$ | $[B, C, H, 1]$ | Horizontal pooled feature (height-preserved) |
| $X_w$ | $[B, C, 1, W]$ | Vertical pooled feature (width-preserved) |
| $A$ | $[B, C, H, W]$ | Attention weights (after sigmoid) |
| $k_h, k_w$ | scalar | Band convolution kernel sizes |
| $k_s$ | scalar | Square depthwise convolution kernel size |

#### 2.2 RCA Formulas

**Step 1: Local Feature Extraction**

The local feature is extracted using a depthwise convolution with a square kernel:

$$X_{local} = \text{DWConv}_{k_s \times k_s}(X)$$

**Step 2: Cross-shaped Pooling**

Two directional pooling operations preserve spatial information along different axes:

$$X_h = \text{AdaptiveAvgPool}_{(H, 1)}(X) \in \mathbb{R}^{B \times C \times H \times 1}$$

$$X_w = \text{AdaptiveAvgPool}_{(1, W)}(X) \in \mathbb{R}^{B \times C \times 1 \times W}$$

**Step 3: Cross-shaped Feature Aggregation**

The cross-shaped context is formed by combining horizontal and vertical features:

$$X_{cross} = X_h + X_w \in \mathbb{R}^{B \times C \times H \times W}$$

*Note: Broadcasting is applied during addition.*

**Step 4: Band Convolution Excitation**

Two consecutive 1D convolutions with orthogonal orientations model the rectangular region:

$$A = \sigma\left(\text{Conv}_{k_h \times 1}\left(\text{ReLU}\left(\text{BN}\left(\text{Conv}_{1 \times k_w}(X_{cross})\right)\right)\right)\right)$$

where $\sigma$ is the sigmoid activation.

**Step 5: Attention-weighted Output**

$$Y_{RCA} = A \odot X_{local}$$

where $\odot$ denotes element-wise multiplication.

#### 2.3 Complete RCM Formula

The complete RCM block includes RCA, normalization, MLP, layer scaling, and residual connection:

$$X_{mid} = \text{RCA}(X)$$
$$X_{mid} = \text{MLP}(\text{BN}(X_{mid}))$$
$$Y = X + \gamma \cdot X_{mid}$$

where $\gamma$ is a learnable layer scale parameter.

#### 2.4 RCM Pseudocode

```python
def RCA_forward(x):
    """
    Rectangular Self-Calibration Attention.
    
    Args:
        x: Input tensor [B, C, H, W]
    
    Returns:
        out: Attention-weighted features [B, C, H, W]
    """
    # Local feature extraction with square depthwise conv
    local_feat = dwconv_square(x)  # [B, C, H, W]
    
    # Cross-shaped pooling
    x_h = adaptive_avg_pool2d(x, output_size=(H, 1))  # [B, C, H, 1]
    x_w = adaptive_avg_pool2d(x, output_size=(1, W))  # [B, C, 1, W]
    
    # Combine via broadcasting
    x_gather = x_h + x_w  # [B, C, H, W] (broadcasted)
    
    # Band convolution excitation
    # First: horizontal band conv (1 x k_w)
    x_excite = conv_1xk(x_gather)  # [B, C, H, W]
    x_excite = batch_norm(x_excite)
    x_excite = relu(x_excite)
    
    # Second: vertical band conv (k_h x 1)
    attention = conv_kx1(x_excite)  # [B, C, H, W]
    attention = sigmoid(attention)
    
    # Apply attention to local features
    out = attention * local_feat
    
    return out


def RCM_forward(x):
    """
    Rectangular Self-Calibration Module.
    
    Args:
        x: Input tensor [B, C, H, W]
    
    Returns:
        out: Refined features [B, C, H, W]
    """
    shortcut = x
    
    # Token mixing via RCA
    x = RCA_forward(x)
    
    # Normalization
    x = batch_norm(x)
    
    # Feed-forward MLP (channel mixing)
    x = mlp(x)  # Conv1x1 -> GELU -> Conv1x1
    
    # Layer scale (learnable per-channel scaling)
    x = x * gamma.reshape(1, -1, 1, 1)
    
    # Residual connection with optional drop path
    out = drop_path(x) + shortcut
    
    return out
```

---

### 3. DPG Head: Dynamic Prototype Guided Head

The DPG Head addresses the challenge of **pixel-to-class-to-pixel** information flow: it first aggregates pixel features into class-level prototypes, then uses these prototypes to guide pixel-level refinement.

#### 3.1 Variable Definitions

| Variable | Shape | Description |
|----------|-------|-------------|
| $X$ | $[B, C, H, W]$ | Input pixel features to be refined |
| $Y$ | $[B, C, H, W]$ | Feature map for prototype extraction (can be same as X) |
| $P$ | $[B, K, H, W]$ | Initial segmentation probability map (K = num_classes) |
| $\hat{P}$ | $[B, K, HW]$ | Softmax-normalized probability (summing to 1 over spatial dim) |
| $Z$ | $[B, C, K]$ | Class prototypes (context vectors) |
| $W_{att}$ | $[B, 1, HW]$ | Spatial attention weights for prototype extraction |
| $\alpha$ | $[B, C, 1, 1]$ | Channel multiplication modulation vector |
| $\beta$ | $[B, C, 1, 1]$ | Channel addition modulation vector |

#### 3.2 Information Flow: Pixel → Class → Pixel

**Stage A: Pixel Space → Class Space (Prototype Extraction)**

Given pixel features and initial predictions, we compute class-conditional prototypes via soft spatial pooling:

$$\hat{P}_{k,i} = \frac{\exp(s \cdot P_{k,i})}{\sum_{j=1}^{HW} \exp(s \cdot P_{k,j})} \quad \text{(Softmax over spatial positions)}$$

$$Z_k = \sum_{i=1}^{HW} \hat{P}_{k,i} \cdot Y_i \in \mathbb{R}^{C}$$

In matrix form:

$$Z = \hat{P} \cdot Y^T \in \mathbb{R}^{B \times K \times C}$$

This produces K class-specific prototypes, each being a C-dimensional vector.

**Attention-based Global Context Pooling (used in DPG Head):**

When no per-class predictions are available, an attention-based global context is extracted:

$$W_{att} = \text{softmax}(\text{Conv}_{1\times1}(Y)) \in \mathbb{R}^{B \times 1 \times HW}$$

$$Z_{global} = Y \cdot W_{att}^T \in \mathbb{R}^{B \times C \times 1 \times 1}$$

This produces a single global context vector used for channel modulation.

**Stage B: Class Space → Pixel Space (Prototype-Guided Modulation)**

The extracted class prototypes are transformed into channel modulation parameters:

$$\alpha = \sigma\left(\text{MLP}_{mul}(Z)\right) \in \mathbb{R}^{B \times C \times 1 \times 1}$$

$$\beta = \text{MLP}_{add}(Z) \in \mathbb{R}^{B \times C \times 1 \times 1}$$

where MLP consists of: Conv1×1 → LayerNorm → ReLU → Conv1×1.

The final modulated output combines multiplicative and additive terms:

$$X_{out} = \alpha \odot X + \beta$$

With residual connection in CGRSeg:

$$X_{refined} = X_{out} + X$$

#### 3.3 Complete DPG Pipeline

```
Pixel Features X ──────────────────────────────────────────────┐
       ↓                                                        │
Initial Prediction P                                            │
       ↓                                                        │
┌──────────────────────────────────────┐                       │
│   Pixel → Class Transformation       │                       │
│   (Spatial Gather / Attention Pool)  │                       │
│                                      │                       │
│   Z = Σᵢ (P̂ₖᵢ · Yᵢ)  or             │                       │
│   Z = Softmax(Conv(Y)) · Y^T         │                       │
└──────────────────────────────────────┘                       │
       ↓                                                        │
   Class Prototypes Z [B, C, K, 1]                             │
       ↓                                                        │
┌──────────────────────────────────────┐                       │
│   Class → Pixel Transformation       │                       │
│   (Channel Modulation)               │                       │
│                                      │                       │
│   α = σ(MLP_mul(Z))                  │                       │
│   β = MLP_add(Z)                     │                       │
│   X_out = α ⊙ X + β                  │◄──────────────────────┘
└──────────────────────────────────────┘
       ↓
Refined Features (+ Residual) → Final Prediction
```

#### 3.4 DPG Pseudocode

```python
def spatial_gather(feats, probs, scale=1.0):
    """
    Aggregate pixel features into class prototypes.
    Pixel Space → Class Space.
    
    Args:
        feats: Pixel features [B, C, H, W]
        probs: Class probability map [B, K, H, W]
        scale: Temperature scaling factor
    
    Returns:
        context: Class prototypes [B, C, K, 1]
    """
    B, K, H, W = probs.shape
    C = feats.shape[1]
    
    # Flatten spatial dimensions
    probs = probs.view(B, K, H * W)  # [B, K, HW]
    feats = feats.view(B, C, H * W)  # [B, C, HW]
    
    # Transpose features for matrix multiplication
    feats = feats.permute(0, 2, 1)  # [B, HW, C]
    
    # Softmax over spatial dimension (normalize to sum=1)
    probs = softmax(scale * probs, dim=2)  # [B, K, HW]
    
    # Weighted aggregation: [B, K, HW] @ [B, HW, C] -> [B, K, C]
    context = matmul(probs, feats)  # [B, K, C]
    
    # Reshape for convolution operations
    context = context.permute(0, 2, 1).unsqueeze(3)  # [B, C, K, 1]
    
    return context


def DPG_spatial_pool(x, pool_type='att'):
    """
    Attention-based or average pooling for context extraction.
    Pixel Space → Global Context.
    
    Args:
        x: Input features [B, C, H, W]
        pool_type: 'att' for attention pooling, 'avg' for average pooling
    
    Returns:
        context: Global context vector [B, C, 1, 1]
    """
    B, C, H, W = x.shape
    
    if pool_type == 'att':
        # Compute spatial attention weights
        # x: [B, C, H, W] -> view to [B, C, HW] -> unsqueeze to [B, 1, C, HW]
        input_x = x.view(B, C, H * W).unsqueeze(1)
        
        # Attention mask via 1x1 conv
        context_mask = conv_1x1(x)  # [B, 1, H, W]
        context_mask = context_mask.view(B, 1, H * W)  # [B, 1, HW]
        context_mask = softmax(context_mask, dim=2)  # [B, 1, HW]
        context_mask = context_mask.unsqueeze(3)  # [B, 1, HW, 1]
        
        # Weighted sum: [B, 1, C, HW] @ [B, 1, HW, 1] -> [B, 1, C, 1]
        context = matmul(input_x, context_mask)
        context = context.view(B, C, 1, 1)  # [B, C, 1, 1]
    else:
        # Simple global average pooling
        context = adaptive_avg_pool2d(x, output_size=(1, 1))
    
    return context


def DPG_head_forward(x, y):
    """
    Dynamic Prototype Guided Head forward pass.
    Class Space → Pixel Space modulation.
    
    Args:
        x: Features to be refined [B, C, H, W]
        y: Input for context extraction. Can be:
           - Feature map [B, C, H, W] for general attention pooling
           - Class prototypes [B, C, K, 1] from spatial_gather (K=num_classes)
             In this case, spatial_pool aggregates K class prototypes into
             a single global context vector via attention.
    
    Returns:
        out: Refined features [B, C, H, W]
    """
    # Extract global context via attention pooling
    # If y has shape [B, C, K, 1], this pools over K class prototypes
    # If y has shape [B, C, H, W], this pools over spatial positions
    context = DPG_spatial_pool(y, pool_type='att')  # [B, C, 1, 1]
    
    out = x
    
    # Channel multiplication (gating)
    if use_channel_mul:
        # MLP: Conv1x1 -> LayerNorm -> ReLU -> Conv1x1 -> Sigmoid
        mul_term = conv1x1_1(context)      # [B, mid_C, 1, 1]
        mul_term = layer_norm(mul_term)
        mul_term = relu(mul_term)
        mul_term = conv1x1_2(mul_term)     # [B, C, 1, 1]
        mul_term = sigmoid(mul_term)       # α ∈ (0, 1)
        
        out = out * mul_term  # Channel-wise gating
    
    # Channel addition (bias)
    if use_channel_add:
        # MLP: Conv1x1 -> LayerNorm -> ReLU -> Conv1x1
        add_term = conv1x1_1(context)      # [B, mid_C, 1, 1]
        add_term = layer_norm(add_term)
        add_term = relu(add_term)
        add_term = conv1x1_2(add_term)     # [B, C, 1, 1] = β
        
        out = out + add_term  # Channel-wise bias
    
    return out


def CGRSeg_DPG_integration(x, prev_output):
    """
    Complete DPG integration in CGRSeg.
    
    In the actual CGRSeg implementation:
    1. spatial_gather extracts K class prototypes from x weighted by prev_output
    2. DPG_head pools these K prototypes into a single global context via attention
    3. The global context modulates x via channel multiplication
    
    Args:
        x: Fused features [B, C, H, W]
        prev_output: Initial segmentation logits [B, K, H, W]
    
    Returns:
        output: Final refined segmentation [B, K, H, W]
    """
    # Step 1: Pixel → Class (extract K class prototypes)
    # Each prototype is a weighted average of pixel features for that class
    context = spatial_gather(x, prev_output)  # [B, C, K, 1]
    
    # Step 2: Class → Global (pool K prototypes into single context)
    # DPG_head internally applies spatial_pool to the class prototypes
    # This aggregates information across all K classes via attention
    # The result modulates pixel features via channel multiplication
    object_context = DPG_head_forward(x, context)  # [B, C, H, W]
    # Note: DPG_head.spatial_pool(context) pools [B, C, K, 1] -> [B, C, 1, 1]
    #       Then applies channel modulation: x * sigmoid(MLP(global_context))
    
    # Step 3: Residual connection
    object_context = object_context + x
    
    # Step 4: Final classification
    output = cls_seg(object_context)  # [B, K, H, W]
    
    return output
```

---

### Summary

| Component | Function | Key Innovation |
|-----------|----------|----------------|
| **CGRSeg Pipeline** | End-to-end segmentation | Pyramid aggregation + Top-down fusion + Prototype refinement |
| **RCM** | Spatial feature reconstruction | Cross-shaped pooling + Band convolutions for rectangular attention |
| **DPG Head** | Class-aware feature refinement | Pixel→Class prototype extraction + Class→Pixel channel modulation |

---


