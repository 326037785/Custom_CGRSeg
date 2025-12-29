import os
import cv2
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------
globalH = 256
globalW = 256
global_dim = 256
BATCH_SIZE = 8       # Increased from 4 to 8 for better BN stability
LR = 2e-4            # Slightly lower initial LR for AdamW + Cosine
NUM_EPOCHS = 300
AUX_WEIGHT = 0.2     # Increased from 0.0001 to 0.2 to actually utilize strip supervision


# -----------------------------------------------------------------------
# Data Generation (Kept Same)
# -----------------------------------------------------------------------
def make_fake_data(num: int = 8, height: int = 64, width: int = 64):
    datas = []
    targets = []

    for _ in range(num):
        image = np.zeros((height, width), dtype=np.float32)
        mask = np.zeros((height, width), dtype=np.int64)

        rect_num = np.random.randint(1, 4)
        for rect_idx in range(rect_num):
            rect_h = np.random.randint(8, height // 3)
            rect_w = np.random.randint(8, width // 3)
            y = np.random.randint(0, height - rect_h)
            x = np.random.randint(0, width - rect_w)

            image[y:y + rect_h, x:x + rect_w] = 1.0
            mask[y:y + rect_h, x:x + rect_w] = 1

        noise = np.random.normal(0, 0.1, (height, width)).astype(np.float32)
        image = image + noise

        if np.random.random() > 0.5:
            image = scipy.ndimage.binary_erosion(image > 0.5).astype(np.float32)
        else:
            image = scipy.ndimage.binary_dilation(image > 0.5).astype(np.float32)

        image = scipy.ndimage.gaussian_filter(image, sigma=1.0).astype(np.float32)
        image = image + np.random.normal(0, 0.05, (height, width)).astype(np.float32)
        image = np.clip(image, 0, 1).astype(np.float32)
        mask = (image > 0.3).astype(np.int64)

        datas.append(torch.from_numpy(image).unsqueeze(0))
        targets.append(torch.from_numpy(mask))

    datas = torch.stack(datas, dim=0)
    targets = torch.stack(targets, dim=0)
    return datas.float(), targets.long()


class FakeRectDataset(Dataset):
    def __init__(self, num: int = 8, height: int = 64, width: int = 64):
        super().__init__()
        self.data, self.target = make_fake_data(num, height, width)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


# -----------------------------------------------------------------------
# Model Components (RCA, RCM - Kept Same)
# -----------------------------------------------------------------------
class RCA(nn.Module):
    def __init__(self, inp, ratio=1, band_kernel_size=11, square_kernel_size=3):
        super().__init__()
        self.dwconv_hw = nn.Conv2d(inp, inp, kernel_size=square_kernel_size,
                                   padding=square_kernel_size // 2, groups=inp)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        gc = max(1, inp // ratio)
        self.conv_h = nn.Conv2d(inp, gc, (1, band_kernel_size),
                                padding=(0, band_kernel_size // 2), groups=gc)
        self.bn = nn.BatchNorm2d(gc)
        self.relu = nn.ReLU(inplace=True)
        self.conv_w = nn.Conv2d(gc, inp, (band_kernel_size, 1),
                                padding=(band_kernel_size // 2, 0), groups=gc)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        loc = self.dwconv_hw(x)
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        s = x_h + x_w
        feat_h = self.conv_h(s)
        feat_h = self.bn(feat_h)
        feat_h = self.relu(feat_h)
        feat_w = self.conv_w(feat_h)
        att = self.sigmoid(feat_w)
        return att * loc, x_h, x_w


class RCM(nn.Module):
    def __init__(self, dim, token_mixer: nn.Module = RCA, norm_layer: nn.Module = nn.BatchNorm2d,
                 mlp_ratio: int = 2, act_layer: nn.Module = nn.GELU, ls_init_value: float = 1e-6,
                 drop_path: float = 0.0, dw_size: int = 11, square_kernel_size: int = 3, ratio: int = 1):
        super().__init__()
        self.token_mixer = token_mixer(dim, band_kernel_size=dw_size,
                                       square_kernel_size=square_kernel_size, ratio=ratio)
        self.norm = norm_layer(dim)
        hidden_dim = int(mlp_ratio * dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.Identity(),
            act_layer(),
            nn.Dropout(0.0),
            nn.Conv2d(hidden_dim, dim, 1),
        )
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x_mixed, x_h, x_w = self.token_mixer(x)
        x_enc = self.mlp(self.norm(x_mixed))
        if self.gamma is not None:
            x_enc = x_enc * self.gamma.view(1, -1, 1, 1)
        out = self.drop_path(x_enc) + shortcut
        return out, x_h, x_w


class SegWithRCM(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, dim: int = 64, dw_size: int = 11):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(True))
        self.enc4 = nn.Sequential(nn.Conv2d(128, dim, 3, stride=2, padding=1), nn.ReLU(True))

        self.rcm = RCM(dim, dw_size=dw_size)
        self.head = nn.Conv2d(dim, num_classes, 3, padding=1)
        
        # Auxiliary Heads: Output raw logits for BCEWithLogitsLoss
        self.head_h = nn.Conv2d(dim, num_classes, 1)
        self.head_w = nn.Conv2d(dim, num_classes, 1)

    def forward(self, x, return_att: bool = False):
        H, W = x.shape[2], x.shape[3]
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)

        y, x_h, x_w = self.rcm(f4)
        
        logits = self.head(y)
        logits_up = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

        if return_att:
            # Visualization logic (kept same)
            with torch.no_grad():
                rca = self.rcm.token_mixer
                rcm = self.rcm
                _x_h = rca.pool_h(f4)
                _x_w = rca.pool_w(f4)
                s = _x_h + _x_w
                att_raw = s.mean(dim=1, keepdim=True)
                feat_h = rca.conv_h(s)
                feat_h = rca.bn(feat_h)
                feat_h = rca.relu(feat_h)
                att_h = feat_h.mean(dim=1, keepdim=True)
                feat_w = rca.conv_w(feat_h)
                att_final = rca.sigmoid(feat_w)
                att_v = att_final.mean(dim=1, keepdim=True)
                loc = rca.dwconv_hw(f4)
                token_mixer_out = att_final * loc
                normed = rcm.norm(token_mixer_out)
                mlp_out = rcm.mlp(normed)
                if rcm.gamma is not None:
                    mlp_out = mlp_out * rcm.gamma.view(1, -1, 1, 1)
                res_after_mlp = rcm.drop_path(mlp_out) + f4
                att_mlp_res = res_after_mlp.mean(dim=1, keepdim=True)
                def up(t): return F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
                att_dict = {
                    "raw": up(att_raw),
                    "after_h": up(att_h),
                    "after_v": up(att_v),
                    "mlp_res": up(att_mlp_res),
                }
            return logits_up, att_dict

        # Training: Output interpolated logits for strip supervision
        pred_h = self.head_h(x_h)
        pred_h_up = F.interpolate(pred_h, size=(H, 1), mode="bilinear", align_corners=False)
        
        pred_w = self.head_w(x_w)
        pred_w_up = F.interpolate(pred_w, size=(1, W), mode="bilinear", align_corners=False)

        return logits_up, pred_h_up, pred_w_up


# -----------------------------------------------------------------------
# NEW: Soft Dice Loss for better segmentation precision
# -----------------------------------------------------------------------
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, activation='softmax'):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.activation = activation

    def forward(self, logits, targets):
        # logits: (B, C, H, W)
        # targets: (B, H, W) with class indices
        num_classes = logits.shape[1]
        
        if self.activation == 'softmax':
            probs = torch.softmax(logits, dim=1)
        else:
            probs = torch.sigmoid(logits)
            
        # One-hot encode targets
        true_1_hot = torch.eye(num_classes, device=logits.device)[targets]  # (B, H, W, C)
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()               # (B, C, H, W)

        # Intersection
        dims = (0, 2, 3)
        intersection = torch.sum(probs * true_1_hot, dims)
        cardinality = torch.sum(probs + true_1_hot, dims)
        
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Loss is 1 - Mean Dice across classes (or just foreground)
        return 1. - torch.mean(dice)


# -----------------------------------------------------------------------
# Improved Training Phase
# -----------------------------------------------------------------------
def trainPhase():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Increase batch size for better gradient estimation
    dataset = FakeRectDataset(num=64, height=globalH, width=globalW)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SegWithRCM(in_channels=1, num_classes=2, dim=global_dim, dw_size=5).to(device)

    # 1. Improved Loss: Combination of CE and Dice
    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = SoftDiceLoss()
    
    # 2. Improved Auxiliary Loss: BCEWithLogits (more stable than MSE on prob)
    criterion_bce = nn.BCEWithLogitsLoss()

    # 3. Improved Optimizer: Weight Decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    
    # 4. Improved Scheduler: Cosine Annealing is smoother than StepLR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    print(f"Starting training on {device} with BS={BATCH_SIZE}, Epochs={NUM_EPOCHS}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        total = len(dataloader.dataset)

        for inputs, masks in dataloader:
            inputs = inputs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            
            # Forward
            logits, strip_h_logits, strip_w_logits = model(inputs)  
            # logits: (B,2,H,W), strip_h: (B,2,H,1), strip_w: (B,2,1,W)

            # --- Main Segmentation Loss ---
            loss_ce = criterion_ce(logits, masks)
            loss_dice = criterion_dice(logits, masks)
            loss_main = 0.5 * loss_ce + 0.5 * loss_dice  # Balanced combination

            # --- Auxiliary Strip Loss ---
            # Generate GT strip targets (Does this row/col contain foreground?)
            bin_masks = (masks > 0).float()
            target_h = bin_masks.max(dim=2)[0]  # (B, H) - 1 if row has object
            target_w = bin_masks.max(dim=1)[0]  # (B, W) - 1 if col has object

            # Select the "foreground" channel (index 1) from the strip logits
            # Remove dimensions of size 1 for BCE
            pred_h_fg = strip_h_logits[:, 1, :, :].squeeze(-1) # (B, H)
            pred_w_fg = strip_w_logits[:, 1, :, :].squeeze(1)  # (B, W)

            loss_h = criterion_bce(pred_h_fg, target_h)
            loss_w = criterion_bce(pred_w_fg, target_w)

            # Total Loss: Increase aux weight to make it effective
            loss = loss_main + AUX_WEIGHT * (loss_h + loss_w)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)

        scheduler.step()

        epoch_loss = running_loss / total
        if (epoch + 1) % 10 == 0:
            print(f"[Train] Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {epoch_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    torch.save(model.state_dict(), "rect_smooth_segnet.pth")
    print("Model saved to rect_smooth_segnet.pth")


# -----------------------------------------------------------------------
# Inference Phase (Visual Changes mostly in plotting)
# -----------------------------------------------------------------------
def _to_uint8(img: np.ndarray):
    img = img.astype(np.float32)
    vmin, vmax = img.min(), img.max()
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    else:
        img = np.zeros_like(img)
    return (img * 255.0).astype(np.uint8)


def inferencePhase():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SegWithRCM(in_channels=1, num_classes=2,
                       dim=global_dim, dw_size=5)
    
    weights_path = "rect_smooth_segnet.pth"
    if not os.path.exists(weights_path):
        print("Weights not found, please train first.")
        return

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    dataset = FakeRectDataset(num=4, height=globalH, width=globalW)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    out_dir = "inference_results_rcm"
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (inputs, masks) in enumerate(dataloader):
            inputs = inputs.to(device)
            # logits_up: (1, 2, H, W)
            logits_up, att_dict = model(inputs, return_att=True)

            # Apply softmax
            probs = torch.softmax(logits_up, dim=1)
            
            # More precise thresholding strategy
            # Since we used Dice loss, probability maps are usually cleaner.
            foreground_prob = torch.argmax(probs, dim=1)
            preds = (foreground_prob > 0.5).long().cpu().numpy().squeeze()
            
            input_np = inputs.cpu().numpy().squeeze()
            
            att_raw_np = att_dict["raw"].cpu().numpy().squeeze()
            att_h_np = att_dict["after_h"].cpu().numpy().squeeze()
            att_v_np = att_dict["after_v"].cpu().numpy().squeeze()
            att_mlp_np = att_dict["mlp_res"].cpu().numpy().squeeze()

            fig, axes = plt.subplots(2, 3, figsize=(12, 8))

            axes[0, 0].imshow(input_np, cmap="gray")
            axes[0, 0].set_title("Input")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(preds, cmap="gray")
            axes[0, 1].set_title("Prediction (Dice+CE)")
            axes[0, 1].axis("off")

            axes[0, 2].imshow(att_mlp_np, cmap="jet")
            axes[0, 2].set_title("Feature Refined")
            axes[0, 2].axis("off")

            axes[1, 0].imshow(att_raw_np, cmap="jet")
            axes[1, 0].set_title("Context (H+W)")
            axes[1, 0].axis("off")
            
            axes[1, 1].imshow(att_h_np, cmap="jet")
            axes[1, 1].set_title("Horizontal Strip Attn")
            axes[1, 1].axis("off")
            
            axes[1, 2].imshow(att_v_np, cmap="jet")
            axes[1, 2].set_title("Vertical Strip Attn")
            axes[1, 2].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"viz_progressive_{idx}.png"), dpi=150)
            plt.close(fig)

            print(f"[Infer] Saved sample {idx} to {out_dir}/")

if __name__ == "__main__":
    trainPhase()
    inferencePhase()
