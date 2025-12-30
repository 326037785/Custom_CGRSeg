import os
import json
from typing import Any, Dict, List, Tuple, Optional

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
# Config
# -----------------------------------------------------------------------
DATA_ROOT = r"path\car-segmentation"   # contains images/, masks/, classes.txt, masks.json
#The original data can be found in kaggle car-segmentation.
IMAGE_DIRNAME = "images"
MASK_DIRNAME = "masks"

globalH = 256
globalW = 256
global_dim = 256

BATCH_SIZE = 8
LR = 2e-4
NUM_EPOCHS = 10           # adjust as you like
AUX_WEIGHT = 0.2

FORCE_GRAY = True         # keep model in_channels=1; set False to use RGB with in_channels=3
LIMIT_TRAIN_NUM = 10      # like FakeRectDataset: only use first N
LIMIT_INFER_NUM = 8

SAVE_WEIGHTS = "seg_rcm_from_folder.pth"
OUT_DIR = "inference_results_rcm_folder"

# -----------------------------------------------------------------------
# Dataset (mask png already encodes class index: background=0, classes start from 1)
# -----------------------------------------------------------------------
def _read_lines(path: str) -> List[str]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for s in f:
            s = s.strip()
            if s:
                lines.append(s)
    return lines

def _infer_num_classes_from_masks(mask_paths: List[str], max_scan: int = 32) -> int:
    n = len(mask_paths)
    k = min(max_scan, n)
    vmax = 0
    for i in range(k):
        m = cv2.imread(mask_paths[i], cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        v = int(m.max()) if m.size > 0 else 0
        if v > vmax:
            vmax = v
    return 5 #max(2, vmax)  # at least background+1

class FolderSegDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        *,
        height: int,
        width: int,
        num: int = -1,
        image_dirname: str = "images",
        mask_dirname: str = "masks",
        force_gray: bool = True,
        allowed_img_ext: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
        allowed_mask_ext: Tuple[str, ...] = (".png", ".tif", ".tiff"),
        sort_files: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, image_dirname)
        self.msk_dir = os.path.join(root_dir, mask_dirname)
        self.H = int(height)
        self.W = int(width)
        self.force_gray = bool(force_gray)

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"images dir not found: {self.img_dir}")
        if not os.path.isdir(self.msk_dir):
            raise FileNotFoundError(f"masks dir not found: {self.msk_dir}")

        img_files: List[str] = []
        for fn in os.listdir(self.img_dir):
            ext = os.path.splitext(fn)[1].lower()
            if ext in allowed_img_ext:
                img_files.append(fn)
        if sort_files:
            img_files.sort()

        pairs: List[Tuple[str, str]] = []
        mask_paths: List[str] = []
        for fn in img_files:
            stem = os.path.splitext(fn)[0]
            img_path = os.path.join(self.img_dir, fn)

            msk_path = None
            for me in allowed_mask_ext:
                cand = os.path.join(self.msk_dir, stem + me)
                if os.path.exists(cand):
                    msk_path = cand
                    break
            if msk_path is None:
                continue

            pairs.append((img_path, msk_path))
            mask_paths.append(msk_path)

        if len(pairs) == 0:
            raise RuntimeError("No (image, mask) pairs found. Check filename stems and folders.")

        if num is not None and num > 0:
            pairs = pairs[:num]
            mask_paths = mask_paths[:num]

        self.pairs = pairs

        # Determine num_classes (optional utility)
        classes_txt = os.path.join(root_dir, "classes.txt")
        if os.path.exists(classes_txt):
           # cls = _read_lines(classes_txt)
            # classes.txt typically lists labels excluding background; background assumed 0
            self.num_classes = 5
        else:
            self.num_classes = _infer_num_classes_from_masks(mask_paths, max_scan=32)

    def __len__(self):
        return len(self.pairs)

    def _read_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")

        if self.force_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # (H,W)
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32) / 255.0
            img = img[None, ...]  # (1,H,W)
        else:
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)  # (3,H,W)

        return img

    def _read_mask(self, path: str) -> np.ndarray:
        msk = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if msk is None:
            raise FileNotFoundError(f"Failed to read mask: {path}")
        if msk.ndim == 3:
            msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
        msk = cv2.resize(msk, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return msk.astype(np.int64)

    def __getitem__(self, idx: int):
        img_path, msk_path = self.pairs[idx]
        img = self._read_image(img_path)
        msk = self._read_mask(msk_path)
        return torch.from_numpy(img).float(), torch.from_numpy(msk).long()

# -----------------------------------------------------------------------
# Model Components (RCA, RCM)
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

        pred_h = self.head_h(x_h)
        pred_h_up = F.interpolate(pred_h, size=(H, 1), mode="bilinear", align_corners=False)

        pred_w = self.head_w(x_w)
        pred_w_up = F.interpolate(pred_w, size=(1, W), mode="bilinear", align_corners=False)

        return logits_up, pred_h_up, pred_w_up

# -----------------------------------------------------------------------
# Soft Dice Loss
# -----------------------------------------------------------------------
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B,C,H,W), targets: (B,H,W) int64
        B, C, H, W = logits.shape
        probs = torch.softmax(logits, dim=1)  # (B,C,H,W)

        # one-hot targets
        true_1_hot = torch.zeros((B, C, H, W), device=logits.device, dtype=torch.float32)
        true_1_hot.scatter_(1, targets.unsqueeze(1).clamp(min=0, max=C - 1), 1.0)

        dims = (0, 2, 3)
        intersection = torch.sum(probs * true_1_hot, dims)
        cardinality = torch.sum(probs + true_1_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - torch.mean(dice)

# -----------------------------------------------------------------------
# Train / Infer
# -----------------------------------------------------------------------
def trainPhase():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FolderSegDataset(
        root_dir=DATA_ROOT,
        height=globalH,
        width=globalW,
        num=LIMIT_TRAIN_NUM,
        image_dirname=IMAGE_DIRNAME,
        mask_dirname=MASK_DIRNAME,
        force_gray=FORCE_GRAY,
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)

    in_ch = 1 if FORCE_GRAY else 3
    num_classes = int(dataset.num_classes)

    model = SegWithRCM(in_channels=in_ch, num_classes=num_classes, dim=global_dim, dw_size=5).to(device)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = SoftDiceLoss()
    criterion_bce = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    print(f"[Train] device={device} | in_ch={in_ch} | num_classes={num_classes} | N={len(dataset)} | BS={BATCH_SIZE}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        total = len(dataloader.dataset)

        for inputs, masks in dataloader:
            inputs = inputs.to(device, non_blocking=False)
            masks = masks.to(device, non_blocking=False)

            optimizer.zero_grad()

            logits, strip_h_logits, strip_w_logits = model(inputs)
            # logits: (B,C,H,W), strip_h: (B,C,H,1), strip_w: (B,C,1,W)

            loss_ce = criterion_ce(logits, masks)
            loss_dice = criterion_dice(logits, masks)
            loss_main = 0.5 * loss_ce + 0.5 * loss_dice

            # Binary strip supervision: "does any non-background class appear in this row/col?"
            bin_masks = (masks > 0).float()
            target_h = bin_masks.max(dim=2)[0]  # (B,H)
            target_w = bin_masks.max(dim=1)[0]  # (B,W)

            # Use aggregated non-background logits for BCE
            # strip_h_logits: (B,C,H,1) -> (B,H)
            if strip_h_logits.shape[1] > 1:
                pred_h_fg = torch.logsumexp(strip_h_logits[:, 1:, :, :], dim=1).squeeze(-1)
                pred_w_fg = torch.logsumexp(strip_w_logits[:, 1:, :, :], dim=1).squeeze(1)
            else:
                pred_h_fg = strip_h_logits[:, 0, :, :].squeeze(-1)
                pred_w_fg = strip_w_logits[:, 0, :, :].squeeze(1)

            loss_h = criterion_bce(pred_h_fg, target_h)
            loss_w = criterion_bce(pred_w_fg, target_w)

            loss = loss_main + AUX_WEIGHT * (loss_h + loss_w)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * int(inputs.size(0))

        scheduler.step()

        epoch_loss = running_loss / max(1, total)
        print(f"[Train] Epoch {epoch+1}/{NUM_EPOCHS} | Loss={epoch_loss:.4f} | LR={optimizer.param_groups[0]['lr']:.6f}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_ch": in_ch,
            "num_classes": num_classes,
            "force_gray": FORCE_GRAY,
            "H": globalH,
            "W": globalW,
        },
        SAVE_WEIGHTS
    )
    print(f"[Train] Saved: {SAVE_WEIGHTS}")

def inferencePhase():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(SAVE_WEIGHTS):
        print(f"[Infer] Weights not found: {SAVE_WEIGHTS}")
        return

    ckpt = torch.load(SAVE_WEIGHTS, map_location=device)
    in_ch = int(ckpt.get("in_ch", 1))
    num_classes = int(ckpt.get("num_classes", 2))

    model = SegWithRCM(in_channels=in_ch, num_classes=num_classes, dim=global_dim, dw_size=5)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()

    dataset = FolderSegDataset(
        root_dir=DATA_ROOT,
        height=globalH,
        width=globalW,
        num=LIMIT_INFER_NUM,
        image_dirname=IMAGE_DIRNAME,
        mask_dirname=MASK_DIRNAME,
        force_gray=(in_ch == 1),
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[Infer] device={device} | in_ch={in_ch} | num_classes={num_classes} | N={len(dataset)}")

    with torch.no_grad():
        for idx, (inputs, masks) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=False)
            masks_np = masks.numpy().squeeze().astype(np.int32)

            logits_up, att_dict = model(inputs, return_att=True)
            preds = torch.argmax(logits_up, dim=1).cpu().numpy().squeeze().astype(np.int32)

            if in_ch == 1:
                input_np = inputs.cpu().numpy().squeeze().astype(np.float32)
            else:
                input_np = inputs.cpu().numpy().squeeze().transpose(1, 2, 0).astype(np.float32)

            att_raw_np = att_dict["raw"].cpu().numpy().squeeze()
            att_h_np = att_dict["after_h"].cpu().numpy().squeeze()
            att_v_np = att_dict["after_v"].cpu().numpy().squeeze()
            att_mlp_np = att_dict["mlp_res"].cpu().numpy().squeeze()

            fig, axes = plt.subplots(2, 4, figsize=(16, 8))

            if in_ch == 1:
                axes[0, 0].imshow(input_np, cmap="gray")
            else:
                axes[0, 0].imshow(np.clip(input_np, 0.0, 1.0))
            axes[0, 0].set_title("Input")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(masks_np, cmap="gray" if num_classes <= 2 else None)
            axes[0, 1].set_title("GT Mask")
            axes[0, 1].axis("off")

            axes[0, 2].imshow(preds, cmap="gray" if num_classes <= 2 else None)
            axes[0, 2].set_title("Prediction")
            axes[0, 2].axis("off")

            axes[0, 3].imshow(att_mlp_np, cmap="jet")
            axes[0, 3].set_title("Feature Refined")
            axes[0, 3].axis("off")

            axes[1, 0].imshow(att_raw_np, cmap="jet")
            axes[1, 0].set_title("Context (H+W)")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(att_h_np, cmap="jet")
            axes[1, 1].set_title("Horizontal Strip Attn")
            axes[1, 1].axis("off")

            axes[1, 2].imshow(att_v_np, cmap="jet")
            axes[1, 2].set_title("Vertical Strip Attn")
            axes[1, 2].axis("off")

            # simple error map
            err = (preds != masks_np).astype(np.uint8)
            axes[1, 3].imshow(err, cmap="gray")
            axes[1, 3].set_title("Error (pred!=gt)")
            axes[1, 3].axis("off")

            plt.tight_layout()
            out_path = os.path.join(OUT_DIR, f"viz_{idx:03d}.png")
            plt.savefig(out_path, dpi=150)
            plt.close(fig)

            print(f"[Infer] Saved: {out_path}")

if __name__ == "__main__":
    trainPhase()
    inferencePhase()
