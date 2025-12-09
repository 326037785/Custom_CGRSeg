import os
import cv2
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

globalH = 256
globalW = 256


def make_fake_data(num: int = 8, height: int = 64, width: int = 64):
    """
    生成 (N, 1, H, W) 的图像和 (N, H, W) 的二值 mask
    raw data 通过腐蚀/膨胀 + blur + 噪声，不再是完美方块，
    方便观察条带卷积对"矩形注意力"的修饰效果。
    """
    datas = []
    targets = []

    for _ in range(num):
        image = np.zeros((height, width), dtype=np.float32)
        mask = np.zeros((height, width), dtype=np.int64)

        rect_num = np.random.randint(1, 4)  # 1~3 个矩形
        for _rect_idx in range(rect_num):
            rect_h = np.random.randint(8, height // 3)
            rect_w = np.random.randint(8, width // 3)
            y = np.random.randint(0, height - rect_h)
            x = np.random.randint(0, width - rect_w)

            image[y:y + rect_h, x:x + rect_w] = 1.0
            mask[y:y + rect_h, x:x + rect_w] = 1  # 最终就做 2 类：背景 + 前景

        # 噪声
        noise = np.random.normal(0, 0.1, (height, width)).astype(np.float32)
        image = image + noise

        # 形态学操作：让边缘更 irregular
        if np.random.random() > 0.5:
            image = scipy.ndimage.binary_erosion(image > 0.5).astype(np.float32)
        else:
            image = scipy.ndimage.binary_dilation(image > 0.5).astype(np.float32)

        # blur 一下
        image = scipy.ndimage.gaussian_filter(image, sigma=1.0).astype(np.float32)

        # 再加一点噪声
        image = image + np.random.normal(0, 0.05, (height, width)).astype(np.float32)

        image = np.clip(image, 0, 1).astype(np.float32)

        # 最终 mask：二值
        mask = (image > 0.3).astype(np.int64)

        datas.append(torch.from_numpy(image).unsqueeze(0))  # (1,H,W)
        targets.append(torch.from_numpy(mask))              # (H,W)

    datas = torch.stack(datas, dim=0)      # (N,1,H,W)
    targets = torch.stack(targets, dim=0)  # (N,H,W)
    return datas.float(), targets.long()


class FakeRectDataset(Dataset):
    def __init__(self, num: int = 8, height: int = 64, width: int = 64):
        super().__init__()
        self.data, self.target = make_fake_data(num, height, width)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


class FakeDataLoader(DataLoader):
    def __init__(self, num: int = 8, height: int = 64, width: int = 64,
                 batch_size: int = 4, shuffle: bool = True):
        dataset = FakeRectDataset(num, height, width)
        super(FakeDataLoader, self).__init__(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle)

class SmoothLayer(nn.Module):
    """
    对 1 通道注意力图做 horizontal + vertical strip conv 平滑。
    channels=1 即注意力通道数为 1。
    """
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size

        # 横向条带卷积 (1, k)
        self.horizontal_kernel = nn.Conv2d(
            1, 1, kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
            bias=False
        )
        # 纵向条带卷积 (k, 1)
        self.vertical_kernel = nn.Conv2d(
            1, 1, kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
            bias=False
        )

        # 初始化为平均平滑
        with torch.no_grad():
            self.horizontal_kernel.weight.fill_(1.0 / kernel_size)
            self.vertical_kernel.weight.fill_(1.0 / kernel_size)

    def forward(self, x: torch.Tensor, return_intermediate: bool = False):
        """
        x: (B,1,H,W) 注意力图
        return_intermediate=True 时返回 horizontal 之后和 vertical 之后的结果
        """
        inter_list = []

        x = self.horizontal_kernel(x)
        if return_intermediate:
            inter_list.append(x.clone())

        x = self.vertical_kernel(x)
        if return_intermediate:
            inter_list.append(x.clone())
            return x, inter_list

        return x


# ----------------------------------------------------------
# 3. 简单编码器 + 最高层矩形注意力 + 条带平滑 + 分割 head
# ----------------------------------------------------------
class RectSmoothSegNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, kernel_size: int = 5):
        super().__init__()
        # encoder：三层，逐步下采样，最后得到高语义特征 F
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True)
        )

        # 高层矩形注意力分支：F -> 1 通道 attention logits
        self.attn_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # 条带平滑放在最高层注意力上
        self.smooth = SmoothLayer(kernel_size=kernel_size)

        # 分割 head：用平滑后的注意力调节 F，再预测类别
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """
        return_attn=False: 只返回分割 logits（训练阶段）
        return_attn=True:  返回 logits, attn_raw_up, attn_horizontal_up, attn_smooth_up（推理阶段看效果用）
        """
        H, W = x.shape[2], x.shape[3]

        f1 = self.enc1(x)   # (B,16,H/2,W/2)
        f2 = self.enc2(f1)  # (B,32,H/4,W/4)
        f3 = self.enc3(f2)  # (B,64,H/8,W/8)

        # 最高语义层 attention（矩形相关语义）
        attn_logits = self.attn_conv(f3)           # (B,1,h,w)
        attn_raw = torch.sigmoid(attn_logits)      # (B,1,h,w)

        # 条带卷积在这里作用：horizontal + vertical，并返回中间结果
        if return_attn:
            attn_smooth, inter_list = self.smooth(attn_raw, return_intermediate=True)
            attn_horizontal = inter_list[0]  # 经过横向卷积后
            attn_vertical = inter_list[1]     # 经过纵向卷积后（即最终）
        else:
            attn_smooth = self.smooth(attn_raw)

        # 用平滑后的注意力调节 feature（一个简单的 gating 例子）
        f3_enhanced = f3 * (1.0 + attn_smooth)     # 1+attn 防止全抑制

        logits = self.seg_head(f3_enhanced)        # (B,num_classes,h,w)
        logits_up = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

        if not return_attn:
            return logits_up

        # 为了可视化，把注意力图上采样到输入分辨率
        attn_raw_up = F.interpolate(attn_raw, size=(H, W), mode="bilinear", align_corners=False)
        attn_horizontal_up = F.interpolate(attn_horizontal, size=(H, W), mode="bilinear", align_corners=False)
        attn_smooth_up = F.interpolate(attn_smooth, size=(H, W), mode="bilinear", align_corners=False)

        return logits_up, attn_raw_up, attn_horizontal_up, attn_smooth_up


# ----------------------------------------------------------
# 4. 训练：只用分割 loss，attention 是内部机制
# ----------------------------------------------------------
def trainPhase():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FakeRectDataset(num=32, height=64, width=64)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = RectSmoothSegNet(in_channels=1, num_classes=2, kernel_size=5).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 120
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total = len(dataloader.dataset)

        for inputs, masks in dataloader:
            inputs = inputs.to(device)  # (B,1,H,W)
            masks = masks.to(device)    # (B,H,W)

            optimizer.zero_grad()
            outputs = model(inputs)     # (B,2,H,W)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / total
        if (epoch + 1) % 5 == 0:
            print(f"[Train] Epoch {epoch+1}/{num_epochs}  Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "rect_smooth_segnet.pth")
    print("Model saved to rect_smooth_segnet.pth")


# ----------------------------------------------------------
# 5. 推理 + 可视化 strip conv 对矩形注意力的修饰效果
# ----------------------------------------------------------
def _to_uint8(img: np.ndarray):
    """归一化到 0~255，防止除 0。"""
    img = img.astype(np.float32)
    vmin, vmax = img.min(), img.max()
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    else:
        img = np.zeros_like(img)
    return (img * 255.0).astype(np.uint8)


def inferencePhase():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RectSmoothSegNet(in_channels=1, num_classes=2, kernel_size=5)
    model.load_state_dict(torch.load("rect_smooth_segnet.pth", map_location=device))
    model.to(device)
    model.eval()

    dataset = FakeRectDataset(num=4, height=64, width=64)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    out_dir = "inference_results_rect_smooth"
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (inputs, masks) in enumerate(dataloader):
            inputs = inputs.to(device)         # (1,1,H,W)
            masks = masks.to(device)           # (1,H,W)

            logits, attn_raw_up, attn_horizontal_up, attn_smooth_up = model(inputs, return_attn=True)

            preds = torch.argmax(logits, dim=1).cpu().numpy()     # (1,H,W)
            attn_raw_up_np = attn_raw_up.cpu().numpy()            # (1,1,H,W)
            attn_horizontal_up_np = attn_horizontal_up.cpu().numpy()  # (1,1,H,W)
            attn_smooth_up_np = attn_smooth_up.cpu().numpy()      # (1,1,H,W)

            input_np = inputs.cpu().numpy().squeeze()             # (H,W)
            mask_np = masks.cpu().numpy().squeeze()               # (H,W)
            pred_np = preds.squeeze()                             # (H,W)
            attn_raw_np = attn_raw_up_np.squeeze()                # (H,W)
            attn_horizontal_np = attn_horizontal_up_np.squeeze()  # (H,W)
            attn_smooth_np = attn_smooth_up_np.squeeze()          # (H,W)

            # 保存渐进式结果
            cv2.imwrite(os.path.join(out_dir, f"input_{idx}.png"), _to_uint8(input_np))
            cv2.imwrite(os.path.join(out_dir, f"mask_{idx}.png"), _to_uint8(mask_np))
            cv2.imwrite(os.path.join(out_dir, f"pred_{idx}.png"), _to_uint8(pred_np))
            cv2.imwrite(os.path.join(out_dir, f"attn_0_raw_{idx}.png"), _to_uint8(attn_raw_np))
            cv2.imwrite(os.path.join(out_dir, f"attn_1_horizontal_{idx}.png"), _to_uint8(attn_horizontal_np))
            cv2.imwrite(os.path.join(out_dir, f"attn_2_vertical_{idx}.png"), _to_uint8(attn_smooth_np))

            # 可视化渐进式变化（使用matplotlib）
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            axes[0, 0].imshow(input_np, cmap='gray')
            axes[0, 0].set_title('Input')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(mask_np, cmap='gray')
            axes[0, 1].set_title('Ground Truth')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(pred_np, cmap='gray')
            axes[0, 2].set_title('Prediction')
            axes[0, 2].axis('off')
            
            axes[1, 0].imshow(attn_raw_np, cmap='jet')
            axes[1, 0].set_title('Attention Raw')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(attn_horizontal_np, cmap='jet')
            axes[1, 1].set_title('After Horizontal Conv')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(attn_smooth_np, cmap='jet')
            axes[1, 2].set_title('After Vertical Conv')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"progressive_viz_{idx}.png"), dpi=150, bbox_inches='tight')
            plt.close()

            print(f"[Infer] Saved sample {idx} to {out_dir}/")


if __name__ == "__main__":
    trainPhase()
    inferencePhase()
