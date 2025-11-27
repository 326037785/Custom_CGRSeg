"""
Quick helper to assemble the CGRSeg network with the default EfficientFormerV2
backbone and print its structure.

The script mirrors the official ADE20K single-scale config
(`local_configs/cgrseg/cgrseg-t_ade20k_160k.py`) but skips loading pretrained
weights so it can run anywhere. Use the ``__main__`` block to instantiate the
model and run a dummy forward pass to inspect the shapes.
"""

import torch
from mmcv import Config
from mmseg.models import build_segmentor


def build_default_cgrseg(config_path: str = "local_configs/cgrseg/cgrseg-t_ade20k_160k.py"):
    """Build the CGRSeg model with the default EfficientFormerV2 backbone.

    Args:
        config_path: Path to the base config that defines the model.

    Returns:
        torch.nn.Module: Instantiated segmentation model ready for inspection.
    """
    cfg = Config.fromfile(config_path)

    # Avoid loading pretrained weights to keep the helper lightweight.
    if "pretrained" in cfg.model:
        cfg.model["pretrained"] = None
    if cfg.model.get("backbone") and "init_cfg" in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    model = build_segmentor(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    model.init_weights()
    return model


if __name__ == "__main__":
    net = build_default_cgrseg()
    net.eval()

    dummy_input = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        if hasattr(net, "forward_dummy"):
            output = net.forward_dummy(dummy_input)
        else:
            output = net(dummy_input)

    print(net)
    print("\nOutput type:", type(output))
    if isinstance(output, torch.Tensor):
        print("Output shape:", output.shape)
    else:
        print("Output structure:", {k: v.shape for k, v in output.items()})
