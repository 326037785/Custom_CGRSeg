## [ECCV 2024] Context-Guided Spatial Feature Reconstruction for Efficient Semantic Segmentation

Zhenliang Ni, Xinghao Chen, Yingjie Zhai, Yehui Tang, and Yunhe Wang

## 🔥 Updates
* **2025/01/01**: Updated for 2025 with PyTorch 2.x and OpenCV 4.x support. Added a user-friendly single-GPU training/evaluation entrypoint.
* **2024/10/08**: The training code is released and fixed bugs in the issue.
* **2024/07/01**: The paper of CGRSeg is accepted by ECCV 2024.
* **2024/05/10**: Codes of CGRSeg are released in [Pytorch](https://github.com/nizhenliang/CGRSeg/) and paper in [[arXiv]](https://arxiv.org/abs/2405.06228).

## 📸 Overview
<img width="784" alt="cgrseg2" src="https://github.com/nizhenliang/CGRSeg/assets/48742170/eef8c502-599d-48aa-b05b-51a682ac7456">

The overall architecture of CGRSeg. The Rectangular Self-Calibration Module (RCM) is designed for spatial feature reconstruction and pyramid context extraction. 
The rectangular self-calibration attention (RCA) explicitly models the rectangular region and calibrates the attention shape. The Dynamic Prototype Guided (DPG) head
is proposed to improve the classification of the foreground objects via explicit class embedding.

<img width="731" alt="flops" src="https://github.com/nizhenliang/CGRSeg/assets/48742170/2bdf4e0c-d4a7-4b83-b091-394d1ee0afaa">

##  1️⃣ Results

#### ADE20K

<img width="539" alt="ade20k" src="https://github.com/nizhenliang/CGRSeg/assets/48742170/98e14385-8f41-417c-84d9-3cc6db0d32c1">

COCO-Stuff-10k

<img width="491" alt="coco" src="https://github.com/nizhenliang/CGRSeg/assets/48742170/9bf2487f-27d6-41d1-8e94-26f3fd994ce0">

Pascal Context

<img width="481" alt="pc" src="https://github.com/nizhenliang/CGRSeg/assets/48742170/d0b3f524-523f-4fc3-a809-691f4617ebb4">

##  2️⃣ Quick Start (2025-ready)

1. **Create environment** (PyTorch 2.x + OpenCV 4.x):

   ```shell
   conda create --name cgrseg python=3.10 -y
   conda activate cgrseg

   # Install PyTorch 2.x (pick the CUDA build that matches your driver)
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
       --index-url https://download.pytorch.org/whl/cu118

   # Install OpenCV 4.x and other dependencies
   pip install opencv-python>=4.8.0 timm>=0.9.0 mmcv>=2.0.0 mmsegmentation>=1.0.0
   ```

2. **(Optional) Legacy environment** (for reproducibility with the original release):

   ```shell
   conda create --name ssa python=3.8 -y
   conda activate ssa
   pip install torch==1.8.2+cu102 torchvision==0.9.2+cu102 torchaudio==0.8.2
   pip install timm==0.6.13 mmcv-full==1.6.1 opencv-python==4.1.2.30
   pip install "mmsegmentation==0.27.0"
   ```

CGRSeg is built on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). If you see missing-operator errors, upgrade MMCV/MMDetection/MMEngine to the latest compatible versions for your CUDA toolkit.

3. **IDE-friendly defaults**: open `run_cgrseg.py`, locate the `DEFAULT_HPARAMS` dict near the top, and edit values (learning rate, batch size, data root, etc.). Both the CLI and Python API pick up those defaults automatically so you can hit “Run” in VSCode/PyCharm without extra flags.

## 3️⃣ Dataset Layout (copy & paste ready)

Place your datasets under a single `data/` directory so the configs work out of the box. The ADE20K layout expected by `local_configs/cgrseg/cgrseg-t_ade20k_160k.py` is:

```
data/
└── ade/
    └── ADEChallengeData2016/
        ├── images/
        │   ├── training/
        │   └── validation/
        └── annotations/
            ├── training/
            └── validation/
```

Other supported datasets follow a similar pattern:

- **COCO-Stuff-10k**

  ```
  data/
  └── coco_stuff10k/
      ├── images/
      │   ├── train2014/
      │   └── test2014/
      └── annotations/
          ├── train2014/
          └── test2014/
  ```

- **Pascal Context**

  ```
  data/
  └── VOCdevkit/
      └── VOC2010/
          ├── JPEGImages/
          └── SegmentationClassContext/
  ```

**Tips for newcomers**
- Download the raw data from the official pages (ADE20K: http://sceneparsing.csail.mit.edu/, COCO-Stuff: https://github.com/nightrome/cocostuff, Pascal Context: https://cs.stanford.edu/~roozbeh/pascal-context/).
- Unzip directly into `data/` so the folder names match the trees above.
- If you store data elsewhere, pass `--data-root /path/to/ADEChallengeData2016` to `run_cgrseg.py`.

## 4️⃣ Train & Evaluate (single GPU)

Use the unified helper `run_cgrseg.py` for both training and inference. All hyperparameters are exposed as flags so you can adjust them without editing the config.

```shell
# Train with the default config (ADE20K single-scale)
python run_cgrseg.py --mode train

# Train with custom hyperparameters and a custom data root
python run_cgrseg.py --mode train \
    --lr 0.0001 --batch-size 4 --max-iters 80000 \
    --eval-interval 4000 --data-root ./data/ade/ADEChallengeData2016 \
    --work-dir ./work_dirs/my_experiment

# Evaluate an existing checkpoint (optionally save visualizations)
python run_cgrseg.py --mode eval \
    --checkpoint ./work_dirs/cgrseg-t_ade20k_160k/latest.pth \
    --show-dir ./results
```

Available knobs for beginners:
- `--lr`: Learning rate (defaults to config value `0.00012`).
- `--batch-size`: Samples per GPU (default: `4` for training, `1` for eval).
- `--max-iters`: Total training iterations (default: `160000`).
- `--eval-interval`: Validation interval during training (default: `4000`).
- `--data-root`: Point the dataset somewhere else without editing configs.
- `--work-dir`: Where to save logs/checkpoints (default: `./work_dirs/<config-name>`).

Prefer Python APIs? Call the built-in helper from any script or notebook:

```python
from run_cgrseg import run_single_gpu

# Train
run_single_gpu('train', work_dir='./work_dirs/quick_start')

# Evaluate
run_single_gpu('eval', checkpoint='./work_dirs/cgrseg-t_ade20k_160k/latest.pth')
```

**VSCode / PyCharm debugging**
- Set your run configuration to `run_cgrseg.py`.
- Adjust quick knobs in `DEFAULT_HPARAMS` (e.g., `batch_size`, `lr`, `data_root`).
- For evaluation, add `checkpoint` there or pass it in your Run/Debug parameters.
- Start debugging; the script prints the active settings before training/eval begins.

### Original Scripts

- Train
  
  ```shell
  # Single-gpu training
  python tools/train.py local_configs/cgrseg/cgrseg-t_ade20k_160k.py
  
  # Multi-gpu (4-gpu) training
  sh tools/dist_train.sh local_configs/cgrseg/cgrseg-t_ade20k_160k.py 4
  ```

- Test
  
  ```shell
  # Single-gpu testing
  python tools/test.py local_configs/cgrseg/cgrseg-t_ade20k_160k.py ${CHECKPOINT_FILE} --eval mIoU
  
  # Multi-gpu (4-gpu) testing
  sh tools/dist_test.sh local_configs/cgrseg/cgrseg-t_ade20k_160k.py ${CHECKPOINT_FILE} 4 --eval mIoU
  ```
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=nizhenliang/CGRSeg&type=date&legend=top-left)](https://www.star-history.com/#nizhenliang/CGRSeg&type=date&legend=top-left)

## ✏️ Reference
If you find CGRSeg useful in your research or applications, please consider giving a star ⭐ and citing using the following BibTeX:
```
@inproceedings{ni2024context,
  title={Context-guided spatial feature reconstruction for efficient semantic segmentation},
  author={Ni, Zhenliang and Chen, Xinghao and Zhai, Yingjie and Tang, Yehui and Wang, Yunhe},
  booktitle={European Conference on Computer Vision},
  pages={239--255},
  year={2024},
  organization={Springer}
}
```

