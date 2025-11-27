## [ECCV 2024] Context-Guided Spatial Feature Reconstruction for Efficient Semantic Segmentation

Zhenliang Ni, Xinghao Chen, Yingjie Zhai, Yehui Tang, and Yunhe Wang

## 🔥 Updates
* **2025/01/01**: Updated for 2025 with PyTorch 2.x and OpenCV 4.x support. Added user-friendly training and evaluation script.
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

##  2️⃣ Requirements

CGRSeg is built based on **PyTorch** and **OpenCV**. Below are the installation instructions:

### For 2025 (Recommended - PyTorch 2.x and OpenCV 4.x)

```shell
conda create --name cgrseg python=3.10 -y
conda activate cgrseg

# Install PyTorch 2.x (adjust CUDA version as needed)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install OpenCV 4.x
pip install opencv-python>=4.8.0

# Install other dependencies
pip install timm>=0.9.0
pip install mmcv>=2.0.0
pip install mmsegmentation>=1.0.0
```

### Legacy Installation (Original - for compatibility with older environments)

```shell
conda create --name ssa python=3.8 -y
conda activate ssa
pip install torch==1.8.2+cu102 torchvision==0.9.2+cu102 torchaudio==0.8.2
pip install timm==0.6.13
pip install mmcv-full==1.6.1
pip install opencv-python==4.1.2.30
pip install "mmsegmentation==0.27.0"
```

CGRSeg is built based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), which can be referenced for data preparation.

## 3️⃣ Dataset Structure

Before training or evaluation, organize your dataset in the following structure:

### ADE20K Dataset

```
data/
└── ade/
    └── ADEChallengeData2016/
        ├── images/
        │   ├── training/
        │   │   ├── ADE_train_00000001.jpg
        │   │   ├── ADE_train_00000002.jpg
        │   │   └── ...
        │   └── validation/
        │       ├── ADE_val_00000001.jpg
        │       ├── ADE_val_00000002.jpg
        │       └── ...
        └── annotations/
            ├── training/
            │   ├── ADE_train_00000001.png
            │   ├── ADE_train_00000002.png
            │   └── ...
            └── validation/
                ├── ADE_val_00000001.png
                ├── ADE_val_00000002.png
                └── ...
```

### COCO-Stuff-10k Dataset

```
data/
└── coco_stuff10k/
    ├── images/
    │   ├── train2014/
    │   │   └── ...
    │   └── test2014/
    │       └── ...
    └── annotations/
        ├── train2014/
        │   └── ...
        └── test2014/
            └── ...
```

### Pascal Context Dataset

```
data/
└── VOCdevkit/
    └── VOC2010/
        ├── JPEGImages/
        │   └── ...
        └── SegmentationClassContext/
            └── ...
```

**Note:** You can download these datasets from their official sources:
- ADE20K: http://sceneparsing.csail.mit.edu/
- COCO-Stuff: https://github.com/nightrome/cocostuff
- Pascal Context: https://cs.stanford.edu/~roozbeh/pascal-context/

## 4️⃣ Training & Testing

### Quick Start with Unified Script (Recommended)

We provide a user-friendly script `run_cgrseg.py` that combines training and evaluation in one place with easy hyperparameter adjustment:

```shell
# Training (single-GPU)
python run_cgrseg.py --mode train --config local_configs/cgrseg/cgrseg-t_ade20k_160k.py

# Training with custom hyperparameters
python run_cgrseg.py --mode train \
    --config local_configs/cgrseg/cgrseg-t_ade20k_160k.py \
    --lr 0.0001 \
    --batch-size 4 \
    --max-iters 80000 \
    --work-dir ./work_dirs/my_experiment

# Evaluation (single-GPU)
python run_cgrseg.py --mode eval \
    --config local_configs/cgrseg/cgrseg-t_ade20k_160k.py \
    --checkpoint ./work_dirs/cgrseg-t_ade20k_160k/latest.pth

# Show evaluation results with visualization
python run_cgrseg.py --mode eval \
    --config local_configs/cgrseg/cgrseg-t_ade20k_160k.py \
    --checkpoint ./work_dirs/cgrseg-t_ade20k_160k/latest.pth \
    --show-dir ./results
```

**Available Hyperparameters:**
- `--lr`: Learning rate (default: 0.00012)
- `--batch-size`: Samples per GPU (default: 4)
- `--max-iters`: Maximum training iterations (default: 160000)
- `--eval-interval`: Evaluation interval during training (default: 4000)
- `--seed`: Random seed for reproducibility
- `--work-dir`: Directory to save logs and checkpoints

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

