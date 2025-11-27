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




