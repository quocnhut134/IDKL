# Visible-Infrared Person Re-Identification on Resource-Constrained Hardware using Optimized IDKL Model

![Base-Model](https://img.shields.io/badge/Base--Model-IDKL-blue)

## Introduction

This project addresses the challenge of training **IDKL** - a Visible-Infrared Person Re-Identification (VI-ReID) model, on **weak hardware devices**.

We propose an **Optimized IDKL Model** that builds upon the state-of-the-art [IDKL framework](https://arxiv.org/abs/2403.11708). While the original IDKL focuses on maximizing accuracy using complex dual-stream architectures, our work focuses on **inference efficiency**.

By leveraging **Knowledge Distillation** and **Attention Mechanisms**, we transfer discriminative knowledge from a heavy "Specific Branch" to a lightweight "Shared Branch". This allows us to deploy a compact model that retains high accuracy but runs with significantly lower latency and memory footprint.

## Authors

This project was managed and developed by:

-   **Duong Quoc Nhut** - GitHub: `https://github.com/quocnhut134`

- **Le Hong Phong** - GitHub: `https://github.com/hongphong1504`
---

## Key Optimizations for Hardware Efficiency

To achieve high accuracy on resource-constrained devices, we moved away from simply adding heavy layers. Instead, we optimized the "core" of the network and the training strategy. Our approach focuses on **Efficiency-Driven Design**:

### 1. Backbone Optimization: ResNet-IBN (Replacing Standard ResNet)
* **The Change:** We replaced the standard ResNet50 backbone with **IBN-Net** (ResNet with Instance Batch Normalization).
* **Why for Weak Hardware?** Standard CNNs struggle with the domain gap between RGB and IR images. Instead of adding heavy alignment modules, IBN-Net integrates invariance learning directly into the backbone blocks.
* **Benefit:** Improves domain generalization significantly with **negligible computational overhead** compared to the baseline.

### 2. Attention-Guided Feature Refining (Multi-Head Attention)
* **The Change:** Integration of a **Multi-Head Attention (MHA)** mechanism.
* **Why for Weak Hardware?** In limited-resolution scenarios (common in CCTV), details matter. MHA allows the model to actively attend to fine-grained discriminative patterns (specific body parts, logos) while suppressing background noise.
* **Benefit:** Maximizes the "Information Density" of extracted features, ensuring the lightweight model focuses only on pixels that count.

### 3. Data-Centric Optimization (Advanced Augmentation)
* **The Change:** Implementation of robust **Cross-Modality Data Augmentation** strategies (e.g., Channel Augmentation, Random Erasing adapted for ReID).
* **Why for Weak Hardware?** This is a **"Zero-Cost" improvement**. We artificially increase the diversity of training data to simulate challenging environments.
* **Benefit:** The deployed model becomes robust to lighting changes and occlusions without needing a larger architecture.

### 4. Adaptive Training Strategy
* **The Change:** We introduce an **Adaptive Training** mechanism (including dynamic loss weighting and optimized curriculum learning).
* **Why for Weak Hardware?** A well-trained small model often outperforms a poorly-trained large model. We optimize the convergence path to squeeze the maximum performance out of the Shared Branch.
* **Benefit:** Ensures the model reaches its peak potential (Rank-1: 75.53%) within the constraints of the architecture.


## Installation

```bash
# Clone this repository
git clone [https://github.com/quocnhut134/IDKL](https://github.com/quocnhut134/IDKL)
cd idkl

# Install dependencies
pip install -r requirements.txt

```

## Performance on SYSU-MM01
Comparison of CMC (%) and mAP (%) performances with the Origin IDKL Model on SYSU-MM01 dataset

<img width="847" height="189" alt="Image" src="https://github.com/user-attachments/assets/a37b6fbd-714b-4d94-a3cd-2740f03134e1" />

## Training Command

To reproduce our results with the optimized hyper-parameters:

```bash
python train.py --cfg ./configs/SYSU.yml --p_size 8 --k_size 4 --weight_sep 0.1 --weight_KL 0.6 --weight_sid 0.8

```

## Acknowledgement & Reference

### 1. Dataset Usage (SYSU-MM01)

This project utilizes the **SYSU-MM01** dataset for training and evaluation.

**Important Notice regarding Data Access:**
The SYSU-MM01 dataset is **not open for public download** without permission. It is restricted to academic research purposes only.

* We do not distribute the dataset.
* To obtain access, please follow the official instructions provided by the original authors (usually via email request or an application form).
* Please refer to their official repository or paper for the release agreement: **[SYSU-MM01 Official Page](https://github.com/wuancong/SYSU-MM01)**

If you use this dataset, please cite the original paper:

```bibtex
Ancong Wu, Wei-Shi Zheng, Hong-Xing Yu, Shaogang Gong and Jianhuang Lai. RGB-IR Person Re-Identification by Cross-Modality Similarity Preservation. International Journal of Computer Vision (IJCV), 2020}

```

### 2. Base Framework (IDKL)

Our code is developed based on the **Implicit Discriminative Knowledge Learning (IDKL)** framework. We sincerely thank the authors for their excellent contribution to the open-source community.

* **Paper:** [Implicit Discriminative Knowledge Learning for Visible-Infrared Person Re-Identification (CVPR)](https://arxiv.org/abs/2403.11708)
* **Official Code:** [github.com/1KK077/IDKL](https://github.com/1KK077/IDKL)

If you find our code useful, please also cite the IDKL paper:

```bibtex
@article{ren2024implicit,
  title={Implicit Discriminative Knowledge Learning for Visible-Infrared Person Re-Identification},
  author={Kaijie Ren, Lei Zhang},
  journal={arXiv preprint arXiv:2403.11708},
  year={2024}
}

```
