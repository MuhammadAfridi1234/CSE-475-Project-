# CSE-475-Project(machine Learning)
**Turmeric plant disease dataset ( Self Supervised Image classification)**

Dataset link: https://data.mendeley.com/datasets/g46dvrcvwn/2

<div align="center">

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/Self_Supervised_Learning-000000?style=for-the-badge&logo=facebook&logoColor=white" alt="SSL">
<img src="https://img.shields.io/badge/BYOL-5A9E6F?style=for-the-badge&logo=python&logoColor=white" alt="BYOL">
<img src="https://img.shields.io/badge/DINO-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="DINO">
<br />
<br />

# 🌿 TurmericGuard: Self-Supervised Plant Disease Classification

## BYOL (ViT‑B/16) + DINO (Swin‑T) for Turmeric Disease Detection

[![Paper](https://img.shields.io/badge/arXiv-Paper-B31B1B?logo=arxiv)](https://arxiv.org/abs/)
[![Dataset](https://img.shields.io/badge/Mendeley-Dataset-FF6B6B?logo=mendeley)](https://data.mendeley.com/datasets/g46dvrcvwn/2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

</div>

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Methods & Architectures](#-methods--architectures)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Benchmark Results](#-benchmark-results)
- [Group Members](#-group-members)
- [Citations & References](#-citations--references)
- [License](#-license)

---

## 🎯 Overview

**TurmericGuard** is a comprehensive self‑supervised learning (SSL) benchmark for plant disease classification. We implement and compare **two state‑of‑the‑art SSL methods** – **BYOL** and **DINO** – on the **Turmeric Plant Disease Dataset**. Unlike supervised baselines that require thousands of labelled images, our SSL pipelines learn rich visual representations from **unlabelled data**, achieving competitive accuracy with only 10% labelled examples for linear probing.

| Method | Backbone | Core Idea | Key Advantage |
|:-------|:---------|:----------|:---------------|
| **BYOL** | ViT‑B/16 (pretrained) | Predicts target network representation from online view | No negative pairs, stable training |
| **DINO** | Swin‑T (trained from scratch) | Self‑distillation with multi‑crop & teacher‑student | Learns local-to-global correspondences |

> ⚡ **Why Self‑Supervised Learning?** Agricultural datasets are expensive to label. SSL methods leverage **unlabelled images** (80% of our data) to pre‑train a backbone, then only a tiny labelled set (10%) is needed for a linear classifier or k‑NN – dramatically reducing annotation cost.

---

## 🌾 Dataset

We use the **Turmeric Plant Disease Dataset**, version 2, from Mendeley Data [1]. The dataset contains images of turmeric plants affected by various diseases, organised into **5 classes**:

| Class | Example Count (Original) |
|:------|:-------------------------:|
| Dry Leaf | 221 |
| Healthy Leaf | 213 |
| Leaf Blotch | 238 |
| Rhizome Disease Root | 165 * |
| Rhizome Healthy Root | 140 * |

*Approximate – exact numbers may vary. Total original images: **~1,063**.  
(All images are augmented on‑the‑fly during SSL pre‑training with random crops, colour jitter, blur, etc.)

### 📸 Sample Visualisations

| Healthy Leaf | Leaf Blotch | Dry Leaf | Rhizome Disease |
|:------------:|:-----------:|:--------:|:---------------:|
| <img src="https://via.placeholder.com/100?text=Healthy" width="100"> | <img src="https://via.placeholder.com/100?text=Blotch" width="100"> | <img src="https://via.placeholder.com/100?text=Dry" width="100"> | <img src="https://via.placeholder.com/100?text=Rhizome" width="100"> |

*(Placeholder – actual images from dataset)*

### 📥 Download
Dataset DOI: [10.17632/g46dvrcvwn.2](https://data.mendeley.com/datasets/g46dvrcvwn/2)  
After download, place the ZIP file in `data/raw/` and run the extraction script (see [Usage](#-usage-guide)).

---

## 🧠 Methods & Architectures

### 1. BYOL (Bootstrap Your Own Latent) with ViT‑B/16

**Architecture**  
- **Backbone**: ViT‑B/16 (`vit_base_patch16_224`) – pretrained on ImageNet‑21k  
- **Projector**: 3‑layer MLP (`768→4096→4096→256`) with BatchNorm  
- **Predictor**: 2‑layer MLP (`256→4096→256`) with BatchNorm  
- **EMA target network**: momentum‑updated copy of online network (cosine schedule τ: 0.996 → 1.0)

**Loss**  
Symmetric negative cosine similarity between the predictor’s output and the target’s projection.

**Key Hyperparameters**  
| Parameter | Value |
|:----------|:-----:|
| SSL epochs | 60 |
| Batch size | 128 |
| Optimizer | AdamW (lr=3e-4, wd=1e-6) |
| Projection dim | 256 |

### 2. DINO (Self‑Distillation with No Labels) with Swin‑T

**Architecture**  
- **Backbone**: Swin‑Tiny (`swin_tiny_patch4_window7_224`) – trained from scratch (no pretrained weights)  
- **Projection head**: 3‑layer MLP (`768→2048→2048→256`) → ℓ2‑normalisation → WeightNorm Linear (`256→65536`)  
- **Teacher‑student**: same architecture, teacher updated via EMA (cosine λ: 0.996 → 1.0)  
- **Multi‑crop**: 2 global crops (224²) + 6 local crops (96²) – only student sees local crops

**Loss**  
Cross‑entropy between student’s sharpened probabilities and teacher’s centred+sharpened probabilities.

**Key Hyperparameters**  
| Parameter | Value |
|:----------|:-----:|
| SSL epochs | 60 |
| Batch size | 16 (due to multi‑crop memory) |
| Optimizer | AdamW (lr=5e-4, wd cosine 0.04→0.4) |
| Teacher centering momentum | 0.9 |
| Output dimension | 65,536 |

---

## ✨ Key Features

- **Two full SSL pipelines** – BYOL and DINO, ready to run on the turmeric dataset.
- **Unlabelled pre‑training** – 80% of data used without any labels.
- **Linear probing** – Train a linear classifier on frozen features (supports both L‑BFGS and SGD).
- **k‑NN evaluation** – Cosine similarity k‑NN on L2‑normalised features.
- **Attention visualisation** – For DINO (Swin‑T), we provide activation‑based attention maps to inspect what the model focuses on.
- **Ablation study** – EMA momentum ablation (τ = 0.99, 0.996, 0.999) to analyse sensitivity.
- **Reproducible** – Fixed random seeds, all hyperparameters centralised in config cells.

---

## 📁 Project Structure

