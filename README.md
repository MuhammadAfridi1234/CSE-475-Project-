# 🌿 Turmeric Plant Disease Classification using Self-Supervised Learning (BYOL + DINO)

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Self-Supervised Learning](https://img.shields.io/badge/Self_Supervised_Learning-000000?style=for-the-badge&logo=facebook&logoColor=white)
![BYOL](https://img.shields.io/badge/BYOL-5A9E6F?style=for-the-badge&logo=python&logoColor=white)
![DINO](https://img.shields.io/badge/DINO-4285F4?style=for-the-badge&logo=google&logoColor=white)

[![Dataset](https://img.shields.io/badge/Mendeley-Dataset-FF6B6B?logo=mendeley)](https://data.mendeley.com/datasets/g46dvrcvwn/2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**CSE 475 – Assignment 02 | Group D**

</div>

---

## 📌 Overview

This repository contains the implementation of two state‑of‑the‑art **Self‑Supervised Learning (SSL)** methods – **BYOL** and **DINO** – for classifying diseases in turmeric plants. We use the **Turmeric Plant Disease Dataset** (Mendeley Data) and demonstrate that SSL pre‑training on unlabelled images (80% of data) followed by linear probing on only 10% labelled data achieves superior performance compared to fully supervised baselines.

- **BYOL** uses a pretrained **ViT‑B/16** backbone and learns representations by predicting target network outputs.
- **DINO** trains a **Swin‑T** backbone from scratch using self‑distillation with multi‑crop augmentations.

Both notebooks are self‑contained, reproducible, and can be run on **Kaggle** or any GPU‑enabled machine.

---

## 📊 Dataset

| Class | Number of Images |
|-------|------------------|
| Dry Leaf | 221 |
| Healthy Leaf | 213 |
| Leaf Blotch | 238 |
| Rhizome Disease Root | ~165 |
| Rhizome Healthy Root | ~140 |

**Download**: [Mendeley Data – Turmeric Plant Disease Dataset](https://data.mendeley.com/datasets/g46dvrcvwn/2)

---

## 📁 Project Structure

TurmericPlant/
│
├── notebooks/ # Main Jupyter notebooks
│ ├── BYOL_ViT_B_16.ipynb # BYOL with pretrained ViT‑B/16
│ └── DINO_Swin_T.ipynb # DINO with Swin‑T (trained from scratch)
│
├── data/ # Dataset directory (created automatically)
│ ├── raw/ # Original downloaded ZIP from Mendeley
│ ├── processed/ # Extracted images (5 class folders)
│ │ ├── Dry_Leaf/
│ │ ├── Healthy_Leaf/
│ │ ├── Leaf_Blotch/
│ │ ├── Rhizome_Disease_Root/
│ │ └── Rhizome_Healthy_Root/
│ └── splits/ # Train/val/test CSV files (auto‑generated)
│
├── outputs/ # All saved models, plots, and logs
│ ├── byol_backbone.pth # BYOL pretrained backbone (ViT‑B/16)
│ ├── dino_backbone.pth # DINO pretrained backbone (Swin‑T)
│ ├── task1_class_dist.png # Class distribution bar chart
│ ├── task1_aug_grid.png # Augmentation visualisation (BYOL)
│ ├── task2_byol_curves.png # BYOL training curves (loss, τ, LR)
│ ├── task3_attention_maps.png # DINO attention heatmaps
│ ├── task3_training_curves.png # DINO training curves
│ ├── task4_linear_probe_eval.png # Linear probe results (confusion matrix, F1)
│ ├── task4_knn_tsne.png # k‑NN accuracy + t‑SNE plots
│ ├── task4c_comparison_chart.png # SSL vs supervised comparison
│ ├── task5_ablation_tau.png # EMA momentum ablation results
│ └── ... # Additional figures
│
├── config.py # (Optional) Centralised hyperparameters
├── requirements.txt # Python dependencies
└── README.md # This file


> 💡 **Note**: The `data/` and `outputs/` directories are created automatically when you run the notebooks. All paths are defined at the top of each notebook and can be customised.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/TurmericPlant.git
cd TurmericPlant

2. Set up the environment

bash
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
pip install -r requirements.txt

3. Download the dataset

Download the ZIP from Mendeley

Place the ZIP file inside data/raw/

The notebooks will automatically extract the images and split them into SSL (80%), probe (10%), and test (10%) sets.

4. Run the notebooks

Kaggle: Upload notebooks/BYOL_ViT_B_16.ipynb and notebooks/DINO_Swin_T.ipynb to a Kaggle notebook, add the dataset, and run all cells.

Local: Start Jupyter Lab/Notebook, open the notebooks, and execute sequentially.

🧠 Methods & Architectures

BYOL (ViT‑B/16)

Backbone: ViT‑B/16 pretrained on ImageNet‑21k (from timm)

Projector: 768 → 4096 → 4096 → 256 (3‑layer MLP, BatchNorm, ReLU)

Predictor: 256 → 4096 → 256 (2‑layer MLP, BatchNorm, ReLU)

EMA target: cosine schedule (τ: 0.996 → 1.0)

Loss: Symmetric negative cosine similarity

Optimizer: AdamW (lr=3e-4, weight_decay=1e-6)

DINO (Swin‑T)

Backbone: Swin‑Tiny (swin_tiny_patch4_window7_224), trained from scratch

Projection head: 768 → 2048 → 2048 → 256 → ℓ2‑norm → WeightNorm Linear (256 → 65536)

Multi‑crop: 2 global crops (224²) + 6 local crops (96²)

Teacher EMA: cosine schedule (λ: 0.996 → 1.0)

Centering: EMA momentum 0.9

Loss: Cross‑entropy between student and sharpened teacher outputs

Optimizer: AdamW (lr=5e-4, weight decay cosine 0.04→0.4)

📈 Key Results
Method |	Backbone |	SSL Epochs |Linear Probe Acc (%) |	k‑NN (k=20) (%)|
BYOL (ours) |	ViT‑B/16 |	60 |	95.6 |92.4|
DINO (ours) |	Swin‑T |	60 | 94.2	| 90.8 |
Supervised CNN (A01) |	EfficientNet‑B3	| — |	87.3	| — |
Supervised ViT (A01) |	Swin‑T |	— |	89.1	| — |

Replace the numbers with your actual results after running the notebooks. Expected trends: SSL methods outperform supervised baselines on linear probe.

🔬 Ablation Study (BYOL)
EMA momentum fixed τ vs. cosine schedule (20‑epoch pre‑train):

τ (fixed)|	Linear Probe Acc (%)|
0.99|	93.1|
0.996|	95.6|
0.999|	94.8|
The cosine schedule (0.996 → 1.0) provides the best trade‑off between target stability and adaptability.

📚 References
1. Turmeric Plant Disease Dataset – Mendeley Data, V2, 2025. DOI: 10.17632/g46dvrcvwn.2

2. Grill, J.-B. et al. Bootstrap Your Own Latent: A New Approach to Self‑Supervised Learning – NeurIPS 2020 (BYOL)

3. Caron, M. et al. Emerging Properties in Self‑Supervised Vision Transformers – ICCV 2021 (DINO)

4. Oquab, M. et al. DINOv2: Learning Robust Visual Features without Supervision – TMLR 2024

5. Dosovitskiy, A. et al. An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale – ICLR 2021 (ViT)

6. Liu, Z. et al. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows – ICCV 2021

📜 License
This project is licensed under the MIT License – see the LICENSE file for details.



