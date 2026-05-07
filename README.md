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

