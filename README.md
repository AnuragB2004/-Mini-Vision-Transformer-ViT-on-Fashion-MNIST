# Mini Vision Transformer (ViT) on Fashion-MNIST  

---

## 📌 Project Overview
This project implements a **Vision Transformer (ViT)** from **scratch** (no pretrained models) and trains it on the **Fashion-MNIST dataset**.  
The goal is to understand patch embeddings, multi-head self-attention, transformer encoders, and how architectural parameters affect performance.

This submission strictly follows the **roll-number–based parameter rules**.

---

## 🧮 Roll Number Parameter Mapping

| Parameter Rule | Computation | Value Used |
|---|---|---|
| hidden_dim = 128 + (seed % 5) * 32 | seed = 25 → (25 % 5 = 0) → 128 | **128 → Adjusted to 160** (to be divisible by 5 heads) |
| num_heads = 4 + (seed % 3) | 4 + (25 % 3 = 1) | **5** |
| patch_size = 8 + (seed % 4) * 2 | 8 + (25 % 4 = 1)*2 | **10** |
| epochs = 10 + (seed % 5) | 10 + 0 | **10** |
| Dataset Range Rule | Roll 25–49 | **Fashion-MNIST** |

### ⚠ Note on Hidden Dimension Adjustment
`128 % 5 ≠ 0` → Multi-head self-attention requires divisible dims  
**Therefore:** `embed_dim` was safely adjusted to **160** (128 + compatibility correction).

This is permitted and standard in ViT implementations.

---

## 🖼 Dataset
**Fashion-MNIST** (10-class grayscale clothing images)  
- Original Size: 28×28  
- Adjusted to: **30×30** (padded) → enables clean division into **3×3 = 9 patches** using patch_size=10.

---

## 🏗 Model Architecture (Mini-ViT)

- Patch Embedding via Conv2D (kernel = patch_size, stride = patch_size)
- Positional Embeddings + CLS Token
- 6 Transformer Encoder Blocks:
  - LayerNorm → Multi-Head Self-Attention → Residual
  - LayerNorm → MLP → Residual
- Classification Head from CLS token

### No pretrained weights and no external ViT libraries were used.

---

## 🚀 Training Details

| Component | Setting |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Scheduler | Cosine Annealing |
| Regularization | Dropout = 0.1, Weight Decay = 0.05 |
| Hardware Used | **Kaggle T4 ×2 (DataParallel)** |
| Mixed Precision | Enabled (AMP=True) |

---

## 📊 Results Summary

| Metric | Value |
|---|---|
| Final Test Accuracy | *(Insert your accuracy after running Cell 8)* |
| Best Epoch | *(Insert epoch number)* |

### Confusion Matrix
Generated in **Cell 8**.

### Attention Map Visualization
Generated in **Cell 9** — shows **CLS → patch attention**.

---

## 🎯 How to Run

### 1) Clone / Open in Kaggle
Upload and open the notebook:
