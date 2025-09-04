# MedMNIST PyTorch Benchmarks

Baseline benchmarks for [MedMNIST](https://medmnist.com/) classification datasets using **PyTorch**.  
This repo currently implements **ResNet18 on PathMNIST** as a reproducible example.

---

## 📘 Overview
- **Dataset:** PathMNIST (colon pathology images, 9 classes, 107,180 images, 28x28 RGB -> resized to 64x64).  
- **Model:** ResNet18 (training from scratch, no pretrained weights).  
- **Framework:** PyTorch + Torchvision.  
- **Environment:** Designed for Google Colab with GPU (works locally too).

---

## 🚀 How to Run

### ▶️ Option 1: Google Colab
Open directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/karthik-ganeshan/medmnist-pytorch-benchmarks/blob/main/PathMNIST_ResNet18.ipynb)

The notebook will:
1. Install dependencies
2. Download PathMNIST
3. Train ResNet18
4. Evaluate and save results (`metrics.json`, confusion matrix, curves)

---

### 💻 Option 2: Local CLI

Clone repo and install requirements:
```bash
git clone https://github.com/karthik-ganeshan/medmnist-pytorch-benchmarks.git
cd medmnist-pytorch-benchmarks
pip install -r requirements.txt
