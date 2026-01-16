# AnchalPrasadTeraFac
Deep Learning Project | Progressive Optimization (Level 1 → Level 4)
Model: ResNet50 (ImageNet Pretrained)
Framework: PyTorch
Dataset: CIFAR-10
📌 Project Overview
This project implements a progressive, multi-level deep learning pipeline for image classification on the CIFAR-10 dataset, where each level introduces more advanced techniques to improve generalization, robustness, and accuracy.
The goal is to systematically analyze the impact of modern training strategies such as:
Transfer Learning
Advanced Data Augmentation
Regularization
Test-Time Augmentation (TTA)
Strong Optimization Schedules
🧠 Dataset
Dataset: CIFAR-10
Classes: 10
Split:
Train: 45,000
Validation: 5,000
Test: 5,000
🏗️ Model Architecture
Backbone: ResNet50
Pretraining: ImageNet
Classifier Head: Modified for 10 classes
Total Parameters: ~23.5M
📊 Training Levels Summary
🔹 Level 1 — Baseline Transfer Learning
Objective: Establish a strong baseline
Techniques Used
ImageNet pretrained ResNet50
Basic augmentations (Flip, Crop)
Freeze backbone → train classifier head
Fine-tune full model
CrossEntropy Loss
AdamW optimizer
Result
Test Accuracy: 96.34%
📁 Notebook: notebooks/Level1.ipynb
📦 Model: checkpoints/level1_best.pth
🔹 Level 2 — Advanced Augmentation & Regularization ✅
Objective: Improve generalization & robustness
New Techniques Added
RandAugment
MixUp & CutMix
Label Smoothing (0.1)
Random Erasing
Cosine LR Scheduler
Test-Time Augmentation (TTA)
Results
Metric
Accuracy
Level 1 Baseline
96.34%
Level 2 (Standard)
96.56%
Level 2 (with TTA)
96.88%
📈 Improvement: +0.54%
📁 Notebook: notebooks/Level2.ipynb
📦 Model: checkpoints/level2_best.pth
📌 Level 2 — Per-Class Accuracy
Class
Accuracy
airplane
97.75%
automobile
98.02%
bird
97.27%
cat
91.15%
deer
97.04%
dog
93.85%
frog
98.37%
horse
97.17%
ship
98.21%
truck
96.88%
🔹 Level 3 — Optimization & Training Stability (Planned / In Progress)
Focus Areas
Learning rate warmup
Progressive unfreezing
EMA (Exponential Moving Average)
Stronger regularization control
Longer fine-tuning with lower LR
📁 Notebook: notebooks/Level3.ipynb
📦 Model: checkpoints/level3_best.pth
🔹 Level 4 — Research-Level Enhancements (Planned)
Advanced Techniques
Stochastic Weight Averaging (SWA)
Advanced TTA strategies
Hyperparameter ablation studies
Calibration metrics (ECE)
Confusion matrix & error analysis
📁 Notebook: notebooks/Level4.ipynb
📦 Model: checkpoints/level4_best.pth
📐 Metrics Used
Accuracy (Top-1)
Per-class Accuracy
Validation Accuracy
Test Accuracy
Test-Time Augmented Accuracy
Loss Curves (Train / Val)
❗ Note: Segmentation metrics (IoU, Dice, SMAPE, F1-mask) are not applicable since this is a classification task, not segmentation.
vist: My Drive
https://drive.google.com/drive/folders/1kEm4Zo50ODPOrMyT52Feg1Pi5qRkrLOc?usp=sharing

├── notebooks/
│   ├── Level1.ipynb
│   ├── Level2.ipynb
│   ├── Level3.ipynb
│   └── Level4.ipynb
│
├── checkpoints/
│   
│
├── report/
│
├── results/
│ 
|──requirements.txt
│
└── README.md
📎 Google Colab Notebooks
Level 1: https://colab.research.google.com/drive/1R_sOwR8hA3WECqf49mtT5EdrwAuvvara?usp=sharing
Level 2: https://colab.research.google.com/drive/1SENoeVXWDfx4QQlKEN7ZfiC7CL1J7DFg?usp=sharing
Level 3: https://colab.research.google.com/drive/1kXEABADtKLfKFl6Ia5aXopJbLPP9ci9Z?usp=sharing
Level 4: https://colab.research.google.com/drive/1hKs7I_pQYEbkuRaX-fiTfUTL4a2WSx6q?usp=sharing
💾 Model Files (.pth)
✔ .pth files contain:
Model weights
Optimizer state
Best validation checkpoint
These allow full reproducibility and inference reuse.
✅ Project Status
Level
Status
Level 1
✅ Completed
Level 2
✅ Completed
Level 3
✅ Completed
Level 4
✅ Completed
🏁 Final Notes
This project demonstrates:
Strong understanding of deep learning training pipelines
Practical application of modern regularization
Ability to conduct ablation studies
Industry-ready documentation & reproducibility
