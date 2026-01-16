AnchalPrasadTeraFac
Deep Learning Project: Progressive Optimization (Level 1 → Level 4)
Links:
====================================================================================
📎 Google Colab Notebooks

Level 1:
https://colab.research.google.com/drive/1R_sOwR8hA3WECqf49mtT5EdrwAuvvara?usp=sharing

Level 2:
https://colab.research.google.com/drive/1SENoeVXWDfx4QQlKEN7ZfiC7CL1J7DFg?usp=sharing

Level 3:
https://colab.research.google.com/drive/1kXEABADtKLfKFl6Ia5aXopJbLPP9ci9Z?usp=sharing

Level 4:
https://colab.research.google.com/drive/1hKs7I_pQYEbkuRaX-fiTfUTL4a2WSx6q?usp=sharing

📂 Drive Link:
https://drive.google.com/drive/folders/1kEm4Zo50ODPOrMyT52Feg1Pi5qRkrLOc?usp=sharing

===========================================================================================


Model: ResNet50 (ImageNet Pretrained)
Framework: PyTorch
Dataset: CIFAR-10

📌 Project Overview

This project presents a progressive, multi-level deep learning pipeline for image classification on the CIFAR-10 dataset.
Each level incrementally introduces advanced training strategies to improve accuracy, generalization, robustness, and training stability.

The objective is to systematically analyze the impact of modern deep learning techniques, including:

Transfer Learning

Advanced Data Augmentation

Regularization Techniques

Test-Time Augmentation (TTA)

Strong Optimization & Scheduling Strategies

This structured approach mirrors real-world research and industry workflows.

🧠 Dataset Details

Dataset: CIFAR-10

Number of Classes: 10

Data Split:

Training: 45,000

Validation: 5,000

Test: 5,000

🏗️ Model Architecture

Backbone: ResNet50

Pretraining: ImageNet

Classifier Head: Modified for 10 classes

Total Parameters: ~23.5M

📊 Progressive Training Levels
🔹 Level 1 — Baseline Transfer Learning

Objective: Establish a strong baseline using transfer learning.

Techniques Used

ImageNet pretrained ResNet50

Basic augmentations (Random Crop, Horizontal Flip)

Frozen backbone → train classifier head

Full model fine-tuning

CrossEntropy Loss

AdamW Optimizer

Results

Test Accuracy: 96.34%

📁 Notebook: notebooks/Level1.ipynb
📦 Model: checkpoints/level1_best.pth

🔹 Level 2 — Advanced Augmentation & Regularization

Objective: Improve generalization and robustness.

New Techniques Introduced

RandAugment

MixUp & CutMix

Label Smoothing (ε = 0.1)

Random Erasing

Cosine Learning Rate Scheduler

Test-Time Augmentation (TTA)

Results

Metric	Accuracy
Level 1 Baseline	96.34%
Level 2 (Standard)	96.56%
Level 2 (with TTA)	96.88%

📈 Overall Improvement: +0.54%

📁 Notebook: notebooks/Level2.ipynb
📦 Model: checkpoints/level2_best.pth

📌 Per-Class Accuracy (Level 2)
Class	Accuracy
airplane	97.75%
automobile	98.02%
bird	97.27%
cat	91.15%
deer	97.04%
dog	93.85%
frog	98.37%
horse	97.17%
ship	98.21%
truck	96.88%
🔹 Level 3 — Optimization & Training Stability

Objective: Improve convergence behavior and training stability.

Techniques Applied

Learning rate warmup

Progressive layer unfreezing

Exponential Moving Average (EMA)

Stronger regularization control

Extended fine-tuning with lower learning rate

📁 Notebook: notebooks/Level3.ipynb
📦 Model: checkpoints/level3_best.pth

🔹 Level 4 — Research-Level Enhancements

Objective: Apply advanced research-oriented techniques and analysis.

Advanced Methods

Stochastic Weight Averaging (SWA)

Advanced Test-Time Augmentation strategies

Hyperparameter ablation studies

Model calibration (Expected Calibration Error – ECE)

Confusion matrix & error analysis

📁 Notebook: notebooks/Level4.ipynb
📦 Model: checkpoints/level4_best.pth

📐 Evaluation Metrics

Top-1 Accuracy

Per-Class Accuracy

Validation Accuracy

Test Accuracy

Test-Time Augmented Accuracy

Training & Validation Loss Curves


📁 Project Structure
├── notebooks/
│   ├── Level1.ipynb
│   ├── Level2.ipynb
│   ├── Level3.ipynb
│   └── Level4.ipynb
│
├── checkpoints/
│
├── report/
│
├── results/
│
├── requirements.txt
│
└── README.md

📎 Google Colab Notebooks

Level 1:
https://colab.research.google.com/drive/1R_sOwR8hA3WECqf49mtT5EdrwAuvvara?usp=sharing

Level 2:
https://colab.research.google.com/drive/1SENoeVXWDfx4QQlKEN7ZfiC7CL1J7DFg?usp=sharing

Level 3:
https://colab.research.google.com/drive/1kXEABADtKLfKFl6Ia5aXopJbLPP9ci9Z?usp=sharing

Level 4:
https://colab.research.google.com/drive/1hKs7I_pQYEbkuRaX-fiTfUTL4a2WSx6q?usp=sharing

💾 Model Checkpoints (.pth)

Each .pth file includes:

Trained model weights

Optimizer state

Best validation checkpoint

This enables full reproducibility and inference reuse.

✅ Project Status
Level	Status
Level 1	✅ Completed
Level 2	✅ Completed
Level 3	✅ Completed
Level 4	✅ Completed
🏁 Final Notes

This project demonstrates:

Strong understanding of deep learning pipelines

Practical application of modern regularization techniques

Ability to conduct ablation and optimization studies

Industry-ready documentation and reproducibility

Research-oriented mindset with structured experimentation

📂 Drive Link:
https://drive.google.com/drive/folders/1kEm4Zo50ODPOrMyT52Feg1Pi5qRkrLOc?usp=sharing
