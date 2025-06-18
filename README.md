# SRED: Secure and Robust Emotion Detection for ADAS

This repository contains the implementation of **SRED**, a Secure and Robust Emotion Detection framework designed for Advanced Driver Assistance Systems (ADAS). The system detects driver emotions from facial images while defending against adversarial attacks.

## 🔍 Project Overview

Facial emotion recognition is critical in monitoring driver states in ADAS. However, deep learning models are vulnerable to adversarial examples that can mislead emotion prediction. This project implements a hybrid defense strategy combining **attention masking** and **adversarial training** to improve robustness against such attacks.

## 🧰 Features

- Face detection using **MTCNN**
- Emotion classification using:
  - ResNet18
  - MobileNetV2
  - EfficientNetB0
- Attention masking for facial region enhancement
- Adversarial training using **FGSM**
- Robustness evaluation under **FGSM**, **BIM**, and **PGD** attacks
- Saliency map visualization
- Evaluation on **KMU-FED**, **KDEF**, and **FER2013** datasets

## 📁 Project Structure

emotion-detection/
│
├── models/ # Model architectures
├── data/ # Preprocessing scripts
├── attacks/ # FGSM, BIM, PGD attack implementations
├── defense/ # Adversarial training & attention masking
├── utils/ # Helper functions
├── notebooks/ # Jupyter notebooks for experiments
├── results/ # Plots, confusion matrices, saliency maps
└── main.py # Main training & evaluation script

markdown
Copy
Edit

## ⚙️ Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy, matplotlib, scikit-learn
- facenet-pytorch (for MTCNN)

Install with:

```bash
pip install -r requirements.txt
