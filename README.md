# ECSE 415 Course Project: Dog vs. Cat Classification & Localization

## 1. Executive Summary
This project implements a complete computer vision pipeline for classifying pet images (Dogs vs. Cats) and localizing dogs within complex scenes (Stanford Dogs Dataset). By leveraging an ensemble of modern deep learning architectures (EfficientNetV2-M and ConvNeXt-Small), we achieved a classification accuracy of **99.65%** on our internal test split and a mean IoU of **0.438** for object localization.

## 2. Methodology

### 2.1 Part 1: Classification Benchmark
We explored three distinct approaches for the classification of 25,000 labeled images:
- **Feature-Based (Option A):** Extraction of handcrafted features (HOG, SIFT) combined with SVM/Random Forest.
- **Dimensionality Reduction (Option B):** PCA on raw pixels and features to reduce complexity before classification.
- **Deep Learning (Option C):** Fine-tuning of pre-trained state-of-the-art CNNs.

**Final Selection:** Our best-performing model is a weighted ensemble of **EfficientNetV2-M** and **ConvNeXt-Small**, utilizing Test-Time Augmentation (TTA) with horizontal flips to improve robustness.

### 2.2 Part 2: Detection and Localization
The ensemble model was converted into a localization pipeline using a **Patch-Based Sliding Window** approach:
1. **Multi-scale Scanning:** The image is scanned at various window sizes (scales) to detect dogs of different proportions.
2. **Confidence Thresholding:** Only patches with a "Dog" probability > 0.9 were considered.
3. **Non-Maximum Suppression (NMS):** Overlapping detections were merged based on their confidence scores and an IoU threshold of 0.3 to produce a single final bounding box.

## 3. Results & Evaluation

### 3.1 Classification Performance
Evaluated on a 20% internal validation split (4,000 images):
| Metric | Value |
| :--- | :--- |
| **Overall Accuracy** | **99.65%** |
| Precision (Dog) | ~0.997 |
| Recall (Dog) | ~0.997 |

*Note: Confusion matrices and detailed per-class metrics are available in `notebooks/Final_complete.ipynb`.*

### 3.2 Localization Performance (Stanford Dogs)
Evaluated on 20,580 images from the Stanford Dogs Dataset:
| Metric | Value |
| :--- | :--- |
| **Mean IoU** | **0.438** |
| Median IoU | 0.436 |
| **Success Rate (IoU >= 0.5)** | **37.0%** |
| Maximum IoU Achieved | 0.992 |

## 4. Discussion & Analysis

### 4.1 Successes and Failures
- **Success Cases:** The model performed exceptionally well on images with clear, centered subjects and minimal background clutter (achieving IoUs as high as 0.99).
- **Failure Modes:** 
    - **Occlusion:** When the dog is partially hidden by objects, the sliding window often misses the main features.
    - **Background Clutter:** Complex textures similar to fur (e.g., rugs, bushes) occasionally triggered false positives.
    - **Scale Variance:** Extreme close-ups or very small subjects sometimes fell outside our sliding window scale range.

### 4.2 Trade-offs
While the deep learning ensemble provided superior accuracy compared to HOG+SVM, it was significantly more computationally expensive, requiring GPU acceleration for reasonable inference times during the localization phase.

## 5. Directory Structure
- `data/`: Datasets (Kaggle Dogs vs. Cats, Stanford Dogs).
- `docs/`: Project requirements and official description.
- `models/`: Weights for ResNet50, EfficientNetV2-M, and ConvNeXt-Small.
- `notebooks/`: Implementation details (Primary: `Final_complete.ipynb`).
- `submissions/`: Kaggle Leaderboard submission files.

---
*Developed for ECSE 415: Introduction to Computer Vision, McGill University (Winter 2026).*
