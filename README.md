# Deep-Learning-tugas
# Oxford-IIIT Pet Breed Classification & Model Explainability

This project implements a complete, end-to-end computer vision pipeline using PyTorch to classify pet breeds from the Oxford-IIIT Pet Dataset. It explores multiple transfer learning strategies, progressive fine-tuning, alternative architectures (ResNet-18 vs. EfficientNet-B0), data augmentation impacts, error/failure analyses, and model explainability using Grad-CAM visualizations.

## Project Overview

- **Target Dataset**: Oxford-IIIT Pet Dataset (Subset of 10 selected classes including cats like Abyssinian, Bengal, Birman, Bombay, British Shorthair, and dogs like beagle, boxer, chihuahua, pug, shiba_inu).
- **Data Pipeline**: Stratified splitting into Train (70%), Validation (15%), and Test (15%) sets to handle class representation proportionally.
- **Architectures**: ResNet-18 and EfficientNet-B0.
- **Learning Strategies Studied**:
  1. **B1**: Baseline trained from scratch.
  2. **B2**: Feature extraction (classifier head only).
  3. **B3**: Progressive Fine-Tuning (unfreezing late feature extraction layers + classifier).
  4. **B4_NoAug vs. B4_WithAug**: Checking the influence of explicit color, rotation, and crop augmentations.
  5. **B5**: EfficientNet-B0 fine-tuned on late blocks.

## Key Highlights & Results

- **Progressive Fine-Tuning (B3)** significantly outperforms training from scratch (B1) and basic feature extraction (B2).
- **Grad-CAM Visualizations** show that the fine-tuned model focuses much more sharply on discriminative facial and body patterns rather than background context compared to its pre-trained but untuned counterpart.
- **Failure Analysis** indicates that the model's rare mistakes happen primarily on highly occluded, non-standard poses, or when multiple objects/people are prominent in the frame.

## Installation & Setup

1. **Clone the repository** (or save the source files locally).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
