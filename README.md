# Advanced Semantic Segmentation Framework

This repository implements a professional-grade semantic segmentation pipeline utilizing a modified DeepLabV3+ architecture with a MobileNetV3 backbone. A core innovation of this project is the integration of the Segment Anything Model (SAM) to generate high-fidelity training masks from bounding box annotations, expanding the usable dataset to approximately 17,000 images and significantly enhancing model robustness.

---

# 🚀 Key Features

**SAM-Powered Pipeline:**  
Automated generation of pixel-level segmentation masks using Meta's Segment Anything Model (SAM) from PASCAL VOC bounding box annotations.

**Optimized Architecture:**  
DeepLabV3+ modified with Efficient Channel Attention (ECA) and Depthwise Separable Atrous Convolutions.

**Robust Data Augmentation:**  
Integration of advanced techniques including CCMBA (Class-Centric Motion-Blur Augmentation) and "Painting-by-Numbers" for shape-bias reinforcement.

**Efficient Execution:**  
Lightweight MobileNetV3 backbone with Group Normalization (GN) for stable training on edge-ready hardware.

**Hybrid Loss Regime:**  
Dual-optimization using a combination of Weighted Cross-Entropy and Multi-Class Dice Loss.

---

# 🏗️ Methodology and Project Pipeline

The project follows a multi-stage workflow designed to maximize the utility of the PASCAL VOC 2012 dataset and its 17,000+ available images.

---

## 1. Segmentation Map Generation (SAM Integration)

Since many images in the expanded dataset lack pixel-level ground truth, we utilize a SAM ViT-H based script (`generate_sam_mask1.py`) to automate mask creation.

**Prompting:**  
The system parses VOC XML annotations to extract bounding boxes.

**Inference:**  
Bounding boxes are fed as sparse prompts to the SAM predictor.

**Output:**  
High-quality binary masks are generated and mapped to the 21 VOC semantic classes (20 objects + 1 background).

---

## 2. Dataset Construction

The framework constructs a **CombinedVOC directory** containing:

- **17,000+ Images:** A mix of original RGB images and corresponding masks.
- **Split Logic:** A custom 80% Training / 20% Testing split ensuring unbiased evaluation.
- **Mask Hierarchy:** The dataloader prioritizes original VOC ground truth; if missing, it falls back to the SAM-generated segmentation maps.

---

## 3. Model Architecture (DeepLabV3+ Mobile)

The model is an optimized variant of DeepLabV3+ designed for high performance with low computational overhead:

**Encoder:**  
MobileNetV3 Small, pre-trained on COCO, featuring a 1×1 convolution projection layer to reduce channels from  
`1280 → 256` before ASPP.

**ASPP Optimization:**  
Standard atrous convolutions are replaced with Depthwise Separable Atrous Convolutions, drastically reducing FLOPs while maintaining the receptive field.

**ECA Attention:**  
An Efficient Channel Attention (ECA) module is integrated after the high-level projection to perform local cross-channel interaction without dimensionality reduction.

**Decoder:**  
A high-resolution skip-connection decoder fuses low-level features (from MobileNetV3 bottleneck block 2) with high-level ASPP output for precise boundary localization.

---

# 🛠️ Advanced Training Techniques

To ensure the model generalizes to real-world image corruptions, we utilize several state-of-the-art augmentation strategies:

| Technique | Description | Benefit |
|----------|-------------|---------|
| CCMBA | Class-Centric Motion-Blur Augmentation | Robustness against space-variant motion blur |
| Painting-by-Numbers | Alpha-blending images with fixed-color class labels | Forces the model to rely on shape cues rather than texture |
| NoisyMix | Leveraging noisy augmentations in input and feature space | Improves stability against input perturbations |
| Adaptive Weighting | Dynamic class weighting based on pixel counts | Mitigates VOC class imbalance (e.g., Sheep vs. Person) |

---

# 📊 Loss Function & Metrics

The framework employs a Hybrid Loss to align training directly with the competition ranking metric (Dice Similarity Coefficient):

```
L_total = L_CrossEntropy + (1 - DSC)
```

**Cross-Entropy:** Handles pixel-level classification.  
**Multi-Class Dice Loss:** Optimizes region overlap, critical for macro-average performance across imbalanced classes.

---

# 🚀 Setup and Usage

## Installation

```bash
git clone https: https://github.com/Eshani-R-Sawant/PASCAL-VOC-Efficient-Segmentation-Challenge/

```

---



## Training


```bash
python train.py 
```

---

## Evaluation

The model can be evaluated using our internal script or the Hugging Face Evaluator Space:

**Internal**
```bash
python3 inference.py --in_dir classroom_test/JPEGImages --out_dir output --gt_dir classroom_test/SegmentationClass 
```

**External**

```
https://huggingface.co/spaces/priyadip/voc-seg-evaluator
1)Here the user have to add the checkpoints file.
2)model.py
3)Test_images on which the segmentation need to be genrated
```

---

# 📈 Evaluation Results

| Metric | Score |
|------|------|
| Mean Dice (DSC) | ~76.2% |
| Model FLOPs | 0.241 |
| Inference Speed | Real-time on CPU/Mobile GPU |

**Note:**  
Detailed logs and class-wise Dice scores are generated in the `outputs/` folder after each validation epoch.
