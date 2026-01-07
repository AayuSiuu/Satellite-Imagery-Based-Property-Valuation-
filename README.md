# Satellite Imagery Based Property Valuation 🛰️🏠

This project combines satellite imagery features with tabular housing data to create a multimodal regression system that predicts residential property prices.
This method incorporates visual neighborhood context (green cover, roads, surrounding structures) extracted from satellite images, in contrast to traditional price prediction models that solely rely on numerical attributes.---

---

## 📌 Problem Statement

Real estate valuation depends not only on property size and amenities, but also on **neighborhood characteristics and curb appeal**.  
This project aims to improve valuation accuracy by building a **multimodal regression pipeline** that integrates:

- Structured tabular features (e.g. size, condition, location)
- Visual features extracted from satellite imagery using a CNN

---

## 🧠 Approach Overview

The pipeline follows these steps:

1. **Tabular Data Processing**
   - Housing attributes such as bedrooms, bathrooms, living area, condition, grade, etc.
   - Features are scaled using `StandardScaler`.

2. **Satellite Image Processing**
   - Satellite images are fetched using latitude and longitude.
   - A pretrained CNN is used to extract high-dimensional image embeddings that capture neighborhood context.

3. **Multimodal Fusion Model**
   - Separate neural network branches process:
     - Tabular features
     - Image embeddings
   - Features are fused and passed through a regression head.
   - The model is trained to predict **log-scaled property prices**.

4. **Inference**
   - The trained PyTorch fusion model (`fusion_model.pth`) is used to generate predictions on unseen data.
   - Predictions are converted back to real prices using exponential transformation.

---

## 🧱 Model Architecture

- **Tabular Branch**
  - Linear → BatchNorm → ReLU → Dropout

- **Image Branch**
  - Linear → BatchNorm → ReLU → Dropout

- **Fusion Head**
  - Concatenation → Fully connected layers → Price prediction

This design allows the model to learn complementary information from both numeric and visual inputs.

---

## 🔍 Explainability (Grad-CAM)

To interpret the visual component of the model, **Grad-CAM** is applied to the CNN feature extractor.  
The resulting heatmaps highlight regions such as:

- Tree cover and greenery
- Road connectivity
- Surrounding building density

This demonstrates that the model captures meaningful neighborhood features from satellite imagery rather than irrelevant pixels.

An example Grad-CAM visualization is included in the repository (`output.png`).

---

## ▶️ How to Run Inference

1. Ensure dependencies are installed:
   ```bash
   pip install numpy pandas torch scikit-learn

2. Run the following command to generate predictions using the trained multimodal model:
   ```bash
   python inference.py

3. Output is 24115004_final.csv

---

## 📊 Output

- `24115004_final.csv` containing predicted property prices
- Predictions are generated **only for properties with both tabular data and satellite imagery**
- The model uses **true multimodal fusion** during inference

---

## 🚀 Key Highlights

- True **multimodal learning** (tabular + visual data)
- **PyTorch-based** fusion architecture
- **CNN-derived** satellite image embeddings
- Explainability using **Grad-CAM**
- End-to-end **inference pipeline**

---

## Conclusion

By capturing neighborhood context and visual cues that are otherwise challenging to quantify, this project shows how combining satellite imagery with conventional housing data can improve property valuation models.

---

## Author:
 AAYUSHI SINHA (IIT-Roorkee)
 ## 📄 Project Report

The detailed project report is available here:  
👉 [Download the full report (PDF)](24115004_report.pdf)
