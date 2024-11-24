# Multi-Model Image Classification Using Transfer Learning with Attention Mechanism

## Project Overview

This project implements an advanced image classification pipeline by leveraging multiple pre-trained deep learning models—**VGG16, VGG19, MobileNet, and Xception**—combined with an **Attention Mechanism** to enhance feature extraction and classification accuracy. The pipeline is designed for classifying datasets, such as **medical images**, into four distinct categories.

---

## Key Features

### 1. **Data Preprocessing and Augmentation**
- Dynamic generation of image paths and labels stored in a DataFrame.
- **Class imbalance** handled using **Random Oversampling** for improved model training.
- Dataset split into **training, validation, and testing subsets** with stratified sampling for balanced distributions.
- Images normalized (`rescale=1./255`) for optimized model performance.

### 2. **Exploratory Data Analysis (EDA)**
- Visualizes category distributions with **count plots** and **pie charts**.
- Displays a grid of sample images from each category for dataset insights.

### 3. **Transfer Learning**
- Integrates pre-trained models (**VGG16, VGG19, MobileNet, Xception**) as **feature extractors**.
- Pre-trained layers are frozen to preserve learned features, while classification layers are fine-tuned.

### 4. **Attention Mechanism**
- Incorporates a **Multi-Head Attention Layer** to focus on significant features in the extracted feature maps.
- Improves model interpretability and boosts classification performance.

### 5. **Regularization Techniques**
- Mitigates overfitting using:
  - **Gaussian Noise**
  - **Batch Normalization**
  - **Dropout Layers**

### 6. **Model Training and Evaluation**
- Models compiled using the **Adam optimizer** with a custom learning rate.
- Tracks **accuracy** and **loss** metrics for training and validation sets.
- Employs **Early Stopping** to prevent overfitting and secure optimal weights.
- Training progress visualized with accuracy and loss plots.

### 7. **Performance Metrics**
- Comprehensive classification report with:
  - **Precision**
  - **Recall**
  - **F1-score**
  - **Support** for each class.
- Confusion matrix heatmap for detailed evaluation of model predictions.

---

## Applications
- **Medical Image Classification**: Tumor detection, disease diagnosis.
- **Object Detection and Categorization**: Across various domains.
- **Learning Case Study**: Showcasing **Transfer Learning** and **Attention Mechanisms** in practice.

---

## Technologies and Tools Used
- **Programming**: Python
- **Deep Learning**: TensorFlow/Keras
- **Data Manipulation**: Pandas, NumPy
- **Data Visualization**: Seaborn, Matplotlib
- **Image Processing**: OpenCV, PIL
- **Pre-trained Models**: VGG16, VGG19, MobileNet, Xception
- **Other Tools**:
  - **scikit-learn**: Stratified splitting, classification metrics
  - **imblearn**: Oversampling for imbalanced datasets

---

## Visual Highlights
- **Category Distributions**: EDA visualizations for class insights.
- **Attention Visualizations**: Insights into focused features via the attention mechanism.
- **Model Performance**: Plots for training/validation accuracy and loss.

---

## Summary

This project demonstrates a robust pipeline for **multi-model image classification** using cutting-edge deep learning techniques, making it adaptable for diverse datasets and classification tasks. The inclusion of **attention mechanisms** and **transfer learning** ensures enhanced performance and generalizability. 

--- 

Feel free to explore the code and try it on your datasets! 🚀
