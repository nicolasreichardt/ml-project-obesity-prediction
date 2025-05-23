<img src="plots/hertie_logo.png" alt="Hertie Logo" width="90"/>

# Obesity Prediction  
## The Scale Doesn’t Lie — But Does Our Model?

**Final Report**  
Supervised Machine Learning · Spring 2025  
Hertie School · MDS

**Authors**  
Nadine Daum · Ashley Razo · Jasmin Mehnert · Nicolas Reichardt

GitHub: https://github.com/nicolasreichardt/ml-project-obesity-prediction  
Submission: 12 May 2025

<div style="page-break-after: always;"></div>

## Summary

This project applies supervised machine learning to classify individuals into obesity risk categories based on biometric and lifestyle data. We implemented and evaluated multiple models — including logistic regression, KNN, tree-based models, and a neural network — using a shared preprocessed dataset to ensure consistent and fair comparison.

Our best-performing models achieved test accuracy scores above 85%, with interpretable insights from tree-based approaches and strong generalization from the neural network.

---

## Table of Contents

- [Team](#team)
- [Project Overview](#project-overview)
- [1. Dataset Description](#1-dataset-description)
- [2. Preprocessing & Feature Engineering](#2-preprocessing--feature-engineering)
- [3. Model Overviews](#3-model-overviews)
- [4. Model Comparison](#4-model-comparison)
- [5. Reflections](#5-reflections)
- [Appendix A: Links & Files](#appendix-a-links--files)
- [Appendix B: Team Contributions](#appendix-b-team-contributions)

---

## Team
- Nadine Daum – [GitHub](https://github.com/NadineDaum) | [Email](mailto:n.daum@students.hertie-school.org)
- Jasmin Mehnert – [GitHub](https://github.com/jasmin-mehnert) | [Email](mailto:j.mehnert@students.hertie-school.org)
- Ashley Razo – [GitHub](https://github.com/ashley-razo) | [Email](mailto:a.razo@students.hertie-school.org)
- Nicolas Reichardt – [GitHub](https://github.com/nicolasreichardt) | [Email](mailto:n.reichardt@students.hertie-school.org)

## Project Overview

This project aims to classify individuals into seven obesity risk categories based on various biometric and behavioral factors. Using a labeled dataset of 2,111 individuals from Mexico, Peru, and Colombia, our models predict obesity levels ranging from *Insufficient Weight* to *Obesity Type III*.

The goal is to explore how well machine learning models can predict obesity status — and how these predictions might support future public health decisions, risk assessment tools, or individual recommendations.

GitHub repo: [nicolasreichardt/ml-project-obesity-prediction](https://github.com/nicolasreichardt/ml-project-obesity-prediction)

<div style="page-break-after: always;"></div>

## 1. Dataset Description

We used the **Obesity Levels Estimation Dataset**, which contains demographic, behavioral, and biometric data for 2,111 individuals from Mexico, Peru, and Colombia. The dataset was designed for multi-class classification and is labeled with 7 obesity categories.

### Dataset Overview:
- **Size**: 2,111 samples × 17 features + 1 target
- **Features**: mix of categorical (e.g., gender, transport_mode) and numerical (e.g., height, weight, age)
- **Target variable**: `obesity_level` with 7 classes:
  - Insufficient Weight
  - Normal Weight
  - Overweight Level I
  - Overweight Level II
  - Obesity Type I
  - Obesity Type II
  - Obesity Type III
- **ML relevance**: Multi-class classification task with imbalanced class distribution
- **Input shape for models**: ~43 features after encoding (based on one-hot transformation)

The data was collected via a cross-sectional survey and is publicly available on [Kaggle](https://www.kaggle.com/datasets/ruchikakumbhar/obesity-prediction), supported by this [research article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6710633/).

> 📝 **Jasmin – please add 1–2 sentences here about EDA findings**  
> For example: were there correlations, outliers, imbalances, or interesting clusters?

All team members used a shared train/test split to ensure model comparability.

<div style="page-break-after: always;"></div>

## 2. Preprocessing & Feature Engineering

Before modeling, the dataset required thorough cleaning and transformation. This step was led primarily by **Ashley Razo** and **Jasmin Mehnert**, with feedback and reviews from all team members.

### Preprocessing Goals
- Ensure consistent input format across models
- Improve model performance and comparability
- Reduce noise, redundancy, and scaling-related bias

### Key Steps

- **Feature selection**: Retained 17 relevant input features capturing diet, behavior, and biometrics
- **Target formatting**: Standardized and renamed the class column to `obesity_level`
- **Encoding**: Applied one-hot encoding to 13 categorical features (e.g., `gender`, `transport_mode`)
- **Scaling**: Used `StandardScaler` to normalize all numerical features (e.g., `age`, `height_m`, `weight_kg`)
- **Output dimensions**: Final input to the models included ~43 encoded features
- **Train/test split**: 80/20 split applied uniformly to ensure fair model evaluation
- **File formats**: Datasets exported as both `.csv` and `.feather` (for faster access)

> 📝 **@Ashley** – feel free to insert 1–2 sentences on your preprocessing pipeline: decisions around feature selection, encoding strategies, or challenges during cleaning  
> 📝 **@Jasmin** – you can briefly note how you supported the pipeline and flag any edge cases or quirks in the data

### Implementation
📒 Notebook: [`notebooks/preprocessing.ipynb`](notebooks/preprocessing.ipynb)  
🧾 Script: [`processed_data/data_preparation.py`](processed_data/data_preparation.py)

All models consumed the same cleaned and scaled training and testing data.

<div style="page-break-after: always;"></div>

## 3. Model Overviews

All models used the same preprocessed data for consistency.

### Logistic Regression
📒 [logistic_regression.ipynb](https://github.com/nicolasreichardt/ml-project-obesity-prediction/blob/main/notebooks/logistic_regression.ipynb)

- Simple baseline with good interpretability

### Ridge Logistic Regression
📒 [ridge_logistic_regression.ipynb](https://github.com/nicolasreichardt/ml-project-obesity-prediction/blob/main/notebooks/ridge_logistic_regression.ipynb)

- Regularized version of logistic regression

### K-Nearest Neighbors (KNN)
📒 [PCA_KNN.ipynb](https://github.com/nicolasreichardt/ml-project-obesity-prediction/blob/main/notebooks/PCA_KNN.ipynb)

- PCA helped reduce dimensionality and improved KNN performance

### Neural Network
📒 [neural_network.ipynb](https://github.com/nicolasreichardt/ml-project-obesity-prediction/blob/main/notebooks/neural_network.ipynb)

- Multi-layer architecture with ReLU and softmax
- Test accuracy: **83.9%**
- Balanced performance across all obesity categories

![Neural Network Training Curves](plots/training_curves_nn.png)

### Tree-Based Models
📒 [tree-based-models.ipynb](https://github.com/nicolasreichardt/ml-project-obesity-prediction/blob/main/notebooks/tree-based-models.ipynb)

- Random Forest & XGBoost achieved top performance (~86%)
- Screen time, calorie tracking, and water intake were key features

<div style="page-break-after: always;"></div>

## 4. Model Comparison

| Model                   | Test Accuracy | Notes                                              |
|-------------------------|----------------|----------------------------------------------------|
| Logistic Regression     | ~75%           | Simple, interpretable                              |
| Ridge Logistic Regression | ~76%         | Slight improvement with regularization             |
| KNN                     | ~77%           | Better with PCA                                    |
| Neural Network          | **83.9%**      | Strong generalization                              |
| Random Forest           | ~85–86%        | Robust, interpretable                              |
| XGBoost                 | ~86%           | Top performer with best generalization             |

### Feature Importance – Tree-Based Models
![Top Features](plots/top_features_tree_based_models.png)

### Model Comparison Overview
![Model Comparison](plots/tree_based_model_comparison.png)

### Model Comparison with Feature Exclusion
![Model Comparison (Excluded Features)](plots/tree_based_model_comparison_feature_exclusion.png)

<div style="page-break-after: always;"></div>

## 5. Reflections

- Preprocessing made a big difference across all models
- Tree-based models helped us understand what mattered most
- Neural networks were surprisingly manageable and performed well
- Sharing the same train/test split helped standardize evaluation
- We improved our understanding of ML pipelines, GitHub collaboration, and reproducibility

<div style="page-break-after: always;"></div>

## Appendix A: Links & Files

- **GitHub Repository**: [nicolasreichardt/ml-project-obesity-prediction](https://github.com/nicolasreichardt/ml-project-obesity-prediction)
- **Cleaned dataset (CSV)**: `processed_data/obesity_cleaned.csv`
- **Train/Test files**:
  - `processed_data/train_data.feather`
  - `processed_data/test_data.feather`
- **Model notebooks**: in `notebooks/`
- **Generated plots**: in `plots/`

## Appendix B: Team Contributions

- **Nadine Daum** – Neural network, Ridge/Lasso regression  
- **Ashley Razo** – Preprocessing, logistic regression  
- **Jasmin Mehnert** – PCA & KNN, preprocessing support  
- **Nicolas Reichardt** – Random Forest, XGBoost, evaluation  
All team members contributed to meetings, reviews, and report writing.
