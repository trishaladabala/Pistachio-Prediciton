# 🟢 Pistachio Classification - ML Model Evaluation

This repository presents a comprehensive evaluation of machine learning models on a pistachio classification dataset, including both pre- and post-feature-optimization results.

---

## 📂 Dataset

- **Kaggle**: https://www.kaggle.com/datasets/muratkokludataset/pistachio-dataset/data
- **Features**: 28 quantitative measurements per sample
- **Target**: `Class` (encoded via `LabelEncoder`)
- **Preprocessing Steps**:
  - Label encoding of target classes
  - Correlation analysis and CSV export (`correlation_matrix.csv`)
  - Removal of low-importance features (`Kurtosis_RB`, `Kurtosis_RG`, `Shapefactor_4`, `StdDev_RG`) for optimization

---

## 📈 Model Performance Comparison

### 🔍 Before Feature Optimization

| Model                          | Accuracy  |
|--------------------------------|-----------|
| Random Forest Classifier       | 89.76%    |
| K-Nearest Neighbors (KNN)      | 89.75%    |
| Logistic Regression            | 93.25%    |
| Artificial Neural Network (ANN)| 93.95%    |

### 🚀 After Feature Optimization

| Model                          | Accuracy  |
|--------------------------------|-----------|
| Random Forest Classifier       | 88.85%    |
| K-Nearest Neighbors (KNN)      | 90%       |
| Logistic Regression            | 92.70%    |
| Artificial Neural Network (ANN)| 94.41%    |

> **Key Insight:** The ANN remains the top performer after optimization (94.42%), while logistic regression also shows strong robustness (92.79%).
> **Key Insight:** Surprisingly enough, the accuracy score for the ANN increased when the columns with correlation coefficient less than 0.1 were removed, while for others, the accuracy score decreased.

---

## 🧰 Tech Stack

- **Language**: Python 3.10
- **Libraries**:
  - Data handling: `pandas`, `numpy`
  - Modeling: `scikit-learn` (RF, KNN, LR), `TensorFlow`/`Keras` (ANN)

---

## 🚀 Future Work

- 🔄 Hyperparameter tuning for all models
- ✅ K-fold cross-validation and stratified sampling
- 🔍 Feature importance visualization and dimensionality reduction

---
