# Breast Cancer Diagnosis - Machine Learning Project

This project focuses on building a machine learning pipeline to classify breast cancer tumors as benign or malignant using clinical data. The primary goal is to support early diagnosis with a clean, modular, and interpretable machine learning workflow.

## 🧠 Project Overview

* **Dataset**: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
* **Problem Type**: Binary Classification
* **Target Variable**: `target` (0 = benign, 1 = malignant) — this was adjusted for clarity
* **Positive Class**: Malignant (`1`)

## 📁 File Structure

```
notebooks/
├── 01_eda_breast_cancer.ipynb
├── 02_preprocessing_breast_cancer.ipynb
├── 03_modeling_logreg_breast_cancer.ipynb
├── 04_modeling_svm_breast_cancer.ipynb
src/
├── 01_eda_breast_cancer.py
├── 02_preprocessing_breast_cancer.py
├── 03_modeling_logreg_breast_cancer.py
├── 04_modeling_svm_breast_cancer.py
├── models/                             # Saved model artifacts
│   ├── logistic_model.joblib
│   ├── svm_model.joblib
├── README.md
```

## 🔍 Notebooks

* **01\_EDA\_Breast\_Cancer.ipynb**: Overview of dataset, distribution of features, diagnosis class balance.
* **02\_Preprocessing\_BreastCancer.ipynb**: Handles missing values, label encoding, feature scaling, and data splits.
* **03\_Modeling\_BreastCancer.ipynb**: Trains and evaluates models (Logistic Regression). Uses accuracy, precision, recall, F1-score, and ROC-AUC for evaluation.
* **04\_Modeling\_SVM\_BreastCancer.ipynb**: Trains, tunes, and evaluates a Support Vector Machine (SVM) model, with a focus on maximizing recall.

## 🧪 Models

| Model                  | Accuracy | Recall | ROC AUC | Notes                         |
| ---------------------- | -------- | ------ | ------- | ----------------------------- |
| Logistic Regression    | 0.97     | 0.94   | 0.9947  | Baseline model                |
| Support Vector Machine | 0.80     | 0.98   | 0.9787  | Tuned to prioritize recall    |

## 🛠️ Dependencies

* Python >= 3.8
* scikit-learn
* pandas
* numpy
* matplotlib / seaborn
* joblib

## 🚀 How to Run

```bash
# Clone the repo
$ git clone https://github.com/oaPiet/breast-cancer-diagnosis.git

# Navigate to project folder
$ cd breast-cancer-diagnosis

# Create virtual environment
$ python -m venv venv
$ source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
$ pip install -r requirements.txt

# Run notebooks sequentially
```

## 📬 Author

**Omar Alejandro**
[GitHub](https://github.com/oaPiet) | [LinkedIn](https://www.linkedin.com/in/omar-alejandro-b258a4b8/)

---

> This repository is part of my transition into machine learning. Feel free to suggest improvements or collaborate!
