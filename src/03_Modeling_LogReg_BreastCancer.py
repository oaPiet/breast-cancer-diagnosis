#!/usr/bin/env python
# coding: utf-8

# Logistic Regression for Breast Cancer Diagnosis
# Train and evaluate a logistic regression model using cross-validation, ROC curves, and feature importance visualization.

# === Imports ===

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# === Load Preprocessed Data ===

X_train_scaled = np.load(os.path.join('..', 'data', 'X_train_scaled.npy'))
X_test_scaled = np.load(os.path.join('..', 'data', 'X_test_scaled.npy'))
y_train = np.load(os.path.join('..', 'data', 'y_train.npy'))
y_test = np.load(os.path.join('..', 'data', 'y_test.npy'))

# === Train Logistic Regression Model with Cross-Validation ===

logistic_regression_model = LogisticRegressionCV(
    penalty='l2',
    solver='saga',
    cv=5,
    Cs=10,
    random_state=42,
    max_iter=10000
)

logistic_regression_model.fit(X_train_scaled, y_train)

# === Evaluate Model: Accuracy and Coefficients ===

y_pred = logistic_regression_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Best C (Regularization strength): {logistic_regression_model.C_}')
print(f'Coefficients:\n{logistic_regression_model.coef_}')

# === Confusion Matrix & Classification Report ===

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

# === ROC Curve & AUC ===

y_prob = logistic_regression_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# === Feature Importance (Coefficient Analysis) ===

feature_names = joblib.load(os.path.join('..', 'models', 'feature_names.joblib'))

coefs = logistic_regression_model.coef_[0]
non_zero_indices = np.where(coefs != 0)[0]

non_zero_coefs = coefs[non_zero_indices]
non_zero_features = [feature_names[i] for i in non_zero_indices]

sorted_idx = np.argsort(np.abs(non_zero_coefs))[::-1]
sorted_features = [non_zero_features[i] for i in sorted_idx]
sorted_coefs = non_zero_coefs[sorted_idx]

top_n = 10
plt.figure(figsize=(10, 6))
plt.barh(sorted_features[:top_n], sorted_coefs[:top_n])
plt.gca().invert_yaxis()
plt.title(f"Top {top_n} Feature Coefficients")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()

# === Cross-Validation Score Plot ===

cv_scores = logistic_regression_model.scores_[1].mean(axis=0)

plt.figure(figsize=(8, 6))
plt.plot(logistic_regression_model.Cs_, cv_scores, marker='o', color='orange')
plt.xscale('log')
plt.xlabel('Regularization Parameter (C)')
plt.ylabel('Mean CV Accuracy')
plt.title('Cross-Validation Accuracy vs. C')
plt.grid(True)
plt.show()

print("Cross-validation scores by C value:")
for c, score in zip(logistic_regression_model.Cs_, cv_scores):
    print(f"C = {c:.4f} -> CV Accuracy = {score:.4f}")

# === Misclassified Samples ===

misclassified_indices = np.where(y_test != y_pred)[0]
print(f"\nNumber of misclassified samples: {len(misclassified_indices)}")

for idx in misclassified_indices[:5]:
    print(f"Index {idx}: True = {y_test[idx]}, Predicted = {y_pred[idx]}")

# === Save Trained Model ===

try:
    joblib.dump(logistic_regression_model, os.path.join('..', 'models', 'log_reg_model.joblib'))
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")
