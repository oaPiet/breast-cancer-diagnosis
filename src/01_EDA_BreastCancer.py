#!/usr/bin/env python
# coding: utf-8

# === Breast Cancer Prediction - Exploratory Data Analysis ===
# 
# Author: @oaPiet  
# GitHub: https://github.com/oaPiet  
# Repo: https://github.com/oaPiet/breast-cancer-diagnosis  
# Dataset: scikit-learn Breast Cancer  
# 
# Goal: Understand patterns in the data to guide preprocessing and modeling decisions.
# Why: Early breast cancer detection improves outcomes. EDA helps identify key features.

# === Import libraries ===

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# === Load dataset ===

dataset = load_breast_cancer()

# === Map target labels ===

for i in np.unique(dataset.target):
    print(f"Label {i} -> {dataset.target_names[i]}")

# === Create DataFrame ===

df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target

# === Basic checks ===

print("Missing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# === Label target for visualization ===

df['target_labeled'] = df['target'].map({0: 'malignant', 1: 'benign'})

# === Class distribution ===

sns.countplot(data=df, x='target_labeled', palette='Set2')
plt.title('Benign vs Malignant Tumor Distribution')
plt.xlabel('Tumor Type')
plt.ylabel('Count')
plt.show()

# === Target correlation with features ===

correlation = df.corr(numeric_only=True)['target'].sort_values(ascending=False)
print("\nCorrelation with Target:\n", correlation)

# === Top 10 features (absolute correlation) ===

top_features = correlation.abs().sort_values(ascending=False).head(10).index
print("\nTop Correlated Features:\n", top_features)

# === Correlation heatmap ===

plt.figure(figsize=(9, 7))
sns.heatmap(df[top_features].corr(), annot=True, cmap='coolwarm')
plt.title('Top Correlated Features')
plt.tight_layout()
plt.show()

# === Pairplot of top features ===

selected = ['worst concave points', 'worst perimeter', 'mean concave points', 'worst radius', 'target_labeled']
sns.pairplot(df[selected], hue='target_labeled', palette='Set1')
plt.show()

# === Boxplots for important features ===

important_features = ['worst concave points', 'worst perimeter', 'mean concave points', 'mean radius']

for feature in important_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='target_labeled', y=feature, palette='pastel')
    plt.title(f'{feature} by Tumor Type')
    plt.xlabel('Tumor Type')
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()