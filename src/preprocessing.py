#!/usr/bin/env python
# coding: utf-8

# === Preprocessing for Breast Cancer Diagnosis ===

# === Import required libraries ===

import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def preprocess():
    # === Create Output Folders ===
    os.makedirs(os.path.join('..', 'data'), exist_ok=True)
    os.makedirs(os.path.join('..', 'models'), exist_ok=True)

    # === Load the Dataset ===
    dataset = load_breast_cancer()
    breast_cancer_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    breast_cancer_df['target'] = dataset.target

    # === Split Features and Target ===
    X = breast_cancer_df.drop(columns='target')
    y = breast_cancer_df['target']

    # === Check for Missing Values ===
    assert not X.isnull().any().any(), "❌ Missing values found in features!"
    assert not y.isnull().any(), "❌ Missing values found in target!"

    # === Train-Test Split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # === Save Raw Split Data ===
    np.save(os.path.join('..', 'data', 'X_train.npy'), X_train)
    np.save(os.path.join('..', 'data', 'X_test.npy'), X_test)
    np.save(os.path.join('..', 'data', 'y_train.npy'), y_train)
    np.save(os.path.join('..', 'data', 'y_test.npy'), y_test)

    # === Feature Scaling ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # === Save Scaled Data ===
    np.save(os.path.join('..', 'data', 'X_train_scaled.npy'), X_train_scaled)
    np.save(os.path.join('..', 'data', 'X_test_scaled.npy'), X_test_scaled)

    # === Save Scaler and Feature Names ===
    joblib.dump(scaler, os.path.join('..', 'models', 'scaler.joblib'))
    joblib.dump(X.columns.tolist(), os.path.join('..', 'models', 'feature_names.joblib'))

    print('Preprocessing complete!')


if __name__ == "__main__":
    preprocess()
