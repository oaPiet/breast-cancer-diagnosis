#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Prediction - Exploratory Data Analysis
# 
# **Author**: @oaPiet  
# **GitHub Profile**: [github.com/oaPiet](https://github.com/oaPiet)  
# **Project Repository**: [github.com/oaPiet/breast-cancer-diagnosis](https://github.com/oaPiet/breast-cancer-diagnosis)  
# **Dataset**: Scikit-learn Breast Cancer Dataset  
# 
# **Goal**: Understand data patterns to inform preprocessing and modeling decisions.  
# **Why It Matters**: Early detection of breast cancer can significantly improve patient outcomes. This analysis aims to identify key features that differentiate benign and malignant tumors.

# ### Import libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ### Load Data

# In[2]:


from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()


# #### Identify Target Labels

# In[3]:


for i in np.unique(dataset.target):
    print(f"Label {i} -> {dataset.target_names[i]}")


# ### Create the DataFrame from the dataset

# In[4]:


breast_cancer_df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
breast_cancer_df['target'] = dataset.target
breast_cancer_df.head()


# ### Check Missing Values

# In[5]:


print(breast_cancer_df.isnull().sum())


# ### Check Duplicate Rows

# In[6]:


print(breast_cancer_df.duplicated().sum())


# ### Create Labeled Target Column

# In[7]:


breast_cancer_df['target_labeled'] = breast_cancer_df['target'].map({0: 'malignant', 1: 'benign'})


# ### Check Class Distribution

# In[8]:


sns.countplot(data=breast_cancer_df, x='target_labeled')
plt.title('Distribution of Benign and  Malignant tumor')
plt.xlabel('Tumor type')
plt.ylabel('Count')
plt.show()


# > **Class Distribution**:
# > - **Benign**: Approx. 350 samples (**~63%**)
# > - **Malignant**: Approx. 200 samples (**~37%**)
# >
# > While benign cases are more frequent (<70–80% of the data), the dataset is **not severely imbalanced**.  
# > This allows us to use **accuracy** as a valid metric, but we’ll still monitor **recall** to ensure malignant tumors aren’t missed.

# ### Check Correlations between Features and Target

# In[9]:


correlation_with_target = breast_cancer_df.corr(numeric_only=True)['target'].sort_values(ascending=False)
print(correlation_with_target)


# # Selecting Features with Highest Absolute Correlation to the Target

# In[10]:


top_features_abs_correlation = correlation_with_target.abs().sort_values(ascending=False).head(10).index
print(top_features_abs_correlation)


# In[11]:


plt.figure(figsize=(9,7))
sns.heatmap(data=breast_cancer_df[top_features_abs_correlation].corr(), annot=True, cmap='coolwarm')
plt.title('Top Correlated Features with Target')
plt.tight_layout()
plt.show()


# > The heatmap reveals strong negative correlations with the target variable, particularly for features such as *'worst concave points'* (−0.79), *'worst perimeter'* (−0.78), and *'mean concave points'* (−0.78), indicating that higher values are associated with malignant tumors. Additionally, high inter-feature correlations — such as between *'mean perimeter'* and *'mean radius'* (0.99) — suggest multicollinearity, which can be addressed during modeling using techniques like **L1 regularization** or **feature selection**.

# ## Visualize Key Features
# 
# To better understand the relationships between the most predictive features and the target variable, we'll visualize them using a pairplot. This will help us see how benign and malignant tumors differ across these features.

# In[12]:


selected_features = ['worst concave points', 'worst perimeter', 'mean concave points', 'worst radius', 'target_labeled']
sns.pairplot(breast_cancer_df[selected_features], hue='target_labeled', palette='Set1')
plt.show()


# >**Benign tumors are typically associated with lower values of features** such as *'worst concave points'*, *'worst perimeter'*, and *'worst radius'*, **whereas higher values in these same features are strong indicators of malignancy.**

# # Boxplots (for outliers and distribution)
# 
# Boxplots provide a clear view of the distribution and spread of key features across benign and malignant tumors. These plots highlight differences in central tendency (median) and variability (IQR).

# In[13]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=breast_cancer_df, x='target_labeled', y='worst concave points')
plt.title('Worst Concave Points by Tumor Type')
plt.xlabel('Tumor Type')
plt.ylabel('Worst Concave Points')
plt.show()


# >Benign tumors have a lower median and show less variance in *'worst concave points'* compared to malignant tumors.

# In[14]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=breast_cancer_df, x='target_labeled', y='worst perimeter')
plt.title('Worst Perimeter by Tumor Type')
plt.xlabel('Tumor Type')
plt.ylabel('Worst Perimeter')
plt.show()


# >Benign tumors have a lower median and show less variance in *'worst perimeter'* compared to malignant tumors.

# In[15]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=breast_cancer_df, x='target_labeled', y='mean concave points')
plt.title('Mean Concave Points by Tumor Type')
plt.xlabel('Tumor Type')
plt.ylabel('Mean Concave Points')
plt.show()


# >Benign tumors have a lower median and show less variance in *'mean concave points'* compared to malignant tumors.

# In[16]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=breast_cancer_df, x='target_labeled', y='mean radius')
plt.title('Mean Radius by Tumor Type')
plt.xlabel('Tumor Type')
plt.ylabel('Mean Radius')
plt.show()


# >Benign tumors have a lower median and show less variance in *'mean radius'* compared to malignant tumors.
