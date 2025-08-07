#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler


# ### Load the dataset

# In[9]:


from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()

breast_cancer_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)


# In[10]:


breast_cancer_df['target'] = dataset.target

breast_cancer_df.head()


# In[11]:


X = breast_cancer_df.drop(columns='target')
y = breast_cancer_df['target']


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# In[13]:


# Optional: Save scaled data for reuse
np.save('../data/y_train.npy', y_train)
np.save('../data/y_test.npy', y_test)


# In[14]:


#Binazer

