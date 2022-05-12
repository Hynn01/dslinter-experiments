#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (EDA)
# ## Tabular playground series May-2022 Dataset <br>
# By: Jhonnatan Torres
# ___

# Importing required or key libraries

# In[ ]:


import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Reading the data

# In[ ]:


train = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv')
test = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/test.csv')


# The feature **f_27** contains 10 letters, in the following cell each letter is assigned to a column and the result is transformed to a **category** type

# In[ ]:


for i in np.arange(10):
    train['f_27_'+str(i)]=train['f_27'].apply(lambda x: x[i])
    train['f_27_'+str(i)] = train['f_27_'+str(i)].astype('category')
    test['f_27_'+str(i)]=test['f_27'].apply(lambda x: x[i])
    test['f_27_'+str(i)] = test['f_27_'+str(i)].astype('category')


# Dropping columns **'id'** and **'f_27'** for this EDA

# In[ ]:


train.drop(columns=['id', 'f_27'], inplace=True)
test.drop(columns=['id', 'f_27'], inplace=True)


# Assigning a **category** type to the features with an **"int64"** data type

# In[ ]:


CATEGORICAL_COLUMNS = list(train.drop(columns=['target']).select_dtypes(include=["int64"]).columns)
for col in CATEGORICAL_COLUMNS:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')


# In[ ]:


CATEGORICAL_COLUMNS = list(train.select_dtypes(include='category').columns)
print(CATEGORICAL_COLUMNS)


# Summary of the **train** dataset, it does not contain null values

# In[ ]:


train.info()


# ### Train Categorical Features

# In[ ]:


plt.figure(figsize=(16, 12))
for i, c in enumerate(CATEGORICAL_COLUMNS):
        plt.subplot(5, 5, i+1)
        sns.countplot(data=train, x=c, hue='target')
        plt.title("Feature_Name: "+c)
plt.tight_layout()


# for the 8th letter of the original **f_27** feature, an "inverse corelation" between the letter and the target value is observed, for the other features there is similar target distribution, as a result, there is not a clear *decision boundary*

# ### Train Numeric Features

# In[ ]:


NUM_COLUMNS = list(train.select_dtypes(include=["float64"]).columns)
plt.figure(figsize=(16, 12))
for i, c in enumerate(NUM_COLUMNS):
        plt.subplot(4, 4, i+1)
        sns.boxplot(data=train, x='target', y=c, hue='target')
        plt.title("Feature_Name: "+c)
plt.tight_layout()


# All the numeric features are zero centered and contain some outliers, there is not a clear difference or *decision boundary* when these features are compared to the target

# ### Correlation of numeric features (train)

# In[ ]:


plt.figure(figsize=(16,8))
sns.heatmap(train[NUM_COLUMNS].corr(), cmap='bwr', annot=True, fmt='.2f')
plt.title("Correlation of Numeric (float) features")
plt.show()


# There is some correlation between **f_00** - **f_06** and **f_28** features, also, a "cluster" is observed for the features **f_19** - **f_26**

# ## Comparing Train and Test datasets

# In[ ]:


train['Category'] = "train"
test['Category'] = "test"
train.drop(columns='target', inplace=True)
print(train.shape, test.shape)


# In[ ]:


full_dataset = pd.concat(objs=[train, test])


# ### Categorical features

# In[ ]:


plt.figure(figsize=(16, 12))
for i, c in enumerate(CATEGORICAL_COLUMNS):
        plt.subplot(5, 5, i+1)
        sns.countplot(data=full_dataset, x=c, hue='Category')
        plt.title("Feature_Name: "+c)
plt.tight_layout()


# No anomalies are observed between the distributions of the two datasets

# ### Numeric Features

# In[ ]:


plt.figure(figsize=(16, 12))
for i, c in enumerate(NUM_COLUMNS):
        plt.subplot(4, 4, i+1)
        sns.boxplot(data=full_dataset, x='Category', y=c, hue='Category')
        plt.title("Feature_Name: "+c)
plt.tight_layout()


# Like in the training dataset, all the features are zero centered and show a similar distribution

# ## Closing Comments
# 
# * Based on this EDA and just using the available features (w/o interactions) there is not a clear *decision boundary*, Categorical and Numeric Features have a similar distribution when compared to the target
# * In my humble opinion, this is a competition or challenge for *feature engineering*
