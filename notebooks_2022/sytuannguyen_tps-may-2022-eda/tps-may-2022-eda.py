#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['axes.facecolor'] = 'gray'


# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv', index_col='id')
train.head(3)


# In[ ]:


train.info()


# # <span style="color:#A80808">Target</span>

# In[ ]:


plt.figure(figsize=(7,5))
train.target.hist(bins=5, color='w')
plt.xlabel('Target', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.show()


# A small imbalance can be observed for the target.

# # <span style="color:#A80808">String feature: f_27</span>

# In[ ]:


print('Len of f_27:', train.f_27.apply(lambda x: len(x)).unique())

for idx in range(10):
    train[f'f_27_{idx}'] = train.f_27.apply(lambda x: x[idx])
    print(f'Unique values of f_27[{idx}]', sorted(train[f'f_27_{idx}'].unique()))


# In[ ]:


plt.figure(figsize=(15,25))

for idx in range(10):
    plt.subplot(5,2,idx+1)
    train[f'f_27_{idx}'].hist(bins=40, color='w')
    plt.ylabel('Count', fontsize=16)
    plt.xlabel(f'f_27_{idx}', fontsize=16)
    plt.tight_layout()
plt.show()


# # <span style="color:#A80808">Int features</span>

# In[ ]:


int_features = [col for col in train.columns if train[col].dtype=='int64' and col!='target']
print('Number of int features:', len(int_features))
for feature in int_features:
    print(f'Unique values of {feature}]', sorted(train[feature].unique()))


# In[ ]:


plt.figure(figsize=(15,25))

for idx, feature in enumerate(int_features):
    plt.subplot(5,3,idx+1)
    train[feature].hist(bins=40, color='w')
    plt.ylabel('Count', fontsize=16)
    plt.xlabel(feature, fontsize=16)
    plt.tight_layout()
plt.show()


# The distributions of the int features from f_07 to f_18 are quite similar. Besides, the three unique values of the feature f_30 are quite balance.

# # <span style="color:#A80808">Real features</span>

# In[ ]:


float_features = [col for col in train.columns if train[col].dtype=='float64' and col!='f_28']


# In[ ]:


train_float = train[float_features]
train_float_stat = train_float.describe().loc[['mean','std','min','max']]


# In[ ]:


plt.figure(figsize=(10,10))

for i, idx in enumerate(train_float_stat.index):
    plt.subplot(4,1,i+1)
    plt.bar(float_features, train_float_stat.loc[idx], color='w')
    plt.ylabel(idx,fontsize=16)
    plt.tight_layout()
plt.show()


# The features f_00 to f_06 are quite similar. Also, we can observe the similarity for the group of float features from f_19 to f_26.

# In[ ]:


plt.figure(figsize=(10,10))
for idx, feature in enumerate(['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06']):
    plt.subplot(4,2,idx+1)
    train[feature].hist(bins=300, color='w')
    plt.title(feature, fontsize=16)
    plt.tight_layout()
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
for idx, feature in enumerate(['f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26']):
    plt.subplot(4,2,idx+1)
    train[feature].hist(bins=300, color='w')
    plt.title(feature, fontsize=16)
    plt.tight_layout()
plt.show()


# # <span style="color:#A80808">Real feature f_28</span>

# In[ ]:


plt.figure(figsize=(7,5))
train.f_28.hist(bins=300, color='w')
plt.title('f_28', fontsize=16)
plt.show()

