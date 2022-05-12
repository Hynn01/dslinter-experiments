#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # install

# In[ ]:


get_ipython().system('pip install --user pycaret')


# # Library Import

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


from pycaret.classification import *


# # Data Description

# For this challenge, you are given (simulated) manufacturing control data and are tasked to predict whether the machine is in state 0 or state 1. The data has various feature interactions that may be important in determining the machine state.
# 
# Files train.csv - the training data, which includes normalized continuous data and categorical data test.csv - the test set; your task is to predict binary target variable which represents the state of a manufacturing process sample_submission.csv - a sample submission file in the correct format

# # Data loading

# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')
test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv')
sub = pd.read_csv('../input/tabular-playground-series-may-2022/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


pd.set_option('display.max_columns', None)
train.describe()


# No missing values

# In[ ]:


train.isnull().sum()


# No duplicate data

# In[ ]:


train.duplicated().sum()


# **f_27 data is str.
# So I make histgram without f_27.Let's compare target_0 and target_1.**

# In[ ]:


n_columns = train.columns[1:28]
n_columns = n_columns.append(train.columns[29:32])
n_columns


# In[ ]:


train_0 = train[train['target'] ==0]
train_1 = train[train['target'] ==1]


# In[ ]:


fig1 = plt.figure(figsize=(20,16))
fig1.suptitle('each histgram without f_27', fontsize =16)
plt.subplots_adjust(wspace=0.4, hspace=0.3)
for i, column in enumerate(n_columns):
    plt.subplot(6, 6, i+1)
    plt.hist(train_0[column], bins =100, alpha=0.5, label='target=0')
    plt.hist(train_1[column], bins =100, alpha=0.5, label='target=1')
    plt.title(f'{n_columns[i]}')
    plt.legend()
plt.show()


# **Next, Let's see f_27.
# All f_27 data is 10 characters**

# In[ ]:


train['f27_len']=train['f_27'].str.len()
train['f27_len'].value_counts()


# **Separate f27 data character by character
# and Let's visualize them.**

# In[ ]:


for i in range(10):
  train[f'f27_{i}'] = train['f_27'].str[i]


# In[ ]:


train_0 = train[train['target'] ==0]
train_1 = train[train['target'] ==1]


# In[ ]:


fig2 = plt.figure(figsize=(20, 12))
fig2.suptitle('f_27_histgram_string', fontsize=16)
plt.subplots_adjust(wspace=0.4, hspace=0.3)
for i in range(10):
  plt.subplot(3, 4, i+1)
  temp0 = train_0[f'f27_{i}'].value_counts().sort_index()
  temp1 = train_1[f'f27_{i}'].value_counts().sort_index()
  plt.bar(temp0.index, temp0.values, label='target:0', align="edge", width=-0.3)
  plt.bar(temp1.index, temp1.values, label='target:1', align="edge", width=0.3)
  plt.title(f'f_27_{i}')
  plt.xlabel('str')
  plt.ylabel('counts')
  plt.legend()
plt.show()


# # Modeling with Pycaret

# In[ ]:


join_data =  pd.concat([train, test], sort=False, ignore_index=True)
join_data['f27_len']=join_data['f_27'].str.len()
for i in range(10):
  join_data[f'f27_{i}'] = join_data['f_27'].str[i]
train_fe= join_data[:len(train)]
test_fe= join_data[len(train):]


# In[ ]:


train_fe.columns


# In[ ]:


ignore_cols = ['id', 'f_27', 'f27_len']
c_columns = ['f_07', 'f_08', 'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15',
       'f_16', 'f_17', 'f_18', 'f_29', 'f_30', 'f27_0', 'f27_1', 'f27_2',
       'f27_3', 'f27_4', 'f27_5', 'f27_6', 'f27_7', 'f27_8', 'f27_9']
n_columns = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_19', 'f_20',
       'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_28']


# I tried modeling with Pycaret, but I needed high memory and couldn't run to the end.
# Therefore, I will reduce the amount of data and perform screening first.

# In[ ]:


train_fe10, train_90 = train_test_split(train_fe, train_size = 0.1, random_state=0)
print(train_fe.shape)
print(train_fe10.shape)


# In[ ]:


exp = setup(data = train_fe10,
            ignore_features = ignore_cols,
            target = 'target',
            categorical_features = c_columns,
            session_id=123)


# In[ ]:


best_model = compare_models(fold=5)


# catboost is best model.
# Next, I will model using all the data.

# In[ ]:


exp = setup(data = train_fe,
            ignore_features = ignore_cols,
            target = 'target',
            categorical_features = c_columns,
            session_id=123)


# In[ ]:


catboost = create_model('catboost')


# In[ ]:


predictions = predict_model(catboost, data=test_fe)


# In[ ]:


sample = predictions[['id', 'Label']]
sample.rename(columns={'Label': 'target'}, inplace=True)


# In[ ]:


sample.to_csv("submission_test.csv", index=False)


# In[ ]:




