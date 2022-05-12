#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# basic library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ##### Data Read

# In[ ]:


# read the train and test data
df_train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv', index_col = 0)
df_test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv', index_col = 0)


# In[ ]:


# check the shape for training set
df_train.shape


# In[ ]:


# check the shape for testing set
df_test.shape


# In[ ]:


# check the amount for duplicated sample in training set
print(f"Duplicated sample: {df_train.duplicated().sum()}")


# In[ ]:


# check the missing value in the training set
for i in df_train.columns:
    print(f"Missing value for {i}: {df_train[i].isna().sum()}")


# #### Statistical analysis

# ##### Distribution

# In[ ]:


int_list = []
float_list = []
object_list = []
for i in df_train.columns:
    if df_train[i].dtypes == 'int64':
        int_list.append(i)
    elif df_train[i].dtypes == 'float64':
        float_list.append(i)
    else:
        object_list.append(i)


# In[ ]:


print(f"int type features: {int_list}")
print(f"float type features: {float_list}")
print(f"object type features: {object_list}")


# ##### int type

# In[ ]:


# checking the range for the int_type feature
for i in int_list:
    temp_list = list(df_train[i].unique())
    temp_list.sort()
    print(f"{i}, min: {temp_list[0]}, max: {temp_list[-1]}, number of value: {len(temp_list)}")


# By looking up the value of target, it was treated as a binary classification.

# In[ ]:


# check with the distribution for the int type features
f = pd.melt(df_train, value_vars = int_list)
f['counts'] = 1
f = f.groupby(['variable','value']).sum()
ncols = 3
nrows = round(len(int_list) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(16, round(nrows*16/ncols)))
ax = axes.ravel()
for i in range(len(int_list)):
    ax[i].bar(data = f.loc[int_list[i]], x = f.loc[int_list[i]].index, height = 'counts')
    ax[i].set_title(int_list[i])


# ##### float type

# In[ ]:


# Coming soon


# ##### Correlation

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (15,8))
sns.heatmap(df_train.corr(), annot = False)


# According to the heatmap, there is some correlation between f_28 and the other feature.
# 
# The coefficient of some features and target will be extracted and check with to milticolinearity.

# ##### f_07

# In[ ]:


# check the absolute linear coefficient with other for f_07
abs(df_train.corr()['f_07']).sort_values(ascending = False)[1:6]


# ##### f_28

# In[ ]:


# check the absolute linear coefficient with other for f_28
abs(df_train.corr()['f_28']).sort_values(ascending = False)[1:6]


# ##### f_30

# In[ ]:


# check the absolute linear coefficient with other for f_30
abs(df_train.corr()['f_30']).sort_values(ascending = False)[1:6]


# ##### target

# In[ ]:


# check the absolute linear coefficient with other for target
abs(df_train.corr()['target']).sort_values(ascending = False)[1:6]


# Multicolinearity was found in some feature. Although linear regression may not be applied, it still needed to be fixed to help with the model performance.

# #### Feature engineering

# In[ ]:


# Coming soon


# #### Simple Binary Classification

# ##### Data Slicing

# In[ ]:


# slice the dataset for features and target
X = df_train.iloc[:,:-1].values
y = df_train.iloc[:,-1].values.reshape(-1,1)


# ##### Stochastic Gradient Descent Classifier

# In[ ]:


# Coming soon

