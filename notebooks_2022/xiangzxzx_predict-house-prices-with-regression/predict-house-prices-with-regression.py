#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


df_train.head()


# In[ ]:


print('Shape of train set: {}'.format(df_train.shape))
print('Shape of test set: {}'.format(df_test.shape))


# In[ ]:


#do a cross check between train and test sets to verify all columns are present in both sets, except for the target variable in train set.

variables_not_in_train_set = [i for i in df_train.columns if i not in df_test.columns]
print('Columns not in test set but present in train set: {}'.format(variables_not_in_train_set))


# In[ ]:


df_train.info()


# We can see quite a number of columns with missing values. We will deal with that later. 

# In[ ]:


#drop the Id column for both train and test sets

df_train.drop(columns=['Id'], inplace=True)
df_test.drop(columns=['Id'], inplace=True)


# # 1. Numerical Data 

# We first deal with the data by grouping them into numerical and categorical data. We start off with the numerical data and perform the following investigations:
# 
# * Determine the Pearson Correlation coefficient of each features with the target variable. We will drop any features which are deemed poorly correlated with the target variable.
# 
# * Drop any features with multicollinearity.
# 
# * Determine the distribution of the features and compute their skewness and kurtosis.
#  
# * Impute any missing values or drop any features when imputation doesnt help. 
# 

# In[ ]:


#Filter out a list containing only numerical variables
df_train_numerical = [i for i in df_train.columns if df_train[i].dtype != 'object']

#compute the correlation heatmap to see which features are highly correlated with target
corr_matrix = df_train[df_train_numerical].corr()
plt.figure(figsize=(20,20), dpi=70)
sns.heatmap(corr_matrix, cmap=plt.cm.Reds, annot=True)
plt.show()


# In[ ]:


corr_with_SalePrice = df_train[df_train_numerical].corr()['SalePrice'][:-1]
print('Features highly correlated with SalePrice: {}'.format(corr_with_SalePrice[corr_with_SalePrice >= 0.5].sort_values(ascending=False).round(2)))


# In[ ]:


print('Features fairly correlated with SalePrice: {}'.format(corr_with_SalePrice[(corr_with_SalePrice < 0.5) & (corr_with_SalePrice > 0.3)].sort_values().round(2)))


# In[ ]:


print('Features poorly correlated with SalePrice: {}'.format(corr_with_SalePrice[corr_with_SalePrice > 0.3].sort_values().round(2)))


# In[ ]:


#strong features with corr more than 0.5
strong_features = corr_with_SalePrice[corr_with_SalePrice >= 0.5].sort_values(ascending=False).index
strong_list = [x for x in strong_features]
strong_list.append('SalePrice')

def reg_plot(df, features, rows, col):
    fig = plt.figure(figsize=(19,19), dpi=70)
    for i, feature in enumerate(features):
        if feature != 'SalePrice':
            ax = fig.add_subplot(rows, col, i+1)
            sns.regplot(x=feature, y='SalePrice', data=df, line_kws={'color':'black'})
            ax.set_xlabel(feature)
            ax.set_ylabel('SalePrice')
    
reg_plot(df_train[strong_list], strong_list, 4, 3)


# In[ ]:


#features with fair corr between 0.3 to 0.5
fair_features = corr_with_SalePrice[(corr_with_SalePrice < 0.5) & (corr_with_SalePrice > 0.3)].sort_values(ascending=False).index
fair_list = [x for x in fair_features]
fair_list.append('SalePrice')

reg_plot(df_train[fair_list], fair_list, 4, 3)


# In[ ]:


df_train[df_train_numerical].corr().values

