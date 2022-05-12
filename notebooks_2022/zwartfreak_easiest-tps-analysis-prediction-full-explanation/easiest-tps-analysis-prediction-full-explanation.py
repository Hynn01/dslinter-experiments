#!/usr/bin/env python
# coding: utf-8

# #### If you like my work, please, leave an upvote: it will be really appreciated and it will motivate me in offering more content to the Kaggle community ! :)
# 
# #### We will go through all the processes:
# * Data Loading
# * Data Analysis
# * Data Preprocessing
# * Data Visualization
# * Data Modelling
# * Prediction

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor


# > ## Loading the data

# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')
test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv')


# > ## Data Analysis

# In[ ]:


train.head()


# In[ ]:


train.shape, test.shape


# In[ ]:


train.isnull().sum().sum(), test.isnull().sum().sum()


# #### There are no NULL values. That's a relief.

# In[ ]:


train.dtypes, test.dtypes


# #### Ok so all are numerical except `f_27`. That's a relief again.
# 
# ##### Let's drop them.

# In[ ]:


train = train.drop(['f_27'], axis=1)
test = test.drop(['f_27'], axis=1)


# > ## Data Preprocessing

# In[ ]:


cols = train.columns


# In[ ]:


d_train = preprocessing.normalize(train)
train_scaled = pd.DataFrame(d_train, columns=cols)
train_scaled.head()

d_test = preprocessing.normalize(train)
test_scaled = pd.DataFrame(d_test, columns=cols)


# In[ ]:


train.head()


# > ## Data Visualization

# In[ ]:


plt.figure(figsize=(10, 6))
plt.title('Target distribution')
ax = sns.countplot(x=train['target'], data=train)


# In[ ]:





# > ## Data Modelling

# In[ ]:


x = train.drop(['id', 'target'], axis=1)
y = train['target']
x_scaled = train_scaled.drop(['id', 'target'], axis=1)
y_scaled = train_scaled['target']


# In[ ]:


x.shape, y.shape


# In[ ]:


# set up the model
catboost_model = CatBoostRegressor(n_estimators=100,
                                   loss_function = 'RMSE',
                                   eval_metric = 'RMSE',)
# fit model
catboost_model.fit(x, y)

# set up the model
catboost_model_scaled = CatBoostRegressor(n_estimators=100,
                                   loss_function = 'RMSE',
                                   eval_metric = 'RMSE',)
# fit model
catboost_model_scaled.fit(x_scaled, y_scaled)


# > ## Prediction

# In[ ]:


preds = catboost_model.predict(test)
preds_scaled = catboost_model_scaled.predict(test_scaled)


# In[ ]:


#preds, preds_scaled


# ### Work in progress!!
