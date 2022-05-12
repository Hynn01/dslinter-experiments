#!/usr/bin/env python
# coding: utf-8

# # House Prices: Advanced Regression Techniques

# This is a perfect notebook for data science students who have completed an online course in machine learning and are looking to expand their skill set before trying a featured competition. 

# ### Table of contents
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
#     <ol>
#         <li><a href="#ref1">Exploratory Data Analysis and Visualization</a></li>
#         <li><a href="#ref2">Predictive Modeling: Simple Linear Regression, Ridge Regression, Lasso Regression, Neural Network, Liner Regression with Polynomial Feature</a></li>
#         <li><a href="#ref3">Building a Simple Pipeline</a></li>
#     </ol>
# </div>
# <br>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import warnings
warnings.filterwarnings('ignore')


# # Exploratory Data Analysis and Visualization

# In[ ]:


test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
sample=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
print(train.shape)
print(test.shape)
print(sample.shape)


# In[ ]:


train['RoofStyle']


# In[ ]:


train.info()


# In[ ]:


train.drop(columns=['Id','SaleCondition','SaleType','MiscFeature','Fence','PoolQC','PavedDrive','GarageCond','GarageQual','GarageFinish','GarageType','FireplaceQu','Functional','KitchenQual','Electrical','CentralAir','HeatingQC','Heating','BsmtFinType2','BsmtFinType1','BsmtExposure','BsmtCond','BsmtQual','Foundation','ExterCond','ExterQual','MasVnrType','Exterior2nd','Exterior1st','RoofMatl','RoofStyle','HouseStyle','BldgType','Condition2','Condition1','Neighborhood','LandSlope','LotConfig','Utilities','LandContour','LotShape','Alley','Street','MSZoning'],inplace=True)
train.head()


# In[ ]:


train.info()


# In[ ]:


train.shape


# In[ ]:


test.drop(columns=['Id','SaleCondition','SaleType','MiscFeature','Fence','PoolQC','PavedDrive','GarageCond','GarageQual','GarageFinish','GarageType','FireplaceQu','Functional','KitchenQual','Electrical','CentralAir','HeatingQC','Heating','BsmtFinType2','BsmtFinType1','BsmtExposure','BsmtCond','BsmtQual','Foundation','ExterCond','ExterQual','MasVnrType','Exterior2nd','Exterior1st','RoofMatl','RoofStyle','HouseStyle','BldgType','Condition2','Condition1','Neighborhood','LandSlope','LotConfig','Utilities','LandContour','LotShape','Alley','Street','MSZoning'],inplace=True)
test.head()


# In[ ]:


test.info()


# In[ ]:


test.shape


# In[ ]:


train.info()


# **Missing Value Handling**

# In[ ]:


train['LotFrontage'].fillna(train['LotFrontage'].mean(),inplace=True)
train['MasVnrArea'].fillna(train['MasVnrArea'].mean(),inplace=True)
train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean(),inplace=True)
train.info()


# In[ ]:


test['LotFrontage'].fillna(test['LotFrontage'].mean(),inplace=True)
test['MasVnrArea'].fillna(test['MasVnrArea'].mean(),inplace=True)
test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean(),inplace=True)
test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mean(),inplace=True)
test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean(),inplace=True)
test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean(),inplace=True)
test['BsmtFullBath'].fillna(test['BsmtFullBath'].mean(),inplace=True)
test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mean(),inplace=True)
test['GarageYrBlt'].fillna(test['GarageYrBlt'].mean(),inplace=True)
test['GarageCars'].fillna(test['GarageCars'].mean(),inplace=True)
test['GarageArea'].fillna(test['GarageArea'].mean(),inplace=True)
test.info()


# **Visualization**

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(train['LotArea'],train['SalePrice'])


# In[ ]:


plt.scatter(train['MSSubClass'],train['SalePrice'])


# In[ ]:


plt.scatter(train['LotFrontage'],train['SalePrice'])


# In[ ]:


print(train.shape)
print(test.shape)
print(sample.shape)


# # Predictive Modeling

# **Simple Linear Regression**

# In[ ]:


X=train.drop(columns=['SalePrice']) #trying to understand the model with just a single feature
y=train[['SalePrice']]
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
X_trans=PolynomialFeatures(2).fit_transform(X)
test_trans=PolynomialFeatures(2).fit_transform(test)
model_lr=LinearRegression().fit(X_trans,y)
y_pred=model_lr.predict(test_trans)
y_pred


# In[ ]:


print(model_lr.score(X_trans,y))
print(model_lr.score(test_trans,sample['SalePrice']))


# In[ ]:


sample.head()


# In[ ]:


sample['SalePrice']=y_pred
sample.head()


# In[ ]:


"""sample.to_csv('/kaggle/working/output.csv', index=False)
data=pd.read_csv('/kaggle/working/output.csv')
data.head()"""


# **Ridge Regresssion**

# In[ ]:


from sklearn.linear_model import Ridge
model_ridge=Ridge(normalize=True).fit(X,y)
"""y_pred_ridge=model_ridge.predict(test)
sample['SalePrice']=y_pred_ridge
sample.to_csv('/kaggle/working/output_ridge.csv', index=False)
data=pd.read_csv('/kaggle/working/output_ridge.csv')
data.head()"""


# In[ ]:


model_ridge.score(X,y)


# **Lasso Regression**

# In[ ]:


from sklearn.linear_model import Lasso
model_lasso=Lasso(normalize=True).fit(X,y)
"""y_pred_lasso=model_lasso.predict(test)
sample['SalePrice']=y_pred_lasso
sample.to_csv('/kaggle/working/output_lasso.csv',index=False)
data_lasso=pd.read_csv('/kaggle/working/output_lasso.csv')
data_lasso.head()"""


# In[ ]:


model_lasso.score(X,y)


# **Neural Network**
# 

# In[ ]:


import keras
import tensorflow as tf


# In[ ]:


X.shape


# In[ ]:


model_nn=keras.Sequential([
    keras.layers.Dense(36, activation=tf.nn.relu,input_shape=[36]),
    keras.layers.Dense(36, activation=tf.nn.relu),
    keras.layers.Dense(36, activation=tf.nn.relu),
    keras.layers.Dense(1)
    ])


# In[ ]:


optimizer=tf.keras.optimizers.RMSprop(0.001)
model_nn.compile(loss='mean_squared_error',optimizer=optimizer, metrics=['mean_absolute_error','mean_squared_error'])
model_nn.fit(X,y,epochs=10)


# In[ ]:


"""y_pred_nn=model_nn.predict(test)
sample['SalePrice']=y_pred_nn
sample.to_csv('/kaggle/working/output_neural.csv',index=False)
data_nn=pd.read_csv('/kaggle/working/output_neural.csv')
data_nn.head()"""


# # Building a Simple Pipeline

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# In[ ]:


test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
sample=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')


# In[ ]:


X_train=train.drop(columns=['SalePrice'])
Y_train=train[['SalePrice']]


# In[ ]:


num_feat=X_train.select_dtypes(include='number').columns.to_list()
cat_feat=X_train.select_dtypes(exclude='number').columns.to_list()


# In[ ]:


num_pipe=Pipeline([
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipe=Pipeline([
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])


# In[ ]:


ct=ColumnTransformer(remainder='drop',
                    transformers=[
                        ('numeric', num_pipe, num_feat),
                        ('categorical', cat_pipe, cat_feat)
                    ])

model_ranf=Pipeline([
    ('transformer',ct),
    ('predictor', RandomForestRegressor())
])


# In[ ]:


model_ranf.fit(X_train, Y_train);


# In[ ]:


print(model_ranf.score(X_train, Y_train))


# **Submission Function**

# In[ ]:


def submission(test, model):
    y_pred=model.predict(test)
    sample['SalePrice']=y_pred
    date=pd.datetime.now().strftime(format='%d_%m_%Y_%H-%M_')
    sample.to_csv(f'/kaggle/working/{date}result.csv',index=False)


# In[ ]:


submission(test,model_ranf)


# **Linear Regression with Polynomial Feature**

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


ct=ColumnTransformer(remainder='drop',
                    transformers=[
                        ('numeric', num_pipe, num_feat),
                        ('categorical', cat_pipe, cat_feat)
                    ])
model=Pipeline([
    ('transformer',ct),
    ('poly',PolynomialFeatures(2)),
    ('predictor', Lasso())
])


# In[ ]:


model.fit(X_train, Y_train);


# In[ ]:


model.score(X_train, Y_train)


# **Please upvote if you like this or find this notebook useful, thanks.**
