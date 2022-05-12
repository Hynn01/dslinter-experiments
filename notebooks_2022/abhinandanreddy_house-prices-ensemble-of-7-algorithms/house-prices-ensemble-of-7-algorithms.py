#!/usr/bin/env python
# coding: utf-8

# **I have used the following 7 algorithms to train the model:**
# 
# * RandomForestRegressor
# * RidgeCV
# * CatBoostRegressor
# * XGBRegressor
# * KNeighborsRegressor
# * GaussianProcessRegressor
# * Support Vector Regression
# 
# 
# CatBoostRegressor turned out to be the most accurate model without much fine-tuning. 
# 
# The final results based on the median of four most accurate models.

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error 

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#Read the data to dataframe
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')


# In[ ]:


#EDA
train.columns


# In[ ]:


#EDA
train.head(2)


# In[ ]:


#EDA
train['SalePrice'].describe()


# In[ ]:


#Histogram of sales price
plt.hist(train['SalePrice'])
plt.plot()

sns.displot(train['SalePrice']);


# In[ ]:


#Check skewness and kurtosis - can ignore this as beginner
print(train['SalePrice'].skew())
print(train['SalePrice'].kurt())


# In[ ]:


var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis =1) 
data.plot.scatter(var, 'SalePrice')


# In[ ]:


#Scatter Plot - Method 1
sns.scatterplot(train[var], train['SalePrice'])


# In[ ]:


#Scatter Plot - Method 2
plt.scatter( train[var],train['SalePrice'])


# In[ ]:


#Scatter Plot
sns.scatterplot(train['TotalBsmtSF'], train['SalePrice'])


# In[ ]:


var = 'YearBuilt'

plt.subplots(figsize=(16, 8))
sns.boxplot(x = var, y = 'SalePrice', data=train)
plt.title(var)   #matplotlib and seaborn work together in some cases
plt.xticks(rotation=90)
plt.ylim(0, 800000) 
plt.show()


# In[ ]:


# Since we cant keep exploring all the variables, lets make a heat map and explore the corelation of variables

corrmat = train.corr()
plt.subplots(figsize = (13,10))
sns.heatmap(corrmat)


# In[ ]:


#To focus on the largest corelations

col1 = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index
corrmat1 = train[col1].corr()
plt.subplots(figsize = (10,6))
sns.heatmap(corrmat1, annot=True)
plt.show()


# In[ ]:


#To focus on the largest negative corelations

col2 = corrmat.nsmallest(10, 'SalePrice')['SalePrice'].index
corrmat2 = pd.concat([train[col2], train['SalePrice']], axis=1).corr()
plt.subplots(figsize = (10,6))
sns.heatmap(corrmat2, annot=True)
plt.show()


# In[ ]:


#Scatter plot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols])


# In[ ]:


# Finding out missing data 
missing = train.isnull().sum().sort_values(ascending=False)
percentage = missing/len(train['SalePrice'])
missing_data = pd.concat([missing, percentage], axis = 1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[ ]:


# Removing these columns from train data
train1 = train.drop(columns=missing_data[missing_data['Total'] > 1].index)
train2 = train1.drop(train1[train1['Electrical'].isnull()].index)


# In[ ]:


# Removing these columns from test data as well
test1 = test.drop(columns=missing_data[missing_data['Total'] > 1].index)
len(test1.columns)


# In[ ]:


# Checking for missing values in train data again 
train2.isnull().sum().max()


# In[ ]:


# Reshaping to apply standard scaler
train_np = np.array(train2['SalePrice'])
train_np1 = train_np.reshape(-1,1)


# In[ ]:


# Applying standard scaler to look for outliers 
from sklearn.preprocessing import StandardScaler
scaled_price = StandardScaler().fit_transform(np.array(train2['SalePrice']).reshape(-1,1))
low_range = np.sort(scaled_price, axis = 0)[:10]
high_range = np.sort(scaled_price, axis = 0)[-10:]
print(low_range ,'\n', high_range)


# In[ ]:


#Looking for outliers
var = 'GrLivArea'
sns.scatterplot(data = pd.concat([train2[var], train2['SalePrice']], axis = 1), x = var, y = 'SalePrice')
plt.show()


# In[ ]:


#Index the two outliers
train2.sort_values(by = 'GrLivArea', ascending = False)[:2].index


# In[ ]:


#Drop the two records which have a high LivArea and seem like outliers
train4 = train2.drop([1298, 523])


# In[ ]:


# Visualise without outliers
var = 'GrLivArea'
sns.scatterplot(data = pd.concat([train4[var], train4['SalePrice']], axis = 1), x = var, y = 'SalePrice')
plt.show()


# In[ ]:


#Check for outliers
var = 'TotalBsmtSF'
sns.scatterplot(data = pd.concat([train4[var], train4['SalePrice']], axis = 1), x = var, y = 'SalePrice')
plt.show()


# In[ ]:


# Histogram and normal probability plot
from scipy.stats import norm
sns.distplot(train4['SalePrice'], fit = norm)
fig = plt.figure()
from scipy import stats
res = stats.probplot(train4['SalePrice'], plot = plt)


# In[ ]:


# Apply log transformation to SalePrice to narmalise it
train5 = train4.copy(deep = True)
train5['SalePrice'] = np.log(train4['SalePrice'])#train4['SalePrice']#


# In[ ]:


#Convert categorical features to numbers
train6 = pd.get_dummies(train5)


# In[ ]:


# Seperate the features and labels
x = train6.drop(['Id','SalePrice'], axis = 1)
y = train6['SalePrice']


# In[ ]:


# Create test and validation set
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1, random_state= 42)
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)


# In[ ]:


#Linear regression model - Linear regression model doesn't seem to be working well

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error 
# reg = LinearRegression().fit(x_train, y_train)
# y_pred = reg.predict(x_val)
# RMSE_score = mean_squared_error(y_val, y_pred, squared = False)
# print(RMSE_score)

#Score  = 301


# # Evaluating the accuracy on various models

# In[ ]:


# Model 1 - Random Forest Regression 
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor().fit(x_train, y_train)
y_pred = forest.predict(x_val)

RMSE_score = mean_squared_error(y_val, y_pred, squared = False)
print(RMSE_score)


# In[ ]:


# Model 2 - RidgeCV Regression 
from sklearn.linear_model import RidgeCV

ridge = RidgeCV().fit(x_train, y_train)
y_pred = ridge.predict(x_val)

RMSE_score = mean_squared_error(y_val, y_pred, squared = False)
print(RMSE_score)


# In[ ]:


# Model 3 - Catboost Regression 
from catboost import CatBoostRegressor

catb = CatBoostRegressor(iterations=3500,verbose=1000)
catb.fit(x_train, y_train)
y_pred = catb.predict(x_val)

RMSE_score = mean_squared_error(y_val, y_pred, squared = False)
print(RMSE_score)


# In[ ]:


# Model 4 - XGBoost Regression 
from xgboost import XGBRegressor

xgb_reg = XGBRegressor(n_estimators=3000, learning_rate=0.005)
xgb_reg.fit(x_train, y_train, early_stopping_rounds=5, eval_set=[(x_val, y_val)],verbose = 100)
y_pred = xgb_reg.predict(x_val)

RMSE_score = mean_squared_error(y_val, y_pred, squared = False)
print(RMSE_score)


# In[ ]:


# Model 5 - KNN Regression
from sklearn.neighbors import KNeighborsRegressor

knn_reg = KNeighborsRegressor().fit(x_train, y_train)
y_pred = knn_reg.predict(x_val)

RMSE_score = mean_squared_error(y_val, y_pred, squared = False)
print(RMSE_score)


# In[ ]:


# Model 6 - SVM Regression
from sklearn import svm

svm_reg = svm.SVR().fit(x_train, y_train)
y_pred = svm_reg.predict(x_val)

RMSE_score = mean_squared_error(y_val, y_pred, squared = False)
print(RMSE_score)


# In[ ]:


# Model 7 - Gaussian Regression
from sklearn.gaussian_process import GaussianProcessRegressor

g_reg = GaussianProcessRegressor().fit(x_train, y_train)
y_pred = g_reg.predict(x_val)

RMSE_score = mean_squared_error(y_val, y_pred, squared = False)
print(RMSE_score)


# In[ ]:


#Creating test data
x_test = test1.drop('Id', axis = 1)

#Convert to numbers
x_test = pd.get_dummies(x_test)


# In[ ]:


# Few columns are missing in test data since 
print(len(x_train.columns))
print(len(x_test.columns))


# In[ ]:


#Add the missing columns to test set
missing_cols = set(x_train.columns ) - set(x_test.columns )

for c in missing_cols:
    x_test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
x_test = x_test[x_train.columns]


# In[ ]:


x_test.isnull().sum().sort_values(ascending=False)


# In[ ]:


x_test.fillna(value = 0, inplace = True)


# In[ ]:


# Training on full data for submission on 4 most accurate models

# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
forest1 = RandomForestRegressor().fit(x, y)
y_pred_rf = forest1.predict(x_test)

# RidgeCV Regression
from sklearn.linear_model import RidgeCV
ridge = RidgeCV().fit(x, y)
y_pred_rcv = ridge.predict(x_test)

# Catboost Regression 
from catboost import CatBoostRegressor
catb = CatBoostRegressor(iterations=3500,verbose=1000).fit(x, y)
y_pred_cb = catb.predict(x_test)

# XGBoost Regression
xgb_reg = XGBRegressor(n_estimators=1800, learning_rate=0.005)
xgb_reg.fit(x, y, verbose = 1000)
y_pred_xgb = xgb_reg.predict(x_test)


# In[ ]:


#Combine results of four models and plot
#Converting the sale price back to normal number since we had applied log initially

rf  = np.exp(y_pred_rf) 
rcb = np.exp(y_pred_rcv) 
cat = np.exp(y_pred_cb) 
xgb = np.exp(y_pred_xgb)

results = pd.DataFrame({'RF': rf, 'RCB': rcb, 'CAT': cat, 'XGB': xgb}, columns=['RF', 'RCB', 'CAT', 'XGB'])

sns.pairplot(results)


# In[ ]:


np.round(results.head())


# In[ ]:


#Taking median of the 4 model's results
submission['SalePrice'] = np.median(results, axis = 1)

submission.head()


# In[ ]:


submission.to_csv('submission.csv', index = False)

