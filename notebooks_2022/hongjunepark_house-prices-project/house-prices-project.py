#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # 1. House Prices?
# 
# House Prices Project is one of well known datasets. The target is to predict a house price with many explanatory variables, such as LotArea, YearBuilt, and so on. Basically, the dataset is made of train and test data sets. With 79 variables, I have to build my model to forecast individual house price and submit my prediction over test dataset. Let's start!
# 
# First of all, I import the data set. Let's look at basic stats.

# In[ ]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')

train.info()
train.describe().T


# # 2. First Model
# 
# At first, let's build basic model and get RMSE value. RMSE (Root Mean Square Error) will be my criteria, which will determine whether my treatment is valuable. If RMSE value gets higher after I make a adjustment, I have to drop the decision.

# In[ ]:


from sklearn.model_selection import train_test_split

#print(train.info())
y_1 = train["SalePrice"]
train_1 = train.copy()
train_1.drop(["SalePrice"], axis=1, inplace=True)

X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(train_1, y_1,
                                              train_size=0.8, test_size=0.2,random_state=0)
print("\nX_train_1 shape:",X_train_1.shape)
print("\nX_valid_1 shape:",X_valid_1.shape)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def rmse_score(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    mse = mean_squared_error(y_valid, preds)
    return np.sqrt(mse)


# To obtain my first RMSE, I have to delete categorical columns. Unlike R, python panda has no tools for categorical data. For none numberic data, I have some options, but to start easily, at first I get my first RMSE without categorical data.

# In[ ]:


X_train_1 = X_train_1.select_dtypes(exclude=['object'])
X_valid_1 = X_valid_1.select_dtypes(exclude=['object'])

#X_train_1


# Also, I have to check whether there is any missing values. If there is any missing column, I have to consider an optional way to deal with.

# In[ ]:


missing_val_count_by_column = (X_train_1.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


# After the check, I conclude that there are 3 columns with missing values. So, I have to consider a way - imputation. I will use "simple Imputer", which will fill missing values with the mean value along each column.

# In[ ]:


from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer() 
X_train_1 =  pd.DataFrame(my_imputer.fit_transform(X_train_1))
X_valid_1 =  pd.DataFrame(my_imputer.transform(X_valid_1))

#X_train_1
#X_valid_1


# In[ ]:


print("1st RMSE Score:",rmse_score(X_train_1, X_valid_1, y_train_1, y_valid_1))


# # 3. Dependent Variable
# 
# In my project, dependent variable is "SalePrice". The goal of this project is to predict the sale price of a housewith various data features. Let's check the "SalePrice" variable.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(y_1,kde=True)
plt.show()
print("SalePrice's Mean :",y_1.mean())
print("SalePrice's Median :",y_1.median())
print("SalePrice's Mode :",y_1.mode(dropna=True))
#


# I think dependent variable, 'SalePrice' is right skewed. Mean value is much bigger thant mode, (180,921 > 140,000). Therefore I need some scaling process over 'SalePrice' variable. I will apply log transformation. Let's build model with transformation of y variable.

# In[ ]:


y_2 = np.log1p(y_1)
sns.histplot(y_2,kde=True)
plt.show()
print("Transformed SalePrice's Mean :",y_2.mean())
print("Transformed SalePrice's Median :",y_2.median())
print("Transformed SalePrice's Mode :",y_2.mode(dropna=True))


# After log transforamtion, mean, median, and mode value of "SalePrice" is close and the shape is much more like normal distribution.

# In[ ]:


# No change for X variables
X_train_2 = X_train_1 
X_valid_2 = X_valid_1 
# log transformation for y variable
y_train_2 = np.log1p(y_train_1)
y_valid_2 = np.log1p(y_valid_1)

print(X_train_2.shape)
print(X_valid_2.shape)
print(y_train_2.shape)
print(y_valid_2.shape)


# In[ ]:


print("2nd RMSE Score:",rmse_score(X_train_2, X_valid_2, y_train_2, y_valid_2))


# # 4. Explanatory Variables
# 
# In my project, the dataset has total 79 independent variables. At first, I think categorical data has to be dealt with. Let's check how many non-numberic variables and null values.

# In[ ]:


print(train.dtypes.value_counts(),'\n\n')

for i in train.select_dtypes("object"):
    if(train[i].isnull().sum()>0):
        print(i,train[i].isnull().sum())    
    
print("Total number of train dataset:",train.shape[0])


# After reviewing null values, I think that some features should be removed. For "MiscFeature" variable, 1,406 values are null. Considering that total count of train set is 1,460, I conclude that too many null value will not make any difference over my research. Therefore I have to delete some features, such as "MiscFeature", "PoolQC", "Alley" and "Fence". Let's move the next step without those four variables.  

# In[ ]:


drop_columns = ['MiscFeature','PoolQC','Alley','Fence']
train_3 = train.drop(drop_columns,axis=1).copy()
print(train_3.shape)


# # 5. One-hot encoding for categorical data
# 
# Of techniques to deal with categorical data, one-hot encoding is my choice. If I need to improve my model, I will consider other options, ordinal encoding. At first, I will start with one-hot encoding.

# In[ ]:


y_3 = y_2
train_3.drop(["SalePrice"], axis=1, inplace=True)

X_train_3, X_valid_3, y_train_3, y_valid_3 = train_test_split(train_3, y_3,
                                       train_size=0.8, test_size=0.2, random_state=0)

print(X_train_3.shape)
print(X_valid_3.shape)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

object_cols = [col for col in X_train_3.columns if X_train_3[col].dtype == "object"]
numberic_cols = [col for col in X_train_3.columns if X_train_3[col].dtype != "object"]

# imputer : fill missing value with mean of the column
my_imputer_2 = SimpleImputer()
X_train_3[numberic_cols] = my_imputer_2.fit_transform(X_train_3[numberic_cols])
X_valid_3[numberic_cols] = my_imputer_2.transform(X_valid_3[numberic_cols])

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train_3[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid_3[object_cols]))

OH_cols_train.index = X_train_3.index
OH_cols_valid.index = X_valid_3.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train_3.drop(object_cols, axis=1)
num_X_valid = X_valid_3.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

X_train_3 = OH_X_train
X_valid_3 = OH_X_valid

print("\nX_train_3 shape:",X_train_3.shape)
print("\nX_valid_3 shape:",X_valid_3.shape)


# In[ ]:


# RMSE with log-transformed y and one-hot Encoding
print("3rd RMSE Score:",rmse_score(X_train_3, X_valid_3, y_train_3, y_valid_3))


# Comparing between 2nd and 3rd RMSE, I will use 3rd RMSE as benchmark value. Because 2nd RMSE is calculated without any categorical variables, 3rd RMSE is a much better value, which is based on the model with none-numberic features.

# # 6. Numberic Variables - Skewness
# 
# Let's check number variables.

# In[ ]:


print(train.dtypes.value_counts(),'\n\n')
from scipy.stats import skew

features_index = train.dtypes[train.dtypes != 'object'].index
skew_values = train[features_index].apply(lambda x : skew(x))
skew_values


# After reviewing skew values, I think some variables - MiscVal, LotArea, PoolArea - has huge positive skewness values. There are some negative values, but I think they are okay. I have to do something for some big positive-valued variables which are right skewed.

# In[ ]:


skew_value_more_than_3 = skew_values[skew_values >= 4]
print(skew_value_more_than_3.sort_values(ascending=False))

skewed_variables = skew_value_more_than_3.index
print('\n\n',skewed_variables)

X_train_4 = X_train_3.copy()
X_valid_4 = X_valid_3.copy()
y_train_4 = y_train_3.copy()
y_valid_4 = y_valid_3.copy()

X_train_4[skewed_variables] = np.log1p(X_train_4[skewed_variables])
X_valid_4[skewed_variables] = np.log1p(X_valid_4[skewed_variables])


# So, I think that 9 variables - MiscVal, PoolArea, LotArea, and so on - are right skewed so try to fix the problem with log transformation. Let's see what happen.

# In[ ]:


# RMSE : log-transformed y & one-hot Encoding & log-transformed skewed variables
print("4th RMSE Score:",rmse_score(X_train_4, X_valid_4, y_train_4, y_valid_4))


# Comparing with 3rd RMSE value, 0.138766, I think that 4th RMSE value, 0.138814 is slight high. I think, however, that the difference is quite small. So I will stick to this transformation. I hope the log transformation can improve my model.

# # 7. Numberic Variables - Outliers
# 
# Let's check scatter plot and outliers of number variables.

# In[ ]:


X_train_5 = X_train_4.copy()
X_valid_5 = X_valid_4.copy()
y_train_5 = y_train_4.copy()
y_valid_5 = y_valid_4.copy()

#for i in X_train_5[numberic_cols]:
#    plt.scatter(x = X_train_5[i], y = y_train_5)
#    plt.ylabel('SalePrice', fontsize=15)
#    plt.xlabel(i, fontsize=15)
#    plt.show()


# At first, let's look over "LotFrontage" variable.

# In[ ]:


plt.scatter(x = X_train_5["LotFrontage"], y = y_train_5)
plt.ylabel('SalePrice', fontsize=15)
plt.xlabel("LotFrontage", fontsize=15)
plt.show()
print(y_train_5[(X_train_5.LotFrontage>=150)])


# I think that "index number 1338" item should be removed. I have few domain knowledge about property market, but I think the bigger lot frontage the higher sale price.
# 
# Let's move to "GrLivArea" variable.

# In[ ]:


plt.scatter(x = X_train_5["GrLivArea"], y = y_train_5)
plt.ylabel('SalePrice', fontsize=15)
plt.xlabel("GrLivArea", fontsize=15)
plt.show()
print(y_train_5[(X_train_5.GrLivArea>=4000)])


# I think that "index number 524" item should be removed, because I think the bigger Ground Living Area the higher sale price.
# 
# Let's move to "OpenPorchSF" variable.

# In[ ]:


plt.scatter(x = X_train_5["OpenPorchSF"], y = y_train_5)
plt.ylabel('SalePrice', fontsize=15)
plt.xlabel("OpenPorchSF", fontsize=15)
plt.show()
print(y_train_5[(X_train_5.OpenPorchSF>=500)])


# I think that "index number 496" item should be removed, because I think the open porch area the higher sale price.

# In[ ]:


#for i in train.select_dtypes("object"):
#    sns.catplot(x=i, y='SalePrice', data=train)


# Finally, through checking outlier process, I will delete total 3 items - 1338, 524, and 496 by index number. I remove those data to improve my model. Let's compare RMSE numbers!

# In[ ]:


#X_train_5 = X_train_4.copy()
#X_valid_5 = X_valid_4.copy()
#y_train_5 = y_train_4.copy()
#y_valid_5 = y_valid_4.copy()

print(X_train_5.shape)
print(y_train_5.shape)

drop_index = [1338,524,496]
X_train_5.drop(drop_index,axis=0,inplace=True)
y_train_5.drop(drop_index,axis=0,inplace=True)

print(X_train_5.shape)
print(y_train_5.shape)


# In[ ]:


# log-trans Y & one-hot encoding & log-trans X's & deleting outliers
print("5th RMSE Score:",rmse_score(X_train_5, X_valid_5, y_train_5, y_valid_5))


# After deleting some outliers, I did not make any improvement RMSE 0.1435 - from previous 0.1388 (Model 4). In this case, I have to consider more carefully deleting outliers.

# # 8. Final Model - 1
# 
# I will prepare for my final model to submission.

# In[ ]:


#model = RandomForestRegressor(n_estimators=100, random_state=0)
#model.fit(X_train_4, y_train_4)

#X_test = test.drop(drop_columns,axis=1)
#print(X_test.shape)

#X_test[numberic_cols] = my_imputer_2.transform(X_test[numberic_cols])
#OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[object_cols]))
#OH_cols_test.index = X_test.index
#num_X_test = X_test.drop(object_cols, axis=1)

#X_test_final = pd.concat([num_X_test, OH_cols_test], axis=1)
#preds_test = model.predict(X_test_final)


# In[ ]:


#print(preds_test.shape)
#test_result = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
#output = pd.DataFrame({'Id': test_result.Id,'SalePrice': preds_test})
#output.to_csv('submission.csv', index=False)


# # 9. Regularization - Rigde & Lasso Regression
# 
# Ridge Regression was introduced to deal with the case in which linearly independent variables are highly correlated. I think regularization method can be applied to this project using trade-off between variance and bias. So, I will try Ridge Regression.
# 

# In[ ]:


from sklearn.linear_model import Ridge

# To compare with the best model's value, I use X_train_4 data set
ridge = Ridge()
ridge.fit(X_train_4, y_train_4)
pred_ridge = ridge.predict(X_valid_4)
mse_ridge = mean_squared_error(y_valid_4 , pred_ridge)
rmse_rideg = np.sqrt(mse_ridge)

print("6th RMSE Score:",rmse_rideg)


# In[ ]:


from sklearn.model_selection import GridSearchCV

params_ridge =  {'alpha':[0.05,0.1,0,5,7,10,15,20,30,50]}
model_ridge_params= GridSearchCV(ridge,param_grid=params_ridge,scoring='neg_mean_squared_error')
model_ridge_params.fit(X_train_4, y_train_4)
print('Best Score:',model_ridge_params.best_score_)
print('Best Parameter value:',model_ridge_params.best_params_)


# In[ ]:


ridge_1 = Ridge(alpha=10)
ridge_1.fit(X_train_4, y_train_4)
pred_ridge_1 = ridge_1.predict(X_valid_4)
mse_ridge_1 = mean_squared_error(y_valid_4 , pred_ridge)
rmse_rideg_1 = np.sqrt(mse_ridge_1)

print("6th RMSE Score:",rmse_rideg_1)


# After applying Ridge Regression, I get RMSE value, 0.2001, bigger than Model 4 value, 0.1388. At first, I tried with default parameters. Next, I try to find out the proper regularization strength with "GridSearchCV". After that, I tried once more ridge regression with alpha value, 10. The result, however, is almost the same.
# 
# Let's try another Regularization method, Lasso Regression.

# In[ ]:


from sklearn.linear_model import Lasso

# To compare with the best model's value, I use X_train_4 data set
lasso = Lasso()
lasso.fit(X_train_4, y_train_4)
pred_lasso = lasso.predict(X_valid_4)
mse_lasso = mean_squared_error(y_valid_4 , pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)

print("7th RMSE Score:",rmse_lasso)


# In[ ]:


params_lasso =  {'alpha':[0.001, 0.005, 0.01, 0.05, 0.03, 0.1, 0.5, 1]}
model_lasso_params= GridSearchCV(lasso,param_grid=params_lasso,scoring='neg_mean_squared_error')
model_lasso_params.fit(X_train_4, y_train_4)
print('Best Score:',model_lasso_params.best_score_)
print('Best Parameter value:',model_lasso_params.best_params_)


# In[ ]:


lasso = Lasso(alpha=0.001)
lasso.fit(X_train_4, y_train_4)
pred_lasso = lasso.predict(X_valid_4)
mse_lasso = mean_squared_error(y_valid_4 , pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)

print("7th RMSE Score:",rmse_lasso)


# Final Lasso Regression's score is 0.1935. This value is still bigger than Model 4's value, 0.1388. 

# # 10. Decision Tree - Gradient Boosting Regressor
# 
# This time, I will try a new regression model - Gradient Boosting Regressor.

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor

#Let's start with default option
gbr=GradientBoostingRegressor(learning_rate= 0.1, max_depth= 3, n_estimators= 100)
gbr.fit(X_train_4, y_train_4)
pred_gbr = gbr.predict(X_valid_4)
mse_gbr = mean_squared_error(y_valid_4 , pred_gbr)
rmse_gbr = np.sqrt(mse_gbr)

print("8th RMSE Score:",rmse_gbr)


# In[ ]:


# Let's find out best parameters
params_gbr =  {'learning_rate':[0.1],
               'max_depth':[2,3],
               'n_estimators':[100,200,300,500,1000]}
model_gbr_params= GridSearchCV(gbr,param_grid=params_gbr,scoring='neg_mean_squared_error')
model_gbr_params.fit(X_train_4, y_train_4)
print('Best Score:',model_gbr_params.best_score_)
print('Best Parameter value:',model_gbr_params.best_params_)


# In[ ]:


gbr_1=GradientBoostingRegressor(learning_rate= 0.1, max_depth= 2, n_estimators= 500)
gbr_1.fit(X_train_4, y_train_4)
pred_gbr_1 = gbr_1.predict(X_valid_4)
mse_gbr_1 = mean_squared_error(y_valid_4 , pred_gbr_1)
rmse_gbr_1 = np.sqrt(mse_gbr_1)

print("8th RMSE Score:",rmse_gbr_1)


# At first, I applied Gradient Boosting Regressor(GBR) with default parameters. Next, with GridSearchCV, I found out optimal parameter values. Finally my RMSE score with GBR is much lower than Model 4. 0.1254 < 0.1388.

# # 11. Final Model - 2
# 
# Gradient Boosting Regressor(GBR) is my final model.

# In[ ]:


model = GradientBoostingRegressor(learning_rate= 0.1, max_depth= 2, n_estimators= 500)
model.fit(X_train_4, y_train_4)

X_test = test.drop(drop_columns,axis=1)
print(X_test.shape)

X_test[numberic_cols] = my_imputer_2.transform(X_test[numberic_cols])
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[object_cols]))
OH_cols_test.index = X_test.index
num_X_test = X_test.drop(object_cols, axis=1)

X_test_final = pd.concat([num_X_test, OH_cols_test], axis=1)
preds_test = model.predict(X_test_final)


# In[ ]:


print(preds_test.shape)
test_result = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
output = pd.DataFrame({'Id': test_result.Id,'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

