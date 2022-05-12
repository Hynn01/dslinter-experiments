#!/usr/bin/env python
# coding: utf-8

# We'll be trying to predict hause price with regression models. 
# 
# **Let's get started!**
# ## Check out the data
# We've been able to get some data from your neighbor for housing prices as a csv set, let's get our environment ready with the libraries we'll need and then import the data!
# ### Import Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import stats
from scipy.stats import norm, skew

import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


print(train.head())
print('**'* 50)
print(test.head())


# In[ ]:


print(train.info())
print('**'* 50)
print(test.info())


# What is the data trying to say to us ? We need to analyse the data. Analysing data is the most important thing to understand what the data is telling us. 

# Here's a brief version of what you'll find in the data description file.
# 
# * SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
# * MSSubClass: The building class
# * MSZoning: The general zoning classification
# * LotFrontage: Linear feet of street connected to property
# * LotArea: Lot size in square feet
# * Street: Type of road access
# * Alley: Type of alley access
# * LotShape: General shape of property
# * LandContour: Flatness of the property
# * Utilities: Type of utilities available
# * LotConfig: Lot configuration
# * LandSlope: Slope of property
# * Neighborhood: Physical locations within Ames city limits
# * Condition1: Proximity to main road or railroad
# * Condition2: Proximity to main road or railroad (if a second is present)
# * BldgType: Type of dwelling
# * HouseStyle: Style of dwelling
# * OverallQual: Overall material and finish quality
# * OverallCond: Overall condition rating
# * YearBuilt: Original construction date
# * YearRemodAdd: Remodel date
# * RoofStyle: Type of roof
# * RoofMatl: Roof material
# * Exterior1st: Exterior covering on house
# * Exterior2nd: Exterior covering on house (if more than one material)
# * MasVnrType: Masonry veneer type
# * MasVnrArea: Masonry veneer area in square feet
# * ExterQual: Exterior material quality
# * ExterCond: Present condition of the material on the exterior
# * Foundation: Type of foundation
# * BsmtQual: Height of the basement
# * BsmtCond: General condition of the basement
# * BsmtExposure: Walkout or garden level basement walls
# * BsmtFinType1: Quality of basement finished area
# * BsmtFinSF1: Type 1 finished square feet
# * BsmtFinType2: Quality of second finished area (if present)
# * BsmtFinSF2: Type 2 finished square feet
# * BsmtUnfSF: Unfinished square feet of basement area
# * TotalBsmtSF: Total square feet of basement area
# * Heating: Type of heating
# * HeatingQC: Heating quality and condition
# * CentralAir: Central air conditioning
# * Electrical: Electrical system
# * 1stFlrSF: First Floor square feet
# * 2ndFlrSF: Second floor square feet
# * LowQualFinSF: Low quality finished square feet (all floors)
# * GrLivArea: Above grade (ground) living area square feet
# * BsmtFullBath: Basement full bathrooms
# * BsmtHalfBath: Basement half bathrooms
# * FullBath: Full bathrooms above grade
# * HalfBath: Half baths above grade
# * Bedroom: Number of bedrooms above basement level
# * Kitchen: Number of kitchens
# * KitchenQual: Kitchen quality
# * TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# * Functional: Home functionality rating
# * Fireplaces: Number of fireplaces
# * FireplaceQu: Fireplace quality
# * GarageType: Garage location
# * GarageYrBlt: Year garage was built
# * GarageFinish: Interior finish of the garage
# * GarageCars: Size of garage in car capacity
# * GarageArea: Size of garage in square feet
# * GarageQual: Garage quality
# * GarageCond: Garage condition
# * PavedDrive: Paved driveway
# * WoodDeckSF: Wood deck area in square feet
# * OpenPorchSF: Open porch area in square feet
# * EnclosedPorch: Enclosed porch area in square feet
# * 3SsnPorch: Three season porch area in square feet
# * ScreenPorch: Screen porch area in square feet
# * PoolArea: Pool area in square feet
# * PoolQC: Pool quality
# * Fence: Fence quality
# * MiscFeature: Miscellaneous feature not covered in other categories
# * MiscVal: $Value of miscellaneous feature
# * MoSold: Month Sold
# * YrSold: Year Sold
# * SaleType: Type of sale
# * SaleCondition: Condition of sale

# We are going to convert train ad test data. And We are going to drop SalePrice column for predict.

# # Data Visualization

# Let's look at the point of visualization

# In[ ]:


sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show() 


# As you see the sale price value is right skewed. We need to make this normal distributed.

# In[ ]:


plt.figure(figsize=(30,8))
sns.heatmap(train.corr(),cmap='coolwarm',annot = True)
plt.show()


# we can see the most corelated parameters in numerical values above plotting. And we can pick these as features for our macine learning model.

# In[ ]:


corr = train.corr()


# In[ ]:


corr[corr['SalePrice']>0.3].index


# In[ ]:


train.info()


# In[ ]:


train = train[['LotFrontage', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
       'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
       'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
       'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'SalePrice']]
test=test[['LotFrontage', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
       'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
       'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
       'GarageArea', 'WoodDeckSF', 'OpenPorchSF']]


# We droped some columns that less than 0.3 of correlation of Sale Prices.

# In[ ]:


sns.lmplot(x='1stFlrSF',y='SalePrice',data=train) # 1stFlrSF seems very corelated with SalePrice.


# In[ ]:


plt.scatter(x= 'GrLivArea', y='SalePrice', data = train)


# In[ ]:


plt.figure(figsize=(16,8))
sns.boxplot(x='GarageCars',y='SalePrice',data=train)
plt.show()


# In[ ]:


sns.lmplot(x='OverallQual',y='SalePrice',data=train)


# In[ ]:


sns.lmplot(x='GarageArea',y='SalePrice',data=train)


# In[ ]:


plt.figure(figsize=(16,8))
sns.barplot(x='FullBath',y = 'SalePrice',data=train)
plt.show()


# # Feature Engineering

# We have to convert all columns into numeric or categorical data.

# In[ ]:


#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(25)


# As we can see some of paremeters have a lot of missing values. That's why we should drop these from data. And we are going to drop parematers which total value is larger than 81.

# In[ ]:


#dealing with missing data
train = train.drop((missing_data[missing_data['Total'] > 81]).index,1)


# In[ ]:


train.isnull().sum().sort_values(ascending=False).head(20)


# We are going to do same thing to the test data

# In[ ]:


#missing data
total_test = test.isnull().sum().sort_values(ascending=False)
percent_test = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total_test, percent_test], axis=1, keys=['Total', 'Percent'])
missing_data.head(25)


# In[ ]:


#dealing with missing data
test = test.drop((missing_data[missing_data['Total'] > 78]).index,1)


# In[ ]:


test.isnull().sum().sort_values(ascending=False).head(20)


# In[ ]:


train.isnull().sum().sort_values(ascending = False).head(20)


# we need to handle missing data.

# In[ ]:


# Categorical boolean mask
categorical_feature_mask = train.dtypes==object
# filter categorical columns using mask and turn it into alist
categorical_cols = train.columns[categorical_feature_mask].tolist()


# In[ ]:


categorical_cols


# In[ ]:


#data = pd.get_dummies(data, columns=categorical_cols)


# In[ ]:


#from sklearn.preprocessing import LabelEncoder
#labelencoder = LabelEncoder()
#train[categorical_cols] = train[categorical_cols].apply(lambda col: labelencoder.fit_transform(col.astype(str)))


# In[ ]:


# Categorical boolean mask
categorical_feature_mask_test = test.dtypes==object
# filter categorical columns using mask and turn it into alist
categorical_cols_test = test.columns[categorical_feature_mask_test].tolist()


# In[ ]:


#from sklearn.preprocessing import LabelEncoder
#labelencoder = LabelEncoder()
#test[categorical_cols_test] = test[categorical_cols_test].apply(lambda col: labelencoder.fit_transform(col.astype(str)))


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.isnull().sum().sort_values(ascending=False).head(20)


# In[ ]:


test.isnull().sum().sort_values(ascending=False).head(20)


# In[ ]:


train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())


# Now we are going to pick some features for the model. For this we are going to use correlation matrix and we are going to pick most correlated with sale price.

# In[ ]:


#saleprice correlation matrix
k = 15 #number of variables for heatmap
plt.figure(figsize=(16,8))
corrmat = train.corr()
# picking the top 15 correlated features
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:


train = train[cols]


# In[ ]:


cols


# In[ ]:


test=test[cols.drop('SalePrice')]


# In[ ]:


test.isnull().sum().sort_values(ascending=False).head(20)


# In[ ]:


test.head()


# In[ ]:


test['GarageYrBlt'] = test['GarageYrBlt'].fillna(test['GarageYrBlt'].mean())
test['MasVnrArea'] = test['MasVnrArea'].fillna(test['MasVnrArea'].mean())
test['GarageCars'] = test['GarageCars'].fillna(test['GarageCars'].mean())
test['GarageArea'] = test['GarageArea'].fillna(test['GarageArea'].mean())
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean())
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean())


# <a id="1"></a> <br>
# # **Linear Regression **
# 
# Let's now begin to train out regression model! We will need to first split up our data into an X array that contains the features to train on, and a y array with the target variable, in this case the Price column. We will toss out the Address column because it only has text info that the linear regression model can't use.

# <a id="1"></a> <br>
# ### **Train Test Split **
# Now let's split the data into a training set and a testing set. We will train out model on the training set and then use the test set to evaluate the model.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('SalePrice', axis=1), train['SalePrice'], test_size=0.3, random_state=101)


# In[ ]:


# we are going to scale to data

y_train= y_train.values.reshape(-1,1)
y_test= y_test.values.reshape(-1,1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
y_train = sc_X.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)


# In[ ]:


X_train


# <a id="1"></a> <br>
# ### **Creating and Training the Model **

# In[ ]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# In[ ]:


lm.fit(X_train,y_train)
print(lm)


# <a id="1"></a> <br>
# ### **Model Evaluation **
# Let's evaluate the model by checking out it's coefficients and how we can interpret them.

# In[ ]:


# print the intercept
print(lm.intercept_)


# In[ ]:


print(lm.coef_)


# <a id="1"></a> <br>
# ### **Predictions from our Model **
# Let's grab predictions off our test set and see how well it did!

# In[ ]:


predictions = lm.predict(X_test)
predictions= predictions.reshape(-1,1)


# In[ ]:


plt.figure(figsize=(15,8))
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


# In[ ]:


plt.figure(figsize=(16,8))
plt.plot(y_test,label ='Test')
plt.plot(predictions, label = 'predict')
plt.show()


# <a id="1"></a> <br>
# ### **Regression Evaluation Metrics**
# 
# Here are three common evaluation metrics for regression problems:
# 
# **Mean Absolute Error (MAE)** is the mean of the absolute value of the errors:
# 
# 1𝑛∑𝑖=1𝑛|𝑦𝑖−𝑦̂ 𝑖|
#  
# **Mean Squared Error (MSE)** is the mean of the squared errors:
# 
# 1𝑛∑𝑖=1𝑛(𝑦𝑖−𝑦̂ 𝑖)2
#  
#  
# **Root Mean Squared Error (RMSE)** is the square root of the mean of the squared errors:
# 
# 1𝑛∑𝑖=1𝑛(𝑦𝑖−𝑦̂ 𝑖)
#  
# **Comparing these metrics**:
# 
# MAE is the easiest to understand, because it's the average error.
# MSE is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
# RMSE is even more popular than MSE, because RMSE is interpretable in the "y" units.
# All of these are loss functions, because we want to minimize them.

# In[ ]:


from sklearn import metrics


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# <a id="1"></a> <br>
# # **Gradient Boosting Regression **
# 
# Gradient Boosting trains many models in a gradual, additive and sequential manner. The major difference between AdaBoost and Gradient Boosting Algorithm is how the two algorithms identify the shortcomings of weak learners (eg. decision trees). While the AdaBoost model identifies the shortcomings by using high weight data points, gradient boosting performs the same by using gradients in the loss function (y=ax+b+e , e needs a special mention as it is the error term). The loss function is a measure indicating how good are model’s coefficients are at fitting the underlying data. A logical understanding of loss function would depend on what we are trying to optimise. We are trying to predict the sales prices by using a regression, then the loss function would be based off the error between true and predicted house prices.

# In[ ]:


from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score


# In[ ]:


params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.05, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)


# In[ ]:


clf_pred=clf.predict(X_test)
clf_pred= clf_pred.reshape(-1,1)


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, clf_pred))
print('MSE:', metrics.mean_squared_error(y_test, clf_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, clf_pred)))


# In[ ]:


plt.figure(figsize=(15,8))
plt.scatter(y_test,clf_pred, c= 'brown')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


# In[ ]:


plt.figure(figsize=(16,8))
plt.plot(y_test,label ='Test')
plt.plot(clf_pred, label = 'predict')
plt.show()


# <a id="1"></a> <br>
# # **Decision Tree Regression **

# 
# The decision tree is a simple machine learning model for getting started with regression tasks.
# 
# **Background**
# A decision tree is a flow-chart-like structure, where each internal (non-leaf) node denotes a test on an attribute, each branch represents the outcome of a test, and each leaf (or terminal) node holds a class label. The topmost node in a tree is the root node. (see here for more details).
# 
# 

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dtreg = DecisionTreeRegressor(random_state = 100)
dtreg.fit(X_train, y_train)


# In[ ]:


dtr_pred = dtreg.predict(X_test)
dtr_pred= dtr_pred.reshape(-1,1)


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, dtr_pred))
print('MSE:', metrics.mean_squared_error(y_test, dtr_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dtr_pred)))


# In[ ]:


plt.figure(figsize=(15,8))
plt.scatter(y_test,dtr_pred,c='green')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


# <a id="1"></a> <br>
# # **Support Vector Machine Regression **

# Support Vector Machine can also be used as a regression method, maintaining all the main features that characterize the algorithm (maximal margin). The Support Vector Regression (SVR) uses the same principles as the SVM for classification, with only a few minor differences. First of all, because output is a real number it becomes very difficult to predict the information at hand, which has infinite possibilities. In the case of regression, a margin of tolerance (epsilon) is set in approximation to the SVM which would have already requested from the problem. But besides this fact, there is also a more complicated reason, the algorithm is more complicated therefore to be taken in consideration. However, the main idea is always the same: to minimize error, individualizing the hyperplane which maximizes the margin, keeping in mind that part of the error is tolerated.
# 
# 
# ![](https://www.saedsayad.com/images/SVR_1.png)
# ![](https://www.saedsayad.com/images/SVR_2.png)
# ***Linear SVR***
#                                     ![](https://www.saedsayad.com/images/SVR_4.png)                     
#                                     
#                                     
#                                     
#                                     
# ***Non Linear SVR***
# 
# ![](https://www.saedsayad.com/images/SVR_6.png)
# ![](https://www.saedsayad.com/images/SVR_5.png)

# In[ ]:


from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(X_train, y_train)


# In[ ]:


svr_pred = svr.predict(X_test)
svr_pred= svr_pred.reshape(-1,1)


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, svr_pred))
print('MSE:', metrics.mean_squared_error(y_test, svr_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, svr_pred)))


# In[ ]:


plt.figure(figsize=(15,8))
plt.scatter(y_test,svr_pred, c='red')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


# In[ ]:


plt.figure(figsize=(16,8))
plt.plot(y_test,label ='Test')
plt.plot(svr_pred, label = 'predict')
plt.show()


# <a id="1"></a> <br>
# # **Random Forest Regression **

# A Random Forest is an ensemble technique capable of performing both regression and classification tasks with the use of multiple decision trees and a technique called Bootstrap Aggregation, commonly known as bagging. What is bagging you may ask? Bagging, in the Random Forest method, involves training each decision tree on a different data sample where sampling is done with replacement.
# 
# ![](https://cdn-images-1.medium.com/max/800/1*jEGFJCm4VSG0OzoqFUQJQg.jpeg)

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 100, random_state = 0)
rfr.fit(X_train, y_train)


# In[ ]:


rfr_pred= rfr.predict(X_test)
rfr_pred = rfr_pred.reshape(-1,1)


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, rfr_pred))
print('MSE:', metrics.mean_squared_error(y_test, rfr_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfr_pred)))


# In[ ]:


plt.figure(figsize=(15,8))
plt.scatter(y_test,rfr_pred, c='orange')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


# In[ ]:


plt.figure(figsize=(16,8))
plt.plot(y_test,label ='Test')
plt.plot(rfr_pred, label = 'predict')
plt.show()


# # LightGBM

# In[ ]:


import lightgbm as lgb


# In[ ]:


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.1, n_estimators=500,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# In[ ]:


model_lgb.fit(X_train,y_train)


# In[ ]:


lgb_pred = model_lgb.predict(X_test)
lgb_pred = lgb_pred.reshape(-1,1)


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, lgb_pred))
print('MSE:', metrics.mean_squared_error(y_test, lgb_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lgb_pred)))


# In[ ]:


plt.figure(figsize=(15,8))
plt.scatter(y_test,lgb_pred, c='orange')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


# In[ ]:


plt.figure(figsize=(16,8))
plt.plot(y_test,label ='Test')
plt.plot(lgb_pred, label = 'predict')
plt.show()


# # Model Comparison

# **We can say the best working model by loking MSE rates The best working model is Support Vector Machine.**
# We are going to see the error rate. which one is better?
# 

# In[ ]:


error_rate=np.array([metrics.mean_squared_error(y_test, predictions),metrics.mean_squared_error(y_test, clf_pred),metrics.mean_squared_error(y_test, dtr_pred),metrics.mean_squared_error(y_test, svr_pred),metrics.mean_squared_error(y_test, rfr_pred)])


# In[ ]:


plt.figure(figsize=(16,5))
plt.plot(error_rate)


# Now we will use test data .

# In[ ]:


a = pd.read_csv('../input/test.csv')


# In[ ]:


test_id = a['Id']
a = pd.DataFrame(test_id, columns=['Id'])


# In[ ]:


test = sc_X.fit_transform(test)


# In[ ]:


test.shape


# In[ ]:


test_prediction_lgbm=model_lgb.predict(test)
test_prediction_lgbm= test_prediction_lgbm.reshape(-1,1)


# In[ ]:


test_prediction_lgbm


# In[ ]:


test_prediction_lgbm =sc_y.inverse_transform(test_prediction_lgbm)


# In[ ]:


test_prediction_lgbm = pd.DataFrame(test_prediction_lgbm, columns=['SalePrice'])


# In[ ]:


test_prediction_lgbm.head()


# In[ ]:


result = pd.concat([a,test_prediction_lgbm], axis=1)


# In[ ]:


result.head()


# In[ ]:


result.to_csv('submission.csv',index=False)


# If you like it, please vote :)
