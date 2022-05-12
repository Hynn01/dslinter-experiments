#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# Hi
# 
# Thanks to all who upvoted for the gold medal! I really appriciate it.
# _____________________
# This is my first attempt at DL/ML of something other than images. Therefore, I wanted to start with well explored dataset that has a lot of good kernels, relatively small amount of data and this competiotion seems to suit it. This kernel is mostly a blend of top/highly voted kernels, my attemp at learning from them and hopefully implementing them in future competitions. This work is still in progress and I will use it as learning platform to DL/ML techniques
# 
# <br> Upvotes, notes and remarks are very appriciated!
# <brr>
# Special thanks to this kernel: 
#     <br> https://www.kaggle.com/alfredmaboa/advanced-regression-techniques-regularization/data
# 
#       
# Also learneda lot and used those kernels: 
# <br>
# Source1 : https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
# <br>
# Source2 : https://www.kaggle.com/bsivavenu/house-price-calculation-methods-for-beginners
# <br> 
# Source3 : https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# <br>
# Model source1 : https://www.kaggle.com/niteshx2/top-50-beginners-stacking-lgb-xgb
# 
# 
# ### Table of interest:
# > ### 1. Importing libraries
# > ### 2. Importing and inquiring data
# > > #### 2.1 Importing data
# > > #### 2.2 Quiring data
# > ### 3. The predicted variable - Sales price Skew & kurtosis analysis
# > > #### 3.1 Observing histogram
# > > #### 3.2 Tansforming log or box cox: 
# > ### 4. Missing data
# > > #### 4.1 Presenting and locating missing data
# > > #### 4.2 Replacing the missing data
# > > Examples: Replacing NaN with "No", "0" most common value and such
# > ### 5. Numerical and Categorial features
# > > #### 5.1 Splitting the data into categorial and numerical features
# > > #### 5.2 Box cox transform for skewd numerical data
# >  ### 6. Adding Features
# > > #### 6.1 Creating features from the data
# > > Adding features at this section to be able to view them at the visualization section next 
# > > #### 6.2 Deleting features
# > > Features that cant be skewd or unsignificant.
# > ### 7.Plotting the data
# > > #### 7.1 Visually comparing data to sale prices
# > > Boxplot for categorial features and 2D plot for numerical
# > > #### 7.2 Comparing data to sale price through correlation matrix
# > > #### 7.3 Pairplot for the most intresting parameters
# > ### 8. Preparing the data 
# > ### 9. Creating the model
# > > #### 9.1 Importing learning libraries
# > > #### 9.2 Defining folds and score functions
# > > #### 9.3 Defining models
# > > #### 9.4 Training the models
# > > #### 9.5 Validating each model
# > > #### 9.6 Blend model prediction
# > ### 10 Submission
# 

# ## 1. Importing libraries

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 2. Import And quiring data

# ### 2.1 Importing data
# Importing, dropping ID and some outliers based on future EDA

# In[ ]:


# Read files
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

# From EDA obvious outliers
train = train[train.GrLivArea < 4500]
train.reset_index(drop=True, inplace=True)

outliers = [30, 88, 462, 631, 1322]
train = train.drop(train.index[outliers])


print (train.columns)
print(test.columns)
print(train.shape,test.shape)


# ### 2.2 Quiring the data
# Just watching what's out there

# In[ ]:


train.describe()


# In[ ]:


train.head(7)


# ## 3. The predicted variable - Sales price Skew & kurtosis analysis
# The predicted variable is probably the most important variable, therefore it should be inspected throughly. 
# <br> It turns out models work better with symmetric gaussian distributions, therefore we want to get rid of the skewness by using log transformation. More on log transformation later
# <br> <br> Skew: 
# \begin{equation} 
# skew \left( X  \right) = E[ \frac{X-\mu}{\sigma} ]^3
# \end{equation} https://en.wikipedia.org/wiki/Skewness
# ![](https://www.managedfuturesinvesting.com/images/default-source/default-album/measure-of-skewness.jpg?sfvrsn=0)
#  
# <br> Kurtosis: $$kurtosis(X) = E[ (\frac{X-\mu}{\sigma})^4  ]$$
# https://en.wikipedia.org/wiki/Kurtosis
# ![](https://siemensplm.i.lithium.com/t5/image/serverpage/image-id/38460iB0F0D63C4F9B568A/image-size/large?v=1.0&px=999)

# ### 3.1 Observing Sale price histogram
# 

# In[ ]:


train['SalePrice'].describe()
sns.distplot(train['SalePrice']);
#skewness and kurtosis
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())


# ### 3.2 Tansforming: \begin{equation*} Y = log(1 + X)) \end{equation*}
# Should correct for skew.
# <br> A random example of a different log transformation
# ![](http://www.biostathandbook.com/pix/transformfig1.gif)

# In[ ]:


from scipy import stats
from scipy.stats import norm, skew #for some statistics

# Plot histogram and probability
fig = plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(train['SalePrice'] , fit=norm);
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.subplot(1,2,2)
res = stats.probplot(train['SalePrice'], plot=plt)
plt.suptitle('Before transformation')

# Apply transformation
train.SalePrice = np.log1p(train.SalePrice )
# New prediction
y_train = train.SalePrice.values
y_train_orig = train.SalePrice


# Plot histogram and probability after transformation
fig = plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(train['SalePrice'] , fit=norm);
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.subplot(1,2,2)
res = stats.probplot(train['SalePrice'], plot=plt)
plt.suptitle('After transformation')


# Concatenate train and test

# In[ ]:


# y_train_orig = train.SalePrice
# train.drop("SalePrice", axis = 1, inplace = True)
data_features = pd.concat((train, test)).reset_index(drop=True)
print(data_features.shape)

# print(train.SalePrice)


# ## 4. Missing data

# ### 4.1 Locating missing data
# 

# In[ ]:


# Missing data in train
data_features_na = data_features.isnull().sum()
data_features_na = data_features_na[data_features_na>0]
data_features_na.sort_values(ascending=False)


# In[ ]:


#missing data percent plot
total = data_features.isnull().sum().sort_values(ascending=False)
percent = (data_features.isnull().sum()/data_features.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# ### 4.2 Replacing the missing data
# 
# Correcting for the format, mostly filling NaN with "No" or "0"

# #### String Values
# For numbers that have no significance and should actually be strings

# In[ ]:


str_vars = ['MSSubClass','YrSold','MoSold']
for var in str_vars:
    data_features[var] = data_features[var].apply(str)


# Most common (frequent) string transform

# In[ ]:


# Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string

common_vars = ['Exterior1st','Exterior2nd','SaleType','Electrical','KitchenQual']
for var in common_vars:
    data_features[var] = data_features[var].fillna(data_features[var].mode()[0])
    
# 'RL' is by far the most common value. So we can fill in missing values with 'RL'
data_features['MSZoning'] = data_features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))


# Turn Nan to None 

# In[ ]:


# # data description says NA means "No Pool", majority of houses have no Pool at all in general.
# features[] = features["PoolQC"].fillna("None")
# Replacing missing data with None
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual',
            'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',"PoolQC"
           ,'Alley','Fence','MiscFeature','FireplaceQu','MasVnrType','Utilities']:
    data_features[col] = data_features[col].fillna('None')
# # For all these categorical basement-related features, NaN means that there is no basement
# for col in (:
#     features[col] = features[col].fillna('None')


# Fill numerical data - 0 or median

# In[ ]:


# Replacing missing data with 0 (Since No garage = no cars in such garage.)
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','MasVnrArea','BsmtFinSF1','BsmtFinSF2'
           ,'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BsmtUnfSF','TotalBsmtSF'):
    data_features[col] = data_features[col].fillna(0)

# group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
data_features['LotFrontage'] = data_features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
print('Features size:', data_features.shape)


# Uniqe

# In[ ]:


# data description says NA means typical
data_features['Functional'] = data_features['Functional'].fillna('Typ')


# In[ ]:


#missing data
total = data_features.isnull().sum().sort_values(ascending=False)
percent = (data_features.isnull().sum()/data_features.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)


# ## 5. Numerical and Categorial features

# ### 5.1 Splitting the data into categorial and numerical features

# In[ ]:


# Differentiate numerical features (minus the target) and categorical features
categorical_features = data_features.select_dtypes(include=['object']).columns
print(categorical_features)
numerical_features = data_features.select_dtypes(exclude = ["object"]).columns
print(numerical_features)

print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
feat_num = data_features[numerical_features]
feat_cat = data_features[categorical_features]


# In[ ]:


feat_num.head(10)


# In[ ]:


feat_cat.head(10)


# ### 5.2 Box cox transform for skewd numerical data
# Another transformation to reduce skew. 
# <br> Equation:
# ![](https://www.statisticshowto.datasciencecentral.com/wp-content/uploads/2015/07/boxcox-formula-1.png)
# <br> Transformation example:
# ![](https://www.itl.nist.gov/div898/handbook/eda/section3/gif/boxcox.gif)

# In[ ]:


# Plot skew value for each numerical value
from scipy.stats import skew 
skewness = feat_num.apply(lambda x: skew(x))
skewness.sort_values(ascending=False)


# Encode categorial features: can and should be replaced.

# In[ ]:


skewness = skewness[abs(skewness) > 0.5]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
print("Mean skewnees: {}".format(np.mean(skewness)))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    feat_num[feat] = boxcox1p(feat_num[feat], boxcox_normmax(feat_num[feat] + 1))
    data_features[feat] = boxcox1p(data_features[feat], boxcox_normmax(data_features[feat] + 1))
    
    
from scipy.stats import skew 
skewness.sort_values(ascending=False)


# Observe the correction. 
# We can see that a lot of parameters remained skewd. I suspect that's for variables that have a lot of 0. 

# In[ ]:


skewness = feat_num.apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]

print("There are {} skewed numerical features after Box Cox transform".format(skewness.shape[0]))
print("Mean skewnees: {}".format(np.mean(skewness)))
skewness.sort_values(ascending=False)


# ## 6. Adding features
# 

# ### 6.1 Creating features from the data
# Adding features at this section to be able to view them at the visualization section next 
# 

# In[ ]:


# Calculating totals before droping less significant columns

#  Adding total sqfootage feature 
data_features['TotalSF']=data_features['TotalBsmtSF'] + data_features['1stFlrSF'] + data_features['2ndFlrSF']
#  Adding total bathrooms feature
data_features['Total_Bathrooms'] = (data_features['FullBath'] + (0.5 * data_features['HalfBath']) +
                               data_features['BsmtFullBath'] + (0.5 * data_features['BsmtHalfBath']))
#  Adding total porch sqfootage feature
data_features['Total_porch_sf'] = (data_features['OpenPorchSF'] + data_features['3SsnPorch'] +
                              data_features['EnclosedPorch'] + data_features['ScreenPorch'] +
                              data_features['WoodDeckSF'])


# data_features['Super_quality'] = OverallQual * 
# vars = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'FullBath']


# ### 6.2 Deleting features
# Features that cant be skewd or are unsignificant.

# In[ ]:


data_features['haspool'] = data_features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
data_features['hasgarage'] = data_features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
data_features['hasbsmt'] = data_features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
data_features['hasfireplace'] = data_features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# Not normaly distributed can not be normalised and has no central tendecy
data_features = data_features.drop(['MasVnrArea', 'OpenPorchSF', 'WoodDeckSF', 'BsmtFinSF1','2ndFlrSF'], axis=1)
# data_features = data_features.drop(['MasVnrArea', 'OpenPorchSF', 'WoodDeckSF', 'BsmtFinSF1','2ndFlrSF',
#                          'PoolArea','3SsnPorch','LowQualFinSF','MiscVal','BsmtHalfBath','ScreenPorch',
#                          'ScreenPorch','KitchenAbvGr','BsmtFinSF2','EnclosedPorch','LotFrontage'
#                          ,'BsmtUnfSF','GarageYrBlt'], axis=1)

print('data_features size:', data_features.shape)


# ### 5.9 Splitting the data back to train and test
# 

# In[ ]:


train = data_features.iloc[:len(y_train), :]
test = data_features.iloc[len(y_train):, :]
print(['Train data shpe: ',train.shape,'Prediction on (Sales price) shape: ', y_train.shape,'Test shape: ', test.shape])


# ## 7.Plotting the data

# ### 7.1 Visually comparing data to sale prices
# One can observe the behaviour of the variables, locate outlier and more.

# In[ ]:


vars = data_features.columns
# vars = numerical_features
figures_per_time = 4
count = 0 
y = y_train
for var in vars:
    x = train[var]
#     print(y.shape,x.shape)
    plt.figure(count//figures_per_time,figsize=(25,5))
    plt.subplot(1,figures_per_time,np.mod(count,4)+1)
    plt.scatter(x, y);
    plt.title('f model: T= {}'.format(var))
    count+=1
    


# remove outliers

# In[ ]:


# Removes outliers 
# outliers = [30, 88, 462, 631, 1322]
# train = train.drop(train.index[outliers])
y_train = train['SalePrice']


# ### Optional: Box plot
# 
# Box plot is heavy, one can manualy choose the intresting parameters

# In[ ]:



# vars_box = ['OverallQual','YearBuilt','BedroomAbvGr']
vars_box = feat_cat
for var in vars_box:
    data = pd.concat([train['SalePrice'], train[var]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y="SalePrice", data=data)


# ### 7.2 Comparing data to sale price through correlation matrix

# Numerical values correlation matrix, to locate dependencies between different variables. 

# In[ ]:


# Complete numerical correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=1, square=True);


# #### Largest correlation with Sale Price
# Its important to remmber that this are 2D correlations, between sale price and another variable. When stacking all of the parameters the dependencies the picture gets more complex.

# In[ ]:


# saleprice correlation matrix
corr_num = 15 #number of variables for heatmap
cols_corr = corrmat.nlargest(corr_num, 'SalePrice')['SalePrice'].index
corr_mat_sales = np.corrcoef(train[cols_corr].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(12, 9))
hm = sns.heatmap(corr_mat_sales, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=cols_corr.values, xticklabels=cols_corr.values)
plt.show()


# ### 7.3 Pairplot for the most intresting parameters

# In[ ]:


# pair plots for variables with largest correlation
var_num = 8
vars = cols_corr[0:var_num]

sns.set()
sns.pairplot(train[vars], size = 2.5)
plt.show();


# ## 8. Preparing the data 
# Dropping Sale price, Creating dummy variable for the categorial variables and matching dimentions between train and test

# In[ ]:


data_features = data_features.drop("SalePrice", axis = 1)
final_features = pd.get_dummies(data_features)

print(final_features.shape)
X = final_features.iloc[:len(y), :]
X_test = final_features.iloc[len(y):, :]
X.shape, y_train.shape, X_test.shape


print(X.shape,y_train.shape,X_test.shape)


# Removing overfit

# In[ ]:


# Removes colums where the threshold of zero's is (> 99.95), means has only zero values 
overfit = []
for i in X.columns:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 99.95:
        overfit.append(i)

overfit = list(overfit)
overfit.append('MSZoning_C (all)')

X = X.drop(overfit, axis=1).copy()
X_test = X_test.drop(overfit, axis=1).copy()

print(X.shape,y_train.shape,X_test.shape)


# ## 9. Creating the model

# ### 9.1 Importing learning libraries

# In[ ]:


from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error , make_scorer
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ### 9.2 Defining folds and score functions

# In[ ]:


kfolds = KFold(n_splits=18, shuffle=True, random_state=42)

# model scoring and validation function
def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y,scoring="neg_mean_squared_error",cv=kfolds))
    return (rmse)

# rmsle scoring function
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# ### 9.3 Defining models

# In[ ]:


lightgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4, #was 3
                                       learning_rate=0.01, 
                                       n_estimators=9000, #8000
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.2, # 'was 0.2'
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       )

# xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,
#                                      max_depth=3, min_child_weight=0,
#                                      gamma=0, subsample=0.7,
#                                      colsample_bytree=0.7,
#                                      objective='reg:linear', nthread=-1,
#                                      scale_pos_weight=1, seed=27,
#                                      reg_alpha=0.00006)



# setup models hyperparameters using a pipline
# The purpose of the pipeline is to assemble several steps that can be cross-validated together, while setting different parameters.
# This is a range of values that the model considers each time in runs a CV
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]




# Kernel Ridge Regression : made robust to outliers
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))

# LASSO Regression : made robust to outliers
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, 
                    alphas=alphas2,random_state=42, cv=kfolds))

# Elastic Net Regression : made robust to outliers
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, 
                         alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))


stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, lightgbm),
                                meta_regressor=elasticnet,
                                use_features_in_secondary=True)


svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))

# store models, scores and prediction values 
models = {'Ridge': ridge,
          'Lasso': lasso, 
          'ElasticNet': elasticnet,
          'lightgbm': lightgbm,
          'Svd': svr}
#           'xgboost': xgboost}
predictions = {}
scores = {}


# ### 9.4 Training the models

# In[ ]:


for name, model in models.items():
    
    model.fit(X, y)
    predictions[name] = np.expm1(model.predict(X))
    
    score = cv_rmse(model, X=X)
    scores[name] = (score.mean(), score.std())


# ### 9.5 Validating and training each model

# In[ ]:


# get the performance of each model on training data(validation set)
print('---- Score with CV_RMSLE-----')
score = cv_rmse(ridge)
print("Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(lasso)
print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(elasticnet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(lightgbm)
print("lightgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(svr)
print("SVR: {:.4f} ({:.4f})".format(score.mean(), score.std()))
# scores['svr'] = (score.mean(), score.std())

# score = cv_rmse(xgboost)
# print("xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


#Fit the training data X, y
print('----START Fit----',datetime.now())
print('Elasticnet')
elastic_model = elasticnet.fit(X, y)
print('Lasso')
lasso_model = lasso.fit(X, y)
print('Ridge')
ridge_model = ridge.fit(X, y)
print('lightgbm')
lgb_model_full_data = lightgbm.fit(X, y)
print('Svr')
svr_model_full_data = svr.fit(X, y)

# print('xgboost')
# xgb_model_full_data = xgboost.fit(X, y)


print('stack_gen')
stack_gen_model = stack_gen.fit(np.array(X), np.array(y))


# ### 9.6 Blend model prediction

# In[ ]:


def blend_models_predict(X):
    return ((0.16  * elastic_model.predict(X)) +             (0.16 * lasso_model.predict(X)) +             (0.11 * ridge_model.predict(X)) +             (0.2 * lgb_model_full_data.predict(X)) +             (0.1 * svr_model_full_data.predict(X)) + #             (0.1 * xgb_model_full_data.predict(X)) + \
            (0.27 * stack_gen_model.predict(np.array(X))))


# In[ ]:


print('RMSLE score on train data:')
print(rmsle(y, blend_models_predict(X)))


# ## 10 Submission

# In[ ]:


print('Predict submission')
submission = pd.read_csv("../input/sample_submission.csv")
submission.iloc[:,1] = (np.expm1(blend_models_predict(X_test)))


# In[ ]:


# q1 = submission['SalePrice'].quantile(0.0042)
# q2 = submission['SalePrice'].quantile(0.99)
# # Quantiles helping us get some extreme values for extremely low or high values 
# submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
# submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
submission.to_csv("submission.csv", index=False)

