#!/usr/bin/env python
# coding: utf-8

# ## Intro

# we can work in training file to build model to predict price of house, <br>
# and we use this model to predict price in test file.<br>
# 
# we use description.txt file to understand columns and range of effect in price him.

# ## import Tools(Toolkit)

# In[ ]:


# manipulation libs
import pandas as pd
import numpy as np


# In[ ]:


# visuals Libs
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# model Libs
from sklearn.metrics import mean_squared_error,mean_absolute_error,median_absolute_error
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,VotingRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


# In[ ]:


# warnings Libs
import warnings
warnings.filterwarnings('ignore')


# ## Load DataSet

# In[ ]:


train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
submit=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')


# In[ ]:


train.head()


# ## Exploratory dataset

# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


submit.shape


# In[ ]:


train_null=train.isnull().sum()
train_null=pd.DataFrame(train_null,columns=['number']).sort_values(by='number',ascending=False)
train_null.reset_index(inplace=True)
train_null.head(20)


# In[ ]:


test_null=test.isnull().sum()
test_null=pd.DataFrame(test_null,columns=['number']).sort_values(by='number',ascending=False)
test_null.reset_index(inplace=True)
test_null.head(34)


# In[ ]:


# array of train null value
train_null_array=np.array(train_null.iloc[:19]['index'])
train_null_array


# In[ ]:


# array of test null value
test_null_array=np.array(test_null.iloc[:33]['index'])
test_null_array


# we split array has null values to fill it by 3 section, <br>
# first section :- fill numerical value by median / mean.<br>
# second section :- fill object value by None without null value. <br>
# third section :- fill object value by mode.<br>
# 
# we must read description file to know columns which Null is part of values

# In[ ]:


train.columns


# In[ ]:


object_columns=train.select_dtypes(include='object').columns


# In[ ]:


object_columns


# In[ ]:


for i in object_columns:
    print(i)
    print(train[i].value_counts())


# we saw there are some of columns it effect in price and part of this not effect,
# so we should use feature selection to choose columns which it predict in price

# In[ ]:


train.describe().T


# In[ ]:


train.describe(include='object').T


# In[ ]:


c=train.corr()


# In[ ]:


c


# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(c,annot=True)


# In[ ]:


# we can choose top 10 columns effect in price without this Correlation above
correlation=c["SalePrice"].sort_values(ascending=False)
correlation=pd.DataFrame(correlation)
# correlation


# In[ ]:


correlation.head(11)


# we show Overqual,GrLivArea is more columns effect in HousePrice

# ## data cleaning

# In[ ]:


# drop columns not import before feature selection
train.drop(columns=['Id'],axis=1,inplace=True)
test.drop(columns=['Id'],axis=1,inplace=True)


# fill null value

# In[ ]:


train_null_array


# In[ ]:


test_null_array


# In[ ]:


# array contain name of columns which Null is part of value
arr_null=['MiscFeature','Fence','PoolQC','GarageCond','GarageQual','GarageFinish','GarageType','FireplaceQu','BsmtFinType2',
         'BsmtFinType1','BsmtExposure','BsmtCond','BsmtQual','Alley']


# In[ ]:


arr_null


# In[ ]:


train.MasVnrType.value_counts()


# In[ ]:


combain=[train,test]


# In[ ]:


# fill object value by None without null value in train / test
for i in arr_null:
    train[i].fillna('NA',inplace=True)
    test[i].fillna('NA',inplace=True)    


# In[ ]:


train_null=train.isnull().sum()
train_null=pd.DataFrame(train_null,columns=['number']).sort_values(by='number',ascending=False)
train_null.reset_index(inplace=True)
train_null.head(10)


# In[ ]:


test_null=test.isnull().sum()
test_null=pd.DataFrame(test_null,columns=['number']).sort_values(by='number',ascending=False)
test_null.reset_index(inplace=True)
test_null.head(20)


# In[ ]:


# extract values which not intersection between train_null_array,arr_null
train_null_array=list(set(train_null_array).symmetric_difference(arr_null))


# In[ ]:


train_null_array


# In[ ]:


# extract values which not intersection between test_null_array,arr_null
test_null_array=list(set(test_null_array).symmetric_difference(arr_null))


# In[ ]:


test_null_array


# In[ ]:


# fill null value (object by mode) and (numeric by median) 
for i in train_null_array:
    if train[i].dtype == 'object':
        train[i].fillna(train[i].mode()[0],inplace=True)
    else:
        train[i].fillna(train[i].median(),inplace=True)


# In[ ]:


#  fill null value (object by mode) and (numeric by median) in test file
for i in test_null_array:
    if test[i].dtype == 'object':
        test[i].fillna(test[i].mode()[0],inplace=True)
    else:
        test[i].fillna(test[i].median(),inplace=True)


# In[ ]:


train_null=train.isnull().sum()
train_null=pd.DataFrame(train_null,columns=['number']).sort_values(by='number',ascending=False)
train_null.reset_index(inplace=True)
train_null.head(5)


# In[ ]:


test_null=test.isnull().sum()
test_null=pd.DataFrame(test_null,columns=['number']).sort_values(by='number',ascending=False)
test_null.reset_index(inplace=True)
test_null.head(5)


# drop CarageCars column because it correlated with GarageArea perfect connection

# In[ ]:


train.drop(columns=['GarageCars'],axis=1,inplace=True)
test.drop(columns=['GarageCars'],axis=1,inplace=True)


# In[ ]:


train.shape,test.shape


# ## EDA and Visuals

# In[ ]:


def bar_chart(col):
    HousePrice=train.groupby([col])['SalePrice'].mean()
    df_HousePrice=pd.DataFrame(HousePrice).sort_values(by=['SalePrice'],ascending=False)
    df_HousePrice.reset_index(inplace=True)
    plt.bar(x=df_HousePrice[col],height=df_HousePrice['SalePrice'])
    plt.title(f'house price effect ber {col}')
    plt.xlabel(col)
    plt.ylabel('Price')


# how fence affect to houseprice

# In[ ]:


bar_chart('Fence')


# how GarageType affect to price

# In[ ]:


bar_chart('GarageType')


# we show GarageType is effect in price

# how roofstyle affect to price

# In[ ]:


bar_chart('RoofStyle')


# how RoofMatl affect to price

# In[ ]:


plt.figure(figsize=(14,6))
bar_chart('RoofMatl')


# how SaleType affect to price

# In[ ]:


bar_chart('SaleType')


# In[ ]:


# top 10 number elements effect in SalePrice
correlation.head(11)


# In[ ]:


plt.scatter(train['OverallQual'],train['SalePrice'])
plt.title('how OverallQual effect in SalePrice')
plt.xlabel('OverallQual')
plt.ylabel('HousePrice')


# In[ ]:


plt.scatter(train['GrLivArea'],train['SalePrice'])
plt.title('how GrLivArea effect in SalePrice')
plt.xlabel('GrLivArea')
plt.ylabel('HousePrice')


# In[ ]:


plt.scatter(train['GrLivArea'],train['SalePrice'])
plt.title('how GrLivArea effect in SalePrice')
plt.xlabel('GrLivArea')
plt.ylabel('HousePrice')


# In[ ]:


plt.scatter(train['FullBath'],train['SalePrice'])
plt.title('how FullBath effect in SalePrice')
plt.xlabel('FullBath')
plt.ylabel('HousePrice')


# In[ ]:


plt.scatter(train['TotalBsmtSF'],train['SalePrice'])
plt.title('how TotalBsmtSF effect in SalePrice')
plt.xlabel('TotalBsmtSF')
plt.ylabel('HousePrice')


# In[ ]:


plt.scatter(train['1stFlrSF'],train['SalePrice'])
plt.title('how 1stFlrSF effect in SalePrice')
plt.xlabel('1stFlrSF')
plt.ylabel('HousePrice')


# In[ ]:


plt.scatter(train['TotRmsAbvGrd'],train['SalePrice'])
plt.title('how TotRmsAbvGrd effect in SalePrice')
plt.xlabel('TotRmsAbvGrd')
plt.ylabel('HousePrice')


# In[ ]:


plt.scatter(train['LotFrontage'],train['SalePrice'])
plt.title('how LotFrontage effect in SalePrice')
plt.xlabel('LotFrontage')
plt.ylabel('HousePrice')


# In[ ]:


sns.distplot(x=train['SalePrice'])


# In[ ]:


sns.distplot(x=train['LotArea'])


# In[ ]:


sns.distplot(x=train['GrLivArea'])


# In[ ]:


sns.distplot(x=train['1stFlrSF'])


# In[ ]:


# some feature we must convert it to ball shape
num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']

for i in num_features:
    train[i]=np.log(train[i])    
    test[i]=np.log(test[i])


# In[ ]:


for i in num_features:
    sns.distplot(train[i])
    plt.show()
plt.show()


# In[ ]:


for i in num_features:
    sns.boxplot(train[i])
    plt.show()
plt.show()


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


object_cols=train.select_dtypes(include='object')


# In[ ]:


le=LabelEncoder()
for i in object_cols:
    train[i]=le.fit_transform(train[i])
    test[i]=le.fit_transform(test[i])


# In[ ]:





# In[ ]:





# ## Model

# In[ ]:


from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2 , f_classif


# In[ ]:


# split data
X=train.iloc[:,:-1]
y=train.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[ ]:


def feature_selection(x,y,percent):
    x_top_columns=SelectPercentile(score_func = f_classif, percentile=percent)
    x_top_80=x_top_columns.fit_transform(x,y)
    
    # to return columns is still found
    X_train_top_80 = list(x_train.columns[x_top_columns.get_support()])
    
    X_train_feature = x_train[x_train.columns[x_train.columns.isin(X_train_top_80)]]
    X_test_feature = x_test[x_test.columns[x_test.columns.isin(X_train_top_80)]]
    return X_train_feature,X_test_feature


# In[ ]:


x_train_feature_80,x_test_feature_80=feature_selection(x_train,y_train,80)


# In[ ]:


feature_80_test=test[test.columns[test.columns.isin(x_train_feature_80)]]


# In[ ]:


feature_80_test


# ### LinearRegression

# In[ ]:


model_lr_feature_selection_80=LinearRegression(normalize=True)
model_lr_feature_selection_80.fit(x_train_feature_80,y_train)


# In[ ]:


model_lr_feature_selection_80.score(x_train_feature_80,y_train)


# In[ ]:


model_lr_feature_selection_80.score(x_test_feature_80,y_test)


# In[ ]:


predictions=model_lr_feature_selection_80.predict(x_test_feature_80)


# In[ ]:


from sklearn.metrics import explained_variance_score,r2_score


# In[ ]:


print('Mean Absolute Error(MAE):', mean_absolute_error(y_test, predictions))
print('Mean Squared Error(MSE):', mean_squared_error(y_test, predictions))
print('Root Mean Squared Error (RMSE):', np.sqrt(mean_squared_error(y_test, predictions)))
print('Explained Variance Score (EVS):',explained_variance_score(y_test,predictions))
print('R2:',r2_score(y_test, predictions))


# ### Ridge

# In[ ]:


from sklearn.linear_model import Ridge
model_ridge_80=Ridge()
model_ridge_80.fit(x_train_feature_80,y_train)


# In[ ]:


model_ridge_80.score(x_train_feature_80,y_train)


# ### RandomForest

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


n_estimators = [5,20,50,100] # number of trees in the random forest
max_features = ['auto', 'sqrt'] # number of features in consideration at every split
max_depth = [int(x) for x in np.linspace(10, 120, num = 12)] # maximum number of levels allowed in each decision tree
min_samples_split = [2, 6, 10] # minimum sample number to split a node
min_samples_leaf = [1, 3, 4] # minimum sample number that can be stored in a leaf node
bootstrap = [True, False] # method used to sample data points

random_grid = {'n_estimators': n_estimators,
'max_features': max_features,
'max_depth': max_depth,
'min_samples_split': min_samples_split,
'min_samples_leaf': min_samples_leaf,
'bootstrap': bootstrap}


rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf,param_distributions = random_grid,
               n_iter = 100, cv = 5, verbose=2, random_state=35, n_jobs = -1)
rf_random.fit(x_train_feature_80,y_train)


# In[ ]:


rf_random.score(x_train_feature_80,y_train)


# In[ ]:


rf_random.score(x_test_feature_80,y_test)


# In[ ]:


model_tree=DecisionTreeRegressor(max_depth=100)
model_tree.fit(x_train_feature_80,y_train)


# In[ ]:


model_tree.score(x_train_feature_80,y_train)


# In[ ]:


model_tree.score(x_test_feature_80,y_test)


# SVM

# In[ ]:


model_svr=SVR(C=1)
model_svr.fit(x_train_feature_80,y_train)


# In[ ]:


model_svr.score(x_train_feature_80,y_train)


# In[ ]:


model_svr.score(x_test_feature_80,y_test)


# In[ ]:





# In[ ]:





# ## Evalution model

# In[ ]:


y_predction=rf_random.predict(feature_80_test)


# In[ ]:


print('Mean Absolute Error(MAE):', mean_absolute_error(submit['SalePrice'], y_predction))
print('Mean Squared Error(MSE):', mean_squared_error(submit['SalePrice'], y_predction))
print('Root Mean Squared Error (RMSE):', np.sqrt(mean_squared_error(submit['SalePrice'], y_predction)))
print('Explained Variance Score (EVS):',explained_variance_score(submit['SalePrice'],y_predction))
print('R2:',r2_score(submit['SalePrice'], y_predction))


# In[ ]:


y_predction


# ## Submission

# In[ ]:


submission=pd.DataFrame({
    'Id':submit['Id'],
    'SalePrice':y_predction
})

submission.to_csv('./submission.csv',index=False)


# ## end
