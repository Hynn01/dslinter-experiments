#!/usr/bin/env python
# coding: utf-8

# # Problem 1 :  Predicting house prices  

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

# Any results you write to the current directory are saved as output


# In[ ]:


#import using library
from sklearn.ensemble import RandomForestRegressor
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
from scipy import stats 
from scipy.stats import norm, skew ,zscore#for some statistics
import matplotlib.pyplot as plt  # Matlab-style plotting
from sklearn.linear_model import LinearRegression,Lasso,ElasticNet,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


# In[ ]:


#load our data train and test (test dat does not include the target feature)
train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# **We can see total 81 variables for train and 80 variables for test data. And we don't have *SalePrice* variable for test set because this will be our task to infer *SalePrice* for test set by learning from train set. So *SalePrice* is our target variable and rest of the variables are our predictor variables.**
# #### Here comes the description of a few variables:
# * MSSubClass — The building class
# * MSZoning — The general zoning classification
# * LotFrontage — Linear feet of street connected to property
# * LotArea — Lot size in square feet
# * Street — Type of road access
# * Alley — Type of alley access
# * LotShape — General shape of property
# * LandContour — Flatness of the property
# * Utilities — Type of utilities available
# * LotConfig — Lot configuration
# * LandSlope: Slope of property
# * Neighborhood: Physical locations within Ames city limits
# * Condition1: Proximity to main road or railroad
# * Condition2: Proximity to main road or railroad (if a second is present)
# * BldgType: Type of dwelling
# * HouseStyle: Style of dwelling
# * OverallQual: Overall material and finish quality
# * OverallCond: Overall condition rating
# * YearBuilt: Original construction date
# 

# In[ ]:


train.shape,test.shape


# **perform appropriate exploratory data analysis **

# In[ ]:


#let know more info about features data
train.info()


# In[ ]:


#let's know the target feature by applying XOR Function between train and test
test.columns^train.columns


# In[ ]:


#let's select the numeric features 
numerc_fet=train.select_dtypes(include=np.number)


# In[ ]:


numerc_fet.head()


# # **Measure the correlation between those numeric feaures regarding to the target**

# In[ ]:



corr=numerc_fet.corr()
corr['SalePrice'].sort_values(ascending=False)[:9]


# **As above the most important variables is OverallQual  and GrLivArea  features   **

# [](http://)**heatmap of correlation**

# In[ ]:


sns.heatmap(corr)


# In[ ]:


#Check if there any outliers
sns.boxplot(x=train['OverallQual'])


# In[ ]:


sns.boxplot(x=train['GarageCars'])


# In[ ]:


sns.boxplot(x=train['TotRmsAbvGrd'])


# In[ ]:


y=train['SalePrice']


# **visualize some data distrbution!**

# In[ ]:



six_cols = ['GrLivArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'LotArea', 'SalePrice']
sns.pairplot(train[six_cols]) 
plt.show()


# In[ ]:


totalData=pd.concat([train,test])


# Creating New Features
# 
# We will create a new feature named TotalSF combining TotalBsmtSF, 1stFlrSF, and 2ndFlrSF.

# In[ ]:


#Create Feature TotalSF
totalData['TotalSF'] = totalData['TotalBsmtSF'] + totalData['1stFlrSF'] + totalData['2ndFlrSF']


# **Measure the correlation between those numeric feaures regarding to the targe after adding TotalSF feature**

# In[ ]:


x=len(y)
train_fea=totalData.iloc[:x,:]


# In[ ]:


numerc_fet2=train_fea.select_dtypes(include=np.number)
#Measure the correlation between those numeric feaures regarding to the target
corr2=numerc_fet2.corr()
corr2['SalePrice'].sort_values(ascending=False)[:9]


# As above the most important variables is OverallQual and TotalSF features

# In[ ]:


totalData.shape


# In[ ]:


#Xtrain=totalData.drop('SalePrice',axis=1)
ytrain=train['SalePrice']


# In[ ]:


#Compute the missing data
missing=totalData.isnull().sum().sort_values(ascending=False)
missing=missing[missing>0]
missing


# In[ ]:


#Dealing With Missing data
#for catergorical variables, we replece missing data with None
Miss_cat=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
          'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 
          'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass']
for col in Miss_cat:
    totalData[col].fillna('None',inplace=True)
# for numerical variables, we replace missing value with 0
Miss_num=['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 
          'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'] 
for col in Miss_num:
    totalData[col].fillna(0, inplace=True)


# In[ ]:


rest_val=['MSZoning','Functional','Utilities','Exterior1st', 'SaleType','Electrical', 'Exterior2nd','KitchenQual']
for col in rest_val:
    totalData[col].fillna(totalData[col].mode()[0],inplace=True)  #fill with most frequency data


# In[ ]:


totalData['LotFrontage']=totalData.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))#fill with median


# In[ ]:


totalData=totalData.drop('Id',axis=1)  #not important feature


# In[ ]:


totalData.head()


# In[ ]:


#convert the numeric values into string becuse there are many repetition 
totalData['YrSold'] = totalData['YrSold'].astype(str)
totalData['MoSold'] = totalData['MoSold'].astype(str)
totalData['MSSubClass'] = totalData['MSSubClass'].astype(str)
totalData['OverallCond'] = totalData['OverallCond'].astype(str)


# In[ ]:


totalData.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(totalData[c].values)) 
    totalData[c] = lbl.transform(list(totalData[c].values))


# In[ ]:


# shape        
print('Shape totalData: {}'.format(totalData.shape))


# In[ ]:


totalData.head()


# In[ ]:


numeric_feats = totalData.dtypes[totalData.dtypes != "object"].index
string_feats=totalData.dtypes[totalData.dtypes == "object"].index


# In[ ]:


string_feats


# In[ ]:


#dealing with string_feats
dumies = pd.get_dummies(totalData[string_feats])
print(dumies.shape)


# In[ ]:


totalData=pd.concat([totalData,dumies],axis='columns')


# In[ ]:


totalData.shape


# In[ ]:


totalData=totalData.drop(string_feats,axis=1)


# In[ ]:


totalData.shape


# In[ ]:


#Dealing with out liers
len(totalData)   # number of rows befor remove the outliers


# In[ ]:


x=len(ytrain)


# In[ ]:


train_feature=totalData.iloc[:x,:]
test_feature=totalData.iloc[x:,:]


# In[ ]:


train_feature.head()


# In[ ]:


#Here we will not apply scalling for data features because there are many str features and by compare the result with and without scalling

#sc_X = MinMaxScaler()
#all_data_train_normalized = sc_X.fit_transform(train_feature)
#all_data_test_normalized = sc_X.transform(test_feature)


# In[ ]:


all_data_train_normalized=train_feature
all_data_test_normalized=test_feature


# In[ ]:


all_data_train_normalized.head()


# **visualization  of traget varible distribution**

# In[ ]:


sns.distplot(ytrain , fit=norm)

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(ytrain)
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(ytrain, plot=plt)
plt.show()


# In[ ]:


ytrain.skew()


# In[ ]:


ytrain.head()


# **log function in python dealing with skewness**

# In[ ]:


ytrain=np.log(ytrain)


# In[ ]:


ytrain.skew()


# In[ ]:


ytrain=pd.DataFrame(ytrain)


# **plot Log-transformation of the target variable**

# In[ ]:


sns.distplot(ytrain , fit=norm)

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(ytrain)
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


# In[ ]:


ytrain.head()


# In[ ]:


len(ytrain),len(all_data_train_normalized)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(all_data_train_normalized,ytrain,test_size=0.2,random_state=42)


# **First Model is linear Regression**

# In[ ]:


model1= LinearRegression()
model1.fit(X_train,Y_train)


# In[ ]:


ypre1=model1.predict(X_test)


# In[ ]:


mean = mean_squared_error(y_pred=ypre1,y_true=Y_test)
r2_scor = r2_score(y_pred=ypre1,y_true=Y_test)
absloute = mean_absolute_error(y_pred=ypre1,y_true=Y_test)
print(mean,r2_scor,absloute)


# In[ ]:


model1.score(X_test,Y_test)


# # note: we here achieved more than 95 percent accuracy after adding TotalSF feature it was 91 percent before adding this feature and the mean_squared_error decresed to be 0.0089  from 0.016 

# In[ ]:


#predicting on the test set
predictions = model1.predict(X_test)


# In[ ]:


actual_values = Y_test
plt.scatter(predictions, actual_values, alpha= 0.75, color = 'b')

plt.xlabel('Predicted price')
plt.ylabel('Actual price')
plt.title('Linear Regression Model')
plt.show()


# **Describe the various models **

# In[ ]:


#Try more Models

# Test Options and Evaluation Metrics
num_folds = 5
scoring = "neg_mean_squared_error"
# Spot Check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))
models.append(('RFR', RandomForestRegressor()))


results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=0)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold,    scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(),   cv_results.std())
    print(msg)


# **Note: the best pramester here is RandomForestRegressor with 0.000579 mean_squared_error**

# In[ ]:


RFR=RandomForestRegressor()
RFR.fit(X_train,Y_train)
ypreRFR=RFR.predict(X_test)


# In[ ]:


mean = mean_squared_error(y_pred=ypreRFR,y_true=Y_test)
r2_scor = r2_score(y_pred=ypreRFR,y_true=Y_test)
absloute = mean_absolute_error(y_pred=ypreRFR,y_true=Y_test)
print(mean,r2_scor,absloute)


# In[ ]:


RFR.score(X_test,Y_test)


# **note: we here achieved more than 99 percent accuracy in this model after adding TotalSF feature and the mean_squared_error decresed to be 0.00057 **

# **#ensemble methods (gradient boosting)**

# In[ ]:



from sklearn import ensemble
# Fit regression model
params = {'n_estimators': 1000, 'max_depth': 2, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
model2 = ensemble.GradientBoostingRegressor(**params)

model2.fit(X_train, Y_train)


# In[ ]:


ypre2=model2.predict(X_test)


# In[ ]:


mean = mean_squared_error(y_pred=ypre2,y_true=Y_test)
r2_scor = r2_score(y_pred=ypre2,y_true=Y_test)
absloute = mean_absolute_error(y_pred=ypre2,y_true=Y_test)
print(mean,r2_scor,absloute)


# In[ ]:


model2.score(X_test,Y_test)


# Note: More better Accuracy 99.9 percent

# In[ ]:


predictions2 = model2.predict(X_test)


# In[ ]:


actual_values = Y_test
plt.scatter(predictions2, actual_values, alpha= 0.75, color = 'b')

plt.xlabel('Predicted price')
plt.ylabel('Actual price')
plt.title('gradient boosting Model')
plt.show()


# Linear regression with L2 regularization kind of optimization

# In[ ]:



for i in range(-2, 3):
    alpha = 10**i
    rm = Ridge(alpha = alpha)
    ridge_model = rm.fit(X_train, Y_train)
    preds_ridge = ridge_model.predict(X_test)
    
    plt.scatter(preds_ridge,Y_test, alpha= 0.75, c= 'b')
    plt.xlabel('Predicted price')
    plt.ylabel('Actual price')
    plt.title('Ridge redularization with alpha {}'.format(alpha))
    overlay = 'R square: {} \nMSE: {}'.format(ridge_model.score(X_test, Y_test), mean_squared_error(Y_test, preds_ridge))
    plt.annotate(s = overlay, xy = (12.1, 10.6), size = 'x-large')
    plt.show()


# **So After trying more than model we found the linear regression is the best one in terms in mean_squared_error so we can  select linear regrision Model or Ridge Regression **

# **here we try more value for alpha in Ridge Regression Model**

# In[ ]:



alphas = np.linspace(0.0002, 100, num=50)
scores = [
     np.sqrt(-cross_val_score(Ridge(alpha), X_train,Y_train, 
       scoring="neg_mean_squared_error")).mean()
     for alpha in alphas
]
scores = pd.Series(scores, index=alphas)
scores.plot(title = "Alphas vs error (Lowest error is best)")


# In[ ]:


rm = Ridge(alpha = 18)
ridge_model = rm.fit(X_train, Y_train)
preds_ridge = ridge_model.predict(X_test)


# In[ ]:


mean = mean_squared_error(y_pred=preds_ridge,y_true=Y_test)
r2_scor = r2_score(y_pred=preds_ridge,y_true=Y_test)
absloute = mean_absolute_error(y_pred=preds_ridge,y_true=Y_test)
print(mean,r2_scor,absloute)


# **Finally we found gradient boosting give best result compare to Random forest**

# **The predictor variables for gradient boosting**[](http://)

# In[ ]:


model2.get_params


# **Build MLP for Regression and see the result**

# In[ ]:


from keras.models import Sequential


# In[ ]:


from keras.layers import Dense


# In[ ]:


#Build MLP Model
NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
#NN_model.add(Dropout(.2))   # Add Dropout layer
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
#NN_model.add(Dropout(.2)) # Add another Dropout layer
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()


# In[ ]:


from keras.callbacks import ModelCheckpoint

checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]


# In[ ]:


MLPR_model=NN_model.fit(X_train,Y_train, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)


# In[ ]:


test_loss = NN_model.evaluate(X_test, Y_test)
test_loss[0]


# **not better accuracy in terms of val_loss**

# In[ ]:





# In[ ]:




