#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import all necessary library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#  Import dataset

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
train


# In[ ]:


test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
test


# In[ ]:


train.columns


# In[ ]:


train.info()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.isnull().sum()


# Preprocessing Data

# In[ ]:


#  here, some columns contain null value. So, drop it.

train.drop(['Id', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType'], axis=1, inplace=True)
test.drop(['Id','Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType'], axis=1, inplace=True)
train.shape


# In[ ]:


test.shape


# In[ ]:


train.duplicated().sum()


# In[ ]:


test.duplicated().sum()


# In[ ]:


train['LotFrontage'].fillna(method = 'bfill', inplace=True)
train['MasVnrArea'].fillna(0, inplace=True)
train['BsmtQual'].fillna(method = 'bfill', inplace=True)
train['BsmtCond'].fillna(method = 'bfill', inplace=True)
train['BsmtExposure'].fillna('No', inplace=True)
train['BsmtFinType1'].fillna(method = 'bfill', inplace=True)
train['BsmtFinType2'].fillna(method = 'bfill', inplace=True)
train['Electrical'].fillna(method = 'bfill', inplace=True)
train['GarageType'].fillna(method = 'bfill', inplace=True)
train['GarageYrBlt'].fillna(method = 'bfill', inplace=True)
train['GarageFinish'].fillna(method = 'bfill', inplace=True)
train['GarageQual'].fillna(method = 'bfill', inplace=True)
train['GarageCond'].fillna(method = 'bfill', inplace=True)



test['LotFrontage'].fillna(method = 'bfill', inplace=True)
test['MasVnrArea'].fillna(0, inplace=True)
test['BsmtQual'].fillna(method = 'bfill', inplace=True)
test['BsmtCond'].fillna(method = 'bfill', inplace=True)
test['BsmtExposure'].fillna('No', inplace=True)
test['BsmtFinType1'].fillna(method = 'bfill', inplace=True)
test['BsmtFinType2'].fillna(method = 'bfill', inplace=True)
test['Electrical'].fillna(method = 'bfill', inplace=True)
test['Electrical'].fillna(method = 'bfill', inplace=True)
test['BsmtFinSF1'].fillna(method = 'bfill', inplace=True)
test['BsmtFinSF2'].fillna(method = 'bfill', inplace=True)
test['BsmtUnfSF'].fillna(method = 'bfill', inplace=True)
test['TotalBsmtSF'].fillna(method = 'bfill', inplace=True)
test['BsmtFullBath'].fillna(method = 'bfill', inplace=True)
test['BsmtHalfBath'].fillna(method = 'bfill', inplace=True)
test['GarageYrBlt'].fillna(method = 'bfill', inplace=True)
test['GarageCars'].fillna(method = 'bfill', inplace=True)
test['GarageArea'].fillna(method = 'bfill', inplace=True)


# In[ ]:


# train = train.dropna(how='any',axis=0)
# test = test.dropna(how='any',axis=0)


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.head()


# **Label Encoding on object value**

# In[ ]:


from sklearn import preprocessing


# In[ ]:


print('MSZoning: ', len(train['MSZoning'].unique()))
print('Street: ', len(train['Street'].unique()))
print('LotShape: ', len(train['LotShape'].unique()))
print('LandContour: ', len(train['LandContour'].unique()))
print('Utilities: ', len(train['Utilities'].unique()))
print('LotConfig: ', len(train['LotConfig'].unique()))
print('LandSlope: ', len(train['LandSlope'].unique()))
print('Neighborhood: ', len(train['Neighborhood'].unique()))
print('Condition1: ', len(train['Condition1'].unique()))
print('Condition2: ', len(train['Condition2'].unique()))
print('BldgType: ', len(train['BldgType'].unique()))
print('HouseStyle: ', len(train['HouseStyle'].unique()))
print('RoofStyle: ', len(train['RoofStyle'].unique()))
print('RoofMatl: ', len(train['RoofMatl'].unique()))
print('Exterior1st: ', len(train['Exterior1st'].unique()))
print('Exterior2nd: ', len(train['Exterior2nd'].unique()))
print('ExterQual: ', len(train['ExterQual'].unique()))
print('ExterCond: ', len(train['ExterCond'].unique()))
print('Foundation: ', len(train['Foundation'].unique()))
print('BsmtQual: ', len(train['BsmtQual'].unique()))
print('BsmtCond: ', len(train['BsmtCond'].unique()))
print('BsmtExposure: ', len(train['BsmtExposure'].unique()))
print('BsmtFinType1: ', len(train['BsmtFinType1'].unique()))
print('BsmtFinType2: ', len(train['BsmtFinType2'].unique()))
print('Heating: ', len(train['Heating'].unique()))
print('HeatingQC: ', len(train['HeatingQC'].unique()))
print('CentralAir: ', len(train['CentralAir'].unique()))
print('Electrical: ', len(train['Electrical'].unique()))
print('KitchenQual: ', len(train['KitchenQual'].unique()))
print('Functional: ', len(train['Functional'].unique()))
print('GarageType: ', len(train['GarageType'].unique()))
print('GarageFinish: ', len(train['GarageFinish'].unique()))
print('GarageQual: ', len(train['GarageQual'].unique()))
print('GarageCond: ', len(train['GarageCond'].unique()))
print('PavedDrive: ', len(train['PavedDrive'].unique()))
print('SaleType: ', len(train['SaleType'].unique()))
print('SaleCondition: ', len(train['SaleCondition'].unique()))


# In[ ]:




label_encoder = preprocessing.LabelEncoder()

train['MSZoning'] = label_encoder.fit_transform(train['MSZoning'])
train['Street'] = label_encoder.fit_transform(train['Street'])
train['LotShape'] = label_encoder.fit_transform(train['LotShape'])
train['LandContour'] = label_encoder.fit_transform(train['LandContour'])
train['Utilities'] = label_encoder.fit_transform(train['Utilities'])
train['LotConfig'] = label_encoder.fit_transform(train['LotConfig'])
train['LandSlope'] = label_encoder.fit_transform(train['LandSlope'])
train['Neighborhood'] = label_encoder.fit_transform(train['Neighborhood'])
train['Condition1'] = label_encoder.fit_transform(train['Condition1'])
train['Condition2'] = label_encoder.fit_transform(train['Condition2'])
train['BldgType'] = label_encoder.fit_transform(train['BldgType'])
train['HouseStyle'] = label_encoder.fit_transform(train['HouseStyle'])
train['RoofStyle'] = label_encoder.fit_transform(train['RoofStyle'])
train['RoofMatl'] = label_encoder.fit_transform(train['RoofMatl'])
train['Exterior1st'] = label_encoder.fit_transform(train['Exterior1st'])
train['Exterior2nd'] = label_encoder.fit_transform(train['Exterior2nd'])
train['ExterQual'] = label_encoder.fit_transform(train['ExterQual'])
train['ExterCond'] = label_encoder.fit_transform(train['ExterCond'])
train['Foundation'] = label_encoder.fit_transform(train['Foundation'])
train['BsmtQual'] = label_encoder.fit_transform(train['BsmtQual'])
train['BsmtCond'] = label_encoder.fit_transform(train['BsmtCond'])
train['BsmtExposure'] = label_encoder.fit_transform(train['BsmtExposure'])
train['BsmtFinType1'] = label_encoder.fit_transform(train['BsmtFinType1'])
train['BsmtFinType2'] = label_encoder.fit_transform(train['BsmtFinType2'])
train['Heating'] = label_encoder.fit_transform(train['Heating'])
train['HeatingQC'] = label_encoder.fit_transform(train['HeatingQC'])
train['CentralAir'] = label_encoder.fit_transform(train['CentralAir'])
train['Electrical'] = label_encoder.fit_transform(train['Electrical'])
train['KitchenQual'] = label_encoder.fit_transform(train['KitchenQual'])
train['Functional'] = label_encoder.fit_transform(train['Functional'])
train['GarageType'] = label_encoder.fit_transform(train['GarageType'])
train['GarageFinish'] = label_encoder.fit_transform(train['GarageFinish'])
train['GarageQual'] = label_encoder.fit_transform(train['GarageQual'])
train['GarageCond'] = label_encoder.fit_transform(train['GarageCond'])
train['PavedDrive'] = label_encoder.fit_transform(train['PavedDrive'])
train['SaleType'] = label_encoder.fit_transform(train['SaleType'])
train['SaleCondition'] = label_encoder.fit_transform(train['SaleCondition'])

train.head()


# In[ ]:


train.info()


# In[ ]:


test['MSZoning'] = label_encoder.fit_transform(test['MSZoning'])
test['Street'] = label_encoder.fit_transform(test['Street'])
test['LotShape'] = label_encoder.fit_transform(test['LotShape'])
test['LandContour'] = label_encoder.fit_transform(test['LandContour'])
test['Utilities'] = label_encoder.fit_transform(test['Utilities'])
test['LotConfig'] = label_encoder.fit_transform(test['LotConfig'])
test['LandSlope'] = label_encoder.fit_transform(test['LandSlope'])
test['Neighborhood'] = label_encoder.fit_transform(test['Neighborhood'])
test['Condition1'] = label_encoder.fit_transform(test['Condition1'])
test['Condition2'] = label_encoder.fit_transform(test['Condition2'])
test['BldgType'] = label_encoder.fit_transform(test['BldgType'])
test['HouseStyle'] = label_encoder.fit_transform(test['HouseStyle'])
test['RoofStyle'] = label_encoder.fit_transform(test['RoofStyle'])
test['RoofMatl'] = label_encoder.fit_transform(test['RoofMatl'])
test['Exterior1st'] = label_encoder.fit_transform(test['Exterior1st'])
test['Exterior2nd'] = label_encoder.fit_transform(test['Exterior2nd'])
test['ExterQual'] = label_encoder.fit_transform(test['ExterQual'])
test['ExterCond'] = label_encoder.fit_transform(test['ExterCond'])
test['Foundation'] = label_encoder.fit_transform(test['Foundation'])
test['BsmtQual'] = label_encoder.fit_transform(test['BsmtQual'])
test['BsmtCond'] = label_encoder.fit_transform(test['BsmtCond'])
test['BsmtExposure'] = label_encoder.fit_transform(test['BsmtExposure'])
test['BsmtFinType1'] = label_encoder.fit_transform(test['BsmtFinType1'])
test['BsmtFinType2'] = label_encoder.fit_transform(test['BsmtFinType2'])
test['Heating'] = label_encoder.fit_transform(test['Heating'])
test['HeatingQC'] = label_encoder.fit_transform(test['HeatingQC'])
test['CentralAir'] = label_encoder.fit_transform(test['CentralAir'])
test['Electrical'] = label_encoder.fit_transform(test['Electrical'])
test['KitchenQual'] = label_encoder.fit_transform(test['KitchenQual'])
test['Functional'] = label_encoder.fit_transform(test['Functional'])
test['GarageType'] = label_encoder.fit_transform(test['GarageType'])
test['GarageFinish'] = label_encoder.fit_transform(test['GarageFinish'])
test['GarageQual'] = label_encoder.fit_transform(test['GarageQual'])
test['GarageCond'] = label_encoder.fit_transform(test['GarageCond'])
test['PavedDrive'] = label_encoder.fit_transform(test['PavedDrive'])
test['SaleType'] = label_encoder.fit_transform(test['SaleType'])
test['SaleCondition'] = label_encoder.fit_transform(test['SaleCondition'])

test.head()


# **Apply corelation technique**

# In[ ]:


X = train.drop('SalePrice', axis=1)
y = train['SalePrice']


# In[ ]:


# now, plot the dataset

plt.figure(figsize=(22,18))
ax = sns.heatmap(X.corr(), annot=True)
plt.show()


# In[ ]:


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# In[ ]:


corr_features = correlation(X, 0.7)
len(set(corr_features))


# In[ ]:


corr_features


# In[ ]:


X_corr = X.drop(corr_features,axis=1)
X_corr


# In[ ]:


test_data = test.drop(corr_features, axis=1)
test_data.head()


# **Splitting**

# In[ ]:


x_train, x_test,y_train,y_test = train_test_split(X_corr,y,test_size =0.3)

# print the data
x_train


# In[ ]:


print('x_train: ', x_train.shape)
print('x_test: ', x_test.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)


# In[ ]:


from sklearn.svm import SVR
model = SVR()

model.fit(x_train, y_train)


# In[ ]:


test_data


# In[ ]:


pred = clf.predict(test_data)
pred


# In[ ]:


# Model Evaluation

clf.score(x_test,y_test)


# Store in csv file

# In[ ]:


final = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
final


# In[ ]:


pred = pd.DataFrame(pred,columns=['SalePrice'])
sub = pd.concat([final['Id'],pred],axis=1)

sub.set_index('Id',inplace=True)

sub.to_csv("submission.csv")


# In[ ]:


cf = pd.read_csv("submission.csv")
cf.head()

