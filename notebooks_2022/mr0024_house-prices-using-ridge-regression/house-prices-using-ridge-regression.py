#!/usr/bin/env python
# coding: utf-8

# # Loading libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from math import ceil

import warnings

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from scipy.stats import skew,norm

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, make_scorer, mean_squared_log_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

from sklearn.linear_model import Ridge, LinearRegression


# In[ ]:


warnings.filterwarnings('ignore')
sns.set_theme()
pd.set_option('display.max_columns', None)


# # Importing the dataset

# In[ ]:


orig_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
orig_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


train = orig_train.copy()
test = orig_test.copy()


# # Missing values

# A quick glance at missing values.

# In[ ]:


train.info()


# In[ ]:


train.isnull().sum()[train.isnull().sum() != 0].sort_values(ascending=False)


# In[ ]:


test.isnull().sum()[test.isnull().sum() != 0].sort_values(ascending=False)


# In[ ]:


null_val = pd.concat([train, test], axis=0).isnull().sum()[pd.concat([train, test], axis=0).isnull().sum() != 0].sort_values(ascending=False)
null_val


# In[ ]:


pd.concat([train, test], axis=0).shape


# In[ ]:


def find_types (df):
    types = {}
    cat = []
    num = [] 
    flo = []
    for i in df.columns:
        types[i] = df[i].dtype
    
    for key, value in types.items():
        if (key == 'SalePrice') or (key == 'Id') or (key == 'const') or (key == 'FootPrice'):
            continue
        elif value == 'O':
            cat.append(key)
        elif value in ['int64', 'int8', 'int16']:
            num.append(key)
        elif value == 'float64':
            flo.append(key)
    
    return cat, num, flo  


# In[ ]:


cat, num, flo = find_types(train.loc[:, null_val.index])
print(cat)
print(num)
print(flo)


# In[ ]:


train.loc[train.PoolArea>0, 'PoolQC']


# In[ ]:


test.loc[test.PoolArea>0, 'PoolQC']


# In[ ]:


train[(~train.GarageType.isnull()) & train.GarageQual.isnull()]


# In[ ]:


test[(~test.GarageType.isnull()) & test.GarageQual.isnull()]


# In[ ]:


test[(test.TotalBsmtSF.isnull())]


# In[ ]:


train[(train.TotalBsmtSF>0) & train.BsmtQual.isnull()]


# Filling missing values.

# In[ ]:


def fill_na (df, cat, num, flo):
    # Categorical features
    for i in cat:
        if i == 'Functional':
            df[i].fillna('Typ', inplace=True)
        elif i == 'Exterior1':
            df[i].fillna('Other', inplace=True)
        elif i == 'Exterior2':
            df[i].fillna('Other', inplace=True) 
        elif i == 'SaleType':
            df[i].fillna('Oth', inplace=True)
        elif i == 'Electrical':
            df[i].fillna('Mix', inplace=True)
        elif i == 'KitchenQual':
            df[i].fillna('TA', inplace=True)
        else:
            df[i].fillna('None', inplace=True)

    df.loc[df.PoolArea==0, 'PoolQC'] = df.PoolQC.apply(lambda x: 'None' if x!='None' else x)
    df.loc[df.Neighborhood=='IDOTRR', 'MSZoning'] = df.MSZoning.apply(lambda x: 'RM' if x=='None' else x)
    df.loc[df.Neighborhood=='Mitchel', 'MSZoning'] = df.MSZoning.apply(lambda x: 'RL' if x=='None' else x)
    df.loc[df.MasVnrArea==0, 'MasVnrType'] = df.MasVnrType.apply(lambda x: 'None' if x!='None' else x)
    df.loc[df.MiscVal==0, 'MiscFeature'] = df.MiscFeature.apply(lambda x: 'None' if x!='None' else x)
    df.loc[df.GarageType!='None', 'GarageQual'] = df.GarageQual.apply(lambda x: 'TA' if x=='None' else x)
    df.loc[df.GarageType!='None', 'GarageCond'] = df.GarageCond.apply(lambda x: 'TA' if x=='None' else x)
    df.loc[df.GarageType!='None', 'GarageFinish'] = df.GarageFinish.apply(lambda x: 'Unf' if x=='None' else x)

    # Numerical features
    for i in num:
        df[i].fillna(0, inplace=True)
        
    for i in flo:
        if i == 'LotFrontage':
            df[i] = df.groupby(['Neighborhood', 'MSZoning'])['LotFrontage'].transform(lambda x: x.fillna(x.median() if x.median()>0 else 85))
        elif i == 'GarageYrBlt':
            df[i].fillna(df[i].min(), inplace=True)
        else:
            df[i].fillna(0, inplace=True)

    df.loc[df.PoolArea>0, 'PoolQC'] = df.PoolQC.apply(lambda x: 'Gd' if x=='None' else x)
    df.loc[df.MasVnrArea>0, 'MasVnrType'] = df.MasVnrType.apply(lambda x: 'BrkFace' if x=='None' else x)
    df.loc[df.MiscVal>0, 'MiscFeature'] = df.MiscFeature.apply(lambda x: 'Other' if x=='None' else x)
    df.loc[df.GarageType!='None', 'GarageCars'] = df.GarageCars.apply(lambda x: 1 if x==0 else x)
    df.loc[df.GarageType!='None', 'GarageArea'] = df.GarageArea.apply(lambda x: 470 if x==0 else x)

    return df


# In[ ]:


fill_na(train, cat, num, flo)
fill_na(test, cat, num, flo)


# In[ ]:


test.loc[[666, 1116],'GarageYrBlt'] = test.GarageYrBlt.apply(lambda x: 1980 if x==1895 else x)


# In[ ]:


pd.concat([train, test], axis=0).isnull().sum()[pd.concat([train, test], axis=0).isnull().sum() != 0].sort_values(ascending=False)


# # Encoding categorical features

# I will transform features which should be considered ordinal.

# In[ ]:


def type_ftr (df):
    df['NumBsmtQual'] = df.BsmtQual.replace({'None': 0, 'NA': 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5})
    df['NumBsmtCond'] = df.BsmtCond.replace({'None': 0, 'NA': 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5})
    df['NumBsmtExposure'] = df.BsmtExposure.replace({'None': 0, 'NA': 0, 'No' : 0, 'Mn' : 1, 'Av' : 2, 'Gd' : 3})
    df['NumBsmtFinType1'] = df.BsmtFinType1.replace({'None': 0, 'NA': 0, 'Unf' : 1, 'LwQ' : 2, 'Rec' : 3, 'BLQ' : 4, 'ALQ' : 5, 'GLQ' : 6})
    df['NumBsmtFinType2'] = df.BsmtFinType2.replace({'None': 0, 'NA': 0, 'Unf' : 1, 'LwQ' : 2, 'Rec' : 3, 'BLQ' : 4, 'ALQ' : 5, 'GLQ' : 6})
    df['NumHeatingQC'] = df.HeatingQC.replace({'None': 0, 'NA': 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5})
    df['NumKitchenQual'] = df.KitchenQual.replace({'None': 0, 'NA': 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5})
    df['NumGarageFinish'] = df.GarageFinish.replace({'None': 0, 'NA': 0, 'Unf' : 1, 'RFn' : 2, 'Fin' : 3})
    df['NumGarageQual'] = df.GarageQual.replace({'None': 0, 'NA': 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5})
    df['NumGarageCond'] = df.GarageCond.replace({'None': 0, 'NA': 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5})
    df['NumExterQual'] = df.ExterQual.replace({'None': 0, 'NA': 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5})
    df['NumExterCond'] = df.ExterCond.replace({'None': 0, 'NA': 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5})
    df['NumFireplaceQu'] = df.FireplaceQu.replace({'None': 0, 'NA': 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5})
    
    df['MSSubClass'] = df.MSSubClass.astype('str')
    df['OverallCond'] = df.OverallCond.astype('int64')
    df['NumBsmtQual'] = df.NumBsmtQual.astype('int64')
    df['NumBsmtCond'] = df.NumBsmtCond.astype('int64')
    df['NumBsmtExposure'] = df.NumBsmtExposure.astype('int64')
    df['NumBsmtFinType1'] = df.NumBsmtFinType1.astype('int64')
    df['NumBsmtFinType2'] = df.NumBsmtFinType2.astype('int64')
    df['NumHeatingQC'] = df.NumHeatingQC.astype('int64')
    df['NumKitchenQual'] = df.NumKitchenQual.astype('int64')
    df['NumFireplaceQu'] = df.NumFireplaceQu.astype('int64')
    df['NumGarageFinish'] = df.NumGarageFinish.astype('int64')
    df['NumGarageQual'] = df.NumGarageQual.astype('int64')
    df['NumGarageCond'] = df.NumGarageCond.astype('int64')
    df['NumExterQual'] = df.NumExterQual.astype('int64')
    df['NumExterCond'] = df.NumExterCond.astype('int64') 
    
    return df


# In[ ]:


train = type_ftr(train)
test = type_ftr(test)


# In[ ]:


cat, num, flo = find_types(train)


# In[ ]:


lflo = len(flo)
nflo = train.loc[:, train.columns.map(lambda x: x in flo)].nunique()
fig, axes = plt.subplots(ceil((lflo)/3), 3, figsize=(15, ceil(lflo/3)*4))
for ax, i in zip(axes.flat, flo):
    sns.scatterplot(ax=ax, x=train[i], y=train.SalePrice, alpha=0.2)
    ax.set_title(i) 
    
fig.tight_layout()


# In[ ]:


lnum = len(num)
fig, axes = plt.subplots(ceil((lnum)/5), 5, figsize=(15, ceil(lnum/5)*3))

for ax, i in zip(axes.flat, num):
    sns.scatterplot(ax=ax, x=train[i], y=train.SalePrice, alpha=0.2)
    ax.set_title(i) 
    
fig.tight_layout()


# In[ ]:


for i in train.loc[:, train.columns.map(lambda x: x in cat)].columns:
    print(train.loc[:, train.columns.map(lambda x: x in cat)][i].value_counts(), '\n')


# In[ ]:


for i in test.loc[:, test.columns.map(lambda x: x in cat)].columns:
    print(test.loc[:, test.columns.map(lambda x: x in cat)][i].value_counts(), '\n')


# # Plotting data

# Since in the house market the yardstick is normally the square foot, at first I will create two new features that could be useful in the next analysis: the total of finished area usually considered useful in the assessment of house value, and the price per square foot. 

# In[ ]:


train['FootPrice'] = train.SalePrice / (train['1stFlrSF'] + train['2ndFlrSF'] + train['LowQualFinSF'] +                             train['BsmtFinSF1'] + train['BsmtFinSF2'])
train['SqrFeet'] = train['1stFlrSF'] + train['2ndFlrSF'] + train['LowQualFinSF'] +                             train['BsmtFinSF1'] + train['BsmtFinSF2']


# In[ ]:


test['SqrFeet'] = test['1stFlrSF'] + test['2ndFlrSF'] + test['LowQualFinSF'] +                             test['BsmtFinSF1'] + test['BsmtFinSF2']


# The most important physical factors affecting house pricing are the finished areas and the overall quality, so I will check those at first. 

# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(12,5))

sns.scatterplot(ax=axes[0], x=train.SqrFeet, y=train.SalePrice, hue=train.OverallQual, alpha=1)
sns.scatterplot(ax=axes[1], x=train.OverallQual, y=train.SalePrice, hue=train.OverallQual, alpha=1)

fig.tight_layout()


# In[ ]:


outliers = train.query("SqrFeet>6600").index.values


# In[ ]:


train.iloc[outliers,:]


# There are two houses having a big area and high quality but a low price. The sale condition of these houses is "Partial", so the value could not reflect the real quote of market.

# In[ ]:


train.query("SaleCondition=='Partial' or SaleCondition=='Abnorml' or SaleCondition=='Family'").shape


# There are 246 observations with unusual sale conditions.

# In[ ]:


train.query("(SaleCondition!='Partial' and SaleCondition!='Abnorml' and SaleCondition!='Family') and (SaleType=='New')").shape


# In[ ]:


train.query("SaleType=='New'").shape


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(12,5))

sns.scatterplot(ax=axes[0], x=train.query("SaleCondition=='Partial' or SaleCondition=='Abnorml' or SaleCondition=='Family'").SqrFeet, y=train.query("SaleCondition=='Partial' or SaleCondition=='Abnorml' or SaleCondition=='Family'").SalePrice, hue=train.OverallQual, alpha=1)
sns.scatterplot(ax=axes[1], x=train.query("SaleCondition=='Partial' or SaleCondition=='Abnorml' or SaleCondition=='Family'").OverallQual, y=train.query("SaleCondition=='Partial' or SaleCondition=='Abnorml' or SaleCondition=='Family'").SalePrice, hue=train.OverallQual, alpha=1)

fig.tight_layout()


# I will drop the two observations and divide the data in two parts in order to have a non influenced analysis of normal house market.

# In[ ]:


train.drop(index=[*outliers], inplace=True)
train = train.reset_index(drop=True)


# In[ ]:


sc_train = train.query("SaleCondition=='Partial' or SaleCondition=='Abnorml' or SaleCondition=='Family'")
train = train.query("SaleCondition!='Partial' and SaleCondition!='Abnorml' and SaleCondition!='Family'")


# In[ ]:


sc_train.shape


# In[ ]:


train.shape


# I will start to check data in detail with the following topics:
# 1) Sale conditions;\
# 2) Age;\
# 3) Surfaces;\
# 4) Garage;\
# 5) Living zones;\
# 6) Facilities/Other.

# ## Sale Conditions

# In[ ]:


sns.lmplot(
    x='GrLivArea', y='SalePrice', hue='SaleCondition', scatter_kws={"edgecolor": 'w'}, 
    data=train)


# In[ ]:


sns.lmplot(
    x='GrLivArea', y='SalePrice', hue='SaleCondition', scatter_kws={"edgecolor": 'w'}, 
    data=sc_train)


# In[ ]:


train.groupby(by=['SaleCondition', 'SaleType'])['SaleCondition', 'YrSold', 'YearBuilt', 'GrLivArea', 'OverallQual', 'FootPrice'].agg({'SaleCondition':'count', 'YrSold':'mean', 'YearBuilt':'mean', 'GrLivArea':'mean', 'OverallQual':'mean', 'FootPrice':'mean'})


# In[ ]:


sc_train.groupby(by=['SaleCondition', 'SaleType'])['SaleCondition', 'YrSold', 'YearBuilt', 'GrLivArea', 'OverallQual', 'FootPrice'].agg({'SaleCondition':'count', 'YrSold':'mean', 'YearBuilt':'mean', 'GrLivArea':'mean', 'OverallQual':'mean', 'FootPrice':'mean'})


# In[ ]:


sns.lmplot(
    x='GrLivArea', y='SalePrice', hue='SaleType', col='SaleType',
    data=sc_train)


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(15,5))

sns.lineplot(ax=axes[0], x=train.MoSold, y=train.FootPrice)
sns.lineplot(ax=axes[1], x=train.YrSold, y=train.FootPrice)
fig.tight_layout()


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(15,5))

sns.lineplot(ax=axes[0], x=train.MoSold, y=train.SalePrice)
sns.lineplot(ax=axes[1], x=train.YrSold, y=train.SalePrice)
fig.tight_layout()


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(15,5))

sns.lineplot(ax=axes[0], x=sc_train.MoSold, y=sc_train.FootPrice)
sns.lineplot(ax=axes[1], x=sc_train.YrSold, y=sc_train.FootPrice)
fig.tight_layout()


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(15,5))

sns.lineplot(ax=axes[0], x=sc_train.MoSold, y=sc_train.SalePrice)
sns.lineplot(ax=axes[1], x=sc_train.YrSold, y=sc_train.SalePrice)
fig.tight_layout()


# In[ ]:


sns.lmplot(
    x='GrLivArea', y='SalePrice', hue='MSSubClass', col='MSSubClass',
    data=train, col_wrap=7, height=6,)


# In[ ]:


sns.lmplot(
    x='GrLivArea', y='SalePrice', hue='MSSubClass', col='MSSubClass',
    data=sc_train, col_wrap=7, height=6,)


# ## Age

# In[ ]:


def style_negative(v, props=''):
    return props if v < 0 else None


# In[ ]:


corrMatrix = train.loc[:,train.columns[train.columns.str.match(r'SalePric|OverallCond|OverallQual|YearBuilt|YearRemodAdd|GarageYrBlt')]].corr()
crrm = corrMatrix.loc[((abs(corrMatrix) > 0.5) & (abs(corrMatrix) < 1.0)).any(),((abs(corrMatrix) > 0.5) & (abs(corrMatrix) < 1.0)).any()]
crrm = crrm.style.applymap(style_negative, props='color:red;')              .applymap(lambda v: 'opacity: 20%;' if (v < 0.7) and (v > -0.7) else None)
crrm


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(12,5))

sns.scatterplot(ax=axes[0], x=train.YearBuilt, hue=train.GarageYrBlt, y=train.FootPrice, alpha=1)
sns.scatterplot(ax=axes[1], x=train.YearBuilt, hue=train.YearRemodAdd, y=train.FootPrice, alpha=1)
fig.tight_layout()


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(12,5))

sns.scatterplot(ax=axes[0], x=sc_train.YearBuilt, hue=sc_train.GarageYrBlt, y=sc_train.FootPrice, alpha=1)
sns.scatterplot(ax=axes[1], x=sc_train.YearBuilt, hue=sc_train.YearRemodAdd, y=sc_train.FootPrice, alpha=1)
fig.tight_layout()


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(12,5))

sns.scatterplot(ax=axes[0], x=train.YearBuilt, y=train.FootPrice, hue=train.OverallQual, alpha=1)
sns.scatterplot(ax=axes[1], x=train.YearRemodAdd, y=train.FootPrice, hue=train.OverallQual, alpha=1)
fig.tight_layout()


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(12,5))

sns.scatterplot(ax=axes[0], x=train.YearBuilt, y=train.FootPrice, hue=train.OverallCond, alpha=1)
sns.scatterplot(ax=axes[1], x=train.YearRemodAdd, y=train.FootPrice, hue=train.OverallCond, alpha=1)
fig.tight_layout()


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(12,5))

sns.scatterplot(ax=axes[0], x=sc_train.YearBuilt, y=sc_train.FootPrice, hue=sc_train.OverallCond, alpha=1)
sns.scatterplot(ax=axes[1], x=sc_train.YearRemodAdd, y=sc_train.FootPrice, hue=sc_train.OverallCond, alpha=1)
fig.tight_layout()


# ### Surfaces

# In[ ]:


(train['GrLivArea'] - (train['1stFlrSF'] + train['2ndFlrSF'] + train['LowQualFinSF'])).unique()


# In[ ]:


(train['TotalBsmtSF'] - (train['BsmtFinSF1'] + train['BsmtFinSF2'] + train['BsmtUnfSF'])).unique()


# In[ ]:


corrMatrix = train.loc[:,train.columns[train.columns.str.match(r'SalePric|OverallCond|OverallQual|GrLivArea|1stFlrSF|2ndFlrSF|LowQualFinSF|TotalBsmtSF|BsmtUnfSF|BsmtFinSF1|BsmtFinSF2|MasVnrArea|FullBath|HalfBath|Fireplaces|NumBsmtFinType1|NumBsmtFinType2|NumBsmtQual|NumBsmtCond|NumBsmtExposure|OpenPorchSF|EnclosedPorch|3SsnPorch|ScreenPorch|WoodDeckSF|LotArea|LotFrontage|GarageArea')]].corr()
crrm = corrMatrix.loc[((abs(corrMatrix) > 0.5) & (abs(corrMatrix) < 1.0)).any(),((abs(corrMatrix) > 0.5) & (abs(corrMatrix) < 1.0)).any()]
crrm = crrm.style.applymap(style_negative, props='color:red;')              .applymap(lambda v: 'opacity: 20%;' if (v < 0.5) and (v > -0.5) else None)
crrm


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(15,5))

sns.scatterplot(ax=axes[0], x=train.GrLivArea, y=train.SalePrice, hue=train.OverallQual, alpha=1)
sns.scatterplot(ax=axes[1], x=train['1stFlrSF'], y=train.SalePrice, hue=train.OverallQual, alpha=1)
sns.scatterplot(ax=axes[2], x=train['2ndFlrSF'], y=train.SalePrice, hue=train.OverallQual, alpha=1)
sns.scatterplot(ax=axes[3], x=train.LowQualFinSF, y=train.SalePrice, hue=train.OverallQual, alpha=1)

fig.tight_layout()


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(12,5))

sns.scatterplot(ax=axes[0], x=train['TotalBsmtSF'], y=train.SalePrice, hue=train.BsmtUnfSF, alpha=1)
sns.scatterplot(ax=axes[1], x=train['GrLivArea'], y=train.SalePrice, hue=train.LowQualFinSF, alpha=1)
fig.tight_layout()


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(15,5))

sns.scatterplot(ax=axes[0], x=train.GrLivArea, y=train.SalePrice, hue=train.MasVnrArea, alpha=1)
sns.scatterplot(ax=axes[1], x=train.GrLivArea, y=train.SalePrice, hue=train.TotRmsAbvGrd, alpha=1)
sns.scatterplot(ax=axes[2], x=train.GrLivArea, y=train.SalePrice, hue=train.FullBath, alpha=1)
sns.scatterplot(ax=axes[3], x=train.GrLivArea, y=train.SalePrice, hue=train.Fireplaces, alpha=1)

fig.tight_layout()


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(15,5))

sns.scatterplot(ax=axes[0], x=train.TotalBsmtSF, y=train.SalePrice, hue=train.OverallQual, alpha=1)
sns.scatterplot(ax=axes[1], x=train.BsmtFinSF1, y=train.SalePrice, hue=train.OverallQual, alpha=1)
sns.scatterplot(ax=axes[2], x=train.BsmtFinSF2, y=train.SalePrice, hue=train.OverallQual, alpha=1)
sns.scatterplot(ax=axes[3], x=train.BsmtUnfSF, y=train.SalePrice, hue=train.OverallQual, alpha=1)

fig.tight_layout()


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(15,5))

sns.scatterplot(ax=axes[0], x=train.TotalBsmtSF, y=train.SalePrice, hue=train.BsmtQual, alpha=1)
sns.scatterplot(ax=axes[1], x=train.BsmtFinSF1, y=train.SalePrice, hue=train.BsmtQual, alpha=1)
sns.scatterplot(ax=axes[2], x=train.BsmtFinSF2, y=train.SalePrice, hue=train.BsmtQual, alpha=1)
sns.scatterplot(ax=axes[3], x=train.BsmtUnfSF, y=train.SalePrice, hue=train.BsmtQual, alpha=1)

fig.tight_layout()


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(15,5))

sns.scatterplot(ax=axes[0], x=train.TotalBsmtSF, y=train.SalePrice, hue=train.BsmtCond, alpha=1)
sns.scatterplot(ax=axes[1], x=train.BsmtFinSF1, y=train.SalePrice, hue=train.BsmtCond, alpha=1)
sns.scatterplot(ax=axes[2], x=train.BsmtFinSF2, y=train.SalePrice, hue=train.BsmtCond, alpha=1)
sns.scatterplot(ax=axes[3], x=train.BsmtUnfSF, y=train.SalePrice, hue=train.BsmtCond, alpha=1)

fig.tight_layout()


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(15,5))

sns.scatterplot(ax=axes[0], x=train.TotalBsmtSF, y=train.SalePrice, hue=train.BsmtFinType1, alpha=1)
sns.scatterplot(ax=axes[1], x=train.BsmtFinSF1, y=train.SalePrice, hue=train.BsmtFinType1, alpha=1)
sns.scatterplot(ax=axes[2], x=train.BsmtFinSF2, y=train.SalePrice, hue=train.BsmtFinType1, alpha=1)
sns.scatterplot(ax=axes[3], x=train.BsmtUnfSF, y=train.SalePrice, hue=train.BsmtFinType1, alpha=1)

fig.tight_layout()


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(15,5))

sns.scatterplot(ax=axes[0], x=train.TotalBsmtSF, y=train.SalePrice, hue=train.BsmtFinType2, alpha=1)
sns.scatterplot(ax=axes[1], x=train.BsmtFinSF1, y=train.SalePrice, hue=train.BsmtFinType2, alpha=1)
sns.scatterplot(ax=axes[2], x=train.BsmtFinSF2, y=train.SalePrice, hue=train.BsmtFinType2, alpha=1)
sns.scatterplot(ax=axes[3], x=train.BsmtUnfSF, y=train.SalePrice, hue=train.BsmtFinType2, alpha=1)

fig.tight_layout()


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(16,4))

sns.scatterplot(ax=axes[0], x=train['OpenPorchSF'], y=train.SalePrice, hue=train.GrLivArea, alpha=1)
sns.scatterplot(ax=axes[1], x=train['EnclosedPorch'], y=train.SalePrice, hue=train.GrLivArea, alpha=1)
sns.scatterplot(ax=axes[2], x=train['3SsnPorch'], y=train.SalePrice, hue=train.GrLivArea, alpha=1)
sns.scatterplot(ax=axes[3], x=train['ScreenPorch'], y=train.SalePrice, hue=train.GrLivArea, alpha=1)
fig.tight_layout()


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(16,4))

sns.scatterplot(ax=axes[0], x=train['OpenPorchSF'], y=train.SalePrice, hue=train.OverallQual, alpha=1)
sns.scatterplot(ax=axes[1], x=train['EnclosedPorch'], y=train.SalePrice, hue=train.OverallQual, alpha=1)
sns.scatterplot(ax=axes[2], x=train['3SsnPorch'], y=train.SalePrice, hue=train.OverallQual, alpha=1)
sns.scatterplot(ax=axes[3], x=train['ScreenPorch'], y=train.SalePrice, hue=train.OverallQual, alpha=1)
fig.tight_layout()


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(16,4))

sns.scatterplot(ax=axes[0], x=train.LotArea, y=train.SalePrice, hue=train.WoodDeckSF, alpha=1)
sns.scatterplot(ax=axes[1], x=train.LotArea, y=train.SalePrice, hue=train.LotFrontage, alpha=1)
sns.scatterplot(ax=axes[2], x=train.LotArea, y=train.SalePrice, hue=train.MiscFeature, alpha=1)
sns.scatterplot(ax=axes[3], x=train.WoodDeckSF, y=train.SalePrice, hue=train.PoolArea, alpha=1)
fig.tight_layout()


# ## Garage

# In[ ]:


corrMatrix = train.loc[:,train.columns[train.columns.str.match(r'SalePric|OverallCond|OverallQual|GrLivArea|GarageArea|GarageCars|GarageFinish|GarageQual|GarageCond')]].corr()
crrm = corrMatrix.loc[((abs(corrMatrix) > 0.5) & (abs(corrMatrix) < 1.0)).any(),((abs(corrMatrix) > 0.5) & (abs(corrMatrix) < 1.0)).any()]
crrm = crrm.style.applymap(style_negative, props='color:red;')              .applymap(lambda v: 'opacity: 20%;' if (v < 0.5) and (v > -0.5) else None)
crrm


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(16,4))

sns.scatterplot(ax=axes[0], x=train['GarageArea'], y=train.SalePrice, hue=train.GarageCars, alpha=1)
sns.scatterplot(ax=axes[1], x=train['GarageArea'], y=train.SalePrice, hue=train.GarageFinish, alpha=1)
sns.scatterplot(ax=axes[2], x=train['GarageArea'], y=train.SalePrice, hue=train.GarageQual, alpha=1)
sns.scatterplot(ax=axes[3], x=train['GarageArea'], y=train.SalePrice, hue=train.GarageCond, alpha=1)
fig.tight_layout()


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(16,4))

sns.scatterplot(ax=axes[0], x=train['GarageArea'], y=train.SalePrice, hue=train.GarageType, alpha=1)
sns.scatterplot(ax=axes[1], x=train['GarageArea'], y=train.SalePrice, hue=train.TotalBsmtSF, alpha=1)
sns.scatterplot(ax=axes[2], x=train['GarageArea'], y=train.SalePrice, hue=train.GrLivArea, alpha=1)
sns.scatterplot(ax=axes[3], x=train['GarageArea'], y=train.SalePrice, hue=train.MiscFeature, alpha=1)
fig.tight_layout()


# ## Living zones

# In[ ]:


train.groupby(by='MSZoning')['MSZoning', 'YearBuilt', 'GrLivArea', 'OverallQual', 'FootPrice'].agg({'MSZoning':'count', 'YearBuilt':'mean', 'GrLivArea':'mean', 'OverallQual':'mean', 'FootPrice':'mean'}).sort_values(by='FootPrice')


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(16,4))

sns.scatterplot(ax=axes[0], x=train.GrLivArea, y=train.SalePrice, hue=train.MSZoning, alpha=1)
sns.scatterplot(ax=axes[1], x=train.TotalBsmtSF, y=train.SalePrice, hue=train.MSZoning, alpha=1)
sns.scatterplot(ax=axes[2], x=train.OverallQual, y=train.SalePrice, hue=train.MSZoning, alpha=1)
sns.scatterplot(ax=axes[3], x=train.GarageCars, y=train.SalePrice, hue=train.MSZoning, alpha=1)

fig.tight_layout()


# In[ ]:


sns.lmplot(
    x='SqrFeet', y='SalePrice', hue='MSZoning', col='MSZoning',
    data=train)


# In[ ]:


train.groupby(by=['Neighborhood', 'MSZoning'])['Neighborhood', 'YearBuilt', 'GrLivArea', 'OverallQual', 'SalePrice', 'FootPrice'].agg({'Neighborhood':'count', 'YearBuilt':'mean', 'GrLivArea':'mean', 'OverallQual':'mean', 'FootPrice':'mean'}).sort_values(by='FootPrice')


# In[ ]:


train.query("Neighborhood=='OldTown'").YearBuilt.unique()


# In[ ]:


train.query("Neighborhood=='OldTown'").YearRemodAdd.unique()


# In[ ]:


train.groupby(by='Neighborhood')['Neighborhood', 'YearRemodAdd', 'GrLivArea', 'OverallQual', 'SalePrice', 'FootPrice'].agg({'Neighborhood':'count', 'YearRemodAdd':'mean', 'GrLivArea':'mean', 'OverallQual':'mean', 'FootPrice':'mean'}).sort_values(by='FootPrice')


# In[ ]:


nei_conc = pd.concat([sc_train, train, test], axis=0)
nei = nei_conc.groupby(by='Neighborhood')['Neighborhood', 'YearRemodAdd', 'GrLivArea', 'OverallQual', 'GarageCars', 'NumExterQual'].agg({'Neighborhood':'count', 'YearRemodAdd':'mean', 'GrLivArea':'mean', 'OverallQual':'mean', 'GarageCars':'mean', 'NumExterQual':'mean'})
nei = pd.DataFrame(MinMaxScaler().fit_transform(nei), index=nei.index, columns=nei.columns)
(nei.YearRemodAdd + nei.OverallQual + nei.GarageCars + nei.NumExterQual).sort_values()


# In[ ]:


sns.lmplot(
    x='SqrFeet', y='SalePrice', hue='Neighborhood', col='Neighborhood',
    data=train, col_wrap=7, height=4,)


# In[ ]:


train.groupby(by=['Neighborhood', 'Condition1'])['Neighborhood', 'YearBuilt', 'GrLivArea', 'OverallQual', 'FootPrice'].agg({'Neighborhood':'count', 'YearBuilt':'mean', 'GrLivArea':'mean', 'OverallQual':'mean', 'FootPrice':'mean'}).tail(50)


# In[ ]:


train.groupby(by=['Condition1', 'Condition2'])['Condition1', 'YearBuilt', 'GrLivArea', 'OverallQual', 'FootPrice'].agg({'Condition1':'count', 'YearBuilt':'mean', 'GrLivArea':'mean', 'OverallQual':'mean', 'FootPrice':'mean'}).tail(50)


# In[ ]:


train.groupby(by=['Neighborhood', 'BldgType'])['Neighborhood', 'YearBuilt', 'GrLivArea', 'OverallQual', 'FootPrice'].agg({'Neighborhood':'count', 'YearBuilt':'mean', 'GrLivArea':'mean', 'OverallQual':'mean', 'FootPrice':'mean'}).tail(50)


# In[ ]:


train.groupby(by='BldgType')['BldgType', 'YearBuilt', 'GrLivArea', 'OverallQual', 'FootPrice'].agg({'BldgType':'count', 'YearBuilt':'mean', 'GrLivArea':'mean', 'OverallQual':'mean', 'FootPrice':'mean'}).sort_values(by='FootPrice')


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(16,4))

sns.scatterplot(ax=axes[0], x=train.GrLivArea, y=train.SalePrice, hue=train.BldgType, alpha=1)
sns.scatterplot(ax=axes[1], x=train['1stFlrSF'], y=train.SalePrice, hue=train.BldgType, alpha=1)
sns.scatterplot(ax=axes[2], x=train['2ndFlrSF'], y=train.SalePrice, hue=train.BldgType, alpha=1)
sns.scatterplot(ax=axes[3], x=train.TotalBsmtSF, y=train.SalePrice, hue=train.BldgType, alpha=1)

fig.tight_layout()


# In[ ]:


sns.lmplot(
    x='SqrFeet', y='SalePrice', hue='BldgType', col='BldgType',
    data=train, col_wrap=7, height=6,)


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(16,4))

sns.scatterplot(ax=axes[0], x=train.GrLivArea, y=train.SalePrice, hue=train.HouseStyle, alpha=1)
sns.scatterplot(ax=axes[1], x=train['1stFlrSF'], y=train.SalePrice, hue=train.HouseStyle, alpha=1)
sns.scatterplot(ax=axes[2], x=train['2ndFlrSF'], y=train.SalePrice, hue=train.HouseStyle, alpha=1)
sns.scatterplot(ax=axes[3], x=train.TotalBsmtSF, y=train.SalePrice, hue=train.HouseStyle, alpha=1)

fig.tight_layout()


# In[ ]:


sns.lmplot(
    x='1stFlrSF', y='SalePrice', hue='HouseStyle', col='HouseStyle',
    data=train, col_wrap=7, height=6,)


# ## Facilities/Other

# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(12,5))

sns.scatterplot(ax=axes[0], x=train.YearBuilt, y=train.FootPrice, hue=train.Foundation, alpha=1)
sns.scatterplot(ax=axes[1], x=train.YearBuilt, y=train.FootPrice, hue=train.MasVnrType, alpha=1)
fig.tight_layout()


# In[ ]:


sns.lmplot(
    x='YearBuilt', y='SalePrice', hue='Foundation', col='Foundation',
    data=train)


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(16,4))

sns.scatterplot(ax=axes[0], x=train.GrLivArea, y=train.SalePrice, hue=train.Foundation, alpha=1)
sns.scatterplot(ax=axes[1], x=train['1stFlrSF'], y=train.SalePrice, hue=train.Foundation, alpha=1)
sns.scatterplot(ax=axes[2], x=train['2ndFlrSF'], y=train.SalePrice, hue=train.Foundation, alpha=1)
sns.scatterplot(ax=axes[3], x=train.TotalBsmtSF, y=train.SalePrice, hue=train.Foundation, alpha=1)
fig.tight_layout()


# In[ ]:


sns.lmplot(
    x='SqrFeet', y='SalePrice', hue='Foundation', col='Foundation',
    data=train)


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(16,4))

sns.scatterplot(ax=axes[0], x=train.GrLivArea, y=train.SalePrice, hue=train.MasVnrType, alpha=1)
sns.scatterplot(ax=axes[1], x=train['1stFlrSF'], y=train.SalePrice, hue=train.MasVnrType, alpha=1)
sns.scatterplot(ax=axes[2], x=train['2ndFlrSF'], y=train.SalePrice, hue=train.MasVnrType, alpha=1)
sns.scatterplot(ax=axes[3], x=train.TotalBsmtSF, y=train.SalePrice, hue=train.MasVnrType, alpha=1)

fig.tight_layout()


# In[ ]:


sns.lmplot(
    x='GrLivArea', y='SalePrice', hue='MasVnrType', col='MasVnrType',
    data=train)


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(16,4))

sns.scatterplot(ax=axes[0], x=train.GrLivArea, y=train.SalePrice, hue=train.LotConfig, alpha=1)
sns.scatterplot(ax=axes[1], x=train['1stFlrSF'], y=train.SalePrice, hue=train.LotConfig, alpha=1)
sns.scatterplot(ax=axes[2], x=train['2ndFlrSF'], y=train.SalePrice, hue=train.LotConfig, alpha=1)
sns.scatterplot(ax=axes[3], x=train.TotalBsmtSF, y=train.SalePrice, hue=train.LotConfig, alpha=1)

fig.tight_layout()


# In[ ]:


sns.lmplot(
    x='SqrFeet', y='SalePrice', hue='LotConfig', col='LotConfig',
    data=train, col_wrap=7, height=6,)


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(16,4))

sns.scatterplot(ax=axes[0], x=train.GrLivArea, y=train.SalePrice, hue=train.LotShape, alpha=1)
sns.scatterplot(ax=axes[1], x=train['1stFlrSF'], y=train.SalePrice, hue=train.LotShape, alpha=1)
sns.scatterplot(ax=axes[2], x=train['2ndFlrSF'], y=train.SalePrice, hue=train.LotShape, alpha=1)
sns.scatterplot(ax=axes[3], x=train.TotalBsmtSF, y=train.SalePrice, hue=train.LotShape, alpha=1)

fig.tight_layout()


# In[ ]:


sns.lmplot(
    x='GrLivArea', y='SalePrice', hue='LotShape', col='LotShape',
    data=train, col_wrap=7, height=6,)


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(16,4))

sns.scatterplot(ax=axes[0], x=train.GrLivArea, y=train.SalePrice, hue=train.LandContour, alpha=1)
sns.scatterplot(ax=axes[1], x=train['1stFlrSF'], y=train.SalePrice, hue=train.LandContour, alpha=1)
sns.scatterplot(ax=axes[2], x=train['2ndFlrSF'], y=train.SalePrice, hue=train.LandContour, alpha=1)
sns.scatterplot(ax=axes[3], x=train.TotalBsmtSF, y=train.SalePrice, hue=train.LandContour, alpha=1)

fig.tight_layout()


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(16,4))

sns.scatterplot(ax=axes[0], x=train.GrLivArea, y=train.SalePrice, hue=train.LandSlope, alpha=1)
sns.scatterplot(ax=axes[1], x=train['1stFlrSF'], y=train.SalePrice, hue=train.LandSlope, alpha=1)
sns.scatterplot(ax=axes[2], x=train['2ndFlrSF'], y=train.SalePrice, hue=train.LandSlope, alpha=1)
sns.scatterplot(ax=axes[3], x=train.TotalBsmtSF, y=train.SalePrice, hue=train.LandSlope, alpha=1)

fig.tight_layout()


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(16,4))

sns.scatterplot(ax=axes[0], x=train.GrLivArea, y=train.SalePrice, hue=train.CentralAir, alpha=1)
sns.scatterplot(ax=axes[1], x=train['1stFlrSF'], y=train.SalePrice, hue=train.CentralAir, alpha=1)
sns.scatterplot(ax=axes[2], x=train['2ndFlrSF'], y=train.SalePrice, hue=train.CentralAir, alpha=1)
sns.scatterplot(ax=axes[3], x=train.TotalBsmtSF, y=train.SalePrice, hue=train.CentralAir, alpha=1)

fig.tight_layout()


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(16,4))

sns.scatterplot(ax=axes[0], x=train.GrLivArea, y=train.SalePrice, hue=train.Electrical, alpha=1)
sns.scatterplot(ax=axes[1], x=train['1stFlrSF'], y=train.SalePrice, hue=train.Electrical, alpha=1)
sns.scatterplot(ax=axes[2], x=train['2ndFlrSF'], y=train.SalePrice, hue=train.Electrical, alpha=1)
sns.scatterplot(ax=axes[3], x=train.TotalBsmtSF, y=train.SalePrice, hue=train.Electrical, alpha=1)

fig.tight_layout()


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(16,4))

sns.scatterplot(ax=axes[0], x=train.GrLivArea, y=train.SalePrice, hue=train.Heating, alpha=1)
sns.scatterplot(ax=axes[1], x=train['1stFlrSF'], y=train.SalePrice, hue=train.Heating, alpha=1)
sns.scatterplot(ax=axes[2], x=train['2ndFlrSF'], y=train.SalePrice, hue=train.Heating, alpha=1)
sns.scatterplot(ax=axes[3], x=train.TotalBsmtSF, y=train.SalePrice, hue=train.Heating, alpha=1)

fig.tight_layout()


# In[ ]:


sns.lmplot(
    x='GrLivArea', y='SalePrice', hue='Heating', col='Heating',
    data=train, col_wrap=7, height=6,)


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(16,4))

sns.scatterplot(ax=axes[0], x=train.GrLivArea, y=train.SalePrice, hue=train.Fence, alpha=1)
sns.scatterplot(ax=axes[1], x=train['1stFlrSF'], y=train.SalePrice, hue=train.Fence, alpha=1)
sns.scatterplot(ax=axes[2], x=train['2ndFlrSF'], y=train.SalePrice, hue=train.Fence, alpha=1)
sns.scatterplot(ax=axes[3], x=train.TotalBsmtSF, y=train.SalePrice, hue=train.Fence, alpha=1)

fig.tight_layout()


# In[ ]:


sns.lmplot(
    x='GrLivArea', y='SalePrice', hue='Fence', col='Fence',
    data=train, col_wrap=7, height=6,)


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(16,4))

sns.scatterplot(ax=axes[0], x=train.GrLivArea, y=train.SalePrice, hue=train.RoofMatl, alpha=1)
sns.scatterplot(ax=axes[1], x=train['1stFlrSF'], y=train.SalePrice, hue=train.RoofMatl, alpha=1)
sns.scatterplot(ax=axes[2], x=train['2ndFlrSF'], y=train.SalePrice, hue=train.RoofMatl, alpha=1)
sns.scatterplot(ax=axes[3], x=train.TotalBsmtSF, y=train.SalePrice, hue=train.RoofMatl, alpha=1)

fig.tight_layout()


# In[ ]:


sns.lmplot(
    x='GrLivArea', y='SalePrice', hue='RoofMatl', col='RoofMatl',
    data=train, col_wrap=7, height=6,)


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(16,4))

sns.scatterplot(ax=axes[0], x=train.GrLivArea, y=train.SalePrice, hue=train.RoofStyle, alpha=1)
sns.scatterplot(ax=axes[1], x=train['1stFlrSF'], y=train.SalePrice, hue=train.RoofStyle, alpha=1)
sns.scatterplot(ax=axes[2], x=train['2ndFlrSF'], y=train.SalePrice, hue=train.RoofStyle, alpha=1)
sns.scatterplot(ax=axes[3], x=train.TotalBsmtSF, y=train.SalePrice, hue=train.RoofStyle, alpha=1)

fig.tight_layout()


# In[ ]:


sns.lmplot(
    x='GrLivArea', y='SalePrice', hue='RoofStyle', col='RoofStyle',
    data=train, col_wrap=7, height=6,)


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(16,4))

sns.scatterplot(ax=axes[0], x=train.GrLivArea, y=train.SalePrice, hue=train.Functional, alpha=1)
sns.scatterplot(ax=axes[1], x=train['1stFlrSF'], y=train.SalePrice, hue=train.Functional, alpha=1)
sns.scatterplot(ax=axes[2], x=train['2ndFlrSF'], y=train.SalePrice, hue=train.Functional, alpha=1)
sns.scatterplot(ax=axes[3], x=train.TotalBsmtSF, y=train.SalePrice, hue=train.Functional, alpha=1)

fig.tight_layout()


# In[ ]:


sns.lmplot(
    x='GrLivArea', y='SalePrice', hue='Functional', col='Functional',
    data=train, col_wrap=7, height=6,)


# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(7,7))

sns.scatterplot(
    x='GrLivArea', y='SalePrice', hue='Exterior1st', alpha=0.9, 
    data=train)


# In[ ]:


sns.lmplot(
    x='GrLivArea', y='SalePrice', hue='Exterior1st', col='Exterior1st',
    data=train, col_wrap=7, height=6,)


# In[ ]:


sns.lmplot(
    x='SqrFeet', y='SalePrice', hue='PoolQC', scatter_kws={"edgecolor": 'w'}, 
    data=train)


# In[ ]:


sns.lmplot(
    x='SqrFeet', y='SalePrice', hue='MiscFeature', scatter_kws={"edgecolor": 'w'}, 
    data=train)


# Now I can merge again the dataframe.

# In[ ]:


train = pd.concat([train, sc_train])


# In[ ]:


train.shape


# # Feature engineering

# ### Summary after plotting

# #####  Considerations
# 
# 0) Since many features have few instances, I will simplify some of them aggregating the values.
# 
# 1) I will keep all features related to sale conditions. I think they can affect the price more than quality, condition or facilities of the house.
# 
# 2) GarageYrBlt could be explained by YearBuilt, so I will drop it.
# 
# 3) I will create the number of other kinds of rooms, I will check if there are more than one kitchen, and I will aggregate the number of baths and the areas of porch.
# 
# 4) Since an detached garage needs an own framework and roof, this will impact on the value as well as the area, so I will distinguish if it is attached or not to the house.
# 
# 4) Regarding the external factors, I will consider: neighborhoods, MSZoning and conditions. There are many neighborhoods and some of them have few instances and the same for conditions, so I will aggregate them building hypothetical desirable living zones based on general quality, remod year, number of cars, near conditions and residential density. Since every neighborhood has different conditions and residential density, it's likely that within every of them there are some areas more desirable and others less. I.e. Oldtown has tree kinds of density and a large YearBuilt range. I will make a measure of conditions considering: adjacency to arterial, feeder streets and railroads as a negative condition, and as positive conditions the ones indicated as positive. 
# Looking at the kind of house I will divide in two categories: Townhouse and not. 
# About HouseStyle, I believe that, more important than the style of dwelling is how the house was declared during the sale deal, so I will drop it as there is already MSSubClass.
# 
# 5) I will keep only some of the features related to facilities to identify some kinds of specific houses. 

# In[ ]:


def add_features (df):

    add_df = df.copy() 
    
    ### Sale conditions
    add_df['SplSaleCondition'] = add_df.SaleCondition.apply(lambda x : 'Family' if x == 'Family' else ('Abnorml' if x == 'Abnorml' else ('Partial' if x == 'Partial' else 'Normal')))
    add_df['SplSaleType'] = add_df.SaleType.apply(lambda x: 'COD' if x == 'COD' else 'other')
    add_df['SplMSSubClass'] = add_df.MSSubClass.replace({'85': '80', '75': '70', '180': '120', '45': '40', '150': '120'})
   
    # Year
    add_df['AgeHouse'] = (add_df.YrSold - add_df.YearBuilt).apply(lambda x: x if x>0 else 0)
    add_df['AgeRemod'] = (add_df.YrSold - add_df.YearRemodAdd).apply(lambda x: x if x>0 else 0)
    
    # Surfaces
    add_df['TotOtherRoomAbvGr'] = add_df['TotRmsAbvGrd'] - add_df['KitchenAbvGr'] - add_df['BedroomAbvGr'] 
    add_df['TotRmsAbvGrd'] = add_df.TotRmsAbvGrd.apply(lambda x: 10 if x > 10 else x) 
    add_df['SplKitchenAbvGr'] = add_df.KitchenAbvGr.apply(lambda x: 'Y' if x > 1 else 'N') 
    add_df['TotFullBath'] = add_df.FullBath + add_df.BsmtFullBath
    add_df['TotHalfBath'] = add_df.HalfBath + add_df.BsmtHalfBath 
    add_df['TotPorchSF'] = add_df['OpenPorchSF'] + add_df['EnclosedPorch'] + add_df['3SsnPorch'] + add_df['ScreenPorch']
    add_df['Fireplaces'] = add_df.Fireplaces.apply(lambda x: 2 if x > 2 else x)

    add_df['SplBsmtCond'] = add_df.BsmtCond.replace({'None': 'N', 'NA': 'N', 'Po' : 'Low', 'Fa' : 'Low', 'TA' : 'Norm', 'Gd' : 'High', 'Ex' : 'High'})
    add_df['SplBsmtExposure'] = add_df.BsmtExposure.replace({'None': 'N', 'NA': 'N', 'No' : 'N', 'Mn' : 'Mn', 'Av' : 'Av', 'Gd' : 'Gd'})
    add_df['SplExterCond'] = add_df.ExterCond.replace({'None': 'Low', 'NA': 'Low', 'Po' : 'Low', 'Fa' : 'Low', 'TA' : 'Norn', 'Gd' : 'High', 'Ex' : 'High'})
    
    # Garage
    add_df['GarageCars'] = add_df.GarageCars.apply(lambda x: 3 if x > 3 else x)
    add_df['SplGarageType'] = add_df.GarageType.replace({'CarPort': 'Attchd', 'Detchd': 'Detchd', 'Basment': 'Attchd', '2Types': 'Attchd', 'Attchd': 'Attchd', 'BuiltIn': 'Attchd'})
    add_df['SplGarageFinish'] = add_df.GarageFinish.replace({'None': 'N', 'NA': 'N', 'Unf' : 'Unf', 'RFn' : 'RFn', 'Fin' : 'Fin'})
     
    # Living zones
    add_df['SplMSZoning'] = add_df.MSZoning.replace({'C (all)': 'Low', 'None':'Norm', 'RM': 'Low', 'RH' : 'Low', 'RL' : 'Norm', 'FV': 'Norm'})
    add_df['DesMSZoning'] = add_df.MSZoning.replace({'C (all)': 1, 'None':2, 'RM': 1, 'RH' : 1, 'RL' : 2, 'FV': 2})
    add_df['SplCondition1'] = add_df.Condition1.replace({'Feedr': -1, 'Artery': -1, 'RRAn': -1, 'RRAe': -1,
                                                        'Norm': 0,  'RRNn': 0, 'RRNe': 0,  
                                                        'PosN': 1, 'PosA': 1,})    
    add_df['SplCondition2'] = add_df.Condition2.replace({'Feedr': -1, 'Artery': -1, 'RRAn': -1, 'RRAe': -1,
                                                        'Norm': 0,  'RRNn': 0, 'RRNe': 0,  
                                                        'PosN': 1, 'PosA': 1,})      
    add_df['SplTotCondition'] = (add_df['SplCondition1'] + add_df['SplCondition2']).apply(lambda x: -1 if x < 0 else(1 if x > 0 else 0)) 
    add_df['DesNeighborhood'] = add_df.Neighborhood.replace({'MeadowV': 1, 'IDOTRR': 1, 'BrkSide': 1, 'SWISU': 1,
                                                             'Edwards': 2, 'OldTown': 2,  'BrDale': 2, 'NAmes': 2, 'Sawyer': 2,
                                                             'Mitchel': 3, 'NPkVill': 3, 'ClearCr': 3, 'Crawfor': 3, 'Blueste': 3, 'NWAmes': 3,  
                                                             'SawyerW': 4, 'CollgCr': 4, 'Gilbert': 4, 'Veenker': 4,  'Timber': 4, 
                                                             'Somerst': 5, 'Blmngtn': 5, 'NoRidge': 5, 'StoneBr': 5, 'NridgHt': 5})
    add_df['DesZoning'] = (add_df['DesNeighborhood'] + add_df['DesMSZoning'] + add_df['SplTotCondition']).apply(lambda x: 2 if x < 2 else(7 if x > 7 else x))
    add_df['SplBldgType'] = add_df.BldgType.replace({'1Fam': 'Other', 'Twnhs':'Twn', 'TwnhsE': 'Twn', 'Duplex' : 'Other', '2fmCon' : 'Other'})

    # Facilities/Other
    add_df['SplMasVnrType'] = add_df.MasVnrType.apply(lambda x : 'None' if x == 'None' else ('Stone' if x == 'Stone' else 'Other'))
    add_df['SplFoundation'] = add_df.Foundation.replace({'PConc': 'Mod', 'CBlock': 'Old', 'BrkTil': 'Anc',
                                                         'Wood': 'Mod',  'Slab': 'Old', 'Stone': 'Old'})
    add_df['SplLotShape'] = add_df.LotShape.apply(lambda x : 'Reg' if x == 'Reg' else 'Ir')
    add_df['SplRoofStyle'] = add_df.RoofStyle.apply(lambda x : 'Flat' if x == 'Flat' else ('Shed' if x == 'Shed' else 'Other'))
    add_df['SplFunctional'] = add_df.Functional.apply(lambda x: 'Other' if x == 'Min2' or x == 'Mod' or x == 'Min1' else 'Typ')
    add_df['SplPavedDrive'] = add_df.PavedDrive.apply(lambda x: 'Y' if x == 'Y' or x == 'Mix' else 'N') 

    add_df['YrSold'] = add_df.YrSold.astype('str')
    add_df['MoSold'] = add_df.MoSold.astype('str')
    add_df['SplCondition1'] = add_df.SplCondition1.astype('int64') 
    add_df['SplCondition2'] = add_df.SplCondition2.astype('int64') 
    add_df['SplTotCondition'] = add_df.SplTotCondition.astype('int64') 
    add_df['DesNeighborhood'] = add_df.DesNeighborhood.astype('str') 
    add_df['DesZoning'] = add_df.DesZoning.astype('str') 

    return add_df
  


# In[ ]:


tmp_train = add_features(train)
tmp_test = add_features(test)


# In[ ]:


cat, num, flo = find_types(tmp_train)


# In[ ]:


fig = plt.figure(figsize=(15,15))
corrMatrix = tmp_train[[*flo, *num, 'SalePrice']].corr()
sns.heatmap(corrMatrix, annot=False, cmap="YlGnBu")


# In[ ]:


tmp_train.corr()['SalePrice'].sort_values(ascending=False).head(11)


# In[ ]:


tmp_train.corr()['SalePrice'].sort_values(ascending=False).tail(5)


# In[ ]:


lflo = len(flo)
nflo = train.loc[:, train.columns.map(lambda x: x in flo)].nunique()
fig, axes = plt.subplots(ceil((lflo)/3), 3, figsize=(15, ceil(lflo/3)*4))
for ax, i in zip(axes.flat, flo):
    sns.scatterplot(ax=ax, x=tmp_train[i], y=tmp_train.SalePrice, alpha=0.2)
    ax.set_title(i) 
    
fig.tight_layout()


# In[ ]:


lnum = len(num)
fig, axes = plt.subplots(ceil((lnum)/5), 5, figsize=(15, ceil(lnum/5)*3))

for ax, i in zip(axes.flat, num):
    sns.scatterplot(ax=ax, x=tmp_train[i], y=tmp_train.SalePrice, alpha=0.2)
    ax.set_title(i) 
    
fig.tight_layout()


# # Creating dummy features

# In[ ]:


tmp_train = tmp_train.drop(columns='FootPrice')


# In[ ]:


tmp_test.insert(48, "SalePrice", pd.Series())


# In[ ]:


conc = pd.concat([tmp_train, tmp_test], axis=0)


# In[ ]:


cat, num, flo = find_types(tmp_train)


# In[ ]:


def dum_cat(df):
    for i in cat:
        cls_dum = pd.get_dummies(df[i], prefix=i[0:12], drop_first=True)
        for x in cls_dum.columns:
            df[x] = cls_dum[x]
    
    df.drop(columns=cat, inplace=True)    
    return df 


# In[ ]:


dum_cat(conc)


# In[ ]:


conc.isnull().sum()[conc.isnull().sum()!=0]


# In[ ]:


tmp_test = conc.loc[conc.SalePrice.isnull(),:]
tmp_train = conc.loc[~conc.SalePrice.isnull(),:]
tmp_test.drop(columns='SalePrice', inplace=True)


# In[ ]:


cat, num, flo = find_types(tmp_train)
cat


# # Trying linear regression

# I will test candidate features with a simple linear regression and, since the evalutation will be based on the logarithm of the predicted value and on the logarithm of the observed sales price, I will create a simple function of performance using neg_mean_squared_log_error.

# In[ ]:


def perf_mod (model, X, y):
    kf = KFold(10, shuffle=True, random_state=0).get_n_splits(X)
    rmse= (np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_log_error", cv=kf)))
    lin = model.fit(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)  
    test_pred = lin.predict(X_test)
    print('RMSE_Cv: ', np.round(rmse, 4))
    print('RMSE_Cv_Max: ', np.round(rmse.max(), 4), ', RMSE_Cv_min: ', np.round(rmse.min(), 4))
    print('RMSE_Cv_Mean: ', np.round(np.median(rmse), 4), ', RMSE_Cv_Std: ', np.round(rmse.std(), 4))
    print('RMSE_test: ', (np.round(np.sqrt(mean_squared_log_error(y_test, test_pred)), 4)))


# In[ ]:


features = tmp_train.columns[tmp_train.columns.str.match(r'LotArea|OverallQual|OverallCond|MasVnrArea|BsmtUnfSF|TotalBsmtSF|TotHalfBath|TotOtherRoomAbvGr|GrLivArea|Fireplaces|AgeRemod|EnclosedPorch|ScreenPorch|NumExterQual|GarageCars|GarageArea|WoodDeckSF|MiscVal|NumHeatingQC|NumKitchenQual|AgeHouse|TotFullBath|TotPorchSF|SplTotCondition|SplSaleCond|SplKitchenAb|SplMSZoni|DesZon|SplMasVnrTy|SplMSSubCl|CentralAir_Y|SplFound|SplBsmtExpo|SplGarageFi|NumBsmtQual|SplExterCond|YrSold|MoSold|SplSaleTy|SplFunction|SplBldgType|SplRoofStyle|SplLotShap|SplPavedDr|AgeHouseOverallCond|MiscValLotArea')]


# In[ ]:


y = tmp_train.SalePrice
X = tmp_train.loc[:,features]
X = sm.add_constant(X, prepend=False)
result = sm.OLS(y, X).fit()
pred = result.predict(X)
result.summary()


# In[ ]:


perf_mod(LinearRegression(), X.drop(columns='const'), y)


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(20,4))
sns.residplot(ax=axes[0], x=pred, y=stats.zscore(result.resid), data=X, lowess=True, scatter_kws={'alpha': 0.5}, 
                     line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
sns.histplot(ax=axes[1], x=result.resid, element="step", color="orange", kde=True)
stats.probplot(result.resid, dist="norm", plot=axes[2])
sns.scatterplot(x=pred, y=y)
(mu, sigma) = norm.fit(result.resid)
plt.legend(['$\mu=$ {:.2f} and $\sigma=$ {:.2f}'.format(mu, sigma)],
            loc='best')


# # Violated assumptions

# Looking at the residual plot, it is clear that many of the linear regression assumptions are violated. There is an evident problem of linearity and of heteroscedasticity, that was hinted looking at the plot of areas.
# Expectation value of residuals is zero and, as I only want to predict the target value, I can avoid concern about the normality of residuals.
# I fixed the problems through a log transforming of SalePrice, but before that I tried the following methods: redefining the target using the price of square feet instead of SalePrice; creating polynomial features associated with a WLS model. 
# So I will go on with log transformation.

# In[ ]:


y = np.log(tmp_train.SalePrice)
X = tmp_train.loc[:,features]
X = sm.add_constant(X, prepend=False)
result = sm.OLS(y, X).fit()
pred = result.predict(X)
result.summary()


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(20,4))
sns.residplot(ax=axes[0], x=pred, y=stats.zscore(result.resid), data=X, lowess=True, scatter_kws={'alpha': 0.5}, 
                     line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
sns.histplot(ax=axes[1], x=result.resid, element="step", color="orange", kde=True)
stats.probplot(result.resid, dist="norm", plot=axes[2])
sns.scatterplot(x=pred, y=y)
(mu, sigma) = norm.fit(result.resid)
plt.legend(['$\mu=$ {:.2f} and $\sigma=$ {:.2f}'.format(mu, sigma)],
            loc='best')


# In[ ]:


fig, ax = plt.subplots(figsize=(20,80))
sm.graphics.plot_ccpr_grid(result, fig=fig)
fig.tight_layout(pad=2.0)


# Now that I have redefined the target variable, I need to change the performance function using neg_mean_squared_error instead of neg_mean_squared_log_error, since the predict values and the target are already in the log terms.

# In[ ]:


def perf_mod (model, X, y):
    kf = KFold(10, shuffle=True, random_state=0).get_n_splits(X)
    rmse= (np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf)))
    lin = model.fit(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)  
    test_pred = lin.predict(X_test)
    print('RMSE_Cv: ', np.round(rmse, 4))
    print('RMSE_Cv_Max: ', np.round(rmse.max(), 4), ', RMSE_Cv_min: ', np.round(rmse.min(), 4))
    print('RMSE_Cv_Mean: ', np.round(np.median(rmse), 4), ', RMSE_Cv_Std: ', np.round(rmse.std(), 4))
    print('RMSE_test: ', (np.round(np.sqrt(mean_squared_error(y_test, test_pred)), 4)))


# In[ ]:


corrMatrix = X.drop(columns=['const']).corr()
crrm = corrMatrix.loc[((abs(corrMatrix) > 0.7) & (abs(corrMatrix) < 1.0)).any(),((abs(corrMatrix) > 0.7) & (abs(corrMatrix) < 1.0)).any()]
crrm = crrm.style.applymap(style_negative, props='color:red;')              .applymap(lambda v: 'opacity: 20%;' if (v < 0.7) and (v > -0.7) else None)
crrm


# After resolving the two problems with assumptions, the next problems is multicollinearity. As I would like to use all the features I selected instead of removing the high correlated ones, I will use a ridge regression as a final model.

# # Interaction effects

# Now I will add two interaction terms. From the plot of AgeHouse vs OverallCond I noticed that the conditions of all new houses have a medium value, even though recent houses have a high quality score, so it's likely that a standard condition score is applied to the new buildings. This means that high values of both features do not always lead to a high price.
# Regarding LotArea, some houses have additional features, like a shed, which have impact on the lot size, but the additional value of the lot is already explained by MiscVal.

# In[ ]:


def poly_add (df):
    
    poly_df = df.copy()

    poly_df['AgeHouseOverallCond'] = poly_df['AgeHouse'] * poly_df['OverallCond']
    poly_df['MiscValLotArea'] = poly_df['MiscVal'] * poly_df['LotArea']
 
    return poly_df


# In[ ]:


poly_train = poly_add(tmp_train)
poly_test = poly_add(tmp_test)


# In[ ]:


features = poly_train.columns[poly_train.columns.str.match(r'LotArea|OverallQual|OverallCond|MasVnrArea|BsmtUnfSF|TotalBsmtSF|TotHalfBath|TotOtherRoomAbvGr|GrLivArea|Fireplaces|AgeRemod|EnclosedPorch|ScreenPorch|NumExterQual|GarageCars|GarageArea|WoodDeckSF|MiscVal|NumHeatingQC|NumKitchenQual|AgeHouse|TotFullBath|TotPorchSF|SplTotCondition|SplSaleCond|SplKitchenAb|SplMSZoni|DesZon|SplMasVnrTy|SplMSSubCl|CentralAir_Y|SplFound|SplBsmtExpo|SplGarageFi|NumBsmtQual|SplExterCond|YrSold|MoSold|SplSaleTy|SplFunction|SplBldgType|SplRoofStyle|SplLotShap|SplPavedDr|AgeHouseOverallCond|MiscValLotArea')]


# In[ ]:


y = np.log(poly_train.SalePrice)
X = poly_train.loc[:,features]
X = sm.add_constant(X, prepend=False)
result = sm.OLS(y, X).fit()
pred = result.predict(X)
result.summary()


# In[ ]:


perf_mod(LinearRegression(), X.drop(columns='const'), y)


# In[ ]:


fig, axes = plt.subplots(1, 4, figsize=(20,4))
sns.residplot(ax=axes[0], x=pred, y=stats.zscore(result.resid), data=X, lowess=True, scatter_kws={'alpha': 0.5}, 
                     line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
sns.histplot(ax=axes[1], x=result.resid, element="step", color="orange", kde=True)
stats.probplot(result.resid, dist="norm", plot=axes[2])
sns.scatterplot(x=pred, y=y)
(mu, sigma) = norm.fit(result.resid)
plt.legend(['$\mu=$ {:.2f} and $\sigma=$ {:.2f}'.format(mu, sigma)],
            loc='best')


# # Ridge regression

# In[ ]:


rid = Ridge(random_state=0).fit(RobustScaler().fit_transform(X.drop(columns='const')), y)
fig, axes = plt.subplots(figsize=(10,10))

ftr = X.columns
indices = np.argsort(rid.coef_)[::-1]
sns.barplot(y=ftr[indices][:50], x=rid.coef_[indices][:50] , orient='h')


# # Tuning parameters

# I will use GridSearchCV to tune parameters.

# In[ ]:


kf = KFold(10, shuffle=True, random_state=0)

def search_pars (model, X, y):
    kf = KFold(10, shuffle=True, random_state=0).get_n_splits(X)
    model.fit(X, y) 
    print('Best_params: ', model.best_params_)
    
    return model.best_params_


# In[ ]:


pipeline = Pipeline([('scale', RobustScaler()),
                     ('ridge', Ridge(random_state=0))])

pars = {#'ridge__alpha': [1e-15, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 5, 10, 15, 20, 25, 30, 45, 50],
        #'ridge__alpha': np.arange(6,14, 1),
        #'ridge__alpha': np.arange(8,10, 0.1),
        #'ridge__alpha': np.arange(8.81,8.99, 0.01),
        'ridge__alpha': [8.81],
       }

ridge = GridSearchCV(pipeline, 
                     pars, 
                     scoring="neg_mean_squared_error",
                     cv=kf)


# In[ ]:


search_pars(ridge, X.drop(columns='const'), y)


# # Final result

# In[ ]:


ridge_final_result = ridge.fit(X.drop(columns='const'), y).predict(poly_test.loc[:,features])


# In[ ]:


tmp_y = np.exp(ridge.predict(X.drop(columns='const')))
tmp_diff = pd.DataFrame({'TmpSalePrice': tmp_y, 'MiscV': X.MiscVal, 'SalePrice': np.exp(y)})
tmp_diff.query("MiscV!=0").sum()


# In[ ]:


final_result = ridge_final_result
price = np.exp(final_result) - poly_test.MiscVal
price


# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(15,5))

sns.scatterplot(ax=axes[0], x=poly_train.GrLivArea, y=train.SalePrice, alpha=1, color='green')
sns.scatterplot(ax=axes[1], x=poly_test.GrLivArea, y=price, alpha=1, color='orange')
sns.scatterplot(ax=axes[2], x=poly_train.GrLivArea, y=train.SalePrice, alpha=1, color='green')
sns.scatterplot(ax=axes[2], x=poly_test.GrLivArea, y=price, alpha=1, color='orange')

fig.tight_layout()


# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(6, 4))
sns.boxplot(x=price)
plt.xlabel('Price') 


# In[ ]:


output = pd.DataFrame({'Id': poly_test.Id, 'SalePrice': price})
output.to_csv('submission.csv', index=False)
print("Submitted successfully!")

