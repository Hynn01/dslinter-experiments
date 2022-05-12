#!/usr/bin/env python
# coding: utf-8

# # Importing all the modlules

# In[ ]:


import pandas as pd #pandas use for reading the data
import numpy as np#numpy used for the mathematics
import matplotlib.pyplot as plt#matplotlib used for the data visualisation
import seaborn as sns#seaborn is also for visualise the data
from sklearn.preprocessing import StandardScaler#Standardising the data
from sklearn.ensemble import IsolationForest #isolation forest for the remove outliers
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping # Early Stopping Callback in the NN
from kerastuner.tuners import RandomSearch # HyperParameter Tunining
import plotly
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go


# # Desciption of the data

# SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
# 
# MSSubClass: The building class
# 
# MSZoning: The general zoning classification
# 
# LotFrontage: Linear feet of street connected to property
# 
# LotArea: Lot size in square feet
# 
# Street: Type of road access
# 
# Alley: Type of alley access
# 
# LotShape: General shape of property
# 
# LandContour: Flatness of the property
# 
# Utilities: Type of utilities available
# 
# LotConfig: Lot configuration
# 
# LandSlope: Slope of property
# 
# Neighborhood: Physical locations within Ames city limits
# 
# Condition1: Proximity to main road or railroad
# 
# Condition2: Proximity to main road or railroad (if a second is present)
# 
# BldgType: Type of dwelling
# 
# HouseStyle: Style of dwelling
# 
# OverallQual: Overall material and finish quality
# 
# OverallCond: Overall condition rating
# 
# YearBuilt: Original construction date
# 
# YearRemodAdd: Remodel date
# 
# RoofStyle: Type of roof
# 
# RoofMatl: Roof material
# 
# Exterior1st: Exterior covering on house
# 
# Exterior2nd: Exterior covering on house (if more than one material)
# 
# MasVnrType: Masonry veneer type
# 
# MasVnrArea: Masonry veneer area in square feet
# 
# ExterQual: Exterior material quality
# 
# ExterCond: Present condition of the material on the exterior
# 
# Foundation: Type of foundation
# 
# |BsmtQual: Height of the basement
# 
# BsmtCond: General condition of the basement
# 
# BsmtExposure: Walkout or garden level basement walls
# 
# BsmtFinType1: Quality of basement finished area
# 
# BsmtFinSF1: Type 1 finished square feet
# 
# BsmtFinType2: Quality of second finished area (if present)
# 
# BsmtFinSF2: Type 2 finished square feet
# 
# BsmtUnfSF: Unfinished square feet of basement area
# 
# TotalBsmtSF: Total square feet of basement area
# 
# Heating: Type of heating
# 
# HeatingQC: Heating quality and condition
# 
# CentralAir: Central air conditioning
# 
# Electrical: Electrical system
# 
# 1stFlrSF: First Floor square feet
# 
# 2ndFlrSF: Second floor square feet
# 
# LowQualFinSF: Low quality finished square feet (all floors)
# 
# GrLivArea: Above grade (ground) living area square feet
# 
# BsmtFullBath: Basement full bathrooms
# 
# BsmtHalfBath: Basement half bathrooms
# 
# FullBath: Full bathrooms above grade
# 
# HalfBath: Half baths above grade
# 
# Bedroom: Number of bedrooms above basement level
# 
# Kitchen: Number of kitchens
# 
# KitchenQual: Kitchen quality
# 
# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# 
# Functional: Home functionality rating
# 
# Fireplaces: Number of fireplaces
# 
# FireplaceQu: Fireplace quality
# 
# GarageType: Garage location
# 
# GarageYrBlt: Year garage was built
# 
# GarageFinish: Interior finish of the garage
# 
# GarageCars: Size of garage in car capacity
# 
# GarageArea: Size of garage in square feet
# 
# GarageQual: Garage quality
# 
# GarageCond: Garage condition
# 
# PavedDrive: Paved driveway
# 
# WoodDeckSF: Wood deck area in square feet
# 
# OpenPorchSF: Open porch area in square feet
# 
# EnclosedPorch: Enclosed porch area in square feet
# 
# 3SsnPorch: Three season porch area in square feet
# 
# ScreenPorch: Screen porch area in square feet
# 
# PoolArea: Pool area in square feet
# 
# PoolQC: Pool quality
# 
# Fence: Fence quality
# 
# MiscFeature: Miscellaneous feature not covered in other categories
# 
# MiscVal: $Value of miscellaneous feature
# 
# MoSold: Month Sold
# 
# YrSold: Year Sold
# 
# SaleType: Type of sale
# 
# SaleCondition: Condition of sale
# 
# "

# # Reading the data

# In[ ]:


train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")


# In[ ]:


train.head()


# **Checking How many null variable it have**

# In[ ]:


train.isna().sum()


# # assigning train and test data 

# In[ ]:


test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
y = train['SalePrice'].values
data = pd.concat([train,test],axis=0,sort=False)
data.drop(['SalePrice'],axis=1,inplace=True)
data.head()


# In[ ]:


data.info()


# # Checking how many column have a categorical and nymerical data type

# In[ ]:


column_data_type = []
for col in data.columns:
    data_type = data[col].dtype
    if data[col].dtype in ['int64','float64']:
        column_data_type.append('numeric')
    else:
        column_data_type.append('categorical')
plt.figure(figsize=(15,5))
sns.countplot(x=column_data_type)
plt.show()


# **Checking how many column have missing values**

# In[ ]:


missing_values = data.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending = False)
missing_values
NAN_col = list(missing_values.to_dict().keys())
missing_values_data = pd.DataFrame(missing_values)
missing_values_data.reset_index(level=0, inplace=True)
missing_values_data.columns = ['Feature','Number of Missing Values']
missing_values_data['Percentage of Missing Values'] = (100.0*missing_values_data['Number of Missing Values'])/len(data)
missing_values_data


# # Filling all the missing values

# In[ ]:


data['BsmtFinSF1'].fillna(0, inplace=True)
data['BsmtFinSF2'].fillna(0, inplace=True)
data['TotalBsmtSF'].fillna(0, inplace=True)
data['BsmtUnfSF'].fillna(0, inplace=True)
data['Electrical'].fillna('FuseA',inplace = True)
data['KitchenQual'].fillna('TA',inplace=True)
data['LotFrontage'].fillna(data.groupby('1stFlrSF')['LotFrontage'].transform('mean'),inplace=True)
data['LotFrontage'].interpolate(method='linear',inplace=True)
data['MasVnrArea'].fillna(data.groupby('MasVnrType')['MasVnrArea'].transform('mean'),inplace=True)
data['MasVnrArea'].interpolate(method='linear',inplace=True)


# In[ ]:


for col in NAN_col:
    data_type = data[col].dtype
    if data_type == 'object':
        data[col].fillna('NA',inplace=True)
    else:
        data[col].fillna(data[col].mean(),inplace=True)


# In[ ]:


data['Total_Square_Feet'] = (data['BsmtFinSF1'] + data['BsmtFinSF2'] + data['1stFlrSF'] + 
                                                                 data['2ndFlrSF'] + data['TotalBsmtSF'])

data['Total_Bath'] = (data['FullBath'] + (0.5 * data['HalfBath']) + data['BsmtFullBath'] + 
                                                                  (0.5 * data['BsmtHalfBath']))

data['Total_Porch_Area'] = (data['OpenPorchSF'] + data['3SsnPorch'] + 
                                                data['EnclosedPorch'] + data['ScreenPorch'] + data['WoodDeckSF'])

data['SqFtPerRoom'] = data['GrLivArea'] / (data['TotRmsAbvGrd'] + data['FullBath'] +
                                                       data['HalfBath'] + data['KitchenAbvGr'])


# In[ ]:


data=pd.get_dummies(data)
data.head()


# In[ ]:


train = data[:1460].copy()
test = data[1460:].copy()
train['SalePrice'] = y
train.head()


# # Checking the corelation beetween the predicted and training column

# In[ ]:


top_features = train.corr()[['SalePrice']].sort_values(by=['SalePrice'],ascending=False).head(30)
plt.figure(figsize=(5,10))
sns.heatmap(top_features,cmap='rainbow',annot=True,annot_kws={"size": 16},vmin=-1)


# **Making Function for plotting and checking the relation between two**

# In[ ]:


def plot_data(col, discrete=False):
    if discrete:
        fig, ax = plt.subplots(1,2,figsize=(14,6))
        sns.stripplot(x=col, y='SalePrice', data=train, ax=ax[0])
        sns.countplot(train[col], ax=ax[1])
        fig.suptitle(str(col) + ' Analysis')
    else:
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        sns.scatterplot(x=col, y='SalePrice', data=train, ax=ax[0])
        sns.distplot(train[col], kde=False, ax=ax[1])
        fig.suptitle(str(col) + ' Analysis')


# In[ ]:


plot_data('OverallQual',True)


# In[ ]:


train = train.drop(train[(train['OverallQual'] == 10) & (train['SalePrice'] < 200000)].index)


# In[ ]:


plot_data('GrLivArea')


# In[ ]:


plot_data('Total_Bath')


# # Dropping the outliers in the data

# In[ ]:


train = train.drop(train[(train['Total_Bath'] > 4) & (train['SalePrice'] < 200000)].index)


# In[ ]:


plot_data('TotalBsmtSF')


# In[ ]:


train = train.drop(train[(train['TotalBsmtSF'] > 3000) & (train['SalePrice'] < 400000)].index)


# In[ ]:


train.reset_index()


# In[ ]:


clf = IsolationForest(max_samples = 100, random_state = 42)
clf.fit(train)
y_noano = clf.predict(train)
y_noano = pd.DataFrame(y_noano, columns = ['Top'])
y_noano[y_noano['Top'] == 1].index.values

train = train.iloc[y_noano[y_noano['Top'] == 1].index.values]
train.reset_index(drop = True, inplace = True)
print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
print("Number of rows without outliers:", train.shape[0])


# In[ ]:


X = train.copy()
X.drop(['SalePrice'],axis=1,inplace=True)
y = train['SalePrice'].values
X.shape,y.shape


# # Standardising the data

# In[ ]:


scale = StandardScaler()
X = scale.fit_transform(X)


# # Building Model 

# In[ ]:


def build_model(hp):
    model = Sequential()
    for i in range(hp.Int('layers', 2, 10)):
        model.add(Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
    model.add(Dense(1))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mse',
        metrics=['mse'])
    return model


# In[ ]:


tuner = RandomSearch(
    build_model,
    objective='val_mse',
    max_trials=10,
    executions_per_trial=3,
    directory='model_dir',
    project_name='House_Price_Prediction')
tuner.search_space_summary()


# In[ ]:


def create_model():
    # create model
    model = Sequential()
    model.add(Dense(320, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(384, activation='relu'))
    model.add(Dense(352, activation='relu'))
    model.add(Dense(448, activation='relu'))
    model.add(Dense(160, activation='relu'))
    model.add(Dense(160, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    # Compile model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss = 'mse')
    return model


# In[ ]:


model = create_model()
model.summary()


# # Testing the model

# In[ ]:


early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
history = model.fit(x=X,y=y,
          validation_split=0.1,
          batch_size=128,epochs=1000, callbacks=[early_stop])


# In[ ]:


losses = pd.DataFrame(model.history.history)
losses.plot()


# # Making the final model

# In[ ]:


model=create_model()
history = model.fit(x=X,y=y,
          batch_size=128,epochs=170)


# In[ ]:


losses = pd.DataFrame(model.history.history)
losses.plot()


# In[ ]:


model.evaluate(X,y)


# In[ ]:


X_test = scale.transform(test)
result = model.predict(X_test)
result = pd.DataFrame(result,columns=['SalePrice'])
result.head()
result['Id'] = test['Id']
result = result[['Id','SalePrice']]
result.head()


# # Doing the submission

# In[ ]:


result.to_csv('submission.csv',index=False)


# In[ ]:




