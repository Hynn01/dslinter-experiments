#!/usr/bin/env python
# coding: utf-8

# <a class="anchor" id="0"></a>
# # Bitcoin (BTC) Forecasting with Advanced FE (original approach)
# 
# ## Dataset "[Forecasting Top Cryptocurrencies](https://www.kaggle.com/datasets/vbmokin/forecasting-top-cryptocurrencies)"
# ## Data download from API [Yahoo.Finance](https://finance.yahoo.com/cryptocurrencies/)

# ## Acknowledgements:
# * FE - the notebook [GResearch Simple LGB Starter](https://www.kaggle.com/code1110/gresearch-simple-lgb-starter)
# * about cryptocurrencies - dataset [Forecasting Top Cryptocurrencies](https://www.kaggle.com/datasets/vbmokin/forecasting-top-cryptocurrencies)
# * data source via API: https://finance.yahoo.com/cryptocurrencies/

# <a class="anchor" id="0.1"></a>
# 
# ## Table of Contents
# 
# 1. [Import libraries](#1)
# 1. [Download data](#2)
# 1. [FE](#3)
# 1. [Model training](#4)
# 1. [Prediction](#5)
#     - [Training data](#5.1)
#     - [Test data](#5.2)

# ## 1. Import libraries <a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# !pip install tsfresh


# In[ ]:


# Import libraries
import random
import os
import numpy as np 
import pandas as pd 
import requests
import pandas_datareader as web

# Date
import datetime as dt
from datetime import date, timedelta, datetime

# EDA
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

# FE
from tsfresh import extract_features, select_features
from functools import reduce

# Metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

# Modeling and preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


# Set main parameters
cryptocurrency = 'BTC'
target = 'Close'
forecasting_days = 10    # requires > 1


# In[ ]:


# Set random state
def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

random_state = 42
fix_all_seeds(random_state)


# In[ ]:


# Set time interval of data for given cryptocurrency - the period of coronavirus in 2020-2021
date_start = dt.datetime(2020, 4, 1)
# date_end = dt.datetime.now()
date_end = dt.datetime(2021, 12, 31)
print(f"Time interval: from {date_start} to {date_end}")


# ## 2. Download data <a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Download information about cryptocurrencies
df_about = pd.read_csv("../input/forecasting-top-cryptocurrencies/about_top_cryptocurrencies_1B_information.csv", sep=";")
df_about


# In[ ]:


def get_rank_cryptocurrency(df_about, cryptocurrency):
    # Get rank by Market Cap for code of the cryptocurrency
    # df_about from the dataset from https://www.kaggle.com/datasets/vbmokin/forecasting-top-cryptocurrencies
    
    place = df_about.index[df_about['code'] == cryptocurrency].tolist()[0]
    if place==0:
        place_end = 'st'
    elif place < 3:
        place_end = 'nd'
    else: place_end = 'th'
        
    print(f"{df_about.loc[place, 'name']} was {place+1}{place_end}",
          "among the world's cryptocurrencies by market capitalization (2022-04-11)")


# In[ ]:


# Get rank by Market Cap of the cryptocurrency
get_rank_cryptocurrency(df_about, cryptocurrency)


# In[ ]:


def get_data(cryptocurrency, date_start, date_end=None):
    # Get data for given cryptocurrency in USD from Yahoo.finance and https://coinmarketcap.com/
    # date_end = None means that the date_end is the current day
    
    if date_end is None:
        date_end = dt.datetime.now()
    df = web.DataReader(f'{cryptocurrency}-USD', 'yahoo', date_start, date_end)
    df = df[['High', 'Low', 'Open', 'Close', 'Volume']].reset_index(drop=False)
    
    return df


# In[ ]:


# Download data of the cryptocurrency via API
data = get_data(cryptocurrency, date_start, date_end)
data


# ## 3. FE <a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


def get_target(data):
    # Get df and target (regression task)
    #data['Close2'] = data['Close'].diff()                            # for classification task
    #data['Close_growth'] = (data['Close2'] > 0).astype('int')        # for classification task
    #data['target'] = data['Close_growth'].shift(-forecasting_days)   # for classification task
    data['target'] = data['Close'].shift(-forecasting_days)
    data = data.dropna().reset_index(drop=True)
    data['target'] = data['target'].astype('int')
    #data = data.drop(columns=['Close2', 'Close_growth', 'Volume'])   # for classification task
    data = data.drop(columns=['Volume'])
    target = data.pop('target')
    return data, target


# In[ ]:


data, target = get_target(data)
data


# In[ ]:


def get_stat_features(data):
    # Get statistic features using library TSFRESH 
    
    data = data.reset_index(drop=False)
    
    # Extract features
    extracted_features = extract_features(data, column_id="index", column_sort="Date")
    
    # Drop features with NaN
    extracted_features_clean = extracted_features.dropna(axis=1, how='all').reset_index(drop=True)
    
    # Drop features with constants
    extracted_features_clean = extracted_features_clean.loc[:, (extracted_features_clean != extracted_features_clean.iloc[0]).any()]
    
    extracted_features_clean['Date'] = data['Date']   # For the merging
    
    return extracted_features_clean


# In[ ]:


def get_relations_for_one_feature(data, feature):
    # Get relations for feature in data
    
    df = pd.DataFrame(index=data.index)
    df['Date'] = data['Date']   # For For the merging
    time_intervals = [1, 2, 3, 7, 14]
    for time_item in time_intervals:
        df[feature+str(time_item)] = data[feature]/data[feature].shift(-time_item)
    
    return df


# In[ ]:


def get_math_for_one_feature(data, feature):
    # get mathematic relations for feature in data
    
    #print(feature)
    df = pd.DataFrame(index=data.index)
    df['Date'] = data['Date']   # For the merging
    
    try:
        df[feature+'_sqr'] = data[feature] ** 2
    except: pass
        
    try:
        df[feature+'_sqrt'] = data[feature] ** .5
    except: pass

    try:
        df[feature+'_log1p'] = np.log1p(data[feature])
    except: pass
    
    return df


# In[ ]:


def get_features_for_one_feature(data, function_type="get_math_for_one_feature"):
    # Get features
    
    # Result
    data_F = pd.DataFrame(index=data.index)
    data_F['Date'] = data['Date']   # For the merging
    
    # Get features
    data_cols = data.columns.tolist()
    if 'Date' in data_cols:
        data_cols.remove('Date')
    
    # Calculation
    if function_type=="get_math_for_one_feature":
        for col in data_cols:
            df_i = get_math_for_one_feature(data, col)
            #data_F = pd.concat([data_F, df_i], axis=1)
            data_F = data_F.merge(df_i, how='left', on='Date')
    
    elif function_type=="get_relations_for_one_feature":
        pass
    
    return data_F


# In[ ]:


def get_math_for_many_features(data):
    # get mathematic relations for OHLC-features in data
    
    df = pd.DataFrame(index=data.index)
    df['Date'] = data['Date']   # For the merging
    
    df['Upper_Shadow'] = data['High'] - np.maximum(data['Close'], data['Open'])
    df['Lower_Shadow'] = np.minimum(data['Close'], data['Open']) - data['Low']
    df['high2low'] = data['High'] / data['Low']
    
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Advanced FE\n\n# Statistic features\ndata_S = get_stat_features(data)\nprint(f'Received {data_S.shape[1]} statistic features')\n\n# One-Function features\ndata_F = get_features_for_one_feature(data)\nprint(f'Received {data_F.shape[1]} One-Function features')\n\n# Many-Functions features\ndata_Fs = get_math_for_many_features(data)\nprint(f'Received {data_Fs.shape[1]} Many-Function features')")


# In[ ]:


# Statistic features for One-Function features 
data_SF = get_stat_features(data_F)
print(f'Received {data_SF.shape[1]} Statistic features for One-Function features')

# Statistic features for Many-Function features 
data_SFs = get_stat_features(data_Fs)
print(f'Received {data_SFs.shape[1]} Statistic features for Many-Function features')

# One-Function features for Many-Functions features
data_FFs = get_features_for_one_feature(data_Fs)
print(f'Received {data_FFs.shape[1]} One-Function features for Many-Functions features')


# In[ ]:


data_FFs.head(3)


# In[ ]:


def get_data_for_training(df, test_option, test_size):
    # Get test and training datasets
    # test_option = "end" for prediction problem or "random" for EDA
    # test_size in days in the end of the time interval for prediction problem (test_option = "random")
    # test_size in part of the days in all data (1 = 100%) for EDA problem (test_option = "end")
    
    df = df.drop(columns = ['Date'])
    df = df.dropna(how="any")
    y = df.pop('target')
    
    # Standartization data
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
    #display(df)

    if test_option == "random":
        train, test, ytrain, ytest = train_test_split(df, y, test_size=test_size, random_state=random_state)
    else:
        # test_option == "end"
        train_len = len(df)-test_size
        test = df[train_len:]
        train = df[:train_len]
        ytest = y[train_len:]
        ytrain = y[:train_len]
        
    print(f'Get training dataset with {len(train)} rows and test dataset with {len(test)} rows for test option - "{test_option}"')
    
    return train, test, ytrain, ytest


# In[ ]:


def get_data(data_frames, target, forecasting_days):
    # Get data for training from list of dataframes with id in 'Date'
    
    for i in range(len(data_frames)):
        if 'target' in data_frames[i].columns.tolist():
            data_frames[i] = data_frames[i].drop(columns=['target'])
    
    df = reduce(lambda left, right: pd.merge(left,right,on=['Date'], 
                                             how='left'), data_frames)
    print(f'Total received {df.shape[1]} features')
    
    df['target'] = target
    
    train, test, target_train, target_test = get_data_for_training(df, 'end', forecasting_days)
    
    return train, test, target_train, target_test


# ## 4. Model training and prediction <a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


def print_acc(x,y):
    # Calculation and printing errors and metrics for list with real values x and prediction values y
    # Metrics: r2_score, the relative error (WAPE), RMSE
    
    def error_wape(x,y):
        # Calculation of the relative error for list with real values x and prediction values y
        return mean_absolute_error(x,y)*len(x)/sum(x)

    r2 = round(r2_score(x, y),4)
    wape = round(100*error_wape(x,y),2)
    rmse = round(mean_squared_error(x,y, squared=False),4)
    print(f"Errors: r2_score - {r2}, relative error (WAPE) - {wape}%, RMSE - {rmse}")
    
    return r2, wape, rmse


# In[ ]:


def xgb_training(train, target_train):
    # XGB Regressor Training
    
    xgbr = xgb.XGBRegressor() 
    param_grid_xgb = {'max_depth': [5],
                      'n_estimators': [50], 
                      'learning_rate': [0.05],
                      'random_state': [random_state]}
    
    # Training model
    xgb_CV = GridSearchCV(xgbr, param_grid=param_grid_xgb, cv=3, verbose=False)
    xgb_CV.fit(train, target_train)
    
    return xgb_CV


# In[ ]:


def main(res, res_acc, data_frames, target, forecasting_days, col_name):
    # Model tuning and prediction for given list of the dataframes and save it into res[col_name]
    # Errors save into dataframe res_acc
    
    print(col_name)
    train, test, target_train, target_test = get_data(data_frames, target, forecasting_days)
    
    # Model tuning
    model = xgb_training(train, target_train)
    
    # Prediction of test data
    ypred_test = model.predict(test)
    res[col_name] = ypred_test
    
    # Errors
    _, wape, rmse = print_acc(target_test, ypred_test)
    n = len(res_acc)
    res_acc.loc[n,'FE_model'] = col_name
    res_acc.loc[n,'WAPE'] = wape
    res_acc.loc[n,'RMSE'] = rmse
    
    return res, res_acc


# In[ ]:


# Test predictions of the models
res_pred = pd.DataFrame()
target_test = target[(len(target)-forecasting_days):]
res_pred['test_target'] = target_test

res_acc = pd.DataFrame(columns = ['FE_model', 'WAPE', 'RMSE'])


# In[ ]:


# Statistic features
res_pred, res_acc = main(res_pred, res_acc, [data_S], target, forecasting_days, 'S')

# One-Function features
res_pred, res_acc = main(res_pred, res_acc, [data_F], target, forecasting_days, 'F')

# Many-Functions features
res_pred, res_acc = main(res_pred, res_acc, [data_Fs], target, forecasting_days, 'Fs')

# Statistic features for One-Function features 
res_pred, res_acc = main(res_pred, res_acc, [data_SF], target, forecasting_days, 'SF')

# Statistic features for Many-Function features 
res_pred, res_acc = main(res_pred, res_acc, [data_SFs], target, forecasting_days, 'SFs')

# One-Function features for Many-Functions features
res_pred, res_acc = main(res_pred, res_acc, [data_FFs], target, forecasting_days, 'FFs')

# S, SF
res_pred, res_acc = main(res_pred, res_acc, [data_S, data_SF], target, forecasting_days, 'S_SF')

# F, SF
res_pred, res_acc = main(res_pred, res_acc, [data_F, data_SF], target, forecasting_days, 'F_SF')

# F, Fs, FFs
res_pred, res_acc = main(res_pred, res_acc, [data_F, data_Fs, data_FFs], target, forecasting_days, 'F_Fs_FFs')

# All features
data_frames = [data_S, data_F, data_Fs, data_SF, data_SFs, data_FFs]
res_pred, res_acc = main(res_pred, res_acc, data_frames, target, forecasting_days, 'All features')


# In[ ]:


display(res_pred)


# In[ ]:


res_acc.columns = ['FE_model', 'WAPE,%', 'RMSE']
display(res_acc.sort_values(['WAPE,%', 'RMSE']))


# In[ ]:


models = res_acc['FE_model'].tolist()


# In[ ]:


# Building plot for prediction for the test data 
x = np.arange(len(target_test))
plt.figure(figsize=(12,8))
plt.plot(x, target_test, label = "Target values of the test data", color = 'g')
for model_name in models:
    plt.plot(x, res_pred[model_name], label = model_name)
plt.title('Prediction for the test data - relative errors (WAPE), %')
plt.legend(loc='best')
plt.grid(True)


# I hope you find this notebook useful and enjoyable.
# 
# Your comments and feedback are most welcome.

# [Go to Top](#0)
