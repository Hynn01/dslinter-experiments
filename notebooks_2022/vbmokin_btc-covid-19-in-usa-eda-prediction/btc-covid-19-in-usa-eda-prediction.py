#!/usr/bin/env python
# coding: utf-8

# <a class="anchor" id="0"></a>
# # BTC & COVID-19 in USA : EDA
# 
# ## Dataset "[Forecasting Top Cryptocurrencies](https://www.kaggle.com/datasets/vbmokin/forecasting-top-cryptocurrencies)"
# ## Data download from API [Yahoo.Finance](https://finance.yahoo.com/cryptocurrencies/)

# ## Acknowledgements
# 
# * dataset [Forecasting Top Cryptocurrencies](https://www.kaggle.com/datasets/vbmokin/forecasting-top-cryptocurrencies)
# * data source via API: https://finance.yahoo.com/cryptocurrencies/
# * notebook [Topic 9. Time series. ARIMA example](https://www.kaggle.com/code/kashnitsky/topic-9-time-series-arima-example)

# <a class="anchor" id="0.1"></a>
# ## Table of Contents
# 
# 1. [Import libraries](#1)
# 1. [Download data](#2)
# 1. [EDA & FE](#3)
# 1. [Modeling and prediction](#4)

# ## 1. Import libraries<a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# !pip install pandas-datareader
# !pip install xgboost


# In[ ]:


import os
import io
import pandas as pd
import numpy as np
#import pywt
import requests
import pandas_datareader as web

import seaborn as sns

import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = 12, 10

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PIL import Image
from IPython.display import FileLink

import datetime as dt
from datetime import date, timedelta, datetime

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.tree import DecisionTreeRegressor

from scipy import stats
import statsmodels.api as sm
from itertools import product

import xgboost as xgb
from xgboost import plot_tree
from matplotlib.pylab import rcParams

import warnings
warnings.simplefilter('ignore')


# In[ ]:


# Set main parameters
cryptocurrency = 'BTC'
target = 'Close'
covid_feature = 'New_Deaths'  # or "New_Cases"
country = 'USA'  # or ISO-code of country
country_covid_feature = f"{country}_{covid_feature}"
print('country_covid_feature =', country_covid_feature)
year_start = 2020
date_start = dt.datetime(year_start, 2, 1)
date_start


# ## 2. Download data<a class="anchor" id="2"></a>
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


def get_data(crypto_currency, date_start):
    if crypto_currency != 'GOLD':
        data_name = f'{crypto_currency}-USD'
    else: data_name = 'GC=F'
    end= dt.datetime.now()
    return web.DataReader(f'{data_name}', 'yahoo', date_start, end)


# In[ ]:


data_btc = get_data('BTC', date_start)
data_btc


# In[ ]:


def get_covid_data(date_start, covid_feature, country):

    # Thanks https://www.kaggle.com/vbmokin/covid-19-in-70-countries-daily-prophet-forecast
    # Source: https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series
    
    if covid_feature=='New_Cases':
        file = "time_series_covid19_confirmed_global.csv"
        name_feature = 'Cases'
    elif covid_feature=="New_Deaths":
        file = "time_series_covid19_deaths_global.csv"
        name_feature = 'Deaths'
    
    myfile = requests.get(f'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/{file}')
    open('data', 'wb').write(myfile.content)
    global_df = pd.read_csv('data')
    
    if country=='USA':
        code = 'US'
    else: code = country
    
    try:
        global_df = global_df[global_df['Country/Region']==code]
    except:
        print('Non-existent country code given')
        return None

    def convert_date_str(df):
        try:
            df.columns = list(df.columns[:4]) + [datetime.strptime(d, "%m/%d/%y").date().strftime("%Y-%m-%d") for d in df.columns[4:]]
        except:
            print('_convert_date_str failed with %y, try %Y')
            df.columns = list(df.columns[:4]) + [datetime.strptime(d, "%m/%d/%Y").date().strftime("%Y-%m-%d") for d in df.columns[4:]]

    convert_date_str(global_df)
    
    global_df2 = global_df.melt(
        id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], value_vars=global_df.columns[4:], var_name='Date', value_name=name_feature)

    df_covid = global_df2[['Date', name_feature]]
    df_covid[name_feature] = df_covid[name_feature].astype('int').diff()
    #df_covid = df_covid.dropna()
    df_covid = df_covid.fillna(0)

    df_covid['ds'] = pd.to_datetime(df_covid['Date'])
    df_covid = df_covid[df_covid['ds'] > date_start][['ds', name_feature]].reset_index(drop=True)
    df_covid.columns = ['Date', country_covid_feature]

    return df_covid


# In[ ]:


df_covid = get_covid_data(date_start, covid_feature, country)
df_covid


# In[ ]:


def pd_imputing(df, date1, date2, col):
    x1 = float(df[df['Date']==date1][col].head(1))
    x2 = float(df[df['Date']==date2][col].head(1))
    return (x1+x2)/2


# In[ ]:


def df_add(df, date_middle, date1, date2, col=country_covid_feature):
    df = df.append({'Date': datetime.strptime(date_middle, '%Y-%m-%d'), col : pd_imputing(df, date1, date2, col=col)}, ignore_index=True)
    return df


# In[ ]:


df_covid['USA_New_Deaths'].plot()


# In[ ]:


# Only for USA - the imputing missing data
date_anomal = ['2020-10-08', '2020-10-11', '2020-10-12', '2020-10-25']
df_covid = df_add(df_covid, '2020-10-08', '2020-10-07', '2020-10-09')
df_covid = df_add(df_covid, '2020-10-11', '2020-10-10', '2020-10-13')
df_covid = df_add(df_covid, '2020-10-12', '2020-10-11', '2020-10-14')
df_covid = df_add(df_covid, '2020-10-25', '2020-10-24', '2020-10-26')
df_covid.tail(6)


# In[ ]:


df_covid = df_covid.sort_values(by=['Date']).reset_index(drop=True)
df_covid


# In[ ]:


df_covid[country_covid_feature].plot()


# ## 3. EDA & FE<a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


data = pd.merge(data_btc, df_covid, on = 'Date')
data.index = data['Date']
data = data[[target, 'Volume', country_covid_feature]][53:]
data


# In[ ]:


data = data.reset_index(drop=False)


# In[ ]:


data.describe()


# In[ ]:


data.info()


# In[ ]:


def df_minmax_scaler(df):
    # Data Scalling
    index_df = df.pop('Date')
    scaler = MinMaxScaler().fit(df)
    df = pd.DataFrame(scaler.transform(df), columns = df.columns, index = index_df)
    return scaler, df


# In[ ]:


scaler, df = df_minmax_scaler(data.copy())
df


# In[ ]:


df.describe()


# In[ ]:


df[[target, country_covid_feature]].plot()


# In[ ]:


target2 = target+"-7days-MA"
df[target2] = df[target].rolling(7).mean()
covid2 = country_covid_feature + "-7days-MA"
df[covid2] = df[country_covid_feature].rolling(7).mean()
df['covid'] = df[covid2].shift(80)
df[[target2, covid2]].plot(lw=4)


# In[ ]:


df2 = df[[target2, covid2]]
for i in range(20):
    df2['covid'+str(i)] = df2[covid2].shift(i*7)
df2 = df2.fillna(0)
df2.plot()
df2[target2].plot()


# In[ ]:


df2.head(2)


# In[ ]:


df2 = df2[[target2, covid2, 'covid1', 'covid4', 'covid14', 'covid16']]
# df2 = df2[[target2, covid2, 'covid1', 'covid2', 'covid3', 'covid4', 
#                             'covid5', 'covid6', 'covid10', 'covid11']]
df2.plot()
df2[target2].plot()


# ## 4. Modeling and prediction<a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


def error_wape(x,y):
    # Calculation of the relative error for list with real values x and prediction values y
    return mean_absolute_error(x,y)*len(x)/sum(x)


# In[ ]:


def print_acc(x,y):
    # Calculation and printing errors and metrics for list with real values x and prediction values y
    # Metrics: r2_score, the relative error (WAPE), RMSE
    r2 = round(r2_score(x, y),4)
    wape = round(100*error_wape(x,y),2)
    rmse = round(mean_squared_error(x,y, squared=False),4)
    print(f"Errors: r2_score - {r2}, relative error (WAPE) - {wape}%, RMSE - {rmse}")


# In[ ]:


# Split training set to validation subsets
target_new = df2.pop(target2)
Xtrain, Xtest, Ztrain, Ztest = train_test_split(df2, target_new, test_size=0.2, random_state=0)

# Random permutation cross-validator
cv_train = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)


# In[ ]:


get_ipython().run_cell_magic('time', '', "decision_tree = DecisionTreeRegressor()\nparam_grid = {'max_depth': [i for i in range(2,15)], \n              'min_samples_leaf': [i for i in range(2,5)]}\ndecision_tree_CV = GridSearchCV(decision_tree, param_grid=param_grid, cv=cv_train, verbose=False)\ndecision_tree_CV.fit(Xtrain, Ztrain)\nprint(decision_tree_CV.best_params_)")


# In[ ]:


Ytest = decision_tree_CV.predict(Xtest)
print_acc(Ztest, Ytest)


# In[ ]:


def xgb_training(train, target_train, cv_train):
    # XGB Classifier Training
    #eval_metric_model = 'error'
    xgbr = xgb.XGBRegressor() 
    param_grid_xgb = {'max_depth': [i for i in range(2,10)],
                      'random_state': [0]}

    # Training model
    xgb_CV = GridSearchCV(xgbr, param_grid=param_grid_xgb, cv=cv_train, verbose=False)
    xgb_CV.fit(train, target_train)
    xgbp = xgb_CV.best_params_
    print(xgbp)

    # Feature importance diagrame
    xgb_model = xgb.XGBRegressor(max_depth=5,
                                  random_state=0).fit(train, target_train)
    fig =  plt.figure(figsize = (15,15))
    axes = fig.add_subplot(111)
    xgb.plot_importance(xgb_model,ax = axes,height = 0.5)
    plt.show();plt.close()
    
    return xgb_CV, xgbp, xgb_model


# In[ ]:


xgb_CV, xgbp, xgb_model = xgb_training(Xtrain, Ztrain, cv_train)


# In[ ]:


Ytest = xgb_model.predict(Xtest)
print_acc(Ztest, Ytest)


# In[ ]:


df_res = df2[[covid2]]
df_res[target2] = target_new
df_res['DTR prediction'] = decision_tree_CV.predict(df2)
df_res['XGB prediction'] = xgb_model.predict(df2)


# In[ ]:


df_res.plot()


# In[ ]:


df_res


# In[ ]:


df = df.reset_index(drop=False)
res = pd.merge(df_res, df[['Date', 'Close', covid2]], on='Date', how='left').dropna().reset_index(drop=True)


# In[ ]:


res['Close_without_covid_DTR'] = res['Close'] - res['DTR prediction']
res['Close_without_covid_XGB'] = res['Close'] - res['XGB prediction']
res[['Close', 'Close_without_covid_DTR', 'Close_without_covid_XGB']].plot()


# In[ ]:


res


# In[ ]:


print("Dickey-Fuller criterion for 'Close_without_covid_DTR': p=%f", sm.tsa.stattools.adfuller(res['Close_without_covid_DTR']))
print("Dickey-Fuller criterion for 'Close_without_covid_XGB': p=%f", sm.tsa.stattools.adfuller(res['Close_without_covid_XGB']))


# I hope you find this kernel useful and enjoyable.
# 
# Your comments and feedback are most welcome.

# [Go to Top](#0)
