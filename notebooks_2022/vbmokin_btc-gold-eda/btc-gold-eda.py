#!/usr/bin/env python
# coding: utf-8

# <a class="anchor" id="0"></a>
# # BTC & Gold : EDA
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
# 1. [EDA](#3)
# 1. [Time Series EDA](#4)

# ## 1. Import libraries<a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


#!pip install pandas-datareader


# In[ ]:


import os
import io
import pandas as pd
import numpy as np
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

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from scipy import stats
import statsmodels.api as sm
from itertools import product

import warnings
warnings.simplefilter('ignore')


# In[ ]:


# Set main parameters
cryptocurrency = 'BTC'
target = 'Close'
year_start = 2017
date_start = dt.datetime(year_start, 1, 1)
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


data = get_data(cryptocurrency, date_start)
data


# In[ ]:


data_gold = get_data('GOLD', date_start)
data_gold


# ## 3. EDA<a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


data = pd.merge(data, data_gold, on = 'Date', suffixes=('', '_gold'))
data


# In[ ]:


def df_minmax_scaler(df):
    # Data Scalling
    scaler = MinMaxScaler().fit(df)
    df = pd.DataFrame(scaler.transform(df), columns = df.columns, index = df.index)
    df = df.dropna()
    return scaler, df


# In[ ]:


scaler, df = df_minmax_scaler(data.copy())
df


# In[ ]:


df[['Close', 'Close_gold']].plot()


# In[ ]:


df['Close_week'] = df['Close'].rolling(7, center=True).mean()
df['Close_gold_week'] = df['Close_gold'].rolling(7, center=True).mean()


# In[ ]:


df[['Close_week', 'Close_gold_week']].plot(lw=4)


# ## 4. Time Series EDA<a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# Thanks to https://www.kaggle.com/kashnitsky/topic-9-time-series-arima-example

# In[ ]:


df.head()


# In[ ]:


print("Dickey-Fuller criterion: p=%f") 
sm.tsa.stattools.adfuller(df['Close'])[1]


# In[ ]:


shift_days = 10
df['Close_gold_diff'] = df['Close_gold']-df['Close_gold'].shift(shift_days)
df['Close_diff'] = df['Close']-df['Close'].shift(shift_days)


# In[ ]:


def plot_acf_pacf(df, col, lag_list):
    print(f"Lags: {lag_list}")
    #s=lag_list[0]
    s=21
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(211)
    sm.graphics.tsa.plot_acf(df[col][s:].values.squeeze(), lags=lag_list, ax=ax)
    plt.grid()

    ax = plt.subplot(212)
    sm.graphics.tsa.plot_pacf(df[col][s:].values.squeeze(), lags=lag_list, ax=ax)
    ax.axhline(y=1.96/np.sqrt(len(df)), color='r')
    ax.axhline(y=-1.96/np.sqrt(len(df)), color='r')
    plt.grid()
    plt.show()


# In[ ]:


lag_list = [7, 30, 120, 182, 240, 480, 365]
plot_acf_pacf(df, 'Close_diff', lag_list)


# In[ ]:


plot_acf_pacf(df, 'Close', lag_list)


# In[ ]:


lag_list_num = list(range(0, 200, 7))
plot_acf_pacf(df, 'Close', lag_list_num)


# In[ ]:


lag_list_num = list(range(0, 200, 7))
plot_acf_pacf(df, 'Close_diff', lag_list_num)


# In[ ]:


lag_list = [7, 30, 120, 182, 240, 480, 365]
plot_acf_pacf(df, 'Close_gold_diff', lag_list)


# In[ ]:


plot_acf_pacf(df, 'Close_gold', lag_list)


# In[ ]:


lag_list_num = list(range(0, 200, 7))
plot_acf_pacf(df, 'Close_gold_diff', lag_list_num)


# In[ ]:


plot_acf_pacf(df, 'Close_gold', lag_list_num)


# In[ ]:


df = df.dropna()


# In[ ]:


def plot_axcorr(df, col1, col2, maxlags):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(211)
    plt.acorr(x=df[col2], maxlags=maxlags, usevlines=True)
    plt.grid()
    plt.title(f'Autocorrelation function for "{col2}"')
    ax = plt.subplot(212)
    plt.xcorr(x=df[col1], y=df[col2], maxlags=maxlags, usevlines=True)
    plt.grid()
    plt.title(f'Cross-correlation function between "{col1}" and "{col2}"')
    plt.show()


# In[ ]:


plot_axcorr(df, 'Close_gold', 'Close', maxlags=200)


# In[ ]:


plot_axcorr(df, 'Close_gold', 'Close_diff', maxlags=200)


# In[ ]:


plot_axcorr(df, 'Close_gold_diff', 'Close_diff', maxlags=200)


# I hope you find this kernel useful and enjoyable.
# 
# Your comments and feedback are most welcome.

# [Go to Top](#0)
