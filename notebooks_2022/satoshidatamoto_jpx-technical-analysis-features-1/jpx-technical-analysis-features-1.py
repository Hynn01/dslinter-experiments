#!/usr/bin/env python
# coding: utf-8

# >### Technical Analysis Indicators #1
# >- Here are some simple indexes to analyze the charts. some can even be used as features to a model.
# >- Ta-lib is very good and very helpful library for calculating various indexes, but kernel doesn't support.
# >- Enjoy the short scripts to obtain them! 
# >
# >Based on: https://www.kaggle.com/youhanlee/simple-quant-features-using-python
# 
# [1]: TBD

# #### Code starts here ⬇

# In[ ]:


import os
import gc
import traceback
import numpy as np
import pandas as pd
import datatable as dt
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.core.common.SettingWithCopyWarning)
    
plt.style.use('bmh')
plt.rcParams['figure.figsize'] = [14, 8]  # width, height


# # <span class="title-section w3-xxlarge" id="loading">Data Loading 🗃️</span>
# <hr>
# 
# The data organisation has already been done and saved to Kaggle datasets. Here we choose which years to load. We can use either 2017, 2018, 2019, 2020, 2021, Original, Supplement by changing the `INC2021`, `INC2020`, `INC2019`, `INC2018`, `INC2017`, `INCCOMP`, `INCSUPP` variables in the preceeding code section. These datasets are discussed [here][1].
# 
# [1]: https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/285726
# 

# In[ ]:


# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype.name

        if col_type not in ['object', 'category', 'datetime64[ns, UTC]']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# In[ ]:


stock_list = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/stock_list.csv")
prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
stock_list = stock_list.loc[stock_list['SecuritiesCode'].isin(prices['SecuritiesCode'].unique())]
stock_name_dict = {stock_list['SecuritiesCode'].tolist()[idx]: stock_list['Name'].tolist()[idx] for idx in range(len(stock_list))}

def load_training_data(asset_id = None):
    prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
    supplemental_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv")
    df_train = pd.concat([prices, supplemental_prices]) if INCSUPP else prices
    df_train = pd.merge(df_train, stock_list[['SecuritiesCode', 'Name']], left_on = 'SecuritiesCode', right_on = 'SecuritiesCode', how = 'left')
    df_train['date'] = pd.to_datetime(df_train['Date'])
    df_train['year'] = df_train['date'].dt.year
    if not INC2022: df_train = df_train.loc[df_train['year'] != 2022]
    if not INC2021: df_train = df_train.loc[df_train['year'] != 2021]
    if not INC2020: df_train = df_train.loc[df_train['year'] != 2020]
    if not INC2019: df_train = df_train.loc[df_train['year'] != 2019]
    if not INC2018: df_train = df_train.loc[df_train['year'] != 2018]
    if not INC2017: df_train = df_train.loc[df_train['year'] != 2017]
    # asset_id = 1301 # Remove before flight
    if asset_id is not None: df_train = df_train.loc[df_train['SecuritiesCode'] == asset_id]
    # df_train = df_train[:1000] # Remove before flight
    return df_train


# In[ ]:


# WHICH YEARS TO INCLUDE? YES=1 NO=0
INC2022 = 1
INC2021 = 1
INC2020 = 1
INC2019 = 1
INC2018 = 1
INC2017 = 1
INCSUPP = 1

train = load_training_data().sort_values('date').set_index("date")
print("Loaded all data!")


# # <span class="title-section w3-xxlarge" id="features">Feature Engineering 🔬</span>
# <hr>

# In[ ]:


import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
plt.style.use('seaborn')
sns.set(font_scale=2)
import warnings; warnings.filterwarnings('ignore')


# In[ ]:


train_data = train.copy()
train_data['date'] = pd.to_datetime(train_data['Date'])


# In[ ]:


df = train_data.loc[train_data['SecuritiesCode'] == 1301]


# In[ ]:


N=100

df['timestamp'] = df['date']
df.set_index(df['timestamp'], inplace=True)
df.drop('timestamp', axis=1, inplace=True)

convertion={
    'Open':'first',
    'High':'max',
    'Low':'min',
    'Close':'mean',
    'Volume':'sum',    
}
ds_df = df.resample('W').apply(convertion)


# # Moving average

# > An example of two moving average curves
# In statistics, a moving average (rolling average or running average) is a calculation to analyze data points by creating series of averages of different subsets of the full data set. It is also called a moving mean (MM)[1] or rolling mean and is a type of finite impulse response filter.
# 
# ref. https://en.wikipedia.org/wiki/Moving_average

# ## Moving average

# - Moving average is simple

# In[ ]:




ds_df['rolling_mean' + str(N) + '_' + str(5)] = ds_df.Close.rolling(window=5).mean()
ds_df['rolling_mean' + str(N) + '_' + str(10)] = ds_df.Close.rolling(window=10).mean()



fig = go.Figure(go.Candlestick(x=ds_df.index,open=ds_df['Open'],high=ds_df['High'],low=ds_df['Low'],close=ds_df['Close']))
fig.update_layout(title='Bitcoin Price', yaxis_title='BTC')
fig.update_yaxes(type="log")
fig.add_trace(go.Scatter(x=ds_df.index, y=ds_df['Close'],mode='lines',name='Close'))
fig.add_trace(go.Scatter(x=ds_df.index, y=ds_df['rolling_mean' + str(N) + '_' + str(5)], mode='lines', name='MEAN_5' + str(N),line=dict(color='royalblue', width=2)))
fig.add_trace(go.Scatter(x=ds_df.index, y=ds_df['rolling_mean' + str(N) + '_' + str(10)], mode='lines', name='MEAN_10' + str(N), line=dict(color='#555555', width=2)))
fig.show()


# ## Exponential Moving Average

# > An exponential moving average (EMA), also known as an exponentially weighted moving average (EWMA),[5] is a first-order infinite impulse response filter that applies weighting factors which decrease exponentially.
# 
# ref. https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average

# In[ ]:


ewma = pd.Series.ewm


# In[ ]:


ds_df['rolling_ema_'+ str(N)]  = ds_df.Close.ewm(min_periods=N, span=N).mean()


# In[ ]:




ds_df['rolling_ema_' + str(N)] = ds_df.Close.ewm(min_periods=10, span=10).mean()



fig = go.Figure(go.Candlestick(x=ds_df.index,open=ds_df['Open'],high=ds_df['High'],low=ds_df['Low'],close=ds_df['Close']))
fig.update_layout(title='Bitcoin Price', yaxis_title='BTC')
fig.update_yaxes(type="log")
fig.add_trace(go.Scatter(x=ds_df.index, y=ds_df['Close'],mode='lines',name='Close'))
fig.add_trace(go.Scatter(x=ds_df.index, y=ds_df['rolling_ema_' + str(N)], mode='lines', name='EMA_10',line=dict(color='royalblue', width=2)))
fig.show()


# # MACD
# - MACD: (12-day EMA - 26-day EMA)

# > Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of prices. The MACD is calculated by subtracting the 26-day exponential moving average (EMA) from the 12-day EMA
# 
# ref. https://www.investopedia.com/terms/m/macd.asp

# In[ ]:


ds_df['close_5EMA'] = ewma(ds_df["Close"], span=5).mean()
ds_df['close_2EMA'] = ewma(ds_df["Close"], span=2).mean()

ds_df['MACD'] = ds_df['close_5EMA'] - ds_df['close_2EMA']

fig = go.Figure()
fig.update_layout(title='Bitcoin Price', yaxis_title='BTC')
fig.add_trace(go.Scatter(x=ds_df.index, y=ds_df['Close'],mode='lines',name='Close', line=dict(color='#555555', width=2)))
fig.add_trace(go.Scatter(x=ds_df.index, y=ds_df['MACD'], mode='lines', name='MACD_26_12',line=dict(color='royalblue', width=2)))
fig.show()


# ## Bollinger Band

# > Bollinger Bands are a type of statistical chart characterizing the prices and volatility over time of a financial instrument or commodity, using a formulaic method propounded by John Bollinger in the 1980s. Financial traders employ these charts as a methodical tool to inform trading decisions, control automated trading systems, or as a component of technical analysis. Bollinger Bands display a graphical band (the envelope maximum and minimum of moving averages, similar to Keltner or Donchian channels) and volatility (expressed by the width of the envelope) in one two-dimensional chart.
# 
# ref. https://en.wikipedia.org/wiki/Bollinger_Bands

# In[ ]:


window = 7
no_of_std = 2

ds_df[f'MA_{window}MA'] = ds_df['Close'].rolling(window=window).mean()
ds_df[f'MA_{window}MA_std'] = ds_df['Close'].rolling(window=window).std() 
ds_df[f'MA_{window}MA_BB_high'] = ds_df[f'MA_{window}MA'] + no_of_std * ds_df[f'MA_{window}MA_std']
ds_df[f'MA_{window}MA_BB_low'] = ds_df[f'MA_{window}MA'] - no_of_std * ds_df[f'MA_{window}MA_std']

fig = go.Figure()
fig.update_layout(title='Bitcoin Price', yaxis_title='BTC')
fig.add_trace(go.Scatter(x=ds_df.index, y=ds_df['Close'],mode='lines',name='Close', line=dict(color='#555555', width=2)))
fig.add_trace(go.Scatter(x=ds_df.index, y=ds_df[f'MA_{window}MA_BB_high'], mode='lines', name=f'BB_high',line=dict(color='royalblue', width=2)))
fig.add_trace(go.Scatter(x=ds_df.index, y=ds_df[f'MA_{window}MA_BB_low'], mode='lines', name=f'BB_high',line=dict(color='royalblue', width=2)))
fig.show()


# In[ ]:


window = 15
no_of_std = 2

ds_df[f'MA_{window}MA'] = ds_df['Close'].rolling(window=window).mean()
ds_df[f'MA_{window}MA_std'] = ds_df['Close'].rolling(window=window).std() 
ds_df[f'MA_{window}MA_BB_high'] = ds_df[f'MA_{window}MA'] + no_of_std * ds_df[f'MA_{window}MA_std']
ds_df[f'MA_{window}MA_BB_low'] = ds_df[f'MA_{window}MA'] - no_of_std * ds_df[f'MA_{window}MA_std']

fig = go.Figure()
fig.update_layout(title='Bitcoin Price', yaxis_title='BTC')
fig.add_trace(go.Scatter(x=ds_df.index, y=ds_df['Close'],mode='lines',name='Close', line=dict(color='#555555', width=2)))
fig.add_trace(go.Scatter(x=ds_df.index, y=ds_df[f'MA_{window}MA_BB_high'], mode='lines', name=f'BB_high',line=dict(color='royalblue', width=2)))
fig.add_trace(go.Scatter(x=ds_df.index, y=ds_df[f'MA_{window}MA_BB_low'], mode='lines', name=f'BB_high',line=dict(color='royalblue', width=2)))
fig.show()


# In[ ]:


window = 30
no_of_std = 2

ds_df[f'MA_{window}MA'] = ds_df['Close'].rolling(window=window).mean()
ds_df[f'MA_{window}MA_std'] = ds_df['Close'].rolling(window=window).std() 
ds_df[f'MA_{window}MA_BB_high'] = ds_df[f'MA_{window}MA'] + no_of_std * ds_df[f'MA_{window}MA_std']
ds_df[f'MA_{window}MA_BB_low'] = ds_df[f'MA_{window}MA'] - no_of_std * ds_df[f'MA_{window}MA_std']

fig = go.Figure()
fig.update_layout(title='Bitcoin Price', yaxis_title='BTC')
fig.add_trace(go.Scatter(x=ds_df.index, y=ds_df['Close'],mode='lines',name='Close', line=dict(color='#555555', width=2)))
fig.add_trace(go.Scatter(x=ds_df.index, y=ds_df[f'MA_{window}MA_BB_high'], mode='lines', name=f'BB_high',line=dict(color='royalblue', width=2)))
fig.add_trace(go.Scatter(x=ds_df.index, y=ds_df[f'MA_{window}MA_BB_low'], mode='lines', name=f'BB_high',line=dict(color='royalblue', width=2)))
fig.show()


# # RSI

# > The Relative Strength Index (RSI), developed by J. Welles Wilder, is a momentum oscillator that measures the speed and change of price movements. The RSI oscillates between zero and 100. Traditionally the RSI is considered overbought when above 70 and oversold when below 30. Signals can be generated by looking for divergences and failure swings. RSI can also be used to identify the general trend.
# 
# ref. https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/RSI

# In[ ]:


def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi


# In[ ]:


rsi_6 = rsiFunc(ds_df['Close'].values, 6)
rsi_14 = rsiFunc(ds_df['Close'].values, 14)
rsi_20 = rsiFunc(ds_df['Close'].values, 20)


# In[ ]:


ds_df['rsi_6'] = rsi_6
ds_df['rsi_14'] = rsi_14
ds_df['rsi_20'] = rsi_20

fig = go.Figure()
fig.update_layout(title='Bitcoin Price', yaxis_title='BTC')
fig.add_trace(go.Scatter(x=ds_df.index, y=ds_df[f'rsi_6'], mode='lines', name=f'rsi_6',line=dict(color='royalblue', width=2)))
fig.add_trace(go.Scatter(x=ds_df.index, y=ds_df[f'rsi_14'], mode='lines', name=f'rsi_14',line=dict(color='royalblue', width=2)))
fig.add_trace(go.Scatter(x=ds_df.index, y=ds_df[f'rsi_20'], mode='lines', name=f'rsi_20',line=dict(color='royalblue', width=2)))
fig.show()


# # Volume Moving Avreage

# > A Volume Moving Average is the simplest volume-based technical indicator. Similar to a price moving average, a VMA is an average volume of a security (stock), commodity, index or exchange over a selected period of time. Volume Moving Averages are used in charts and in technical analysis to smooth and describe a volume trend by filtering short term spikes and gaps.
# 
# ref. https://www.marketvolume.com/analysis/volume_ma.asp

# In[ ]:


ds_df['VMA_7MA'] = ds_df['Volume'].rolling(window=7).mean()
ds_df['VMA_15MA'] = ds_df['Volume'].rolling(window=15).mean()
ds_df['VMA_30MA'] = ds_df['Volume'].rolling(window=30).mean()
ds_df['VMA_60MA'] = ds_df['Volume'].rolling(window=60).mean()


# In[ ]:


fig = go.Figure()
fig.update_layout(title='Bitcoin Price', yaxis_title='BTC')
fig.add_trace(go.Scatter(x=ds_df.index, y=ds_df[f'VMA_7MA'], mode='lines', name=f'VMA_7MA',line=dict(color='royalblue', width=2)))
fig.add_trace(go.Scatter(x=ds_df.index, y=ds_df[f'VMA_15MA'], mode='lines', name=f'VMA_15MA',line=dict(color='royalblue', width=2)))
fig.add_trace(go.Scatter(x=ds_df.index, y=ds_df[f'VMA_30MA'], mode='lines', name=f'VMA_30MA',line=dict(color='royalblue', width=2)))
fig.add_trace(go.Scatter(x=ds_df.index, y=ds_df[f'VMA_60MA'], mode='lines', name=f'VMA_60MA',line=dict(color='royalblue', width=2)))
fig.show()


# # More to come..
