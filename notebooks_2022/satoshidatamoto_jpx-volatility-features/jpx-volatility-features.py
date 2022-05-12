#!/usr/bin/env python
# coding: utf-8

# ## Volatility Estimation Features
# >- In this notebook there are serveral functions to estimate the volatility of an asset. some can be used as features to a model.
# >- Enjoy the short scripts to obtain them! 
# >

# # <span class="title-section w3-xxlarge" id="features">Volatility Features</span>
# <hr>
# 
# In this notebook I will introduce six  methods for calculating volatility that can be used for feature engineering:
# 
# | Volatility | Price Information |
# | ----------------------- | ---------------------- |
# | Realized | Close |
# | Parkinson | High, Low |
# | Garman-Klass | Open, High, Low, Close |
# | Roger-Satchell | Open, High, Low, Close |
# | Garman-Klass-Yang-Zhang | Open, High, Low, Close |
# | Yang-Zhang | Open, High, Low, Close |
# 
# The last five methods all use a continuous rate of return, resulting in an underestimation of the real volatility.
# 
# Realized, Garman-Klass-Yang-Zhang, and Yang-Zhang all use the price information of the previous day and the current day, which can be regarded as day-to-day volatility. 
# 
# ## 1. Realized Volatility: Close-Close
# 
# $$\sigma_{realized} = \sqrt{ \frac{N}{n-2} \sum\limits_{i=1} ^{n-1} (r_t-\bar r)^2 }$$
# 
# $r_t =\log\frac{C_t}{C_{t-1}}$: rate of return
# 
# $ \bar r =\frac{1}{n} \sum\limits_{n}^{t=1}r_t$: average rate of return
# 
# ## 2. Parkinson Volatility: High-Low Volatility
# 
# $$\sigma_{parkinson} = \sqrt{ {\frac{1}{4*\ln{2}} * \frac{252}{n} * \sum\limits_{t=1}^{n} { \ln{(\frac{H_t}{L_t})}}^2}}$$
# 
# The general volatility only considers the closing price. Parkinson Volatility takes the highest price and the lowest price into consideration. 
# 
# ## 3. Garman-Klass Volatility: OHLC volatility
# 
# Assumes Brown motion with zero drift and no opening spike.
# 
# $$\sigma_{garman-klass} = \sqrt{\frac{N}{n} \sum\limits_{i=1}^{N} \lbrack {\frac{1}{2} * (\log{ \frac{H_i}{L_i}})^2 -(2*\log2 -1) * (\log\frac{C_i}{O_i})^2\rbrack}}$$
# 
# Compared with Parkinson Volatility, it further considers the opening and closing prices, and incorporates more price information, 
# 
# ## 4. Roger-Satchell Volatility: OHLC Volatility
# 
# Assumes for non-zero drift, but assumed no opening spike.
# 
# $$\sigma_{roger-satchel} = \sqrt{ \frac{N}{n} \sum\limits_{i=1}^{n} \lbrack \log \frac{H_i}{L_i} * \log \ frac{H_i}{O_i} + \log \frac{HL_i}{L_i} * \log \frac{L_i}{O_i} \rbrack }$$
# 
# ## 5. Garman-Klass-Yang-Zhang Volatility: OHLC Volatility
# 
# A modified version of Garman-Klass estimator that allows for opening spikes.
# 
# $$\sigma_{garkla-yangzh} = \sqrt {\frac{N}{n} \sum\limits_{i=1}^{n} \lbrack (\log \frac{O_i}{C_{i-1 }})^2 + {\frac{1}{2} * (\log{\frac{H_i}{L_i}})^2 -(2*\log2 -1) * (\log\frac{C_i} {O_i})^2}\rbrack }$$
# 
# In the cases where the return on assets is not zero, the volatility will be overestimated.
# 
# ## 6. Yang-Zhang Volatility: OHLC Volatility
# 
# $$\sigma_{yang-zhang} = \sqrt {\sigma_o^2 + k * \sigma_c^2 + (1-k) * \sigma_{rs}^2}$$
# 
# $\mu_o = \frac{1}{n} \sum\limits_{i=1}^{n} \log \frac {O_i}{C_{i-1}}$
# 
# $\sigma_o^2 = \frac{N}{n-1} \sum\limits_{i=1}^{n} (\log \frac {O_i}{C_{i-1}}-\mu_o)^ 2$, Open-Close Volatility or Overnight Volatility
# 
# $\mu_c = \frac{1}{n} \sum\limits_{i=1}^{n} \log \frac {C_i}{O_i}$, Close-Open Volatility
# 
# $\sigma_c^2 = \frac{N}{n-1} \sum\limits_{i=1}^{n} (\log \frac {C_i}{O_i}-\mu_c)^2$
# 
# $\sigma_{rs}^2 = \sigma_{roger-satchel}^2$
# 
# $k^* = \frac {\alpha} {1+ \alpha + \frac{n+1}{n-1}}, \alpha$ 
# 
# can also be interpreted as a weighted average of the Roger-Satchell estimator, the Close-Open Volatility and the Open-Close Volatility.
# 
# ## References
# 
# 1. [Volatility and its Measurements](https://www.eurexchange.com/blob/116048/47ca53f0178cec31caeecdf94cc18f6e/data/volatility_and_its_measurements.pdf.pdf)
# 
# 2. [Drift Independent Volatility Estimation Based on High, Low, Open and Close Price](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.628.4037&rep=rep1&type=pdf)
# 
# 3. [volatility function | R Documentation](https://www.rdocumentation.org/packages/TTR/versions/0.23-3/topics/volatility)
# 
# 4. [Parkinson volatility-Breaking Down Finance](http://breakingdownfinance.com/finance-topics/risk-management/parkinson-volatility/)
# 
# 5. [MEASURING HISTORICAL VOLATILITY](http://www.todaysgroep.nl/media/236846/measuring_historic_volatility.pdf)
# 

# In[ ]:


import os
import gc
import traceback
import numpy as np
import pandas as pd
import datatable as dt
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.core.common.SettingWithCopyWarning)
    
plt.style.use('bmh')
plt.rcParams['figure.figsize'] = [14, 8]  # width, height


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

train = load_training_data().sort_values('date') #.set_index("date")
print("Loaded all data!")

train_data = train.copy()
train_data['date'] = pd.to_datetime(train_data['Date'])
df = train_data.loc[train_data['SecuritiesCode'] == 1301]
df = df[-500000:]


# # <span class="title-section w3-xxlarge" id="features">Feature Engineering 🔬</span>
# <hr>

# In[ ]:


import numpy
import pandas as pd
from math import sqrt, log

def plot_feature(feat, df, feat_name):
    try: plt.close()
    except: pass   
    df2 = df[-len(feat):].reset_index().set_index('date')
    fig = plt.figure(figsize = (12, 6))
    # fig, ax_left = plt.subplots(figsize = (12, 6))
    ax_left = fig.add_subplot(111)
    ax_left.set_facecolor('azure')    
    ax_right = ax_left.twinx()
    ax_left.plot(feat, color = 'crimson', label = feat_name)
    ax_right.plot(df['Close'], color = 'darkgrey', label = "Price")
    plt.legend()
    plt.grid()
    plt.xlabel('Time')
    plt.title('3 month rolling %s vs close price' % (feat_name))
    plt.show()


# # <span class="title-section w3-xxlarge" id="features">Realized Volatility: Close-to-Close</span>
# <hr>
# 
# Close-to-Close volatility is a classic and most commonly used volatility measure, sometimes referred to as historical volatility.
# 
# Volatility is an indicator of the speed of a stock price change. A stock with high volatility is one where the price changes rapidly and with a bigger amplitude. The more volatile a stock is, the riskier it is.
# 
# Close-to-close historical volatility calculated using only stock's closing prices. It is the simplest volatility estimator. But in many cases, it is not precise enough. Stock prices could jump considerably during a trading session, and return to the open value at the end. That means that a big amount of price information is not taken into account by close-to-close volatility.
# 
# Despite its drawbacks, Close-to-Close volatility is still useful in cases where the instrument doesn't have intraday prices. For example, mutual funds calculate their net asset values daily or weekly, and thus their prices are not suitable for more sophisticated volatility estimators.
# 
# [source](https://portfolioslab.com/tools/close-to-close-volatility)
# 
# 
# <br>
# 
# Defined as:
# 
# $$\sigma_{realized} = \sqrt{ \frac{N}{n-2} \sum\limits_{i=1} ^{n-1} (r_t-\bar r)^2 }$$
# 
# Where the **Rate of return**:
# $$r_t =\log\frac{C_t}{C_{t-1}}$$
# 
# and the **Average rate of return**:
# $$ \bar r =\frac{1}{n} \sum\limits_{n}^{t=1}r_t$$

# In[ ]:


def realized(close, N=240):
    rt = list(log(C_t / C_t_1) for C_t, C_t_1 in zip(close[1:], close[:-1]))
    rt_mean = sum(rt) / len(rt)
    return sqrt(sum((r_i - rt_mean) ** 2 for r_i in rt) * N / (len(rt) - 1))

feat = df['Close'].rolling(60).apply(realized).bfill()
plot_feature(feat, df, 'realized volatility')


# # <span class="title-section w3-xxlarge" id="features">Parkinson Volatility</span>
# <hr>
# 
# Parkinson volatility is a volatility measure that uses the stock’s high and low price of the day.
# 
# The main difference between regular volatility and Parkinson volatility is that the latter uses high and low prices for a day, rather than only the closing price. That is useful as close to close prices could show little difference while large price movements could have happened during the day. Thus Parkinson's volatility is considered to be more precise and requires less data for calculation than the close-close volatility.
# 
# One drawback of this estimator is that it doesn't take into account price movements after market close. Hence it systematically undervalues volatility. 
# 
# [source](https://portfolioslab.com/tools/parkinson)
# 
# <br>
# 
# 
# Defined as: 
# 
# $$\sigma_{parkinson} = \sqrt{ {\frac{1}{4*\ln{2}} * \frac{252}{n} * \sum\limits_{t=1}^{n} { \ln{(\frac{H_t}{L_t})}}^2}}$$
# 

# In[ ]:


def parkinson(high, low, N=240):
    sum_hl = sum(log(H_t / L_t) ** 2 for H_t, L_t in zip(high, low))
    return sqrt(sum_hl * N / (4 * len(high) *log(2)))

feat = df.rolling(60).apply(lambda x: parkinson(df.loc[x.index, 'High'], df.loc[x.index, 'Low'])).bfill()
plot_feature(feat, df, 'parkinson volatility')


# # <span class="title-section w3-xxlarge" id="features">Garman-Klass Volatility</span>
# <hr>
# 
# Garman Klass is a volatility estimator that incorporates open, low, high, and close prices of a security.
# 
# Garman-Klass volatility extends Parkinson's volatility by taking into account the opening and closing price. As markets are most active during the opening and closing of a trading session, it makes volatility estimation more accurate.
# 
# Garman and Klass also assumed that the process of price change is a process of continuous diffusion (geometric Brownian motion). However, this assumption has several drawbacks. The method is not robust for opening jumps in price and trend movements.
# 
# Despite its drawbacks, the Garman-Klass estimator is still more effective than the basic formula since it takes into account not only the price at the beginning and end of the time interval but also intraday price extremums.
# 
# [source](https://portfolioslab.com/tools/garman-klass)
# 
# 
# <br>
# 
# Defined as:
# 
# $$\sigma_{garman-klass} = \sqrt{\frac{N}{n} \sum\limits_{i=1}^{N} \lbrack {\frac{1}{2} * (\log{ \frac{H_i}{L_i}})^2 -(2*\log2 -1) * (\log\frac{C_i}{O_i})^2\rbrack}}$$

# In[ ]:


def garman_klass(open, high, low, close, N=240):
    sum_hl = sum(log(H_t / L_t) ** 2 for H_t, L_t in zip(high, low)) / 2
    sum_co = sum(log(C_t / O_t) ** 2 for C_t, O_t in zip(close, open)) * (2 * log(2) - 1)
    return sqrt((sum_hl - sum_co) * N / len(close))

feat = df.rolling(60).apply(lambda x: garman_klass(df.loc[x.index, 'Open'], df.loc[x.index, 'High'], df.loc[x.index, 'Low'], df.loc[x.index, 'Close'])).bfill()
plot_feature(feat, df, 'garman klass')


# # <span class="title-section w3-xxlarge" id="features">Roger-Satchell Volatility</span>
# <hr>
# 
# Rogers-Satchell is an estimator for measuring the volatility of securities with an average return not equal to zero.
# 
# Unlike Parkinson and Garman-Klass estimators, Rogers-Satchell incorporates drift term (mean return not equal to zero). As a result, it provides a better volatility estimation when the underlying is trending.
# 
# The main disadvantage of this method is that it does not take into account price movements between trading sessions. It means an underestimation of volatility since price jumps periodically occur in the market precisely at the moments between sessions.
# 
# 
# [source](https://portfolioslab.com/tools/rogers-satchell)
# 
# <br>
# 
# 
# 
# $$\sigma_{roger-satchel} = \sqrt{ \frac{N}{n} \sum\limits_{i=1}^{n} \lbrack \log \frac{H_i}{L_i} * \log \ frac{H_i}{O_i} + \log \frac{HL_i}{L_i} * \log \frac{L_i}{O_i} \rbrack }$$

# In[ ]:


def roger_satchell(open, high, low, close, N=240):
    sum_ohlc = sum(log(H_t / C_t) * log(H_t / O_t) + log(L_t / C_t) * log(L_t / O_t) for O_t, H_t, L_t, C_t in zip(open, high, low, close))
    return sqrt(sum_ohlc * N / len(close))

feat = df.rolling(60).apply(lambda x: roger_satchell(df.loc[x.index, 'Open'], df.loc[x.index, 'High'], df.loc[x.index, 'Low'], df.loc[x.index, 'Close'])).bfill()
plot_feature(feat, df, 'roger satchell')


# # <span class="title-section w3-xxlarge" id="features">Yang-Zhang Volatility</span>
# <hr>
# 
# Yang Zhang is a historical volatility estimator that handles both opening jumps and the drift and has a minimum estimation error.
# 
# We can think of the Yang-Zhang volatility as the combination of the overnight (close-to-open volatility) and a weighted average of the Rogers-Satchell volatility and the day’s open-to-close volatility. It considered being 14 times more efficient than the close-to-close estimator.
# 
# [source](https://portfolioslab.com/tools/yang-zhang)
# 
# <br>
# 
# Defined as: 
# 
# $$\sigma_{yang-zhang} = \sqrt {\sigma_o^2 + k * \sigma_c^2 + (1-k) * \sigma_{rs}^2}$$
# 
# $\mu_o = \frac{1}{n} \sum\limits_{i=1}^{n} \log \frac {O_i}{C_{i-1}}$
# 
# $\sigma_o^2 = \frac{N}{n-1} \sum\limits_{i=1}^{n} (\log \frac {O_i}{C_{i-1}}-\mu_o)^ 2$, Open-Close Volatility or Overnight Volatility
# 
# $\mu_c = \frac{1}{n} \sum\limits_{i=1}^{n} \log \frac {C_i}{O_i}$, Close-Open Volatility
# 
# $\sigma_c^2 = \frac{N}{n-1} \sum\limits_{i=1}^{n} (\log \frac {C_i}{O_i}-\mu_c)^2$
# 
# $\sigma_{rs}^2 = \sigma_{roger-satchel}^2$
# 
# $k^* = \frac {\alpha} {1+ \alpha + \frac{n+1}{n-1}}, \alpha$ 
# 
# can also be interpreted as a weighted average of the Roger-Satchell estimator, the Close-Open Volatility and the Open-Close Volatility.

# In[ ]:


def yang_zhang(open, high, low, close, N=240):
    oc = list(log(O_t / C_t_1) for O_t, C_t_1 in zip(open[1:], close[:-1]))
    n = len(oc)
    oc_mean = sum(oc) / n
    oc_var = sum((oc_i - oc_mean) ** 2 for oc_i in oc) * N / (n - 1)   
    co = list(log(C_t / O_t) for O_t, C_t in zip(open[1:], close[1:]))
    co_mean = sum(co) / n
    co_var = sum((co_i - co_mean) ** 2 for co_i in co) * N / (n - 1)    
    rs_var = (roger_satchell(open[1:], high[1:], low[1:], close[1:])) ** 2    
    k = 0.34 / (1.34 + (n +1) / (n - 1))    
    return sqrt(oc_var + k * co_var + (1-k) * rs_var)

feat = df.rolling(60).apply(lambda x: yang_zhang(df.loc[x.index, 'Open'], df.loc[x.index, 'High'], df.loc[x.index, 'Low'], df.loc[x.index, 'Close'])).bfill()
plot_feature(feat, df, 'yang zhang')


# # <span class="title-section w3-xxlarge" id="features">Garman-Klass-Yang-Zhang Volatility: OHLC Volatility</span>
# <hr>
# 
# A modified version of Garman-Klass estimator that allows for opening spikes.
# 
# $$\sigma_{garkla-yangzh} = \sqrt {\frac{N}{n} \sum\limits_{i=1}^{n} \lbrack (\log \frac{O_i}{C_{i-1 }})^2 + {\frac{1}{2} * (\log{\frac{H_i}{L_i}})^2 -(2*\log2 -1) * (\log\frac{C_i} {O_i})^2}\rbrack }$$
# 
# In the cases where the return on assets is not zero, the volatility will be overestimated.

# In[ ]:


def garkla_yangzh(open, high, low, close, N=240):
    sum_oc_1 = sum(log(O_t / C_t_1) ** 2 for O_t, C_t_1 in zip(open[1:], close[:-1]))
    sum_hl = sum(log(H_t / L_t) ** 2 for H_t, L_t in zip(high[1:], low[1:])) / 2
    sum_co = sum(log(C_t / O_t) ** 2 for C_t, O_t in zip(close[1:], open[1:])) * (2 * log(2) - 1)
    return sqrt((sum_oc_1 + sum_hl - sum_co) * N / (len(close) - 1))

feat = df.rolling(60).apply(lambda x: garkla_yangzh(df.loc[x.index, 'Open'], df.loc[x.index, 'High'], df.loc[x.index, 'Low'], df.loc[x.index, 'Close'])).bfill()
plot_feature(feat, df, 'garkla yangzh')


# # More to come..
