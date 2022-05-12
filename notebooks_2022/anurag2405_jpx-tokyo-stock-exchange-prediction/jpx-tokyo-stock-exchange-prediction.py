#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # About the competition
# Japan Exchange Group, Inc. (JPX) is a holding company operating one of the largest stock exchanges in the world, Tokyo Stock Exchange (TSE), and derivatives exchanges Osaka Exchange (OSE) and Tokyo Commodity Exchange (TOCOM). JPX is hosting this competition and is supported by AI technology company AlpacaJapan Co.,Ltd.
# 
# In this competition, historic data for a variety of Japanese stocks and options are given. The competitors are challenged to rank the change ratio of adjusted close price between the next day and 2 days later for each stock. Also, note that the forecasting phase leaderboard will be determined using real market data gathered after the submission period closes.
# 
# https://www.kaggle.com/code/sohier/basic-submission-demo<br>
# https://www.kaggle.com/code/smeitoma/jpx-competition-metric-definition

# Let's Start by importing the libraries

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import lightgbm as lgb


# In[ ]:


stocks = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/stock_list.csv")
df_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")


# In[ ]:


stocks


# In[ ]:


stocks.info()


# In[ ]:


stocks.describe()


# In[ ]:


stocks.SecuritiesCode.nunique()


# In[ ]:


stocks["Section/Products"].value_counts()


# In[ ]:


stocks = stocks[stocks['NewMarketSegment'].notna()]


# In[ ]:


fig = px.pie(stocks,names="Section/Products", title='Stock Indices')
fig.show()


# In[ ]:


fig = px.pie(stocks,names="NewMarketSegment", title='Market Segment')
fig.show()


# In[ ]:


fig = px.pie(stocks,names="33SectorName", title='Sector')
fig.show()


# In[ ]:


fig = px.pie(stocks,names="Universe0",title = "top 2000 stocks by market capitalization")
fig.show()


# In[ ]:


sec_info = stocks[['SecuritiesCode', '33SectorName', '17SectorName']]
df_prices = pd.merge(df_prices, sec_info, on='SecuritiesCode')
df_prices.head()


# In[ ]:


target_mean_33sec = df_prices.groupby(['33SectorName'])['Target'].mean()
target_mean_33sec.sort_values(inplace=True, ascending=False)
fig = px.bar(x=target_mean_33sec.index, y=target_mean_33sec.values,title = "Target Mean of Each Sector")
fig.show()


# In[ ]:


target_std_33sec = df_prices.groupby(['33SectorName'])['Target'].std()
target_std_33sec.sort_values(inplace=True, ascending=False)
fig = px.bar(x=target_std_33sec.index, y=target_std_33sec.values,title = "Target Std of Each Sector")
fig.show()


# In[ ]:


target_std_per_date = df_prices.groupby(['Date'])['Target'].std()
target_std_mean = target_std_per_date.mean()

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(data=target_std_per_date.values, bins=100, kde=True,
             ax=ax)
ax.axvline(x=target_std_mean, color='orange', linestyle='dotted', linewidth=2, 
           label='Mean')
ax.set_title("Target Std Distibution\n"
             f"Min {round(target_std_per_date.min(), 4)} | "
             f"Max {round(target_std_per_date.max(), 4)} | "
             f"Skewness {round(target_std_per_date.skew(), 2)} | "
             f"Kurtosis {round(target_std_per_date.kurtosis(), 2)}")
ax.set_xlabel("Target Std")
ax.set_ylabel("Date Count")
ax.legend()
plt.show()


# In[ ]:


target_mean_per_date = df_prices.groupby(['Date'])['Target'].mean()
target_mean_mean = target_mean_per_date.mean()

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(data=target_mean_per_date.values, bins=100, kde=True,
             ax=ax)
ax.axvline(x=target_mean_mean, color='orange', linestyle='dotted', linewidth=2, 
           label='Mean')
ax.set_title("Target Mean Distibution\n"
             f"Min {round(target_mean_per_date.min(), 4)} | "
             f"Max {round(target_mean_per_date.max(), 4)} | "
             f"Skewness {round(target_mean_per_date.skew(), 2)} | "
             f"Kurtosis {round(target_mean_per_date.kurtosis(), 2)}")
ax.set_xlabel("Target Mean")
ax.set_ylabel("Date Count")
ax.legend()
plt.show()


# In[ ]:


n_stocks_per_date = df_prices.groupby(['Date'])['SecuritiesCode'].count()
n_dates_per_stock = df_prices.groupby(['SecuritiesCode'])['Date'].count()
target_mean_per_stock = df_prices.groupby(['SecuritiesCode'])['Target'].mean()
ax = sns.jointplot(x=n_dates_per_stock, y=target_mean_per_stock, 
                   kind="reg", height=6, marginal_ticks=True, 
                   joint_kws={'line_kws':{'color':'orange'}})
ax.ax_joint.set_xlabel('#Dates per Stock')
ax.ax_joint.set_ylabel('Target Mean')
plt.show()


# ## Data modelling

# Let's take a look at the train data. stock_prices.csv in train_files<br>
# The core file of interest, including the daily closing price for each stock and the target column. Following is column information recorded in stock_price_spec.csv:
# <ul>
#     <li>RowId: Unique ID of price records, the combination of Date and SecuritiesCode.</li>
# <li>Date: Trade date.</li>
# <li>SecuritiesCode: Local securities code.</li>
# <li>Open: First traded price on a day.</li>
# <li>High: Highest traded price on a day.</li>
# <li>Low: Lowest traded price on a day.</li>
# <li>Close: Last traded price on a day.</li>
# <li>Volume: Number of traded stocks on a day.</li>
# <li>AdjustmentFactor: Used to calculate theoretical price/volume when split/reverse-split happens (NOT including dividend/allotment of shares).</li>
# <li>ExpectedDividend: Expected dividend value for ex-right date. This value is recorded 2 business days before ex-dividend date.</li>
# <li>SupervisionFlag: Flag of securities under supervision and securities to be delisted, for more information, please see here.</li>
# <li>Target: Change ratio of adjusted closing price between t+2 and t+1 where t+0 is trade date.</li>
# </ul>

# In[ ]:


stock_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
stock_prices


# In[ ]:


stock_prices.shape


# In[ ]:


stock_prices.info()


# ### The date attribute is in object form let's convert it to date time format

# In[ ]:


stock_prices["Date"] = pd.to_datetime(stock_prices["Date"])


# ### Let's look at only one stock

# In[ ]:


nintendo_data = stock_prices.loc[stock_prices["SecuritiesCode"] == 7974].copy()


# In[ ]:


nintendo_data.shape


# In[ ]:


nintendo_data


# In[ ]:


sns.lineplot(data=nintendo_data,x="Date", y="Open");


# In[ ]:


fig = px.line(nintendo_data,x="Date", y="Open", title='Nintendo Opening Prices')
fig.show() 


# In[ ]:


fig = px.line(nintendo_data,x="Date",y="Close", title='Nintendo closing Prices')
fig.show()


# In[ ]:


fig = px.line(nintendo_data,x="Date",y="Volume", title='Nintendo closing Prices')
fig.show()


# In[ ]:


import plotly.graph_objects as go

import pandas as pd
from datetime import datetime

fig = go.Figure(data=[go.Candlestick(x=nintendo_data['Date'],
                open=nintendo_data['Open'],
                high=nintendo_data['High'],
                low=nintendo_data['Low'],
                close=nintendo_data['Close'])])

fig.show()


# ## Predicting using Light GBM

# In[ ]:


def upper_shadow(df): return df['High'] - np.maximum(df['Close'], df['Open'])
def lower_shadow(df): return np.minimum(df['Close'], df['Open']) - df['Low']

# A utility function to build features from the original df
# It works for rows to, so we can reutilize it.
def get_features(df):
    df_feat = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_feat['Upper_Shadow'] = upper_shadow(df_feat)
    df_feat['Lower_Shadow'] = lower_shadow(df_feat)
    return df_feat


# In[ ]:


def get_Xy_and_model(df_train):

    df_proc = get_features(df_train)
    df_proc['y'] = df_train['Target']
    df_proc = df_proc.dropna(how = "any")
    
    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]
    
    model = lgb.LGBMRegressor()
    model.fit(X, y)
    return X, y, model


# In[ ]:


print(f"Training model")
X, y, model = get_Xy_and_model(stock_prices)
Xs, ys, models = X, y, model


# In[ ]:


x = get_features(stock_prices.iloc[1])
y_pred = models.predict(pd.DataFrame([x]))
y_pred[0]


# ## Predicting using Metrics given in Competition Metric

# Consider a temporary df

# In[ ]:


tmpdf = stock_prices[stock_prices["SecuritiesCode"]==1301].reset_index(drop=True)
tmpdf


# ## Calculating Target

# The model will use the closing price ( C(k,t) ) until that business day ( t ) and other data every business day as input data for a stock ( k ), and predict rate of change ( r(k,t) ) of closing price of the top 200 stocks and bottom 200 stocks on the following business day ( C(k,t+1) ) to next following business day ( C(k,t+2) )
# 
# ![](https://i.imgur.com/H3RJLUr.png)

# In[ ]:


tmpdf["Close_shift1"] = tmpdf["Close"].shift(-1)
tmpdf["Close_shift2"] = tmpdf["Close"].shift(-2)
tmpdf["rate"] = (tmpdf["Close_shift2"] - tmpdf["Close_shift1"]) / tmpdf["Close_shift1"]
tmpdf


# ### we can confirm that the target given is equal to the calculated rate

# ## Rank calculation.
# First, let's take a look at only one day.

# In[ ]:


tmpdf2 = stock_prices[stock_prices["Date"]=="2021-12-02"].reset_index(drop=True)
tmpdf2


# In[ ]:


tmpdf2["rank"] = tmpdf2["Target"].rank(ascending=False,method="first") -1 
tmpdf2 = tmpdf2.sort_values("rank").reset_index(drop=True)


# In[ ]:


tmpdf2


# In terms of meaning, the smaller the rank, the larger the rate of change to +, so it is profitable to buy it. The larger the rank value, the greater the rate of change, so it is profitable to sell it.

# ## Calculation of daily spread return of this day

# If you look at, for the top 200 and bottom 200 of the rank, multiply the rate of change target of the closing price you just understood by weight (1 ~ 2).
# ![](https://i.imgur.com/Oqw26Q7.png)

# Consider only a rank of 200 from the top. (200 larger Targets)

# In[ ]:


tmpdf2_top200 = tmpdf2.iloc[:200,:]
tmpdf2_top200


# In[ ]:


weights = np.linspace(start=2, stop=1, num=200)
weights


# In[ ]:


tmpdf2_top200["weights"] = weights
tmpdf2_top200.head(5)


# Multiply this by target

# In[ ]:


tmpdf2_top200["calc_weights"] = tmpdf2_top200["Target"] * tmpdf2_top200["weights"]
tmpdf2_top200.head(5)


# The sum of this is divided by weight, Sup
# ![Imgur](https://i.imgur.com/r9PWJKb.png?1)

# In[ ]:


Sup = tmpdf2_top200["calc_weights"].sum()/np.mean(weights)
Sup


# ### Similarly, calculate Sdown by calculating 200 bottoms.

# In[ ]:


tmpdf2_bottom200 = tmpdf2.iloc[-200:,:]
tmpdf2_bottom200 = tmpdf2_bottom200.sort_values("rank",ascending = False).reset_index(drop=True)
tmpdf2_bottom200


# In[ ]:


tmpdf2_bottom200["weights"] = weights
tmpdf2_bottom200.head(3)


# In[ ]:


tmpdf2_bottom200["calc_weights"] = tmpdf2_bottom200["Target"] * tmpdf2_bottom200["weights"]
tmpdf2_bottom200.head(3)


# In[ ]:


Sdown = tmpdf2_bottom200["calc_weights"].sum()/np.mean(weights)
Sdown


# ![Imgur](https://i.imgur.com/kNyRPR2.png?1)

# In[ ]:


daily_spread_return = Sup - Sdown
daily_spread_return


# Function provided in Competition Metrics

# In[ ]:


import numpy as np
import pandas as pd


def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    """
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        """
        Args:
            df (pd.DataFrame): predicted results
            portfolio_size (int): # of equities to buy/sell
            toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
        Returns:
            (float): spread return
        """
        assert df['Rank'].min() == 0
        assert df['Rank'].max() == len(df['Rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio


# In[ ]:


stock_prices2 = stock_prices.loc[stock_prices["Date"]>= "2021-01-01"].reset_index(drop=True)
stock_prices2


# In[ ]:


stock_prices2["Rank"] = stock_prices2.groupby("Date")["Target"].rank(ascending=False,method="first") -1 
stock_prices2["Rank"] =stock_prices2["Rank"].astype("int")


# In[ ]:


stock_prices2


# In[ ]:


stock_prices2["Rank"].min()


# In[ ]:


score = calc_spread_return_sharpe(stock_prices2, portfolio_size= 200, toprank_weight_ratio= 2)
score


# # Predicting Ranks

# In[ ]:


import jpx_tokyo_market_prediction
env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()


# In[ ]:


for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    
    break


# In[ ]:


supplemental_stock_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv")
supplemental_stock_prices


# In[ ]:


supplemental_stock_prices["Rank"] = supplemental_stock_prices.groupby("Date")["Target"].rank(ascending=False,method="first") -1
supplemental_stock_prices.head(3)


# In[ ]:


supplemental_stock_prices[supplemental_stock_prices["Date"]=="2022-02-28"].nunique()


# In[ ]:


finday = supplemental_stock_prices[supplemental_stock_prices["Date"]=="2022-02-28"].reset_index(drop=True)
finday


# In[ ]:


finday[finday["Rank"]==finday["Rank"].iloc[0]]


# In[ ]:


finday["Rank"] = finday["Rank"].astype("int")


# In[ ]:


findaydict = dict(zip(finday["SecuritiesCode"],finday["Rank"]))


# In[ ]:


sample_prediction.head(3)


# In[ ]:


env.predict(sample_prediction)


# In[ ]:


sample_prediction["Rank"]  = sample_prediction["SecuritiesCode"].map(findaydict)
sample_prediction


# In[ ]:


for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    
    sample_prediction['Rank'] = sample_prediction["SecuritiesCode"].map(findaydict)
    env.predict(sample_prediction)

