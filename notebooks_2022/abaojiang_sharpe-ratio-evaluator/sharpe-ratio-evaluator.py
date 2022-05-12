#!/usr/bin/env python
# coding: utf-8

# # JPX - Sharpe Ratio Evaluator
# 
# ## About this Notebook
# In this kernel, a simple `Evaluator` is defined, which can be used for further evaluation and prediction analysis; furthermore, a baseline is implemented to demonstrate how to use `Evaluator`.
# 
# <a id='toc'></a>
# ## Table of Contents
# * [1. Definition of `Evaluator`](#def-eval)
# * [2. Baseline - Intraday Return as Proxy](#baseline)
# * [3. Demonstration of `Evaluator`](#demo-eval)
# * [4. Submission](#submission)
# 
# ## Import Packages

# In[ ]:


import os
import gc
import warnings

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.graph_objects as go

# Configuration
warnings.simplefilter('ignore')
pd.set_option('max_column', None)
sns.set_style("darkgrid")
colors = sns.color_palette('Set2')


# In[ ]:


TRAIN_DIR = "../input/jpx-tokyo-stock-exchange-prediction/train_files/"
PRICE_COLS = ["Open", "High", "Low", "Close"]
PK = ["Date", "SecuritiesCode"]


# ## Load Data

# In[ ]:


df = pd.read_csv(os.path.join(TRAIN_DIR, 'stock_prices.csv'))


# <a id="def-eval"></a>
# ## 1. Definition of `Evaluator`
# [**<span style="color:#FEF1FE; background-color:#535d70;border-radius: 5px; padding: 2px">Go to Table of Content</span>**](#toc)
# 
# Following is the definition of the `Evaluator`.

# In[ ]:


class Evaluator(object):
    """JPX sharpe ratio evaluator.

    The evaluator is used to evaluate performance of the model on a
    single dataset (e.g., training set, validation set). Moreover, to
    facilitate the prediction analysis, rolling sharpe ratio can be
    derived. For running evaluation (e.g., evaluation on one epoch), 
    rolling sharpe ratio derivation can be disabled.

    Parameters:
        derive_rolling_sr: whether to derive rolling sharpe ratio
    """

    WTS: np.ndarray = np.linspace(2, stop=1, num=200)
    WTS_MEAN: float = np.mean(WTS)
    ddsr: pd.DataFrame
    rolling_sr: pd.DataFrame

    def __init__(self, derive_rolling_sr: bool = False):
        self.derive_rolling_sr = derive_rolling_sr
        self.ddsr = None
        self.rolling_sr = None

    def evaluate(self, pred: pd.DataFrame, window: int = 60) -> float:
        """Run evalution.

        Parameters:
            pred: prediction results on different datasets following
                  the format of `sample_submission.csv`
            window: size of sliding window to derive rolling sr

        Return:
            sr: sharpe ratio
        """
        pred = pred.sort_values(["Date"])
        self.ddsr = pred.groupby("Date").apply(self._derive_daily_spread_return)
        sr = self._derive_overall_sr()
        if self.derive_rolling_sr:
            self.rolling_sr = self._derive_rolling_sr(window)

        return sr

    def get_daily_spread_return(self) -> pd.DataFrame:
        """Return daily spread return for current fold.

        Return:
            self.ddsr: daily spread return
        """
        return self.ddsr

    def get_rolling_sr(self) -> pd.DataFrame:
        """Return rolling sharpe ratio for current fold.

        Return:
            self.rolling_sr: rolling sharpe ratio
        """
        return self.rolling_sr

    def _derive_daily_spread_return(self, pred_date: pd.DataFrame) -> float:
        """Derive daily spread return for one date.

        Parameters:
            pred_date: prediction for one date

        Return:
            daily_spread_return: daily spread return
        """
        pred_date.sort_values("Rank", inplace=True)
        s_up = np.dot(pred_date[:200]["Target"], Evaluator.WTS) / Evaluator.WTS_MEAN
        s_down = (
            np.dot(pred_date[-200:]["Target"][::-1], Evaluator.WTS) / Evaluator.WTS_MEAN
        )
        daily_spread_return = s_up - s_down

        return daily_spread_return

    def _derive_rolling_sr(self, window: int = 60) -> pd.DataFrame:
        """Derive rolling sharpe ratio.

        Parameters:
            window: size of sliding window to derive rolling sr

        Return:
            rolling_sr: rolling sharpe ratio
        """
        rolling_sr = self.ddsr.rolling(window).mean() / self.ddsr.rolling(window).std()

        return rolling_sr

    def _derive_overall_sr(self) -> float:
        """Derive overall sharpe ratio for different datasets.

        Return:
            sr: overall sharpe ratio
        """
        sr = self.ddsr.mean() / self.ddsr.std()

        return sr


# <a id="baseline"></a>
# ## 2. Baseline - Intraday Return as Proxy
# [**<span style="color:#FEF1FE; background-color:#535d70;border-radius: 5px; padding: 2px">Go to Table of Content</span>**](#toc)
# 
# Following is a simple baseline ranking the stocks based on **intraday return**, which is defined as: $$Intraday\ Return = \frac{C_{t}^{adj} - O_{t}^{adj}}{O_{t}^{adj}}$$

# In[ ]:


def adjust_pnv(df_stock: pd.DataFrame) -> pd.DataFrame:
    """Adjust prices and volume of a single stock.
    
    Parameters:
        df_stock: raw data of a single stock
    
    Return:
        df_stock_: processed data with adjusted prices and volume
    """
    df_stock.sort_values(by=["Date"], inplace=True)
    adj_factor_cum_prod = df_stock["AdjustmentFactor"][::-1].cumprod()

    for price in PRICE_COLS:
        df_stock[f"Adj{price}"] = df_stock[price] * adj_factor_cum_prod
        df_stock[f"Adj{price}"] = df_stock[f"Adj{price}"].ffill()   # Tmp. workaround
    df_stock[f"AdjVol"] = df_stock["Volume"] / adj_factor_cum_prod

    return df_stock


# First, let's adjust price and volume information. 

# In[ ]:


df = df.groupby("SecuritiesCode").apply(adjust_pnv)


# Then, we can use adjusted prices to derive **intraday return**.

# In[ ]:


df["IntradayReturn"] = (df["AdjClose"] - df["AdjOpen"]) / df["AdjOpen"]

# Forward fill missing values (temporary workaround)
df["IntradayReturn"].fillna(0, inplace=True)
df["Target"].fillna(0, inplace=True)


# Finally, stock ranking can be done based on **intraday return**.

# In[ ]:


# Rank stocks
rank = df[PK + ["IntradayReturn"]].set_index("SecuritiesCode")
rank = (rank.groupby("Date")
            .apply(lambda x: x["IntradayReturn"].rank(method='first').astype(int) - 1))
rank.name = "Rank"

# Merge prediction with target
pred = rank.reset_index()
target = df[PK + ['Target']]
pred = pred.merge(target, left_on=PK, right_on=PK)
pred.head(3)


# <a id="demo-eval"></a>
# ## 3. Demonstration of `Evaluator`
# [**<span style="color:#FEF1FE; background-color:#535d70;border-radius: 5px; padding: 2px">Go to Table of Content</span>**](#toc)
# 
# After obtaining the predicting results (*i.e.*, stock ranking), the evaluation can be done by `Evaluator`. Remember to turn on `derive_rolling_sr` to facilitate further observation (I choose 30 and 60 to do the demonstration).

# In[ ]:


evaluator = Evaluator(derive_rolling_sr=True)
w1, w2 = 30, 60   # Window size to derive rolling sharpe ratio

sr = evaluator.evaluate(pred, window=w1)
print(f"Sharpe ratio of baseline during training period: {sr}")


# In[ ]:


daily_spread_return = evaluator.get_daily_spread_return()

fig = go.Figure()
fig.add_trace(go.Scatter(x=daily_spread_return.index, y=daily_spread_return.values, 
                         mode='lines+markers'))
fig.update_layout(title="Daily Spread Return during Training Period")
fig.show()


# In[ ]:


rolling_sr_30 = evaluator.get_rolling_sr()

# Second evaluation using window=60
_ = evaluator.evaluate(pred, window=w2)
rolling_sr_60 = evaluator.get_rolling_sr()

fig = go.Figure()
fig.add_trace(go.Scatter(x=rolling_sr_30.index, y=rolling_sr_30.values, 
                         mode='lines+markers', name=f"Window = {w1}"))
fig.add_trace(go.Scatter(x=rolling_sr_60.index, y=rolling_sr_60.values, 
                         mode='lines+markers', name=f"Window = {w2}"))
fig.update_layout(title="Rolling Sharpe Ratio during Training Period")
fig.show()


# <a id="submission"></a>
# ## 4. Submission
# [**<span style="color:#FEF1FE; background-color:#535d70;border-radius: 5px; padding: 2px">Go to Table of Content</span>**](#toc)

# In[ ]:


import jpx_tokyo_market_prediction

env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

for (prices, _, _, _, _, sample_prediction) in iter_test:
    prices["IntradayReturn"] = (prices["Close"] - prices["Open"]) / prices["Open"]
    prices["IntradayReturn"].fillna(0, inplace=True)
    df = prices[["SecuritiesCode", "IntradayReturn"]].set_index("SecuritiesCode")
    df["Rank"] = df["IntradayReturn"].rank(method='first').astype(int) - 1
    rank = df["Rank"]
    
    sample_prediction['Rank'] = sample_prediction["SecuritiesCode"].map(rank)
    env.predict(sample_prediction)

