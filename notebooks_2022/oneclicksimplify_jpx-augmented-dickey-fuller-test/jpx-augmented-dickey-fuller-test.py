#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Augmented Dickey-Fuller test
# 
# The idea and example is taken from here:
# 
# https://machinelearningmastery.com/time-series-data-stationary-python/

# Here, we are trying to determine whether the time series(that we are going to use) in this competition is stationary.
# 
# The Augmented Dickey-Fuller uses an autoregressive model and optimizes an information criterion across multiple different lag values.
# 
# The null hypothesis of the test is that the time series can be represented by a unit root, that it is not stationary (has some time-dependent structure). 
# 
# The alternate hypothesis (rejecting the null hypothesis) is that the time series is stationary.
# 
# Null Hypothesis (H0): If failed to be rejected, it suggests the time series has a unit root, meaning it is non-stationary. It has some time dependent structure.
# 
# Alternate Hypothesis (H1): The null hypothesis is rejected; it suggests the time series does not have a unit root, meaning it is stationary. It does not have time-dependent structure.
# 
# We interpret this result using the p-value from the test. 
# 
# A p-value below a threshold (such as 5% or 1%) suggests we reject the null hypothesis (stationary), otherwise a p-value above the threshold suggests we fail to reject the null hypothesis (non-stationary).
# 
# p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.
# 
# p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
window_size=20


# In[ ]:


import jpx_tokyo_market_prediction
from lightgbm import LGBMRegressor

from decimal import ROUND_HALF_UP, Decimal


# In[ ]:


def concat_df(df1, df2):
    df1 = pd.concat([df1, df2],
                    ignore_index=True, sort=False
                    ).drop_duplicates(["RowId"], keep="first")
    return df1


# In[ ]:


path = "../input/jpx-tokyo-stock-exchange-prediction/"
prices = pd.read_csv(f"{path}train_files/stock_prices.csv")


# In[ ]:


df_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv")


# In[ ]:


prices = concat_df(prices, df_prices)


# In[ ]:


prices = prices[prices.Date>"2021-10-02"]
prices.info(show_counts=True)


# In[ ]:


prices.isnull().sum()


# In[ ]:


def adjust_price(price):
    """
    We will generate AdjustedClose using AdjustmentFactor value. 
    This should reduce historical price gap caused by split/reverse-split.
    
    Args:
        price (pd.DataFrame)  : pd.DataFrame include stock_price
    Returns:
        price DataFrame (pd.DataFrame): stock_price with generated AdjustedClose
    """
    # transform Date column into datetime
    price.loc[: ,"Date"] = pd.to_datetime(price.loc[: ,"Date"], format="%Y-%m-%d")

    def generate_adjusted_close(df):
        """
        Args:
            df (pd.DataFrame)  : stock_price for a single SecuritiesCode
        Returns:
            df (pd.DataFrame): stock_price with AdjustedClose for a single SecuritiesCode
        """
        # sort data to generate CumulativeAdjustmentFactor
        df = df.sort_values("Date", ascending=False)
        # generate CumulativeAdjustmentFactor
        df.loc[:, "CumulativeAdjustmentFactor"] = df["AdjustmentFactor"].cumprod()
        # generate AdjustedClose
        df.loc[:, "AdjustedClose"] = (
            df["CumulativeAdjustmentFactor"] * df["Close"]
        ).map(lambda x: float(
            Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        ))
        # reverse order
        df = df.sort_values("Date")
        # to fill AdjustedClose, replace 0 into np.nan
        df.loc[df["AdjustedClose"] == 0, "AdjustedClose"] = np.nan
        # forward fill AdjustedClose
        df.loc[:, "AdjustedClose"] = df.loc[:, "AdjustedClose"].ffill()
        return df

    # generate AdjustedClose
    price = price.sort_values(["SecuritiesCode", "Date"])
    price = price.groupby("SecuritiesCode").apply(generate_adjusted_close).reset_index(drop=True)

    # price.set_index("Date", inplace=True)
    return price


# In[ ]:


# adjust close
df = adjust_price(prices)


# In[ ]:


df.isnull().sum()


# In[ ]:


df=df.dropna()


# In[ ]:


df.head(1)


# In[ ]:


df.isnull().sum()


# In[ ]:


del prices


# Compute daily returns and 20 day moving historical volatility

# In[ ]:


df['returns']=df['Close'].pct_change()
df['volatility']=df['returns'].rolling(window_size).std()*(252**0.5)


# In[ ]:


df.isnull().sum()


# In[ ]:


df=df.dropna()


# In[ ]:


X = df['volatility'].values


# # Stability Check

# In[ ]:


from statsmodels.tsa.stattools import adfuller
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

