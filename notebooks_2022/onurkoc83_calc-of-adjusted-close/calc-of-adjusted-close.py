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


# **Notes:**
# * Calculation of Adjusted Close Price(AdjustmentFactor_diff = 730)
# * useful links: https://www.tradewithscience.com/stock-split-adjusting-with-python/
# * https://finance.zacks.com/adjusted-closing-price-vs-closing-price-9991.html

# * When individual stocks become very expensive, companies can split the stocks into smaller units. These splits, like new offerings, reduce the overall value of each share because the number of total shares increases. While the initial overall value of each individual stock decreases with a stock split, the overall value of the company can actually increase because new investors snatch up the newly reduced stocks and drive the price up.
# 
# * For example, a company with 10,000 shares of 200 stocks might split the shares in half so there are 20,000 shares. Each of the 20,000 shares is then worth 50. When investors buy up the new 50 stocks because of their perceived good value, the value of each of the 50 stocks might then rise. Adjusted closing price accounts for stock splits, both because of a decrease in value caused by the split itself, and also the subsequent possible increase in value due to the new demand.

# 

# In[ ]:


df  = pd.read_csv('/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv', parse_dates=True)


# In[ ]:


df.head()


# In[ ]:


print(df["AdjustmentFactor"].value_counts())


# In[ ]:


AdjustmentFactor_eq_1 = df[df["AdjustmentFactor"]==1]["AdjustmentFactor"].value_counts().item()


# In[ ]:


print("Number of the AdjustmentFactor other than 1  = ",df.shape[0] - AdjustmentFactor_eq_1)


# In[ ]:


def adjust_price(price):
    
    def calculate_adjusted(df, dividends=False):
        # we will go from today to the past
        new = df.sort_index(ascending=False)

        split_coef = new['AdjustmentFactor'].shift(1
            ).fillna(1).cumprod()

        for col in ['Open', 'High', 'Low', 'Close']:
            new['adj_' + col] = new[col] / split_coef
        new['adj_Volume'] = split_coef * new['Volume']

        if dividends:
            new['adj_dividends'] = new['dividend amount'] / split_coef

        return new.sort_index(ascending=True)
    
    price = price.groupby("SecuritiesCode").apply(calculate_adjusted).reset_index(drop=True)
    price.set_index("Date", inplace=True)
    return price


# In[ ]:


df_adj = adjust_price(df)


# In[ ]:


df_adj

