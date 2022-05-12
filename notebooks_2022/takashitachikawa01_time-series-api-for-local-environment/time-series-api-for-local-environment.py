#!/usr/bin/env python
# coding: utf-8

# * This code emulates the time-series-API behavior of JPX Tokyo Stock Exchange Prediction in your local environment. 
# * このコードは、JPX Tokyo Stock Exchange Predictionのtime-series-APIをあなたのローカル環境で再現することができます。
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


#Load supplemental files which is loaded in time-series API
trades_sup = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/trades.csv")
secondary_stock_prices_sup = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/secondary_stock_prices.csv")
financials_sup = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/financials.csv")
options_sup = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/options.csv")
stock_prices_sup = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv")
stock_prices_sup["Date"] = pd.to_datetime(stock_prices_sup["Date"])


# In[ ]:


#timeseriesAPI emulator
#you can test your code for time-series-API in the following code
for date in stock_prices_sup["Date"].unique():
    prices = stock_prices_sup[stock_prices_sup["Date"] == date]
    options = options_sup[options_sup["Date"] == date]
    financials = financials_sup[financials_sup["Date"] == date]
    secondary_prices = secondary_stock_prices_sup[secondary_stock_prices_sup["Date"] == date]
    sample_submission = prices[["Date","SecuritiesCode"]].sort_values(by = "SecuritiesCode",ascending = True)
    sample_submission["Rank"] = np.arange(len(sample_submission.index)) 

