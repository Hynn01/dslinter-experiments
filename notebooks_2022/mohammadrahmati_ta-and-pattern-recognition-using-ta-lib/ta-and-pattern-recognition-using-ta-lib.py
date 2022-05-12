#!/usr/bin/env python
# coding: utf-8

# # Technical Analysis and Pattern Recognition:
# 
# TA-Lib is widely used by trading software developers requiring to perform technical analysis of financial market data. It includes 200 indicators such as ADX, MACD, RSI, Stochastic, Bollinger Bands, and has Candlestick pattern recognition. Since Talib does not come with a standard python package, the installation is a bit difficult. In this notebook, I am using 'how to install ta-lib in Google Drive' method to install and import the talib library. Finally, with usage of Abstract API, I will generate new features for all of the candlestick patterns available in talib.
# 
# Important Links:
# 
# * TA-Lib Python Wrapper documentation: <a href="https://github.com/mrjbq7/ta-lib" target="_blank">here</a>
# * TA-Lib Website: <a href="https://www.ta-lib.org/" target="_blank">here</a>
# * How to install talib in python: <a href="https://blog.quantinsti.com/install-ta-lib-python/" target="_blank">here</a>
# * How to install ta-lib in Google Drive: <a href="https://stackoverflow.com/questions/49648391/how-to-install-ta-lib-in-google-colab?newreg=5d6b616eb23c4a2f9f6e78a8c948d56d" target="_blank">here</a>
# * All Tech Indicators: <a href="https://www.programcreek.com/python/index/7769/talib" target="_blank">here</a>
# 
# * Using abstract API: <a href="https://mrjbq7.github.io/ta-lib/abstract.html" target="_blank">here</a>
# 

# In[ ]:


url = 'https://launchpad.net/~mario-mariomedina/+archive/ubuntu/talib/+files'
ext = '0.4.0-oneiric1_amd64.deb -qO'
get_ipython().system('wget $url/libta-lib0_$ext libta.deb')
get_ipython().system('wget $url/ta-lib0-dev_$ext ta.deb')
get_ipython().system('dpkg -i libta.deb ta.deb')
get_ipython().system('pip install ta-lib')


# In[ ]:


# loading libraries
import pandas as pd
import talib
from talib import abstract


# In[ ]:


# loading dataset and converting the price column names to lowercase so abstract API can recognize the input
stock_prices_df = pd.read_csv('../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv')
df = stock_prices_df[stock_prices_df['SecuritiesCode'] == 1301].copy()
df.rename(columns={'Open': 'open', 'High': 'high','Low': 'low','Close': 'close', 'Volume': 'volume'}, inplace= True)
df.head()


# In[ ]:


df.shape


# If you're already familiar with using the function API, you should feel right at home using the abstract API. Every function takes the same input. In this notebook we focus on the Pattern Recognition. Candlestick patterns can be a great way of deciding if and when we should open or close a trade. For example, if we get a ‘hammer’ candlestick, this is usually a bullish sign. Paired with a moving average, this can be a powerful strategy. 

# In[ ]:


# talib library has many categories of indicators, feel free to add more indicator groups to your features.
talib.get_function_groups().keys()


# In[ ]:


# list of all indicators
talib.get_function_groups()


# In[ ]:


# all Pattern Recognition indicators start with CDL
all_indicators = [method for method in dir(abstract) if method.startswith('CDL')]


# In[ ]:


all_indicators


# In[ ]:


df_indicator = pd.DataFrame()
for indicator in all_indicators:
    df[str(indicator)] = getattr(abstract, indicator)(df)


# In[ ]:


df

