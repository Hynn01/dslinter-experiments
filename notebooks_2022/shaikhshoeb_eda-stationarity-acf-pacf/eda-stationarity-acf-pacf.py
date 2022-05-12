#!/usr/bin/env python
# coding: utf-8

# **Analyzing and Predicting Bitcoin pricing**

# **Importing useful libraries**

# In[ ]:


import numpy as np # use to handle intigers
import pandas as pd # useful for importing dataframe 
import seaborn as sns # Visualization 
import matplotlib.pyplot as plt 
from statsmodels.tsa.stattools import adfuller # used to check stationarity
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # ACF and PACF Plots
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
from statsmodels.tsa.stattools import acovf,acf,pacf,pacf_yw,pacf_ols

import os
for dirname, _, filenames in os.walk('../input/analyzing-and-prediction-of-bitcoin-pricing'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# importing Data Frame 


# In[ ]:


df = pd.read_csv('/kaggle/input/analyzing-and-prediction-of-bitcoin-pricing/BTC-USD.csv', index_col='Date', parse_dates=True)


# In[ ]:


print(df.head())
print(df.tail())


# In[ ]:


df.info()


# In[ ]:


df['Open'].plot(figsize=(16,5)) #line plot of Open price 


# In[ ]:


df['Close'].plot(figsize=(16,5), color = 'g') # line plot of Closed price


# In[ ]:


df['Volume'].plot(figsize=(16,5)) # line plot of volume 


# In[ ]:


df.plot.hist(y='Volume', figsize=(10,6),bins=200,edgecolor='g') # histogram 


# In[ ]:


#installing MPLfinance library for candle plots


# In[ ]:


get_ipython().system('pip install mplfinance ')


# In[ ]:


import mplfinance as mpf


# In[ ]:


# Candle plot of closed price from 2014 to 2015


# In[ ]:


mpf.plot(df['2014-09-17' : '2015-01-01'],
         type = 'candle',style='yahoo',
         title=' Bitcoin pricing from 2014 to 2015')


# In[ ]:


# Candle plot of closed price from 2015 to 2016


# In[ ]:


mpf.plot(df['2015-01-01' : '2016-01-01'],
         type = 'candle',style='yahoo',
         title=' Bitcoin pricing from 2015 to 2016')


# In[ ]:


# Candle plot of closed price from 2016 to 2017


# In[ ]:


mpf.plot(df['2016-01-01' : '2017-01-01'],
         type = 'candle',style='yahoo',
         title=' Bitcoin pricing from 2016 to 2017')


# In[ ]:


# Candle plot of closed price from 2017 to 2018


# In[ ]:


mpf.plot(df['2017-01-01' : '2018-01-01'],
         type = 'candle',style='yahoo',
         title=' Bitcoin pricing from 2017 to 2018')


# In[ ]:


# Candle plot of closed price from 2018 to 2019


# In[ ]:


mpf.plot(df['2018-01-01' : '2019-01-01'],
         type = 'candle',style='yahoo',
         title=' Bitcoin pricing from 2018 to 2019')


# In[ ]:


# Candle plot of closed price from 2019 to 2020


# In[ ]:


mpf.plot(df['2019-01-01' : '2020-01-01'],
         type = 'candle',style='yahoo',
         title=' Bitcoin pricing from 2019 to 2020')


# In[ ]:


# Candle plot of closed price from 2020 to 2021


# In[ ]:


mpf.plot(df['2020-01-01' : '2021-01-01'],
         type = 'candle',style='yahoo',
         title=' Bitcoin pricing from 2020 to 2021')


# In[ ]:


# Candle plot of closed price from 2021 to 2022


# In[ ]:


mpf.plot(df['2021-01-01' : '2022-04-02 '],
         type = 'candle',style='yahoo',
         title=' Bitcoin pricing from 2021 to 2022')

#corrilation between different features 
# In[ ]:


plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()


# In[ ]:


#Adfuller test to check stationarity


# In[ ]:


A = df["Close"].values
result = adfuller(A)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

if result[0] < result[4]["5%"]:
    print ("Reject Ho - Time Series is Stationary")
else:
    print ("Failed to Reject Ho - Time Series is Non-Stationary")


# In[ ]:


# ACF and PACF plots


# In[ ]:


# Now let's plot the autocorrelation at different lags
title = 'Autocorrelation: Close price of bitcoin'
lags = 40
plot_acf(df['Close'],title=title,lags=lags);


# In[ ]:


title='Partial Autocorrelation: Close price of bitcoin'
lags=40
plot_pacf(df['Close'],title=title,lags=lags);


# In[ ]:


# the plots shows that we can use ARIMA order 1 


# **Thank You**
