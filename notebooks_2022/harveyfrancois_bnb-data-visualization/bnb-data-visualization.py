#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# **CREATE DATA FRAME**

# In[ ]:


# Creating a data frame with panda
df = pd.read_csv('/kaggle/input/binancecoin-data/BNB-USD.csv')


# **CLEAN UP**

# In[ ]:


# Converting date column to date time with panda.
df['Date'] = pd.to_datetime(df['Date'])

# Creating new data frmame with only data that we will be using.
new_df = df.groupby('Date')[['Date','Open','Close']].sum()
new_df.head()


# **PLOTING PRICE DATA**

# In[ ]:


# Creating grap with matplotlib.
plt.figure(figsize=(25,10))

# Selecting data column to plot.
bnb_close = new_df['Close']
bnb_open = new_df['Open']

# Ploting the data.
plt.plot(bnb_close, label='Closing', color='red')
plt.plot(bnb_open, label='Open', color='green')
plt.legend()
sns.set()


# **PREDICTING FUTURE MARKET PRICES**

# In[ ]:


# Creating new data frame with only needed columns. 
prediction_df = df[['Date', 'Close']]
prediction_df.head()


# In[ ]:


# Change names of Date and Close to ds and y.
# Required varibales for forcast
prediction_df.columns = prediction_df.columns.str.replace('Date', 'ds')
prediction_df.columns = prediction_df.columns.str.replace('Close', 'y')
prediction_df.head()


# In[ ]:



# Prediction days.
pre = Prophet()
pre.fit(prediction_df)
future = pre.make_future_dataframe(periods=360)
future.tail(5)


# In[ ]:


#Create a dataset to forcast market prices. 
forecast = pre.predict(future)
data_forecast = forecast[["ds","yhat_lower","yhat_upper"]]
pd.set_option('display.max_rows',data_forecast.shape[0]+1)
data_forecast.tail()


# **RESULTS**

# In[ ]:


# Ploting out Binance BNB market predictions.
dis_pre = pre.plot(forecast)

