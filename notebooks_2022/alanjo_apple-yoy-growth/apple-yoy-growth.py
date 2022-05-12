#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
plt.style.use('seaborn')
sns.set(font_scale=2)
import warnings; warnings.filterwarnings('ignore')

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # AAPL analysis

# In[ ]:


name='Apple'
data=pd.read_csv('../input/apple-stock-price-all-time/AAPL.csv')
data['Date'] = pd.to_datetime(data['Date'])
ds_df=data[-600:].reset_index(drop=True)
ds_df2=data
display(ds_df)


# # Candlestick chart

# In[ ]:


#fig=make_subplots(specs=[[{"secondary_y":True}]])
fig = go.Figure(go.Candlestick(x=ds_df['Date'],open=ds_df['Open'],high=ds_df['High'],low=ds_df['Low'],close=ds_df['Close']))
#fig.add_trace(go.Scatter(x=ds_df['Date'],y=ds_df['Close'],mode='lines',name='Close'))
fig.update_layout(title='Candlestick', yaxis_title='USD',width=800,height=500,)
fig.update_yaxes()
fig.show()


# # Year Over Year: Percentage Growth

# In[ ]:


ds_df2['Close MA200'] = ds_df2['Close'].rolling(window=20).mean()
ds_df2['Close MA200 shift year']=ds_df2['Close MA200'].shift(253)
ds_df2['Yearly Growth']=(ds_df2['Close MA200']-ds_df2['Close MA200 shift year'])/ds_df2['Close MA200 shift year']
fig=make_subplots(specs=[[{"secondary_y":True}]])
fig.add_trace(go.Scatter(x=ds_df2['Date'], y=ds_df2['Close MA200'],mode='lines',name='Close MA200'),secondary_y=False,)
fig.add_trace(go.Scatter(x=ds_df2['Date'], y=ds_df2['Yearly Growth']*100,mode='lines',name='Yearly Growth'),secondary_y=True,)
fig.update_layout(title='Close MA200 and Yearly Growth',width=800,height=400,)
fig.update_xaxes(title_text="Date")
fig.update_yaxes(title_text="Close MA200 USD",secondary_y=False)
fig.update_yaxes(title_text="Yearly Growth %",secondary_y=True)
fig.show()

