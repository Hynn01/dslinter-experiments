#!/usr/bin/env python
# coding: utf-8

# # 0. Intro
# > The point isn't how many data you had, How use.
# 
# 
# ## library
# ![image.png](attachment:3f3ea13c-0ddb-4bec-b356-7980d2a137cd.png)
# 
# ![image.png](attachment:36cd38a9-28a4-4550-a9d2-9f6dde61b131.png)
# 
# ![image.png](attachment:bc0dbed7-9d98-42da-99fd-49e250f96b6d.png)
# 

# In[ ]:


# library import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# # 1. Data

# In[ ]:


# data load
stock_filepath = '../input/faang-stocks-covid190101202004012022/faang_stocks_pandemic_data.csv'
stock_data = pd.read_csv(stock_filepath)
stock_data.head()


# In[ ]:


stock_data.info()


# In[ ]:


stock_data.Name.unique()


# In[ ]:


# FAANG
facebook = stock_data.loc[stock_data.Name=='Facebook']
amazon = stock_data.loc[stock_data.Name=='Amazon']
apple = stock_data.loc[stock_data.Name=='Apple']
netflix = stock_data.loc[stock_data.Name=='Netflix']
google = stock_data.loc[stock_data.Name=='Google']
amazon.head()


# # 2. Visualization
# ![image.png](attachment:d4213118-6202-4398-917b-91746e5a69b9.png)

# ## Line Chart
# 

# In[ ]:


plt.figure(figsize=(12, 6))
plt.title('FAANG 2020-2022')
stock_data['Date'] = stock_data['Date'].apply(str)
sns.lineplot(data=stock_data,x='Date', y='High', hue='Name')


# ## Scatterplot

# In[ ]:


sns.scatterplot(x=amazon['High'], y=amazon['Open'])


# In[ ]:


sns.swarmplot(x=stock_data['Name'], y=stock_data['Volume'])


# ## Bar Chart

# In[ ]:


plt.title('Bar Chart')
sns.barplot(x=stock_data.Name, y = stock_data.High)


# ## Canddle Stock
# ![image.png](attachment:7687bec1-864f-4452-bcc1-4d24b96dea16.png)
# 

# # 3. Amazon Discount day!!

# In[ ]:


#from plotly.subplot import make_subplots
import plotly.graph_objects as go
import datetime
'''
start = datetime.datetime(2020, 3, 1)
finish = datetime.datetime(2020, 3, 31)
fig = plt.figure(figsize=(12, 8))
'''
amazon = stock_data.loc[stock_data.Name == "Amazon"]
amazon = amazon.reset_index()
#amazon['Date'] = amazon['Date'].apply(lambda x:datetime.strftime(x, '%Y-%m-%d'))
fig = go.Figure(data=[go.Candlestick(x=amazon['Date'],
                                     open=amazon['Open'],
                                    high=amazon['High'],
                                    low=amazon['Low'],
                                    close=amazon['Close'])])
fig.show()


# In[ ]:


get_ipython().system('pip install finance-datareader')


# In[ ]:


import FinanceDataReader as fdr
df_nasdaq = fdr.StockListing('NASDAQ')


# In[ ]:


df = fdr.DataReader('AMZN', '2022')
df.head()


# In[ ]:


amazon = df.reset_index()
#amazon['Date'] = amazon['Date'].apply(lambda x:datetime.strftime(x, '%Y-%m-%d'))
fig = go.Figure(data=[go.Candlestick(x=amazon['Date'],
                                     open=amazon['Open'],
                                    high=amazon['High'],
                                    low=amazon['Low'],
                                    close=amazon['Close'])])
fig.show()


# # References
# - [Course - Data Visualization](https://www.kaggle.com/learn/data-visualization)
# - [The reson Why you have to do Visualization](https://www.bloter.net/newsView/blt201901230009)
# - [[Python] FinanceDataReader로 주식데이터 가져오기](https://seong6496.tistory.com/169)
# - [[Python] cufflinks, QunatFig, Plotly 주식 그래프 예쁘고 편하게 그리기 - 파이썬 주식투자](https://chancoding.tistory.com/117?category=846070)
