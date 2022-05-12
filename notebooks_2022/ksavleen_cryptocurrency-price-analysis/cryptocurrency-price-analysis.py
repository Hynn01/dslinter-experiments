#!/usr/bin/env python
# coding: utf-8

# #### Problem Statement
# * Extract OHLCV data from Binance
# * Plot Candlestick patterns for "1d" for last 1 year
# * Exponential Moving Average (EMA) of 200 days on the candlestick chart

# #### Flow of Analysis
# * Import Relevant Libraries
# * Load data from Binance Exchange for Bitcoin(BNB/BTC)/USDT
# * Clean OHLCV data
# * Method for Simple Moving Average(SMA/EMA) for 7 days, 20 days, 200 days
# * Plot Line Graph for Price vs SMA7, SMA20, SMA200

# In[ ]:


get_ipython().system(' pip install ccxt')
#CryptoCurrency eXchange Trading library
#run line of code/script in (linux)terminal and not in python (python environment)
#pip: package that connects our env to PyPi global repository


# In[ ]:


get_ipython().system(' pip install mpl_finance')


# In[ ]:


import pandas as pd 
import ccxt
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
#magical function which renders the graph along with kaggle notebook
#from ccxt.base import exchange
import datetime                 
#libraries to plot candlesticks                                
from mpl_finance import candlestick_ohlc     
import matplotlib.dates as mpdates
import plotly.graph_objects as go


# In[ ]:


exchange = ccxt.binance() 
#will have necessary backend ready for connecting to BINANCE portal
#initializing the connector


# #### API Application Programming Interface:
# one program(kaggle notebook) interacts with other program(Binance portal)
# 
# Service providing the info puts some limitation on the data to be shared as it costs 
# for Computation and Processing at Binance's end
# * we provide the start and end date
# * for every request, we create a batch i.e 20 records/request or 20 days/year
# 

# In[ ]:


exchange.milliseconds() 
#Current timestamp
#View the Human Readable Date using https://www.epochconverter.com/
#standard nomenclature in computer science


# In[ ]:


if exchange.has['fetchOHLCV']:
    #conditional loop to check exchange has fetchOHLCV
    #get info from Binance :opening, high,low,closing price
    ticker = "BTC/USDT"
    #Binance recognizes the crytocurrency
    since = exchange.parse8601("2020-11-01T00:00:00Z") 
    #Unix epoch: number of seconds that have elapsed since January 1, 1970
    all_orders = []  #blank list to be used for appending
    while since < exchange.milliseconds():  
        #start time is less than current time
        #exchange.milliseconds(): converts the current time 
        limit = 20  # 20 days of data will be extracted from 1st of Nov
        ohlcv = exchange.fetch_ohlcv(symbol = ticker, since = since, timeframe = "1d", limit = limit)
        #class exchange has method fetch_ohlcv
        if len(ohlcv) > 0:
            since = ohlcv[len(ohlcv)-1][0] + 1 
            #from 20 days of data,extract first element from the last request i.e milliseconds
            #and increment by 1 i.e now 21st Nov
            all_orders += ohlcv
            #appending the list received from Binance to our empty list
        else:
            break


# In[ ]:


print(f"First 5 records: \n{all_orders[0:5]}")
#we have data for below number of days
print(f"\nTotal records: {len(all_orders)}")
print(f"\nLast record is: {all_orders[-1]}")


# In[ ]:


def cleanData(ohlcv_list):
    temp = []
    for i in ohlcv_list:
        timestamp_with_ms = i[0]
        dt = datetime.datetime.fromtimestamp(timestamp_with_ms / 1000)
        formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        #converts a datetime object containing current date and time to string formats
        #extracting upto final 3 values to avoid high accuracy date
        i[0] = dt
        temp.append(i)
    return temp


# In[ ]:


data = cleanData(ohlcv_list = all_orders)


# In[ ]:


ohlcv_df = pd.DataFrame(data, columns = ["date","open_price","high_price","low_price","close_price","volume_traded"])


# In[ ]:


ohlcv_df.tail(n=6)


# #### Data Exploration

# In[ ]:


ohlcv_df.iloc[523,:]


# In[ ]:


ohlcv_df.loc[254,["date","low_price"]]


# In[ ]:


ohlcv_df.describe()


# * 25 percentile means 25% of data has values less tha 36650 as open value
# * 75 percentile means 75% of datapoints in our dataset are less than 50449 as open value
# * 50 percentile is Median
# * Rolling Average : what is the average based on specific range of dates
# * TimeSeries data: Data collected over a period of time(current use case)
# * EWMA and SMA: 
#     * Exponentially Weighted Moving Average(recent timeseries data has higher weightage in the calculation of average as compared to previous data) 
#     * Simple Moving Average(looks for 100 days of data without giving priority to latest or old days, no weightage to recency of data)

# In[ ]:


#Calculating EWMA on the open prices; for 200 days
ohlcv_df["ewm200"]=ohlcv_df.open_price.ewm(span=200,min_periods = 0,adjust = False,ignore_na = False).mean()


# In[ ]:


ohlcv_df[-3:]


# #### Plotting Line Graph

# In[ ]:


plt.figure(figsize = (20,10))
plt.plot(ohlcv_df.date, ohlcv_df.open_price, color = "black", label = "Opening Value")
plt.plot(ohlcv_df.date, ohlcv_df.ewm200, color = "red", label = "EWMA 200")
plt.title("Opening Price of Bitcoin vs EWMA 200")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Bitcoin Price")
plt.show()


# #### Plotting Candlesticks

# In[ ]:


cs_df = ohlcv_df.copy()


# In[ ]:


try:
    cs_df["date"] = cs_df["date"].map(mpdates.date2num)
except:
    pass
#re-execution of date conversion will throw an error so we use try-except
#try: try line of code : transform/(.map is a in-built python method to)apply transformation across all elements of a list
#date is transformed to numeric value required by matpotlib to plot candlesticks
#if a code fails it comes to except: don't do anything just move on

#Creating Subplots
fig,ax = plt.subplots()
fig.set_figwidth(20)
fig.set_figheight(10)

#Plotting the data
candlestick_ohlc(ax,
                 cs_df[["date","open_price","high_price",
                        "low_price","close_price","volume_traded"]].values,
                 width = 0.8,colorup="green",colordown="red",alpha = 0.8)
#Plotting EWMA
plt.plot(ohlcv_df.date, ohlcv_df.ewm200, color = "red", label = "EWMA 200")
#Setting labels
ax.set_xlabel("Date")
ax.set_ylabel("Price")

#Setting Title
plt.title("Bitcoin Candlesticks")

#Formatting Date
date_format = mpdates.DateFormatter("%d-%m-%Y")
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()

fig.tight_layout()
#Show the plot
plt.show()


# #### Plotly : Interactive Graphs

# In[ ]:


fig = go.Figure(data = [go.Candlestick(x = cs_df["date"],
                                       open = cs_df["open_price"],
                                       high = cs_df["high_price"],
                                       low = cs_df["low_price"],
                                       close = cs_df["close_price"])])
#Line plot with Plotly
fig.add_trace(go.Scatter(x = cs_df["date"], y = cs_df["ewm200"],
                        mode = "lines",name = "ewm200",
                        line = dict(color = "teal")))
#remove grid
fig.update_layout(title = "Bitcoin Candlesticks", plot_bgcolor="white")
#Show the plot
fig.show()

