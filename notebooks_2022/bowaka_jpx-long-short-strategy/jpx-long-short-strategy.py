#!/usr/bin/env python
# coding: utf-8

# # Long Short Strategy
# 
# I made a very basic implementation of the long-short strategy recommanded by tmrtj9999 in his thread:
# https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/discussion/320886
# 
# The idea of the strategy is to long securities that underperformed and short securities that overperformed.
# 
# According to the thread, the strategy should perform well in bull markets, and very badly in bear markets. 

# # Exploration, and backtesting

# In[ ]:


import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def calc_return(X,targets):
    '''Home made function to calculate the revenu of a strategy, given the ranks'''
    longs = X<200
    longs = ((1-(X/199))+1)*longs/600
    shorts = X>1799
    shorts = -((X-1800)/199+1)*shorts/600
    return (targets*(longs+shorts)).sum(axis=1)


# ## Data preparation

# In[ ]:


stocks = pd.read_csv('../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv')
stocks.Date = pd.to_datetime(stocks.Date)
targets = pd.pivot(stocks, index = 'Date', values = 'Target', columns = 'SecuritiesCode')


# ## Create "train" data
# 
# Don't forgeet to shift data by 2 days as targets have a 2 days lag

# In[ ]:


# A rolling factor, averaging on 2 values seems to give better results. Don't forget also to shift values !
roll = 2
train = targets.rolling(roll).mean().shift(2).iloc[3:]


# # Model evaluation
# 
# Model is evaluated against the average of the 2000 stocks

# In[ ]:


# A rolling factor, averaging on 2 values seems to give better results. Don't forget also to shift values !
roll = 2
train = targets.rolling(roll).mean().shift(2).iloc[3:]

#The values are sorted date by day such than values that overperformed are ranked badly
X = np.argsort(np.argsort(train))
y = targets.loc[X.index]

#Benchmark, averaging all securities for a given day
bm = y.mean(axis=1)

#Return of our strategy
r = calc_return(X,y)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x = bm.index,
        y = bm.cumsum().values,
        name = "Benchmark",
        marker = {"color":"black"}
    )
)

fig.add_trace(
    go.Scatter(
        x = r.index,
        y = r.cumsum().values,
        name = "Long Short Strategy",
        marker = {"color":"green"}
    )
)

fig.update_layout(template="presentation", title = "Long-Short strategy cumulative return vs Benchmark")


# ## Sharp ratios
# 
# The yearly sharp ratio is calculated on the whole dataset. 
# For recall, an acceptable yearly sharp ratio should have a value above 1, a good ratio a value above 2, and an excellent sharp ratio a value above 3.

# In[ ]:


print(f"sharp ratio, benchmark: {round(bm.mean()/bm.std()*252**0.5,3)}")
print(f"sharp ratio, Long - Short strategy: {round(r.mean()/r.std()*252**0.5,3)}")


# In[ ]:


dividends = pd.pivot(stocks, index = 'Date', values = 'ExpectedDividend', columns = 'SecuritiesCode')
dvs = dividends.fillna(0)
train2 = train.copy()
subdvs = dvs.loc[train2.index]
train2[subdvs!=0] = train2[subdvs!=0] +1
train2[subdvs.shift(-1)!=0] = train2[subdvs.shift(-1)!=0] +0.5

X = np.argsort(np.argsort(train2))
y = targets.loc[X.index]

#Benchmark, averaging all securities for a given day
bm = y.mean(axis=1)

#Return of our strategy
r = calc_return(X,y)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x = bm.index,
        y = bm.cumsum().values,
        name = "Benchmark",
        marker = {"color":"black"}
    )
)

fig.add_trace(
    go.Scatter(
        x = r.index,
        y = r.cumsum().values,
        name = "Long Short Strategy",
        marker = {"color":"green"}
    )
)

fig.update_layout(template="presentation", title = "Long-Short strategy cumulative return vs Benchmark")


# In[ ]:


print(f"sharp ratio, benchmark: {round(bm.mean()/bm.std()*252**0.5,3)}")
print(f"sharp ratio, Long - Short strategy: {round(r.mean()/r.std()*252**0.5,3)}")


# # Score on evaluation data

# In[ ]:


evalstocks = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv")
evalstocks.Date = pd.to_datetime(evalstocks.Date)
evaltargets = pd.pivot(evalstocks, index = 'Date', values = 'Target', columns = 'SecuritiesCode')

evaldividends = pd.pivot(evalstocks, index = 'Date', values = 'ExpectedDividend', columns = 'SecuritiesCode')
dvs = evaldividends.fillna(0)

roll = 2
train = pd.concat([targets,evaltargets]).rolling(roll).mean().shift(2).loc[evaltargets.index]

subdvs = dvs.loc[train.index]
train[subdvs!=0] = train[subdvs!=0] +1
train[subdvs.shift(-1)!=0] = train[subdvs.shift(-1)!=0] +0.5

X = np.argsort(np.argsort(train))
y = evaltargets.loc[X.index]

#Benchmark, averaging all securities for a given day
bm = y.mean(axis=1)

#Return of our strategy
r = calc_return(X,y)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x = bm.index,
        y = bm.cumsum().values,
        name = "Benchmark",
        marker = {"color":"black"}
    )
)

fig.add_trace(
    go.Scatter(
        x = r.index,
        y = r.cumsum().values,
        name = "Long Short Strategy",
        marker = {"color":"green"}
    )
)

fig.update_layout(template="presentation", title = "Long-Short strategy cumulative return vs Benchmark")

print(f"sharp ratio, benchmark: {round(bm.mean()/bm.std()*252**0.5,3)}")
print(f"sharp ratio, Long - Short strategy: {round(r.mean()/r.std()*252**0.5,3)}")

fig.show()


# # Submit the strat

# In[ ]:


import jpx_tokyo_market_prediction

current_closes = pd.pivot(stocks, index = 'Date', values = 'Close', columns = 'SecuritiesCode')
current_targets = targets.copy().fillna(0)

env = jpx_tokyo_market_prediction.make_env()   # initialize the environment
iter_test = env.iter_test()    # an iterator which loops over the test files
for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    
    date = pd.to_datetime(prices.Date.iloc[0])
    #Update close table
    close_ = pd.pivot_table(prices, values = "Close", columns = "SecuritiesCode", index = "Date")
    current_closes = pd.concat([current_closes, close_]).ffill()
    
    #Update target table
    target_ = pd.DataFrame((current_closes.iloc[-1]-current_closes.iloc[-2])/current_closes.iloc[-1]).rename(columns = {0:date}).T.fillna(0)
    current_targets = pd.concat([current_targets, target_])
    
    #Calculate variations of the two last targets
    scores = current_targets.iloc[-2:].sum()
    divs = pd.pivot_table(prices.fillna(0), values = "ExpectedDividend", columns = "SecuritiesCode", index = "Date").iloc[0]
    scores[divs!=0] = scores[divs!=0]+1
    sample_prediction['Rank'] = scores.argsort().argsort().reindex(sample_prediction["SecuritiesCode"]).values
    env.predict(sample_prediction)   # register your predictions

