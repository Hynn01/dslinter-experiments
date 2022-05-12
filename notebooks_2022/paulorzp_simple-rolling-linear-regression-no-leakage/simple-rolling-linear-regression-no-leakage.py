#!/usr/bin/env python
# coding: utf-8

# ### Rolling Linear Regression (no leakage)
# ### *“Simplicity is the ultimate sophistication.”*
# <div style="text-align: center"><b><i>Leonardo da Vinci</i></b></div>

# In[ ]:


import numpy as np
import pandas as pd
import jpx_tokyo_market_prediction

train_path = "../input/jpx-tokyo-stock-exchange-prediction/train_files/"
usecols = ["Date","SecuritiesCode","Close","Target"]
df = pd.read_csv(train_path+"stock_prices.csv", usecols=usecols)
df = df[df["Date"]==df.Date.iat[-1]] # using only the last day from train set
cods = df.SecuritiesCode.unique()

env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

for (prices, _, _, _, _, submission) in iter_test:
    prices.loc[:,"Target"] = np.nan
    df = pd.concat([df, prices[usecols]])
    df = df.sort_values(["SecuritiesCode", "Date"])
    df.ffill(inplace=True)
    targets = []
    for cod in cods:
        y = df[df.SecuritiesCode==cod].Close[-2:].values # Get two last close values
        targets.append(1-y[1]/(y[1]+(y[1]-y[0]))) # Using the linear regression gradient as a target 
    tr = df[df.Date==prices.Date.iat[0]].copy()
    tr.Target = targets 
    tr.loc[:,"Rank"]=tr.groupby("Date")["Target"].rank(method="first",ascending=False)-1 
    pred = tr.set_index("SecuritiesCode")["Rank"].astype(int).to_dict()
    submission["Rank"] = submission["SecuritiesCode"].map(pred)
    env.predict(submission)

