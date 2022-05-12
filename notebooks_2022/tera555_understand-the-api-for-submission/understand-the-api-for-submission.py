#!/usr/bin/env python
# coding: utf-8

# # Description of Competition Summary
# 
# ```
# You must submit to this competition using the provided python time-series API, which ensures that models do not peek forward in time.
# ```
# From the above, we can expect them to be in time-series order.  
# When using lag features, it is important to check at the code level that they are actually passed in time-series order.

# # Verify that data is passed in time-series order

# In[ ]:


import numpy as np
import time
import jpx_tokyo_market_prediction
env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

prev_date = None
for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    current_date = prices["Date"].iloc[0]
    print(f"prev: {prev_date} current: {current_date}")
    if prev_date != None:
        if current_date <= prev_date:
            raise ValueError("MyError!!")
    prev_date = current_date
    sample_prediction['Rank'] = np.arange(len(sample_prediction))
    env.predict(sample_prediction)


# It is in time-series order because it is not in error.
# 
# # Reference Links
# https://www.kaggle.com/competitions/g-research-crypto-forecasting/discussion/289000#1587704  
# https://www.kaggle.com/competitions/jane-street-market-prediction/discussion/199203#1089839  
