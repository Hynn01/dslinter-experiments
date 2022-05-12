#!/usr/bin/env python
# coding: utf-8

# # Demo on Submission for the JPX Stock Prediction Competition
# 
# 

# Check the source data path

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Submission

# In[ ]:


import jpx_tokyo_market_prediction

env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()


# ## Load the Raw Data

# In[ ]:


old_price_total_df = pd.read_csv('../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv')

new_price_total_df = pd.read_csv('../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv')


# In[ ]:


price_total_df = pd.concat([old_price_total_df,new_price_total_df])

price_total_df = price_total_df.loc[price_total_df['Date']>'2021']


# In[ ]:


price_total_df.tail()


# In[ ]:


trading_dates = np.array(sorted(price_total_df['Date'].unique()))
stock_ids = np.array(sorted(price_total_df['SecuritiesCode'].unique()))

temp_mat = pd.DataFrame(np.nan, index=stock_ids, columns=trading_dates)

def create_factor(item, temp_mat=temp_mat):
    output_mat = pd.pivot_table(price_total_df,
                                values=item,
                                index='SecuritiesCode', columns='Date')
    
    output_factor = temp_mat.copy()
    output_factor.loc[output_mat.index, output_mat.columns] = output_mat.values

    return output_factor

close_factor = create_factor('Close')

# adjustment 
adj_factor = create_factor('AdjustmentFactor')

adjusted_factor = adj_factor.iloc[:,::-1].cumprod(axis=1).iloc[:,::-1].fillna(axis=1,
                                                                              method='bfill')

adjusted_factor  = adjusted_factor.T.div(adjusted_factor.iloc[:,-1]).T

close_adj_factor = adjusted_factor*close_factor

rtn_mat = close_adj_factor.pct_change(1,axis=1)


# ## Baseline Model: Reversal Factor

# In[ ]:


reversal_5d = -np.log(rtn_mat.T+1).rolling(5).sum().T.shift(2, axis=1)


# Access the features from test set

# In[ ]:



counter = 0


for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    pred_dt = prices['Date'][0]
    print(" => {} Now generating ranking at date [{}]".format(counter+1, pred_dt))
    pred_stocks = sample_prediction['SecuritiesCode'].values
    
    # access the predict 
    signal = reversal_5d.loc[pred_stocks,pred_dt]
    # convert to rank
    pred_rank = signal.rank()-1
    #### Convert the Ranking ####
    # assign the rank score
    sample_prediction['Rank'] = np.arange(len(sample_prediction))  #pred_rank[pred_stocks].values.astype(int) 
    
    #### upload prediction ####
    env.predict(sample_prediction)
    ########################
    
    counter+=1
    
    


# In[ ]:


get_ipython().system(' head submission.csv')


# In[ ]:





# In[ ]:




