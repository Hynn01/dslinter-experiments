#!/usr/bin/env python
# coding: utf-8

# This notebook using AutoGluon to predict the sale price of the NFTs based on the last sales. The best RMSE is about 33.7 (the leaderboard prints negative RMSE - [a known issue](https://github.com/awslabs/autogluon/issues/524). Given the mean and std of sales price is 3478 and 3675, the prediction is pretty good. 

# In[ ]:


get_ipython().system('pip install autogluon.tabular')


# In[ ]:


# import packages
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from autogluon.tabular import TabularPredictor


# In[ ]:


# read csv data into pandas dataframe
df = pd.read_csv('/kaggle/input/crypto-coven/witches.csv')


# In[ ]:


# generate pandas profiling report
profile = ProfileReport(df, minimal=True)  # this option turns off many expensive calculations for large datasets


# In[ ]:


profile.to_notebook_iframe()


# In[ ]:


# create a new column to calculate the sales price in USD
df['sales_price'] = df['last_sale.total_price']/1000000000000000000 * df['last_sale.payment_token.usd_price']


# In[ ]:


# the label cannot have missing values
df = df.loc[~df.sales_price.isnull()]
df.sales_price.isnull().sum()


# In[ ]:


# Split the data into a training set and a test set for autogluon
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42)
train.head()


# In[ ]:


df.sales_price.describe()


# In[ ]:


from autogluon.tabular import TabularDataset, TabularPredictor
predictor = TabularPredictor(label='sales_price').fit(train)
predictor.leaderboard(test, silent=True)


# In[ ]:




