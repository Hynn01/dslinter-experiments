#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

TRAIN_PATH = "../input/store-sales-time-series-forecasting/train.csv"
TEST_PATH = "../input/store-sales-time-series-forecasting/test.csv"
SAMPLE_SUBMISSION_PATH = "../input/store-sales-time-series-forecasting/sample_submission.csv"

HOLIDAYS_EVENTS = "../input/store-sales-time-series-forecasting/holidays_events.csv"
OIL = "../input/store-sales-time-series-forecasting/oil.csv"
STORES = "../input/store-sales-time-series-forecasting/stores.csv"
TRANSACTIONS = "../input/store-sales-time-series-forecasting/transactions.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

holidays_events = pd.read_csv(HOLIDAYS_EVENTS)
oil = pd.read_csv(OIL)
stores = pd.read_csv(STORES)
transactions = pd.read_csv(TRANSACTIONS)

NEW_TRAIN_PATH = "train.csv"
NEW_TEST_PATH = "test.csv"


# In[ ]:


sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
len(sub)


# In[ ]:


train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)


# # merge oil

# In[ ]:


train_oil = pd.merge(train,oil,how='left',on="date")
# train_oil = train_oil.reset_index()
train_oil.head(5)


# In[ ]:


len(train_oil)


# In[ ]:


test_oil = pd.merge(test,oil,how='left',on="date")
# test_oil = test_oil.reset_index()
test_oil.head(5)


# In[ ]:


len(test_oil)


# # merge stores

# In[ ]:


train_oil_stores = train_oil.merge(stores, 
                                        on = ["store_nbr"], 
                                        how='left')
# train_oil_stores = train_oil_stores.reset_index()
train_oil_stores.head()


# In[ ]:


len(train_oil_stores)


# In[ ]:


test_oil_stores = test_oil.merge(stores, 
                                        on = ["store_nbr"], 
                                        how='left')
# test_oil_stores = test_oil_stores.reset_index()
test_oil_stores.head()


# In[ ]:


len(test_oil_stores)


# # merge transactions

# In[ ]:


train_oil_stores_transaction = train_oil_stores.merge(transactions, 
                                        on = ["date","store_nbr"], 
                                        how='left')
# train_oil_stores_transaction = train_oil_stores_transaction.reset_index()
train_oil_stores_transaction.head()


# In[ ]:


len(train_oil_stores_transaction)


# In[ ]:


test_oil_stores_transaction = test_oil_stores.merge(transactions, 
                                        on = ["date","store_nbr"], 
                                        how='left')
# test_oil_stores_transaction = test_oil_stores_transaction.reset_index()
test_oil_stores_transaction.head()


# In[ ]:


len(test_oil_stores_transaction)


# # merge holidays_events

# In[ ]:


train_oil_stores_transaction_events = train_oil_stores_transaction.merge(holidays_events, 
                                        on = ["date","type"], 
                                        how='left')
# train_oil_stores_transaction_events = train_oil_stores_transaction_events.reset_index()
train_oil_stores_transaction_events.head()


# In[ ]:


len(train_oil_stores_transaction_events)


# In[ ]:


test_oil_stores_transaction_events = test_oil_stores_transaction.merge(holidays_events, 
                                        on = ["date","type"], 
                                        how='left')
# test_oil_stores_transaction_events = test_oil_stores_transaction_events.reset_index()
test_oil_stores_transaction_events.head()


# In[ ]:


len(test_oil_stores_transaction_events)


# In[ ]:


train_oil_stores_transaction_events.to_csv(NEW_TRAIN_PATH,index=False)
test_oil_stores_transaction_events.to_csv(NEW_TEST_PATH,index=False)

