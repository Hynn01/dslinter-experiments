#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('mkdir -p price_of_books')
get_ipython().system('wget https://automl-mm-bench.s3.amazonaws.com/machine_hack_competitions/predict_the_price_of_books/Data.zip -O price_of_books/Data.zip')
get_ipython().system('cd price_of_books && unzip -o Data.zip')
get_ipython().system('ls price_of_books/Participants_Data')


# In[ ]:


get_ipython().system('pip install openpyxl')


# In[ ]:


import numpy as np 
import pandas as pd 
from sklearn.model_selection import KFold

train = pd.read_excel('./price_of_books/Participants_Data/Data_Train.xlsx')


# In[ ]:


kf = KFold(n_splits=3, random_state=1001,shuffle=True)
for i, (train_index, val_index) in enumerate(kf.split(train)):
    trn= train.iloc[train_index].reset_index()
    val= train.iloc[val_index].reset_index()
    
trn = trn.drop(columns=['index'])
val = val.drop(columns=['index'])

val.to_csv('sub_val.csv',index=False)
trn.to_csv('sub_train.csv',index=False)


# In[ ]:


from sklearn.metrics import mean_squared_log_error
import re
import gc
import matplotlib.pyplot as plt
#H2O
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator

# Pandas setting to display more dataset rows and columns
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# In[ ]:


train = pd.read_csv("sub_train.csv")
test1 = pd.read_csv('sub_val.csv')


# In[ ]:


test = test1.drop(columns=['Price'])


# In[ ]:


test.head(3)


# In[ ]:


h2o.init()
all_train = h2o.H2OFrame(train)
#Get columns names for Building H2O Models
target = 'Price'
predictors = [f for f in all_train.columns if f not in ['Price']]


# In[ ]:


from h2o.automl import H2OAutoML


# In[ ]:


aml = H2OAutoML(max_models=1, seed=1331)
aml.train(x=predictors, y=target, training_frame=all_train)


# In[ ]:


test = h2o.H2OFrame(test)
prediction = aml.leader.predict(test)


# In[ ]:


prediction[['predict']]


# In[ ]:


pred = prediction.as_data_frame()


# In[ ]:


from sklearn.metrics import mean_squared_error
RMSE = np.sqrt(mean_squared_error(test1['Price'], pred))


# In[ ]:


RMSE


# In[ ]:




