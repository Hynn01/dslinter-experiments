#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install autogluon')


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


from autogluon.text import TextPredictor
import pandas as pd

train_data = pd.read_csv('sub_train.csv')
test_data = pd.read_csv('sub_val.csv')
test_data_nolab = test_data.drop(columns=['Price']) 
time_limit = 1 * 60  # set to larger value in your applications
predictor = TextPredictor(label='Price', path='autogluon')
predictor.fit(train_data, time_limit=time_limit)
y_pred = predictor.predict(test_data_nolab)


# In[ ]:


from sklearn.metrics import mean_squared_error
import numpy as np
RMSE = np.sqrt(mean_squared_error(test_data['Price'], y_pred))
RMSE


# In[ ]:




