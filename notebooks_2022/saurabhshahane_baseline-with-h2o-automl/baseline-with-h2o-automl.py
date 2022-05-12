#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import h2o
import matplotlib.pyplot as plt
import seaborn as sns
from h2o.automl import H2OAutoML


# In[ ]:


h2o.init()


# In[ ]:


train_df = h2o.import_file('../input/tabular-playground-series-may-2022/train.csv',destination_frame='train_df')


# In[ ]:


test_df = h2o.import_file("../input/tabular-playground-series-may-2022/test.csv")


# In[ ]:


train_df.describe()


# In[ ]:


train_df = train_df.drop('id')


# In[ ]:


predictor_cols = train_df.columns[:-1]


# In[ ]:


train, valid, test = train_df.split_frame(ratios=[0.6,0.2], seed=1111)
response = "target"
train[response] = train[response].asfactor()
valid[response] = valid[response].asfactor()
test[response] = test[response].asfactor()
print("Number of rows in train, valid and test set : ", train.shape[0], valid.shape[0], test.shape[0])


# In[ ]:


aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=predictor_cols, y=response, training_frame=train)


# In[ ]:


aml.leaderboard


# In[ ]:


test_df = test_df.drop('id')


# In[ ]:


pred = aml.predict(test_df)


# In[ ]:


pred.head()


# In[ ]:


p1 = pred['p1']


# In[ ]:


#convert to pandas
p1_list = h2o.as_list(p1)


# In[ ]:


submission = pd.read_csv('../input/tabular-playground-series-may-2022/sample_submission.csv')


# In[ ]:


submission.target = p1_list['p1'].values


# In[ ]:


submission.to_csv('Submission_AutoML.csv',index=False)

