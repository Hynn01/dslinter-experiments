#!/usr/bin/env python
# coding: utf-8

# Imports

# In[ ]:


TRAIN_PATH = '../input/tabular-playground-series-may-2022/train.csv'
TEST_PATH = '../input/tabular-playground-series-may-2022/test.csv'
SUBMISSION_PATH = '../input/tabular-playground-series-may-2022/test.csv'


# In[ ]:


import pandas as pd
import matplotlib as plt


# # Train Data Expolaration

# In[ ]:


trainDf = pd.read_csv(TRAIN_PATH)
trainDf.describe()


# In[ ]:


trainDf.columns


# Some intuitions:
# 1. As the mean of the target is 0.486488 the train data is fairly balanced
# 2. Some values are categorical and some are continous
# 3. From summary provided in kaggle data, f_07, f_08, f_09, f_10, f_11, f_12, f_13, f_14, f_15, f_16, f_17, f_18, f_29, f_30
# 4. The continuous data are prefectly in gaussian distribution, categoricals are not
# 5. The results have to be given in Probability - Regression Problem

# # Continuous data 

# In[ ]:




