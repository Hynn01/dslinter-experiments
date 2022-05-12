#!/usr/bin/env python
# coding: utf-8

# # Notes and Summary of Kernel 
# 
# This is a experiment on feature extraction and feature selection, the inital idea and part of the code came from this fantastic notebook from [JEROENVDD](https://www.kaggle.com/code/jeroenvdd/tsflex-x-tsfresh-feature-extraction) and the [tsflex github](https://github.com/predict-idlab/tsflex)

# In[ ]:


#!pip install powershap 


# In[ ]:


#!conda install catboost


# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import seaborn as sns 

from catboost import CatBoostClassifier

import gc

from powershap import PowerShap


# In[ ]:


train_feats = pd.read_csv("./train_feats_nopowershap.csv",index_col = 0)
test_feats = pd.read_csv("./test_feats_nopowershap.csv",index_col = 0)


train_lables = pd.read_csv("./train_labels.csv")
# sub= pd.read_csv("../input/tabular-playground-series-apr-2022/sample_submission.csv", index_col = 0)


# In[ ]:


print("train shape:",train_feats.shape)
print("test shape:",test_feats.shape)


# # Downcasting 

# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
 
    return df

# reduce_mem_usage(train_feats)
# reduce_mem_usage(test_feats)


# # Powershap 
# Feature selection 

# In[ ]:


X= train_feats
y = train_lables["state"]


# In[ ]:


from catboost import CatBoostClassifier

selector = PowerShap(
    model=CatBoostClassifier(iterations=3000, verbose=0, od_type='Iter', od_wait=100, task_type="CPU")
)

selector.fit(X, y)  # Fit the PowerShap feature selector


# In[ ]:


print([col for col in train_feats.columns[selector._get_support_mask()]])


# # Save data 

# In[ ]:


selector._processed_shaps_df.to_csv("tsflex_powershap_out.csv")
selector._processed_shaps_df


# In[ ]:


train_feats[train_feats.columns[selector._get_support_mask()]].to_csv("train_add_feats.csv")
test_feats[test_feats.columns[selector._get_support_mask()]].to_csv("test_add_feats.csv")


# In[ ]:


train_feats[train_feats.columns[selector._get_support_mask()]]


# In[ ]:




