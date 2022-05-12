#!/usr/bin/env python
# coding: utf-8

# #### We Will use the last 28 days data on category-store level as prediction for next 28(validation) and later 28 days(as evaluation) as our first submission. We will check the score and improve it later in our next attempt. 
# ## If you do not have any submission and are stuck at Novice level, use this Notebook, run it and submit the output file to become a contributor.
# #### Feel free to fork and update the notebook.

# In[ ]:


ls ../input/m5-forecasting-accuracy/


# In[ ]:


import pandas as pd
sample = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')


# In[ ]:


sample.head()


# In[ ]:


# util functions to reduce pandas dataframe memory
import numpy as np
def df_mem_reduce(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
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
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

sales_train = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
sales_train = df_mem_reduce(sales_train)


# In[ ]:


calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
print(calendar.head())
print(calendar.tail())
calendar['date'].shape


# In[ ]:


sales_train.head()


# In[ ]:


sales_train.shape


# In[ ]:


sales_train.loc[sales_train['item_id']=='HOBBIES_1_001',:]


# In[ ]:


sales_train_dates_columns = sales_train.columns[6:]
sales_train_dates_columns_recent_28 = sales_train_dates_columns[-28:]

sales_train.loc[sales_train['item_id']=='HOBBIES_1_001',sales_train_dates_columns_recent_28].head()


# In[ ]:


sales_train_agg_cat_store = sales_train.groupby(['cat_id','store_id'])[sales_train_dates_columns_recent_28].mean().reset_index()
sales_train_agg_cat_store['_cat_store'] = sales_train_agg_cat_store.apply(lambda x: x['cat_id'] + "_" + x['store_id'], axis=1)
sales_train_agg_cat_store.drop(['cat_id','store_id'],axis=1,inplace=True)
newCols = {x:'F'+str(id_+1) for id_,x in enumerate(sales_train_agg_cat_store.columns) if not x.startswith('_')}
sales_train_agg_cat_store.rename(columns=newCols, inplace=True)
sales_train_agg_cat_store.head()


# In[ ]:


sample['_cat_store'] = sample.apply(lambda x:x['id'].split('_')[0]+"_"+x['id'].split('_')[3]+"_"+x['id'].split('_')[4] , axis=1 )#cat_id	store_id	
print(sample.head())
sample_joint = sample[['id','_cat_store']].merge(sales_train_agg_cat_store, on='_cat_store', how='left')
print(sample_joint.head())


# In[ ]:


import os
os.system("rm -rf output")
os.system("mkdir -p output")
if '_cat_store' in sample_joint.columns:
    sample_joint.drop('_cat_store',axis=1, inplace=True)
sample_joint.to_csv('output/submission_last_28_days.csv', index=False, float_format='%.2f')


# In[ ]:


ls -alh output/


# In[ ]:


get_ipython().system('head output/submission_last_28_days.csv')


# # Use this zipped file,upload and submit. 
# # Contratulations, You are a Contributor now.

# In[ ]:





# In[ ]:





# In[ ]:




