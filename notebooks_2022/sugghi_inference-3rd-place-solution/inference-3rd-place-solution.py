#!/usr/bin/env python
# coding: utf-8

# # [inference] 3rd place solution
# 
# I will post a description of the model in discussion. Training is done in another notebook.
# * Training: https://www.kaggle.com/sugghi/training-3rd-place-solution/
# 
# I referred to various notebooks when coding. In particular, the following notebook was used directly.
# * Submitting Lagged Features via API：
# https://www.kaggle.com/tomforbes/gresearch-submitting-lagged-features-via-api
# 
# In addition, local api published by @jagofc helped me a lot in coding. (Not included in the final submission so it is commented out in this notebook.)
# * local api：https://www.kaggle.com/code/jagofc/local-api/
# 
# As you can see from my code, I am a novice in machine learning and python.
# If you see anything  to improve on or any mistakes, I'd be very happy to hear about them!

# In[ ]:


import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import gresearch_crypto
import time
import datetime

import pickle
import gc

from tqdm import tqdm

n_fold = 7

# List for ensemble but not used due to inference time...
input_list = [
    '../input/training-3rd-place-solution'
]
n_fold_list = [7]

use_supple_for_train = False

TRAIN_CSV = '/kaggle/input/g-research-crypto-forecasting/train.csv'
SUPPLE_TRAIN_CSV = '/kaggle/input/g-research-crypto-forecasting/supplemental_train.csv'
ASSET_DETAILS_CSV = '/kaggle/input/g-research-crypto-forecasting/asset_details.csv'

pd.set_option('display.max_rows', 6)
pd.set_option('display.max_columns', 350)


# In[ ]:


models = [ [  [] for split in range(sum(n_fold_list))     ] for asset_id in range(14)]

for asset_id in range(14):
    for input_number in range(len(input_list)):
        for split in range(n_fold_list[input_number]):
            models[asset_id][sum(n_fold_list[:input_number])+split] = pickle.load(open(f'{input_list[input_number]}/trained_model_id{asset_id}_fold{split}.pkl', 'rb'))


# In[ ]:


lags = [60,300,900]


# In[ ]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
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
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# In[ ]:


df_asset_details = pd.read_csv(ASSET_DETAILS_CSV).sort_values("Asset_ID")
# df_asset_details


# In[ ]:


def get_features(df, train=True):   
    if train == True:
        totimestamp = lambda s: np.int32(time.mktime(datetime.datetime.strptime(s, "%d/%m/%Y").timetuple()))
        valid_window = [totimestamp("12/03/2021")]
#         valid_window = [totimestamp("15/08/2021")]  #検証用
        df['train_flg'] = np.where(df['timestamp']>=valid_window[0], 0,1)

        supple_start_window = [totimestamp("22/09/2021")]
        if use_supple_for_train:
            df['train_flg'] = np.where(df['timestamp']>=supple_start_window[0], 1 ,df['train_flg']  )

   
    for id in range(14):    
        for lag in lags:
            df[f'log_close/mean_{lag}_id{id}'] = np.log( np.array(df[f'Close_{id}']) /  np.roll(np.append(np.convolve( np.array(df[f'Close_{id}']), np.ones(lag)/lag, mode="valid"), np.ones(lag-1)), lag-1)  )
            df[f'log_return_{lag}_id{id}']     = np.log( np.array(df[f'Close_{id}']) /  np.roll(np.array(df[f'Close_{id}']), lag)  )
    for lag in lags:
        df[f'mean_close/mean_{lag}'] =  np.mean(df.iloc[:,df.columns.str.startswith(f'log_close/mean_{lag}_id')], axis=1)
        df[f'mean_log_returns_{lag}'] = np.mean(df.iloc[:,df.columns.str.startswith(f'log_return_{lag}_id')] ,    axis=1)
        for id in range(14):
            df[f'log_close/mean_{lag}-mean_close/mean_{lag}_id{id}'] = np.array( df[f'log_close/mean_{lag}_id{id}']) - np.array( df[f'mean_close/mean_{lag}']  )
            df[f'log_return_{lag}-mean_log_returns_{lag}_id{id}']    = np.array( df[f'log_return_{lag}_id{id}'])     - np.array( df[f'mean_log_returns_{lag}'] )

    if train == True:
        for id in range(14):
            df = df.drop([f'Close_{id}'], axis=1)
        oldest_use_window = [totimestamp("12/01/2019")]
        df = df[  df['timestamp'] >= oldest_use_window[0]   ]

    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', '# generate feature column names\ndf_train = pd.read_csv(SUPPLE_TRAIN_CSV, usecols=[\'timestamp\',\'Asset_ID\', \'Close\', \'Target\'], nrows=max(lags)*20)\nprint(len(df_train[\'Asset_ID\'].unique()))\ndf_train = reduce_mem_usage(df_train)\n# df_train\n\ntrain_merged = pd.DataFrame()\ntrain_merged[df_train.columns] = 0\nfor id in tqdm( range(14) ):\n    train_merged = train_merged.merge(df_train.loc[df_train["Asset_ID"] == id, [\'timestamp\', \'Close\',\'Target\']].copy(), on="timestamp", how=\'outer\',suffixes=[\'\', "_"+str(id)])\n        \ntrain_merged = train_merged.drop(df_train.columns.drop("timestamp"), axis=1)\ndisplay(train_merged.head())\n\nnot_use_features_train = [\'timestamp\', \'train_flg\']\nfor id in range(14):\n    not_use_features_train.append(f\'Target_{id}\')\n\nfeatures = get_features(train_merged, train=True).columns \nfeatures = features.drop(not_use_features_train)\nfeatures = list(features)\n# display(features)  \nlen(features)')


# In[ ]:


del train_merged
del df_train
gc.collect()


# In[ ]:


# define max_lookback - an integer > (greater than) the furthest look back in your lagged features
keep_hist = max(lags)


# In[ ]:


def merge_for_infer(df):
    df_merged = pd.DataFrame()
    df_merged[['timestamp', 'Asset_ID', 'Close']] = 0
    for id in range(14):
        df_merged = df_merged.merge(df.loc[df["Asset_ID"] == id, ['timestamp', 'Close']].copy(), on="timestamp", how='outer',suffixes=['', "_"+str(id)])
 
    df_merged = df_merged.drop(['Asset_ID', 'Close'], axis=1)
#     df_merged = df_merged.sort_values('timestamp', ascending=True)
    return df_merged


# In[ ]:


# create dataframe to store data from the api to create lagged features
history = pd.read_csv(SUPPLE_TRAIN_CSV, usecols=['timestamp','Asset_ID', 'Close'])
history = history.tail(keep_hist*14)

history = reduce_mem_usage(history)
history_merged = merge_for_infer(history)
history_merged = history_merged.tail(keep_hist)
history_merged


# In[ ]:


one_line = history_merged.tail(1).copy()

def merge_infer_2(df, df_one_line):
    for asset_id, close in zip(   df['Asset_ID'].values,  df['Close'].values   ): 
        df_one_line[f'Close_{asset_id}'] = close
    return df_one_line


# In[ ]:


env = gresearch_crypto.make_env()
iter_test = env.iter_test()


# In[ ]:


start = time.time()

for i, (df_test, df_pred) in enumerate(iter_test):
    df_test_merged = merge_infer_2(df_test, one_line)
    history_merged = pd.concat([history_merged, df_test_merged])
    x_test = get_features(history_merged, train=False)
    x_calc = x_test.iloc[-1]
    for j , (asset_id,row_id) in enumerate(  zip(   df_test['Asset_ID'].values,  df_test['row_id'].values   )   ): 
        y_pred_list = []
        try:
            for split in range(n_fold):
                y_pred_list.append(models[ asset_id ][split].predict(x_calc[features]))
            y_pred = np.median(y_pred_list)
        except Exception:
            y_pred = 0
        df_pred.loc[  df_pred['row_id'] == row_id ,  'Target'  ] = y_pred

    history_merged = history_merged.tail(keep_hist)
    
    # Send submissions
    env.predict(df_pred)
stop = time.time()
print(stop-start)


# # local test

# In[ ]:


# # Thanks to @jagofc
# # https://www.kaggle.com/code/jagofc/local-api/

# import local_api as la
# train_df = la.read_csv_slice('../input/g-research-crypto-forecasting/train.csv')
# # example_window = (la.datestring_to_timestamp("2021-07-02T00:00"),
# #                   la.datestring_to_timestamp("2021-08-17T05:00"))
# # la.LB_WINDOW
# # api = la.API(train_df, use_window=example_window)
# api = la.API(train_df, use_window=la.LB_WINDOW)

# start_time = time.time()

# for i, (df_test, df_pred) in enumerate(tqdm(api)):
#     df_test_merged = merge_infer_2(df_test, one_line)    
#     history_merged = pd.concat([history_merged, df_test_merged])    
#     x_test = get_features(history_merged, False)
#     x_calc = x_test.iloc[-1]
    
#     for j , (asset_id,row_id) in enumerate(  zip(   df_test['Asset_ID'].values,  df_test['row_id'].values   )   ): 
#         y_pred = 0       
#         y_pred_list = []
#         for split in range(n_fold):
#             y_pred_list.append(models[ asset_id ][split].predict(x_calc[features]))
#         y_pred = np.median(y_pred_list)     
#         df_pred.loc[  df_pred['row_id'] == row_id ,  'Target'  ] = y_pred

#     history_merged = history_merged.tail(keep_hist)
    
#     # Send submissions
#     api.predict(df_pred)    
# stop = time.time()
# print(stop-start)

# finish_time = time.time()

# total_time = finish_time - start_time
# iter_speed = api.init_num_times/total_time

# print(f"    num_fold = {n_fold}")
# print(f"Iterations/s = {round(iter_speed, 3)}")
# print(f"s/Iterations = {round(1/iter_speed, 3)}")
# test_iters = 60 * 24 * 100
# print(f"Expected number of iterations in test set is approx. {test_iters}",
#       f"which will take {round(test_iters / (iter_speed * 3600), 2)} hours",
#       "using this API emulator while making dummy predictions.")

# df, score = api.score()
# print(f"Your LB score is {round(score, 5)}")

