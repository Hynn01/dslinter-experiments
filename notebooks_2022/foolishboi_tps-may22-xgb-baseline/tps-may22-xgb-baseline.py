#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv', index_col = 0)
test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv', index_col = 0)
for df in [train, test]:
    for i in range(10):
        df[f'ch{i}'] = df.f_27.str.get(i).apply(ord) - ord('A')
    # Next feature is from https://www.kaggle.com/code/cabaxiom/tps-may-22-eda-lgbm-model
    df["unique_characters"] = df.f_27.apply(lambda s: len(set(s)))
features = [f for f in test.columns if f != 'id' and f != 'f_27']
test[features].head(2)


# In[ ]:


def reduce_mem_usage(df, verbose=True):
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
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


reduce_mem_usage(train)
reduce_mem_usage(test)


# In[ ]:


train


# In[ ]:


params = {'tree_method':'gpu_hist',
          'n_estimators': 10000,
          'lambda': 0.001583005792255653, 
          'alpha': 9.826088526413647, 
          'colsample_bytree': 0.5, 
          'subsample': 0.5, 
          'learning_rate': 0.02, 
          'max_depth': 11, 
          'random_state': 2020, 
          'min_child_weight': 135}


# In[ ]:


preds = np.zeros(test.shape[0])
kf = KFold(n_splits=5,random_state=48,shuffle=True)
auc=[]  # list contains roc_auc score for each fold
n=0
for trn_idx, test_idx in kf.split(train[features],train['target']):
    X_tr,X_val=train[features].iloc[trn_idx],train[features].iloc[test_idx]
    y_tr,y_val=train['target'].iloc[trn_idx],train['target'].iloc[test_idx]
    model = XGBRegressor(**params)
    model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],early_stopping_rounds=100,verbose=False)
    preds+=model.predict(test[features])/kf.n_splits
    auc.append(roc_auc_score(y_val, model.predict(X_val)))
    print(f"fold: {n+1} ==> rmse: {auc[n]}")
    n+=1


# In[ ]:


sub = pd.read_csv('../input/tabular-playground-series-may-2022/sample_submission.csv')
sub['target'] = preds
sub.to_csv('sub_tpsmay22_xgbregressor_v1.csv', index = False)


# In[ ]:





# In[ ]:





# In[ ]:




