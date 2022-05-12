#!/usr/bin/env python
# coding: utf-8

# ## Iterative Imputation
# * I wrote manuel code for iterative imputation, but you can use scikitlearn api if u want (https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html)
# * If you impute only Torque using this notebook, you will have 24 RMSE score on public LB. After impute all features, the score will be 23 or something. For more robust results, i am using ensemble of Catboost and Lightgbm with cross validation.

# In[ ]:


#As usual
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import warnings
import itertools
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

#Modelleme
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit,train_test_split
from catboost import CatBoostRegressor
import lightgbm as lgbm
import xgboost as xg
from sklearn import preprocessing

import datetime


# In[ ]:


power=pd.read_csv(r"../input/enerjisa-uretim-hackathon/power.csv")
features=pd.read_csv(r"../input/enerjisa-uretim-hackathon/features.csv")
sample_submission=pd.read_csv(r"../input/enerjisa-uretim-hackathon/sample_submission.csv")


# In[ ]:


power.Timestamp=pd.to_datetime(power.Timestamp)
sample_submission.Timestamp=pd.to_datetime(sample_submission.Timestamp)
features.Timestamp=pd.to_datetime(features.Timestamp)
sample_submission['Power(kW)']=np.nan


# In[ ]:


df=pd.concat([power,sample_submission],axis=0)
df.Timestamp=pd.to_datetime(df.Timestamp)
df=pd.merge(df,features,on='Timestamp',how='left')
df=df.sort_values('Timestamp')
df=df.replace(99999.0,np.nan)
df.set_index('Timestamp',inplace=True)


# In[ ]:


#short by correlation
corr_order=df.corr()['Power(kW)'].abs().reset_index().sort_values('Power(kW)')['index'].to_list()[:-1]


# In[ ]:


corr_order


# In[ ]:


#we don't need 'power' for imputation
df=df.drop('Power(kW)',axis=1)


# In[ ]:


#4 random fold for robust imputation
kfold=KFold(n_splits=4, random_state=42, shuffle=True) 


# In[ ]:


df.isnull().sum()


# In[ ]:


start_time = time.time()
df_imputed=df.copy()
for i in corr_order:
    print('Trying:',i)
    target=i 
    forecast=df_imputed[df_imputed[target].isnull()] #to predict
    historical=df_imputed[~df_imputed[target].isnull()] #train
    forecast.drop([target],axis=1,inplace=True) 
    y=historical[target]
    X=historical.drop([target],axis=1)
    
    unseen_preds = []
    fold = 1
    print('catboost')
    for train_index,test_index in kfold.split(X,y):
        X_train,X_val = X.iloc[train_index],X.iloc[test_index]
        y_train,y_val = y.iloc[train_index],y.iloc[test_index]
        cat = CatBoostRegressor(#**params,
                            iterations = 2500,loss_function='RMSE', eval_metric='RMSE',allow_writing_files=False)
        cat.fit(X_train,y_train,eval_set=[(X_val,y_val)],early_stopping_rounds=200,verbose=2500)
        forecast_pred=cat.predict(forecast)
        unseen_preds.append(forecast_pred)
        fold+=1
    
    print("--- %s seconds ---" % (time.time() - start_time))
    fold = 1
    print('Lightgbm')
    for train_index,test_index in kfold.split(X,y):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
        dtrain = lgbm.Dataset(X_train, y_train)
        dvalid = lgbm.Dataset(X_val, y_val)
        params = {"objective": "regression","metric": "rmse","verbosity": -1,"boosting_type": "gbdt","feature_fraction":0.5,"num_leaves": 250,"lambda_l1":4,"lambda_l2":2,
                  "learning_rate":0.01,'min_child_samples': 35,"bagging_fraction":0.75,"bagging_freq":1,"seed":0
             }           
        model = lgbm.train(params,
                           dtrain,
                           valid_sets=[dtrain, dvalid],
                           early_stopping_rounds=200,
                           verbose_eval=2100,
                           num_boost_round=2200
                           
                    )
        forecast_pred=model.predict(forecast)
        unseen_preds.append(forecast_pred)
        fold+=1
        
    print("--- %s seconds ---" % (time.time() - start_time))
    
    first = pd.DataFrame(np.mean(unseen_preds,axis=0)) #ensemble
    forecasted = pd.DataFrame(first).rename(columns={0:target}).set_index(forecast.index)
    will_replace = pd.concat([pd.DataFrame(y),forecasted],axis=0)
    will_replace = will_replace.sort_index()
    
    df_imputed.drop(target,axis=1,inplace=True)
    df_imputed[target]=will_replace[target]
    
print('----------DONE----------')


# In[ ]:


df_imputed.reset_index(inplace=True)


# In[ ]:


df_imputed.to_csv('df_imputed_cat_lgb.csv',index=False)

