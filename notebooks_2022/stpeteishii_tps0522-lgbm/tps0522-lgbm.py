#!/usr/bin/env python
# coding: utf-8

# # TPS0522 LGBM

# In[ ]:


import lightgbm as lgb
import numpy as np
import pandas as pd
import random
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


# In[ ]:


train = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv")
test = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")


# In[ ]:


display(train.head())


# In[ ]:


data=pd.concat([train,test],axis=0)


# In[ ]:


from sklearn.preprocessing import LabelEncoder

def labelencoder(df):
    for c in df.columns:
        if df[c].dtype=='object': 
            df[c] = df[c].fillna('N')
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(df[c].values)
    return df


# In[ ]:


data=labelencoder(data)
train=data[0:len(train)]
test=data[len(train):]


# In[ ]:


target = train['target']
data = train.drop(['target','id'],axis=1)
test = test.drop('id',axis=1)


# In[ ]:


columns=data.columns.to_list()
print(columns)


# In[ ]:


def objective(trial,data=data,target=target):
    
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2,random_state=42)
    param =   {
        'num_leaves': trial.suggest_int('num_leaves', 280, 300),
        'objective': trial.suggest_categorical('objective',['regression','rmse']),  
        'max_depth': trial.suggest_int('max_depth',16, 20),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.2, 0.22),
        "boosting": "gbdt",
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-6, 1e-3),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-3, 0.003),
        "bagging_freq": 5,
        "bagging_fraction": trial.suggest_uniform('bagging_fraction', 0.7, 1.0),
        "feature_fraction": trial.suggest_uniform('feature_fraction', 0.9, 1.0),
        "verbosity": -1,
    }
    model = lgb.LGBMClassifier(**param)      
    model.fit(train_x,train_y,eval_set=[(test_x,test_y)],early_stopping_rounds=100,verbose=False)
    preds = model.predict(test_x)
    rmse = mean_squared_error(test_y, preds,squared=False)
    
    return rmse


# In[ ]:


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=300)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)


# In[ ]:


study.trials_dataframe()


# In[ ]:


# shows the scores from all trials
optuna.visualization.plot_optimization_history(study)


# In[ ]:


# interactively visualizes the hyperparameters and scores
optuna.visualization.plot_parallel_coordinate(study)


# In[ ]:


# shows the evolution of the search
optuna.visualization.plot_slice(study)


# In[ ]:


# parameter interactions on an interactive chart.
optuna.visualization.plot_contour(study, params=['num_leaves','learning_rate'])


# In[ ]:


# Visualize parameter importances.
optuna.visualization.plot_param_importances(study)


# In[ ]:


# Visualize empirical distribution function
optuna.visualization.plot_edf(study)


# In[ ]:


Best_trial=study.best_trial.params
print(Best_trial)


# In[ ]:


sample = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
display(sample[0:3])


# In[ ]:


preds = np.zeros((sample.shape[0]))
kf = KFold(n_splits=5,random_state=48,shuffle=True)
for trn_idx, test_idx in kf.split(train[columns],target):
    X_tr,X_val=train[columns].iloc[trn_idx],train[columns].iloc[test_idx]
    y_tr,y_val=target.iloc[trn_idx],target.iloc[test_idx]
    model = lgb.LGBMClassifier(**Best_trial)
    model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],early_stopping_rounds=100,verbose=False)
    preds+=model.predict(test[columns])/kf.n_splits   ###### predict_proba
    rmse=mean_squared_error(y_val, model.predict(X_val),squared=False)
    print(rmse)


# In[ ]:


subm = sample
subm['target'] = np.where(preds<0.5,0,1).astype(int)
subm.to_csv('submission.csv',index=False)
subm


# In[ ]:




