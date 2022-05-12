#!/usr/bin/env python
# coding: utf-8

# ##### Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning. It features an imperative, define-by-run style user API. Thanks to our define-by-run API, the code written with Optuna enjoys high modularity, and the user of Optuna can dynamically construct the search spaces for the hyperparameters.

# # Basic Concepts

# In[ ]:


#!pip install optuna


# In[ ]:


import optuna

# Define an objective function to be minimized.
def objective(trial):

    # Invoke suggest methods of a Trial object to generate hyperparameters.
    regressor_name = trial.suggest_categorical('regressor', ['SVR', 'RandomForest'])
    if regressor_name == 'SVR':
        svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log=True)
        regressor_obj = sklearn.svm.SVR(C=svr_c)
    else:
        rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32)
        regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=rf_max_depth)

    X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, random_state=0)

    regressor_obj.fit(X_train, y_train)
    y_pred = regressor_obj.predict(X_val)

    error = sklearn.metrics.mean_squared_error(y_val, y_pred)

    return error  # An objective value linked with the Trial object.

#study = optuna.create_study()  # Create a new study.
#study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.


# # Application

# In[ ]:


from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split


# ## Import Data

# In[ ]:


df = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')


# In[ ]:


#Use only 30,000 for quick experiments
df = df[:30000]
df.head()


# ## EDA

# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.dtypes


# ## Split Data

# In[ ]:


df['f_27'] = df['f_27'].astype('category')  ##Category column processing for LGBM

X = df.drop(['id'],axis=1)
y = df['target']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0 ,shuffle=True)


# ## Optuna Tuning

# In[ ]:


def objectives(trial):
    
    ### Set a range of parameters
    params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
        
            'num_leaves': trial.suggest_int('num_leaves', 100, 5000),
            'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
            'max_bin': trial.suggest_int('max_bin', 2, 100),
            'learning_rate': trial.suggest_uniform('learning_rate',0, 1),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
            'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
            'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.2, 1.0),
    }
    
    model = LGBMClassifier(**params)
    model.fit(X_train,y_train)
    score = model.score(X_train,y_train)
    return score


# In[ ]:


## optuna pactice
studyLGBM = optuna.create_study(direction='maximize',sampler=optuna.samplers.RandomSampler(seed=0))
studyLGBM.optimize(objectives, n_trials=20)  ## Set the number to n_trials

trial = studyLGBM.best_trial
params_best = dict(trial.params.items())
params_best['random_seed'] = 0
    
model_o = LGBMClassifier(**params_best)#


# In[ ]:


print('study.best_params:', studyLGBM.best_trial.value)
print('Number of finished trials:', len(studyLGBM.trials))
print('Best trial:', studyLGBM.best_trial.params)
print('study.best_params:', studyLGBM.best_params)


# In[ ]:


# Visualize parameter importance



#optuna.visualization.plot_param_importances(studyLGBM)

