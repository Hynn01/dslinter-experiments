#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <div style="color:white;
#            display:fill;
#            border-radius:13px;
#            background-color:Aquamarine;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# Practice AutoML - mljar
# </h1>
# </div>

# <div style="color:white;
#            display:fill;
#            border-radius:13px;
#            background-color:Lavender;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# 1. Ready Package - install mljar
# </h1>
# </div>

# In[ ]:


## mljar 설치

get_ipython().system('pip install -q -U git+https://github.com/mljar/mljar-supervised.git@master')


# In[ ]:


## 데이터 로드 
train = pd.read_csv("/kaggle/input/titanic/train.csv")
train.head()


# <div style="color:white;
#            display:fill;
#            border-radius:13px;
#            background-color:Lavender;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# 2. Simple EDA by mljar
# </h1>
# </div>

# In[ ]:


train.info()


# In[ ]:


from supervised.preprocessing.eda import EDA

train = train
X = train[train.columns[2:]]
y = train["Survived"]

EDA.extensive_eda(X, y, save_path = './')


# I think Cabin has too many Null Value
# 
# Drop 'Cabin'

# In[ ]:


X = X.drop("Cabin",axis = 1)


# <div style="color:white;
#            display:fill;
#            border-radius:13px;
#            background-color:Lavender;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# 3. AutoML Operate - Explain
# </h1>
# </div>

# In[ ]:


from supervised import AutoML

a = AutoML(total_time_limit=60*20,
                model_time_limit = 120,
                mode = "Explain",
                eval_metric="accuracy",
                algorithms = ['Decision Tree','Xgboost', 'Linear', 'Random Forest', 'Extra Trees', 'LightGBM', 'CatBoost', 'Neural Network', 'Nearest Neighbors'],
                ml_task = 'binary_classification',
                train_ensemble=True,
                n_jobs = -1,
                random_state = 1004)
a.fit(X,y)


# In[ ]:


pd.set_option('display.max_rows', None)
a.get_leaderboard()


# In[ ]:


a.report()


# Best model : LightGBM
# 
# I think bad models : NearestNeighbors, Neural

# <div style="color:white;
#            display:fill;
#            border-radius:13px;
#            background-color:Lavender;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# 4. AutoML Operate - Perform
# </h1>
# </div>

# In[ ]:


b = AutoML(total_time_limit=60*10,
                model_time_limit = 60,
                mode = "Perform",
                eval_metric="accuracy",
                algorithms = ['Xgboost', 'Random Forest', 'Extra Trees', 'LightGBM', 'CatBoost'],
                ml_task = 'binary_classification',
                train_ensemble=True,
                n_jobs = -1,
                validation_strategy = { 
                    "validation_type": "kfold",
                    "k_folds": 5,
                    "shuffle": True,
                    "stratify": True,
                    "random_seed": 46
                },
                random_state = 1004)
b.fit(X,y)


# <div style="color:white;
#            display:fill;
#            border-radius:13px;
#            background-color:Lavender;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# 5. AutoML Operate - Compete
# </h1>
# </div>

# In[ ]:


c = AutoML(total_time_limit=60*10,
                model_time_limit = 60,
                mode = "Compete",
                eval_metric="accuracy",
                algorithms = ['Xgboost', 'LightGBM', 'CatBoost'],
                ml_task = 'binary_classification',
                train_ensemble=True,
                n_jobs = -1,
                random_state = 1004)
c.fit(X,y)


# <div style="color:white;
#            display:fill;
#            border-radius:13px;
#            background-color:Lavender;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# 6. AutoML Predict
# </h1>
# </div>

# In[ ]:


test = pd.read_csv("/kaggle/input/titanic/test.csv")
test = test.drop("Cabin",axis = 1)
test.head()


# In[ ]:


sub = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
sub.head()


# In[ ]:


pred_c = c.predict(test)
sub.loc[:,"Survived"] = pred_c
sub.head()


# In[ ]:


sub.to_csv("submission_c.csv",index=False)


# ## Optuna
# 
# Next todo,,,, Optuna Vs AutoML

# In[ ]:


# # !pip install optuna
# import optuna
# from optuna import Trial
# from optuna.samplers import TPESampler
# from sklearn.metrics import rmse
# from sklearn.model_selection import train_test_split

# def objective(trial: Trial) -> float:

#     params_lgb = {
#         "random_state": 42,
#         "verbosity": -1,
#         "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.1),
#         "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
#         "objective": "regression",
#         "metric": "rmse",
#         "max_depth": trial.suggest_int("max_depth", 1, 30),
#         "num_leaves": trial.suggest_int("num_leaves", 2, 256),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.6),
#         "subsample": trial.suggest_float("subsample", 0.3, 1.0),
#         "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
#         "min_child_samples": trial.suggest_int("min_child_samples", 2, 30),
#     }

#     model = LGBMRegressor(**params_lgb)
#     model.fit(
#         X_train,
#         y_train,
#         eval_set=[(X_train, y_train), (X_test, y_test)],
#         early_stopping_rounds=100,
#         verbose=False,
#     )

#     lgb_pred = model.predict(X_test)
#     eval_rmse = np.round(np.sqrt(mean_squared_error(y_test, lgb_pred)),4)
#     return eval_rmse

# sampler = TPESampler(seed=42)
# study = optuna.create_study(
#     study_name="lgbm_parameter_opt",
#     direction="minimize",
#     sampler=sampler,
# )

# study.optimize(objective, n_trials=500)
# print("Best Score:", study.best_value)
# print("Best trial:", study.best_trial.params)


# ## Reference
# 
# <a href="https://www.kaggle.com/code/kimchanyoung/titanic-easiest-automl">@CHANYOUNG KIM</a>
