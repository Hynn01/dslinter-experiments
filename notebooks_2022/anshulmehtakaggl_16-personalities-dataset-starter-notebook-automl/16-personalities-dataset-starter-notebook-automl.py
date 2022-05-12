#!/usr/bin/env python
# coding: utf-8

# # <center> The 16 Personalities Dataset: By Anshul </center>
# 
# 
# #### Link to the Dataset:
# #### [Dataset](https://www.kaggle.com/datasets/anshulmehtakaggl/60k-responses-of-16-personalities-test-mbt)
# <img src='https://i.pinimg.com/originals/a8/9f/5f/a89f5ff47344c8329e54706767eac545.jpg' >
# 
# 

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


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
train_data=pd.read_csv('../input/60k-responses-of-16-personalities-test-mbt/16P.csv',encoding='cp1252')
train_data=train_data.drop(columns={'Response Id'})
train_data.head()


# In[ ]:


pip install pycaret


# In[ ]:


from sklearn.preprocessing import StandardScaler
X=train_data.drop(columns="Personality")
y=train_data["Personality"]


# In[ ]:


# Standardizing the features
X = StandardScaler().fit_transform(X)
from sklearn.decomposition import PCA
pca = PCA(n_components=21)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc1', 'pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11','pc12','pc13','pc14','pc15','pc16','pc17','pc18','pc19','pc20','pc21'])
finalDf = pd.concat([principalDf, train_data[['Personality']]], axis = 1)


# In[ ]:


finalDf.head()


# In[ ]:


from pycaret.classification import *
#df = pd.read_csv("../input/60k-responses-of-16-personalities-test-mbt/16_Personalities_v1.csv")

clf = setup(
    data=finalDf,
    target="Personality",
    remove_multicollinearity=True,
    remove_outliers=True,
    remove_perfect_collinearity=True,
    fix_imbalance=True,
    log_experiment=True,
    normalize=True,
    transformation=True,
    verbose=True,
    silent=True,
    feature_interaction=True,
    feature_selection=True,
    pca=True
)
get_ipython().run_line_magic('time', 'best_model = compare_models()')


# In[ ]:


# import optuna
# import optuna.integration.lightgbm as lgb
# import pandas as pd
# from lightgbm import early_stopping, log_evaluation
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import train_test_split


# def objective(trial: optuna.Trial):
# #     df = pd.read_csv("../input/60k-responses-of-16-personalities-test-mbt/16_PERSONALITIES.csv",encoding='cp1252')

#     train_x, test_x, train_y, test_y = train_test_split(
#         finalDf.drop(columns="Personality"), finalDf["Personality"], test_size=0.2
#     )

#     params = {
#         "metric": "auc",
#         "objective": "binary",
#         "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
#         "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
#         "n_estimators": trial.suggest_int("n_estimators", 1, 100),
#         "num_leaves": trial.suggest_int("num_leaves", 2, 256),
#         "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
#         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
#         "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
#     }

#     dtrain = lgb.Dataset(train_x, label=train_y)
#     dval = lgb.Dataset(test_x, label=test_y)

#     model = lgb.train(
#         params,
#         dtrain,
#         valid_sets=[dtrain, dval],
#         callbacks=[early_stopping(100), log_evaluation(100)],
#     )

#     prediction = model.predict(test_x, num_iteration=model.best_iteration)
#     return roc_auc_score(test_y, prediction)


# study = optuna.create_study()
# study.optimize(objective, n_jobs=-1, n_trials=100)
# print(study.best_params)

