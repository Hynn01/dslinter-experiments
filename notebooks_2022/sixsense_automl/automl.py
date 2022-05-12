#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -q -U git+https://github.com/mljar/mljar-supervised.git@master')


# In[ ]:


import pandas as pd

pd_train = pd.read_csv('../input/titanic/train.csv')
pd_train.info()


# In[ ]:


from supervised.preprocessing.eda import EDA

train = pd_train
X = train[train.columns[2:]]
y = train["Survived"]

EDA.extensive_eda(X, y, save_path = './')


# In[ ]:


import numpy as np
from supervised import AutoML

automl = AutoML(total_time_limit=60*10,
                model_time_limit = 300,
                mode = "Compete",
                eval_metric="accuracy",
                algorithms = ['Random Forest', 'Extra Trees', 'Neural Network', 'Nearest Neighbors'],
                ml_task = 'binary_classification',
                train_ensemble=True,
                n_jobs = -1,
                random_state = 42
               )

automl.fit(X, y)


# In[ ]:


pd.set_option('display.max_rows', None)
automl.get_leaderboard()


# In[ ]:


automl.report()


# In[ ]:


import numpy as np
from supervised import AutoML

automl2 = AutoML(mode = "Optuna",
                eval_metric="accuracy",
                algorithms = ['Random Forest', 'Extra Trees', 'Neural Network', 'Nearest Neighbors'],
                optuna_time_budget=60,   # tune each algorithm for 30 minutes
                total_time_limit=600,  # total time limit, set large enough to have time to compute all steps
                )


automl2.fit(X, y)


# In[ ]:


automl2.report()

