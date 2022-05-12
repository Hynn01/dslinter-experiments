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


# <div class="text_cell_render border-box-sizing rendered_html">
# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:skyblue;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# [Titanic]Easiest AutoML
# </h1>
# </div>
# </div>

# <img src = "https://storage.googleapis.com/kaggle-competitions/kaggle/3136/logos/header.png" alt="" width = '2000'>

# <div class="text_cell_render border-box-sizing rendered_html">
# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:skyblue;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# Install MLJAR AutoML
# </h1>
# </div>
# </div>

# <img src="https://raw.githubusercontent.com/mljar/mljar-examples/master/media/AutoML_overview_mljar_v3.svg" alt="" width = '700'>
# 
# **Documentation: https://supervised.mljar.com/**
# 
# **Source Code: https://github.com/mljar/mljar-supervised**
# 
# The **mljar-supervised** is an Automated Machine Learning Python package that works with tabular data.  
# It is designed to save time for a data scientist. It abstracts the common way to preprocess the data,   
# construct the machine learning models, and perform hyper-parameters tuning to find the best model
# 
# source : https://github.com/mljar/mljar-supervised

# In[ ]:


get_ipython().system('pip install -q -U git+https://github.com/mljar/mljar-supervised.git@master')


# <div class="text_cell_render border-box-sizing rendered_html">
# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:skyblue;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# Load Dataset
# </h1>
# </div>
# </div>

# In[ ]:


pd_train = pd.read_csv('../input/titanic/train.csv')
pd_train.info()


# Looks like there is a lot of missing data.  
# But we will leave this missing data as it is. Because our AutoML will takes care of it!

# <div class="text_cell_render border-box-sizing rendered_html">
# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:skyblue;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# Easiest EDA using MLJAR
# </h1>
# </div>
# </div>

# In[ ]:


from supervised.preprocessing.eda import EDA

train = pd_train
X = train[train.columns[2:]]
y = train["Survived"]

EDA.extensive_eda(X, y, save_path = './')


# <div class="text_cell_render border-box-sizing rendered_html">
# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:skyblue;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# Training using MLJAL AutoML
# </h1>
# </div>
# </div>

# We will find our best model automatically in 30 minutes.  
# 'Decision Tree', 'Linear', 'Random Forest', 'Extra Trees', 'LightGBM', 'Xgboost', 'CatBoost', 'Neural Network', 'Nearest Neighbors' will be used to find our best model!

# In[ ]:


import pandas as pd
import numpy as np
from supervised import AutoML

automl = AutoML(total_time_limit=60*30,
                model_time_limit = 300,
                mode = "Compete",
                eval_metric="accuracy",
                algorithms = ['Decision Tree', 'Linear', 'Random Forest', 'Extra Trees', 'LightGBM', 'CatBoost', 'Neural Network', 'Nearest Neighbors'],
                ml_task = 'binary_classification',
                train_ensemble=True,
                n_jobs = -1,
                random_state = 1004
               )

automl.fit(X, y)


# <div class="text_cell_render border-box-sizing rendered_html">
# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:skyblue;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# <h1 style="text-align: center;
#            padding: 10px;
#               color:white">
# Training Result
# </h1>
# </div>
# </div>

# In[ ]:


pd.set_option('display.max_rows', None)
automl.get_leaderboard()


# In[ ]:


automl.report()

