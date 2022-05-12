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
#            border-radius:5px;
#            background-color:#D0F0C0;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="text-align:center;" "color:white;">
# <span style="font-size:30px;"> 
# <b> Predicting Titanic Data Using Auto ML </b>
# </div>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for plotting
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# mljar install
get_ipython().system('pip install -q -U git+https://github.com/mljar/mljar-supervised.git@master')


# In[ ]:


# import data
train_data = pd.read_csv("../input/titanic/train.csv") 
test_data = pd.read_csv("../input/titanic/test.csv")


# In[ ]:


print(train_data.shape)
print(train_data.info())


# In[ ]:


print(test_data.shape)
print(test_data.info())


# In[ ]:


# train data 
train_data.describe().T.style.bar(subset=['mean'], color ='#205ff2')                            .background_gradient(subset=['std'], cmap='coolwarm')                            .background_gradient(subset=['50%'], cmap='coolwarm')


# In[ ]:


# test data 
test_data.describe().T.style.bar(subset=['mean'], color ='#205ff2')                            .background_gradient(subset=['std'], cmap='coolwarm')                            .background_gradient(subset=['50%'], cmap='coolwarm')


# In[ ]:


# set up train x data and y target
X_train = train_data[train_data.columns[2:]]
y_train = train_data["Survived"]


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#D0F0C0;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="text-align:center;" "color:white;">
# <span style="font-size:25px;"> 
# <b> Perform Mode </b>
# </div>

# In[ ]:


#Perfrom mode

# MLJAR-supervised
from supervised.automl import AutoML 

# Create the model
mljar = AutoML(mode = "Perform", # 모드 종류: Exlplain, Compete, Optuna
               total_time_limit = 1800, # 30 minutes
               model_time_limit = 300, # 5 minutes
               ml_task = "binary_classification",  
               algorithms = ['Random Forest', 'LightGBM', 'Xgboost', 'CatBoost'
                            ,'Decision Tree', 'Extra Trees', 'Neural Network'], #7 models selected
               eval_metric = "accuracy",               
               train_ensemble = True,
               features_selection = True,
               n_jobs = -1,
               random_state = 1234 #default= 1234
              )

# Fit the training data
mljar.fit(X_train, y_train)


# In[ ]:


# View Results
pd.set_option('display.max_rows', None)
mljar.get_leaderboard()


# In[ ]:


mljar.report()


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#D0F0C0;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="text-align:center;" "color:white;">
# <span style="font-size:25px;"> 
# <b> Compete Mode </b>
# </div>

# In[ ]:


#Compete mode

# MLJAR-supervised
from supervised.automl import AutoML 

# Create the model
mljar_2 = AutoML(mode = "Perform", # 모드 종류: Exlplain, Compete, Optuna
               total_time_limit = 1800, # 30 minutes
               model_time_limit = 300, # 5 minutes
               ml_task = "binary_classification",  
               algorithms = ['Random Forest', 'LightGBM', 'Xgboost', 'CatBoost'
                            ,'Decision Tree', 'Extra Trees', 'Neural Network'], #7 models selected
               eval_metric = "accuracy",               
               train_ensemble = True,
               features_selection = True,
               n_jobs = -1,
               random_state = 1234 #default= 1234
              )

# Fit the training data
mljar_2.fit(X_train, y_train)


# In[ ]:


# View Results
pd.set_option('display.max_rows', None)
mljar_2.get_leaderboard()


# In[ ]:


mljar_2.report()


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#D0F0C0;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="text-align:center;" "color:white;">
# <span style="font-size:25px;"> 
# <b> References </b>
# </div>

# 1. https://supervised.mljar.com/
# 2. https://www.analyticsvidhya.com/blog/2021/09/10-automated-machine-learning-for-supervised-learning-part-2
# 3. https://supervised.mljar.com/api/
# 4. https://www.kaggle.com/code/kimchanyoung/titanic-easiest-automl
# 
# Especially, thank you Chanyoung Kim for guiding auto ML!

# In[ ]:




