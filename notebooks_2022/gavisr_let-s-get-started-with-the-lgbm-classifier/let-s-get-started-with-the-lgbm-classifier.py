#!/usr/bin/env python
# coding: utf-8

# # <center> **Tabular Playground Series - May 2022** </center>

# # Table of Contents
# <a id="toc"></a>
# - [1. Libraries](#1)
# - [2. Load Datasets](#2)
# - [3. Data Exploration](#3)
# - [4. EDA](#4)
# - [5. Modelling](#5)    
# - [6. Submit To Kaggle](#6)
# 

# <p id="1"></p>
# 
# # <span style="font-family: cursive; font-size: 30px; font-style: normal; font-weight: bold; letter-spacing: 0.5px; color:#543348">Libraries  📚</span>
#  

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split

import lightgbm as lgb

import time
import warnings
warnings.filterwarnings('ignore')


# <a href="#toc" role="button" aria-pressed="true" style="color:#543348;"> Click here to go back to: Table of Contents </a>

# <p id="2"></p>
# # <span style="font-family: cursive; font-size: 30px; font-style: normal; font-weight: bold; letter-spacing: 0.5px; color:#543348"> Load Datasets 🗃️</span>

# In[ ]:


train = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv")
test = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")
submission = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")


# <a id="3"></a>
# # <span style="font-family: cursive; font-size: 30px; font-style: normal; font-weight: bold; letter-spacing: 0.5px; color:#543348">Data Exploration 🔎🛠</span>

# ### <span style="color:#543348;"> Below are the first 5 rows of train dataset:</span>

# In[ ]:


train.head(5)


# ### <span style="color:#543348;"> Below are the first 5 rows of test dataset: </span>

# In[ ]:


test.head(5)


# In[ ]:


print(f'Train data set: {train.shape[0]} rows and {train.shape[1]} columns')
print(f'Test data set: {test.shape[0]} rows and {test.shape[1]} columns') 


# ### <span style="color:#543348;"> Below are the number of missing values in the train and test datasets: </span>

# In[ ]:


print(f'Missing values in the train dataset: {train.isna().sum().sum()}')
print(f'Missing values in the test dataset: {test.isna().sum().sum()}')


# ### <span style="color:#543348;"> The number of unique values for each variable in the train dataset: </span> 

# In[ ]:


train.nunique().sort_values(ascending = True)


# ### <span style="color:#543348;"> Overview of the the train dataset including number of columns, column labels, data types, memory usage, range index, and the number of cells in each column (non-null values): </span> 

# In[ ]:


train.info()


# ### <span style="color:#543348;"> Basic statistics of training data for each variables which contain information on count, mean, standard deviation, minimum, 1st quartile, median, 3rd quartile and maximum: </span>

# In[ ]:


train.describe().T


# <a href="#toc" role="button" aria-pressed="true" style="color:#543348;" > Click here to go back to: Table of Contents </a>

# <a id="4"></a>
# # <span style="font-family: cursive; font-size: 30px; font-style: normal; font-weight: bold; letter-spacing: 0.5px; color:#543348">  EDA 👩🏽‍💻</span>

# ### <span style="color:#543348;"> Training features: </span>

# In[ ]:


TARGET = 'target'
Features = [col for col in train.columns if col != TARGET]
print('Training data colummn names:', Features)


# ### <span style="color:#543348;"> Encoding Categorical Features </span>

# In[ ]:


label_cols = ["f_27"]
def label_encoder(train,test,columns):
    for col in columns:
        train[col] = LabelEncoder().fit_transform(train[col])
        test[col] =  LabelEncoder().fit_transform(test[col])
    return train, test

train ,test = label_encoder(train,test ,label_cols)


# <a href="#toc" role="button" aria-pressed="true" style="color:#543348;"> Click here to go back to: Table of Contents </a>

# <a id="5"></a>
# # <span style="font-family: cursive; font-size: 30px; font-style: normal; font-weight: bold; letter-spacing: 0.5px; color:#543348">  Modelling ⚙️</span>

# In[ ]:


Feats_ignore = ['target', 'id']
Features = [col for col in train.columns if col not in Feats_ignore]


# ### <span style="color:#543348;"> Declare feature vector and target variable </span> 
# 

# In[ ]:


X = train[Features]
y = train[TARGET]
test = test.drop(columns=["id"])


# ### <span style="color:#543348;"> Build LGBM Classifier </span>

# In[ ]:


SEED = 2022


# In[ ]:


X_train, X_eval, y_train, y_eval = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.20, 
                                                    random_state = SEED)
print("Train/Eval Sizes : ", X_train.shape, X_eval.shape, y_train.shape, y_eval.shape)


# In[ ]:


fit_params = {"early_stopping_rounds":100, 
            "eval_metric" : 'auc', 
            "eval_set" : [(X_eval,y_eval)],
            'verbose': 500,
           }


# In[ ]:


lgb_model = lgb.LGBMClassifier (
                                n_estimators = 5000,
                                max_depth = 11,
                                num_leaves = 15,
                                learning_rate = 0.05,
                                subsample = 0.9,
                                colsample_bytree = 0.7,
                                random_state = SEED )


# In[ ]:


lgb_model.fit(X, y, **fit_params)


# ### <span style="color:#543348;"> Predict the results </span>

# In[ ]:


lgb_predictions = lgb_model.predict(test)


# <a href="#toc" role="button" aria-pressed="true" style="color:#543348;"> Click here to go back to: Table of Contents </a>

# <a id="6"></a>
# # <span style="font-family: cursive; font-size: 30px; font-style: normal; font-weight: bold; letter-spacing: 0.5px; color:#543348">  Submit To Kaggle 🇰</span>

# In[ ]:


submission[TARGET] = lgb_predictions
submission.to_csv("sample_submission.csv",index=False)
submission.head()


# <a href="#toc" role="button" aria-pressed="true" style="color:#543348;"> Click here to go back to: Table of Contents </a>

# ## <center> If you find this notebook useful, please support with an upvote 👍</center>
# ### <center> I'm here to learn and improve, so feel free to comment on anything that will make me a better Kaggler! <center>
