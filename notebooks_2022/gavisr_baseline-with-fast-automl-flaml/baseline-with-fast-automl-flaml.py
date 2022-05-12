#!/usr/bin/env python
# coding: utf-8

# <div>
# <img src="https://raw.githubusercontent.com/microsoft/FLAML/c1e1299855dcea378591628ae4ecb818426cb98f/website/static/img/flaml.svg" width="750" height="750">   
# </div>
# 
# <span style = "font-family:cursive; font-size:20px; letter-spacing:0.5px;  border-radius:5px;  background-color:#eff4f4; "> 
# <p style="padding: 10px; color:#543348; text-align: justify;">
# FLAML is a lightweight Python library that finds accurate machine learning models automatically, efficiently and economically. It frees users from selecting learners and hyperparameters for each learner.
# 
#     
# * For common machine learning tasks like classification and regression, it quickly finds quality models for user-provided data with low computational resources. It supports both classifcal machine learning models and deep neural networks.
# * It is easy to customize or extend. Users can find their desired customizability from a smooth range: minimal customization (computational resource budget), medium customization (e.g., scikit-style learner, search space and metric), or full customization (arbitrary training and evaluation code).
# * It supports fast automatic tuning, capable of handling complex constraints/guidance/early stopping. FLAML is powered by a new, cost-effective hyperparameter optimization and learner selection method invented by Microsoft Research.
# </span>
# 

# <p id="1"></p>
# 
# # <span style="font-family: cursive; font-size: 30px; font-style: normal; font-weight: bold; letter-spacing: 0.5px; color:#543348">Libraries  📚</span>

# In[ ]:


# base
import pandas as pd
import numpy as np
import os
import random

# flaml
get_ipython().system('pip install flaml')
get_ipython().system('pip install -U scikit-learn')
from flaml import AutoML

# warning
import warnings
warnings.simplefilter('ignore')

# parameters
SEED = 2022
FOLDS = 5


# <p id="2"></p>
# 
# # <span style="font-family: cursive; font-size: 30px; font-style: normal; font-weight: bold; letter-spacing: 0.5px; color:#543348"> Load Datasets 🗃️</span>

# In[ ]:


def reduce_mem_usage(df):
    """ Iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
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
    print('Memory usage after optimisation is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def import_data(file):
    """Create a dataframe and optimise its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df


# In[ ]:


print('-' * 100)
print('train')
train = import_data('../input/tabular-playground-series-may-2022/train.csv')

print('-' * 100)
print('test')
test = import_data('../input/tabular-playground-series-may-2022/test.csv')

submission = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")


# <a id="3"></a>
# 
# # <span style="font-family: cursive; font-size: 30px; font-style: normal; font-weight: bold; letter-spacing: 0.5px; color:#543348">Data Exploration 🔎🛠</span>

# ### <span style="color:#543348;"> Below are the first 5 rows of train and test datasets:</span>

# In[ ]:


train.head(5)


# In[ ]:


test.head(5)


# In[ ]:


print(f'Train data set: {train.shape[0]} rows and {train.shape[1]} columns')
print(f'Test data set: {test.shape[0]} rows and {test.shape[1]} columns') 


# ### <span style="color:#543348;"> Below are the number of missing values in the train and test datasets: </span>

# 📌 Observation: There are no missing values in both the train and test datasets

# In[ ]:


print(f'Missing values in the train dataset: {train.isna().sum().sum()}')
print(f'Missing values in the test dataset: {test.isna().sum().sum()}')


# ### <span style="color:#543348;"> The number of unique values for each variable in the train dataset: </span> 

# In[ ]:


train.nunique().sort_values(ascending = True)


# ### <span style="color:#543348;"> Overview of the the train dataset including number of columns, column labels, data types, memory usage, range index, and the number of cells in each column (non-null values): </span> 

# 📌 Observation: f_27 is a categorical data type, the rest are float and int data types

# In[ ]:


train.info()


# ### <span style="color:#543348;"> Basic statistics of training data for each variables which contain information on count, mean, standard deviation, minimum, 1st quartile, median, 3rd quartile and maximum: </span>

# In[ ]:


train.describe().T


# In[ ]:


Feats_ignore = ['target', 'id','f_27']
Features = [col for col in train.columns if col not in Feats_ignore]

df_train = train[Features]
df_target = train.target
TARGET = 'target'


# <a id="4"></a>
# 
# # <span style="font-family: cursive; font-size: 30px; font-style: normal; font-weight: bold; letter-spacing: 0.5px; color:#543348">  FLAML - AUTOML ⭐🚀</span>
#            

# 📝 Note: Increasing time_budget will yield better hyperparameters and therefore the final model will be more accurate.

# In[ ]:


# Initialize an AutoML instance
automl = AutoML()

# Specify automl goal and constraint
automl_settings = {
    "time_budget": 60*10,  # in seconds
    "metric": 'roc_auc',
    "task": 'classification',
    "n_jobs": -1,
    "eval_method": 'cv',
    "n_splits": FOLDS,
    "seed" : SEED}

# Train with labeled input data
automl.fit(df_train, df_target, **automl_settings)


# In[ ]:


print('Best ML leaner:', automl.best_estimator)
print('Best hyperparmeter config:', automl.best_config)
print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))


# In[ ]:


# Make prediction
y_pred= automl.predict_proba(test)[:,1]


# <a id="5"></a>
# 
# # <span style="font-family: cursive; font-size: 30px; font-style: normal; font-weight: bold; letter-spacing: 0.5px; color:#543348">  Submit To Kaggle 🇰</span>

# In[ ]:


submission[TARGET] = y_pred
submission.to_csv("submission.csv",index=False)
submission.head()
print("Your submission was successfully saved!")


# # References
# https://microsoft.github.io/FLAML/docs/getting-started
# 
# https://github.com/microsoft/FLAML

# ## <center> If you find this notebook useful, please support with an upvote 👍</center>
# ### <center> I'm here to learn and improve, so feel free to comment on anything that will make me a better Kaggler! <center>
