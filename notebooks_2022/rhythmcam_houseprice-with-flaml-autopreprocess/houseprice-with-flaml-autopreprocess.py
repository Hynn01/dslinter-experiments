#!/usr/bin/env python
# coding: utf-8

# # Define Data

# In[ ]:


from IPython.display import clear_output
get_ipython().system('pip3 install flaml ')
clear_output()

import numpy as np
import pandas as pd 
import os 
import random
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler,MaxAbsScaler,RobustScaler

from flaml import AutoML

TRAIN_PATH = "../input/house-prices-advanced-regression-techniques/train.csv"
TEST_PATH = "../input/house-prices-advanced-regression-techniques/test.csv"
SAMPLE_SUBMISSION_PATH = "../input/house-prices-advanced-regression-techniques/sample_submission.csv"
SUBMISSION_PATH = "submission.csv"

ID = "Id"
TARGET = "SalePrice"

SEED = 2022
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything()


# # Preprocess Data

# In[ ]:


def autoPreProcess(train,test,DROP_COLS,TARGET):
    train_len = len(train)

    train_test = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
    train_test = train_test.drop(DROP_COLS, axis = 1)

    def checkNull_fillData(df):
        for col in df.columns:
            if len(df.loc[df[col].isnull() == True]) != 0:
                if df[col].dtype == "float64" or df[col].dtype == "int64":
                    df.loc[df[col].isnull() == True,col] = df[col].median()
                else:
                    df.loc[df[col].isnull() == True,col] = "Missing"
            
    checkNull_fillData(train_test)

    str_list = [] 
    num_list = []
    for colname, colvalue in train_test.iteritems():
        if colname == TARGET:
            continue
            
        if type(colvalue[1]) == str:
            str_list.append(colname)
        else:
            num_list.append(colname)

    train_test = pd.get_dummies(train_test, columns=str_list)

    scaler = StandardScaler()
    train_test[num_list] = scaler.fit_transform(train_test[num_list])
    
    train = train_test[:train_len]
    test = train_test[train_len:]

    test.drop(labels=[TARGET],axis = 1,inplace=True)
    
    return train,test

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
DROP_COLS = [ID]

train,test = autoPreProcess(train,test,DROP_COLS,TARGET)
train.head()


# # Build Model

# In[ ]:


MODEL_TIME_BUDGET = 60*5
MODEL_METRIC = 'rmse'
MODEL_TASK = "regression"
MODEL_LIST = ['lgbm', 'catboost']
MODEL_LOG_FILE_PATH = "flaml_log.log"

X = train.drop([TARGET],axis=1)
y = train[TARGET]

model = AutoML()
params = {
    "time_budget": MODEL_TIME_BUDGET,  
    "metric": MODEL_METRIC,
    "estimator_list": MODEL_LIST, 
    "task": MODEL_TASK,
    "seed":SEED,
    "log_file_name":MODEL_LOG_FILE_PATH,
}
model.fit(X, y, **params)


# # Predict Data

# In[ ]:


sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
sub[TARGET] = model.predict(test)
sub.to_csv(SUBMISSION_PATH,index=False)
sub.head()

