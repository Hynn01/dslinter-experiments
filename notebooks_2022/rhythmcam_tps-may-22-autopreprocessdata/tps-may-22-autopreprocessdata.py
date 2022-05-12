#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# imports 
import numpy as np
import pandas as pd 
import os,random,gc
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler,MaxAbsScaler,RobustScaler

# variables
TRAIN_PATH = "../input/tabular-playground-series-may-2022/train.csv"
TEST_PATH = "../input/tabular-playground-series-may-2022/test.csv"
SAMPLE_SUBMISSION_PATH = "../input/tabular-playground-series-may-2022/sample_submission.csv"
SUBMISSION_PATH = "submission.csv"

ID = "id"
TARGET = "target"

SEED = 2022
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything()


# # Preprocess Data

# In[ ]:


def autoPreprocLabelEncoder(train,test,DROP_COLS,TARGET):
    
    train = train.drop(DROP_COLS, axis = 1)
    test = test.drop(DROP_COLS, axis = 1)
    
    for col in train.columns:
        if col == TARGET:
            continue
            
        if len(train.loc[train[col].isnull() == True]) != 0:
            if train[col].dtype == "float64" or train[col].dtype == "int64":
                train.loc[train[col].isnull() == True,col] = train[col].median()
                test.loc[test[col].isnull() == True,col] = train[col].median()
            elif train[col].dtype == "float32" or train[col].dtype == "int32":
                train.loc[train[col].isnull() == True,col] = train[col].median()
                test.loc[test[col].isnull() == True,col] = train[col].median()
            elif train[col].dtype == "float16" or train[col].dtype == "int16":
                train.loc[train[col].isnull() == True,col] = train[col].median()
                test.loc[test[col].isnull() == True,col] = train[col].median()
            else:
                train.loc[train[col].isnull() == True,col] = "Missing"
                test.loc[test[col].isnull() == True,col] = "Missing"
            
    train_len = len(train)
    train_test = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
  
    str_list = []
    num_list = []
    for col in train:
        if col == TARGET:
            continue
            
        if train[col].dtypes == "object":
            str_list.append(col)
        else:
            num_list.append(col)
            
    print("str_list",str_list)
    print("num_list",num_list)

    for col in str_list:
        encoder = LabelEncoder()
        train_test[col] = encoder.fit_transform(train_test[col])
    
    scaler = StandardScaler()
    train_test[num_list] = scaler.fit_transform(train_test[num_list])
    
    train = train_test[:train_len]
    test = train_test[train_len:]

    test.drop(labels=[TARGET],axis = 1,inplace=True)
    
    return train,test

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
DROP_COLS = [ID]

train,test = autoPreprocLabelEncoder(train,test,DROP_COLS,TARGET)
train.head()

