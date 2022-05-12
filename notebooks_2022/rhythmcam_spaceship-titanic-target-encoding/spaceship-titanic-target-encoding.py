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

from xgboost import XGBClassifier

# variables
TRAIN_PATH = "../input/spaceship-titanic/train.csv"
TEST_PATH = "../input/spaceship-titanic/test.csv"
SAMPLE_SUBMISSION_PATH = "../input/spaceship-titanic/sample_submission.csv"
SUBMISSION_PATH = "submission.csv"

ID = "PassengerId"
TARGET = "Transported"

SEED = 2022
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything()


# In[ ]:


train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(train.describe(include="O"))


# In[ ]:


train.info()


# In[ ]:


# import module we'll need to import our custom module
from shutil import copyfile

# copy our file into the working directory (make sure it has .py suffix)
copyfile(src = "../input/autosplitfeatureutil/auto_splited_feature_util.py", dst = "../working/auto_splited_feature_util.py")

from auto_splited_feature_util import *


# In[ ]:


def autoPreProcess(train,test,DROP_COLS,BOOL_COLS,TARGET):
    
    train = train.drop(DROP_COLS, axis = 1)
    test = test.drop(DROP_COLS, axis = 1)
    
    def checkNull_fillData(train,test):
        for col in train.columns:
            if len(train.loc[train[col].isnull() == True]) != 0:
                if train[col].dtype == "float64" or train[col].dtype == "int64":
                    train.loc[train[col].isnull() == True,col] = 0
                    test.loc[test[col].isnull() == True,col] = 0
                elif col in BOOL_COLS:
                    train.loc[train[col].isnull() == True,col] = train[col].mode()[0]
                    test.loc[test[col].isnull() == True,col] = train[col].mode()[0]
                else:
                    train.loc[train[col].isnull() == True,col] = "Missing"
                    test.loc[test[col].isnull() == True,col] = "Missing"
        
            
    checkNull_fillData(train,test)
    
    for col in BOOL_COLS:
        train[col] = train[col].astype(int)
        test[col] = test[col].astype(int)
            
    num_list = train.describe().columns.tolist()

    scaler = StandardScaler()
    train[num_list] = scaler.fit_transform(train[num_list])
    test[num_list] = scaler.transform(test[num_list])
    
    train_len = len(train)

    train_test = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
    
#     COL = "Cabin"
#     SPLIT_SIZE = 3
#     SEP = "/"

#     train_test = addSplitedColumn(train_test,COL,SPLIT_SIZE,SEP)
    
    str_col_list = train_test.describe(include="O").columns.tolist()

    for col in str_col_list:
        if col == TARGET or col == ID:
            continue

        meanData = train_test.groupby(col)[TARGET].mean()
        train_test[f"{col}Mean"] = train_test[col].map(meanData)
        train_test = train_test.drop(col,axis=1)

    train = train_test[:train_len]
    test = train_test[train_len:]

    test.drop(labels=[TARGET],axis = 1,inplace=True)
    
    return train,test

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
DROP_COLS = [ID,"Name","Cabin","Destination"]
BOOL_COLS = ["CryoSleep","VIP"]

train,test = autoPreProcess(train,test,DROP_COLS,BOOL_COLS,TARGET)
train.head(10)


# In[ ]:


train.head(10)


# In[ ]:


y = train[TARGET]
X = train.drop([TARGET],axis=1)
X_test = test

MODEL_TREE_METHOD = 'gpu_hist'
# MODEL_EVAL_METRIC = "accuracy"

model = XGBClassifier(tree_method=MODEL_TREE_METHOD,
#                       eval_metric=MODEL_EVAL_METRIC,
#                       max_depth=12,
                      random_state = SEED
                      
                     ) 
model.fit(X, y)


# In[ ]:


pred_test = model.predict_proba(X_test)[:, 1]
print(pred_test[:5])

sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
sub[TARGET] = (pred_test > 0.5).astype(bool)
sub.to_csv(SUBMISSION_PATH, index=False)
sub.head()

