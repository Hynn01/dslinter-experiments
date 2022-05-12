#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import random,os
import warnings
warnings.filterwarnings('ignore')

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn import ensemble

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


# In[ ]:


train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

DROP_COLS = [ID]
train = train.drop(DROP_COLS,axis=1)
test = test.drop(DROP_COLS,axis=1)

def checkNull_fillData(train,test):
    for col in train.columns:
        if len(train.loc[train[col].isnull() == True]) != 0:
            if train[col].dtype == "float64" or train[col].dtype == "int64":
                train.loc[train[col].isnull() == True,col] = train[col].median()
                test.loc[test[col].isnull() == True,col] = train[col].median()
            else:
                train.loc[train[col].isnull() == True,col] = "Missing"
                test.loc[test[col].isnull() == True,col] = "Missing"
                
checkNull_fillData(train,test)


str_list = [] 
num_list = []
for colname, colvalue in train.iteritems():
    if colname == TARGET:
        continue

    if type(colvalue[1]) == str:
        str_list.append(colname)
    else:
        num_list.append(colname)
        
X = train.drop([TARGET],axis=1)
y = train[TARGET]
        
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_list),
        ('cat', categorical_transformer, str_list)])

model = ensemble.RandomForestRegressor(
    n_estimators = 2000, 
    max_depth = 15, 
    random_state = SEED, 
    verbose = 1)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),("model", model)])
pipeline.fit(X, y)


# In[ ]:


X_test = test
pred_test = pipeline.predict(X_test)

submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
submission[TARGET] = pred_test
submission.to_csv(SUBMISSION_PATH, index=False)
submission.head()

