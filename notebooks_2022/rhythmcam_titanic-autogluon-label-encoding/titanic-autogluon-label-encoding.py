#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import clear_output
get_ipython().system('pip install autogluon --user')
clear_output()

import numpy as np
import pandas as pd 
import os 
import random
import warnings
warnings.filterwarnings(action='ignore')

from autogluon.tabular import TabularPredictor


# In[ ]:


TRAIN_PATH = "../input/titanic/train.csv"
TEST_PATH = "../input/titanic/test.csv"
SAMPLE_SUBISSION_PATH = "../input/titanic/gender_submission.csv"
SUBMISSION_PATH = "submission.csv"
TARGET = "Survived"

SEED = 2022
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything()

MODEL_EVAL_METRIC = 'accuracy'
MODEL_TIME_LIMIT = 60*1
MODEL_SAVE_PATH = "autogluon_model/"
MODEL_VERBOSE = 3
MODEL_PRESETS = "best_quality"


# In[ ]:


DROP_ELEMENTS = ['Name', 'Ticket', 'Cabin']

def toLabelEncode(TRAIN_PATH,TEST_PATH):
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    
    train = train.drop(DROP_ELEMENTS, axis = 1)
    test = test.drop(DROP_ELEMENTS, axis = 1)
    
    cat_col = train.describe(include="O").columns.tolist()
    
    for col in cat_col:
        train[col] = train[col].astype("category").cat.codes
        test[col] = test[col].astype("category").cat.codes
        
    return train,test

train,test = toLabelEncode(TRAIN_PATH,TEST_PATH)


# In[ ]:


# train = pd.read_csv(TRAIN_PATH)

predictor = TabularPredictor(
    label=TARGET, 
    eval_metric=MODEL_EVAL_METRIC, 
    path=MODEL_SAVE_PATH, 
    verbosity=MODEL_VERBOSE).fit(
    train, 
    presets=MODEL_PRESETS, 
    time_limit=MODEL_TIME_LIMIT
)


# In[ ]:


results = predictor.fit_summary()
results


# In[ ]:


pred_test = predictor.predict_proba(test,as_multiclass=False)

sub = pd.read_csv(SAMPLE_SUBISSION_PATH)
sub[TARGET] = (pred_test > 0.5).astype(int)
sub.to_csv(SUBMISSION_PATH,index=False)
sub.head()

