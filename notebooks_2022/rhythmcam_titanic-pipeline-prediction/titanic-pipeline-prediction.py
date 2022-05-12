#!/usr/bin/env python
# coding: utf-8

# # Define Data

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

TRAIN_PATH = "../input/titanic/train.csv"
TEST_PATH = "../input/titanic/test.csv"
SAMPLE_SUBMISSION_PATH = "../input/titanic/gender_submission.csv"
SUBMISSION_PATH = "submission.csv"

ID = "PassengerId"
TARGET = "Survived"

SEED = 2022
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything()


# # Build Model

# In[ ]:


train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

DROP_COLS = ['PassengerId', 'Name','Ticket', 'Cabin']
train = train.drop(DROP_COLS,axis=1)
test = test.drop(DROP_COLS,axis=1)

str_list = [] 
num_list = []
for colname, colvalue in test.iteritems():
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
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_list),
        ('cat', categorical_transformer, str_list)])

model = ensemble.RandomForestClassifier(
    n_estimators = 750, 
    criterion = 'gini', 
    max_features = 'sqrt', 
    max_depth = 5, 
    min_samples_split = 4, 
    min_samples_leaf = 2, 
    n_jobs = -1, 
    random_state = SEED, 
    verbose = 1)

clf = Pipeline(steps=[('preprocessor', preprocessor),("model", model)])
clf.fit(X, y)


# # Predict Data

# In[ ]:


X_test = test
pred_test = clf.predict(X_test)

submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
submission[TARGET] = pred_test.astype(int)
submission.to_csv(SUBMISSION_PATH, index=False)
submission.head()

