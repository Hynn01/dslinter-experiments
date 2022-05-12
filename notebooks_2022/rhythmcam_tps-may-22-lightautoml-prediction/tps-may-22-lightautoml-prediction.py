#!/usr/bin/env python
# coding: utf-8

# # Imports and Defines

# In[ ]:


from IPython.display import clear_output
get_ipython().system('pip install lightautoml --user')
clear_output()

import numpy as np
import pandas as pd 
import random,os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from lightautoml.report.report_deco import ReportDeco

TRAIN_PATH = "../input/tabular-playground-series-may-2022/train.csv"
TEST_PATH = "../input/tabular-playground-series-may-2022/test.csv"
SAMPLE_SUBMISSION_PATH = "../input/tabular-playground-series-may-2022/sample_submission.csv"
SUBMISSION_PATH = "submission.csv "

ID = "id"
TARGET = "target"

SEED = 2022
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything()

MODEL_TASK = Task('binary')
MODEL_ROLES = {'target': TARGET}
MODEL_THREADS = 2
MODEL_FOLDS = 2
MODEL_TIMEOUT = 60*5


# # Preprocess Data

# In[ ]:


train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

for df in [train, test]:
    for i in range(10):
        df[f'ch{i}'] = df.f_27.str.get(i).apply(ord) - ord('A')
        
    df["unique_characters"] = df.f_27.apply(lambda s: len(set(s)))
    
scaler = StandardScaler()
col = [col for col in train.columns if col not in [ID, 'f_27', TARGET]]
train[col] = scaler.fit_transform(train[col])
test[col] = scaler.transform(test[col])

train.drop(columns = [ID,'f_27'], inplace = True)
test.drop(columns = [ID,'f_27'], inplace = True)


# # Build Model

# In[ ]:


automl = TabularAutoML(task = MODEL_TASK, timeout = MODEL_TIMEOUT, cpu_limit = MODEL_THREADS,
                       reader_params = {'n_jobs': MODEL_THREADS, 'cv': MODEL_FOLDS, 'random_state': SEED})
pred = automl.fit_predict(train, roles = MODEL_ROLES, verbose=3)

predict_data = pred.data[:, 0]
predict_data


# # Predict Data

# In[ ]:


pred_test = automl.predict(test)

sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
sub[TARGET] =  pred_test.data[:, 0]
sub.to_csv(SUBMISSION_PATH, index=False)
sub.head()

