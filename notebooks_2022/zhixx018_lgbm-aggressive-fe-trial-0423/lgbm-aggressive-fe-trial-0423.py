#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb
import catboost as cgb

from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# #### refer to this notebook for engineered data
# https://www.kaggle.com/code/siukeitin/tps042022-fe-2500-features-with-tsfresh-catch22

# In[ ]:


df_train = pd.read_csv('../input/tps042022-fe-2500-features-with-tsfresh-catch22/tps042022_train.csv')
df_train_labels = pd.read_csv('../input/tabular-playground-series-apr-2022/train_labels.csv')
df_test = pd.read_csv('../input/tps042022-fe-2500-features-with-tsfresh-catch22/tps042022_test.csv')
df_smpl = pd.read_csv('../input/tabular-playground-series-apr-2022/sample_submission.csv')


# In[ ]:


# lgb model

steps = []
steps.append(('minmax_scaler', MinMaxScaler()))
steps.append(('std_scaler', StandardScaler()))

steps.append(('model', lgb.LGBMClassifier()))

pipe_lgb = Pipeline(steps)

cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=3)

scores = cross_val_score(
    pipe_lgb,
    df_train,
    df_train_labels['state'],
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1
)

print('mean score: %.3f' % scores.mean())
print('std: %.3f' % scores.std())


# In[ ]:


pipe_lgb.fit(df_train, df_train_labels['state'])
pred = pipe_lgb.predict_proba(df_test) # use predict_prob instead of predict
df_smpl['state'] = pd.Series(pred[:, 1]) # pred per se is composed of two columns

df_smpl[['sequence','state']].to_csv('submission.csv', index=False)


# In[ ]:




