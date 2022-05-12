#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')
test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv')


# In[ ]:


x_data = train.drop(['id', 'target'], axis=1)
y_data = train.target


# In[ ]:


x_test = test.drop('id', axis=1)


# In[ ]:


data = pd.concat([x_data, x_test])


# In[ ]:


encoder = LabelEncoder()
encoder.fit(data.f_27)
x_data.f_27 = encoder.transform(x_data.f_27)


# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)


# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(x_data, y_data)


# In[ ]:


model = LogisticRegression()
model.fit(x_train, y_train)

y_predict = model.predict_proba(x_val)[:, 1]


# In[ ]:


roc_auc_score(y_val, y_predict)


# In[ ]:


model = DecisionTreeClassifier()
model.fit(x_train, y_train)
y_predict = model.predict_proba(x_val)[:,1]

roc_auc_score(y_val, y_predict)


# In[ ]:


model = XGBClassifier()
model.fit(x_train, y_train)
y_predict = model.predict_proba(x_val)[:, 1]

roc_auc_score(y_val, y_predict)


# In[ ]:


roc_auc_score(y_val, y_predict)


# In[ ]:


x_test.f_27 = encoder.transform(x_test.f_27)
x_test = scaler.transform(x_test)


# In[ ]:


y_test = model.predict_proba(x_test)[:, 1]

submission = pd.read_csv('../input/tabular-playground-series-may-2022/sample_submission.csv')
submission.target = y_test
submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:




