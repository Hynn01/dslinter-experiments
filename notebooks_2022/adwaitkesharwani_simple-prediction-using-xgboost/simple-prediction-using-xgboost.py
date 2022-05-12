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


#  ***Loading training data***

# In[ ]:


x_train = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv")
x_train.head()


# ***Getting info form the dataset***

# In[ ]:


x_train.info()


# 1. ***No missing value***
# 2. ***27th column is categorical***

# ***Let's convert the categorical column to a numerical column using ordinal encoding***

# In[ ]:


from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder(handle_unknown='error')
x_train["f_27"] = pd.DataFrame(encoder.fit_transform(x_train["f_27"].to_numpy().reshape(-1,1)))


# In[ ]:


x_train.info()


# ***Training set is ready!!***

# ***Seperating labels and features***

# In[ ]:


y_train = x_train["target"]
x_train = x_train.drop(["target"],axis=1)


# # XGBoost 

# In[ ]:


import xgboost as xgb
xgb_cl = xgb.XGBClassifier()


# ***Model Training***

# In[ ]:


xgb_cl.fit(x_train, y_train)


# 

# # Pre Processing Test Set

# In[ ]:


x_test = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")
x_test["f_27"] = pd.DataFrame(encoder.fit_transform(x_test["f_27"].to_numpy().reshape(-1,1)))


# ***Getting Predictions***

# In[ ]:


predictions = xgb_cl.predict_proba(x_test)


# ***Saving Predictions***

# In[ ]:


output = pd.DataFrame({'id': x_test['id'], 'target': predictions[:,-1]})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:




