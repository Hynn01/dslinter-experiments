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


from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target


# In[ ]:


clf1 = LogisticRegression(random_state=0, solver='newton-cg', max_iter=1000,multi_class='multinomial')
clf1.fit(X,y)


# In[ ]:


for f, w in zip(breast_cancer.feature_names, clf1.coef_[0]) :
 print("{0:<23}: {1:6.2f}". format(f, w))


# In[ ]:


clf2 = DecisionTreeClassifier(max_depth=10)
clf2.fit(X, y)

