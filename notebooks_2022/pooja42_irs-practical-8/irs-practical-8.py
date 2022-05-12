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


from sklearn import metrics,datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB


# In[ ]:


url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'


# In[ ]:


sms = pd.read_table(url, names=["label","message"], header=None)
print(sms.shape)
print(sms)


# In[ ]:


sms["label_name"] = sms.label.map({"ham":0, "spam":1})
print(sms)


# In[ ]:


X, y = sms.message, sms.label_name
print(X.shape)
print(X)
print(y.shape)
print(y)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:


vect = TfidfVectorizer(stop_words='english')
print(vect.get_params())


# In[ ]:


vect.fit(X_train)
X_train_new = vect.transform(X_train)
X_train_new


# In[ ]:


X_test_new = vect.transform(X_test)
X_test_new


# In[ ]:


clf = MultinomialNB()
clf.fit(X_train_new, y_train)


# In[ ]:


y_prediction = clf.predict(X_test_new)
y_prediction


# In[ ]:


metrics.accuracy_score(y_test,y_prediction)


# In[ ]:


print(y_test.value_counts())
ham_count = y_test.value_counts()[0]
spam_count = y_test.value_counts()[1]


# In[ ]:


print(metrics.classification_report(y_test, y_prediction))


# In[ ]:


print(metrics.confusion_matrix(y_test, y_prediction))

