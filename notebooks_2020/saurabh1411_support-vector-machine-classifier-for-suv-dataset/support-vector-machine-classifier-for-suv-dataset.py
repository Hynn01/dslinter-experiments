#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


dataset=pd.read_csv('../input/classification-suv-dataset/Social_Network_Ads.csv')


# Dataset view 

# In[ ]:


dataset.head()


# In[ ]:


X=dataset.iloc[:,[2,3]].values 
y=dataset.iloc[:,-1].values


# Applying SVM 
# 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2
                                              )


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)


# In[ ]:


from sklearn.svm import SVC
classifier=SVC(kernel='rbf',gamma='auto',random_state=0)
classifier.fit(X_train,y_train)


# In[ ]:


y_pred=classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


# In[ ]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
acc


# with linear kernel model

# In[ ]:


from sklearn.svm import SVC
classifier=SVC(kernel='linear',gamma='auto',random_state=0)
classifier.fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
acc


# In[ ]:


from sklearn.svm import SVC
classifier=SVC(kernel='sigmoid',gamma='auto',random_state=0)
classifier.fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
acc


# In[ ]:




