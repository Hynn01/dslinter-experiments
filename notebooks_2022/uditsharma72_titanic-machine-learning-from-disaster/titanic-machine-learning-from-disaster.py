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


import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
sub = test = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


train.shape


# In[ ]:


train.describe()


# In[ ]:


sex =  pd.get_dummies(train['Sex'])


# In[ ]:


embarked = pd.get_dummies(train['Embarked'])


# In[ ]:


# dropiing the category value and merging newly defined variable
train.drop(['Sex','Embarked','Name',"Ticket",'Cabin'],axis=1,inplace=True)
train.head()


# In[ ]:


train = pd.concat([train,sex,],axis=1)
train.head()


# In[ ]:


sns.heatmap(train.isnull(),yticklabels = False,cbar=False,cmap='viridis')


# **Data Cleaning**

# In[ ]:


train.dropna(inplace=True)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train , X_test, y_train, y_test = train_test_split(
    train.drop('Survived',axis=1), train['Survived'],test_size =0.2,
                                                    random_state=42)


# In[ ]:


X_train.shape,y_train.shape


# In[ ]:


X_train.head()


# In[ ]:


X_train.info()


# In[ ]:


y_train


# **LogisticRegression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)


# In[ ]:


predictions = logistic_model.predict(X_test)


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
print(cross_val_score(logistic_model,X_train,y_train,cv=5))
accuracy = accuracy_score(y_test,predictions)*100
accuracy


# **Support vector algorithm**

# In[ ]:


from sklearn.svm import SVC
classifiers = SVC(kernel='linear', random_state=0)  
classifiers.fit(X_train, y_train)
y_predicted = classifiers.predict(X_test)  


# In[ ]:


accuracySVC = accuracy_score(y_predicted,y_test)*100
print("The accuracy OF Support vector :",accuracySVC)
# print(cross_val_score(classifiers,X_train,y_train,cv=5))


# In[ ]:


print(cross_val_score(classifiers,X_train,y_train,cv=5))


# **RandomForestRegressor**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


forestClass = RandomForestClassifier()
forestClass.fit(X_train,y_train)
forestClassPred = forestClass.predict(X_test)


# In[ ]:


forestClassPred


# In[ ]:


forestClassAcc = accuracy_score(forestClassPred,y_test)*100
print("The accuracy OF Support vector :",accuracySVC)
# print(cross_val_score(classifiers,X_train,y_train,cv=5))


# In[ ]:


print(cross_val_score(forestClass,X_train,y_train,cv=5))


# **Acc Table**

# In[ ]:


from prettytable import PrettyTable
Table = PrettyTable(["Algorithm", "Accuracy"])
Table.add_row(["LogisticRegression", accuracy])
Table.add_row(["RandomForestClassifier", forestClassAcc])
Table.add_row(["SVC", accuracySVC])
print(Table)

