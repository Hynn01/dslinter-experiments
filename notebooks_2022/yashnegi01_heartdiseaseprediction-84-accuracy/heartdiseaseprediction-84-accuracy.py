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


import pandas as pd
import numpy as np
import seaborn as sns 


# In[ ]:


data = pd.read_csv("/kaggle/input/heart-failure-prediction/heart.csv")

data.head()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.dtypes


# In[ ]:


cat_feat = []
col = list(data.columns)
for i in col:
    if data[i].dtype=="object":
        cat_feat.append(i)
cat_feat


# In[ ]:


for i in cat_feat:
    print(data[i].unique())


# In[ ]:


X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [1, 2, 6, 8, 10])], remainder="passthrough")
X = np.array(ct.fit_transform(X))


# In[ ]:


from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25, random_state=42)


# In[ ]:


from xgboost import XGBClassifier
xg = XGBClassifier()
xg.fit(X_train, y_train)
y_predxg = xg.predict(X_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)
y_predlr = lr.predict(X_test)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=13,metric="minkowski", p=2)
knn.fit(X_train, y_train)
y_predknn = knn.predict(X_test)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
y_prednb = nb.predict(X_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=20, criterion="entropy")
rf.fit(X_train, y_train)
y_predrf = rf.predict(X_test) 


# In[ ]:


from sklearn.metrics import mean_absolute_error

xgmean = cross_val_score(estimator=xg, X=X_train, y=y_train, cv =10)
logimean = cross_val_score(estimator=lr, X=X_train, y=y_train, cv =10)
knnmean = cross_val_score(estimator=knn, X=X_train, y=y_train, cv =10)
nbmean = cross_val_score(estimator=nb, X=X_train, y=y_train, cv =20)
rfmean = cross_val_score(estimator=rf, X=X_train, y=y_train, cv =20)

print(f"Mean Absolute Error of XGboost : {mean_absolute_error(y_test, y_predxg)}")
print(f"Accuracy of XGBoost : {xgmean.mean()}\n")

print(f"Mean Absolute Error of Logistic Regression : {mean_absolute_error(y_test, y_predlr)}")
print(f"Accuracy of Logistic Regression : {logimean.mean()}\n")

print(f"Mean Absolute Error of KNN : {mean_absolute_error(y_test, y_predknn)}")
print(f"Accuracy of KNN : {knnmean.mean()}\n")

print(f"Mean Absolute Error of Naicve Bayes : {mean_absolute_error(y_test, y_prednb)}")
print(f"Accuracy of Naive Bayes : {nbmean.mean()}\n")

print(f"Mean Absolute Error of Random Forest : {mean_absolute_error(y_test, y_predrf)}")
print(f"Accuracy of Random Forest : {rfmean.mean()}\n")

