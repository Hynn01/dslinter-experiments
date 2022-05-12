#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

test_size = 0.25
random_state = 42

df_train = pd.read_csv("../input/titanic/train.csv")
df_test = pd.read_csv("../input/titanic/test.csv")

df_train.info()


# In[ ]:


#
df_train.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis = 1, inplace = True)
df_test.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis = 1, inplace = True)

#
df_train.Age.fillna(1000, inplace = True)
df_test.Age.fillna(1000, inplace = True)

df_train.isnull().sum()

#
df_train.Fare.interpolate(inplace = True)
df_test.Fare.interpolate(inplace = True)

df_train.isnull().sum()

#
df_train.fillna("U", inplace = True)
df_test.fillna("U", inplace = True)

df_train.Embarked.unique()

#
sex_mapper = {key: value for value, key in enumerate(df_train.Sex.unique())}
embarked_mapper = {key: value for value, key in enumerate(df_train.Embarked.unique())}

df_train.Sex = df_train.Sex.map(sex_mapper)
df_train.Embarked = df_train.Embarked.map(embarked_mapper)

df_test.Sex = df_test.Sex.map(sex_mapper)
df_test.Embarked = df_test.Embarked.map(embarked_mapper)


# In[ ]:


#
x = df_train.iloc[:, 1:]
y = df_train.iloc[:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = random_state)

scaler = StandardScaler()

x_train = pd.DataFrame(scaler.fit_transform(x_train), columns = x_train.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns = x_test.columns)


# In[ ]:


# linear regression
model_lr = LogisticRegression(penalty = "l2", random_state = random_state, max_iter = 10000, n_jobs = -1)
model_lr.fit(x_train, y_train)
model_lr.score(x_test, y_test)

y_predict_lr = model_lr.predict(x_test)
confusion_matrix(y_predict_lr, y_test)

# naive bayes
model_nb = GaussianNB()
model_nb.fit(x_train, y_train)
model_nb.score(x_test, y_test)

y_predict_nb = model_nb.predict(x_test)
confusion_matrix(y_predict_nb, y_test)

# support vector machines
model_svc = SVC()
model_svc.fit(x_train, y_train)
model_svc.score(x_test, y_test)

y_predict_svc = model_svc.predict(x_test)
confusion_matrix(y_predict_svc, y_test)

# random forests
grid_rf = GridSearchCV(RandomForestClassifier(criterion = "gini", random_state = random_state), {"n_estimators": range(2, 502, 10)}, cv = 10)
grid_rf.fit(x_train, y_train)
model_rf = grid_rf.best_estimator_
model_rf.score(x_test, y_test)

y_predict_rf = model_rf.predict(x_test)
confusion_matrix(y_predict_rf, y_test)

model_rf.get_params()


# In[ ]:


predict_result = model_rf.predict(df_test)

final = pd.DataFrame(pd.read_csv("../input/titanic/test.csv").PassengerId)
final["Survived"] = predict_result

final.to_csv("./submission.csv")


# In[ ]:




