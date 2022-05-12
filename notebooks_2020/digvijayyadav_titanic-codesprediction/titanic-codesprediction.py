#!/usr/bin/env python
# coding: utf-8

# # Let's begin
# 

# Loading the libraries is done

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
    

# Any results you write to the current directory are saved as output.


# Train data is viewed using .head() in pandas

# In[ ]:


train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()


# To check the length of datasets

# In[ ]:


print(train_data.shape)


# In[ ]:


train_data.info()


# In[ ]:


test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()


# In[ ]:


sns.lineplot(x=train_data['Survived'], y=train_data.index)


# In[ ]:


test_data.shape


# In[ ]:


test_data.info()


# # Missing values
# you would have obsereved that certain values were either null or NaN so now let's see what proportion of them are there in training and test dataset

# In[ ]:


# Check for missing values
missing_data=train_data.isnull().sum()
missing_data


# In[ ]:


sns.distplot(a=missing_data['Age'], label='Age',kde=False)
sns.distplot(a=missing_data['Cabin'], label= 'Cabin',kde=False)
plt.legend()


# In[ ]:


test_data.isnull().sum()


# Name, Sex, Cabin & Embarked are categorical. Remove Name & Ticket as they are irrelavant. Remove cabin as too many null values. You can also use pipelines or SimpleImputer

# In[ ]:


# Name, Sex, Cabin & Embarked are categorical. Remove Name & Ticket as they are irrelavant. Remove cabin as too many null values.
train_data.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
test_data.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)


# In[ ]:


train_data.isnull().sum()
train_data.head()


# In[ ]:


test_data.isnull().sum()


# Filling the train and test data with mean taken of all the entries

# In[ ]:


train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())

test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())


# In[ ]:


# For embarked, there are 2 missing values, drop them.
train_data.dropna(subset = ["Embarked"], inplace=True)


# Dummy Encoding

# In[ ]:


# dummy encoding of 2 remaining categorical variables.
train_data = pd.get_dummies(train_data, columns=["Sex"], drop_first=True)
train_data = pd.get_dummies(train_data, columns=["Embarked"],drop_first=True)


# In[ ]:


test_data = pd.get_dummies(test_data, columns=["Sex"], drop_first=True)
test_data = pd.get_dummies(test_data, columns=["Embarked"],drop_first=True)


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


y = train_data["Survived"]

X = train_data.drop(['Survived'], axis=1)


# For Certain algorithms to work we must normalize the data so I have normalized using StandardScaler method

# In[ ]:


from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
sc.fit(train_data.drop(['Survived', 'PassengerId'], axis = 1))
X_train = sc.transform(train_data.drop(['Survived', 'PassengerId'], axis = 1))
X_train


# **Training & Scoring**

# In[ ]:


from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.23, random_state = 5)

model_LR = LogisticRegression(max_iter=5000)
model_LR.fit(X_train, y_train)
LR_predict = model_LR.predict(X_test)
LR_score = model_LR.score(X_test,y_test)

model_X = XGBClassifier(eta=0.1, n_estimators=50,
                        max_depth=5, subsample=0.6, colsample_bytree=0.7,objective= 'binary:logistic',
                        scale_pos_weight=1, seed=27)
model_X.fit(X_train, y_train)
X_predict = model_X.predict(X_test)
X_score = model_X.score(X_test,y_test)

rfc = RandomForestClassifier(n_estimators=6)
rfc.fit(X_train, y_train)
RFC_predict = rfc.predict(X_test)
RFC_score = rfc.score(X_test,y_test)

model_DTC = DecisionTreeClassifier(max_depth=7, min_samples_leaf=6, min_samples_split=2)
model_DTC.fit(X_train, y_train)
DTC_predict = model_DTC.predict(X_test)
DTC_score = model_DTC.score(X_test,y_test)

model_GB = GradientBoostingClassifier(random_state=10, n_estimators=1500,min_samples_split=100, max_depth=6)
model_GB.fit(X, y)
GB_predict = model_GB.predict(X_test)
GB_score = model_GB.score(X_test,y_test)

x1=LR_score, X_score, RFC_score, DTC_score, GB_score
print(x1)

sns.distplot(a=x1, kde=True)
plt.legend()
plt.title('Accuracy estimates of Logistic regression, XGB, decision trees, Random Forest, and Gradient Boosting')


# Let us use another algorithm called Naive bayes and now i have used cross validation scoring parameter

# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

gnb= GaussianNB()
gnb.fit(X_train, y_train)
prediction = gnb.predict(X_test)
cross_scores = cross_val_score(gnb,X_train,y_train,cv=8)
print(cross_scores)

sns.distplot(a=cross_scores, kde=True)
plt.legend()


# KNeighbors in imported from sklearn library

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

neigh= KNeighborsClassifier(n_neighbors=5, leaf_size=30)
neigh.fit(X_train, y_train)
KN_predict = neigh.predict(X_test)
cross_scores = cross_val_score(neigh,X_train,y_train,cv=8)
print(cross_scores)
print(accuracy_score(KN_predict, y_test))


# # Submission

# In[ ]:


predictions = model_X.predict(test_data)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")


# Do comment on if any improvement could be done on this
# I would be looking for more possible approaches like neural nets etc.
# instead of mean i would look for pipelines and simpleimputer in future 

# # References
# 1. https://www.kaggle.com/kshivi99/predcting-the-titanic-survivors-minimal-kernal
# 2. https://www.kaggle.com/mdmahmudferdous/titanic-survivor-prediction-0-804-top-8

# In[ ]:




