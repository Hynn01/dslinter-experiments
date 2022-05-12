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


train_url='/kaggle/input/titanic/train.csv'
test_url='/kaggle/input/titanic/test.csv'
gender_submission_url='/kaggle/input/titanic/gender_submission.csv'
df_train=pd.read_csv(train_url)
df_test=pd.read_csv(test_url)


# In[ ]:


y_test=pd.read_csv(gender_submission_url)


# In[ ]:


y_test.shape


# In[ ]:


y_test.head()


# In[ ]:


y_train=df_train['Survived'].to_frame()
x_test=df_test
passengerid=x_test['PassengerId']


# In[ ]:


x_test["PassengerId"].head()


# In[ ]:


x_train = df_train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'])


# In[ ]:


x_test = df_test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])


# In[ ]:


age_mean_train = x_train['Age'].mean()
#x['Age'].fillna(value=age_mean, inplace=True)
age_mean_test = x_test['Age'].mean()


# In[ ]:


fare_mean_test = x_test['Fare'].mean()


# In[ ]:


x_train['Age'].fillna(value=age_mean_train, inplace=True)
x_test['Age'].fillna(value=age_mean_test, inplace=True)
x_test['Fare'].fillna(value=fare_mean_test, inplace=True)


# In[ ]:


x_train['Embarked'].fillna(value='S', inplace=True)
x_test['Embarked'].fillna(value='S', inplace=True)


# In[ ]:


print('x_train Age Empty Cells: ',x_train['Age'].isnull().sum())
print('x_train Embarked Empty Cells: ',x_train['Embarked'].isnull().sum())
print('x_train Fare Empty Cells: ',x_train['Fare'].isnull().sum())
print('x_train Parch Empty Cells: ',x_train['Parch'].isnull().sum())
print('x_train Passenger class Empty Cells: ',x_train['Pclass'].isnull().sum())
print('x_train Sex Empty Cells: ',x_train['Sex'].isnull().sum())
print('x_train SibSp Empty Cells: ',x_train['SibSp'].isnull().sum())
print('x_test Age Empty Cells: ',x_test['Age'].isnull().sum())
print('x_test Embarked Empty Cells: ',x_test['Embarked'].isnull().sum())
print('x_test Fare Empty Cells: ',x_test['Fare'].isnull().sum())
print('x_test Parch Empty Cells: ',x_test['Parch'].isnull().sum())
print('x_test Passenger class Empty Cells: ',x_test['Pclass'].isnull().sum())
print('x_test Sex Empty Cells: ',x_test['Sex'].isnull().sum())
print('x_test SibSp Empty Cells: ',x_test['SibSp'].isnull().sum())


# In[ ]:


x_train['Embarked'].value_counts()


# In[ ]:


x_train['Embarked']=x_train['Embarked'].replace(to_replace="S",value="1")
x_train['Embarked']=x_train['Embarked'].replace(to_replace="C",value="2")
x_train['Embarked']=x_train['Embarked'].replace(to_replace="Q",value="3")
x_train['Sex']=x_train['Sex'].replace(to_replace="male",value="1")
x_train['Sex']=x_train['Sex'].replace(to_replace="female",value="2")


# In[ ]:


x_test['Embarked']=x_test['Embarked'].replace(to_replace="S",value="1")
x_test['Embarked']=x_test['Embarked'].replace(to_replace="C",value="2")
x_test['Embarked']=x_test['Embarked'].replace(to_replace="Q",value="3")
x_test['Sex']=x_test['Sex'].replace(to_replace="male",value="1")
x_test['Sex']=x_test['Sex'].replace(to_replace="female",value="2")


# In[ ]:


x_train.head()


# In[ ]:


x_test.head()


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

tree = DecisionTreeClassifier()


# In[ ]:


tree_cv = GridSearchCV(tree, parameters, cv=10)
tree_cv.fit(x_train, y_train)


# In[ ]:


print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
print("accuracy :",tree_cv.best_score_)


# In[ ]:


#clf.predict([[2., 2.]])
submission = tree_cv.predict(x_test)


# In[ ]:


submission = pd.DataFrame(submission)


# In[ ]:


submission["PassengerId"]=passengerid


# In[ ]:


submission.rename(columns={0: "Survived"}, inplace=True)


# In[ ]:


submission.head()


# In[ ]:


submission = submission[['PassengerId', 'Survived']]


# In[ ]:


submission.head()


# In[ ]:


submission.set_index('PassengerId', inplace=True)


# In[ ]:


#submission.columns=["PassengerId", "Survived"]
submission.shape


# In[ ]:


submission.tail()


# In[ ]:


submission.to_csv('submission.csv')


# In[ ]:




