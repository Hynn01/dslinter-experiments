#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports

import pandas as pd
import re
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv('../input/titanic/train.csv')
data_test = pd.read_csv('../input/titanic/test.csv')
data.head()


# In[ ]:


# Finding unique Title in Test Dataset

titles_test = []
for name in list(data_test['Name']):
  title = re.findall('\s[A-z]{1,}\.\s' , name)[0]
  if(title not in titles_test):
    titles_test.append(title)
titles_test


# In[ ]:


# Marking the titles for test dataset if available in the test dataset and OTH if not available


data['Titles'] = data['Name'].apply(lambda x : re.findall('\s[A-z]{1,}\.\s' , x)[0] if (re.findall('\s[A-z]{1,}\.\s' , x)[0] in titles_test)
                                                                                    else "OTH"
                                                               )
data


# In[ ]:


data['Titles'].value_counts().plot(kind = 'bar')


# In[ ]:


# Assigning Titles column to Test Dataset

data_test['Titles'] = data_test['Name'].apply(lambda x : re.findall('\s[A-z]{1,}\.\s' , x)[0]   )
data_test


# In[ ]:


data_test['Titles'].value_counts().plot(kind = 'bar')


# In[ ]:


print(data['Survived'].value_counts())
data['Survived'].value_counts().plot(kind='bar')


# In[ ]:


sns.heatmap(pd.pivot_table(data = data , index = 'Survived' , columns = 'Sex'  ,values = 'Name', aggfunc = 'count' ))


# In[ ]:


sns.barplot(data = data , y = 'Survived' , x = 'Pclass', estimator = np.mean  )


# In[ ]:


sns.barplot(data = data , y = 'Survived' , x = 'Sex', estimator = np.mean  )


# In[ ]:


sns.barplot(data = data , y = 'Survived' , x = 'SibSp', estimator = np.mean  )


# In[ ]:


sns.barplot(data = data , y = 'Survived' , x = 'Parch', estimator = np.mean  )


# In[ ]:


sns.barplot(data = data , y = 'Survived' , x = 'Titles', estimator = np.mean  )


# In[ ]:


data.info()


# In[ ]:


data_test.info()


# In[ ]:


title_list = [' Mr. ' ,' Miss. ' , ' Mrs. ', ' Master. ' , ' Dr. ' , ' Ms. ' ]

for elem in title_list:
  mean_age = data[  data['Name'].str.contains(elem)]['Age'].mean()
  data.loc[(data['Name'].str.contains(elem)) & (data['Age'].isnull()) , 'Age' ] = mean_age
  data_test.loc[(data_test['Name'].str.contains(elem)) & (data_test['Age'].isnull()) , 'Age' ] = mean_age


# In[ ]:


data.info()


# In[ ]:


data_test.info()


# In[ ]:


data_test[data_test.Fare.isnull()]


# In[ ]:


# Assigning average fare for the PClass = 3 

data_test.loc[data_test.Fare.isnull() , 'Fare'] = data[  data['Pclass'] == 3]['Fare'].mean()


# In[ ]:


data_test.info()


# In[ ]:


data.head()


# In[ ]:


data = data[~data['Embarked'].isnull()]


# In[ ]:


data.info()


# In[ ]:


data_f =  data[['Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Embarked', 'Survived' , 'Titles']]

data_f_test =  data_test[['Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Embarked', 'Titles']]

alpha = {
    'male' : 1,
    'female' : 0
}       

data_f['Sex'] =  data_f['Sex'].map(alpha) 
data_f_test['Sex'] =  data_f_test['Sex'].map(alpha) 


# In[ ]:


data_f_test


# In[ ]:


data_f = pd.concat([data_f , pd.get_dummies(data_f['Embarked']) ] , axis = 1)
data_f


# In[ ]:


data_f_test = pd.concat([data_f_test , pd.get_dummies(data_f_test['Embarked']) ] , axis = 1)
data_f_test


# In[ ]:


data_f.drop(columns=['Embarked' ] , inplace = True)
data_f_test.drop(columns=['Embarked'] , inplace = True)


# In[ ]:


data_f


# In[ ]:


data_f = pd.concat([data_f , pd.get_dummies(data_f['Titles']) ] , axis = 1)
data_f.drop(columns=['Titles'] , inplace = True)
data_f


# In[ ]:


data_f_test = pd.concat([data_f_test , pd.get_dummies(data_f_test['Titles']) ] , axis = 1)
data_f_test.drop(columns=['Titles'] , inplace = True)
data_f_test


# In[ ]:


sns.boxplot(data = data_f , x = 'Survived' , y = 'Age')
plt.show()


# In[ ]:


sns.histplot(data_f['Age'])
plt.show()


# In[ ]:


sns.histplot(data_f['Fare'])
plt.show()


# In[ ]:


# As OTH column is not available in Test Data , removing this column
data_f.drop(columns=['OTH'] , inplace = True)
data_f


# In[ ]:


scaler = MinMaxScaler()
a = list(data_f.columns)
a.remove('Survived')

X = data_f[a]
print(X.columns)

X = scaler.fit_transform(X)

X


# In[ ]:


X_sub = data_f_test[a]
print(X_sub.columns)
X_sub = scaler.transform(X_sub)
X_sub


# In[ ]:


y = data_f['Survived']
y


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,  X_test,y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 42)


# In[ ]:


def performance(model):
  y_pred = model.predict(X_test)
  conf = confusion_matrix(y_test , y_pred)
  print(conf)
  TN = conf[0][0]
  TP = conf[1][1]
  FN = conf[1][0]
  FP = conf[0][1]

  RECALL = TP/(TP + FN)
  PRECISION = TP/(TP + FP)
  ACCURACY = (TP + TN)/(TP + FP+ TN + FN)

  print('Recall : ' , RECALL)
  print('PRECISION : ' , PRECISION)
  print('ACCURACY : ' , ACCURACY)


# In[ ]:


# Logistics regression

LogReg = LogisticRegression()

params = {
    'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter' : [100, 150],
    'penalty' : ['l1', 'l2', 'elasticnet', 'none']
}

Grid = GridSearchCV(
    estimator = LogReg,
    param_grid = params,
    cv = 5

)

Grid.fit(X_train, y_train)
print(Grid.best_params_)

LogReg  = LogisticRegression(**Grid.best_params_)
LogReg.fit(X_train, y_train)

performance(LogReg)


# In[ ]:


# Default Decision Tree

dtc  = DecisionTreeClassifier(max_depth = 6)
dtc.fit(X_train, y_train)
performance(dtc)


# In[ ]:


dtc  = DecisionTreeClassifier()

params = {
    # 'criterion' : ['gini' , 'entropy'], 
    'max_depth' : [4 , 6, 10, 20]


}

Grid = GridSearchCV(
    estimator = dtc,
    param_grid = params,
    cv = 4
)

Grid.fit(X_train, y_train)



# dtc.fit(X,y)


# In[ ]:


print(Grid.best_params_)
results = pd.DataFrame(Grid.cv_results_)
results


# In[ ]:


dtc = DecisionTreeClassifier(**Grid.best_params_)
dtc.fit(X_train, y_train)

performance(dtc)


# In[ ]:




dtc  = RandomForestClassifier()
params = {
    # 'criterion' : ['gini' , 'entropy'], 
    'max_depth' : [
                  #  5,7,10,15 ,
                  #  20 , 30 , 40 , 50
                   200
                   ], 
    'min_samples_split' : [
                          #  5,7
                          #  ,10
                          #  15,20
                           20
                           ], 
    'min_samples_leaf' : [
                          # 2,5,7,10,
                          #  15,20
                          20
                          ],
    'n_estimators' : [  200]


}

Grid = GridSearchCV(
    estimator = dtc,
    param_grid = params,
    cv = 5,
    n_jobs = 1
)

Grid.fit(X_train, y_train)



# dtc.fit(X,y)


# In[ ]:


print(Grid.best_params_)

RFC = RandomForestClassifier(**Grid.best_params_)
RFC.fit(X_train, y_train)


performance(RFC)


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

adc = AdaBoostClassifier(n_estimators=200, learning_rate=0.6, algorithm='SAMME.R')
adc.fit(X_train, y_train)

performance(adc)

