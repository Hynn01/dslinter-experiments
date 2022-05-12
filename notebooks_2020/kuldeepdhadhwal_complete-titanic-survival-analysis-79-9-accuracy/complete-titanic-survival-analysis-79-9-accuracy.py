#!/usr/bin/env python
# coding: utf-8

# **Titanic**: Machine Learning from Disaster
# - **Predict survival on the Titanic**
# - **Defining the problem statement**
# -  **Collecting the data**
# - **Exploratory data analysis**
# - **Feature engineering**
# - **Modelling**
# - **Testing**

# <h1 style= "color:red"> Please UpVote If You find this Kernel helpFul..</h1>

# # 1. Defining the problem statement

# Complete the analysis of what sorts of people were likely to survive.
# In particular, we ask you to apply the tools of machine learning to predict which passengers survived the Titanic tragedy.

# In[ ]:


from IPython.display import Image
Image(url= "https://i.ytimg.com/vi/1PhMWUoPDsk/maxresdefault.jpg")


# # Import required modules

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # 2. Collecting the data

# # load train, test dataset using Pandas

# In[ ]:


train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')
survived = train_df['Survived']
passenger_id = test_df['PassengerId']


# In[ ]:


submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
submission.head()


# # 3. Exploratory data analysis

# In[ ]:


# print first five rows
train_df.head()


# # Data Dictionary
# - Survived: 0 = No, 1 = Yes
# - pclass: Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd
# - sibsp: # of siblings / spouses aboard the Titanic
# - parch: # of parents / children aboard the Titanic
# - ticket: Ticket number
# - cabin: Cabin number
# - embarked: Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton
# Total rows and columns
# 
# We can see that there are 891 rows and 12 columns in our training dataset.

# In[ ]:


# print first five rows
test_df.head()


# In[ ]:


print(train_df.shape)
print(test_df.shape)


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# We can see that Age value is missing for many rows.
# 
# Out of 891 rows, the Age value is present only in 714 rows.
# 
# Similarly, Cabin values are also missing in many rows. Only 204 out of 891 rows have Cabin values.

# In[ ]:


print(train_df.isnull().sum())
print(test_df.isnull().sum())


# There are 177 rows with missing Age, 687 rows with missing Cabin and 2 rows with missing Embarked information.
# 

# Bar Chart for Categorical Features
# - Pclass
# - Sex
# - SibSp ( # of siblings and spouse)
# - Parch ( # of parents and children)
# - Embarked
# - Cabin

# In[ ]:


# The plots gives a good idea about the basic data distribution of any of the attributes.
train_df.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
           xlabelsize=8, ylabelsize=8, grid=False)    
plt.tight_layout(rect=(0, 0, 1.2, 1.2))   


# In[ ]:


# The plots gives a good idea about the basic data distribution of any of the attributes.
test_df.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
           xlabelsize=8, ylabelsize=8, grid=False)    
plt.tight_layout(rect=(0, 0, 1.2, 1.2))   


# In[ ]:


def bar_chart(feature):
    survived = train_df[train_df['Survived']==1][feature].value_counts()
    dead = train_df[train_df['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[ ]:


bar_chart('Sex')


# The Chart confirms Women more likely survivied than Men

# In[ ]:


bar_chart('Pclass')


# - The Chart confirms 1st class more likely survivied than other classes
# - The Chart confirms 3rd class more likely dead than other classes

# In[ ]:


bar_chart('SibSp')


# - The Chart confirms a person aboarded with more than 2 siblings or spouse more likely survived
# - The Chart confirms a person aboarded without siblings or spouse more likely dead

# In[ ]:


bar_chart('Parch')


# - The Chart confirms a person aboarded with more than 2 parents or children more likely survived
# - The Chart confirms a person aboarded alone more likely dead

# In[ ]:


bar_chart('Embarked')


# 
# - The Chart confirms a person aboarded from C slightly more likely survived
# - The Chart confirms a person aboarded from Q more likely dead
# - The Chart confirms a person aboarded from S more likely dead

# # 4. Feature engineering
# Feature engineering is the process of using domain knowledge of the data
# to create features (feature vectors) that make machine learning algorithms work.
# 
# feature vector is an n-dimensional vector of numerical features that represent some object.
# Many algorithms in machine learning require a numerical representation of objects,
# since such representations facilitate processing and statistical analysis.

# In[ ]:


train_df.head()


# # Name
# 

# In[ ]:


train_test_data = [train_df, test_df] # combining train and test dataset
print(train_test_data)
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


train_df['Title'].value_counts()


# In[ ]:


test_df['Title'].value_counts()


# In[ ]:


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


bar_chart('Title')


# In[ ]:


# delete unnecessary feature from dataset
train_df.drop('Name', axis=1, inplace=True)
test_df.drop('Name', axis=1, inplace=True)


# Feature Sex
# - male: 0,  female: 1

# In[ ]:


sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[ ]:


bar_chart('Sex')


# **Age**
# - some age is missing
# - Let's use Title's median age for missing Age

# In[ ]:


# fill missing age with median age for each title (Mr, Mrs, Miss, Others)
train_df["Age"].fillna(train_df.groupby("Title")["Age"].transform("median"), inplace=True)
test_df["Age"].fillna(test_df.groupby("Title")["Age"].transform("median"), inplace=True)


# In[ ]:


train_df.groupby("Title")["Age"].transform("median")


# In[ ]:


facet = sns.FacetGrid(train_df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_df['Age'].max()))
facet.add_legend()
 
plt.show()


# In[ ]:


facet = sns.FacetGrid(train_df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_df['Age'].max()))
facet.add_legend()
plt.xlim(0, 20)


# In[ ]:


facet = sns.FacetGrid(train_df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_df['Age'].max()))
facet.add_legend()
plt.xlim(20, 30)


# In[ ]:


facet = sns.FacetGrid(train_df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_df['Age'].max()))
facet.add_legend()
plt.xlim(30, 40)


# In[ ]:


facet = sns.FacetGrid(train_df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_df['Age'].max()))
facet.add_legend()
plt.xlim(40, 60)


# In[ ]:


facet = sns.FacetGrid(train_df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_df['Age'].max()))
facet.add_legend()
plt.xlim(60)


# Converting Numerical Age to Categorical Variable
# 
# feature vector map:
# - child: 0
# - young: 1
# - adult: 2
# - mid-age: 3
# - senior: 4

# In[ ]:


for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4


# In[ ]:


train_df.head()


# In[ ]:


bar_chart('Age')


# # Embarked
# **filling missing values**

# In[ ]:


Pclass1 = train_df[train_df['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train_df[train_df['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train_df[train_df['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# - more than 50% of 1st class are from S embark
# - more than 50% of 2nd class are from S embark
# - more than 50% of 3rd class are from S embark
# 
# fill out missing embark with S embark

# In[ ]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[ ]:


embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# # Fare

# In[ ]:


# fill missing Fare with median fare for each Pclass
train_df["Fare"].fillna(train_df.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test_df["Fare"].fillna(test_df.groupby("Pclass")["Fare"].transform("median"), inplace=True)
train_df.head(5)


# In[ ]:


facet = sns.FacetGrid(train_df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train_df['Fare'].max()))
facet.add_legend()
 
plt.show()


# In[ ]:


facet = sns.FacetGrid(train_df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train_df['Fare'].max()))
facet.add_legend()
plt.xlim(0, 20)


# In[ ]:


facet = sns.FacetGrid(train_df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train_df['Fare'].max()))
facet.add_legend()
plt.xlim(0, 30)


# In[ ]:


facet = sns.FacetGrid(train_df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train_df['Fare'].max()))
facet.add_legend()
plt.xlim(0)


# In[ ]:


train_df.head()


# # Cabin

# In[ ]:


train_df.Cabin.value_counts()


# In[ ]:


for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


# In[ ]:


Pclass1 = train_df[train_df['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train_df[train_df['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train_df[train_df['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[ ]:


cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


# In[ ]:


# fill missing Fare with median fare for each Pclass
train_df["Cabin"].fillna(train_df.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test_df["Cabin"].fillna(test_df.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


# # FamilySize

# In[ ]:


train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1


# In[ ]:


train_df.loc[train_df['FamilySize'] <= 0.0,'IsAlone'] = 0
train_df.loc[train_df['FamilySize'] > 0.0,'IsAlone'] = 1
test_df.loc[test_df['FamilySize'] <= 0.0,'IsAlone'] = 0
test_df.loc[test_df['FamilySize'] > 0.0,'IsAlone'] = 1


# In[ ]:


train_df.head()


# In[ ]:


facet = sns.FacetGrid(train_df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'FamilySize',shade= True)
facet.set(xlim=(0, train_df['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)


# In[ ]:


family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)


# In[ ]:


# Correlation Matrix Heatmap
f, ax = plt.subplots(figsize=(10, 6))
corr = train_df.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Wine Attributes Correlation Heatmap', fontsize=14)


# In[ ]:


test_df.head()


# In[ ]:


train_df.head()


# In[ ]:


# Pair-wise Scatter Plots
cols = ['Title', 'Sex', 'Cabin', 'Pclass']
pp = sns.pairplot(train_df[cols], size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Titanic Attributes Pairwise Plots', fontsize=14)


# In[ ]:


features_drop = ['Ticket', 'Parch', ]
# features_drop = ['Pclass', 'Fare', 'Cabin', 'Ticket']

train_df = train_df.drop(features_drop, axis=1)
test_df = test_df.drop(features_drop, axis=1)
train_df = train_df.drop(['PassengerId'], axis=1)


# In[ ]:


train_df.head()


# In[ ]:


train_data = train_df.drop('Survived', axis=1)
target = train_df['Survived']

train_data.shape, target.shape


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# # KNN

# In[ ]:


clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


round(np.mean(score)*100, 2)


# # Decision Tree

# In[ ]:


clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


# decision tree Score
round(np.mean(score)*100, 2)


# # Ramdom Forest

# In[ ]:


rand_clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


# Random Forest Score
round(np.mean(score)*100, 2)


# # Naive Bayes

# In[ ]:


clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


# Naive Bayes Score
round(np.mean(score)*100, 2)


# # SVM

# In[ ]:


svm = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


round(np.mean(score)*100,2)


# In[ ]:


svm = SVC()
clf.fit(train_data, target)

test_data = test_df.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)


# In[ ]:


# fit model no training data
model = XGBClassifier()
model.fit(train_data, target)


# In[ ]:


# make predictions for test data
y_pred = clf.predict(test_data)
predictions = [round(value) for value in y_pred]


# In[ ]:


# turn run_gs to True if you want to run the gridsearch again.
run_gs = False

if run_gs:
    parameter_grid = {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [50, 10],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [2, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False],
                 }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation,
                               verbose=1
                              )

    grid_search.fit(train_data, target)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    
else: 
    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
    
    model = RandomForestClassifier(**parameters)
    model.fit(train_data, target)


# In[ ]:


output = model.predict(test_data).astype(int)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": output
    })

submission.to_csv('submission.csv', index=False)


# Thanks ...
