#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import needed moduls
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Load data
titanic_train = pd.read_csv("../input/titanic/train.csv", index_col='PassengerId')
titanic_test = pd.read_csv("../input/titanic/test.csv", index_col='PassengerId')
titanic_train.head()


# In[ ]:


# Delete no needed (for my opinion) columns
titanic_train.drop(labels=["Ticket", "Cabin"], axis=1, inplace=True)
titanic_test.drop(labels=["Ticket", "Cabin"], axis=1, inplace=True)
titanic_train.head()


# In[ ]:


# Check null values
titanic_train.info()


# In[ ]:


# Fill 'Age' column nones by median value
titanic_train["Age"].fillna(titanic_train["Age"].median(), inplace=True)
titanic_train["Fare"].fillna(titanic_train["Fare"].median(), inplace=True)
titanic_train["Embarked"].fillna(titanic_train["Embarked"].mode()[0], inplace=True)

titanic_test["Age"].fillna(titanic_test["Age"].median(), inplace=True)
titanic_test["Fare"].fillna(titanic_test["Fare"].median(), inplace=True)
titanic_test["Embarked"].fillna(titanic_test["Embarked"].mode()[0], inplace=True)
titanic_train.head()


# In[ ]:


# Create new 'Age' column for every age range
for i in range(0, int(titanic_train["Age"].max()//16)):
    titanic_train[f"Age{i}"] = titanic_train["Age"].map(lambda x: int(x >= 16*i and x < 16*(i+1)))
    titanic_test[f"Age{i}"] = titanic_test["Age"].map(lambda x: int(x >= 16*i and x < 16*(i+1)))
titanic_train.head()


# In[ ]:


# Check Fare distribution 
sns.boxplot(x=titanic_train['Fare']);


# In[ ]:


# Create 'Rich' column
titanic_train["Rich"] = titanic_train["Fare"].map(lambda x: int(x >= 80))
titanic_test["Rich"] = titanic_test["Fare"].map(lambda x: int(x >= 80))

# Create new 'Fare' column for every Fare range
for i in range(0, 6):
    titanic_train[f"Fare{i}"] = titanic_train["Fare"].map(lambda x: int(x >= 20*i and x < 20*(i+1)) if x < 80 else 0)
    titanic_test[f"Fare{i}"] = titanic_test["Fare"].map(lambda x: int(x >= 20*i and x < 20*(i+1)) if x < 80 else 0)
titanic_train.head()

# Delete not needed columns
titanic_train.drop(["Fare", "Age"], axis=1, inplace=True)
titanic_test.drop(["Fare", "Age"], axis=1, inplace=True)

titanic_train.head()


# In[ ]:


# Converting categorical string feature to binominal integer
titanic_train["Sex"] = titanic_train["Sex"].apply(lambda x: 1 if x=="male" else 0)
titanic_test["Sex"] = titanic_test["Sex"].apply(lambda x: 1 if x=="male" else 0)
titanic_train.head()


# In[ ]:


# Create new 'Embarked' column for every Embarked place
titanic_train = pd.concat([titanic_train, 
                           pd.get_dummies(titanic_train["Embarked"], prefix='Embarked')],
                           axis=1)
titanic_test = pd.concat([titanic_test, 
                           pd.get_dummies(titanic_test["Embarked"], prefix='Embarked')],
                           axis=1)

titanic_train.drop(["Embarked"], axis=1, inplace=True)
titanic_test.drop(["Embarked"], axis=1, inplace=True)


# In[ ]:


titanic_train.head()


# In[ ]:


# Create 'IsAlone' column
titanic_train['IsAlone'] = ((titanic_train['SibSp']==0) & (titanic_train['Parch']==0))
titanic_test['IsAlone'] = ((titanic_test['SibSp']==0) & (titanic_test['Parch']==0)).astype('int')

# Drop not needed feature
titanic_train = titanic_train.drop(['Parch', 'SibSp'], axis=1)
titanic_test = titanic_test.drop(['Parch', 'SibSp'], axis=1)

# Check survived depending on 'IsAlone' 
titanic_train['IsAlone'].head()
pd.crosstab(titanic_train['IsAlone'], 
            titanic_train['Survived']).plot(kind="bar", figsize=(10,6), color=["salmon", "lightblue"]);

# Check part of survived depending on 'IsAlone' 
print(titanic_train.groupby(['IsAlone'])['Survived'].mean()*100)

# Keep the labels on the x-axis vertical
plt.xticks(rotation=0); 


# In[ ]:


# Find title of each person 
titanic_train['Title'] = titanic_train['Name'].str.extract('([A-Za-z]+)\.')
titanic_train['Title'].value_counts()


# In[ ]:


# Bug fixes (maybe)
titanic_train['Title'] = titanic_train['Title'].replace(['Mlle','Ms'], 'Miss')
titanic_train['Title'] = titanic_train['Title'].replace('Mme', 'Mrs')

titanic_train['Title'].value_counts()


# In[ ]:


# Replace rare (<= 7) title to 'Rare'
titanic_train['Title'] = titanic_train['Title'].replace(['Lady', 'Countess','Capt', 'Col',                                                         'Don', 'Dr', 'Major', 'Rev', 'Sir',                                                         'Jonkheer', 'Dona'], 'Rare')

titanic_train['Title'].value_counts()


# In[ ]:


# Convert to numbers and fill missing
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
titanic_train['Title'] = titanic_train['Title'].map(title_mapping)
titanic_train['Title'] = titanic_train['Title'].fillna(0)

# Drop not needed feature
titanic_train = titanic_train.drop('Name', axis=1)


titanic_train.head()


# In[ ]:


# Same process for test df
titanic_test['Title'] = titanic_test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

titanic_test['Title'] = titanic_test['Title'].replace(['Mlle','Ms'], 'Miss')
titanic_test['Title'] = titanic_test['Title'].replace('Mme', 'Mrs')
    
titanic_test['Title'] = titanic_test['Title'].replace(['Lady', 'Countess','Capt', 'Col',                                                         'Don', 'Dr', 'Major', 'Rev', 'Sir',                                                         'Jonkheer', 'Dona'], 'Rare')

titanic_test['Title'] = titanic_test['Title'].map(title_mapping)
titanic_test['Title'] = titanic_test['Title'].fillna(0)

titanic_test = titanic_test.drop('Name', axis=1)


# In[ ]:


# Splitting the dataset on training and validating
from sklearn.model_selection import train_test_split

X = titanic_train.drop("Survived", axis=1)
y = titanic_train["Survived"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# Import different model
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# In[ ]:


# Create dictionary of models
models = {"LinearSVC": LinearSVC(),
          "KNN": KNeighborsClassifier(),
          "SVC": SVC(),
          "LogisticRegression": LogisticRegression(),
          "RandomForestClassifier": RandomForestClassifier(),
          "XGBoost": xgb.XGBClassifier(use_label_encoder=False)}


# Create an empty dictionary for results
results = {}


# In[ ]:


# Fit and score each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    results[model_name] = model.score(X_val, y_val)

# View the results
results


# In[ ]:


# Tune hyperparams of KNN
# Create dict for KNN params
param_knn = {'n_neighbors': [5, 7, 8, 9, 10, 12, 15, 20],
             'weights': ['uniform', 'distance'],
             'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

# Setup the grid search
grid_knn = GridSearchCV(KNeighborsClassifier(),
                        param_knn,
                        cv=5)

# Fit the grid search to the data
grid_knn.fit(X_train, y_train)

# Find the best parameters
grid_knn.best_params_, grid_knn.best_score_


# In[ ]:


# Check score on validating data
grid_knn.best_estimator_.score(X_val, y_val)


# In[ ]:


# Check all metrics on full data
print(f"Accuracy {np.mean(cross_val_score(grid_knn.best_estimator_, X, y, cv=5, scoring='accuracy'))}")
print(f"Recall {np.mean(cross_val_score(grid_knn.best_estimator_, X, y, cv=5, scoring='recall'))}")
print(f"Precision {np.mean(cross_val_score(grid_knn.best_estimator_, X, y, cv=5, scoring='precision'))}")
print(f"F1 {np.mean(cross_val_score(grid_knn.best_estimator_, X, y, cv=5, scoring='f1'))}")


# In[ ]:


# Tune hyperparams of RandomForestClassifier
# Create dict for RandomForestClassifier params
param_RFC = {'n_estimators': [i for i in range(1, 101, 5)],
             'max_depth': [i for i in range(1, 31, 3)]}

# Setup the grid search
grid_RFC = GridSearchCV(RandomForestClassifier(random_state=42),
                        param_RFC,
                        cv=5)

# Fit the grid search to the data
grid_RFC.fit(X_train, y_train)

# Find the best parameters
grid_RFC.best_params_, grid_RFC.best_score_


# In[ ]:


# Check score on validating data
grid_RFC.best_estimator_.score(X_val, y_val)


# In[ ]:


# Check all metrics on full data
print(f"Accuracy {np.mean(cross_val_score(grid_RFC.best_estimator_, X, y, cv=5, scoring='accuracy'))}")
print(f"Recall {np.mean(cross_val_score(grid_RFC.best_estimator_, X, y, cv=5, scoring='recall'))}")
print(f"Precision {np.mean(cross_val_score(grid_RFC.best_estimator_, X, y, cv=5, scoring='precision'))}")
print(f"F1 {np.mean(cross_val_score(grid_RFC.best_estimator_, X, y, cv=5, scoring='f1'))}")


# In[ ]:


# Tune hyperparams of XGBClassifier
# Create dict for XGBClassifier params
param_xgb = {
    "max_depth": [3, 5, 9],
    "learning_rate": [0.01, 0.1, 0.3, 1],
    "gamma": [0,  1, 3],
    "reg_alpha": [0, 1, 10],
    "reg_lambda": [0, 1, 10],
    "scale_pos_weight": [1, 5]
}

# Setup the grid search
grid_xgb = GridSearchCV(xgb.XGBClassifier(random_state=42, use_label_encoder=False), 
                        param_xgb,
                        cv=5)

# Fit the grid search to the data
grid_xgb.fit(X_train, y_train)

# Find the best parameters
grid_xgb.best_params_, grid_xgb.best_score_


# In[ ]:


# Check score on validating data
grid_xgb.best_estimator_.score(X_val, y_val)


# In[ ]:


# Check all metrics on full data
print(f"Accuracy {np.mean(cross_val_score(grid_xgb.best_estimator_, X, y, cv=5, scoring='accuracy'))}")
print(f"Recall {np.mean(cross_val_score(grid_xgb.best_estimator_, X, y, cv=5, scoring='recall'))}")
print(f"Precision {np.mean(cross_val_score(grid_xgb.best_estimator_, X, y, cv=5, scoring='precision'))}")
print(f"F1 {np.mean(cross_val_score(grid_xgb.best_estimator_, X, y, cv=5, scoring='f1'))}")


# In[ ]:


# Check right shape of test data
X_train.shape, titanic_test.shape


# In[ ]:


# Predict on test data
preds = grid_xgb.best_estimator_.predict(titanic_test)

