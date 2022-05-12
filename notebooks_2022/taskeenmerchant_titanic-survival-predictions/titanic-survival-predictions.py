#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train=pd.read_csv('../input/machine-learning-on-titanic-data-set/train.csv')
test=pd.read_csv('../input/machine-learning-on-titanic-data-set/test.csv')


# In[ ]:


train.head(5)


# In[ ]:


train.describe(include="all")


# In[ ]:


print(train.columns)


# In[ ]:


train.sample(5)


# In[ ]:


train.sample(5)


# In[ ]:


pd.isnull(train).sum()


# In[ ]:


sns.barplot(x="Sex", y="Survived", data=train)


# In[ ]:


print("Percentage of Females survived :", train['Survived'][train['Sex']=='female'].value_counts(normalize=True)[1]*100)
print("Percentage of Males survived :", train['Survived'][train['Sex']=='male'].value_counts(normalize=True)[1]*100)


# In[ ]:


sns.barplot(x="SibSp", y="Survived", data=train)


# In[ ]:


sns.barplot(x="Pclass", y="Survived", data=train)


# In[ ]:


sns.barplot(x="Parch", y="Survived", data=train)


# In[ ]:


train["Age"]=train["Age"].fillna(-0.5)
test["Age"]=test["Age"].fillna(-0.5)
bins=[-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels=['Unknown', 'Baby', 'Child', 'teenage', 'Student', 'Young adult', 'Adult', 'Senior']
train["AgeGroup"]=pd.cut(train["Age"], bins, labels=labels)
test["AgeGroup"]=pd.cut(train["Age"], bins, labels=labels)
sns.barplot(x="AgeGroup", y="Survived", data=train)


# In[ ]:


train["Cabinpool"]=(train["Cabin"].notnull().astype('int'))
test["Cabinpool"]=(test["Cabin"].notnull().astype('int'))
sns.barplot(x="Cabinpool", y="Survived", data=train)


# In[ ]:


train.describe(include="all")


# In[ ]:


train=train.drop(['Cabin'], axis=1)
test=test.drop(['Cabin'], axis=1)


# In[ ]:


train=train.drop(['Ticket'], axis=1)
test=test.drop(['Ticket'], axis=1)


# In[ ]:


print("No of passenger from south")
south=train[train["Embarked"]=="S"].shape[0]
print(south)
print("No of passenger from Chenn")
chenn=train[train["Embarked"]=="C"].shape[0]
print(chenn)
print("No of passenger from Queensland")
queen=train[train["Embarked"]=="Q"].shape[0]
print(queen)


# In[ ]:


train=train.fillna({"Embarked":"S"})
test=test.fillna({"Embarked":"S"})


# In[ ]:


combine=[train,test]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
pd.crosstab(train['Title'], train['Sex'])


# In[ ]:


for dataset in combine:
    dataset['Title']=dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
print(dataset['Title'])


# In[ ]:


train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


title_mapping={"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title']=dataset['Title'].map(title_mapping)
    dataset['Title']=dataset['Title'].fillna(0)


# In[ ]:


train.head(5)


# In[ ]:



age_title_mapping={1:"Young adult", 2:"Student", 3:"Adult", 4:"Baby", 5:"Adult",6:"Adult"}
for x in range(len(train["AgeGroup"])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]
for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]
train.head(205)


# In[ ]:


age_mapping={'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young adult': 5, 'Adult': 6, 'Senior': 7}
train["AgeGroup"]=train["AgeGroup"].map(age_mapping)
test["AgeGroup"]=test["AgeGroup"].map(age_mapping)

train=train.drop(['Age'], axis=1)
test=test.drop(['Age'], axis=1)
train.head(5)


# In[ ]:


train=train.drop(['Name'], axis=1)
test=test.drop(['Name'], axis=1)


# In[ ]:


sex_mapping={"male":0, "female":1}
train['Sex']=train["Sex"].map(sex_mapping)
test['Sex']=test["Sex"].map(sex_mapping)
train.head(3)


# In[ ]:


embark_mapping={"S":1, "C":2, "Q":3}
train['Embarked']=train["Embarked"].map(embark_mapping)
test['Embarked']=test["Embarked"].map(embark_mapping)
train.head(3)


# In[ ]:


for x in range(len(train["Fare"])):
    if pd.isnull(train["Fare"][x]):
        pclass=(train["Pclass"][x])
        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)
train["FareBand"]=pd.qcut(train["Fare"],4,labels=[1,2,3,4])
test["FareBand"]=pd.qcut(test["Fare"],4,labels=[1,2,3,4])
        
train=train.drop(["Fare"], axis=1) 
test=test.drop(["Fare"], axis=1)     
       


# In[ ]:


train.head(496)


# In[ ]:


test.head()


# In[ ]:


train['AgeGroup']=train['AgeGroup'].fillna(0)
train.head()


# In[ ]:


train["AgeGroup"].isnull().sum()


# In[ ]:


from sklearn.model_selection import train_test_split
predictors=train.drop(["Survived", "PassengerId"], axis=1)
target=train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size=0.22, random_state=0)


# In[ ]:


print(x_train)


# In[ ]:


test.head()


# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train, y_train)
y_pred=logreg.predict(x_val)
acc_logreg=round(accuracy_score(y_pred, y_val)*100,2)
print(acc_logreg)


# In[ ]:


from sklearn.svm import SVC
svms=SVC()
svms.fit(x_train, y_train)
y_pred=svms.predict(x_val)
acc_svm=round(accuracy_score(y_pred, y_val)*100,2)
print(acc_svm)


# In[ ]:


from sklearn.svm import LinearSVC
lsvm=LinearSVC()
lsvm.fit(x_train, y_train)
y_pred=lsvm.predict(x_val)
acc_lsvm=round(accuracy_score(y_pred, y_val)*100,2)
print(acc_lsvm)


# In[ ]:


from sklearn.linear_model import Perceptron
per=Perceptron()
per.fit(x_train, y_train)
y_pred=per.predict(x_val)
acc_per=round(accuracy_score(y_pred, y_val)*100,2)
print(acc_per)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(x_train, y_train)
y_pred=dtc.predict(x_val)
acc_dtc=round(accuracy_score(y_pred, y_val)*100,2)
print(acc_dtc)


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
grad=GradientBoostingClassifier()
grad.fit(x_train, y_train)
y_pred=grad.predict(x_val)
acc_grad=round(accuracy_score(y_pred, y_val)*100,2)
print(acc_grad)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
random=RandomForestClassifier()
random.fit(x_train,y_train)
y_pred=dtc.predict(x_val)
acc_random=round(accuracy_score(y_pred, y_val)*100,2)
print(acc_random)


# In[ ]:


models=pd.DataFrame({'Models':["Gaussian","Logistic Regression", "Support Vector Machine", "Linear Support Vector Mahcine", "Perceptron",
                              "Decision Tree Classifier", "Random Forest Classifier", "Gradient Boosting Classifier"],
                    'Score': [acc_gaussian, acc_logreg, acc_svm, acc_lsvm, acc_per, acc_dtc, acc_random, acc_grad]})
models.sort_values(by="Score", ascending=False)


# In[ ]:


test['AgeGroup']=test['AgeGroup'].fillna(0)
test=test.dropna(axis=0, subset=["FareBand"])
test.head()


# In[ ]:


test.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


ids = test['PassengerId']
predictions = grad.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
print(output)


# In[ ]:


output.to_csv("Submission.csv", index=False)


# In[ ]:




