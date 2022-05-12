#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# ### **Load Datasets**

# In[ ]:


train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
both_df = [train_data, test_data]


# In[ ]:


train_data.head()


# ### **Exploratory analysis**

# #### Missing data

# Cabin, Age, and Embarked columns have missing values in the training data. <br>
# Cabin,and Age have missing values in the testing data. 

# In[ ]:


print('TRAIN data:\n', train_data.isnull().sum())
print('\n')

print('TEST data:\n', test_data.isnull().sum())


# **impute missing embarked values with mode of train data, and missign fare value with median. <br>
# We will remove cabin column because most of it is missing**
# 

# In[ ]:


data_corr = train_data.corr(method='pearson').abs().stack().reset_index().sort_values(by=[0], ascending=False)
data_corr[data_corr['level_0'] == 'Age']


# In[ ]:


print('median age of pclass 1 male: ' + str(train_data[(train_data['Pclass'] == 1)&(train_data['Sex'] == 'male')]['Age'].median()))
print('median age of pclass 1 female: ' + str(train_data[(train_data['Pclass'] == 1)&(train_data['Sex'] == 'female')]['Age'].median()))

print('median age of pclass 2 male: ' + str(train_data[(train_data['Pclass'] == 2)&(train_data['Sex'] == 'male')]['Age'].median()))
print('median age of pclass 2 female: ' + str(train_data[(train_data['Pclass'] == 2)&(train_data['Sex'] == 'female')]['Age'].median()))

print('median age of pclass 3 male: ' + str(train_data[(train_data['Pclass'] == 3)&(train_data['Sex'] == 'male')]['Age'].median()))
print('median age of pclass 3 female: ' + str(train_data[(train_data['Pclass'] == 3)&(train_data['Sex'] == 'female')]['Age'].median()))

#impute age 
for df in both_df:
    df['Age'] = train_data.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))


# In[ ]:


#impute  
for df in both_df:
    df['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace = True)
    
    df['Fare'].fillna(train_data['Fare'].median(), inplace = True)
    
    del df['Cabin']


# In[ ]:


for df in both_df:
    print(df.isnull().sum())


# In[ ]:


cols = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

fig, axs = plt.subplots(figsize=(15, 10))

for i, feature in enumerate(cols, 1):    
    plt.subplot(2, 3, i)
    sns.countplot(x=feature, hue='Survived', data=train_data)


# In[ ]:


plt.hist(x=train_data['Fare'], bins='auto', color='#0504aa')
plt.show()


# **Feature Engineering**

# In[ ]:


for df in both_df:
    # get title only 
    df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
#     # create fare and age bin
    df['Fare_bin'] = pd.qcut(df['Fare'], 7)
    df['Age_bin'] = pd.qcut(df['Age'], 5)
    df["Family_size"] = df["SibSp"] + df["Parch"] + 1
    df.loc[df['Family_size']-1 > 0, 'travelled_alone'] = 1
    df.loc[df['Family_size']-1 == 0, 'travelled_alone'] = 0
    


# In[ ]:


#might need grouping
plt.subplots(figsize=(14, 4))
sns.countplot(x='Title', data=train_data)


# In[ ]:


for df in both_df:
    #group the titles
    df["Title"] = df["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df["Title"] = df["Title"].replace([ 'Mrs', 'Miss', "Mme", "Ms", "Mlle"], 'Miss/Mrs')


# In[ ]:


sns.countplot(x='Title', data=train_data)


# In[ ]:


plt.subplots(figsize=(14, 4))
sns.countplot(x='Fare_bin', hue='Survived', data=train_data)


# In[ ]:


sns.countplot(x='Age_bin', hue='Survived', data=train_data)


# In[ ]:


plt.hist(x=train_data['Family_size'], bins='auto', color='#0504aa')
plt.show()


# In[ ]:


sns.countplot(x='travelled_alone', data=train_data)


# In[ ]:


train_data.info()


# In[ ]:


for df in both_df:
    # Mapping Sex
    df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    #mapping embarked
    df['Embarked'] = df['Embarked'].map({"S": 0, "C": 1, "Q": 2}).astype(int)
    #mapping title
    df['Title'] = df['Title'].map({"Mr": 1, "Miss/Mrs": 2, "Master": 3, "Rare": 4})
    
    # Mapping Fare
    df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
    df.loc[ df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)
    
    # Mapping Age
    df.loc[ df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[ df['Age'] > 64, 'Age']  
    
    # Drop Name 
    df.drop(labels = ["Name", 'Ticket', 'Fare_bin', 'Age_bin'], axis = 1, inplace = True)


# In[ ]:


train_data.head()


# In[ ]:


y_train  = train_data["Survived"]
x_train = train_data.drop(labels = ["Survived", "PassengerId"],axis = 1)


# In[ ]:


from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

classifiers = [
    KNeighborsClassifier(),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(random_state=0),
    LogisticRegression(solver='liblinear', random_state=0)]

cv_results = []

for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, x_train, y = y_train, scoring = "accuracy", cv = StratifiedKFold(n_splits=10), n_jobs=4).mean())


# In[ ]:


classifiers_ = [str(x) for x in classifiers] 

CV_df = pd.DataFrame({"CrossValMeans":cv_results,"Algorithm": classifiers_})
CV_df


# In[ ]:


sns.barplot(x='CrossValMeans', y='Algorithm', data=CV_df)


# In[ ]:


model = SVC(probability=True)
model.fit(x_train, y_train)

y_pred = model.predict(test_data.drop('PassengerId', axis=1))

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:




