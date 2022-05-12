#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# # **IMPORTING LIBRARIES AND LOADING DATASET**

# **IMPORTING LIBRARIES**

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree,svm
from sklearn.metrics import accuracy_score


# **LOADING DATASET**

# In[ ]:


train_data = pd.read_csv("../input/titanic/train.csv")
test_data = pd.read_csv("../input/titanic/test.csv")


# In[ ]:


#Show 5 rows of the dataset
train_data.head(10)


# In[ ]:


#Info about data

train_data.info()


# # **EXPLORATORY DATA ANALYSIS**

# Now we will analyze our data to see which variables are actually important to predict the value of the target variable.

# In[ ]:


plt.figure(figsize=(10,6))

heatmap = sns.heatmap(train_data[["Survived", "SibSp", "Parch", "Age", "Fare"]].corr(), annot = True)
sns.set(rc={'figure.figsize':(12,10)})


# Moving on, now we will understand all the features one by one. We’ll visualize the impact of each feature on the target variable. Let us start with SibSp that is the no. of siblings or spouses a passenger has.

# **SibSp – Number of Siblings / Spouses aboard the Titanic**

# In[ ]:


# Find unique values

train_data['SibSp'].unique()


# In[ ]:



bargraph_sibsp = sns.catplot(x = "SibSp", y = "Survived", data = train_data, kind="bar", height = 8)


# Passengers having 1 or 2 siblings have good chances of survival
# More no. of siblings -> Fewer chances of survival

# **AGE COLOMN**

# We can plot a graph so as to see the distribution of age with respect to target variable.

# In[ ]:



 age = sns.FacetGrid(train_data, col="Survived", height = 7)
age = age.map(sns.histplot, "Age")
age = age.set_ylabels("Survival Probability")


#  We can see more age -> less chances of survival!

# **GENDER OF COLOMN**

# For gender we are simply going to use seaborn and will plot a bar graph.

# In[ ]:


sns.barplot(x="Sex", y="Survived", data=train_data)


# We can see from the above graph it’s quite obvious to say that man has less chances of survival over females. 

# **PCLASS COLOMN**

# Let us now see whether the class plays any role in survival probability or not.

# In[ ]:


pclass = sns.catplot(x = "Pclass", y="Survived", data = train_data, kind="bar", height = 7)


# So we can see a first class passenger has more chances of survival over 2nd and 3rd class passengers & Similarly the 2nd class passengers have more chances of survival over 3rd class passengers.
# 

# # **DATA PREPROCESSING**

# In[ ]:


#Check null values

train_data.isnull().sum()


# We can see there are 177 missing entries in Age column. 687 missing entries are in Cabin column and 2 missing are in Embarked.

# **HANDLE MISSING VALUES OF AGE COLUMN**

# In[ ]:


mean = train_data["Age"].mean()
std = train_data["Age"].std()

rand_age = np.random.randint(mean-std, mean+std, size = 177)
age_slice = train_data["Age"].copy()
age_slice[np.isnan(age_slice)] = rand_age
train_data["Age"] = age_slice

train_data["Embarked"].fillna(value="C", inplace=True)


# Again checking for null values
train_data.isnull().sum()


# We can see, we don't have missing values

# **DROP COLUMN**

# In[ ]:


col_to_drop = ["PassengerId", "Ticket", "Cabin", "Name"]
train_data.drop(col_to_drop, axis=1, inplace=True)
train_data.head(10)


# **CONVERTING CATEGORICAL VARIABLES TO NUMERIC**

# In[ ]:


genders = {"male":0, "female":1}
train_data["Sex"] = train_data["Sex"].map(genders)

ports = {"S":0, "C":1, "Q":2}
train_data["Embarked"] = train_data["Embarked"].map(ports)

train_data.head()


# # **BUILDING MACHINE LEARNING MODEL**

# So, this was all about data preprocessing. Now we are good to go with our titanic dataset. Let’s quickly train our machine learning model.

# In[ ]:


df_train_x = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# Target variable column
df_train_y = train_data['Survived']

# Train Test Splitting
x_train, x_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.20, random_state=42)


# **Lastly, We are going to fit our model on 5 different classification algorithms namely RANDOM FOREST CLASSIFIER, LOGISTIC REGRESSION, K-NEIGHBOR CLASSIFIER, DECISSION TREE CLASSIFIER, and SUPPORT VECTOR MACHINE. And eventually will compare them.**

# **RANDOM FOREST**

# In[ ]:


# Creating alias for Classifier
model1 = RandomForestClassifier()

# Fitting the model using training data
model1 = model1.fit(x_train, y_train)

# Predicting on test data
rfc_y_pred = model1.predict(x_test)

# Calculating Accuracy to compare all models
rfc_accuracy = accuracy_score(y_test,rfc_y_pred) * 100
print("accuracy=",rfc_accuracy)


# **LOGISTIC REGRESSION**

# In[ ]:


model2 = LogisticRegression( max_iter=2000 )
model2 = model2.fit(x_train, y_train)
lr_y_pred = model2.predict(x_test)
lr_accuracy = accuracy_score(y_test,lr_y_pred)*100

print("accuracy=",lr_accuracy)


# **K-NEIGHBOR CLASSIFIER**

# In[ ]:


model3 = KNeighborsClassifier(5)
model3 = model3.fit(x_train, y_train)
knc_y_pred = model3.predict(x_test)
knc_accuracy = accuracy_score(y_test,knc_y_pred)*100

print("accuracy=",knc_accuracy)


# **DECISSION TREE CLASSIFIER**

# In[ ]:


model4 = tree.DecisionTreeClassifier()
model4 = model4.fit(x_train, y_train)
dtc_y_pred = model4.predict(x_test)
dtc_accuracy = accuracy_score(y_test,dtc_y_pred)*100

print("accuracy=",dtc_accuracy)


# **SUPPORT VECTOR MACHINE**

# In[ ]:


model5 = svm.SVC()
model5 = model5.fit(x_train, y_train)
svm_y_pred = model5.predict(x_test)
svm_accuracy = accuracy_score(y_test,svm_y_pred)*100
print("accuracy=",svm_accuracy)


# **ACCURACY SCORES OF All CLASSIFIERS**

# In[ ]:


print("Accuracy of RANDOM FOREST CLASSIFIER =",rfc_accuracy)
print("Accuracy of LOGISTIC REGRESSION =",lr_accuracy)
print("Accuracy of K-NEIGHBOR CLASSIFIER =",knc_accuracy)
print("Accuracy of DECISION TREE CLASSIFIER = ",dtc_accuracy)
print("Accuracy of SUPPORT VECTOR MACHINE = ",svm_accuracy)


# Subsequently, we can now rank our evaluation of all the models to choose the best one for our problem. While Random Forest is best.

# # **FINAL PREDICTION WITH MACHINE LEARNING MODEL**

# So, now it’s time to use test.csv for making predictions. For testing data also we need to do the steps of preprocessing that we did earlier. And then only we can predict whether a passenger will survive or not. Hence, I highly encourage you to do all the things for test.csv by yourself.

# In[ ]:


test_data.head(10)


# In[ ]:


test_data.info()


# In[ ]:


test_data.isnull().sum()


# In[ ]:


# Replacing missing values of age column
mean = test_data["Age"].mean()
std = test_data["Age"].std()
rand_age = np.random.randint(mean-std, mean+std, size = 86)
age_slice = test_data["Age"].copy()
age_slice[np.isnan(age_slice)] = rand_age
test_data["Age"] = age_slice

# Replacing missing value of Fare column
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

test_data.isnull().sum()


# In[ ]:


col_to_drop = ["PassengerId", "Ticket", "Cabin", "Name"]
test_data.drop(col_to_drop, axis=1, inplace=True)
test_data.head(10)


# In[ ]:


genders = {"male":0, "female":1}
test_data["Sex"] = test_data["Sex"].map(genders)

ports = {"S":0, "C":1, "Q":2}
test_data["Embarked"] = test_data["Embarked"].map(ports)

test_data.head()


# # **MACHINE LEARNINIG PROJECT SUBMISSION**

# In[ ]:


x_test = test_data
y_pred = model1.predict(x_test)
originaltest_data = pd.read_csv('/kaggle/input/titanic/test.csv')

submission = pd.DataFrame({
        "PassengerId": originaltest_data["PassengerId"],
        "Survived": y_pred
    })
submission.head(20)


# **Thankyou! If you like this article leave a comment “Nice article!” to motivate me. Keep learning, keep coding!**
