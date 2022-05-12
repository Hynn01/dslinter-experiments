#!/usr/bin/env python
# coding: utf-8

# # Introduction:
# *        RMS Titanic was a British passenger liner operated by the White Star Line that sank in the North Atlantic Ocean in the early morning hours of 15 April 1912, after striking an iceberg during her maiden voyage from Southampton to New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making the sinking one of modern history's deadliest peacetime commercial marine disasters. RMS Titanic was the largest ship afloat at the time she entered service and was the second of three Olympic-class ocean liners operated by the White Star Line. She was built by the Harland and Wolff shipyard in Belfast. Thomas Andrews, chief naval architect of the shipyard at the time, died in the disaster.(Wikipedia)
# 
# <font color = "blue">
# 
# # Content:
# 
# * [1. Load and Check Data](#1)
# * [2. Variable Description](#2)
#   * [2.1. Univariate Variable Analysis](#3)
#   * [2.2. Categorical Variable Analysis](#4)
#   * [2.3. Numerical Variable Analysis](#5)
# * [3. Basic Data Analysis](#6)
# * [4. Outlier Detection](#7)
# * [5. Missing Value](#8)
#   * [5.1. Find Missing Value](#9)
#   * [5.2. Fill Missing Value](#10)
# * [6. Visualization](#11)
#   * [6.1. Correlation Between SibSp - Parch - Age - Fare - Survived](#12)
#   * [6.2. SibSp - Survived](#13)
#   * [6.3. Parch - Survived](#14)
#   * [6.4. Pclass - Survived](#15)
#   * [6.5. Age - Survived](#16)
#   * [6.6. Pclass - Survived - Age](#17)
#   * [6.7. Embarked - Sex - Pclass - Survived](#18)
#   * [6.8. Embarked - Sex - Fare - Survived](#19)
# * [7. Fill Missing: Age Feature](#20)
# * [8. Feature Engineering](#21)
#   * [8.1. Name -Title](#22)
#   * [8.2. Family Size](#23)
#   * [8.3. Embarked](#24)
#   * [8.4. Ticket](#25)
#   * [8.5. Pclass](#26)
#   * [8.6. Sex](#27)
#   * [8.7. Drop Passenger ID and Cabin](#28)
# * [9. Modelling](#29)
#   * [9.1. Train Test Split](#30)
#   * [9.2. Simple Logistic Regression](#31)
#   * [9.3. Hyperparameter Tuning - Grid Search - Cross Validation](#32)
#   * [9.4. Ensemble Modelling](#33)
# * [10. Prediction and Submission](#34)  

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import seaborn as sns
from collections import Counter



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <a id = "1"></a><br>
# ## 1. Load and Check Data:
# 

# In[ ]:


train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_passengerID = test_df["PassengerId"]


# In[ ]:


train_df.columns


# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# <a id = "2"></a><br>
# ## 2. Variable Description
# 
# 1. PassengerId: Unique id munber to each passenger.
# 1. Survived: Passenger survided (1) or died (0).
# 1. Pclass: Passenger class.
# 1. Name: Name of passenger.
# 1. Sex: Gender of passenger.
# 1. Age: Age of passenger.
# 1. SibSp: Number of siblings/spouses.
# 1. Parch: Number of parents/children.
# 1. Ticket: Ticket number.
# 1. Fare: Amount of money spent on ticket.
# 1. Cabin: Cabin category.
# 1. Embarked: Port where passenger embarked (C = Cherbourg, Q = Queenstown, S = Southhampton)
#     
#     

# In[ ]:


train_df.info()


# * float64(2): Fare, age.
# * int64(5): Pclass, sibsp, parch, passengerid, survived.
# * objects(5): Cabin, embarked, ticket, name, sex.

# <a id = "3"></a><br>
# ## 2.1. Univariate Variable Analysis
# * Categorical Variable Analysis: Survived, sex, pclass, embarked, cabin, name, ticket, sibsp, parch.
# * Numerical Variable Analysis: Age, passengerid, fare.

# <a id = "4"></a><br>
# ## 2.2. Categorical Variable Analysis:
# 

# In[ ]:


def bar_plot(variable):
    #___
    #    input: variable ex:"sex"
    #    output:bar plot & value count
    #___
    
    # get a feature
    var = train_df[variable]
    # count number of categorical variable(value/sample)
    varValue = var.value_counts()
    
    #visualize
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}:\n {}".format(variable,varValue))


# In[ ]:


category1 = ["Survived", "Sex", "Pclass", "Embarked", "SibSp", "Parch"]
for c in category1:
    bar_plot(c)


# In[ ]:


category2 = ["Cabin", "Name", "Ticket"]
for c in category2:
    print ("{} \n".format(train_df[c].value_counts()))


# <a id = "5"></a><br>
# ## 2.3. Numerical Variable Analysis:

# In[ ]:


def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train_df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()


# In[ ]:


numericVar = ["Fare", "Age","PassengerId"]
for n in numericVar:
    plot_hist(n)


# <a id = "6"></a><br>
# # 3. Basic Data Analysis
# * Pclass - Survived
# * Sex - Survived
# * SibSp - Survived
# * Parch - Survived

# In[ ]:


# Pclass vs Survived
train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by = "Survived", ascending = False)


# In[ ]:


# Sex vs Survived
train_df[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by = "Survived", ascending = False)


# In[ ]:


# Sibsp vs Survived
train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by = "Survived", ascending = False)


# In[ ]:


# Parch vs Survived
train_df[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by = "Survived", ascending = False)


# <a id = "7"></a><br>
# # 4. Outlier Detection

# In[ ]:


def detect_outlier(df, features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        # IQR
        IQR = Q3 - Q1
        # outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indices
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indices
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers


# In[ ]:


train_df.loc[detect_outlier(train_df,["Age", "SibSp", "Parch", "Fare"])]


# In[ ]:


#drop outliers
train_df = train_df.drop(detect_outlier(train_df,["Age", "SibSp", "Parch", "Fare"]), axis = 0).reset_index(drop =True)


# <a id = "8"></a><br>
# # 5. Missing Value
#    *   [Find Missing Value]
#    *   [Fill Missing Value]

# In[ ]:


train_df_len = len(train_df)
train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop = True)


# In[ ]:


train_df.head()


# <a id = "9"></a><br>
# ## 5.1. Find Missing Value

# In[ ]:


train_df.columns[train_df.isnull().any()]


# In[ ]:


train_df.isnull().sum()


# <a id = "10"></a><br>
# ## 5.2. Fill Missing Value
# 
# * embarked has 2 missing value.
# * Fare has 1 missing value.

# In[ ]:


train_df[train_df["Embarked"].isnull()]


# In[ ]:


train_df.boxplot(column="Fare",by = "Embarked")
plt.show()


# In[ ]:


train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()]


# In[ ]:


train_df[train_df["Fare"].isnull()]


# In[ ]:


train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))


# In[ ]:


train_df[train_df["Fare"].isnull()]


# <a id = "11"></a><br>
# # 6. Visualization
# 

# <a id = "12"></a><br>
# ## 6.1. Correlation Between SibSp - Parch - Age - Fare - Survived

# In[ ]:


list1 =["SibSp", "Parch", "Age", "Fare", "Survived"]
sns.heatmap(train_df[list1].corr(), annot = True, fmt = ".2f")
plt.show()


# Fare feature seems to have correiation with survived feature (0.26).

# <a id = "13"></a><br>
# ## 6.2. SibSp - Survived

# In[ ]:


g = sns.catplot(x = "SibSp", y = "Survived", data = train_df, kind = "bar", height = 6)
g.set_ylabels("Survived Probability")
plt.show()


# * Having a lot of SibSp have less chance to survive.
# * If SibSp = 0 or 0r 2 passenger has more chance to survive.
# * We can consider a new feature describing these categories.

# <a id = "14"></a><br>
# ## 6.3. Parch - Survived

# In[ ]:


g = sns.factorplot(x = "Parch", y = "Survived", data = train_df, kind = "bar", size = 6)
g.set_ylabels("Survived Probability")
plt.show()


# * SibSp and Parch can be used for new feature extraction with treshold = 3.
# * Small  families have more chance to survive.
# * There is astandard deviation in survivel of passenger with parch = 3.

# <a id = "15"></a><br>
# ## 6.4. Pclass - Survived

# In[ ]:


g = sns.factorplot(x = "Pclass", y = "Survived", data = train_df, kind = "bar", size = 6)
g.set_ylabels("Survived Probability")
plt.show()


# 

# <a id = "16"></a><br>
# ## 6.5. Age - Survived

# In[ ]:


g = sns.FacetGrid(train_df, col = "Survived")
g.map(sns.distplot, "Age", bins = 25)
plt.show()


# * Age <= 10 has a high survival rate,
# * Oldest passenger (80) survived,
# * large number of 20 years old did not survive,
# * Most passengers are in 15-25 age range,
# * Use age feature in tarining 
# * Use age distribution for missing value of age feature.

# <a id = "17"></a><br>
# ## 6.6. Pclass - Survived - Age

# In[ ]:


g = sns.FacetGrid(train_df, col = "Survived", row = "Pclass", height = 4)
g.map(plt.hist, "Age", bins = 25)
g.add_legend()
plt.show()


# * Pclass is important feature for model training.

# <a id = "18"></a><br>
# ## 6.7. Embarked - Sex - Pclass - Survived

# In[ ]:


g = sns.FacetGrid(train_df, row = "Embarked", height = 2)
g.map(sns.pointplot, "Pclass", "Survived", "Sex")
g.add_legend()
plt.show()


# * Female passengers have much better survival rate than male passengers.
# * Male passengers have better survival rate in pclass 3 in C.
# * Embarked and sex will be used in training.

# <a id = "19"></a><br>
# ## 6.8. Embarked - Sex - Fare - Survived

# In[ ]:


g = sns.FacetGrid(train_df, row = "Embarked", col = "Survived", size = 2.5)
g.map(sns.barplot, "Sex", "Fare")
g.add_legend()
plt.show()


# * Passengers who pay higher fare have better survival.
# * Fare can be used as categorical for training.
# 

# <a id = "20"></a><br>
# # 7. Fill Missing: Age Feature

# In[ ]:


train_df[train_df["Age"].isnull()]


# In[ ]:


sns.factorplot(x= "Sex", y = "Age", data = train_df, kind = "box")
plt.show()


# * Sex is not informative for age prediction. Age distribution seems to be same.

# In[ ]:


sns.factorplot(x= "Sex", y = "Age", hue = "Pclass", data = train_df, kind = "box")
plt.show()


# * First class passenger is older than second class and second class is older than third class.
# 

# In[ ]:


sns.factorplot(x= "Parch", y = "Age", data = train_df, kind = "box")
sns.factorplot(x= "SibSp", y = "Age", data = train_df, kind = "box")
plt.show()


# In[ ]:


df["Sex"] = [1 if i == "male" else 0 for i in df["Sex"]]


# In[ ]:



sns.heatmap(train_df[["Age", "Sex", "SibSp", "Parch", "Pclass"]].corr(), annot = True)


# * Age is not correlated with sex but it is correlated with parch, sibsp and pclass.

# In[ ]:


index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)
for i in index_nan_age:
    age_pred = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) &(train_df["Parch"] == train_df.iloc[i]["Parch"])& (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()
    age_med = train_df["Age"].median()
    if not np.isnan(age_pred):
        train_df["Age"].iloc[i] = age_pred
    else:
        train_df["Age"].iloc[i] = age_med   


# In[ ]:


train_df[train_df["Age"].isnull()]


# <a id = "21"></a><br>
# # 8. Feature Engineering

# <a id = "22"></a><br>
# # 8.1. Name - Title

# In[ ]:


train_df["Name"].head(10)


# In[ ]:


name = train_df["Name"]
train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]


# In[ ]:


train_df["Title"].head(10)


# In[ ]:


sns.countplot(x="Title", data=train_df)
plt.xticks(rotation=60)
plt.show()


# In[ ]:


# convert to categorical
train_df["Title"] = train_df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")
train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["Title"]]
train_df["Title"].head(20)


# In[ ]:


sns.countplot(x="Title", data=train_df)
plt.xticks(rotation=60)
plt.show()


# In[ ]:


g = sns.factorplot(x="Title", y="Survived", data=train_df, kind="bar")
g.set_xticklabels(["Master","Mrs","Mr","Other"])
g.set_ylabels("Survival Probability")
plt.show()


# In[ ]:


train_df.drop(labels = ["Name"], axis=1, inplace=True)


# In[ ]:


train_df.head()


# In[ ]:


train_df = pd.get_dummies(train_df,columns=["Title"])
train_df.head()


# <a id = "23"></a><br>
# # 8.2. Family Size

# In[ ]:


train_df["FSize"] = train_df["SibSp"] + train_df["Parch"] +1


# In[ ]:


train_df.head()


# In[ ]:


g = sns.factorplot(x="FSize", y="Survived", data=train_df, kind="bar")
g.set_ylabels("Survival")
plt.show()


# In[ ]:


train_df["family_size"] = [1 if i<5 else 0 for i in train_df["FSize"]]


# In[ ]:


train_df.head(10)


# In[ ]:


sns.countplot(x="family_size", data=train_df)
plt.show()


# In[ ]:


g = sns.factorplot(x="family_size", y="Survived", data=train_df, kind="bar")
g.set_ylabels("Survival")
plt.show()


# Small families have more chance to survive than large families.

# In[ ]:


train_df = pd.get_dummies(train_df,columns=["family_size"])
train_df.head()


# <a id = "24"></a><br>
# # 8.3. Embarked

# In[ ]:


train_df["Embarked"].head()


# In[ ]:


sns.countplot(x="Embarked", data=train_df)
plt.show()


# In[ ]:


train_df = pd.get_dummies(train_df,columns=["Embarked"])
train_df.head()


# <a id = "25"></a><br>
# ## 8.4. Ticket

# In[ ]:


train_df["Ticket"].head(20)


# In[ ]:


tickets = []
for i in list(train_df.Ticket):
    if not i.isdigit():
        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])
    else:
        tickets.append("x")
train_df["Ticket"] = tickets


# In[ ]:


train_df["Ticket"].head()


# In[ ]:


train_df.head()


# In[ ]:


train_df = pd.get_dummies(train_df,columns=["Ticket"], prefix = "T")


# In[ ]:


train_df.head()


# <a id = "26"></a><br>
# ## 8.5. Pclass

# In[ ]:


sns.countplot(x="Pclass", data=train_df)
plt.show()


# In[ ]:


train_df["Pclass"] = train_df["Pclass"].astype("category")
train_df = pd.get_dummies(train_df,columns=["Pclass"])
train_df


# <a id = "27"></a><br>
# ## 8.6. Sex

# In[ ]:


train_df["Sex"] = train_df["Sex"].astype("category")
train_df = pd.get_dummies(train_df, columns=["Sex"])


# In[ ]:


train_df


# <a id = "28"></a><br>
# ## 8.7. Drop Passenger ID and Cabin

# In[ ]:


train_df.drop(labels= ["PassengerId","Cabin"],axis=1,inplace=True)


# In[ ]:


train_df.columns


# <a id = "29"></a><br>
# # 9. Modelling

# In[ ]:


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# <a id = "30"></a><br>
# ## 9.1. Train Test Split

# In[ ]:


train_df_len


# In[ ]:


test = train_df[train_df_len:]
test.drop(labels = ["Survived"],axis = 1, inplace = True)


# In[ ]:


test.head()


# In[ ]:


train = train_df[:train_df_len]
x_train = train.drop(labels= "Survived",axis=1)
y_train = train["Survived"]
x_train, x_test, y_train ,y_test = train_test_split(x_train, y_train, test_size=0.33, random_state=42)
print("x_train", len(x_train))
print("x_test", len(x_test))
print("y_train", len(y_train))
print("y_test", len(y_test))
print("test", len(test))


# <a id = "31"></a><br>
# ## 9.2. Simple Logistic Regression

# In[ ]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
acc_logreg_train = round(logreg.score(x_train, y_train)*100,2)
acc_logreg_test = round(logreg.score(x_test, y_test)*100,2)
print("Training Accuracy: % {}".format(acc_logreg_train))
print("Testing Accuracy: % {}".format(acc_logreg_test))


# <a id = "32"></a><br>
# ## 9.3. Hyperparameter Tuning - Grid Search - Cross Validation
# We will compare 5 machine learning classifier and evaluate mean accuracy of each them by stratified cross validation.
# * Decision Tree
# * SVM
# * Random Forest
# * KNN
# * Logistic Regression

# In[ ]:


random_state = 42
classifier = [DecisionTreeClassifier(random_state = random_state),
             SVC(random_state = random_state),
             RandomForestClassifier(random_state = random_state),
             LogisticRegression(random_state = random_state),
             KNeighborsClassifier()]

dt_param_grid = {"min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)}

svc_param_grid = {"kernel" : ["rbf"],
                 "gamma": [0.001, 0.01, 0.1, 1],
                 "C": [1,10,50,100,200,300,1000]}

rf_param_grid = {"max_features": [1,3,10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,300],
                "criterion":["gini"]}

logreg_param_grid = {"C":np.logspace(-3,3,7),
                    "penalty": ["l1","l2"]}

knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),
                 "weights": ["uniform","distance"],
                 "metric":["euclidean","manhattan"]}
classifier_param = [dt_param_grid,
                   svc_param_grid,
                   rf_param_grid,
                   logreg_param_grid,
                   knn_param_grid]


# In[ ]:


cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)
    clf.fit(x_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])


# In[ ]:


cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",
             "LogisticRegression",
             "KNeighborsClassifier"]})

g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)
g.set_xlabel("Mean Accuracy")
g.set_title("Cross Validation Scores")


# <a id = "33"></a><br>
# ## 9.4. Ensemble Modelling

# In[ ]:


votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),
                                        ("rfc",best_estimators[2]),
                                        ("lr",best_estimators[3])],
                                        voting = "soft", n_jobs = -1)
votingC = votingC.fit(x_train, y_train)
print(accuracy_score(votingC.predict(x_test),y_test))


# <a id = "34"></a><br>
# # 10. Prediction and Submission

# In[ ]:


test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)
results = pd.concat([test_passengerID, test_survived],axis = 1)
results.to_csv("titanic.csv", index = False)


# In[ ]:




