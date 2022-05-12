#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Cleaning
# Narrow down the data, add the campaign days feature, and drop unnecessary features, all as we did before.

# In[ ]:


kick = pd.read_csv("/kaggle/input/kickstarter-projects/ks-projects-201801.csv")

# narrowing with conditions
kick = kick[(kick["state"] == "successful") | (kick["state"] == "failed")]
kick = kick[(kick["goal"] <= 50000) & (kick["currency"] == "USD") & (kick["country"] == "US")]

# creating the campaign_days feature
kick["campaign_days"] = pd.to_datetime(kick["deadline"]) - pd.to_datetime(kick["launched"])
kick["campaign_days"] = kick["campaign_days"].dt.days + 1

# dropping features
kick = kick.drop(["currency", "country", "usd pledged", "usd_pledged_real", "usd_goal_real"], axis=1)

kick.head()


# # Feature transformation
# 
# We're only going to use the following three features in our models:
# 
# * main_category
# * goal
# * campaign_days
# 
# I'm only using these three features because:
# 
# * **name:** There is no simple way to transform the name of a project into a number that the algorithm can understand and utilize. Further, it's unlikely that the name of a project correlates that strongly to success rate.
# * **pledged and backers:** Before launching the project, a creator would not know the amount pledged or the number of backers at the project deadline. I want to use this model to predict whether my project will succeed or fail based on preliminary factors.
# ![](http://)* **deadline and launched:** There is possibly a correlation between the month and year the project launched with its success rate, but I'll revisit that possibility later because I haven't done any visualizations on these features yet.
# 
# I'm transforming main_category into numbers because most algorithms can only handle numbers. This will be further elaborated on later under the section "main_category" but by "transformation", I'm really saying that the values of main_category, which are strings, will be re-labeled as numbers.
# 
# I'm turning both goal and campaign_days into a set of bins... that means that what was once a goal of, say, 7,800 would go into a bin labeled 2, which contains the values from 5,001 to 10,000. Using this method will take care of the fact that numbers with trailing zeroes are more likely to be chosen as the goal by the creator. In other words, although goals of 4,999 and 5,000 are *very* close to one another, the algorithm might run into some trouble because there are many more projects with goals of 5,000 than there are projects with goals of 4,999, and this discrepency may interfere with machine learning training.

# In[ ]:


kick = kick[["ID", "main_category", "goal", "campaign_days", "state"]]
kick.head()


# # state
# Change the state into either 0 for failed or 1 for successful.

# In[ ]:


kick.loc[kick["state"] == "successful", "state"] = 1
kick.loc[kick["state"] == "failed", "state"] = 0
kick["state"] = kick["state"].astype("int")


# # main_category
# Transform each main category into a number. For instance, "Narrative Film" would be assigned the value 1.

# In[ ]:


kick["main_category"] = kick["main_category"].astype("category")
kick["main_category"] = kick["main_category"].cat.codes


# Graphing the distribution of the goals by density.

# In[ ]:


import seaborn as sns
sns.distplot(kick["goal"])


# # goal_by_5000
# Transforming the goal feature into different bins:
# * if the goal is 350, then it goes into the bin labeled 1, which contains values from 0 - 5,000
# * if the goal is 2600, then it goes into the bin labeled 2, which contains values from 5,000 - 10,000
# * and so on, counting by 5,000s all the way up to 50,000 for a total of 10 bins labeled from 1 to 10

# In[ ]:


kick["goal_by_5000"] = pd.cut(x=kick["goal"], bins=[0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000], labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
kick.head()


# In[ ]:


kick = kick.drop("goal", axis=1)
kick["goal_by_5000"] = kick["goal_by_5000"].astype("int")


# # campaign_days_by_19
# Similar to goals_by_5000, the feature campagin_days is distributed into five different bins with a range of 19.

# In[ ]:


kick["campaign_days_by_19"] = pd.cut(x=kick["campaign_days"], bins=[0,19,38,57,76,95], labels=[1, 2, 3, 4, 5])
kick = kick.drop("campaign_days", axis=1)
kick["campaign_days_by_19"] = kick["campaign_days_by_19"].astype("int")
kick.head()


# In[ ]:


kick.dtypes


# # Modeling

# In[ ]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(kick, test_size=0.3, random_state=42)


# In[ ]:


X_train = train.drop("state", axis=1)
Y_train = train["state"]
X_test  = test.drop("ID", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[ ]:


from sklearn.svm import SVC, LinearSVC

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)

