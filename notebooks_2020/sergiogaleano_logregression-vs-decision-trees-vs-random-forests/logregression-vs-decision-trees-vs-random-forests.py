#!/usr/bin/env python
# coding: utf-8

# # In this notebook I implemented three machine learning algorithms on loan data with the purpose of predicting which lenders would pay back or not.
# 

# ## The Machine Learning Algorithms used are:
# 1. Logistic Regression
# 2. Decision Trees
# 3. Random Forests

# ## The purpose of this notebook will only be to run and compare the results of the three models. Therefore, I won't be creating any visualizations. Prep work will involve cleaning the data. 

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


# # 1. Cleaning the Data

# In[ ]:


loans = pd.read_csv('/kaggle/input/predicting-who-pays-back-loans/loan_data.csv')


# In[ ]:


loans.head()


# In[ ]:


loans.info()


# In[ ]:


loans.isnull().sum()
# NO MISSING VALUES
# with no missing values, we'll turn to the columns and see if any should be dropped or edited.


# In[ ]:


# not fully paid, with a score of 1 or 0, will be our target value and the variable we want to predict


# In[ ]:


# let's observe the one 'object' data type
loans['purpose'].nunique()
# With 7 unique values that may correlate with our target value, we'll turn it into a dummy variable


# In[ ]:


purpose_ = pd.get_dummies(loans['purpose'],drop_first=True)
public_record  = pd.get_dummies(loans['pub.rec'],drop_first=True)


# In[ ]:


# drop the original columns that we're replacing with dummy variables 
loans.drop(['purpose','pub.rec'],axis=1,inplace=True)


# In[ ]:


# dummy decider
# loans['pub.rec'].unique()


# In[ ]:


loans = pd.concat([loans,purpose_,public_record],axis=1)
loans.head()


# # 2. :::Model Selection and Execution:::

# # 2.0 Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(loans.drop('not.fully.paid',axis=1), 
                                                    loans['not.fully.paid'], test_size=0.30)


# # 2.1 Logistic Regression

# ### Train and Predict

# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# ### Evaluate the Results 

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


logistic_confusion_matrix = confusion_matrix(y_test,predictions)
logistic_classification_report = classification_report(y_test,predictions)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# # 2.2 Decision Trees

# ### Train and Predict

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# ### Evaluate the Results

# In[ ]:


predictions = dtree.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


decision_tree_confusion_matrix = confusion_matrix(y_test,predictions)
decision_tree_classification_report = (classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# # 2.3 Random Forests 

# ### Train and Predict

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier(n_estimators=600)


# In[ ]:


rfc.fit(X_train,y_train)


# In[ ]:


predictions = rfc.predict(X_test)


# ### Evaluate the Results

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


random_forests_confusion_matrix = confusion_matrix(y_test,predictions)
random_forests_classification_report = classification_report(y_test,predictions)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# # Let's compare the results of all three models together

# In[ ]:


print(logistic_confusion_matrix) 
print(logistic_classification_report)

print(decision_tree_confusion_matrix) 
print(decision_tree_classification_report)

print(random_forests_confusion_matrix) 
print(random_forests_classification_report)


# # Winner = Random Forests 

# ###  An important point to note is that our model was heavily more balanced towards positive cases. Our target class was not balanced, which would expectedly result in a model's ability to better predidct the more highly represented instance of the variable. 
# 
# ###  Interestingly, while Random Forests appears to be the winner, decision trees accurately identified the most amount of true positives.
# 
# ###  While logisitc models can be more intuitive, random forests are generally better than decision trees, at the cost of having a process more opaque to the data scientist.

# ### That concludes our quick three-model showdown.
# 
# ### - Sergio A. Galeano
# 
# ### #fortheloveofdatascience

# In[ ]:




