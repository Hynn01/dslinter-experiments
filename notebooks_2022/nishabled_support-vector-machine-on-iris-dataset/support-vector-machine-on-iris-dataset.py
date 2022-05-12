#!/usr/bin/env python
# coding: utf-8

# we are going to explore the Iris dataset and analyse the results of classification using support vector machine (SVMs)

# In[ ]:


#importing libraries

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import svm


# In[ ]:


#loading dataset
data = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
data


# In[ ]:


#print the top 5 columns
data.head()


# In[ ]:


# check missing values
data.isnull().sum()


# In[ ]:


# define x,y
x = data.drop('species', axis = 1)
y = data['species']


# In[ ]:


# create a pairplot to visualize the dataset 

import seaborn as sns
sns.pairplot(data, hue = 'species');


# In[ ]:


# split the dataset into training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=30, random_state= 0)


# In[ ]:


# fit and train dataset

from sklearn.svm import SVC
model = svm.SVC()
model.fit(x_train,y_train)


# In[ ]:


# PREDICTIONS
pred = model.predict(x_test)


# In[ ]:


# model evaluation

from sklearn.metrics import classification_report ,confusion_matrix
print("classification_report:\n",classification_report(y_test,pred))
print("confusion_matrix:\n",confusion_matrix(y_test,pred))


# In[ ]:


model.score(x_test,y_test)

