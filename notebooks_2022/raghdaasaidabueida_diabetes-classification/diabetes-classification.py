#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

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


# In[ ]:


df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df.head(10)


# In[ ]:


df.describe().transpose()


# In[ ]:


#dataset spilitig for classification data_attributes and data_target
attributes = df.drop('Outcome',axis=1)
target = df['Outcome']


# In[ ]:


#spiliting the dataset to (attributes + target) into Training and test
from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(attributes,target,test_size=0.3)
Xtrain.head(10)


# In[ ]:


ytrain.head(10)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain=scaler.transform(Xtrain)
Xtest=scaler.transform(Xtest)


# In[ ]:


#Naive bayes Classification
from sklearn.naive_bayes import GaussianNB as gnb
model = gnb()
model.fit(Xtrain,ytrain)
y_pred=model.predict(Xtest)


# In[ ]:


from sklearn import metrics
cm = metrics.confusion_matrix(ytest, y_pred)
acc=metrics.accuracy_score(ytest, y_pred)
precision=metrics.precision_score(ytest, y_pred)
recall=metrics.recall_score(ytest, y_pred)
f_score=metrics.f1_score(ytest, y_pred)
print('confusion',cm)
print("Accuracy:",acc)
print("precision:",precision)
print("Recall:",recall)
print("F-measure:",f_score)


# In[ ]:


# Knn Classification
from sklearn.neighbors import KNeighborsClassifier as KNN
knn_model = KNN(n_neighbors = 5)
knn_model.fit(Xtrain,ytrain)
y_pred_knn=knn_model.predict(Xtest)


# In[ ]:


from sklearn import metrics
cm = metrics.confusion_matrix(ytest, y_pred_knn)
acc=metrics.accuracy_score(ytest, y_pred_knn)
precision=metrics.precision_score(ytest, y_pred_knn)
recall=metrics.recall_score(ytest, y_pred_knn)
f_score=metrics.f1_score(ytest, y_pred_knn)
print('confusion',cm)
print("Accuracy:",acc)
print("precision:",precision)
print("Recall:",recall)
print("F-measure:",f_score)


# In[ ]:


#Neural Network Classification'
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2),max_iter=1000, random_state=1)
clf.fit(Xtrain,ytrain)
y_pred_clf=clf.predict(Xtest)
cm = metrics.confusion_matrix(ytest, y_pred_clf)
acc=metrics.accuracy_score(ytest, y_pred_clf)
precision=metrics.precision_score(ytest, y_pred_clf)
recall=metrics.recall_score(ytest, y_pred_clf)
f_score=metrics.f1_score(ytest, y_pred_clf)
print('confusion',cm)
print("Accuracy:",acc)
print("precision:",precision)
print("Recall:",recall)
print("F-measure:",f_score)

