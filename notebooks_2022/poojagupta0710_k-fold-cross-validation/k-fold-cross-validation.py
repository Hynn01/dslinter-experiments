#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# to work with dataframes
import pandas as pd
import numpy as np

# to split data into train and test
from sklearn.model_selection import train_test_split

# to build logstic regression model
from sklearn.linear_model import LogisticRegression

# to create k folds of data and get cross validation score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# to ignore warnings
import warnings
warnings.filterwarnings('ignore')


# ## Load and view dataset

# In[ ]:


df = pd.read_csv('../input/diabetes-cv/diabetes.csv')

df.head()


# In[ ]:


# separating data into X and Y
X = df.drop(['class'], axis = 1)
Y = df['class']


# In[ ]:


# creating train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=1, stratify = Y)


# In[ ]:


# defining kfold
kfold = KFold(n_splits=10, random_state=1, shuffle = True)

# number of splits = 10


# In[ ]:


# defining the model
model = LogisticRegression(random_state = 1)

# storing accuracy values of model for every fold in "results"
results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')


# In[ ]:


# let's see the value of accuracy for every fold
print(results)


# In[ ]:


# let's see the mean accuracy score
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

