#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# I am glad to welcome all the participants of the kaggle community. I bring to your attention my work on machine learning. As my first job, I chose the legendary Titanic competition. In this kernel you will find my solution to this classification problem.

# ### Table of contents:
# 1.Import
# 
# 1.1.Import of Required Modules
# 
# 1.2.Importing (Reading) Data 
# 
# 2.Exploratory Analysis 
# 
# 2.1.Data Visualization 
# 
# 3.Data Cleaning 
# 
# 4.Feature Engineering / Feature Selection 
# 
# 5.Machine Learning Models 
# 
# 6.Evaluate & Interpret Results 
# 
# 7.(if necessary) Define the Question of Interest/Goal

# ### 1.Import

# ### 1.1.Import of Required Modules

# In[ ]:


import numpy as np                  
import matplotlib.pyplot as plt     
import pandas as pd                
import seaborn as sns              


# ### 1.2.Importing (Reading) Data

# In[ ]:


df = pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')


# ### 2.Exploratory Analysis

# In[ ]:


type(df)


# In[ ]:


df


# In[ ]:


df.values


# In[ ]:


df.info()   


# In[ ]:


df.shape 


# In[ ]:


df.columns 


# In[ ]:


df.head(3)


# In[ ]:


df.tail(3)


# In[ ]:


df.dtypes 


# ### 2.1.Data Visualization

# In[ ]:


df['Age'].plot(kind='hist', bins=20)     
# we build a histogram of the distribution of the age of passengers, the number of columns can be specified


# In[ ]:


df['Age'].plot(kind='kde')    # graph - age distribution


# In[ ]:


df['Pclass'].value_counts().plot.pie(legend=True) # Distribution of passengers by cabin classes in a pie chart


# In[ ]:


df['Survived'].value_counts().plot.pie(legend=True)  
# Distribution of passengers into survivors and non-survivors on a pie chart


# In[ ]:


df.groupby('Sex')['Age'].plot(kind='kde', xlim=[0,100], legend=True) 
# graph - age distribution, you can specify the boundaries, for men and women


# ### 3.Data Cleaning 

# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()   # we get the total number of NaN elements in X


# In[ ]:


# replacing the NaN values in the 'Age' column with the average value
df['Age'].fillna(df['Age'].median(), inplace = True)
df.isnull().sum()   # we make sure that now there are no NaN values in the 'Age' column


# In[ ]:


# deleting uninformative columns if we think it is possible to do so
df = df.drop(['Name', 'Ticket', 'Cabin'], axis = 1)


# In[ ]:


df.head(3)


# ### 4.Feature Engineering / Feature Selection 

# In[ ]:


# Pre-processing
# Recoding categorical features
df['Sex'].value_counts()


# In[ ]:


sex_mapping = {'male':1, "female":0}
df['Sex'] = df['Sex'].map(sex_mapping)


# In[ ]:


df.head(3)


# In[ ]:


# Recoding categorical features
df['Embarked'].value_counts()


# In[ ]:


# Recoding categorical features - one-hot encoding
# Here we actually create three new binary features instead of one 'Embarked' feature
Embarked_dummies = pd.get_dummies(df['Embarked'], prefix='port', dummy_na = False)
Embarked_dummies.head(3)


# In[ ]:


df.head(3)


# In[ ]:


# we combine our data in df with the created DataFrame Embarked_dummies
df = pd.concat([df, Embarked_dummies], axis=1)


# In[ ]:


# We delete the 'Embarked' column, since now in our Data Frame this feature is recoded into 3 new columns 
df = df.drop(['Embarked'], axis=1)
df.head(3)


# ### 5.Machine Learning Models

# In[ ]:


# In this kernel we create a model - k nearest neighbors
df


# In[ ]:


X = df.drop(['Survived'], axis=1)
y = df['Survived']


# In[ ]:


X


# In[ ]:


y


# In[ ]:


# Dividing the original DataFrame by X_trail, X_test, y_train, y_test
# The original X is divided into X_trail, X_test in a certain proportion (in our case 75% and 25%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[ ]:


len(X_train)


# In[ ]:


len(X_test)


# In[ ]:


# In this kernel we create a model - k nearest neighbors
# n_neighbors=1
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)   
# creating an instance object of the KNeighborsClassifier class from the neighbours module


# In[ ]:


# the fit method trains the model
# the fit method returns the knn object itself (and modifies it)
knn.fit(X_train, y_train)   


# In[ ]:


y_pred = knn.predict(X_test)        # the predict method makes a prediction for test set
knn.score(X_test, y_pred)           # the score method of the knn object, which calculates the accuracy  of the model for the test set
print("Accuracy on the test set: {:.2f}".format(knn.score(X_test, y_test)))


# In[ ]:


# n_neighbors=7
knn = KNeighborsClassifier(n_neighbors=7)   


# In[ ]:


knn.fit(X_train, y_train) 


# In[ ]:


y_pred = knn.predict(X_test)        
knn.score(X_test, y_pred)           
print("Accuracy on the test set: {:.2f}".format(knn.score(X_test, y_test)))


# In[ ]:


# n_neighbors=15
knn = KNeighborsClassifier(n_neighbors=15)   


# In[ ]:


knn.fit(X_train, y_train) 


# In[ ]:


y_pred = knn.predict(X_test)        
knn.score(X_test, y_pred)           
print("Accuracy on the test set: {:.2f}".format(knn.score(X_test, y_test)))


# In[ ]:


# n_neighbors=19
knn = KNeighborsClassifier(n_neighbors=19)   


# In[ ]:


knn.fit(X_train, y_train) 


# In[ ]:


y_pred = knn.predict(X_test)        
knn.score(X_test, y_pred)           
print("Accuracy on the test set: {:.2f}".format(knn.score(X_test, y_test)))


# In[ ]:


# n_neighbors=21
knn = KNeighborsClassifier(n_neighbors=21)   


# In[ ]:


knn.fit(X_train, y_train) 


# In[ ]:


y_pred = knn.predict(X_test)        
knn.score(X_test, y_pred)           
print("Accuracy on the test set: {:.2f}".format(knn.score(X_test, y_test)))


# In[ ]:


# n_neighbors=31
knn = KNeighborsClassifier(n_neighbors=31)   


# In[ ]:


knn.fit(X_train, y_train) 


# In[ ]:


y_pred = knn.predict(X_test)        
knn.score(X_test, y_pred)           
print("Accuracy on the test set: {:.2f}".format(knn.score(X_test, y_test))) 


# ### 6.Evaluate & Interpret Results

# In[ ]:


from sklearn.metrics import accuracy_score    # simple metric - the proportion of correct answers
y_pred = knn.predict(X_test)
acc_score = accuracy_score(y_test, y_pred)
print("knn accuracy score: {:.2f}".format(acc_score)) # knn accuracy score
# print(accuracy_score(y_test, y_pred)) # similar instruction


# In[ ]:


# Selection of the number of neighbors
import matplotlib.pyplot as plt
k_range = np.arange(1,50)
acc_scores = list()
for k in k_range:
    knn_clf = KNeighborsClassifier(n_neighbors= k).fit(X_train, y_train) 
    y_pred = knn_clf.predict(X_test)
    acc_scores.append(accuracy_score(y_test, y_pred))
plt.plot(k_range, acc_scores)


# In[ ]:


# On the graph we find the value of k, at which accuracy is the greatest, it is 19
k_range = np.arange(15,25)
acc_scores = list()
for k in k_range:
    knn_clf = KNeighborsClassifier(n_neighbors= k).fit(X_train, y_train) 
    y_pred = knn_clf.predict(X_test)
    acc_scores.append(accuracy_score(y_test, y_pred))
plt.plot(k_range, acc_scores)


# In[ ]:


# n_neighbors=19
knn = KNeighborsClassifier(n_neighbors=19)   
knn.fit(X_train, y_train) 
y_pred = knn.predict(X_test)        
knn.score(X_test, y_pred)           
print("Accuracy on the test set: {:.2f}".format(knn.score(X_test, y_test)))


# In this work, I did not consider other parameters.

# ### 7.(if necessary) Define the Question of Interest/Goal

# I did not fill in this section in this Notebook.
