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


#In this notebook I only used Logistic Regression the basic algorithm for classification.
# Let us make the best out of it and gain maximum accuracy only usng LR
train = pd.read_csv('/kaggle/input/titanic/train.csv') # loading train data
train


# In[ ]:


test = pd.read_csv('/kaggle/input/titanic/test.csv')
test


# In[ ]:


print(train.isna().sum()) # Finding sum of nan values in each Series


# In[ ]:


# In general if there are missing values which are usually as Nan values then find the percaentage of missing values in each column


# In[ ]:


train.drop(train.columns[[0,3,8,10]],axis=1,inplace=True) #Dropping redundant columns like Name, PassengerId,Cabin,Ticket
# In this notebook, I did not perform much of EDA  so I just dropped those columns but I would like to do in my upcoming notebooks.


# In[ ]:


ye = pd.get_dummies(train.Sex) # Converting categorical variables into integer variables
yew = pd.get_dummies(train.Embarked)
train = train.drop(['Sex','Embarked'], axis =1)
train = pd.concat([train, ye], axis=1)
train = pd.concat([train, yew], axis=1)
train


# In[ ]:


tester = test.drop(test.columns[[0,2,7,9]],axis=1) #Dropping redundant columns
#First method remove all rows with nan values #Returns a dataset without nan values
print(tester)
yr = pd.get_dummies(tester.Sex)
yew = pd.get_dummies(tester.Embarked)
tester = tester.drop(['Sex','Embarked'], axis =1)
tester = pd.concat([tester, yr], axis=1)
tester = pd.concat([tester, yew], axis=1)
tester


# In[ ]:


train.describe() # Describes about our train data like mean,count etc


# In[ ]:


train.info() # Gives info about train data its datatype and objects being null/not


# In[ ]:


def impute_missing_data(test, col, median, mode,mean):
    
    """ This function replaces all nan values in a column by mean or median or mode
    INPUT : column name,dataframe,which value we want to be replaced
    OUTPUT: returns the modified dataframe"""

    nanindex = []  #To store indices of null values in a 
    test.loc[:,col] = test[col].fillna('nan')
    for i in range(test.shape[0]):
        if(test.loc[i,col] == 'nan'):
            nanindex.append(i)
    #print(nanindex)
    mod = test[col].value_counts().index[1] 
    a = np.squeeze(np.nanmean(np.array(test[col],dtype = float))) # Finds mean of data excluding nan values
    #print(a)
    #print(mod)
    for j in nanindex:
        if median == True:
            test.loc[j,col] = test[col].median()
            #print(test[col].median())
        elif mode == True:
            test.loc[j,col] = mod
            #print(mod)
        elif mean == True:
            test.loc[j,col] = a
            #print(test.loc[j,col])
    #print(test)
    return test   


# In[ ]:


#Imputing mean values for missing data
train = impute_missing_data(train,'Age' , median =True, mode=False,mean =False)
#As Embarked is a categorical variable mean and median does not make sense. Hence impute missing values
# with mode of column
#print(train)
train.info()


# In[ ]:


#Imputing median values for missing data
tester = impute_missing_data(tester,'Fare' , median =True, mode=False,mean =False)
tester = impute_missing_data(tester,'Age' , median =True, mode=False,mean =False)
#As Embarked is a categorical variable mean and median does not make sense. Hence impute missing values
# with mode of column
tester


# In[ ]:


X_train = np.array(train.iloc[:,1:])
y = np.array(train.iloc[:,0])
X_test = np.array(tester.iloc[:,:])
#print(X_test)
#print(y)
#print(X_train)
i = np.mean(X_train,axis=0,dtype = float)
r = np.std(X_train, axis=0,dtype = float)
X = (X_train - i)/r # Performing mean normalization so that data has 0 mean and unit std
print(X)
i = np.mean(X_test,axis=0,dtype = float)
r = np.std(X_test, axis=0,dtype = float)
Xtest = (X_test - i)/r
print(Xtest)


# In[ ]:


from sklearn.linear_model import LogisticRegression # Importing required libraries
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


#Train the model
model = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.30, random_state=42)
model.fit(X_train, y_train) #Training the model
model.score(X_cv,y_cv)
#print(np.shape(X_train))
#print(np.shape(X_cv))


# In[ ]:


#Using cv set to find best degree
def degree(X_train,y_train,X_cv,y_cv,X,y,X_test,y_test):
    l = []
    lr = LogisticRegression(max_iter = 250, C =0.0012)
    for i in range(1,6):
        poly = PolynomialFeatures(degree = i, include_bias=False)
        X_poly = poly.fit_transform(X_train)
        lr.fit(X_poly,y_train)
        print(i)
        print(f' Accuracy over train is {lr.score(poly.transform(X_train), y_train)}') 
        print(f' Accuracy over cv is {lr.score(poly.transform(X_cv), y_cv)}') 
        #print(f' Accuracy over test is {lr.score(poly.transform(X_test), y_test)}') 
        l.append({i : lr.score(poly.transform(X_cv), y_cv)})
    return l 
        
bestdegree = degree(X_train,y_train,X_cv,y_cv,X,y,X_test,y_test)
print(bestdegree)
#As we have observed that degree=3 performs better than linear logistic logistic regression because its accuracy is 78.6%
# Whereas degree = 3 accuracy over cv set is 82.88% Hence we use degree = 3 in our model


# In[ ]:


lr = LogisticRegression(max_iter = 400, C =0.0012)
poly = PolynomialFeatures(degree = 3, include_bias=False)
X_poly = poly.fit_transform(X)
lr.fit(X_poly,y)
print(f' Accuracy over entire training set  is {lr.score(poly.transform(X), y)}') 


# In[ ]:


#Test the model
X_testpoly = poly.fit_transform(Xtest)
predictions = lr.predict(X_testpoly)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

