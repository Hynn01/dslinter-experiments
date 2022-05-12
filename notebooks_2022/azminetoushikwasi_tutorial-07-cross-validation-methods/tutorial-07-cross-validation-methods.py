#!/usr/bin/env python
# coding: utf-8

# # Cross Validation Methods MASTERCLASS with Codes
# #### Cross-validation is a statistical method used to estimate the performance of machine learning models. It is a method for assessing how the results of a statistical analysis will generalize to an independent data set.

# ## How does it tackle the problem of overfitting?
# In Cross-Validation, we use our initial training data to generate multiple mini train-test splits. Use these splits to tune your model. For example in standard k-fold cross-validation, we partition the data into k subsets. Then, we iteratively train the algorithm on k-1 subsets while using the remaining subset as the test set. In this way, we can test our model on completely unseen data. In this article, you can read about the 7 most commonly used cross-validation techniques along with their pros and cons. I have also provided the code snippets for each technique.

# # The techniques are listed below:
# 
# - Hold Out Cross-validation
# 
# - K-Fold cross-validation
# 
# - Stratified K-Fold cross-validation
# 
# - Leave Pout Cross-validation
# 
# - Leave One Out Cross-validation
# 
# - Monte Carlo (Shuffle-Split)
# 
# - Time Series ( Rolling cross-validation)
# 
# ### Kaggle Discusion about ALL CV Methods: [List of all Cross Validation methods](https://www.kaggle.com/discussions/questions-and-answers/323105) 
# 
# 

# ## Creating Random Samples

# In[ ]:


import random
li=random.sample(range(10, 130), 24)


# # 1.HoldOut Cross-validation or Train-Test Split
# 
# In this technique of cross-validation, the whole dataset is randomly partitioned into a training set and validation set. 
# 
# ### More Details : [HoldOut Cross-validation or Train-Test Split]( https://www.kaggle.com/discussions/questions-and-answers/323109)
# 
# 
# 
# ![HoldOut Cross-validation or Train-Test Split.png](attachment:7a80ffc7-954d-461c-a28a-c8b986404736.png)
# 
# ### Python Code Example
# 

# In[ ]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris=load_iris()
X=iris.data[li]
Y=iris.target[li]
print("Size of Dataset {}".format(len(X)))


logreg=LogisticRegression()

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
logreg.fit(x_train,y_train)
predict=logreg.predict(x_test)


# In[ ]:


print("Accuracy score on training set is {}".format(accuracy_score(logreg.predict(x_train),y_train)))
print("Accuracy score on test set is {}".format(accuracy_score(predict,y_test)))


# # 2. K-Fold Cross-Validation
# 
# The technique is repeated K times until each fold is used as a validation set and the remaining folds as the training set.
# 
# ### More Details : [K-Fold Cross-Validation](https://www.kaggle.com/discussions/questions-and-answers/323106)
# 
# 
# 
# ![kfold.png](attachment:2567948b-3353-4ba4-aa2f-273372605167.png)
# 
# ### Python Code Example
# 
# 

# In[ ]:


from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score,KFold
from sklearn.linear_model import LogisticRegression

iris=load_iris()

X=iris.data[li]
Y=iris.target[li]

logreg=LogisticRegression()
kf=KFold(n_splits=5)
score=cross_val_score(logreg,X,Y,cv=kf)


# In[ ]:


print("Cross Validation Scores are {}".format(score))
print("Average Cross Validation score :{}".format(score.mean()))


# # 3. Stratified K-Fold Cross-Validation
# 
# 
# Stratified K-Fold is an enhanced version of K-Fold cross-validation which is mainly used for imbalanced datasets. Just like K-fold, the whole dataset is divided into K-folds of equal size.
# 
# ### More Details : [Stratified K-Fold Cross-Validation](https://www.kaggle.com/discussions/questions-and-answers/323110)
# 
# 
# 
# ![3. Stratified K-Fold Cross-Validation.png](attachment:50411830-309e-45db-baeb-50326e99d8ff.png)
# 
# ### Python Code Example
# 
# 

# In[ ]:


from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.linear_model import LogisticRegression

iris=load_iris()
X=iris.data[li]
Y=iris.target[li]

logreg=LogisticRegression()
stratifiedkf=StratifiedKFold(n_splits=5)

score=cross_val_score(logreg,X,Y,cv=stratifiedkf)


# In[ ]:


print("Cross Validation Scores are {}".format(score))
print("Average Cross Validation score :{}".format(score.mean()))


# # 4. Leave P Out cross-validation
# 
# 
# Suppose we have 100 samples in the dataset. If we use p=10 then in each iteration 10 values will be used as a validation set and the remaining 90 samples as the training set.
# 
# This process is repeated till the whole dataset gets divided on the validation set of p-samples and n-p training samples.
# 
# ### More Details : [Leave P Out cross-validation](https://www.kaggle.com/discussions/questions-and-answers/323111)
# 
# 
# 
# ### Python Code Example
# 
# 

# In[ ]:


from sklearn.model_selection import LeavePOut,cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris=load_iris()
X=iris.data[li]
Y=iris.target[li]

lpo=LeavePOut(p=2)
lpo.get_n_splits(X)

tree=RandomForestClassifier(n_estimators=10,max_depth=5,n_jobs=-1)

score=cross_val_score(tree,X,Y,cv=lpo)


# 

# In[ ]:


print("Cross Validation Scores are \n{}".format(score))
print("Average Cross Validation score :{}".format(score.mean()))


# # 5. Leave One Out cross-validation
# 
# LeaveOneOut cross-validation is an exhaustive cross-validation technique in which 1 sample point is used as a validation set and the remaining n-1 samples are used as the training set.
# 
# ### More Details : [Leave One Out cross-validation](https://www.kaggle.com/discussions/questions-and-answers/323113)
# 
# 
# 
# ![5. Leave One Out cross-validation.gif](attachment:4b06000d-c5b9-40c5-88af-bbf77db64014.gif)
# 
# ### Python Code Example
# 
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut,cross_val_score

iris=load_iris()
X=iris.data[li]
Y=iris.target[li]

loo=LeaveOneOut()
tree=RandomForestClassifier(n_estimators=10,max_depth=5,n_jobs=-1)
score=cross_val_score(tree,X,Y,cv=loo)


# In[ ]:


print("Cross Validation Scores are {}".format(score))
print("Average Cross Validation score :{}".format(score.mean()))


# # 6. Monte Carlo Cross-Validation(Shuffle Split)
# 
# 
# Monte Carlo cross-validation, also known as Shuffle Split cross-validation, is a very flexible strategy of cross-validation. In this technique, the datasets get randomly partitioned into training and validation sets.
# 
# ### More Details : [Monte Carlo Cross-Validation(Shuffle Split)](https://www.kaggle.com/discussions/questions-and-answers/323114)
# 
# 
# 
# ![monte carlo.png](attachment:87fd491f-2eef-4ca7-ae9b-036b2bfdc0bc.png)
# 
# ### Python Code Example
# 
# 

# In[ ]:


from sklearn.model_selection import ShuffleSplit,cross_val_score
from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

shuffle_split=ShuffleSplit(test_size=0.3,train_size=0.5,n_splits=10)

scores=cross_val_score(logreg,iris.data[li],iris.target[li],cv=shuffle_split)


# In[ ]:


print("cross Validation scores:n {}".format(scores))
print("Average Cross Validation score :{}".format(scores.mean()))


# # 7. Time Series Cross-Validation
# 
# Time series data is data that is collected at different points in time. As the data points are collected at adjacent time periods there is potential for correlation between observations. This is one of the features that distinguishes time-series data from cross-sectional data.
# 
# ### More Details : [Time Series Cross-Validation](https://www.kaggle.com/discussions/questions-and-answers/323115)
# 
# 
# 
# ![ts.png](attachment:8216d0e3-0082-4cfe-b2e5-093050f5bafd.png)
# 
# ### Python Code Example
# 
# 

# In[ ]:


import numpy as np
from sklearn.model_selection import TimeSeriesSplit

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])

time_series = TimeSeriesSplit()

print(time_series)

for train_index, test_index in time_series.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# #### Credits : [Top 7 Cross-Validation Techniques with Python Code](https://www.analyticsvidhya.com/blog/2021/11/top-7-cross-validation-techniques-with-python-code/)

# # Fork, Share, Support <3
