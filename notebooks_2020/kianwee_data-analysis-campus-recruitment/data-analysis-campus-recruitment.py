#!/usr/bin/env python
# coding: utf-8

# ![](https://www.science.unsw.edu.au/sites/default/files/styles/teaser_mobile/public/images/SCIF2199_science.jpeg?h=f4cc0ec0&itok=Q5grmW2p)
# # Introduction
# This week, we are looking into the campus recruitment dataset. 
# 
# First off, what exactly is a job placement? 
# 
# ![](https://www.dotmagazine.online/_Resources/Persistent/5dc7a08f74a692ec2b2e5a6edd476d915efa5b60/iStock-939787916%20copy-720x405.jpg)
# 
# 
# A job placement is very similar to an internship, just much longer. Doing placement as part of the course provides huge benefit by providing great working experience and increase your employability when you are ready to enter the market. No, you won't be just making coffee although you will be responsible for quite a fair amount of general administrative duties. Since you are considered an employee in the company. you will have the opportunity to develop your skills through meatier assignments. After you have completed your job placement, you may be given the opportunity to join the company if you exceed their expectations! 
# 
# For this dataset, we have created the following objectives to answer common questions frequently asked.
# 
# ## Objectives
# 1. Which factor influenced a candidate in getting placed?
# 2. Does percentage matters for one to get placed?
# 3. Which degree specialization is much demanded by corporate?
# 4. Play with the data conducting all statistical tests.

# ### Importing required packages

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# In[ ]:


# Creating a function to print 
def overview():
    data =pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
    print("First 5 lines of data:\n")
    print(data.head())
    print("\n\n\n")
    print("There are {} rows and {} columns".format(data.shape[0], data.shape[1]))
    print("\n\n\n")
    print("Data types:\n")
    print(data.dtypes)
    print("\n\n\n")
    print("% of missing values per column:\n")
    print(data.isnull().mean().round(2)*100)
    print("Statistical summary:\n")
    print(data.describe())
    return data
    
data = overview()


# ### Dealing with NaN values 
# - Since we have 31/215 NaN values, it's not practical to remove them since they can affect the overall integrity of the data. We will use SimpleImputer to replace those NaN with median values. 

# In[ ]:


data = data.fillna(0)

data.isnull().sum()


# ## Can gender affects salary?

# In[ ]:


plt.figure(figsize = (8, 20))

sns.boxplot(data = data, x = 'gender', y = 'salary',showfliers = False).set_title("Barplot showing salary by gender") #outlier not shown here


# - Finding median for female since we can't see clearly.

# In[ ]:


data[data['gender'] == 'F'].salary.median()


# - It seems like the median between males and females are very close. The distribution of salary seems to be wider for males. 

# ## Does academic results affect salary obtained?
# - Do note that these data will only include students that got the job placement.

# In[ ]:


# Secondary Education percentage- 10th Grade vs Salary

sns.regplot(data = data, x ='ssc_p', y = 'salary' ).set_title("Regression plot: Secondary Education percentage- 10th Grade vs Salary")


# In[ ]:


# Higher Secondary Education percentage- 12th Grade vs Salary

sns.regplot(data = data, x ='hsc_p', y = 'salary' ).set_title("Regression plot: Higher Secondary Education percentage- 12th Grade vs Salary")


# In[ ]:


# Degree percentage vs Salary

sns.regplot(data = data, x ='degree_p', y = 'salary' ).set_title("Regression plot: Degree percentage vs Salary")


# In[ ]:


# Employability test percentage vs salary

sns.regplot(data = data, x ='etest_p', y = 'salary' ).set_title("Regression plot: Employability test percentage vs salary")


# In[ ]:


# MBA test percentage vs salary

sns.regplot(data = data, x ='mba_p', y = 'salary').set_title("Regression plot: MBA test percentage vs salary ")


# - We can clearly see that there is no strong correlation between academic scores and salary.

# ## Can type of specialisation affect placement?

# In[ ]:


# Look at placement between gender
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)  
plt.figure(figsize = (8, 10))

sns.countplot(data = data, x = 'gender', hue = 'status', palette = "RdBu").set_title("Barplot showing placement between gender")


# It seems like more males are able to obtain the job placement. 

# In[ ]:


# Look at placement among specialization in higher secondary education
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)  
plt.figure(figsize = (8, 10))

sns.countplot(data = data, x = 'hsc_s', hue = 'status', palette = "RdBu").set_title("Barplot showing placement among specialisation")


# In[ ]:


# Look at placement among degree specialization 
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)  
plt.figure(figsize = (8, 10))

sns.countplot(data = data, x = 'degree_t', hue = 'status', palette = "RdBu").set_title("Barplot showing placement among specialisation (degree)")


# In[ ]:


# Look at placement among master specialization 
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)  
plt.figure(figsize = (8, 10))

sns.countplot(data = data, x = 'specialisation', hue = 'status', palette = "RdBu").set_title("Barplot showing placement among specialisation (masters)")


# - From what we see here, it seems like business related specialisation tend to have a higher chance of getting a job placement.

# ## Will work experience affect placement

# In[ ]:


# Look at placement among work experience
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)  
plt.figure(figsize = (8, 10))

sns.countplot(data = data, x = 'workex', hue = 'status', palette = "RdBu").set_title("Barplot showing placement among different work experience")


# - It seems like work experience has no effect on job placement.

# ## Using logistic regression to predict the chance of getting a job placement

# In[ ]:


# Use label encoder to change categorical data to numerical
le = LabelEncoder()
 
# Implementing LE on gender
le.fit(data.gender.drop_duplicates()) 
data.gender = le.transform(data.gender)

# Implementing LE on ssc_b
le.fit(data.ssc_b.drop_duplicates()) 
data.ssc_b = le.transform(data.ssc_b)

# Implementing LE on hsc_b
le.fit(data.hsc_b.drop_duplicates()) 
data.hsc_b = le.transform(data.hsc_b)

# Implementing LE on hsc_s
le.fit(data.hsc_s.drop_duplicates()) 
data.hsc_s = le.transform(data.hsc_s)

# Implementing LE on degree_t
le.fit(data.degree_t.drop_duplicates()) 
data.degree_t = le.transform(data.degree_t)

# Implementing LE on workex
le.fit(data.workex.drop_duplicates()) 
data.workex = le.transform(data.workex)

# Implementing LE on specialisation
le.fit(data.specialisation.drop_duplicates()) 
data.specialisation = le.transform(data.specialisation)

# Implementing LE on status
le.fit(data.status.drop_duplicates()) 
data.status = le.transform(data.status)


# In[ ]:


plt.figure(figsize=(15,10))
 
corrMatrix = data.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()


# ### Creating test split
# - I will exclude salary from X since people that got the job placement will get a salary. 

# In[ ]:


# Assigning X and y
X = data.drop(['status', 'sl_no', 'salary'], axis=1)
 
y = data['status']
 
# Implementing train and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Looking into the shape of training and test dataset
print(X_train.shape)
print(X_test.shape)


# In[ ]:


# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)

# Fitting the model
logreg.fit(X_train, y_train)
y_pred_test = logreg.predict(X_test)
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))


# - Great accuracy! 

# ## ROC curve
# - The dotted line represents the ROC curve of a purely random classifier; a good classifier stays as far away from that line as possible (toward the top-left corner).

# In[ ]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

