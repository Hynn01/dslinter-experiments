#!/usr/bin/env python
# coding: utf-8

# # Dataset

# In[ ]:


import pandas as pd
df = pd.read_csv('../input/hr-analytics/HR_comma_sep.csv')
df


# In[ ]:


# Check missing value
df.isnull().sum()


# # Data Exploration

# **Spliting Data**

# In[ ]:


left = df[df.left==1]
left.shape 


# In[ ]:


retained = df[df.left==0]
retained.shape


# 3571 left and 11428 not left
# 
# 10 is the number of columns

# **Mean**

# In[ ]:


df.groupby('left').mean()


# From comparing mean beetwen 0 and 1 we see:
# 
# 1. Satisfaction Level = The value is lower than not left (0) that mean this label doesnt have much impact for employees to left the company.
# 2. Average Montly Hours = Employees that left the company have higher hours work.
# 3. Time Spend Company = Similar to Average Monthly Hours.
# 4. Promotion Last 5 Years = Employees that left the company have lower value than that employees who doest left company

# In[ ]:


df_mean = df.groupby('left').mean()
df_mean


# 
# **Data Visualization (Label that have text data)**

# In[ ]:


pd.crosstab(df.salary,df.left).plot(kind='bar')
pd.crosstab(df.Department,df.left).plot(kind='bar')


# As you can see employees who leaving have lower salary than employees who doesnt left

# # Filtering Dataset

# From the data analysis we get 5 label that give impact for emloyees to left company
# 
# 1. Satisfaction Level
# 2. Average Montly Hours
# 3. Promotion Last 5 Years
# 4. Salary

# In[ ]:


df2 = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]
df2


# # Converting Text Based Data (Tackle Salary Dummy Variable)

# In[ ]:


# Make New Label From Salary Category
salary_dummies = pd.get_dummies(df2.salary, prefix="salary")
salary_dummies


# In[ ]:


# Join New Labels to Dataframe
df_with_salary = pd.concat([df2, salary_dummies], axis="columns")
df_with_salary


# In[ ]:


# Delete Salary Label
df_with_salary.drop('salary', axis='columns', inplace=True)
X = df_with_salary
X


# # Model (Logistic Regression)

# **Y Variable**

# In[ ]:


y = df.left


# **Training & Splitting Test**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3)


# **Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
modlr = LogisticRegression()
modlr.fit(X_train, y_train)


# In[ ]:


modlr.predict(X_test)


# In[ ]:


modlr.score(X_test,y_test)


# The model have accuracy by 77%

# # Support Vector Machine

# In[ ]:


from sklearn.svm import SVC
modsvm = SVC()
modsvm.fit(X_train, y_train)
modsvm.score(X_test,y_test)
# SVC() = {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’
# use one to diffrent score


# In[ ]:


modsvm.predict(X_test)


# # Decision Tree

# In[ ]:


from sklearn import tree
moddt = tree.DecisionTreeClassifier()

moddt.fit(X_train, y_train)
moddt.score(X_test,y_test)


# In[ ]:


moddt.predict(X_test)

