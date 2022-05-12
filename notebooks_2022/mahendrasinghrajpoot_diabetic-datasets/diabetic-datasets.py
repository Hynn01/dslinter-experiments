#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data_1 = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')


# In[ ]:


data_1.head()#viewing the head of the dataset


# In[ ]:


data_1.shape#finding the shape of the dataset


# In[ ]:


data_1.isnull().sum()#finding out the null values of the dataset


# In[ ]:


data_1.describe()#this is also a one way to find the missing values in the dataset


# In[ ]:


data_1.query('Glucose > Insulin')


# In[ ]:


data_1.dtypes #finding out the datatypes of the different variables


# In[ ]:


data_1['Outcome'].value_counts() #counting the number of outcomes of the target variable

#as we can see people who do not have the diabetis are outnumbered with the people who have the diabetic


# In[ ]:


data_1.corr()#finding the correlations between different variables


# In[ ]:


#plotting the correlation matrix of the dataset, as we have reached this step because there are no missing values in the dataset as well as there is no class imblance problem in out dataset
#plotting the correlation matrix
sns.heatmap(data_1.corr(),annot=True)

#we have to check out both the correlation between the matrix be it the positive or the negative , but mainly we consider the positive correlation in our dataset


# In[ ]:


from sklearn.linear_model import LinearRegression
#so we have imported the model of our dataset
#from sklearn.linear_model import LinearRegression


# In[ ]:




