#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv('/kaggle/input/patient-survival-prediction-dataset/Dataset.csv')


# **EDA**

# In[ ]:


data.shape


# In[ ]:


data.dtypes


# In[ ]:


data.head(15)


# In[ ]:


data.columns


# In[ ]:


data[['age', 'bmi', 'weight','hospital_death', 'height']].hist(bins = 10, figsize=(10, 10),
    grid = False,
    rwidth = 0.9)
plt.show()


# In[ ]:


data.describe()


# In[ ]:


data.describe(include='object')


# Note: Total no. of rows= 91713 Columns like 'ethnicity','gender','hospital_admit_score', etc above have count values which is less than the row numbers. Hence, there are missing values.

# In[ ]:


#Looking for missing values
data.isnull().sum()


# In[ ]:


for col in data.select_dtypes(include='object'):
    if data[col].nunique() <= 20:
        sns.countplot(y=col, data=data)
        plt.show()


# In[ ]:


num_cols = ['age','bmi','height','weight']
for col in num_cols:
    sns.boxplot(y = data['hospital_death'].astype('category'), x = col, data=data)
    plt.show()

