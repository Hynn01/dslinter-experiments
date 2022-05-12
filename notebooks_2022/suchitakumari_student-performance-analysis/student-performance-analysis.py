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


# ## Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Reading Data Set

# In[ ]:


df = pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")
df.head()


# In[ ]:


df.info()


# In[ ]:


df.shape


# ## Descriptive Statistics

# In[ ]:


df.describe()


# ## Data Cleaning

# #### 1. Checking the Null Values

# In[ ]:


# Checking null values

df.isnull().sum()


# Null values are not present in the data.

# #### Checking the number of unique items present in the data set

# In[ ]:


df.select_dtypes("object").nunique()


# ## EDA

# In[ ]:


# Checking the number of male and female appeard for exam

plt.figure(figsize=[10,7])
plt.title("Percentace of Male and Female Score Card", fontsize=20)

val = df.gender.value_counts()

plt.pie(data=df, x=val, autopct="%1.0f%%", labels=["Female","Male"])
plt.show()


# ### Conclusion
# 1. Out of the 100%; 52 percantage are female and 48 percentage are male candidates.

# ### EDA On Categorical Variables

# In[ ]:


# Getting the list of all categorical variables

cat_var = df.select_dtypes(include="object").columns
cat_var


# In[ ]:


plt.figure(figsize=[15,5])

plt.subplot(1,2,1)
sns.countplot(data=df, x="race/ethnicity", hue="gender")

plt.subplot(1,2,2)
sns.countplot(data=df, x="parental level of education", hue="gender")
plt.xticks(rotation=60)

plt.show()


# In[ ]:


plt.figure(figsize=[15,5])

plt.subplot(1,2,1)
sns.countplot(data=df, x="lunch", hue="gender")

plt.subplot(1,2,2)
sns.countplot(data=df, x="test preparation course", hue="gender")
plt.xticks(rotation=60)

plt.show()


# ### Marks Vs Gender 

# In[ ]:



plt.figure(figsize=[20,5])

plt.subplot(1,3,1)
plt.xlabel('xlabel', fontsize=18)
plt.ylabel('ylabel', fontsize=18)
sns.boxplot(data=df, x="gender", y="math score")

plt.subplot(1,3,2)
plt.xlabel('xlabel', fontsize=18)
plt.ylabel('ylabel', fontsize=18)
sns.boxplot(data=df, x="gender", y="reading score")

plt.subplot(1,3,3)
plt.xlabel('xlabel', fontsize=18)
plt.ylabel('ylabel', fontsize=18)
sns.boxplot(data=df, x="gender", y="writing score")

plt.show()


# ### Conclusion
# 1. Math Score - Males have scored maximum score than Female.
# 2. Females have scored better in both reading and writing

# ### Patental Level of Education Vs Gender 

# In[ ]:


plt.figure(figsize=[20,5])

plt.subplot(1,3,1)
plt.xlabel('xlabel', fontsize=18)
plt.ylabel('ylabel', fontsize=18)
plt.xticks(rotation=90)
sns.boxplot(data=df, x="parental level of education", y="math score")

plt.subplot(1,3,2)
plt.xlabel('xlabel', fontsize=18)
plt.ylabel('ylabel', fontsize=18)
plt.xticks(rotation=90)
sns.boxplot(data=df, x="parental level of education", y="reading score")

plt.subplot(1,3,3)
plt.xlabel('xlabel', fontsize=18)
plt.ylabel('ylabel', fontsize=18)
plt.xticks(rotation=90)
sns.boxplot(data=df, x="parental level of education", y="writing score")

plt.show()


# #### Conclusion
# 
# Childrens whose parents have master degree has performed better

# ### Lunch Vs Gender 

# In[ ]:


plt.figure(figsize=[20,5])

plt.subplot(1,3,1)
plt.xlabel('xlabel', fontsize=18)
plt.ylabel('ylabel', fontsize=18)
sns.boxplot(data=df, x="lunch", y="math score")

plt.subplot(1,3,2)
plt.xlabel('xlabel', fontsize=18)
plt.ylabel('ylabel', fontsize=18)
sns.boxplot(data=df, x="lunch", y="reading score")

plt.subplot(1,3,3)
plt.xlabel('xlabel', fontsize=18)
plt.ylabel('ylabel', fontsize=18)
sns.boxplot(data=df, x="lunch", y="writing score")

plt.show()


# ### Test preparation course VS Score

# In[ ]:


plt.figure(figsize=[20,5])

plt.subplot(1,3,1)
plt.xlabel('xlabel', fontsize=18)
plt.ylabel('ylabel', fontsize=18)
sns.boxplot(data=df, x="test preparation course", y="math score")

plt.subplot(1,3,2)
plt.xlabel('xlabel', fontsize=18)
plt.ylabel('ylabel', fontsize=18)
sns.boxplot(data=df, x="test preparation course", y="reading score")

plt.subplot(1,3,3)
plt.xlabel('xlabel', fontsize=18)
plt.ylabel('ylabel', fontsize=18)
sns.boxplot(data=df, x="test preparation course", y="writing score")

plt.show()


# #### Conclusion
# Form the above we can clearly see that the students who have completed their test prepration course have performed better.`m

# In[ ]:




