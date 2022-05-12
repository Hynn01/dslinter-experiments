#!/usr/bin/env python
# coding: utf-8

# # Data Analysis of NYC Jobs Openings 2022

# Let's start with importing libraries and data.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics


# In[ ]:


df = pd.read_csv('../input/nyc-jobs-openings-2022/NYC_Jobs.csv')
df.head()


# To look at the column names one by one and see if there are any null values:

# In[ ]:


df.info()


# In[ ]:


df.shape


# There is too much NA in the dataset we need to see where so that we can drop them .

# In[ ]:


df.isnull().sum(axis = 0)


# Dropping nulls and columns that we never use

# In[ ]:


df = df.drop(labels=['Additional Information', 'Hours/Shift', 'Work Location 1', 'Recruitment Contact', 'Post Until', 
                    'Title Code No', 'Level', 'To Apply', 'Posting Date', 'Posting Updated', 'Process Date'], axis=1)


# In[ ]:


df.isnull().sum(axis = 0)


# Let's look at same columns.

# In[ ]:


# Job Category
JC = df['Job Category'].value_counts()[:10]
plt.figure(figsize=(10,10))
res=sns.barplot(x=JC, y=JC.index)
res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 16, color='black')
plt.xlabel('Value Counts of Job Category',fontsize = 16, color='black')
plt.ylabel('Top 10 Job Category Names',fontsize = 16, color='black')
plt.title('Job Categories in NYC',fontsize = 16, color='black')
plt.show()


# The first two does not surprise me. I guess it's like that everywhere :D

# In[ ]:


# Civil Service Title
CS = df['Civil Service Title'].value_counts()[:10]
plt.figure(figsize=(10,10))
res=sns.barplot(x=CS, y=CS.index)
res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 16, color='black')
plt.xlabel('Value Counts of Civil Service Title',fontsize = 16, color='black')
plt.ylabel('Top 10 Civil Service Titles',fontsize = 16, color='black')
plt.title('Civil Service Titles in NYC',fontsize = 16, color='black')
plt.show()


# In[ ]:


# Full-Time/Part-Time indicator
FPI = df['Full-Time/Part-Time indicator'].value_counts()
plt.figure(figsize=(10,10))
res=sns.barplot(x=FPI, y=FPI.index)
res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 16, color='black')
plt.xlabel('Value Counts of Full-Time/Part-Time Indicator',fontsize = 16, color='black')
plt.ylabel('Full-Time/Part-Time Indicators',fontsize = 16, color='black')
plt.title('Full-Time/Part-Time Indicators in NYC',fontsize = 16, color='black')
plt.show()


# In[ ]:


# Career Level
CL = df['Career Level'].value_counts()
plt.figure(figsize=(10,10))
res=sns.barplot(x=CL, y=CL.index)
res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 16, color='black')
plt.xlabel('Value Counts of Career Level',fontsize = 16, color='black')
plt.ylabel('Career Levels',fontsize = 16, color='black')
plt.title('Career Levels in NYC',fontsize = 16, color='black')
plt.show()


# It would not be correct to take the mean of the salaries, as a wide range of employment is made from the intern to the manager. We could have taken an mean while evaluating each level among themselves, but we do not do that here. So we take the median.

# In[ ]:


print(statistics.median(df['Salary Range From']))
print(statistics.median(df['Salary Range To']))

