#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import pandas as pd 
import numpy as np


# In[ ]:


df=pd.read_csv('../input/sf-salaries/Salaries.csv')


# In[ ]:


df.head()


# In[ ]:


#info about the columns

df.info()


# In[ ]:


# statistical information

df.describe()


# In[ ]:


# convert numeric columns to float instead of object.

df['BasePay'] = pd.to_numeric(df['BasePay'],errors='coerce')
df['OvertimePay'] = pd.to_numeric(df['OvertimePay'],errors='coerce')
df['OtherPay'] = pd.to_numeric(df['OtherPay'],errors='coerce')
df['Benefits'] = pd.to_numeric(df['Benefits'],errors='coerce')
df['TotalPay'] = pd.to_numeric(df['TotalPay'],errors='coerce')
df['TotalPayBenefits'] = pd.to_numeric(df['TotalPayBenefits'],errors='coerce')

# convert string columns to string instead of object.

df['EmployeeName'] = df['EmployeeName'].astype('string')
df['JobTitle'] = df['JobTitle'].astype('string')
df['Agency'] = df['Agency'].astype('string')
df['Status'] = df['Status'].astype('string')


# In[ ]:


#the mean of basepay in SF

df['BasePay'].mean()


# In[ ]:


#the maximum overtimepay in SF

df['OvertimePay'].max()


# In[ ]:


# the job title of JOSEPH DRISCOLL

df[df['EmployeeName']=='JOSEPH DRISCOLL']['JobTitle']


# In[ ]:


df[df['EmployeeName']=='JOSEPH DRISCOLL']['TotalPayBenefits']


# In[ ]:


#the EmployeeName with the maximum salary 

df[df['TotalPayBenefits'] == df['TotalPayBenefits'].max()]['EmployeeName']


# In[ ]:


#the EmployeeName with the minimum salary 

df[df['TotalPayBenefits'] == df['TotalPayBenefits'].min()]['EmployeeName']


# In[ ]:


# what was the average BasePay of all employees per year? (2011-2014)?
df.groupby('Year').mean()['BasePay']


# In[ ]:


#how many unique job titles are there 

df['JobTitle'].nunique()


# In[ ]:


# what are the most 5 common jobs

df['JobTitle'].value_counts()[:5]


# In[ ]:


# what are the less common jobs display 10

df['JobTitle'].value_counts().tail(10)


# In[ ]:


# how many job titles were represented by only one person in 2013 

(df[df['Year']==2013]['JobTitle'].value_counts()==1).sum()


# In[ ]:


def chief(string):
    if 'chief' in (string.lower()):
        return True
    else:
        return False


# In[ ]:


#how many people have the word cheif in their job title 

(df['JobTitle'].apply(lambda x: chief(x))).sum()


# In[ ]:


# Is there a corolation between length of the job title string and the salary ??!!

df['ntitles']=df['JobTitle'].apply(len)
df['ntitles']


# In[ ]:


# Is there a corolation between length of the job title string and the salary ??!!

df[['ntitles','TotalPayBenefits']].corr()

# No there is no relation between both.

