#!/usr/bin/env python
# coding: utf-8

# # Intro

# In this notebook you will see how I cleaned [this dataset from UCI machine learning repository](https://archive.ics.uci.edu/ml/datasets/Credit+Approval?msclkid=200008bdc4a311ec9f500a3245a2bfb1) on **credit card approval** to create a **clean, easier to use version**. You can find the [resulting dataset on kaggle here](https://www.kaggle.com/datasets/samuelcortinhas/credit-card-approval-clean-data).
# 
# In particular, this notebook takes care of:
# * **missing feature names**,
# * **missing values**.

# # Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.4)


# # Data

# In[ ]:


# Save to df
data = pd.read_csv('../input/credit-card-approval-clean-data/crx.csv', header = None)

# Shape and preview
print('Dataset shape:', data.shape)
data.head()


# # Data Cleaning

# **Feature names**

# The features of the orginal dataset were **anonymised** to protect the **privacy** of the clients, but [this blog post](http://rstudio-pubs-static.s3.amazonaws.com/73039_9946de135c0a49daa7a0a9eda4a67a72.html) has provided a list of the probable feature names. Some of them I have changed based on my own assumptions.

# * **Gender**          : chr  "a" "a" "b" "b" "b" ...
# * **Age**           : chr  "58.67" "24.50" "27.83" "20.17" ...
# * **Debt**          : num  4.46 0.5 1.54 5.62 4 ...
# * **Married**       : chr  "u" "u" "u" "u" ...
# * **BankCustomer**  : chr  "g" "g" "g" "g" ...
# * **Industry**      : chr  "q" "q" "w" "w" ...
# * **Ethnicity**     : chr  "h" "h" "v" "v" ...
# * **YearsEmployed** : num  3.04 1.5 3.75 1.71 2.5 ...
# * **PriorDefault**  : num  1 1 1 1 1 1 1 1 1 0 ...
# * **Employed**      : num  1 0 1 0 0 0 0 0 0 0 ...
# * **CreditScore**   : num  6 0 5 0 0 0 0 0 0 0 ...
# * **DriversLicense**: chr  "f" "f" "t" "f" ...
# * **Citizen**       : chr  "g" "g" "g" "s" ...
# * **ZipCode**       : chr  "00043" "00280" "00100" "00120" ...
# * **Income**        : num  560 824 3 0 0 ...
# * **Approved**      : chr  "+" "+" "+" "+" ...

# 
# Note: these might **not** be **completely accurate** but at least they provide some **context** to the dataset.

# In[ ]:


# Rename all columns
data.columns = ['Gender', 'Age', 'Debt', 'Married', 'BankCustomer','Industry','Ethnicity','YearsEmployed','PriorDefault','Employed','CreditScore','DriversLicense','Citizen','ZipCode','Income','Approved']


# **Missing values**

# Missing values are denoted by '?' in the original dataset.

# In[ ]:


# Calculate missing values
mv_df=pd.DataFrame(columns = ['Feature', 'NumberMissing', 'PercentageMissing'])
for col in data.columns:
    mv_df=mv_df.append({'Feature':col, 'NumberMissing':(data[col]=='?').sum(), 'PercentageMissing': np.round(100*(data[col]=='?').sum()/len(data),2)},  ignore_index=True)

# Show dataframe
mv_df


# **Gender**

# In[ ]:


data['Gender'].value_counts()


# In[ ]:


# Rename entries
data['Gender'].replace('a', 0, inplace=True)
data['Gender'].replace('b', 1, inplace=True)

# Fill missing values with mode
data['Gender'].replace('?', 1, inplace=True)

# Convert to integer type
data['Gender']=data['Gender'].astype(int)


# **Age**

# In[ ]:


# Identify median age
median_age=data.loc[data['Age']!='?','Age'].median()
print('Median age', median_age)

# Fill missing values with median
data.loc[data['Age']=='?','Age']=median_age

# Convert to float type
data['Age']=data['Age'].astype(float)


# **Married**

# In[ ]:


data['Married'].value_counts()


# These likely refer to single (y), married (u) and divorced (l). 

# In[ ]:


# Fill missing values with mode
data.loc[data['Married']=='?','Married']='u'

# Combine divorced with single (as there are only 2 entries)
data.loc[data['Married']=='l','Married']='y'

# Convert to binary feature (u=Married, y=Single/Divorced)
data.loc[data['Married']=='u','Married']=1
data.loc[data['Married']=='y','Married']=0

# Convert to int type
data['Married']=data['Married'].astype(int)


# **BankCustomer**

# In[ ]:


data['BankCustomer'].value_counts()


# These categories could refer to having a bank account (g) and not having a bank account (p). 

# In[ ]:


# Fill missing values with mode
data.loc[data['BankCustomer']=='?','BankCustomer']='g'

# Combine gg with g (as there are only 2 entries)
data.loc[data['BankCustomer']=='gg','BankCustomer']='g'

# Convert to binary feature (gg=has bank account, p=does not have bank account)
data.loc[data['BankCustomer']=='g','BankCustomer']=1
data.loc[data['BankCustomer']=='p','BankCustomer']=0

# Convert to int type
data['BankCustomer']=data['BankCustomer'].astype(int)


# **Industry**

# Industry refers to the sector that a person works in (if they are currently unemployed then it refers to their most recent job).

# In[ ]:


data['Industry'].value_counts()


# In[ ]:


# Fill missing values with mode
data.loc[data['Industry']=='?','Industry']='c'

# Rename categories
data['Industry'].replace({'c':'Energy','q':'Materials','w':'Industrials','i':'ConsumerDiscretionary','aa':'ConsumerStaples','ff':'Healthcare','k':'Financials','cc':'InformationTechnology','m':'CommunicationServices','x':'Utilities','d':'Real Estate','e':'Education','j':'Research','r':'Transport'}, inplace=True)


# **Ethnicity**

# In[ ]:


data['Ethnicity'].value_counts()


# In[ ]:


# Fill missing values with mode
data.loc[data['Ethnicity']=='?','Ethnicity']='v'

# Combine minority groups together
data.loc[data['Ethnicity'].isin(['j','z','dd','n','o']),'Ethnicity']='Other'

# Rename categories
data['Ethnicity'].replace({'v':'White','h':'Black','bb':'Asian','ff':'Latino'}, inplace=True)


# **PriorDefault**

# This feature does not have missing values but we will rename the entries to make it binary.

# In[ ]:


# Rename entries
data['PriorDefault'].replace('t', 1, inplace=True)
data['PriorDefault'].replace('f', 0, inplace=True)


# **Employed**

# Same with this feature.

# In[ ]:


# Rename entries
data['Employed'].replace('t', 1, inplace=True)
data['Employed'].replace('f', 0, inplace=True)


# **DriversLicense**

# And this one.

# In[ ]:


# Rename entries
data['DriversLicense'].replace('t', 1, inplace=True)
data['DriversLicense'].replace('f', 0, inplace=True)


# **Citizen**

# In[ ]:


data['Citizen'].value_counts()


# This feature does not have missing values. I will interpret the categories as citizen by birth (g), citizen by other means (s) and temporary citizen (p).

# In[ ]:


# Rename entries
data['Citizen'].replace('g', 'ByBirth', inplace=True)
data['Citizen'].replace('s', 'ByOtherMeans', inplace=True)
data['Citizen'].replace('p', 'Temporary', inplace=True)


# **ZipCode**

# In[ ]:


data['ZipCode'].value_counts()


# In[ ]:


# Fill missing values with mode
data.loc[data['ZipCode']=='?','ZipCode']='00000'


# **Approved**

# In[ ]:


# Rename entries (1 is approved, 0 is not approved)
data['Approved'].replace('-', 0, inplace=True)
data['Approved'].replace('+', 1, inplace=True)


# # Output

# In[ ]:


# Preview result
data.head()


# In[ ]:


# Data types
data.dtypes


# In[ ]:


# Save to csv
data.to_csv('clean_dataset.csv',index=False)


# **Acknowledgments:**
# * [Analysis of Credit Approval Data](http://rstudio-pubs-static.s3.amazonaws.com/73039_9946de135c0a49daa7a0a9eda4a67a72.html) by Ryan Kuhn.
# * [Credit Card Approval Predictions](https://www.kaggle.com/code/muhammadahmed68/credit-card-approval-predictions-85-accuracy/notebook) by Moezilda.
