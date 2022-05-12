#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import re
sns.set_theme()

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('/kaggle/input/vegetation-fires-in-cape-town/fire_log_cape_town.csv')
df.info()


# # Parsing Dates
# I don't know about you, but I really LOVE parsing specifics from datetime objects. It really makes me happy.

# In[ ]:


df2 =  df.iloc[:,[1,3,4,14,18]].copy()
df2['Datetime'] = pd.to_datetime(df2['Datetime'])
df2['day']  = df2['Datetime'].dt.day_name()
df2['year'] = df2['Datetime'].dt.year
df2['month'] = df2['Datetime'].dt.month_name()
df2.head()


# In[ ]:


df2.info()


# In[ ]:


df2.iloc[:,5].value_counts(normalize=True).apply(lambda x: x*100).sort_values(ascending=False)


# In[ ]:


df2.iloc[:,7].value_counts(normalize=True).apply(lambda x: x*100).sort_values(ascending=False)


# In[ ]:


plt.figure(figsize=(20,7))
sns.countplot(y='Wind', data=df2, palette='magma')
plt.show()


# In[ ]:


fig, axes = plt.subplots(1,2, figsize=(30,7))
plt.suptitle('Day - Month Frequency',fontsize=24)
sns.countplot(x='day', data=df2, ax=axes[0],palette='ocean', order=['Sunday','Monday',
                                                                     'Tuesday','Wednesday',
                                                                     'Thursday','Friday',
                                                                    'Saturday'])
axes[0].set_title('Day Frequency')
sns.countplot(x='month', data=df2, ax=axes[1], palette='magma')
axes[1].set_title('Month Frequency')
plt.show()


# In[ ]:




