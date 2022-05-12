#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('../input/rollercoaster-database/coaster_db.csv')
df


# In[ ]:


a = df.isna().sum()/len(df)
for item, value in a.items():
    if value >= .3:
        print(f'We will now drop {item} column with {value*100} % MISSING')
        df = df.drop(str(item), axis=1)


# In[ ]:


df


# In[ ]:


fig, ax = plt.subplots(7,4,figsize=(20,30))
ax = ax.flatten()
for i, col in enumerate(df.columns):    
#     print(f'Value counts in column : {col}')
    val = df[col].value_counts()
    val.plot(kind='kde', ax=ax[i], title=f'Count plot for {col}')
    ax[i].set_xlabel('Count')
    #print('\n','*'*50,'\n')
plt.tight_layout()


# In[ ]:


# Remove other location
df = df.query('Location != "Other"')


# In[ ]:


ax = df.groupby('Location').speed_mph.mean().dropna().sort_values(ascending=False).loc[lambda x:x>70].plot(kind='barh', title='Locations with Speed (>70) on average')
ax.set_xlabel('Speed (MPH)')


# In[ ]:


ax = df.groupby('Location')['speed_mph'].agg(['mean', 'count','max','min', 'median']) .query('count > 10').sort_values('min').plot.bar(figsize=(20,15))
# ax.set_ylabel('Values')
plt.show()


# In[ ]:


df.Type_Main.unique()


# In[ ]:


sns.pairplot(data=df, vars=['Inversions',
 'year_introduced',
 'latitude',
 'longitude',
 'speed1_value',
 'speed_mph',
 'height_value',
 'Inversions_clean'], hue='Type_Main')


# In[ ]:




