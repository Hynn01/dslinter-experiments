#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df=pd.read_csv('/kaggle/input/steam-games-hours-played-and-peak-no-of-players/Book 1.csv')
for a in ['Hours Played','Peak No. of Players ']:
    df[a]=[i.replace(',','') for i in df[a]]
    df[a]=df[a].astype('int64')


# In[ ]:


df.sort_values(by='Hours Played', inplace=True)
plt.barh(width=df['Hours Played'].head(), y=df['\nName'].head())
plt.title('Top 5 games by hours played')


# In[ ]:


df.sort_values(by='Peak No. of Players ', inplace=True)
plt.barh(width=df['Peak No. of Players '].head(10), y=df['\nName'].head(10))
plt.title('Top 10 games by peak number of players')


# In[ ]:


sns.heatmap(df.corr(),annot=True)
plt.title('Correlation matrix by columns')


# In[ ]:


sns.regplot(x=df['Hours Played'],y=df['Peak No. of Players '])
plt.title('Hours played vs. Peak No. of Players')
plt.xlabel('Hours Played')
plt.ylabel('Peak No. of Players')

