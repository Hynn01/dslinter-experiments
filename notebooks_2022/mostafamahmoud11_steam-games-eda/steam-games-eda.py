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


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv(r'../input/steam-games-hours-played-and-peak-no-of-players/Book 1.csv')


# In[ ]:


df.info()


# In[ ]:


df['Hours Played']=df['Hours Played'].str.replace(',','').astype('float64')


# In[ ]:


df['Peak No. of Players ']=df['Peak No. of Players '].str.replace(',','').astype('float64')


# In[ ]:


df.head()


# In[ ]:


top10gamesplayed=df.sort_values(by='Hours Played',ascending=False).head(10)


# In[ ]:


sns.set_style("darkgrid")
plt.figure(figsize=(16,8))
sns.barplot(x='Hours Played',y='\nName',data=top10gamesplayed,palette='hls')
plt.ylabel('Hours Played',fontsize=15)
plt.xlabel('Game Name',fontsize=15)
plt.title('Top 10 Games By Hours Played',fontsize=18)


# In[ ]:


top10gamesbyplayers=df.sort_values(by='Peak No. of Players ',ascending=False).head(10)


# In[ ]:


plt.figure(figsize=(16,8))
sns.barplot(x='Peak No. of Players ',y='\nName',data=top10gamesbyplayers,palette='viridis')
plt.ylabel('Number Of players',fontsize=15)
plt.xlabel('Game Name',fontsize=15)
plt.title('Top 10 Games By Players Count',fontsize=18)


# In[ ]:


plt.figure(figsize=(16,8))
sns.lmplot(x='Peak No. of Players ',y='Hours Played',data=df,scatter_kws={"s": 780}, 
           line_kws={"lw":5})
plt.ylabel('Hours Played',fontsize=15)
plt.xlabel('Number OF players',fontsize=15)


# In[ ]:





# Thanks Upvote if You like it :)

# In[ ]:




