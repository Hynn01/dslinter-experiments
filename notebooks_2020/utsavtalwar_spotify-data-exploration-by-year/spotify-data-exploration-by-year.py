#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/top-spotify-songs-from-20102019-by-year/top10s.csv", encoding='ISO-8859-1', index_col=0)


# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()


# In[ ]:


df1 = data.groupby(['year']).median()


# In[ ]:


df1['bpm']


# In[ ]:


data.groupby(['year']).max()[['artist','title']]


# Most popular songs and artist by year

# In[ ]:


import seaborn as sns
sns.lineplot(x='year', y='bpm', data = data)


# BPM of songs have droped over the years.

# In[ ]:


sns.lineplot(x='year', y='dB', data = data)


# There is also a slight drop in the loudness of the songs, refers that audience like less loud songs. 

# In[ ]:


sns.lineplot(x='year', y='dur', data = data)


# We see a heavy drop in the duration of the songs from year 2018 onwards

# In[ ]:


sns.lineplot(x='year', y='spch', data = data)


# In[ ]:


sns.lineplot(x='year', y='acous', data = data)


# In[ ]:


sns.lineplot(x='year', y='live', data = data)


# In[ ]:


data['top genre'].value_counts().head(20).plot.pie(figsize=(15,10), autopct='%0.0f%%')


# Dance pop is the most popular song genre

# > Thanks for going through, please upvote if you found it insightful. Thanks!
