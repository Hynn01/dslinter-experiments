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


# Read the dataset
main_df = pd.read_csv("/kaggle/input/spotify-top-100-songs-of-20152019/Spotify 2010 - 2019 Top 100.csv")
df = main_df.copy()
df


# In[ ]:


# Imports
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Shape
df.shape


# In[ ]:


# Missing values in the dataset
df.isnull().sum()


# In[ ]:


# Delete last three rows
df = df.iloc[:-3 , :]


# In[ ]:


# Missing values in the dataset, should be 0
df.isnull().sum()


# In[ ]:


# Check for duplicates
titleUnique = len(set(df.title))
titlesTotal = df.shape[0]
titlesdupe = titlesTotal - titleUnique
print(titlesdupe)
# No need to drop as soome artists have mulitple hits
#train.drop(['Id'],axis =1,inplace=True)


# In[ ]:


# Dataset info
df.info()


# In[ ]:


df.describe()


# In[ ]:


df.corr()


# In[ ]:


df['year released'] = df['year released'].astype('Int64')
df['top year'] = df['top year'].astype('Int64')


# In[ ]:


sns.pairplot(df)


# In[ ]:


sns.heatmap(df.corr(), annot=True)


# In[ ]:


sns.boxplot(df['top year'])


# In[ ]:


df.groupby('top genre')['top genre'].count().sort_values(ascending=False)


# In[ ]:


px.scatter(df, x='year released',y='top year', color=df['artist'])


# In[ ]:


px.density_heatmap(df, y="top year", x="artist", nbinsx=20, nbinsy=20)


# In[ ]:


idsUnique = len(set(df.title))
idsTotal = df.shape[0]
idsdupe = idsTotal - idsUnique
print(idsdupe)

