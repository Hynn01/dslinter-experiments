#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Lets talk Ramen! 
# 
# ### First things first, lets take a look at our dataset.

# In[ ]:


# Getting the CVS into a Pandas DataFrame we can work with and shape.
df = pd.read_csv('/kaggle/input/ramen-ratings/ramen-ratings.csv')
# Now lets take a look at what we have
df = df.head(200)
df


# In[ ]:


df['Country'].unique()


# # Thinking about our raw data.
# Okay, So we have a dataframe, (there is a good ammount of ramen in here!). Lets take a look at our columns and think about what we can ask.
# Lets chart each countries average ramen rating.

# In[ ]:


grouped_by_country = df.groupby('Country')
grouped_by_country.first()
stars = df['Stars'].values
countries = df['Country'].values
plt.rcParams["figure.figsize"] = (20,18.5)
sns.relplot(x=countries, y= stars, data = df,size ='Stars',  kind='line')


# In[ ]:


df['Country'].unique()

grouped_by_country = df.groupby('Country')
grouped_by_country.first()
stars = df['Stars'].unique()
countries = df['Country'].unique()

df_sorted = df.sort_values(by=['Country','Stars'])
df_sorted = df_sorted.drop(df.columns[1:4], axis=1)
df_sorted = df_sorted.drop(df.columns[-1], axis=1)
df_sorted


# In[ ]:


df_sorted = df_sorted.sort_values('Stars', ascending= False) 

sns.relplot(x=df_sorted['Country'], y= df_sorted['Stars'], data = df_sorted,size ='Stars')
plt.gcf().set_size_inches(25,8)


# ### No surprise here, It looks like Asian countries tend to dominate the ramen space, interestingly though the US doesnt do too bad,
# We apear to have some pretty well rated ramen in the states, But fall off somewhere in the high middle ratings. 
