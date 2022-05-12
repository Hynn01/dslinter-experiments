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
plt.style.use('ggplot')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('../input/wordle-tweets/tweets.csv')
df


# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


df.tweet_text.str[7:10].astype('int')


# In[ ]:


def process_tweets(df):
    df['tweet_datetime'] = pd.to_datetime(df['tweet_date'])
    df['tweet_date'] = df['tweet_datetime'].dt.date
    df['wordle_id'] = df.tweet_text.str[:10]
    df['n_attempts'] = df.tweet_text.str[11].astype('int')
    df['id'] = df.tweet_text.str[7:10].astype('int')
    return df
df = process_tweets(df)

df


# In[ ]:


print(df.tweet_text[0])


# # No. of tweets on each day

# In[ ]:


# _, ax = plt.subplots(3,3,figsize=(20,30))
# ax = ax.flatten()

for i, col in enumerate(df.columns):
    print('Value counts for the col', col,'\n')
    display(df[col].value_counts().head(20))
#     val = df[col].value_counts().head(10)
#     val.plot(kind='barh', ax=ax[i])
#     ax[i].set_ylabel(f'{col}')
    #ax[i].set_xlabel('')
    print('\n\n')


# In[ ]:


df['tweet_date'].value_counts().plot(figsize=(20,15), lw=2)


# # No of attempts analysis

# In[ ]:


df.groupby('wordle_id').n_attempts.value_counts().unstack().style.background_gradient(axis=1)


# In[ ]:


df.groupby('wordle_id').count()


# In[ ]:




