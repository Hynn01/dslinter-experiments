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


data = pd.read_csv('/kaggle/input/vader-lexicon/vader_lexicon.txt', sep='\t', header=None)
df_sent = pd.DataFrame()
df_sent['token']=data[0]
df_sent['sentiment_mean']=data[1]
df_sent['sentiment_dev']=data[2]
df_sent['sentiment_values']=data[3]

# df_sent.set_index('token', inplace=True)
df_sent.head()


# In[ ]:


print(type(df_sent["sentiment_values"].values[0]))
df_sent["sentiment_values"].values[0]


# In[ ]:


str_ = df_sent["sentiment_values"].values[0]
str_to_list = [int(i) for i in str_[1:-1].split(",")]
print(type(str_to_list))
str_to_list


# In[ ]:


print(len(df_sent))


# In[ ]:


"happy" in df_sent["token"].values


# In[ ]:


"web" in df_sent["token"].values


# In[ ]:


import statistics

statistics.mean(str_to_list)     # -1.5


# In[ ]:


statistics.pstdev(str_to_list)    # 0.80623


# In[ ]:




