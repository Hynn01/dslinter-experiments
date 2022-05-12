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


df_cpu = pd.read_csv('/kaggle/input/cpu-benchmarks/CPU_benchmark.csv')


# In[ ]:


df_cpu


# In[ ]:


df_cpu['cpuName'].unique


# In[ ]:


df_cpu.columns


# In[ ]:


df_cpu['cores'].value_counts()


# In[ ]:


df_cpu.query('cores == 5')


# In[ ]:


df_cpu.query('cores == 4')


# In[ ]:


df_cpu['category'].value_counts()


# In[ ]:


df_cpu['testDate'].value_counts()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="ticks")


# In[ ]:


sns.pairplot(df_cpu, hue="testDate")


# In[ ]:


sns.set_theme()
plt.figure(figsize = (15,8))
ax = sns.histplot(df_cpu, x='testDate')
plt.xticks(rotation=90)
ax.bar_label(ax.containers[0])


# In[ ]:


sns.set_theme()
plt.figure(figsize = (15,8))
ax = sns.histplot(df_cpu, x='testDate', hue='cores')
plt.xticks(rotation=90)
ax.bar_label(ax.containers[0])


# In[ ]:


df_cpu_cat = df_cpu.sort_values(by=['category'])
df_cat = df_cpu['category'].value_counts().to_frame().sort_values(by=['category'])
df_cat


# In[ ]:


sns.set_theme()
plt.figure(figsize = (15,8))
ax = sns.histplot(df_cpu_cat, x='category')
plt.xticks(rotation=90)
ax.bar_label(ax.containers[0])


# In[ ]:




