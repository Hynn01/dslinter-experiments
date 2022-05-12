#!/usr/bin/env python
# coding: utf-8

# # Introduction

# We introduce how to load the Spanish Wine Quality Dataset.

# # Analysis preparation

# ## Load packages

# In[ ]:


import pandas as pd
import os


# ## Load data

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


wine_data = pd.read_csv("/kaggle/input/spanish-wine-quality-dataset/wines_SPA.csv", sep=",")


# In[ ]:


wine_data.head()


# In[ ]:


import seaborn as sns
g = sns.pairplot(data=wine_data, diag_kind="kde", dropna=True)
g.map_lower(sns.kdeplot, levels=4, color=".2")

