#!/usr/bin/env python
# coding: utf-8

# # Seaborn Project
# 
# Visualization project

# ## The Data
# 
# I will be working with a famous titanic data set 

# In[ ]:


import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize': [9, 9]}, font_scale=1.2)


# In[ ]:


df = sns.load_dataset('titanic')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


sns.distplot(df['age'], kde=False, bins=30, color='m')


# In[ ]:


sns.kdeplot(df['age'], shade=True, color='m')


# In[ ]:


sns.distplot(df['fare'], kde=False, bins=30, color='m')


# In[ ]:


sns.kdeplot(df['fare'], shade=True, color='m')


# In[ ]:


sns.jointplot(x='age', y='fare', data=df, color='m')


# In[ ]:


df.head()


# In[ ]:


sns.countplot(x='survived', data=df, palette='viridis')


# In[ ]:


sns.countplot(x='class', data=df, hue='alive', palette='viridis')


# In[ ]:


sns.countplot(x='class', data=df, hue='sex', palette='BuPu')


# In[ ]:


sns.countplot(x='class', data=df, hue='who', palette='Set1')


# In[ ]:


sns.countplot(x='who', data=df, hue='alive', palette='viridis')


# In[ ]:


sns.countplot(x='class', data=df, hue='alone', palette='Set2')


# In[ ]:


sns.countplot(x='embark_town', data=df)


# In[ ]:


sns.countplot(x='embark_town', data=df, hue='alive')


# In[ ]:


sns.countplot(x='class', data=df, hue='embark_town', palette='viridis')


# In[ ]:


df.head()


# In[ ]:


sns.boxplot(x='survived', y='age', data=df)


# In[ ]:


sns.boxplot(x='class', y='age', data=df)


# In[ ]:


sns.violinplot(x='class', y='age', data=df, hue='survived', split=True)


# In[ ]:


sns.stripplot(x='embark_town', y='age', data=df, hue='survived', dodge=True)


# In[ ]:


sns.swarmplot(x='embark_town', y='age', data=df, hue='survived', dodge=True)


# In[ ]:


sns.violinplot(x='who', y='fare', data=df, hue='survived', split=True)


# In[ ]:


import numpy as np


# In[ ]:


sns.barplot(x='class', y='fare', data=df, estimator=np.sum, hue='embark_town')


# In[ ]:


df.head()


# In[ ]:


df_corr = df.corr()
df_corr


# In[ ]:


sns.heatmap(df_corr, cmap='viridis', linecolor='k', linewidths=2, annot=True)


# 
# ### That is it for now!
