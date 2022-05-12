#!/usr/bin/env python
# coding: utf-8

# # EDA Pokemon Dataset Analysis and Visualization

# ***
# ## 1. Initial Data Analysis
# We start by importing the libraries we will use through this Data Analysis.

# In[ ]:


# Data analysis and wrangling
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# We load the Pokemon dataset and drop the first column.

# In[ ]:


pokedex = pd.read_csv('../input/complete-pokemon-dataset-updated-090420/pokedex_(Update_05.20).csv')
pokedex = pokedex.drop(['Unnamed: 0'], axis=1)


# We take a first look at the Dataset.

# In[ ]:


pokedex.info()


# In[ ]:


pokedex.head(10)


# ## Preliminar Analysis

# We count the Null values in the Dataset

# In[ ]:


null_filter = pokedex.isna().sum()
missing_values = null_filter.loc[pokedex.isna().sum() != 0].to_frame().copy()
missing_values


# In[ ]:


missing_values.columns = ['count']
missing_values['Name'] = missing_values.index
missing_values.reset_index(drop=True, inplace=True)
missing_values

sns.barplot(x='Name', y='count', data=missing_values);
plt.xticks(rotation=90);


# ## Analysing the Pokedex Data
# 
# How many Pokemon are in each Generation?

# In[ ]:


ax = sns.catplot(x="generation", kind="count", data=pokedex);
ax.set(xlabel='Generation', ylabel='Nr of Pokemon', title='Number of Pokemon in each Generation');


# How many Sub-Legendary, Legendary and Mythical Pokemon are in each generation?

# In[ ]:


sns.catplot(x="generation", col="status", kind="count", data=pokedex);


# We leave out the normal Pokemon for better comparing of the Sub Legendary, Legendary and Mythical Pokemon

# In[ ]:


poke_filter = pokedex.loc[pokedex.status != "Normal"]
sns.catplot(x="generation", col="status", kind="count", data=poke_filter);


# Let's analyse the distribution of Pokemon Species

# In[ ]:


pokedex.species.value_counts()


# In[ ]:


pokedex.groupby('generation').species.describe()


# In[ ]:


sns.catplot(x='generation', y='height_m', data=pokedex);


# We can see 2 Outliers in the 8th Generation. We will plot again the graph without the outliers.

# In[ ]:


pok_height_out = pokedex[pokedex.height_m < 20]
sns.catplot(x='generation', y='height_m', data=pok_height_out);


# In[ ]:


sns.catplot(x='generation', y='height_m', kind='box', data=pok_height_out);


# In[ ]:


pokedex.height_m.mean()


# We can see that the majority of Pokemon don't exceed 3 or 4 meters, with a solid average around 1.4 m

# In[ ]:


sns.catplot(x='generation', y='weight_kg', data=pokedex);


# In[ ]:


sns.catplot(x='generation', y='weight_kg', kind='box', data=pokedex);


# In[ ]:


ax = sns.relplot(x='height_m', y='weight_kg', hue='generation', legend='full',palette='Set1', data=pokedex);
ax.set(xlim=(0, None), ylim=(0, None));


# ##  Analysing the base stats

# In[ ]:


sns.set_style('whitegrid')
g = sns.relplot(x='pokedex_number', y='attack', kind="line", hue = 'generation', palette='Set1', height = 8, aspect = 4, data=pokedex)
g.set(xlim=(0, None));
# g.fig.autofmt_xdate()


# In[ ]:


g = sns.relplot(x='pokedex_number', y='defense', kind="line", hue = 'generation', palette='Set1', height = 8, aspect = 4, data=pokedex)
g.set(xlim=(0, None));
g.fig.autofmt_xdate()


# In[ ]:


ax = sns.relplot(x='attack', y='defense', hue='generation', legend='full',palette='Set1', data=pokedex);
ax.set(xlim=(0, None), ylim=(0, None));

