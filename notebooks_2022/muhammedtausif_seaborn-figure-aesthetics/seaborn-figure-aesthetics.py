#!/usr/bin/env python
# coding: utf-8

# # Customising Figure Asthetics in Seaborn

# ## Loading libraries

# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ## Sin-wave function

# In[ ]:


def sinplot(flip=1, start = 0, step = 14, end = 100, r_start=1, r_end=7):
    x = np.linspace(start,step,end)
    for i in range(r_start,r_end):
        plt.plot(x, np.sin(x+i*.5)*(7-i)* flip)


# In[ ]:


sinplot(1)


# In[ ]:


sinplot(5,5)


# In[ ]:


sinplot(3, 5, 1)


# In[ ]:


sns.set_theme()
sinplot()


# ## Seaborn Figure Style

# Set style to **darkgrid**, there are five seaborn themes:
# 1. darkgrid
# 2. whitegrid
# 3. dark
# 4. white
# 5. ticks

# In[ ]:


sns.set_style('dark')


# Load the data randomly

# In[ ]:


data = np.random.normal(size=(20,6)) + np.arange(6)/2


# Box Plot with Seaborn

# In[ ]:


sns.boxplot(data=data)


# In[ ]:


sns.set_style('darkgrid')


# In[ ]:


sinplot()


# ## Removing Axes Spines

# In[ ]:


sinplot(4,5)
sns.despine()


# In[ ]:


fig, axis = plt.subplots()
sns.violinplot(data=data)
sns.despine(offset=10, trim=True)


# **Working on**
# For more detail [visit](https://seaborn.pydata.org/tutorial/aesthetics.html)

# In[ ]:




