#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
sns.set() # set seaborn plotting aesthetics as default

## to allow multiple printing statements from cell
#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"

import warnings
warnings.filterwarnings('ignore')

# Configure notebook display settings to only use 2 decimal places, tables look nicer.
pd.options.display.float_format = '{:,.4f}'.format

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/train.csv", index_col=0)')


# In[ ]:


train.target.value_counts()


# Broadly even spread of target values.

# In[ ]:


# Sets matplotlib figure size defaults to 25x20
plt.rcParams["figure.figsize"] = (25,20)
fig, ax = plt.subplots(5, 6)

i, j = (0, 0) # col/row
for col in train.columns: # don't include id
    if col not in ['f_27', 'target']: #dont plot f_27 or target feature-will error
        ax[j, i].hist(train[col], bins=100) #plots histogram on subplot [j, i]
        ax[j, i].set_title(col, #adds a title to the subplot
                           {'size': '14', 'weight': 'bold'}) 
        if i == 5: #if we reach the last column of the row, drop down a row and reset
            i = 0
            j += 1
        else: #if not at the end of the row, move over a column
            i += 1
            
plt.show()


# Lots of very normal distributions. Others that could be poisson, gamma, exponential? Couple of discrete features.

# In[ ]:


# Displays Pearson correlation coefficients of each feature in relation to one another
# This value ranges from -1 to 1... 0 means no correlation
plt.figure(figsize=(30, 2))         # changes figure size
sns.heatmap(train.corr()[-1:],      # takes correlations of only target feature
            cmap="viridis",         #changes the color palete
            annot=True)             #display coeficient value

plt.title('Correlation with target feature', {'size': '35'})
plt.show()


# In[ ]:


plt.figure(figsize=(30, 30))
sns.heatmap(train.corr(),
            cmap="viridis",
            annot=True)

plt.title('Correlation of all features', {'size': '35'})
plt.show()


# In[ ]:


plt.rcParams["figure.figsize"] = (25,20)
fig, ax = plt.subplots(5, 6)

i, j = (0, 0)
for col in train.columns:
    if col not in ['f_27', 'target']:
        sns.kdeplot(data=train, x=col, hue="target", ax=ax[j, i])
        ax[j, i].set_title(col, {'size': '14', 'weight': 'bold'}) 
        if i == 5:
            i = 0
            j += 1
        else:
            i += 1
plt.show()


# In[ ]:


plt.figure(figsize=(15,15))
sns.scatterplot(data=train, x="f_03", y="f_28", hue='target')
plt.show()


# Look at f_27: character vector
# 
# Based on:
# https://www.kaggle.com/code/ambrosm/tpsmay22-keras-quickstart/notebook and
# https://www.kaggle.com/code/cabaxiom/tps-may-22-eda-lgbm-model

# In[ ]:


train["unique_characters"] = train.f_27.apply(lambda s: len(set(s)))


# In[ ]:


tar_by_chars = train.groupby(["unique_characters", "target"]).size().reset_index()
sns.barplot(data=tar_by_chars, x="unique_characters", y = 0, hue="target")


# In[ ]:



train_chars = train.copy()
for i in range(10):
    train_chars[f'ch{i}'] = train_chars.f_27.str.get(i).apply(ord) - ord('A')
    
char_cols = [f"ch{i}" for i in range(10)]

plt.rcParams["figure.figsize"] = (25,20)
fig, ax = plt.subplots(2, 5)

i, j = (0, 0)
for col in char_cols:
    summed_col = train_chars.groupby([col, "target"]).size().reset_index().rename(columns={0:"freq"})
    sns.barplot(data=summed_col, x=col, y = "freq", hue="target", ax=ax[j, i])
    ax[j, i].set_title(col, {'size': '14', 'weight': 'bold'}) 
    if i == 4:
        i = 0
        j += 1
    else:
        i += 1
plt.show()


# Clearly some interesting interactions among positions of different characters. Particular in ch7.
