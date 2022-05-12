#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import networkx as nx
import time
import seaborn as sns
import re
import math
import PIL
import urllib
sns.set_style("whitegrid")
#sns.set(style="darkgrid")
sns.set_palette("tab10")
import IPython
from IPython.display import display
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
import datetime as dt
plt.style.use('seaborn-notebook')
params = {'legend.fontsize': 15,
          'legend.title_fontsize': 16,
          'figure.figsize': (15, 5),
         'axes.labelsize': 18,
         'axes.titlesize':20,
         'xtick.labelsize':18,
         'ytick.labelsize':18}
plt.rcParams.update(params)
img_fmt = 'svg'


# In[ ]:


df = pd.read_csv('/kaggle/input/icc-test-cricket-runs/ICC Test Batting Figures.csv',encoding= 'unicode_escape')
df.head()
df.info()
df.describe().T


# In[ ]:


df.replace(to_replace='-',inplace=True)
df[["Mat", "Avg","Inn","Runs","100"]] = df[["Mat", "Avg","Inn","Runs","100"]].apply(pd.to_numeric)


# Top 20 players who has the best average sorted by their run aggregate

# In[ ]:


df[['Player', 'Country']] = df['Player'].str.split('(', 1, expand=True)
#remove 'ICC' from country
df.Country = df.Country.apply(lambda x: x.split(')')[0].replace('ICC/',''))


# In[ ]:


#filter the players who played 100+ matches and sort out the top n players with best average
df1 = df.query('Inn > 100').sort_values(by='Avg',ascending=False).head(10).sort_values(by='Runs',ascending=False)
_ = plt.subplots(figsize = (10,8))
_ = plt.xticks(rotation = 60)
_ = plt.title('Players with best batting average ordered by their run aggregate', color='blue', fontsize=20)
sns.barplot(data = df1, x='Player',y = 'Runs', palette='dark')


# In[ ]:


df[df['100']>20]


# In[ ]:


#filter the players who played 100+ matches and sort out the top n players with best average
df2 = df.sort_values(by='100',ascending=False).head(15)
_ = plt.subplots(figsize = (10,8))
_ = plt.xticks(rotation = 60)
_ = plt.title('Players with most average ordered by their 100s', color='blue', fontsize=20)
sns.barplot(data = df2, y='Player',x = '100', palette='dark')


# In[ ]:




