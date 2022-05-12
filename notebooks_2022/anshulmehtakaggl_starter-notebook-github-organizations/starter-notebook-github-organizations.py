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


# <img src='data:image/png;base64,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'>

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import json
import os,re


# In[ ]:


import networkx as nx


# In[ ]:


data=pd.read_csv('../input/github-organizations-social-network-analysis/organization.csv')


# ### Drawing the Graph 🕸

# In[ ]:


from networkx.algorithms import bipartite
B=nx.Graph()
B.add_nodes_from(data['Organisation'],bipartite=0)
B.add_nodes_from(data['member'],bipartite=1)
for i in range(len(data)):
  B.add_edges_from([(data.iloc[i,1],data.iloc[i,0])])
# nx.draw(B,with_labels=1)
plt.figure(figsize=(25,25))
nx.draw(B, with_labels=True, node_size=1)
plt.show()


# ### Closeness Centrality 🤗

# In[ ]:


cc=nx.closeness_centrality(B, u=None, distance=None, wf_improved=True)
cc


# ### Finding Cliques in the Graph

# In[ ]:


cliques=nx.find_cliques(B)
for cl in cliques:
  print(cl)


# ### Drawing a Max Bipartite Clique

# In[ ]:


max_clique_biparti=nx.make_clique_bipartite(B, fpos=None, create_using=None, name=None)
options = {"edgecolors": "black", "node_size": 100, "alpha": 0.9}
nx.draw(max_clique_biparti,node_color="yellow",**options)


# ### Local Bridges

# In[ ]:


lb=nx.local_bridges(B)
sum=0
for bridge in lb:
    sum=sum+1
print("Number of Local Bridges are:",sum)

