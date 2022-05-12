#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


iris = pd.read_csv("/kaggle/input/moby-dick-dataset/iris.csv")


# In[ ]:


print(iris.shape)


# In[ ]:


print(iris.columns)


# In[ ]:


iris['species'].value_counts()


# In[ ]:


iris.plot(kind='scatter', x='sepal_length', y='sepal_width')
plt.show()


# In[ ]:


sns.set_style('whitegrid')


# In[ ]:


sns.FacetGrid(iris, hue='species', size=4).map(plt.scatter, 'sepal_length', 'sepal_width').add_legend();


# In[ ]:


plt.close()
sns.set_style('whitegrid')
sns.pairplot(iris, hue='species', size=3)
plt.show()


# In[ ]:




