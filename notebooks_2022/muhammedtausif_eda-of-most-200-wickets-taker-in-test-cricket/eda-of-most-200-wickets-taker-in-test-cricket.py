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


# # Most Wickets in Test Cricket (2022 April)|

# In[ ]:


import seaborn as sna
import matplotlib.pyplot as plt
import sklearn


# In[ ]:


path_to_csv = '/kaggle/input/most-wicket-in-cricket-200/most-wickets-200.csv'
wkt_record = pd.read_csv(path_to_csv)


# In[ ]:


wkt_record.shape


# In[ ]:


wkt_record.columns


# In[ ]:


wkt_record.head(3)


# In[ ]:


wkt_record.describe()


# In[ ]:




