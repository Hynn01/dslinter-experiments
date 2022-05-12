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


# # Importando a Biblioteca

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import altair as alt


# # Leitura do Dataset

# In[ ]:


df = pd.read_csv('../input/perth-house-prices/all_perth_310121.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# # Install DATAPREP

# In[ ]:


get_ipython().system('pip install dataprep')


# In[ ]:


from dataprep.eda import create_report
report = create_report(df)


# In[ ]:


report


# In[ ]:




