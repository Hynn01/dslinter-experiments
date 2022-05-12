#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv('/kaggle/input/100-years-oil-production/crudeOil.csv')


# In[ ]:


sns.lineplot(x="Month", y="Oil_tbpd", data=df)

