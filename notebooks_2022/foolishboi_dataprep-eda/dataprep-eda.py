#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


get_ipython().system('pip install dataprep')


# In[ ]:


from dataprep.eda import plot, plot_correlation, create_report, plot_missing


# In[ ]:


df = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')


# In[ ]:


plot(df)


# In[ ]:


create_report(df)


# In[ ]:


plot_correlation(df)


# In[ ]:




