#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


import os

origin_path = "/kaggle/working"
repo_path = "/kaggle/working/american_options"


# In[ ]:


get_ipython().run_cell_magic('capture', '', '!rm -rf /kaggle/working/\n!git clone https://github.com/konmue/american_options.git')


# In[ ]:


os.chdir(repo_path)


# In[ ]:


get_ipython().run_cell_magic('capture', '', '!python setup.py install')


# In[ ]:


import pandas as pd

from deep_ao.algorithms.lsm.main import main


# In[ ]:


results = main()


# In[ ]:


results


# In[ ]:


print(results.to_latex(float_format="%.2f", index=False))

