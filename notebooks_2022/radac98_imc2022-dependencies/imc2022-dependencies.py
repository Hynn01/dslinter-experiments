#!/usr/bin/env python
# coding: utf-8

# # Tutorial: Installing dependencies for offline submissions
# 
# This notebook shows how to install dependencies to notebooks for offline submission (this may be obvious to Kaggle veterans).

# In[ ]:


# This example notebook contains pip packages, binaries, or pre-trained models to be used in offline notebooks.
# After you run this, do File -> Save version -> Advanced Settings -> Save output for this version -> Quick save -> Save.
# On the "inference" notebook, do File -> Add or upload data -> Notebook output files -> Your work, find this notebook, and "Add" it.
# It may take a few minutes for it to show up on the list.
# These files will then be available on /kaggle/input/ on the "inference" notebook, and show up under "data" on the right pane.

get_ipython().system('mkdir -p /kaggle/working/pip')

get_ipython().system('mkdir -p wheels')

get_ipython().system('pip wheel -w wheels einops')
get_ipython().system('git clone https://github.com/felipecadar/DKM')



get_ipython().system('mkdir -p /kaggle/working/pretrained')
get_ipython().system('wget https://github.com/Parskatt/storage/releases/download/dkm/dkm_base_v11.pth   -O /kaggle/working/pretrained/dkm.pth')
  
print('Done!')


# In[ ]:


# Now you can flick the "offline" switch on the right pane and test the installation.
get_ipython().system('pip install -f /kaggle/working/wheels --no-index einops')
get_ipython().system('cd /kaggle/working/DKM/; pip install -f /kaggle/working/wheels -e . ')

print('Done!')


# In[ ]:


# Check that everything went well.
import sys
sys.path.append('/kaggle/working/DKM/')
# !pip install -e /kaggle/working/DKM/

from dkm import dkm_base


# In[ ]:





# In[ ]:





# In[ ]:




