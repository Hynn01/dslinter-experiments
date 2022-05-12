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

import os, sys, glob

import json

# Any results you write to the current directory are saved as output.


# ### Listing of tick data files

# In[ ]:


tickdata = dict()
for root, dirs, filenames in os.walk('/kaggle/input'):
    for dirname in dirs:
        if dirname.startswith('histdata-forex-'):
            files_csv_zg = glob.glob(os.path.join(root, dirname,'*.csv.zg'))
            info=dict()
            missing = dict()
            with open(os.path.join(root, dirname, 'info.json')) as f:
              info = json.load(f)
            with open(os.path.join(root, dirname, 'missing.json')) as f:
              missing = json.load(f)
            provider, sectype, ticker = dirname.split('-',3)
            tickdata[str(dirname)] = {'provider':provider, 'ticker':ticker, 'sectype':sectype, 'details':info, 'count_of_missing_days':len(missing['days']), 'missing_days': missing['days'], 'files':files_csv_zg}


# ### Creating tensorflow CsvDataset for time series data

# In[ ]:


import tensorflow as tf


# ### Load the gzipped csv, extension set to .csv.zg to disable decompression in kaggle¶
# 

# In[ ]:


datasets = dict()
for key, values in tickdata.items():
    datasets[key]=tf.data.experimental.CsvDataset(values['files'], [tf.string,tf.float32,tf.float32],header=False, compression_type="GZIP",select_cols=[0,1,2])


# In[ ]:


input_ds = next(iter(datasets.values()))
input_ds.element_spec


# In[ ]:


for f in input_ds.take(5):
    print(f)


# ### Conversion function
# 
# ```(datetime: string, bid: float32, ask: float32) -> (dateime: string, timestamp: float64,  bid: float32, ask: float32, mid: float32, spread: float32)```

# In[ ]:


from datetime import datetime, timedelta


# In[ ]:


def conv_func(dt, bid, ask):
    txt = lambda t : t.numpy().decode('ascii')
    conv = lambda z : pd.Timestamp(datetime.strptime(z.numpy().decode('ascii'), '%Y%m%d %H%M%S%f')).to_datetime64()
    return tf.py_function(txt,[dt], tf.string), tf.py_function(conv, [dt], tf.float64), bid, ask,(bid+ask)/2, ask-bid


# In[ ]:


ts_data = dict()
for key, ds in datasets.items():
    ts_data[key] = ds.map(conv_func)
    


# In[ ]:


test_ds = next(iter(ts_data.values()))
test_ds.element_spec


# In[ ]:


for f in test_ds.take(5):
    print(f)


# #  Click link >> [OpenDrive, Cloud storage with Webdav support](https://www.opendrive.com/?od=5eecb28b9dda9)

# # Connect Opendrive

# ### username and password stored in kaggle secrets

# In[ ]:


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
opendrive_usr = user_secrets.get_secret("OPENDRIVE_USERNAME")
opendrive_passwd = user_secrets.get_secret("OPENDRIVE_PASSWORD")


# ## Access via WebDAV

# In[ ]:


get_ipython().system('pip install webdavclient3')


# In[ ]:


import urllib3
from webdav3.client import Client
options = {
 'webdav_hostname': "https://webdav.opendrive.com",
 'webdav_login':    opendrive_usr,
 'webdav_password': opendrive_passwd
}
client = Client(options)
client.verify = False
urllib3.disable_warnings()


# ## Make a cache directory

# In[ ]:


client.list()


# # Upload dataset cache

# In[ ]:


if not client.check('ds_cache'):
    client.mkdir('ds_cache')


# In[ ]:


os.makedirs('/kaggle/working/cache', exist_ok=True)
for key, value in ts_data.items():
    cachefilepath = os.path.join('cache',"{}.cache".format(key))
    it = value.cache(cachefilepath).prefetch(tf.data.experimental.AUTOTUNE)
    # it = iter(value)
    count = 0
    for i in it:
        count+=1
    print("{} has {} quotes".format(key, count))


# In[ ]:


# Uncomment following to upload cache to opendrive
# client.push('ds_cache','/kaggle/working/cache')

