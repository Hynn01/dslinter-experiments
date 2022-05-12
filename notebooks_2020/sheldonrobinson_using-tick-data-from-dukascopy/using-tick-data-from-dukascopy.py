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

# Any results you write to the current directory are saved as output.


# ### Listing of tick data files

# In[ ]:


tickdata = dict()
for root, dirs, filenames in os.walk('/kaggle/input'):
    for dirname in dirs:
        files_csv_zg = glob.glob(os.path.join(root, dirname,'*.csv.zg'))
        tickdata[str(dirname)] = files_csv_zg


# ### Created tensorflow CsvDataset for time series data

# In[ ]:


import tensorflow as tf


# ### Load the gzipped csv, extension set to .csv.zg to disable decompression in kaggle

# In[ ]:


datasets = dict()
for ccy, files in tickdata.items():
    datasets[ccy]=tf.data.experimental.CsvDataset(files, [tf.string,tf.float32,tf.float32],header=True, compression_type="GZIP",select_cols=[0,1,2])


# In[ ]:


input_ds = next(iter(datasets.values()))
input_ds.element_spec


# In[ ]:


for f in input_ds.take(5):
    print(f)


# ### Conversion function
# 
# ```(datetime: string, bid: float32, ask: float32) -> (dateime: string, timestamp: float64,  bid: float32, ask: float32, mid: float32, spread:float32)```

# In[ ]:


from datetime import datetime, timedelta


# In[ ]:


def conv_func(dt, bid, ask):
    txt = lambda t : t.numpy().decode('ascii')
    conv = lambda z : pd.Timestamp(datetime.strptime(z.numpy().decode('ascii'), '%Y-%m-%d %H:%M:%S.%f')).to_datetime64()
    return tf.py_function(txt,[dt], tf.string), tf.py_function(conv, [dt], tf.float64), bid, ask,(bid+ask)/2, ask-bid


# In[ ]:


ts_data = dict()
for key, ds in datasets.items():
    ts_data[key] = ds.map(conv_func)
    ts_data[key].cache(key)


# In[ ]:


test_ds = next(iter(ts_data.values()))
test_ds.element_spec


# In[ ]:


for f in test_ds.take(5):
    print(f)

