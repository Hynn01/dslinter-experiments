#!/usr/bin/env python
# coding: utf-8

# # Location Matching / Starter Model Using XGBoost 🗺️
# 
# ## Just Getting Started, Please come back soon !

# Hello! Kaggle. The purpose of this Notebook is to provide a easy to understand model to start in the competition and a good framework to improve in subsequent iterations
# 
# **Objectives:**
# 
# Develop a Machine Learning model, using XGBoost to underestand model development in the dataset; Below list of models available in the Notebook.
# 1. Gradient Boosted Trees (XGBoost)
# 
# **Strategy for this Notebook:**
# 1. Understand the Datasets, Simple EDA. 
# 2. Build a baseline model to improve construct on top.
# 3. Implement the model architecture described in the objectives section (XGBoost).
# 
# **Modeling the Problem**
# 
# The purpose of this competition is...
# 
# 
# ---
# 
# **Updates**
# 
# **04/29/2022**
# 
# 1. Started Notebook, loading and exploring the data.
# 
# **04/30/2022**
# 1. Exploring more of the data.
# 2. Thinking in options how to model the problem.
# ---
# 
# **Credits**
# 
# Below a list of notebooks that I have used as a reference.
# 
# Just trying to understand what has been done here...
# https://www.kaggle.com/code/ryotayoshinobu/foursquare-lightgbm-baseline/notebook

# # 1. Loading Libraries...

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


# ---

# # 2. Setting the Notebook Configuration

# In[ ]:


get_ipython().run_cell_magic('time', '', "# I like to disable my Notebook Warnings...\nimport warnings\nwarnings.filterwarnings('ignore')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Notebook Configuration...\n\n# Amount of data we want to load into the Model...\nDATA_ROWS = None # No limit, dataset is small\n\n# Dataframe, the amount of rows and cols to visualize...\nNROWS = 100\nNCOLS = 15\n\n# Main data location path...\nBASE_PATH = '/kaggle/input/foursquare-location-matching/' ")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Configure notebook display settings to only use 2 decimal places, tables look nicer...\npd.options.display.float_format = '{:,.4f}'.format\npd.set_option('display.max_columns', NCOLS) \npd.set_option('display.max_rows', NROWS)")


# ---

# # 3. Loading the Datasets

# In[ ]:


get_ipython().run_cell_magic('time', '', "# Load the CSV information into a Pandas DataFrame...\ntrn_data   = pd.read_csv(BASE_PATH + 'train.csv')\npairs_data = pd.read_csv(BASE_PATH + 'pairs.csv')\ntst_data   = pd.read_csv(BASE_PATH + 'test.csv')\nsumbission = pd.read_csv(BASE_PATH + 'sample_submission.csv')")


# ---

# # 4. Exploring the Information Loaded, Quick EDA

# ## 4.1 Train Dataset...

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Explore the shape of the DataFrame...\ntrn_data.shape')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Display simple information of the variables in the dataset...\ntrn_data.info()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Display the first few rows of the DataFrame...\ntrn_data.head()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Generate a simple statistical summary of the DataFrame, Only Numerical...\ntrn_data.describe()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Calculates the total number of missing values...\ntrn_data.isnull().sum().sum()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Display the number of missing values by variable...\ntrn_data.isnull().sum()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Display the number of unique values for each variable...\ntrn_data.nunique()')


# ---

# ## 4.2 Pairs Dataset

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Explore the shape of the DataFrame...\npairs_data.shape')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Display simple information of the variables in the dataset...\npairs_data.info()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Display the first few rows of the DataFrame...\npairs_data.head()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Display the first few rows of the DataFrame in a Transpose configuration (I can see all the columns)...\npairs_data.head().T')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Generate a simple statistical summary of the DataFrame, Only Numerical...\npairs_data.describe()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Calculates and display the total number of missing values...\npairs_data.isnull().sum().sum()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Display the number of unique values for each variable...\npairs_data.isnull().sum()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Display the number of unique values for each variable...\npairs_data.nunique()')


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Matching 1 variable from the train dataset with the pairs datasets...\n# Trying to see how the data is structured...\npairs_data[pairs_data['id_1'] == 'E_000001272c6c5d'].T")


# ---

# ## 4.3 Test Dataset

# In[ ]:


get_ipython().run_cell_magic('time', '', 'tst_data.head()')


# ---

# ## 4.4 Submission Dataset

# In[ ]:


sumbission.head()


# In[ ]:




