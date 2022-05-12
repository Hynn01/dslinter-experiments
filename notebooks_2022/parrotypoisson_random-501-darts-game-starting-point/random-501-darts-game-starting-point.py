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


# Rather than always starting from 501, it helps to begin from a different point in the match to get used to all the different ways a match can transpire. This notebook will output a potential place in a 501 match that can occur.

# In[ ]:


possibilities = pd.read_csv('/kaggle/input/darts/data/501_optimal_target.csv')
possibilities


# In[ ]:


start = possibilities.sample()


# In[ ]:


print(f'You have a score of {start.score.iloc[0]}, and {start.darts_remaining.iloc[0]} darts remaining this round. You should aim at the {start.target_text.iloc[0]}.')

