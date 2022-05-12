#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('../input/students-adaptability-level-in-online-education/students_adaptability_level_online_education.csv')
df.head()


# In[ ]:


df.describe().T


# No missing value.

# In[ ]:


df.info()


# In[ ]:


# All are objects so to reduce the memory usage we will convert all of them to category
for col in df.columns:
    df[col] = df[col].astype('category')
df.info()


# In[ ]:


plt.figure(figsize=(20,15))
plt.hist(data=df, x='Age')


# In[ ]:


df.columns


# In[ ]:


group = df.groupby(['Gender', 
            'Age', 
#             'Education Level', 
#             'Institution Type', 'IT Student',
#        'Location', 'Load-shedding', 'Financial Condition', 'Internet Type',
#        'Network Type', 'Class Duration', 
#             'Self Lms', 
#             'Device',
            'Adaptivity Level'])['Adaptivity Level'].count().to_frame()

group


# In[ ]:


group.columns = ['Adaptivity_count']
ax = group.sort_values(by='Adaptivity_count', ascending=False).plot(kind='barh', figsize=(20,15), title='Adaptivity count')
ax.set_xlabel('Adaptivity count')
#group


# So Boy with 21-25 age range and moderate adaptivity is most common in the group we were aiming at. One can group differently and observe further relationships. Happy coding!

# In[ ]:




