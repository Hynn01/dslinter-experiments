#!/usr/bin/env python
# coding: utf-8

# # Video Game Stats
# ## Lab 12: Data Analysis with Pandas
# ## Matt Rangel
# ## 5/7/2022 
# ### Resubmit #1

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


# In[ ]:


df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')


# In[ ]:


df.info()


# ## 1. Which company is the most common video game publisher?

# In[ ]:


df.Publisher.mode()


# ## 2. What’s the most common platform?

# In[ ]:


df.Platform.mode()


# ## 3. What about the most common genre?

# In[ ]:


df.Genre.mode()


# ## 4. What are the top 20 highest grossing games?

# In[ ]:


global_sales = df[["Name","Global_Sales"]].sort_values(by="Global_Sales", ascending=False)
global_sales[:20]


# ## 5. For North American video game sales, what’s the median?

# In[ ]:


med_sales = df["NA_Sales"].median()
med_sales


# ## 5. a) Provide a secondary output showing ten games surrounding the median sales output.

# In[ ]:


ten_med_head = df[df['NA_Sales'] == med_sales]

ten_med_head.head(10)


# In[ ]:


ten_med_head.tail(10)


# ## 6. For the top-selling game of all time, how many standard deviations above/below the mean are its sales for North America?

# In[ ]:


na_std = df['NA_Sales'].std()
na_std


# In[ ]:


top_sale = df['NA_Sales'].head(1)
top_sale


# In[ ]:


na_mean = df['NA_Sales'].mean()
na_mean


# In[ ]:


na_sales_minus_mean = top_sale - na_mean
top_sales_value_from_mean = na_sales_minus_mean / na_std
top_sales_value_from_mean


# ## 7. The Nintendo Wii seems to have outdone itself with games. How does its average number of sales compare with all of the other platforms?

# In[ ]:



wii_global_sales_mean =  df[df['Platform'] == 'Wii']['Global_Sales'].mean()


# In[ ]:


not_wii_global_sales_mean = df[df['Platform'] != 'Wii']['Global_Sales'].mean()


# In[ ]:


print('Wii sales,',wii_global_sales_mean,'. Other than Wii sales,', not_wii_global_sales_mean)


# ## 8. Come up with 3 more questions that can be answered with this data set.

# ### Top 3 Genre

# In[ ]:


df.Genre.head(3)


# ### Bottom 3 Genre

# In[ ]:


df.Genre.tail(3)


# ### Top Sega Publisher

# In[ ]:


df[df['Publisher'] == 'Sega'].head()

