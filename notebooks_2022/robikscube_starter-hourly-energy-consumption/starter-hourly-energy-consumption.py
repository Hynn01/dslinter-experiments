#!/usr/bin/env python
# coding: utf-8

# ## Hourly Energy Consumption

# In[ ]:


import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
plt.style.use('ggplot') # Make it pretty


# In[ ]:


# Data is saved in parquet format so schema is preserved.
df = pd.read_parquet('../input/est_hourly.paruqet')


# Data index is the date/hour, columns are for different regions within PJM.
# 
# Regions joined at different times, so not all have data for all dates. Regions also split (PJM_Load split to East and West)

# In[ ]:


#Show PJM Regions
from IPython.display import Image
Image(url= "http://slideplayer.com/4238181/14/images/4/PJM+Evolution.jpg")


# In[ ]:


df.head()


# In[ ]:


df.describe().T


# In[ ]:


_ = df['PJME'].plot.hist(figsize=(15, 5), bins=200, title='Distribution of PJME Load')


# In[ ]:


_ = df['DOM'].plot.hist(figsize=(15, 5), bins=200, title='Distribution of DOMINION Load')


# In[ ]:


_ = df.plot.hist(figsize=(15, 5), bins=200, title='Distribution of Load by Region')


# ## Plot Time Series

# In[ ]:


plot = df.plot(style='.', figsize=(15, 8), title='Entire PJM Load 1998-2001')


# # Plotting Regions

# In[ ]:


_ = df[['PJM_Load','PJME','PJMW']]     .plot(style='.', figsize=(15, 5), title='PJM Load 1998-2002 - Split East and West 2002-2018')


# ## Summer Demand vs Winter Demand
# Note the dips mid-day in the winter months. Conversely in summer months the daily load is more bell shaped. This is due to high mid-day energy consumtion by air conditioning. In winter months people tend to use less energy mid-day.

# In[ ]:


_ = df['PJME'].loc[(df['PJME'].index >= '2017-11-01') &
               (df['PJME'].index < '2017-12-01')] \
    .plot(figsize=(15, 5), title = 'November 2017')


# In[ ]:


_ = df['PJME'].loc[(df['PJME'].index >= '2017-06-01') &
               (df['PJME'].index < '2017-07-01')] \
    .plot(figsize=(15, 5), title = 'June 2017')


#  # Create Time Series Features

# In[ ]:


df['dow'] = df.index.dayofweek
df['doy'] = df.index.dayofyear
df['year'] = df.index.year
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['hour'] = df.index.hour
df['weekday'] = df.index.weekday_name
df['woy'] = df.index.weekofyear
df['dom'] = df.index.day # Day of Month
df['date'] = df.index.date 


# In[ ]:


_ = df[['PJM_Load','hour']].plot(x='hour',
                                     y='PJM_Load',
                                     kind='scatter',
                                     figsize=(14,4),
                                     title='Consumption by Hour of Day')


# Note Saturday and Sunday demand is much less than during a work week. This is also true for holidays.

# In[ ]:


_ = df.pivot_table(index=df['hour'], 
                     columns='weekday', 
                     values='PJME',
                     aggfunc='sum').plot(figsize=(15,4),
                     title='PJM East - Daily Trends')


# ## Trends change depending on time of year

# In[ ]:


fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(df.loc[df['quarter']==1].hour, df.loc[df['quarter']==1].PJME)
ax.set_title('Hourly Boxplot PJME Q1')
ax.set_ylim(0,65000)
fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(df.loc[df['quarter']==2].hour, df.loc[df['quarter']==2].PJME)
ax.set_title('Hourly Boxplot PJME Q2')
ax.set_ylim(0,65000)
fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(df.loc[df['quarter']==3].hour, df.loc[df['quarter']==3].PJME)
ax.set_title('Hourly Boxplot PJME Q3')
ax.set_ylim(0,65000)
fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(df.loc[df['quarter']==4].hour, df.loc[df['quarter']==4].PJME)
ax.set_title('Hourly Boxplot PJME Q4')
_ = ax.set_ylim(0,65000)


# ## Hope you enjoy exploring this dataset!

# In[ ]:




