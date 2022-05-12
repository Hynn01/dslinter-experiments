#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime as dt
import geopandas as gpd
import matplotlib.ticker as ticker

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/corona-virus-brazil/brazil_covid19.csv')
data.head()


# In[ ]:


data.isna().sum()


# ## By day

# In[ ]:


cases_day = pd.DataFrame(data.groupby(['date']).sum().reset_index())
cases_day['date'] = cases_day['date'].apply(lambda x: '{:%m-%d}'.format(dt.strptime(x, '%Y-%m-%d')))
cases_day


# In[ ]:


plt.figure(figsize=(12,8))
plt.title("Cases of Covid-19 in Brazil")
plt.xticks(rotation=45)


sns.set_style("whitegrid")
sns.despine(left=True)
fig = sns.lineplot(x=cases_day['date'], y=cases_day['cases'], color='orange')
fig.set_xlabel('Date')
fig.set_ylabel('Cases')
fig.xaxis.set_major_locator(ticker.MaxNLocator(46))


# In[ ]:


plt.figure(figsize=(12,8))
plt.title("Deaths caused by Covid-19 in Brazil")
plt.xticks(rotation=45)

sns.set_style("whitegrid")
sns.despine(left=True)
fig = sns.lineplot(x=cases_day['date'], y=cases_day['deaths'], color='red')
fig.set_xlabel('Date')
fig.set_ylabel('Deaths')
fig.xaxis.set_major_locator(ticker.MaxNLocator(46))


# In[ ]:


plt.figure(figsize=(12,8))
plt.title("Progress of the Covid-19 in Brazil")
plt.xticks(rotation=45)


sns.set_style("whitegrid")
sns.despine(left=True)
fig = sns.lineplot(x=cases_day['date'], y=cases_day['cases'], label="Cases", color='orange')
fig = sns.lineplot(x=cases_day['date'], y=cases_day['deaths'], label="Deaths", color='red')
fig.set_xlabel('Date')
fig.set_ylabel('')
fig.xaxis.set_major_locator(ticker.MaxNLocator(60))


# In[ ]:


cases_day['new_cases'] = cases_day['cases'].diff()
cases_day['new_deaths'] = cases_day['deaths'].diff()


# In[ ]:


# Set the width and height of the figure
plt.figure(figsize=(20,6))

plt.title("New cases of Covid-19 in Brazil")

sns.barplot(x=cases_day['date'], y=cases_day['new_cases'], color='orange')

plt.xticks(rotation='vertical')
# Add label for vertical axis
plt.xlabel("")
plt.ylabel("New cases")


# In[ ]:


# Set the width and height of the figure
plt.figure(figsize=(20,6))

plt.title("New deaths caused by Covid-19 in Brazil")

sns.barplot(x=cases_day['date'], y=cases_day['new_deaths'], color='red')

plt.xticks(rotation='vertical')
# Add label for vertical axis
plt.xlabel("")
plt.ylabel("New cases")


# ## By region

# In[ ]:


cases_region = pd.DataFrame(data.groupby(['state']).sum().reset_index())
cases_region.head()


# In[ ]:


fp = '../input/brazilianstatesshapefiles/BRA_adm1.shp'
map_df = gpd.read_file(fp)
# check data type so we can see that this is not a normal dataframe, but a GEOdataframe
map_df.drop(['ISO','NAME_0','ID_0','ID_1','HASC_1','CCN_1','CCA_1','TYPE_1','ENGTYPE_1','NL_NAME_1','VARNAME_1'], 
            axis = 1, inplace=True)
map_df.head()


# In[ ]:


merged = map_df.set_index('NAME_1').join(cases_region.set_index('state'))
merged['coords'] = merged['geometry'].apply(lambda x: x.representative_point().coords[:])
merged['coords'] = [coords[0] for coords in merged['coords']]
merged.head()


# In[ ]:


merged.describe()


# In[ ]:


values = 'cases'
vmin, vmax = 26, 36000
fig, ax = plt.subplots(1, figsize=(25, 10))
ax.axis('off')
title = 'Cases of Covid-19 in Brazil'
ax.set_title(title, fontdict={'fontsize': '18', 'fontweight' : '1'})
sm = plt.cm.ScalarMappable(cmap='OrRd', norm=plt.Normalize(vmin=vmin, vmax=vmax))
cbar = fig.colorbar(sm, shrink=0.5)
for idx, row in merged.iterrows():
    plt.annotate(s=row.cases, xy=row['coords'],horizontalalignment='center')
    
merged.plot(column=values, cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='1',
            norm=plt.Normalize(vmin=vmin, vmax=vmax))


# In[ ]:


values = 'deaths'
vmin, vmax = 0, 1500
fig, ax = plt.subplots(1, figsize=(25, 10))
ax.axis('off')
title = 'Deaths caused by Covid-19 in Brazil'
ax.set_title(title, fontdict={'fontsize': '18', 'fontweight' : '1'})
sm = plt.cm.ScalarMappable(cmap='OrRd', norm=plt.Normalize(vmin=vmin, vmax=vmax))
cbar = fig.colorbar(sm, shrink=0.5)
for idx, row in merged.iterrows():
    plt.annotate(s=row.deaths, xy=row['coords'],horizontalalignment='center')
    
merged.plot(column=values, cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='1',
            norm=plt.Normalize(vmin=vmin, vmax=vmax))


# In[ ]:


cases_region.sort_values(['cases'], inplace=True, ascending=False)

plt.figure(figsize=(20,6))
plt.title("Number of cases by state")

pal = sns.color_palette("OrRd_r", len(cases_region))
rank = cases_region['cases'].argsort().argsort()  
sns.barplot(x=cases_region.state, y=cases_region.cases, palette=np.array(pal[::-1])[rank])

plt.xticks(rotation='vertical')
plt.xlabel("")
plt.ylabel("Cases")


# In[ ]:


cases_region.sort_values(['deaths'], inplace=True, ascending=False)

plt.figure(figsize=(20,6))
plt.title("Number of deaths by state")

pal = sns.color_palette("OrRd_r", len(cases_region))
rank = cases_region['deaths'].argsort().argsort()  
sns.barplot(x=cases_region.state, y=cases_region.deaths, palette=np.array(pal[::-1])[rank])

plt.xticks(rotation='vertical')
plt.xlabel("")
plt.ylabel("Cases")


# Progression of the disease by state

# In[ ]:


cases_region_2 = pd.DataFrame(data.groupby(['date','state']).sum().reset_index())
cases_region_2.head()


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 8))
def draw_barchart(date,case="cases"):
    dff= (cases_region_2[cases_region_2['date'].eq(date)].sort_values(by=case,ascending=False).head(10))[::-1]
    ax.clear()
    ax.barh(dff['state'], dff[case], color=["#980505","#CD1212","#D84E4E","#CB6262","#D39B5F","#F7EC10","#D0F710","#9CF710","#B4D67F","#969C8E"][::-1])
    dx = dff[case].max() / 200
    for i, (value, name) in enumerate(zip(dff[case], dff['state'])):
        ax.text(value-dx, i,     name,           size=14, weight=600, ha='right', va='bottom')
        ax.text(value-dx, i-.25, value, size=10, color='#17202A', ha='right', va='baseline')
        ax.text(value+dx, i,     f'{value:,.0f}',  size=14, ha='left',  va='center')
    # ... polished styles
    ax.text(1, 0.4, date, transform=ax.transAxes, color='#777777', size=30, ha='right', weight=800)
    ax.text(0, 1.06, 'Number of cases', transform=ax.transAxes, size=12, color='#777777')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', colors='#777777', labelsize=12)
    ax.set_yticks([])
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    ax.text(0, 1.12, 'Progression of the Covid-19 in Brazil up to April 30th, 2020',
            transform=ax.transAxes, size=24, weight=600, ha='left')
    ax.text(1, 0, 'by @Rodolfo Saldanha; credit to @Zubair @jburnmurdoch', transform=ax.transAxes, ha='right',
            color='#777777', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
    plt.box(False)


# In[ ]:


from matplotlib import animation as F
from IPython.display import HTML
fig,ax = plt.subplots(figsize=(16,10)) #Set figure for plot
animator = F.FuncAnimation(fig,draw_barchart,frames=cases_region_2.date.unique(),interval=500) #Building animation
HTML(animator.to_jshtml())


# In[ ]:


writergif = F.PillowWriter(fps=3) 
animator.save('animation.gif',writer=writergif)

