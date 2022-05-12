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


# In[ ]:


pairs=pd.read_csv('/kaggle/input/foursquare-location-matching/pairs.csv')
train=pd.read_csv('/kaggle/input/foursquare-location-matching/train.csv')


# In[ ]:


pairs.head()


# In[ ]:


train.head()


# In[ ]:


train=train[train['country']=='TR']
train.info()


# # 💹 Plot

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
train['categories']=np.where(train['categories'].values == 'Residential Buildings (Apartments / Condos)', 'Residential Buildings', train['categories'].values)
data=pd.DataFrame((train.groupby(['categories']).count().id).sort_values(ascending=False)[:20])
top20Ctg=data.index
plt.figure(figsize=(20,8))
sns.barplot(data.id,data.index)


# In[ ]:


train['city']=np.where(train['city'].values == 'İstanbul', 'Istanbul', train['city'].values)
train['city']=np.where(train['city'].values == 'istanbul', 'Istanbul', train['city'].values)
train['city']=np.where(train['city'].values == 'İSTANBUL', 'Istanbul', train['city'].values)
data=pd.DataFrame((train.groupby(['city']).count().id).sort_values(ascending=False)[:20])
top20City=data.index
plt.figure(figsize=(20,8))
sns.barplot(data.id,data.index)


# In[ ]:


data=pd.DataFrame((train.groupby(['address']).count().id).sort_values(ascending=False)[:20])
plt.figure(figsize=(20,8))
sns.barplot(data.id,data.index)


# * **Ankara, İstanbul, Antalya, İzmir and Denizli is a city names. Also Türkiye means Turkey in Turkish.**
# * **The others are district names.**
# * **Bahçelievler, Kadıköy, Cumhuriyet Mahallesi, Beşiktaş, Maltepe, Ataşehir Üsküdar and Pendik are the most crowded districts of Istanbul.**
# * **Alsancak, Atatük Bulvarı, Bornova are the most crowded districts of Izmir.**
# 

# In[ ]:


data=pd.DataFrame((train.groupby(['name']).count().id).sort_values(ascending=False)[:20])
top20Name=data.index
plt.figure(figsize=(20,8))
sns.barplot(data.id,data.index)


# * **By far the leading location of place entries is __Bankamatik__ in Turkey, _Starbucks_ and _BurgerKing_ comes second and third.**
# * **Bankamatik means ATM in Turkish**
# * **In this graph Halkbank, TEB, Garanti BBVA, Türkiye İş Bankası, VakıfBank, Akbank, QNB finansbank and Ziraat Bankası are banks name in Turkey**
# * **Starbucks, Burger King, Mado, Özsüt, Simit Sarayı, MCDonald's, Kahve Dünyası are fastfood locations.** 

# In[ ]:


a=0
fig,axs= plt.subplots(4, 5,figsize=(30,10))
fig.suptitle('Sharing count per name for each Top20 City')
for ax in axs.flat:
    rect=sns.barplot(ax=ax,y=pd.DataFrame(train[train['city']==top20City[a]].groupby(['name']).count().id.sort_values(ascending=False)[:4]).index,x=pd.DataFrame(train[train['city']==top20City[a]].groupby(['name']).count().id.sort_values(ascending=False)[:4]).id)
    a=a+1


# In[ ]:


a=0
fig,axs= plt.subplots(4, 5,figsize=(20,10))
fig.suptitle('Sharing count per names, for each Top20 Names')
fig.tight_layout(pad=0.4, w_pad=5, h_pad=1.0)
for ax in axs.flat:
    rect=sns.barplot(ax=ax,y=pd.DataFrame(train[train['name']==top20Name[a]].groupby(['city']).count().id.sort_values(ascending=False)[:4]).index,x=pd.DataFrame(train[train['name']==top20Name[a]].groupby(['city']).count().id.sort_values(ascending=False)[:4]).id)
    a=a+1


# In[ ]:


a=0
fig,axs= plt.subplots(4, 5,figsize=(20,5))
fig.suptitle('Sharing count per names, for each Top20 Categories')
fig.tight_layout(pad=0.4, w_pad=15, h_pad=1.0)
for ax in axs.flat:
    rect=sns.barplot(ax=ax,y=pd.DataFrame(train[train['city']==top20City[a]].groupby(['categories']).count().id.sort_values(ascending=False)[:4]).index,x=pd.DataFrame(train[train['city']==top20City[a]].groupby(['categories']).count().id.sort_values(ascending=False)[:4]).id)
    a=a+1


# In[ ]:


a=0
fig,axs= plt.subplots(4, 5,figsize=(20,10))
fig.suptitle('Sharing count per names, for each Top20 Categories')
fig.tight_layout()
for ax in axs.flat:
    rect=sns.barplot(ax=ax,y=pd.DataFrame(train[train['categories']==top20Ctg[a]].groupby(['city']).count().id.sort_values(ascending=False)[:4]).index,x=pd.DataFrame(train[train['categories']==top20Ctg[a]].groupby(['city']).count().id.sort_values(ascending=False)[:4]).id)
    a=a+1


# # 🌍 Folium Map

# In[ ]:


import folium
from folium.plugins import MarkerCluster
from folium.plugins import MousePosition
from folium.features import DivIcon
map = folium.Map(location=[train['latitude'].values.mean(),train['longitude'].values.mean()], zoom_start=6, control_scale=True)


# In[ ]:


from folium import plugins
plugins.MarkerCluster(train[['latitude','longitude']]).add_to(map)
map


# In[ ]:


from folium.plugins import HeatMap
heatmap= folium.Map(location=[train['latitude'].values.mean(),train['longitude'].values.mean()], zoom_start=6, control_scale=True)
HeatMap(train[['latitude','longitude']], radius = 10, blur = 5).add_to(heatmap)
heatmap


# # Istanbul

# In[ ]:


map = folium.Map(location=[train[train['city']=='Istanbul']['latitude'].values.mean(),train[train['city']=='Istanbul']['longitude'].values.mean()], zoom_start=11, control_scale=True)
plugins.MarkerCluster(train[train['city']=='Istanbul'][['latitude','longitude']],train[train['city']=='Istanbul']['categories'].values.tolist()).add_to(map)
HeatMap(train[train['city']=='Istanbul'][['latitude','longitude']], radius = 10, blur = 5).add_to(map)
map


# # 🚧 Work in Progress.. 

# In[ ]:




