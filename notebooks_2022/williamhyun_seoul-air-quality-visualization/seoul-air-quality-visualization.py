#!/usr/bin/env python
# coding: utf-8

# # Import required packages

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from folium import Map, Marker, Circle


# # Load Dataset

# In[ ]:


df = pd.read_csv("../input/seoulairqualityhistoricdata/seoul_air_1988_2021.csv")
df


# # Select a single location, `101`, and check the missing period

# In[ ]:


df[df['loc'] == 101][['o3']].plot()


# # Select 9AM of Christmas day in 2021

# In[ ]:


christmas_day = df[df['dt'] == 2021122409]
christmas_day


# # Show the locations

# In[ ]:


seoul = [37.5600, 126.9900]
m = Map(location=seoul, tiles='openstreetmap', zoom_start=11)

for idx, row in christmas_day.iterrows():
    Marker([row['lat'], row['long']]).add_to(m)
m


# # Show O3 quality relatively

# In[ ]:


seoul = [37.5600, 126.9900]
m = Map(location=seoul, tiles='openstreetmap', zoom_start=11)

for idx, row in christmas_day.iterrows():
    Circle([row['lat'], row['long']], radius=row['o3']*200000, fill=True, opacity=1, color='red').add_to(m)
m

