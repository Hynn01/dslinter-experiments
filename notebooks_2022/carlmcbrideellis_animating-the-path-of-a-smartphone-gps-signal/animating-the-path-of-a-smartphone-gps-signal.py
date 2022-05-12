#!/usr/bin/env python
# coding: utf-8

# ### Animating the path of a smartphone GPS signal
# For the GPS positions we shall be using data provided by the [Google Smartphone Decimeter Challenge 2022](https://www.kaggle.com/competitions/smartphone-decimeter-2022)

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import rc
from IPython.display import HTML
import matplotlib.animation as animation

import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
#plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 14})
#plt.rcParams["figure.figsize"] = (8, 8)
import seaborn as sns
import plotly.express as px

import datetime
from datetime import timedelta, date

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt


# read in the data

# In[ ]:


GPS_data = pd.read_csv("../input/smartphone-decimeter-2022/train/2020-12-10-US-SJC-2/GooglePixel4/ground_truth.csv")
# convert Unix epoch time ("UnixTimeMillis") to a datetime
GPS_data["time"] = pd.to_datetime(GPS_data["UnixTimeMillis"],unit='ms')


# get a map of the region of interest using [Cartopy](https://scitools.org.uk/cartopy/docs/latest/index.html)

# In[ ]:


Lat_min = round(GPS_data["LatitudeDegrees"].min(),3) - 0.001
Lat_max = round(GPS_data["LatitudeDegrees"].max(),3) + 0.001
Lon_min = round(GPS_data["LongitudeDegrees"].min(),3) - 0.001
Lon_max = round(GPS_data["LongitudeDegrees"].max(),3) + 0.001

request = cimgt.OSM()
extent  = [Lon_min, Lon_max, Lat_min, Lat_max]


# now create our animation

# In[ ]:


get_ipython().run_cell_magic('capture', '', "n_steps = 75\n\nfig = plt.figure(figsize=(20,5))\ndef animate(time_stepper):\n    i = time_stepper * 8 # plot every 8th row, i.e. every 8 seconds\n    ax = fig.add_subplot(1,1,1, projection=request.crs)\n    ax.set_extent(extent)\n    ax.add_image(request,16)\n    tmp_Lat = GPS_data.iloc[i]['LatitudeDegrees']\n    tmp_Lon = GPS_data.iloc[i]['LongitudeDegrees']\n    ax.scatter(tmp_Lon,tmp_Lat, color='red', s=100, transform=ccrs.PlateCarree())\n\nanim = animation.FuncAnimation(fig, animate, frames=n_steps, save_count=n_steps, repeat=False)\nHTML(anim.to_jshtml())")


# Now press play to watch the animation

# In[ ]:


HTML(anim.to_jshtml())

