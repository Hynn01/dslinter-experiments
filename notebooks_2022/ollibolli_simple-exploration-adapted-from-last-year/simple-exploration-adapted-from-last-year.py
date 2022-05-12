#!/usr/bin/env python
# coding: utf-8

# I shamelessly took almost everything and adapted from last years wonderful notebook: https://www.kaggle.com/code/nayuts/let-s-visualize-dataset-to-understand still a work in progress

# In[ ]:



get_ipython().system('pip install pynmea2')

import glob
import itertools
import json
import os
import warnings
warnings.filterwarnings('ignore')

import geopandas as gpd
from geopandas import GeoDataFrame
import geoplot as gplt
from IPython.display import Video
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import plotly.express as px
import pynmea2
import requests
import seaborn
from shapely.geometry import Point, shape
import shapely.wkt

get_ipython().run_line_magic('matplotlib', 'inline')

DATA_PATH = "../input/smartphone-decimeter-2022/"


# In[ ]:


sub = pd.read_csv(DATA_PATH + "sample_submission.csv")
sub.head()


# Devices can be one thing or multiple things. Data collection trials are separated as collectionName like this

# In[ ]:


get_ipython().system('ls ../input/smartphone-decimeter-2022/train')


# Under each collection Name, the data of the device is stored.

# In[ ]:


get_ipython().system('ls ../input/smartphone-decimeter-2022/train/2020-05-15-US-MTV-1')


# In addition, the data collected from each device, groundtruth, and supplemental data are stored under it.

# In[ ]:


get_ipython().system('ls ../input/smartphone-decimeter-2022/train/2020-05-15-US-MTV-1/GooglePixel4XL')


# The supplemental data contains the raw data that was measured, and I'll show way to read nmea.

# In[ ]:


get_ipython().system('ls ../input/smartphone-decimeter-2022/train/2020-05-15-US-MTV-1/GooglePixel4XL/supplemental')


# In[ ]:


# ../input/smartphone-decimeter-2022/train/2020-05-15-US-MTV-1/GooglePixel4XL/device_gnss.csv


# In[ ]:


df_sample_trail_gnss = pd.read_csv(DATA_PATH + "train/2020-05-15-US-MTV-1/GooglePixel4XL/device_gnss.csv")
df_sample_trail_gt = pd.read_csv(DATA_PATH + "train/2020-05-15-US-MTV-1/GooglePixel4XL//ground_truth.csv")
df_sample_trail_imu = pd.read_csv(DATA_PATH + "train/2020-05-15-US-MTV-1/GooglePixel4XL//device_imu.csv")

df_sample_trail_gnss.head()


# In[ ]:


df_sample_trail_gnss.columns


# In[ ]:


df_sample_trail_gt.head()


# In[ ]:


df_sample_trail_imu.head()


# # How to check track in detail?
# 
# We can use plotly to see our model or ground truth like this. To see trafic, you should adjust map centor and scale.

# In[ ]:


def visualize_trafic(df, center, zoom=9):
    fig = px.scatter_mapbox(df,
                            
                            # Here, plotly gets, (x,y) coordinates
                            lat="LatitudeDegrees",
                            lon="LongitudeDegrees",
                            
                            #Here, plotly detects color of series
                            color="phone",
                            labels="phone",
                            
                            zoom=zoom,
                            center=center,
                            height=600,
                            width=800)
    fig.update_layout(mapbox_style='stamen-terrain')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="GPS trafic")
    fig.show()


# In[ ]:


df_sample_trail_gt['phone'] = 'GooglePixel4XL'
center = {"lat":37.423576, "lon":-122.094132}
visualize_trafic(df_sample_trail_gt, center)


# In[ ]:


df_sample_trail_gt2 = pd.read_csv(DATA_PATH + "train/2020-05-21-US-MTV-1/GooglePixel4/ground_truth.csv")
df_sample_trail_gt2['phone'] = 'GooglePixel4'


# Since plotly looks at the phoneName of the dataframe,
# you can visualize multiple series of data by simply concatting dataframes.
df_sample_trail_gt3 = pd.concat([df_sample_trail_gt, df_sample_trail_gt2])

center = {"lat":37.423576, "lon":-122.094132}
visualize_trafic(df_sample_trail_gt3, center)


# # How to check large amounts of tracks?
# 
# Earlier we saw how to use plotly to map data on OpenStreetMap. This time, since there is a certain amount of tracking data in the train data alone, I will also show you how to use geopandas to get a quick overview as a regular diagram.
# 
# From here on, the cells will be hidden for a while, because the procedure is necessary for visualization and we will get tired of following everything. If you have interest, please open and check them accordingly.
# 
# First, I'll download shape file lof bayarea.
# 

# In[ ]:


#Download geojson file of US San Francisco Bay Area.
r = requests.get("https://data.sfgov.org/api/views/wamw-vt4s/rows.json?accessType=DOWNLOAD")
r.raise_for_status()

#get geojson from response
data = r.json()

#get polygons that represents San Francisco Bay Area.
shapes = []
for d in data["data"]:
    shapes.append(shapely.wkt.loads(d[8]))
    
#Convert list of porygons to geopandas dataframe.
gdf_bayarea = pd.DataFrame()

#I'll use only 6 and 7th object.
for shp in shapes[5:7]:
    tmp = pd.DataFrame(shp, columns=["geometry"])
    gdf_bayarea = pd.concat([gdf_bayarea, tmp])
gdf_bayarea = GeoDataFrame(gdf_bayarea)


# In[ ]:


collection_names = [item.split("/")[-1] for item in glob.glob("../input/smartphone-decimeter-2022/train/*")]

gdfs = []
for collection_name in collection_names:
    gdfs_each_collectionName = []
    csv_paths = glob.glob(f"../input/smartphone-decimeter-2022/train/{collection_name}/*/ground_truth.csv")
    for csv_path in csv_paths:
        df_gt = pd.read_csv(csv_path)
        df_gt['collectionName'] = collection_name
        df_gt["geometry"] = [Point(lngDeg, latDeg) for lngDeg, latDeg in zip(df_gt["LongitudeDegrees"], df_gt["LatitudeDegrees"])]
        gdfs_each_collectionName.append(GeoDataFrame(df_gt))
    gdfs.append(gdfs_each_collectionName)


# In[ ]:


colors = ['blue', 'green', 'purple', 'orange']
for collectionName, gdfs_each_collectionName in zip(collection_names, gdfs):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    gdf_bayarea.plot(figsize=(10,10), color='none', edgecolor='gray', zorder=3, ax=axs[0])
    for i, gdf in enumerate(gdfs_each_collectionName):
        g1 = gdf.plot(color=colors[i], ax=axs[0])
        g1.set_title(f"Phone track of {collectionName} with map")
        g2 = gdf.plot(color=colors[i], ax=axs[1])
        g2.set_title(f"Phone track of {collectionName}")


# There are several tracks that have the same form of data with different collectionName. It is easy to understand the positional relationship by overlapping them. There are two roads extending from the northwest to the southeast, and they seem to run along those roads all the time, or occasionally go off those roads. The tracks wandering around the grid-like paths seem to be collected farther southeast than those paths.

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 5))

for collectionName, gdfs_each_collectionName in zip(collection_names, gdfs):   
    for i, gdf in enumerate(gdfs_each_collectionName):
        gdf.plot(color=colors[i], ax=ax, markersize=5, alpha=0.5)


# In[ ]:


all_tracks = pd.DataFrame()

for collectionName, gdfs_each_collectionName in zip(collection_names, gdfs):   
    for i, gdf in enumerate(gdfs_each_collectionName):
        all_tracks = pd.concat([all_tracks, gdf])
        # Tracks they have same collectionName is also same
        break
        
fig = px.scatter_mapbox(all_tracks,
                            
                        # Here, plotly gets, (x,y) coordinates
                        lat="LongitudeDegrees",
                        lon="LatitudeDegrees",
                            
                        #Here, plotly detects color of series
                        color="collectionName",
                        labels="collectionName",
                            
                        zoom=9,
                        center={"lat":37.423576, "lon":-122.094132},
                        height=600,
                        width=800)
fig.update_layout(mapbox_style='stamen-terrain')
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.update_layout(title_text="GPS trafic")
fig.show()


# # How to check tracks in animation?¶
# 
# I am sure that you all would like to see animations of what the data was driving at. I have also prepared code to check the x, y coordinate movement in gif, so try it out.
# 
# The process takes some minutes (about 2 or 3 minutes for my implement).
# 
# Note: The x, y coordinates have been thinned out (to 1/10) because the processing uses much memory. You can adjust them if necessary.

# In[ ]:


def create_gif_track(df, git_path):
    """ Create git animation of phone track.
    """

    fig, ax = plt.subplots()

    imgs = []
    df["geometry"] = [Point(lngDeg, latDeg) for lngDeg, latDeg in zip(df["LongitudeDegrees"], df["LatitudeDegrees"])]
    gdf = GeoDataFrame(df)
    gdf.plot(color="lightskyblue", ax=ax)

    # Here, (x,y) coordinates are thinned out!!!
    for i in range(0, len(gdf), 10):
        # plot data
        p = ax.plot(gdf.iloc[i]["LongitudeDegrees"], gdf.iloc[i]["LatitudeDegrees"], 
                    color = 'dodgerblue', marker = 'o', markersize = 8)
        imgs.append(p)

    # Create animation & save it
    ani = animation.ArtistAnimation(fig, imgs, interval=200)
    ani.save(git_path, writer='imagemagick', dpi = 300)
    

def create_gif_track_on_map(df, gdf_map, git_path):
    """ Create git animation of phone track on bayarea map.
    """

    fig, ax = plt.subplots()
    df["geometry"] = [Point(lngDeg, latDeg) for lngDeg, latDeg in zip(df["LongitudeDegrees"], df["LatitudeDegrees"])]
    gdf = GeoDataFrame(df)
    gdf.plot(color="lightskyblue", ax=ax)
    imgs = []  
    gdf_map.plot(color='none', edgecolor='gray', zorder=3, ax=ax)
    
    # Here, (x,y) coordinates are thinned out!!!
    for i in range(0, len(gdf), 10):
        # plot data on map
        p = ax.plot(gdf.iloc[i]["LongitudeDegrees"], gdf.iloc[i]["LatitudeDegrees"], 
                    color = 'dodgerblue', marker = 'o', markersize = 8)
        imgs.append(p)

    # Create animation & save it
    ani = animation.ArtistAnimation(fig, imgs, interval=200)
    ani.save(git_path, writer='imagemagick', dpi = 300)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nGooglePixel4XLgt = pd.read_csv(DATA_PATH + "train/2020-05-15-US-MTV-1/GooglePixel4XL/ground_truth.csv")\n\ncreate_gif_track(GooglePixel4XLgt, "./GooglePixel4XLgt.gif")')


# 
