#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from pathlib import Path
import plotly.express as px

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# #Thank you Kuto for your magnificent, pleasant script.  By the way, I ruined it.
# 
# https://www.kaggle.com/code/kuto0633/road-detection-and-creating-grid-points

# In[ ]:


#Code by Kuto https://www.kaggle.com/code/kuto0633/road-detection-and-creating-grid-points

data_dir = Path("../input/smartphone-decimeter-2022")
train_df = pd.read_csv(data_dir / "train/2021-04-29-US-MTV-2/SamsungGalaxyS20Ultra/ground_truth.csv")


# In[ ]:


#Code by Kuto https://www.kaggle.com/code/kuto0633/road-detection-and-creating-grid-points

# get all ground truth dataframe
gt_df = pd.DataFrame()
for (MessageType, Provider), df in tqdm(train_df.groupby(["MessageType", "Provider"])):
    path = data_dir / f"train/2021-04-29-US-MTV-2/SamsungGalaxyS20Ultra/ground_truth.csv"
    df = pd.read_csv(path)  
    gt_df = pd.concat([gt_df, df]).reset_index(drop=True)   
gt_df.head()


# In[ ]:


#Code by Kuto https://www.kaggle.com/code/kuto0633/road-detection-and-creating-grid-points

fig = px.scatter_mapbox(gt_df,

                    # Here, plotly gets, (x,y) coordinates
                    lat="LatitudeDegrees",
                    lon="LongitudeDegrees",
                    text='Provider',

                    #Here, plotly detects color of series
                    color="MessageType",
                    labels="MessageType",

                    zoom=9,
                    center={"lat":37.395817, "lon":-122.102916},
                    height=600,
                    width=800)
fig.update_layout(mapbox_style='stamentoner')
#fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.update_layout(title_text="I lost my GPS!")
fig.show()


# In[ ]:


#Code by Kuto https://www.kaggle.com/code/kuto0633/road-detection-and-creating-grid-points

target_message = '2021-04-29-US-SJC-2'
target_gt_df = gt_df[gt_df["MessageType"]==target_message].reset_index(drop=True)
target_message


# In[ ]:


#Code by Kuto https://www.kaggle.com/code/kuto0633/road-detection-and-creating-grid-points

fig = px.scatter_mapbox(target_gt_df,

                    # Here, plotly gets, (x,y) coordinates
                    lat="LatitudeDegrees",
                    lon="LongitudeDegrees",
                    text='Provider',

                    #Here, plotly detects color of series
                    color="MessageType",
                    labels="Provider",

                    zoom=15,
                    center={"lat":37.395817, "lon":-122.102916},
                    height=600,
                    width=800)
fig.update_layout(mapbox_style='stamentoner')
#fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.update_layout(title_text="GPS is lost again!")
fig.show()


# #Geographical library

# ![](https://i.ytimg.com/vi/310Es1ERf1M/maxresdefault.jpg)youtube.com

# In[ ]:


get_ipython().system('pip install osmnx momepy geopandas')


# In[ ]:


from shapely.geometry import Point
import osmnx as ox
import momepy
import geopandas as gpd


# In[ ]:


#Code by Kuto https://www.kaggle.com/code/kuto0633/road-detection-and-creating-grid-points

# change pd.DataFrame -> gpd.GeoDataFrame
target_gt_df["geometry"] = [Point(p) for p in target_gt_df[["LatitudeDegrees", "LongitudeDegrees"]].to_numpy()]
target_gt_gdf = gpd.GeoDataFrame(target_gt_df, geometry=target_gt_df["geometry"])
target_gt_gdf.head(5)


# In[ ]:


#Code by Kuto https://www.kaggle.com/code/kuto0633/road-detection-and-creating-grid-points

target_gt_gdf.plot();


# #Trying to fix the code. And not getting it.

# In[ ]:


#Code by https://stackoverflow.com/questions/62075847/using-qgis-and-shaply-error-geosgeom-createlinearring-r-returned-a-null-pointer

...
from shapely import speedups
...

speedups.disable()


# In[ ]:


#Code by https://stackoverflow.com/questions/62075847/using-qgis-and-shaply-error-geosgeom-createlinearring-r-returned-a-null-pointer

"""
The shapely.speedups module contains performance enhancements written in C.
They are automaticaly installed when Python has access to a compiler and
GEOS development headers during installation, and are enabled by default.
"""


# In[ ]:


#Code by https://stackoverflow.com/questions/62075847/using-qgis-and-shaply-error-geosgeom-createlinearring-r-returned-a-null-pointer

from shapely.geometry import shape


# In[ ]:


#Code by https://stackoverflow.com/questions/62075847/using-qgis-and-shaply-error-geosgeom-createlinearring-r-returned-a-null-pointer

import shapely
shapely.speedups.disable()


# In[ ]:


#Code by Kuto https://www.kaggle.com/code/kuto0633/road-detection-and-creating-grid-points

# get road data from open street map by osmnx
#offset = 0.1**5
#bbox = target_gt_gdf.bounds + [-offset, -offset, offset, offset]
#east = bbox["minx"].min()
#west = bbox["maxx"].max()
#south = bbox["miny"].min()
#north = bbox["maxy"].max()
#G = ox.graph.graph_from_bbox(north, south, east, west, network_type='drive')


# #InvalidGeometryError: Null geometry supports no operations
# 
# I have No clue how to fix that.

# #At least one map. Though I don't know where I am. I'm still lost!

# In[ ]:


#Code by Juan Cruz Martinez https://livecodestream.dev/post/how-to-plot-your-data-on-maps-using-python-and-folium/
#When I wrote on the tooltip: Click Here For More Info, I could only read in the viewer but not in the edit

import folium

#We're in Palo Alto, are we Samsung?

m = folium.Map(location=[37.395817, -122.102916], zoom_start=15)

tooltip = "Where my Samsung brought me"
folium.TileLayer('stamentoner').add_to(m)#That changes the map to black/white

marker = folium.CircleMarker(
    location=[37.395817,-122.102916 ],
    radius=50,
    popup="<stong>Palo Alto with my Samsung Galaxy</stong>",
    tooltip=tooltip)
marker.add_to(m)

m


# #I gave up. No GPS signal. I'm not shy to embarasse myself.

# #Acknowledgement:
# 
# https://www.kaggle.com/code/kuto0633/road-detection-and-creating-grid-points
# 
# I'm sorry to have ruined your code Kuto.

# #Probably, many are laughing at "my" code.
# 
# However, it's exactly what happens when people lose their GPS. They are non-coders like me. And obviously they blame their expensive gadget. Those are the ones that buy anything smart just because they are suppose to be smart (smartphones/smartwatches/smartwhatever). Therefore, it's better not laughing at US. 
