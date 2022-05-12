#!/usr/bin/env python
# coding: utf-8

# **V1**: Basic EDA  
# **V3**: Added GIF

# In[ ]:


import os
import numpy as np 
import pandas as pd

import glob
import itertools

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import plotly.express as px
import seaborn

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')

DATA_PATH = "../input/smartphone-decimeter-2022/"


# In[ ]:


sub = pd.read_csv(DATA_PATH + "sample_submission.csv")
sub.head()


# In[ ]:


#df_sample_trail = pd.read_csv(DATA_PATH + "train/2020-05-15-US-MTV-1/GooglePixel4XL/device_gnss.csv")
df_sample_trail_gt = pd.read_csv(DATA_PATH + "train/2020-05-15-US-MTV-1/GooglePixel4XL/ground_truth.csv")


# In[ ]:


df_sample_trail_gt.head(6)


# In[ ]:


FILE_NAME = '2020-05-15-US-MTV-1'


# In[ ]:


def visualize_trafic(df, center, zoom=8):
    fig = px.scatter_mapbox(df,
                            zoom=zoom,
                            center=center,
                            
                            lat="LatitudeDegrees",
                            lon="LongitudeDegrees",
                            color="MessageType",
                            labels='Provider',
                            
                            height=600,
                            width=800)
    fig.update_layout(mapbox_style='stamen-terrain')
    fig.show()


# In[ ]:


df_sample_trail_gt = pd.read_csv(DATA_PATH + "train/{}/GooglePixel4XL/ground_truth.csv".format(FILE_NAME))

center = {"lat":37.42, "lon":-122.1}
visualize_trafic(df_sample_trail_gt, center)


# In[ ]:


#Borrowed from prev. competition notebook
def add_lack_data(data, phoneNames, collectionNames):
    result = []
    for collectionName in collectionNames:
        for phoneName in phoneNames:
            if [collectionName, phoneName] in data:
                result.append([collectionName, phoneName, 1])
            else:
                result.append([collectionName, phoneName, 0])
    return result


# In[ ]:


train_phones = glob.glob("../input/smartphone-decimeter-2022/train/*/*")
test_phones = glob.glob("../input/smartphone-decimeter-2022/test/*/*")
train_data = [item.split("/")[-2:] for item in train_phones]
test_data = [item.split("/")[-2:] for item in test_phones]


# In[ ]:


phoneNames = [item.split("/")[-1:] for item in train_phones] + [item.split("/")[-1:] for item in test_phones]
phoneNames = set(list(itertools.chain.from_iterable(phoneNames)))
collectionNames = [item.split("/")[-2] for item in train_phones] + [item.split("/")[-2] for item in test_phones]


# In[ ]:


train_data = add_lack_data(train_data, phoneNames, collectionNames)
test_data = add_lack_data(test_data, phoneNames, collectionNames)
train_data = pd.DataFrame(train_data, columns=["collectionNames", "phoneNames", "count"])
test_data = pd.DataFrame(test_data, columns=["collectionNames", "phoneNames", "count"])


# In[ ]:


for phoneName in phoneNames:
    fig, axes = plt.subplots(1, 2, figsize=(20,30))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    g1 = seaborn.barplot(data=train_data[train_data["phoneNames"] == phoneName],
                         x="count", y="collectionNames", ax=axes[0])
    g1.set_title(f"{phoneName} in train data")
    g2 = seaborn.barplot(data=test_data[test_data["phoneNames"] == phoneName],
                         x="count", y="collectionNames", ax=axes[1])
    g2.set_title(f"{phoneName} in test data")


# In[ ]:


from shapely.geometry import Point, shape
from geopandas import GeoDataFrame
from matplotlib import animation
def create_gif_track(df, git_path):
    """ Create git animation of phone track.
    """

    fig, ax = plt.subplots()

    imgs = []
    df["geometry"] = [Point(lngDeg, latDeg) for lngDeg, latDeg in zip(df["LongitudeDegrees"], df["LatitudeDegrees"])]
    gdf = GeoDataFrame(df)
    gdf.plot(color="lightskyblue", ax=ax)
    
    for i in range(0, len(gdf), 10):
        p = ax.plot(gdf.iloc[i]["LongitudeDegrees"], gdf.iloc[i]["LatitudeDegrees"], 
                    color = 'dodgerblue', marker = 'o', markersize = 10)
        imgs.append(p)

    # Create animation and saving it
    ani = animation.ArtistAnimation(fig, imgs, interval=200)
    ani.save(git_path, writer='imagemagick', dpi = 300)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ngif_gt = pd.read_csv(DATA_PATH + "train/2021-04-26-US-SVL-2/SamsungGalaxyS20Ultra/ground_truth.csv")\n\ncreate_gif_track(gif_gt, "./2021_04_29_US_SJC_2_Pixel4_gt.gif")')


# ![](./2021_04_29_US_SJC_2_Pixel4_gt.gif)

# In[ ]:





# In[ ]:




