#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Load data
data = pd.read_csv("../input/prediction-of-music-genre/music_genre.csv")

# Remove data that will not be used
data = data.drop(
    ['instance_id', 'track_name', 'energy', 'acousticness', 'instrumentalness', 'obtained_date', 'tempo', 'valence', 'liveness', 'loudness', 'speechiness', 'key', 'mode'],
    axis=1
)

data


# In[ ]:


# Analazy infos
# data['popularity'].value_counts()

# data['danceability'].value_counts()
# max(data['danceability'])

# data['duration_ms'].value_counts()
# min(data['duration_ms'])
# max(data['duration_ms'])

data.describe()


# In[ ]:


data.dtypes


# In[ ]:


# Processing
dataToShow = data.copy()
dataToShow.columns = ['artist', 'popularity', 'danceability', 'duration', 'genre']

dataToShow['popularity'] = pd.cut(x=dataToShow['popularity'], bins=[0, 25, 35, 45, 55, 75], labels=['Very low', 'Low', 'Medium', 'High', 'Very High'])
dataToShow['danceability'] = pd.cut(x=dataToShow['danceability'], bins=[0, 0.30, 0.40, 0.55, 0.70, 0.80], labels=['Very low', 'Low', 'Medium', 'High', 'Very High'])
dataToShow['duration'] = (dataToShow['duration'].fillna(0) / 100 / 60).astype(int)

dataToShow.head(7)


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(30, 7))
dataToShow['danceability'].value_counts().plot(kind='pie', ax=ax)


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(35, 15))
sns.barplot(x="genre", y="duration", hue="danceability", data=dataToShow, ax=ax)


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(35, 15))
sns.barplot(x="genre", y="duration", hue="popularity", data=dataToShow, ax=ax)

