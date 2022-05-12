#!/usr/bin/env python
# coding: utf-8

# # Spotify-EDA
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#importing the dataset
tracks = pd.read_csv('../input/spotify-tracks-data/tracks.csv')
features = pd.read_csv('../input/music-r/SpotifyFeatures.csv')


# In[ ]:


#checking the null values
tracks.isnull().sum()


# In[ ]:


#getting info of dataset
tracks.info()


# In[ ]:


#Creating a new variable that contatins the ten least popular songs 

least_popular = tracks.sort_values('popularity', ascending=True).head(10)
least_popular


# In[ ]:


#Checking for the 10 most popular songs having popularity greater than 90
most_popular = tracks[tracks['popularity']>90].sort_values('popularity', ascending=False).head(10)
most_popular


# In[ ]:


#changing the datatype of release_date to datetime
tracks['release_date']=pd.to_datetime(tracks.release_date)


# In[ ]:


#extracting month from release_date
tracks['month'] = tracks['release_date'].dt.month


# In[ ]:


#changing the index to release_date
tracks.set_index('release_date',inplace=True)


# In[ ]:


tracks.head(5)


# In[ ]:


#changing the miliseconds to seconds
tracks['duration']=tracks['duration_ms'].apply(lambda x: round(x/1000))
tracks.drop('duration_ms',inplace=True,axis=1)


# In[ ]:


tracks.head(3)


# In[ ]:


#correlation
corr = tracks.drop(['key','mode','explicit'],axis=1).corr(method='pearson')


# In[ ]:


plt.figure(figsize=(10,7))
sns.heatmap(corr, annot=True, vmin=-1,center=0, cmap='inferno',linewidths=1, linecolor='Black')
plt.title('Correlation Heatmap Between Variables');


# In[ ]:


sample_tracks=tracks.sample(int(0.004*len(tracks)))


# In[ ]:


#creating a regression plot for loudness and energy which was showing the moderate high correlation
plt.figure(figsize=(10,7))
sns.regplot(data=sample_tracks, y='loudness', x='energy', color='c')
plt.title('Loudness V/S Energy Correlation');


# In[ ]:


#checking regression plot for Poularity and Acousticness
plt.figure(figsize=(10,7))
sns.regplot(data=sample_tracks, y='popularity', x='acousticness', color='g')
plt.title('Popularity V/S Acousticness Correlation');


# In[ ]:


#Creating year column
tracks['dates'] = tracks.index.get_level_values('release_date')
tracks.dates=pd.to_datetime(tracks.dates)
years=tracks.dates.dt.year


# In[ ]:


#pip install --user seaborn==0.11.0


# In[ ]:


plt.figure(figsize=(10,7))
sns.displot(years)
plt.title('Year-Wise Songs Release Count');


# In[ ]:


#year wise duration of songs
dur = tracks.duration
plt.figure(figsize=(18,7))
sns.barplot(x=years, y=dur, errwidth =False)
plt.title('Year V/s Duration')
plt.xticks(rotation=90);


# In[ ]:


#analysis of features dataset
features.head(5)


# In[ ]:


#checking the duration of the songs as per the different genre
plt.figure(figsize=(10,8))
sns.color_palette('rocket', as_cmap = True)
sns.barplot(x=features.duration_ms, y=features.genre, errwidth=False)
plt.xlabel('duration in millisecond')
plt.ylabel('genre');


# In[ ]:


#Top 5 genre by popularity
best = features.sort_values('popularity', ascending = False)
best10 = best[['genre','popularity']].head(10)
plt.figure(figsize=(10,5))
sns.set_style(style='darkgrid')
sns.barplot(x=best10.genre, y=best10.popularity)
plt.title('Top 5 Genres by Popularity');

