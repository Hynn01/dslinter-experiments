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


# *This dataset contains Netflix data from various TV shows and movies.
# Netflix is a subscription-based streaming service that allows members to watch TV shows and movies on an internet-connected device without commercials.*

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# In[ ]:


data =pd.read_csv("../input/netflix-shows/netflix_titles.csv")
data.head()


# In[ ]:


#sorting the data in descending order by release year
data.sort_values("release_year",ascending=False)


# In[ ]:


#sum of the null values present in the data
data.isnull().sum()


# In[ ]:


#percentage of the null values present in the data

data.isnull().sum()/len(data.index)*100


# The maximum number of null values are present in the director column (~30%).

# In[ ]:


#check for duplicate data in "show_id" column
data["show_id"].duplicated().any()


# In[ ]:


#putting "unknown" in the director column's null values
data['director'].fillna('Unknown', inplace=True)


# In[ ]:


#putting "unknown" in the cast and country column's null values
data['cast'].fillna('Unknown', inplace=True)
data['country'].fillna('Unknown', inplace=True)


# In[ ]:


#removing null values from the data_added, rating, and duration columns
data.dropna(subset=['date_added','rating','duration'],inplace=True)


# In[ ]:


data.head()


# In[ ]:


# unique values present in each column
data.nunique()


# In[ ]:


data.type.value_counts().to_frame()


# In[ ]:


data.type.value_counts(normalize=True).plot(kind='pie')


# The dataset contains two types of records: movies and TV shows.

# 

# In[ ]:


data.release_year.value_counts().sort_index(ascending=False)


# 

# In[ ]:


net_year=data.release_year.value_counts().sort_index(ascending=False).head(20).plot(kind="bar")


# The plot depicts the number of movies and TV shows that are released in a given year.

# In[ ]:


temp1 = data.set_index('title').country.str.split(', ', expand=True).stack().reset_index(level=1, drop=True)

plt.figure(figsize=(15,10))
sns.countplot(y = temp1, order=temp1.value_counts().index[:25], palette="Spectral")


# The plot depicts the number of movies and TV shows released by the country.
# 
# The United States ranked first in terms of the number of movies and TV shows available, followed by India.

# 

# In[ ]:


test = pd.DataFrame(data.cast.str.split(',').tolist(), index=data.type).stack()
test = test.reset_index([0, 'type'])
test.columns = ['type', 'actor']


# In[ ]:


#Top 10 actor in movies
test.loc[(test["type"]=='Movie') & (test["actor"] != 'Unknown')]["actor"].value_counts().head(10).to_frame()


# In[ ]:


#Top 10 actor in TV shows
test.loc[(test["type"]=='TV Show') & (test["actor"] != 'Unknown')]["actor"].value_counts().head(10).to_frame()


# In[ ]:


test = pd.DataFrame(data.listed_in.str.split(',').tolist(), index=data.type).stack()
test = test.reset_index([0, 'type'])
test.columns = ['type', 'list']


# In[ ]:


test.loc[(test["list"] !="unknown")]["list"].value_counts().head(20).drop(" Dramas")


# In[ ]:


#analyzing the movies and TV shows separately
net_movies = data[data['type']=='Movie'].copy()
net_shows = data[data['type']=='TV Show'].copy()


# In[ ]:


net_movies["duration"].isnull().value_counts()


# In[ ]:


net_movies["duration"] = net_movies.duration.str.replace(" min",'').astype(int)


# In[ ]:


net_movies


# In[ ]:


net_movies.value_counts("duration")


# In[ ]:


plt.figure(figsize=(15,10))
sns.displot(net_movies.duration,color='blue')


# It is clear that the majority of the movies range in length from 90 to 100 minutes.

# In[ ]:


net_shows.rename(columns={'duration':'season'}, inplace = True)


# In[ ]:


net_shows["season"] = net_shows.season.str.replace("Season",'')
net_shows["season"] = net_shows.season.str.replace("Seasons",'')


# In[ ]:


net_shows["season"] = net_shows.season.str.replace("s",'')


# In[ ]:


net_shows


# In[ ]:


net_shows.value_counts("season")


# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(net_shows.season.astype(int), color="purple")


# The plot represents that the majority of TV shows have only one season.

# In[ ]:


data.value_counts("rating")


# https://help.netflix.com/en/node/2064/us

# 

# rating indicates that the majority of the movies and TV shows on Netflix are appropriate for mature audiences or should be watched with parental supervision.
# 
# It is not recommended for children under the age of 17 (TV-MA).
# 
# Parents were also strongly cautioned. Children under the age of 14 may not be suitable (TV-14).
# 
# TV-Y rating is appropriate for all children

# In[ ]:


plt.figure(figsize=(15,7))
sns.countplot(data["rating"],hue = data["type"],data = data)


# In[ ]:




