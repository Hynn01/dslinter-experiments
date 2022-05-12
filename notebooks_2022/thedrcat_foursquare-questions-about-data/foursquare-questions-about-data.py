#!/usr/bin/env python
# coding: utf-8

# # Foursquare Location Matching - Questions About the Data
# 
# The goal of this notebook is to collect all questions and answers about the dataset from the Foursquare Location Matching competition. Some of the questions may be answered by looking at the data, and I'll try to answer them myself. Some questions, for example related to how the dataset was developed, are answered by the host on the forums. I'll collect those questions and answers along with links to the forum posts.
# 
# I'm personally excited about this competition because I can see the real life application, and I hope to learn some tabular data techniques I'm not so familiar with. On the other hand, I'm a bit worried that the synthetic nature of the dataset might lead to leakages. Maybe this analysis will help me (and you?) decide how much effort to invest in competing here. 
# 
# * <a href="#q1">How many matching pairs and unique locations are there?</a>
# * <a href="#q1a">Is the dataset balanced?</a>
# * <a href="#q2">What are some types of noise introduced in the dataset?</a>
# * <a href="#q3">Where is the data coming from?</a>
# * <a href="#q4">How was the "pairs.csv" file created?</a>
# * <a href="#q5">Do places in the train and test data overlap?</a>
# * <a href="#q6">How are these categories generated and are the categories of train and test the same distribution?</a>
# * <a href="#q7">What is the goal of this competition?</a>
# 
# More questions to be added...

# <a id="q1"></a>
# ## How many matching pairs and unique locations are there?
# 
# Let's start with some basic numbers from the dataset. 

# In[ ]:


# https://www.kaggle.com/code/sudalairajkumar/flm-additional-match-pairs-data
import numpy as np 
import pandas as pd 
train_df = pd.read_csv("/kaggle/input/foursquare-location-matching/train.csv")
match_df = pd.merge(train_df, train_df, on="point_of_interest", suffixes=('_1', '_2'))
match_df = match_df[match_df["id_1"]!=match_df["id_2"]]
# match_df = match_df.drop(["point_of_interest"], axis=1)
match_df["match"] = True


# In[ ]:


print(f'Number of locations in train: {len(train_df)}')
print(f'Number of unique points of interests in train: {train_df.point_of_interest.nunique()}')
print(f'Number of matching location pairs in train: {len(match_df)}')


# <a id="q1a"></a>
# ## Is the dataset balanced?
# 
# As a follow up to looking at number of unique places in the dataset, how balanced is it? Would we have roughly the same number of entries per unique location? **Answer:** no, the distribution of number of matching pairs across locations doesn't look balanced. 

# In[ ]:


poi_counts = train_df.point_of_interest.value_counts().to_frame().reset_index()
print(f'Number of unique points of interest in train: {len(poi_counts)}')
print(f'Number of POIs without a matching pair: {len(poi_counts[poi_counts.point_of_interest == 1])}')
print(f'Number of POIs with a single matching pair: {len(poi_counts[poi_counts.point_of_interest == 2])}')
print(f'Number of POIs with 3-10 matching pairs: {len(poi_counts[(poi_counts.point_of_interest > 2) & (poi_counts.point_of_interest <= 10)])}')
print(f'Number of POIs with 11-100 matching pairs: {len(poi_counts[(poi_counts.point_of_interest > 10) & (poi_counts.point_of_interest <= 100)])}')
print(f'Number of POIs with 101-332 matching pairs: {len(poi_counts[(poi_counts.point_of_interest > 100) & (poi_counts.point_of_interest <= 1000)])}')


# <a id="q2"></a>
# ## What are some types of noise introduced in the dataset?
# 
# Let's look at a single POI with the most locations in the dataset and see the range of values across the different attributes. 

# In[ ]:


train_df[train_df.point_of_interest == 'P_fb339198a31db3'].name.value_counts()


# In[ ]:


train_df[train_df.point_of_interest == 'P_fb339198a31db3'].categories.value_counts()


# In[ ]:


# Source: https://www.kaggle.com/code/remekkinas/eda-and-additional-feat-distance-adress-zip

from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame

def draw_lon_lat(df, world):
    geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
    gdf = GeoDataFrame(df, geometry=geometry)   
    gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15);
    
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
draw_lon_lat(train_df[train_df.point_of_interest == 'P_fb339198a31db3'], world);


# In[ ]:


train_df[train_df.point_of_interest == 'P_fb339198a31db3'].address.value_counts()


# In[ ]:


train_df[train_df.point_of_interest == 'P_fb339198a31db3'].city.value_counts()


# In[ ]:


train_df[train_df.point_of_interest == 'P_fb339198a31db3'].state.value_counts()


# In[ ]:


train_df[train_df.point_of_interest == 'P_fb339198a31db3'].zip.value_counts()


# In[ ]:


train_df[train_df.point_of_interest == 'P_fb339198a31db3'].country.value_counts()


# In[ ]:


train_df[train_df.point_of_interest == 'P_fb339198a31db3'].url.value_counts()


# In[ ]:


train_df[train_df.point_of_interest == 'P_fb339198a31db3'].phone.value_counts()


# In[ ]:


train_df[(train_df.point_of_interest == 'P_fb339198a31db3') & (train_df.longitude < 0)]


# We can see from this single POI example that the range of noise is significant. Especially the outlier shown above is interesting, the latitude, longitude and country point to US, only the name suggests it's actually the Soekarto airport in Indonesia.

# <a id="q3"></a>
# ## Where is the data coming from?
# 
# **Question by @xhlulu**: Are POIs simulated or from real places? Are Personally Identifiable Information respected?
# 
# **Answer by host:**
# This dataset is intended to provide information pertaining to real business sites/points of interest and originates from three primary sources:
# * User generated – In countries where Foursquare apps are available, user-generated content is collected and subject to an internal validation process prior to inclusion in the Places dataset.
# * Partner contributions – This content comes from trusted partners that, for example, may work with small businesses that want to share data in order to promote their business.
# * Publicly available sources – This content is aggregated from multiple open web sources containing publicly available information with a commercial or points of interest focus.
# 
# The dataset we use in this competition is derived from user generated content, partner contributions, and publicly available sources. It pertains to real business sites and points of interest. However, for purposes of this competition, we have materially altered the competition dataset with additional noise, modifications, and artificial information, which we had initially characterized as simulated and synthetic to avoid the expectation from users that the data was necessarily associated with real businesses.
# 
# 
# [Link](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/319722) [Link](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/318967)
# 

# <a id="q4"></a>
# ## How was the "pairs.csv" file created?
# 
# **Question by @ymatioun:** can you tell us how the file "pairs.csv" was created?
# 
# **Answer by host:** This is a good question. All matching and non-matching pairs in pairs.csv come from the information in train.csv; there is no additional information in pairs.csv that does not exist in the train.csv file. Matches are pairs of places with the same point_of_interest ids, non-matches are samples of places with different point_of_interest ids. pairs.csv contains samples of matching and non-matching pairs with the purpose to help model training, and it is far from inclusive of all combinations that can be generated from train.csv. You may choose to use pairs.csv as is, modify it (remove matching and non-matching pairs or add new ones generated from train.csv), or disregard it completely, depending on your training strategy.
# 
# [Link](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/318967)

# <a id="q5"></a>
# ## Do places in the train and test data overlap?
# 
# **Question by @kaggledummie007:** 
# 
# **Answer by host:** The places in the train and test data do not overlap, you only need to find matches within the test data.
# 
# 
# [Link](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/318967)

# <a id="q6"></a>
# ## How are these categories generated and are the categories of train and test the same distribution?
# 
# **Question by @columbia2131:** How are these categories generated? For example, is the system such that the store can select multiple choices from those that already exist? And, are the categories of train and test the same distribution?
# 
# **Answer by host:** In this competition we assume that the data may be coming from different sources, different providers. Each provider can follow different taxonomies for categories, or even allow free text. That is, some providers may follow a hierarchical schema with different levels of granularity (e.g., "retail store" > "clothes and shoes store" > "clothes store" > "large-size clothes store"), or a flattened schema (e.g. "shopping"). In practice, this means that the categories entries are not expected to follow a single taxonomy of normalized mutually exclusive values, but rather a mix of taxonomies and different levels of the taxonomy used. For the above example, even if two matching places are generated following the same taxonomy, their categories may be entered in different levels of the hierarchy: one place as "clothes store" and the other as "large-size clothes store"). Sometimes the taxonomies may be the same or very similar to encourage a simple one-hot vector encoding, but this is not always the case.
# 
# On your second question, there is no assumption that the category attributes in the train and test sets follow the same distributions. In addition, there can be category values in the test set that do not exist in the train set, and vice versa.
# 
# 
# [Link](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/318967)

# <a id="q7"></a>
# ## What is the goal of this competition?
# 
# **Question by @atulverma:** I wanted to understand the relevance of this dataset and competition to Foursquare in real life context.
# 
# **Answer by host:** The main focus of this competition is to raise awareness on the data science community on one of the most challenging problems we work on at Foursquare. As Foursquare currently maintains a first party database of 100M+ POIs around the globe, and ingests an increasing amount of POI information from multiple sources on a daily basis, performing data matching and entity resolution at scale becomes a significantly complicated problem.
# 
# The dataset we compiled for this competition is a synthetic dataset with the purpose to expose and simulate several of the real-life challenges we face at Foursquare: e.g., POIs with missing information; attributes in multiple languages; and unnormalized, noisy, or inaccurate data, to name but a few. As it is a synthetic dataset, Foursquare does not benefit from the submitted answers, or any processing applied in the data. Our purpose is to provide a dataset and problem definition as realistic as possible to engage the community and offer a glimpse on the interesting and challenging problems we work on on a daily basis.
# 
# 
# [Link](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/318967)
