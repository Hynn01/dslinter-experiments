#!/usr/bin/env python
# coding: utf-8

# ## What is this notebook?
# 
# This notebook will test the following hypothesis: **Distance distribution varies by categories.**
# 
# In the `categories` section of the competition data, there are many types of location categories such as 'Cafés' or 'Shopping Malls'.
# 
# Each category type should have a rough size. For example, 'Cafés' should be relatively small, but not so large as to span the street. 'Shopping Malls', on the other hand, are relatively large and could take up an entire city block.
# 
# The distance between matched entities should be roughly proportional to the size of the entity, so if the size varies from category to category, it will appear in the distance between entities per category.
# 
# Let's see if there is a difference in distance between matched entities by category.

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import haversine_distances
from tqdm.auto import tqdm


pd.set_option("display.max_rows", 900)


# In[ ]:


train = pd.read_csv("../input/foursquare-location-matching/train.csv")
train.head()


# ## Calculate distances between matched entities

# In[ ]:


poi2distances = {}
for poi, df in tqdm(train[["latitude", "longitude", "point_of_interest"]].groupby("point_of_interest"),
                    total=train["point_of_interest"].nunique()):
    if len(df) == 1:
        # no matches
        continue
        
    distances = []
    distances_mat = haversine_distances(df[["latitude", "longitude"]].values)
    for i in range(len(df)):
        for j in range(len(df)):
            if j >= i:
                continue
            # haversine distance -> meters
            distances.append(distances_mat[i, j] * 6371000)
    poi2distances[poi] = distances


# In[ ]:


poi2distances_df = pd.DataFrame({
    "point_of_interest": list(poi2distances.keys()),
    "distances": list(poi2distances.values())
})
poi2distances_df.head()


# ## Split categories, merge with distances per point_of_interest

# In[ ]:


train["categories"] = train["categories"].fillna("").map(lambda x: x.split(", "))
train_ = train[["id", "name", "categories", "point_of_interest"]].merge(
    poi2distances_df, on="point_of_interest", how="inner")
train_["median_distances"] = train_["distances"].map(np.median)
train_.head()


# ## Aggregate distances for each categories

# In[ ]:


# make each row has only one category
exploded = train_.explode("categories").drop_duplicates(["point_of_interest", "categories"])
exploded.head()


# In[ ]:


def percentile25(x):
    return np.percentile(x, q=25)


def percentile75(x):
    return np.percentile(x, q=75)


aggregated_df = exploded.groupby("categories").agg({
    "median_distances": ["max", percentile75, "median", percentile25, "min"],
    "id": "count"
})

aggregated_df.head()


# ## Check the result
# 
# Now, we get some percentile values of distances between matched entities per categories. Let's see some of them.

# In[ ]:


aggregated_df.sort_values(by=("id", "count"), ascending=False).head(200)


# In[ ]:


aggregated_df.loc[aggregated_df.index.isin(["Cafés", "Shopping Malls"])]


# As you see, 'Cafés' and 'Shopping Malls' have different distance distributions.

# In[ ]:


aggregated_df[aggregated_df[("id", "count")] >= 100].sort_values(
    by=("median_distances", "percentile25"), ascending=False).head(20)


# Parks, Rivers, Mountains. It's quite natural to see these distances between matched entities in these categories are in large values.
# 
# How can we use this result to improve our models? One simple and easy idea is, to use `categories` (', ' splitted would be better) as a feature.
# 
# Another idea is to use `categories` when we pick candidate locations to make pairs to judge if they are matched. Naive way to make candidates is to threshold the distance between entities, but threshold should be different between 'Cafés' and 'Shopping Malls'.
# 
# There must be other ways to use this finding. Hope you'll make use of it.

# ## EOF

# In[ ]:




