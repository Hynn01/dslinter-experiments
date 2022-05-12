#!/usr/bin/env python
# coding: utf-8

# **This notebook is an exercise in the [Geospatial Analysis](https://www.kaggle.com/learn/geospatial-analysis) course.  You can reference the tutorial at [this link](https://www.kaggle.com/alexisbcook/manipulating-geospatial-data).**
# 
# ---
# 

# # Introduction
# 
# You are a Starbucks big data analyst ([that’s a real job!](https://www.forbes.com/sites/bernardmarr/2018/05/28/starbucks-using-big-data-analytics-and-artificial-intelligence-to-boost-performance/#130c7d765cdc)) looking to find the next store into a [Starbucks Reserve Roastery](https://www.businessinsider.com/starbucks-reserve-roastery-compared-regular-starbucks-2018-12#also-on-the-first-floor-was-the-main-coffee-bar-five-hourglass-like-units-hold-the-freshly-roasted-coffee-beans-that-are-used-in-each-order-the-selection-rotates-seasonally-5).  These roasteries are much larger than a typical Starbucks store and have several additional features, including various food and wine options, along with upscale lounge areas.  You'll investigate the demographics of various counties in the state of California, to determine potentially suitable locations.
# 
# <center>
# <img src="https://i.imgur.com/BIyE6kR.png" width="450"><br/><br/>
# </center>
# 
# Before you get started, run the code cell below to set everything up.

# In[ ]:


import math
import pandas as pd
import geopandas as gpd
#from geopy.geocoders import Nominatim            # What you'd normally run
from learntools.geospatial.tools import Nominatim # Just for this exercise

import folium 
from folium import Marker
from folium.plugins import MarkerCluster

from learntools.core import binder
binder.bind(globals())
from learntools.geospatial.ex4 import *


# You'll use the `embed_map()` function from the previous exercise to visualize your maps.

# In[ ]:


def embed_map(m, file_name):
    from IPython.display import IFrame
    m.save(file_name)
    return IFrame(file_name, width='100%', height='500px')


# # Exercises
# 
# ### 1) Geocode the missing locations.
# 
# Run the next code cell to create a DataFrame `starbucks` containing Starbucks locations in the state of California.

# In[ ]:


# Load and preview Starbucks locations in California
starbucks = pd.read_csv("../input/geospatial-learn-course-data/starbucks_locations.csv")
starbucks.head()


# Most of the stores have known (latitude, longitude) locations.  But, all of the locations in the city of Berkeley are missing.

# In[ ]:


# How many rows in each column have missing values?
print(starbucks.isnull().sum())

# View rows with missing locations
rows_with_missing = starbucks[starbucks["City"]=="Berkeley"]
rows_with_missing


# Use the code cell below to fill in these values with the Nominatim geocoder.
# 
# Note that in the tutorial, we used `Nominatim()` (from `geopy.geocoders`) to geocode values, and this is what you can use in your own projects outside of this course.  
# 
# In this exercise, you will use a slightly different function `Nominatim()` (from `learntools.geospatial.tools`).  This function was imported at the top of the notebook and works identically to the function from GeoPandas.
# 
# So, in other words, as long as: 
# - you don't change the import statements at the top of the notebook, and 
# - you call the geocoding function as `geocode()` in the code cell below, 
# 
# your code will work as intended!

# In[ ]:


# Create the geocoder
geolocator = Nominatim(user_agent="kaggle_learn")

# Your code here
____

# Check your answer
q_1.check()


# In[ ]:


# Line below will give you solution code
#q_1.solution()


# ### 2) View Berkeley locations.
# 
# Let's take a look at the locations you just found.  Visualize the (latitude, longitude) locations in Berkeley in the OpenStreetMap style. 

# In[ ]:


# Create a base map
m_2 = folium.Map(location=[37.88,-122.26], zoom_start=13)

# Your code here: Add a marker for each Berkeley location
____

# Uncomment to see a hint
#q_2.a.hint()

# Show the map
embed_map(m_2, 'q_2.html')


# In[ ]:


# Get credit for your work after you have created a map
q_2.a.check()

# Uncomment to see our solution (your code may look different!)
#q_2.a.solution()


# Considering only the five locations in Berkeley, how many of the (latitude, longitude) locations seem potentially correct (are located in the correct city)?

# In[ ]:


# View the solution (Run this code cell to receive credit!)
q_2.b.solution()


# ### 3) Consolidate your data.
# 
# Run the code below to load a GeoDataFrame `CA_counties` containing the name, area (in square kilometers), and a unique id (in the "GEOID" column) for each county in the state of California.  The "geometry" column contains a polygon with county boundaries.

# In[ ]:


CA_counties = gpd.read_file("../input/geospatial-learn-course-data/CA_county_boundaries/CA_county_boundaries/CA_county_boundaries.shp")
CA_counties.head()


# Next, we create three DataFrames:
# - `CA_pop` contains an estimate of the population of each county.
# - `CA_high_earners` contains the number of households with an income of at least $150,000 per year.
# - `CA_median_age` contains the median age for each county.

# In[ ]:


CA_pop = pd.read_csv("../input/geospatial-learn-course-data/CA_county_population.csv", index_col="GEOID")
CA_high_earners = pd.read_csv("../input/geospatial-learn-course-data/CA_county_high_earners.csv", index_col="GEOID")
CA_median_age = pd.read_csv("../input/geospatial-learn-course-data/CA_county_median_age.csv", index_col="GEOID")


# Use the next code cell to join the `CA_counties` GeoDataFrame with `CA_pop`, `CA_high_earners`, and `CA_median_age`.
# 
# Name the resultant GeoDataFrame `CA_stats`, and make sure it has 8 columns: "GEOID", "name", "area_sqkm", "geometry", "population", "high_earners", and "median_age".  Also, make sure the CRS is set to `{'init': 'epsg:4326'}`.

# In[ ]:


# Your code here
CA_stats = ____

# Check your answer
q_3.check()


# In[ ]:


# Lines below will give you a hint or solution code
#q_3.hint()
#q_3.solution()


# Now that we have all of the data in one place, it's much easier to calculate statistics that use a combination of columns.  Run the next code cell to create a "density" column with the population density.

# In[ ]:


CA_stats["density"] = CA_stats["population"] / CA_stats["area_sqkm"]


# ### 4) Which counties look promising?
# 
# Collapsing all of the information into a single GeoDataFrame also makes it much easier to select counties that meet specific criteria.
# 
# Use the next code cell to create a GeoDataFrame `sel_counties` that contains a subset of the rows (and all of the columns) from the `CA_stats` GeoDataFrame.  In particular, you should select counties where:
# - there are at least 100,000 households making \$150,000 per year,
# - the median age is less than 38.5, and
# - the density of inhabitants is at least 285 (per square kilometer).
# 
# Additionally, selected counties should satisfy at least one of the following criteria:
# - there are at least 500,000 households making \$150,000 per year,
# - the median age is less than 35.5, or
# - the density of inhabitants is at least 1400 (per square kilometer).

# In[ ]:


# Your code here
sel_counties = ____

# Check your answer
q_4.check()


# In[ ]:


# Lines below will give you a hint or solution code
#q_4.hint()
#q_4.solution()


# ### 5) How many stores did you identify?
# 
# When looking for the next Starbucks Reserve Roastery location, you'd like to consider all of the stores within the counties that you selected.  So, how many stores are within the selected counties?
# 
# To prepare to answer this question, run the next code cell to create a GeoDataFrame `starbucks_gdf` with all of the starbucks locations.

# In[ ]:


starbucks_gdf = gpd.GeoDataFrame(starbucks, geometry=gpd.points_from_xy(starbucks.Longitude, starbucks.Latitude))
starbucks_gdf.crs = {'init': 'epsg:4326'}


# So, how many stores are in the counties you selected?

# In[ ]:


# Fill in your answer
num_stores = ____

# Check your answer
q_5.check()


# In[ ]:


# Lines below will give you a hint or solution code
#q_5.hint()
#q_5.solution()


# ### 6) Visualize the store locations.
# 
# Create a map that shows the locations of the stores that you identified in the previous question.

# In[ ]:


# Create a base map
m_6 = folium.Map(location=[37,-120], zoom_start=6)

# Your code here: show selected store locations
____

# Uncomment to see a hint
#q_6.hint()

# Show the map
embed_map(m_6, 'q_6.html')


# In[ ]:


# Get credit for your work after you have created a map
q_6.check()

# Uncomment to see our solution (your code may look different!)
#q_6.solution()


# # Keep going
# 
# Learn about how **[proximity analysis](https://www.kaggle.com/alexisbcook/proximity-analysis)** can help you to understand the relationships between points on a map.

# ---
# 
# 
# 
# 
# *Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/geospatial-analysis/discussion) to chat with other learners.*
