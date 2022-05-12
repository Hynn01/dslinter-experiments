#!/usr/bin/env python
# coding: utf-8

# # **EDA Simplified: Foursquare Location Matching (BETA)**

# ## Introduction
# When I've entered this competition over locations and matching them with AI, I remembered this:
# 
# > *It's a small world after all, It's a small world after all, It's a small world after all, It's a small, small world...*
# 
# And for sure, I have a challenge. Can you identify the quote mentioned in this introduction quickly within 15 seconds?
# 
# If you incorrectly guessed this quote or you ran out of time, then better luck next time. But if you guess this quote from the beginning correctly, then you're excellent! you've see that it was from the nursery rhyme, **It's a Small World**! Speaking of that, this small world contained possible locations that you may go, since I am a <span style="color: #7393B3">Tsurezure</span> traveller. But when Foursquare released a competition like this, then this may be a great opportunity to explore around the world. But what's this purpose of this competition? The purpose of this competition is that you’ll match POIs together by using a simulated dataset from Foursquare of over one-and-a-half million Place entries. Using this make you produce an algorithm that predicts which Place entries represent the same point-of-interest. Each Place entry includes attributes like the name, street address, and coordinates. Successful submissions will identify matches with the greatest accuracy. And if you've got this done correctly, then you'll make it easier to identify where new stores or businesses would benefit people the most. *Bada-boom, Bada-bing.*

# ### Quick Heads-Up: About Foursquare
# Before let's proceed to our EDA analysis, let's talk about the creator of this competition, Foursquare. Foursquare is the most trusted, independent location data platform for understanding how people move through the real world. With 12+ years of experience perfecting such methods, Foursquare is the first independent provider of global POI data. The leading independent location technology and data cloud platform, Foursquare is dedicated to building meaningful bridges between digital spaces and physical places. Trusted by leading enterprises like Apple, Microsoft, Samsung, and Uber, Foursquare’s tech stack harnesses the power of places and movement to improve customer experiences and drive better business outcomes. With that, let's move in to EDA.

# ## Imports
# For importing modules in this EDA notebook, there's something special to happen. First, we import geopandas, since this competition is based over location. Then, we import the pandas module as pd for data science and the numpy module as np for linear algebra. We also use plotting modules like: matplotlib with the pyplot module as plt, seaborn as sns, and plotly with the graph_objects submodule as go and express submodule as px.

# In[ ]:


import geopandas
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px


# ## Dataframe Creation
# After importing our stuff, then we create two dataframes, out of pairs.csv file and train.csv file by defining two variables: pairs_df and train_df to put the file paths leading to them in the pd module with the read_csv function.

# In[ ]:


pairs_df = pd.read_csv("../input/foursquare-location-matching/pairs.csv")
train_df = pd.read_csv("../input/foursquare-location-matching/train.csv")


# Yet after that, we display the first 5 rows of our newly created two dataframes by plugging the head function into them!

# In[ ]:


pairs_df.head()


# In[ ]:


train_df.head()


# ## First Peek in this Data
# Now, after creating our two dataframes, let's take a quick, quick peek! First, we check whether there's any NaN entities in the two dataframes of pairs_df and train_df by plugging to isna function and the sum function (to calculate how many of them is NaN) to each dataframe.

# In[ ]:


pairs_df.isna().sum()


# In[ ]:


train_df.isna().sum()


# As for the pairs_df dataframe, we see that the data entities of: 
# * address_1 (103,524) 
# * city_1 (65,979)
# * state_1 (126,591)
# * zip_1 (219,398)
# * country_1 (8)
# * url_1 (347,101)
# * phone_1 (308,885)
# * categories_1 (16,294)
# * address_2 (266,410)
# * city_2 (211,417)
# * state_2 (269,218)
# * zip_2 (354,080)
# * country_2 (6)
# * url_2 (494,057)
# * phone_2 (459,944)
# * categories_2 (75,976)
# 
# contained some NaN values.

# And on the other hand, the train_df dataframe data entities of:
# * name (1)
# * address (396,621)
# * city (299,189)
# * state (420,586)
# * zip (595,426)
# * country (11)
# * url (871,088)
# * phone (795,957)
# * categories (98,307)
# 
# also contained some NaN values. Now, let's move on to analyzing the number of observations in both dataframes!

# To find the number of observations in our two dataframes, we installed the shape attribute, containing the number 0 enclosed with square brackets to each of our pairs_df and train_df dataframes and print them.

# In[ ]:


print("No. of observations in pairs_df:", pairs_df.shape[0])


# In[ ]:


print("No. of observations in train_df:", train_df.shape[0])


# As you can see, the number of observations in our pairs_df dataframe is 578,907 entities and the number of observations in our train_df dataframe is 1,138,812 entities. After that, let's use EDA in our Foursquare competition data!

# ## EDA
# Now for our EDA analysis, let's analyze our dataframe "one-at-a-time" in a chapter so that we can explain it in a clear, clear way. So, let's move on and don't dawdle!

# ### Chapter 1: pairs_df
# To see the most countries in the first pair, we define a sub-dataframe variable called country_1 to our pairs_df with the country_1 data attribute along with the to_frame function (to convert the given series of an object to a dataframe, which is the country_1 data in pairs_df) along with the reset_index function (to reset our data indexes) and the rename function, containing the columns parameter that was set to the dictionary with two keys: index and country_1. 
# 
# Now, for plotting the countries, it's that simple, if you are using Seaborn. All we need to do is to call the sns module with the displot function, containing the country_1 dataframe with the head function in which it contained the number 20 to indicate the first 20 rows of this dataframe, the x parameter set to country_1, and the y parameter set to count.

# In[ ]:


country_1 = pairs_df.country_1.value_counts().to_frame().reset_index().rename(columns={'index': 'country_1', 'country_1': 'count'})

sns.catplot(data=country_1.head(20), x='country_1', y='count', kind='bar', aspect=2)


# When we observed this graph, we see that the most counts of records in country_1 is the United States of America whil the least counts of records in country_2 is Italy. Now, let's move to analyzing the country_2 data index with Matplotlib! 

# Next, let's plot down the data entities of country_2! We continue with Seaborn again, vaguely following of what we did in analyzing country_1 data entities, but we define country_2 variable with displaying out the first 20 rows of this data entity over country_2, thus setting the x parameter in the sns module with the catplot function to country_2.

# In[ ]:


country_2 = pairs_df.country_2.value_counts().to_frame().reset_index().rename(columns={'index': 'country_2', 'country_2': 'count'})

sns.catplot(data=country_2.head(20), x='country_2', y='count', kind='bar', aspect=2)


# Now, when we looked at the two plots, we see that both country_1 and country_2 data entities were almost alike. Now, let's find over the categories!
# 
# For finding over the categories, we use Plotly for sure. But first, we vaguely follow of what we do for the counting the countries in the pairs_df dataframe, but with counting the categories_1 with the definition of the categories_1 variable. Then, we define a variable, fig, to the px module with the bar function, containing the categories_1 variable with the first 20 rows by using the head function, x parameter set to categories_1, y parameter set to count. Finally, we show the fig figure variable by using the show variable on the fig variable.

# In[ ]:


categories_1 = pairs_df.categories_1.value_counts().to_frame().reset_index().rename(columns = {'index':'categories_1', 'categories_1':'count'})

fig = px.bar(categories_1.head(20), x='categories_1', y='count')
fig.show()


# Now let's do the same to finding categories_2!

# In[ ]:


categories_2 = pairs_df.categories_2.value_counts().to_frame().reset_index().rename(columns = {'index':'categories_2', 'categories_2':'count'})

fig = px.bar(categories_2.head(20), x='categories_2', y='count')
fig.show()


# As you can see, in the categories_1 graph, the most values is the shopping malls, with 11,606 entities, while the least values is parks, with 3,513 entities.
# 
# Meanwhile, in the categories_2 graph, the most values is Residential Buildings (Apartments/Condos) with counts up to 11,604 data entities and the least values is Pharmacies, with counts up to 3,837 data entities.

# <h1 style="text-align: center; background-color: yellow; color: black;"><b>WORK IN PROGRESS</b></h1>
