#!/usr/bin/env python
# coding: utf-8

# **This notebook is an exercise in the [Pandas](https://www.kaggle.com/learn/pandas) course.  You can reference the tutorial at [this link](https://www.kaggle.com/residentmario/summary-functions-and-maps).**
# 
# ---
# 

# # Introduction
# 
# Now you are ready to get a deeper understanding of your data.
# 
# Run the following cell to load your data and some utility functions (including code to check your answers).

# In[ ]:


import pandas as pd
pd.set_option("display.max_rows", 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.summary_functions_and_maps import *
print("Setup complete.")

reviews.head()


# # Exercises

# ## 1.
# 
# What is the median of the `points` column in the `reviews` DataFrame?

# In[ ]:


median_points = reviews.points.median()

# Check your answer
q1.check()
median_points


# In[ ]:


#q1.hint()
#q1.solution()


# ## 2. 
# What countries are represented in the dataset? (Your answer should not include any duplicates.)

# In[ ]:


countries = reviews.country.unique()


# Check your answer
q2.check()
countries


# In[ ]:


#q2.hint()
#q2.solution()


# ## 3.
# How often does each country appear in the dataset? Create a Series `reviews_per_country` mapping countries to the count of reviews of wines from that country.

# In[ ]:


reviews_per_country = reviews.country.value_counts()

# Check your answer
q3.check()
reviews_per_country


# In[ ]:


#q3.hint()
#q3.solution()


# ## 4.
# Create variable `centered_price` containing a version of the `price` column with the mean price subtracted.
# 
# (Note: this 'centering' transformation is a common preprocessing step before applying various machine learning algorithms.) 

# In[ ]:


centered_price = reviews.price - reviews.price.mean()

# Check your answer
q4.check()
centered_price


# In[ ]:


#q4.hint()
#q4.solution()


# ## 5.
# I'm an economical wine buyer. Which wine is the "best bargain"? Create a variable `bargain_wine` with the title of the wine with the highest points-to-price ratio in the dataset.

# In[ ]:


bargain_index = (reviews.points / reviews.price).idxmax()
bargain_wine = reviews.loc[bargain_index, 'title']

# Check your answer
q5.check()
bargain_wine


# In[ ]:


#q5.hint()
#q5.solution()


# ## 6.
# There are only so many words you can use when describing a bottle of wine. Is a wine more likely to be "tropical" or "fruity"? Create a Series `descriptor_counts` counting how many times each of these two words appears in the `description` column in the dataset. (For simplicity, let's ignore the capitalized versions of these words.)

# In[ ]:


tropical=reviews.description.map(lambda d: "tropical" in d).sum()
fruity=reviews.description.map(lambda d: "fruity" in d).sum()
descriptor_counts = pd.Series([tropical, fruity], index=['tropical', 'fruity'])


# Check your answer
q6.check()
descriptor_counts


# In[ ]:


#q6.hint()
#q6.solution()


# ## 7.
# We'd like to host these wine reviews on our website, but a rating system ranging from 80 to 100 points is too hard to understand - we'd like to translate them into simple star ratings. A score of 95 or higher counts as 3 stars, a score of at least 85 but less than 95 is 2 stars. Any other score is 1 star.
# 
# Also, the Canadian Vintners Association bought a lot of ads on the site, so any wines from Canada should automatically get 3 stars, regardless of points.
# 
# Create a series `star_ratings` with the number of stars corresponding to each review in the dataset.

# In[ ]:


star_ratings = reviews.points.map(lambda p: 3 if (p >= 95) else 2 if (p>= 85) else 1)

# Check your answer
q7.check()


# In[ ]:


#q7.hint()
#q7.solution()


# # Keep going
# Continue to **[grouping and sorting](https://www.kaggle.com/residentmario/grouping-and-sorting)**.

# ---
# 
# 
# 
# 
# *Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/pandas/discussion) to chat with other learners.*
