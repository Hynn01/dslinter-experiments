#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In the last tutorial, we learned how to select relevant data out of a DataFrame or Series. Plucking the right data out of our data representation is critical to getting work done, as we demonstrated in the exercises.
# 
# However, the data does not always come out of memory in the format we want it in right out of the bat. Sometimes we have to do some more work ourselves to reformat it for the task at hand.  This tutorial will cover different operations we can apply to our data to get the input "just right". 
# 
# **To start the exercise for this topic, please click [here](https://www.kaggle.com/kernels/fork/595524).**
# 
# We'll use the Wine Magazine data for demonstration.

# In[ ]:



import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)


# In[ ]:


reviews


# # Summary functions
# 
# Pandas provides many simple "summary functions" (not an official name) which restructure the data in some useful way. For example, consider the `describe()` method:

# In[ ]:


reviews.points.describe()


# This method generates a high-level summary of the attributes of the given column. It is type-aware, meaning that its output changes based on the data type of the input. The output above only makes sense for numerical data; for string data here's what we get:

# In[ ]:


reviews.taster_name.describe()


# If you want to get some particular simple summary statistic about a column in a DataFrame or a Series, there is usually a helpful pandas function that makes it happen. 
# 
# For example, to see the mean of the points allotted (e.g. how well an averagely rated wine does), we can use the `mean()` function:

# In[ ]:


reviews.points.mean()


# To see a list of unique values we can use the `unique()` function:

# In[ ]:


reviews.taster_name.unique()


# To see a list of unique values _and_ how often they occur in the dataset, we can use the `value_counts()` method:

# In[ ]:


reviews.taster_name.value_counts()


# # Maps
# 
# A **map** is a term, borrowed from mathematics, for a function that takes one set of values and "maps" them to another set of values. In data science we often have a need for creating new representations from existing data, or for transforming data from the format it is in now to the format that we want it to be in later. Maps are what handle this work, making them extremely important for getting your work done!
# 
# There are two mapping methods that you will use often. 
# 
# [`map()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.map.html) is the first, and slightly simpler one. For example, suppose that we wanted to remean the scores the wines received to 0. We can do this as follows:

# In[ ]:


review_points_mean = reviews.points.mean()
reviews.points.map(lambda p: p - review_points_mean)


# The function you pass to `map()` should expect a single value from the Series (a point value, in the above example), and return a transformed version of that value. `map()` returns a new Series where all the values have been transformed by your function.
# 
# [`apply()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.apply.html) is the equivalent method if we want to transform a whole DataFrame by calling a custom method on each row.

# In[ ]:


def remean_points(row):
    row.points = row.points - review_points_mean
    return row

reviews.apply(remean_points, axis='columns')


# If we had called `reviews.apply()` with `axis='index'`, then instead of passing a function to transform each row, we would need to give a function to transform each *column*.
# 
# Note that `map()` and `apply()` return new, transformed Series and DataFrames, respectively. They don't modify the original data they're called on. If we look at the first row of `reviews`, we can see that it still has its original `points` value.

# In[ ]:


reviews.head(1)


# Pandas provides many common mapping operations as built-ins. For example, here's a faster way of remeaning our points column:

# In[ ]:


review_points_mean = reviews.points.mean()
reviews.points - review_points_mean


# In this code we are performing an operation between a lot of values on the left-hand side (everything in the Series) and a single value on the right-hand side (the mean value). Pandas looks at this expression and figures out that we must mean to subtract that mean value from every value in the dataset.
# 
# Pandas will also understand what to do if we perform these operations between Series of equal length. For example, an easy way of combining country and region information in the dataset would be to do the following:

# In[ ]:


reviews.country + " - " + reviews.region_1


# These operators are faster than `map()` or `apply()` because they use speed ups built into pandas. All of the standard Python operators (`>`, `<`, `==`, and so on) work in this manner.
# 
# However, they are not as flexible as `map()` or `apply()`, which can do more advanced things, like applying conditional logic, which cannot be done with addition and subtraction alone.
# 
# # Your turn
# 
# If you haven't started the exercise, you can **[get started here](https://www.kaggle.com/kernels/fork/595524)**.

# ---
# 
# 
# 
# 
# *Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/pandas/discussion) to chat with other learners.*
