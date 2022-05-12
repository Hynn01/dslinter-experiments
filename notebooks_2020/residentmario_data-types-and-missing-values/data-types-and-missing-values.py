#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this tutorial, you'll learn how to investigate data types within a DataFrame or Series.  You'll also learn how to find and replace entries.
# 
# **To start the exercise for this topic, please click [here](https://www.kaggle.com/kernels/fork/598826).**
# 
# # Dtypes
# 
# The data type for a column in a DataFrame or a Series is known as the **dtype**.
# 
# You can use the `dtype` property to grab the type of a specific column.  For instance, we can get the dtype of the `price` column in the `reviews` DataFrame:

# In[ ]:



import pandas as pd
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option('max_rows', 5)


# In[ ]:


reviews.price.dtype


# Alternatively, the `dtypes` property returns the `dtype` of _every_ column in the DataFrame:

# In[ ]:


reviews.dtypes


# Data types tell us something about how pandas is storing the data internally. `float64` means that it's using a 64-bit floating point number; `int64` means a similarly sized integer instead, and so on.
# 
# One peculiarity to keep in mind (and on display very clearly here) is that columns consisting entirely of strings do not get their own type; they are instead given the `object` type.
# 
# It's possible to convert a column of one type into another wherever such a conversion makes sense by using the `astype()` function. For example, we may transform the `points` column from its existing `int64` data type into a `float64` data type:

# In[ ]:


reviews.points.astype('float64')


# A DataFrame or Series index has its own `dtype`, too:

# In[ ]:


reviews.index.dtype


# Pandas also supports more exotic data types, such as categorical data and timeseries data. Because these data types are more rarely used, we will omit them until a much later section of this tutorial.

# # Missing data
# 
# Entries missing values are given the value `NaN`, short for "Not a Number". For technical reasons these `NaN` values are always of the `float64` dtype.
# 
# Pandas provides some methods specific to missing data. To select `NaN` entries you can use `pd.isnull()` (or its companion `pd.notnull()`). This is meant to be used thusly:

# In[ ]:


reviews[pd.isnull(reviews.country)]


# Replacing missing values is a common operation.  Pandas provides a really handy method for this problem: `fillna()`. `fillna()` provides a few different strategies for mitigating such data. For example, we can simply replace each `NaN` with an `"Unknown"`:

# In[ ]:


reviews.region_2.fillna("Unknown")


# Or we could fill each missing value with the first non-null value that appears sometime after the given record in the database. This is known as the backfill strategy.

# Alternatively, we may have a non-null value that we would like to replace. For example, suppose that since this dataset was published, reviewer Kerin O'Keefe has changed her Twitter handle from `@kerinokeefe` to `@kerino`. One way to reflect this in the dataset is using the `replace()` method:

# In[ ]:


reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino")


# The `replace()` method is worth mentioning here because it's handy for replacing missing data which is given some kind of sentinel value in the dataset: things like `"Unknown"`, `"Undisclosed"`, `"Invalid"`, and so on.
# 
# # Your turn
# 
# If you haven't started the exercise, you can **[get started here](https://www.kaggle.com/kernels/fork/598826)**.

# ---
# 
# 
# 
# 
# *Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/pandas/discussion) to chat with other learners.*
