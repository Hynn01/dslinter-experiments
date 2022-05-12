#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Maps allow us to transform data in a DataFrame or Series one value at a time for an entire column. However, often we want to group our data, and then do something specific to the group the data is in. 
# 
# As you'll learn, we do this with the `groupby()` operation.  We'll also cover some additional topics, such as more complex ways to index your DataFrames, along with how to sort your data.
# 
# **To start the exercise for this topic, please click [here](https://www.kaggle.com/kernels/fork/598715).**
# 
# # Groupwise analysis
# 
# One function we've been using heavily thus far is the `value_counts()` function. We can replicate what `value_counts()` does by doing the following:

# In[ ]:



import pandas as pd
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)


# In[ ]:


reviews.groupby('points').points.count()


# `groupby()` created a group of reviews which allotted the same point values to the given wines. Then, for each of these groups, we grabbed the `points()` column and counted how many times it appeared.  `value_counts()` is just a shortcut to this `groupby()` operation. 
# 
# We can use any of the summary functions we've used before with this data. For example, to get the cheapest wine in each point value category, we can do the following:

# In[ ]:


reviews.groupby('points').price.min()


# You can think of each group we generate as being a slice of our DataFrame containing only data with values that match. This DataFrame is accessible to us directly using the `apply()` method, and we can then manipulate the data in any way we see fit. For example, here's one way of selecting the name of the first wine reviewed from each winery in the dataset:

# In[ ]:


reviews.groupby('winery').apply(lambda df: df.title.iloc[0])


# For even more fine-grained control, you can also group by more than one column. For an example, here's how we would pick out the best wine by country _and_ province:

# In[ ]:


reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])


# Another `groupby()` method worth mentioning is `agg()`, which lets you run a bunch of different functions on your DataFrame simultaneously. For example, we can generate a simple statistical summary of the dataset as follows:

# In[ ]:


reviews.groupby(['country']).price.agg([len, min, max])


# Effective use of `groupby()` will allow you to do lots of really powerful things with your dataset.

# # Multi-indexes
# 
# In all of the examples we've seen thus far we've been working with DataFrame or Series objects with a single-label index. `groupby()` is slightly different in the fact that, depending on the operation we run, it will sometimes result in what is called a multi-index.
# 
# A multi-index differs from a regular index in that it has multiple levels. For example:

# In[ ]:


countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])
countries_reviewed


# In[ ]:


mi = countries_reviewed.index
type(mi)


# Multi-indices have several methods for dealing with their tiered structure which are absent for single-level indices. They also require two levels of labels to retrieve a value. Dealing with multi-index output is a common "gotcha" for users new to pandas.
# 
# The use cases for a multi-index are detailed alongside instructions on using them in the [MultiIndex / Advanced Selection](https://pandas.pydata.org/pandas-docs/stable/advanced.html) section of the pandas documentation.
# 
# However, in general the multi-index method you will use most often is the one for converting back to a regular index, the `reset_index()` method:

# In[ ]:


countries_reviewed.reset_index()


# # Sorting
# 
# Looking again at `countries_reviewed` we can see that grouping returns data in index order, not in value order. That is to say, when outputting the result of a `groupby`, the order of the rows is dependent on the values in the index, not in the data.
# 
# To get data in the order want it in we can sort it ourselves.  The `sort_values()` method is handy for this.

# In[ ]:


countries_reviewed = countries_reviewed.reset_index()
countries_reviewed.sort_values(by='len')


# `sort_values()` defaults to an ascending sort, where the lowest values go first. However, most of the time we want a descending sort, where the higher numbers go first. That goes thusly:

# In[ ]:


countries_reviewed.sort_values(by='len', ascending=False)


# To sort by index values, use the companion method `sort_index()`. This method has the same arguments and default order:

# In[ ]:


countries_reviewed.sort_index()


# Finally, know that you can sort by more than one column at a time:

# In[ ]:


countries_reviewed.sort_values(by=['country', 'len'])


# # Your turn
# 
# If you haven't started the exercise, you can **[get started here](https://www.kaggle.com/kernels/fork/598715)**.

# ---
# 
# 
# 
# 
# *Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/pandas/discussion) to chat with other learners.*
