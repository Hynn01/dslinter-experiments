#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Selecting specific values of a `pandas` `DataFrame` or `Series` to work on is an implicit step in almost any data operation you'll run. Hence a solid understanding of how to slice and dice a dataset is vital. You'll see it with the familiar wine review dataset.

# In[ ]:


import pandas as pd
reviews = pd.read_csv("../input/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)


# ## Easy Ways to Access Series from A DataFrame
# 
# You're probably familiar with a general Python pattern of accessing an object attribute: If a `book` object has a `title` property, you can access it by calling `book.title`. Columns in a `DataFrame` work the same way. 
# 
# So you can access the `country` property of our `reviews` with

# In[ ]:


reviews.country


# Similarly, you are likely familiar with the "bracket" access to dictionary objects. You can access DataFrame columns in a similar way with the indexing (`[]`) operator. This "just works":

# In[ ]:


reviews['country']


# Doesn't a `pandas` `Series` look kind of like a fancy `dict`? It pretty much is, so it's no surprise that, to drill down to a single specific value, we need only use the indexing operator `[]` once more:

# In[ ]:


reviews['country'][0]


# ## Index-based selection
# 
# The indexing operator and attribute selection are nice because they look like things you've seen before. But `pandas` also has accessor operators, `loc` and `iloc` that are useful for more complex operations.
# 
# `pandas` indexing works in one of two paradigms. The first is **index-based selection**: selecting data based on its numerical position in the data. You'll use `iloc` for this.
# 
# Here is how to select the first row of data:

# In[ ]:


reviews.iloc[0]


# Both `loc` and `iloc` are row-first, column-second. So, you get a column with `iloc` with the following:

# In[ ]:


reviews.iloc[:, 0]


# On its own the `:` operator, which also comes from native Python, means "everything". When combined with other selectors, however, it can be used to indicate a range of values. For example, to select the `country` column from just the first, second, and third row, we would do:

# In[ ]:


reviews.iloc[:3, 0]


# Or, to select just the second and third entries, we would do:

# In[ ]:


reviews.iloc[1:3, 0]


# It's also possible to pass a list:

# In[ ]:


reviews.iloc[[0, 1, 2], 0]


# Finally, it's worth knowing that negative numbers can be used in selection. This will start counting forwards from the _end_ of the values. So for example here are the last five elements of the dataset.

# In[ ]:


reviews.iloc[-5:]


# ## Label-based selection
# 
# You can use the `loc` operator for **label-based selection**. This uses the data index value, not its position.
# 
# For example, to get the first entry in `reviews`, we would now do the following:

# In[ ]:


reviews.loc[0, 'country']


# With `iloc` we treat the dataset like a big matrix (a list of lists), one that we have to index into by position. 
# 
# Since your dataset usually has meaningful indices, it's frequently easier to use `loc`. For example, here's one operation that's much easier using `loc`:

# In[ ]:


reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]


# ## Manipulating the index
# 
# Label-based selection derives its power from the labels in the index. What if you want to select on a different field than the current index?
# 
# The `set_index` method will help you here. For example, you can make `title` the index field with the command:

# In[ ]:


reviews.set_index("title")


# Performing a `set_index` is useful if you can come up with an index for the dataset which is better than the current one.

# ## Conditional selection
# 
# So far you've indexed data using locations or values. But you'll often want to ask questions based on conditions. 
# 
# For example, suppose that you're interested specifically in better-than-average wines produced in Italy.
# 
# We can start by asking each wine if it's Italian or not:

# In[ ]:


reviews.country == 'Italy'


# This produced a `Series` of `True`/`False` booleans based on the `country` of each record.  This result can then be used inside of `loc` to select the relevant data:

# In[ ]:


reviews.loc[reviews.country == 'Italy']


# This `DataFrame` has ~20,000 rows. The original had ~130,000. That means that around 15% of wines originate from Italy.
# 
# We also wanted to know which ones are better than average. Wines are reviewed on a 80-to-100 point scale, so this could mean wines that accrued at least 90 points.
# 
# We can use the ampersand (`&`) to bring the two questions together:

# In[ ]:


reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]


# Suppose we'll buy any wine that's made in Italy _or_ which is rated above average. For this we use a pipe (`|`):

# In[ ]:


reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)]


# `pandas` comes with a few pre-built conditional selectors, two of which we will highlight here. The first is `isin`. `isin` is lets you select data whose value "is in" a list of values. For example, here's how we can use it to select wines only from Italy or France:

# In[ ]:


reviews.loc[reviews.country.isin(['Italy', 'France'])]


# The second is `isnull` (and its companion `notnull`). These methods let you highlight values which are or are not empty (`NaN`). For example, to filter out wines lacking a price tag in the dataset, here's what we would do:

# In[ ]:


reviews.loc[reviews.price.notnull()]


# ## Assigning data
# 
# Going the other way, assigning data to a `DataFrame` is easy. You can assign either a constant value:

# In[ ]:


reviews['critic'] = 'everyone'
reviews['critic']


# Or with an iterable of values:

# In[ ]:


reviews['index_backwards'] = range(len(reviews), 0, -1)
reviews['index_backwards']


# We will see much more `DataFrame` assignment going on in later sections of this tutorial.
