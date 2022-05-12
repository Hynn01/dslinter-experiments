#!/usr/bin/env python
# coding: utf-8

# This competition's metrics is [pearson correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).
# 
# According to [wiki](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient), it is defined as a measure of linear correlation between two sets of data.
# 
# Let's take a look at the following scanerios and understand what it means.

# In[ ]:


import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


# ### Case 1: when actual and pred are the same
# 
# Pearson correlation is undefined in this case, which is expected.

# In[ ]:


actual = [1, 1, 1, 1, 1]
pred = [1, 1, 1, 1, 1]

print(f"pearson: {pearsonr(actual, pred)}")
print(f"mse: {mean_squared_error(y_true=actual, y_pred=pred)}")


# ### Case 2: Let's add 100 to each element in the pred vector
# 
# In these two cases, the pearson correlation coefficient is actually the same when MSE is obviously different. From this observation, I don't think MSE is a good proxy loss to pearson correlation. 
# 
# You can try a diffferent value other than 100. The pearson correlation would stay unchanged.

# In[ ]:


actual = [0, 0.24, 0.25, 0.5, 0]
pred = [0, 0.5, 0.25, 1, 0]

print(f"pearson: {pearsonr(actual, pred)}")
print(f"mse: {mean_squared_error(y_true=actual, y_pred=pred)}")


# In[ ]:


actual = [0, 0.24, 0.25, 0.5, 0]
pred = [item + 100 for item in [0, 0.5, 0.25, 1, 0]]

print(f"pearson: {pearsonr(actual, pred)}")
print(f"mse: {mean_squared_error(y_true=actual, y_pred=pred)}")


# ### Case 3: Let's simulate what we normally do in Kaggle competitions when we only have tiny improvement
# 
# Assume we improve the first element of pred vector from 0.24 to 0.25. Luckily, pearson correlation shows improvement too. 

# In[ ]:


actual = [0.5, 0.25, 1, 0, 0.5]
pred = [0.24, 0.5, 1.25, 0, 0.75]

print(f"pearson: {pearsonr(actual, pred)}")
print(f"mse: {mean_squared_error(y_true=actual, y_pred=pred)}")


# In[ ]:


actual = [0.5, 0.25, 1, 0, 0.5]
pred = [0.25, 0.5, 1.25, 0, 0.75]

print(f"pearson: {pearsonr(actual, pred)}")
print(f"mse: {mean_squared_error(y_true=actual, y_pred=pred)}")


# ### Concluding Remark
# 1. Pearson correlation is a difficult metric to optimize.
# 2. Once you find improvement, the metric will reflect accordingly.
# 3. Having a good pearson correlation doesn't mean your learning performance is good in terms of the typical MSE (case 2).
# 
# Your feedback is welcomed.
