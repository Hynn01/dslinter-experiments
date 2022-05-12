#!/usr/bin/env python
# coding: utf-8

# ## Simple Strategies for Outliners Detection 🦦

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)


# - **In order to illustrate each strategy, I will use Wine dataset and detect outliners based on 2 attributes: volatile acidity+ pH**

# In[ ]:


data= pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
col= ['volatile acidity','pH']
data.head()


# In[ ]:


fig, ax=plt.subplots(figsize=(15,6))
ax.scatter(x=col[0],y=col[1], data=data, color='darkblue')
ax.set_title('Initial Dataset', fontsize=20);


# ### Quantiles 🐝

# - **We determine outliers by those data points that are outside certain thresholds (for both sides, for ex: 0.1-0.9, 0.05-0.95,...) in all considered columns**

# In[ ]:


# Create function 
def quantile(data,col,x1,x2):
    mask1= (data[col]>data[col].quantile(x1)).all(axis=1)    # remove left outliners
    mask2= (data[col]<data[col].quantile(x2)).all(axis=1)    # remove right outliners
    return (mask1&mask2)


# In[ ]:


mask = quantile(data,col,0.05,0.95)
fig, ax=plt.subplots(figsize=(15,6))
ax.scatter(x=col[0],y=col[1], data=data[mask],color='darkblue')
ax.scatter(x=col[0],y=col[1], data=data[~mask], color='darkred')
ax.set_title('Outliner Detection- Quantile', fontsize=20);


# ### Gaussian Distribution 🐝

# - **We determine outliers by those data points that are in 'low density'**
# 
# - Note: This strategy works best for those dataset that follow normal distribution, we can transform our attributes into normal before applying this strategy. 

# In[ ]:


from scipy.stats import multivariate_normal 

# Create function
def gaussian(data,col,thres):
    mean, cov= np.mean(data[col]), np.cov(data[col].T)
    # data points having extremely low probability density value (<thres) is considered as outliers
    mask= multivariate_normal(mean, cov).pdf(data[col])> thres
    return mask


# In[ ]:


mask = gaussian(data,col,0.1)
fig, ax=plt.subplots(figsize=(15,6))
ax.scatter(x=col[0],y=col[1], data=data[mask],color='darkblue')
ax.scatter(x=col[0],y=col[1], data=data[~mask], color='darkred')
ax.set_title('Outliner Detection- Gaussian Distribution', fontsize=20);


# ### Outliner Detection functions in Scikit-Learn 🐝

# [Outlier and Novelty Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)

# - There are 2 outliner detection methods that are available in Sklearn that perform quite well at detecting outliner points. 
#     - Isolation Forest (IF) [link](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest)
#     - Local Outlier Factor (LOF) [link](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor)

# In[ ]:


from sklearn.ensemble import IsolationForest as ISF
from sklearn.neighbors import LocalOutlierFactor as LOF

# Create functions
## Isolation Forest
def isf(data,col):
    mask = ISF(random_state=0).fit_predict(data[col])==1
    return mask
## Local Outlier Factor
def lof(data,col):
    n= int(data[col].shape[0]*0.1)
    mask = LOF(n_neighbors=n).fit_predict(data[col])==1
    return mask


# In[ ]:


mask = isf(data,col)
fig, ax=plt.subplots(figsize=(15,6))
ax.scatter(x=col[0],y=col[1], data=data[mask],color='darkblue')
ax.scatter(x=col[0],y=col[1], data=data[~mask], color='darkred')
ax.set_title('Outliner Detection- Isolation Forest', fontsize=20);


# In[ ]:


mask = lof(data,col)
fig, ax=plt.subplots(figsize=(15,6))
ax.scatter(x=col[0],y=col[1], data=data[mask],color='darkblue')
ax.scatter(x=col[0],y=col[1], data=data[~mask], color='darkred')
ax.set_title('Outliner Detection- Local Outliner Factor', fontsize=20);


# ### 🙊 THE END !!! 🙊
