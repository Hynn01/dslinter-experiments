#!/usr/bin/env python
# coding: utf-8

# # Amazon Bin Image Dataset EDA

# ## Import required packages

# In[ ]:


get_ipython().system('pip3 install awscli')


# In[ ]:


import pandas as pd
import matplotlib


# ## Read data file list

# In[ ]:


df = pd.read_csv("../input/amazon-bin-image-dataset-file-list/quantity.csv")
df


# ## Load image

# In[ ]:


df['location'][46]


# In[ ]:


get_ipython().system('aws s3 cp --no-sign-request s3://aft-vbi-pds/bin-images/00024.jpg .')


# In[ ]:


img = matplotlib.image.imread('00024.jpg')
matplotlib.pyplot.imshow(img)


# ## Statistics

# In[ ]:


df.describe()


# ## Histogram

# In[ ]:


df.hist(bins=100)


# ## 95% of bins have less than or equal to 12 items

# In[ ]:


df.quantile(.95)

