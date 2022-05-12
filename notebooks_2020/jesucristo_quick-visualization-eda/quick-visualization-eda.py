#!/usr/bin/env python
# coding: utf-8

# # <span style="font-family:Papyrus; font-size:2em;">Recursion Cellular Image Classification</span>
# # <span style="font-family:Papyrus; font-size:1em;">CellSignal: Disentangling biological signal from experimental noise in cellular images</span>
# 
# ![](https://assets.website-files.com/5cb63fe47eb5472014c3dae6/5d040176f0a2fd66df939c51_figure1%400.75x.png)

# <br>
# ## [Competition Resources](https://www.kaggle.com/c/recursion-cellular-image-classification/overview/resources)
# ## [RXRX](https://www.rxrx.ai/)
# ## [Tutorials](https://github.com/recursionpharma/rxrx1-utils)

# <br>
# # How to visualize images in RxRx1
# 
# The RxRx1 cellular image dataset is made up of 6-channel images, where each channel illuminates different parts of the cell (visit [RxRx.ai](https://www.rxrx.ai/) for details). This notebook demonstrates how to use the code in [rxrx1-utils](https://github.com/recursionpharma/rxrx1-utils) to load and visualize the RxRx1 images.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import sys
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system('git clone https://github.com/recursionpharma/rxrx1-utils')
print ('rxrx1-utils cloned!')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


sys.path.append('rxrx1-utils')
import rxrx.io as rio


# # Loading a site and visualizing individual channels
# 
# Use load_site to get the 512 x 512 x 6 image tensor for a site. The arguments you pass to load_site tell it which image you want. In the example below, from the train set, we request the image in experiment RPE-05 on plate 3 in well D19 at site 2.

# In[ ]:


t = rio.load_site('train', 'RPE-05', 3, 'D19', 2)
t.shape


# At this point, you can visualize the individual channels.

# In[ ]:


fig, axes = plt.subplots(2, 3, figsize=(24, 16))

for i, ax in enumerate(axes.flatten()):
  ax.axis('off')
  ax.set_title('channel {}'.format(i + 1))
  _ = ax.imshow(t[:, :, i], cmap='gray')


# The function load_site takes an optional base_path argument that defaults to *gs://rxrx1-us-central1/images*, one of the two Google Cloud Storage buckets containing the RxRx1 image set. You can also set base_path to the path of your local copy of the dataset, and this is how you'll typically want to work with this function.

# # Converting a site to RGB format
# 
# In order to visualize all six channels at once, use **convert_tensor_to_rgb**. It associates an RGB color with each channel, then aggregates the color channels across the six cellular channels.
# 
# 

# In[ ]:


x = rio.convert_tensor_to_rgb(t)
x.shape


# Now plot your RGB image.

# In[ ]:


plt.figure(figsize=(8, 8))
plt.axis('off')

_ = plt.imshow(x)


# # Load and convert to RGB
# For convenience, there is a wrapper around these two functions called ```load_site_as_rgb``` with the same signature as ```load_site```.

# In[ ]:


y = rio.load_site_as_rgb('train', 'HUVEC-08', 4, 'K09', 1)

plt.figure(figsize=(8, 8))
plt.axis('off')

_ = plt.imshow(y)


# Beautiful images, aren't they?

# # Combining competition metadata
# The metadata for RxRx1 during the Kaggle competition is broken up into four files: 
# - ```train.csv```
# - ```train_controls.csv```
# - ```test.csv```
# - ```test_controls.csv```. 
# 
# It is often more convenient to view all the metadata at once, so we have provided a helper function called combine_metadata for doing just that.

# In[ ]:


md = rio.combine_metadata()
md.head()


# The combined metadata adds a ```cell_type``` and dataset column, and specifies a well_type of "treament" for all non-control sirna. Note that the sirna column has **NaNs** for the non-control test images since those labels are not available during the competition (they are from the wells that need to be predicted), which forces the sirna column to be of type float.

# # EDA

# ### File Description
# - **[train/test].zip:** the image data. The image paths, such as ```U2OS-01/Plate1/B02_s2_w3.png```, can be read as:
# 
#     - Cell line and batch number (U2OS batch 1)
#     - Plate number (1)
#     - Well location on plate (column B, row 2)
#     - Site (2)
#     - Microscope channel (3)
# 
# Please note that the **[train/test].csv** and **[train/test]_controls.csv** combined describe the images found in **[train/test].zip.** You will only be making predictions on the images listed in **test.csv**, not on all the images found in **test.zip**.
# 
# - **[train/test].csv**
#     - id_code
#     - experiment: the cell type and batch number
#     - plate: plate number within the experiment
#     - well: location on the plate
#     - sirna: the target
# 
# - **[train/test]_controls.csv** In each experiment, the same 30 siRNAs appear on every plate as positive controls. In addition, there is one well per plate with untreated cells as a negative control. It has the same schema as **[train/test].csv**, plus a ```well_type``` field denoting the type of control.
# 
# **pixel_stats.csv** Provides the mean, standard deviation, median, min, and max pixel values for each channel of each image.
# **sample_submission.csv** A valid sample submission.

# In[ ]:


import seaborn as sns


# In[ ]:


md.head(10)


# In[ ]:


md.index


# ### Unique values

# In[ ]:


for i in md.columns:
    print (">> ",i,"\t", md[i].unique())


# In[ ]:


for col in ['cell_type', 'dataset', 'experiment', 'plate',  'site', 'well_type']:
    print (col)
    print (md[col].value_counts())
    sns.countplot(y = col,
              data = md,
              order = md[col].value_counts().index)
    plt.show()
    


# ### Missing

# In[ ]:


missing_values_count = md.isnull().sum()
missing_values_count


# In[ ]:


md = md.fillna(0)
md.head()


# ### sirna distribution

# In[ ]:


train_df = md[md['dataset'] == 'train']
test_df = md[md['dataset'] == 'test']

train_df.shape, test_df.shape


# In[ ]:


plt.figure(figsize=(16,6))
plt.title("Distribution of SIRNA in the train and test set")
sns.distplot(train_df.sirna,color="green", kde=True,bins='auto', label='train')
sns.distplot(test_df.sirna,color="blue", kde=True, bins='auto', label='test')
plt.legend()
plt.show()


# Remember, 0s were NaNs

# In[ ]:


feat1 = 'sirna'
fig = plt.subplots(figsize=(15, 5))

# train
plt.subplot(1, 2, 1)
sns.kdeplot(train_df[feat1][train_df['site'] == 1], shade=False, color="b", label = 'site 1')
sns.kdeplot(train_df[feat1][train_df['site'] == 2], shade=False, color="r", label = 'site 2')
plt.title(feat1)
plt.xlabel('Feature Values')
plt.ylabel('Probability')

# test
plt.subplot(1, 2, 2)
sns.kdeplot(test_df[feat1][test_df['site'] == 1], shade=False, color="b", label = 'site 1')
sns.kdeplot(test_df[feat1][test_df['site'] == 2], shade=False, color="r", label = 'site 2')
plt.title(feat1)
plt.xlabel('Feature Values')
plt.ylabel('Probability')
plt.show()


# In[ ]:





# In[ ]:


# Prevent: Output path '/rxrx1-utils/.git/logs/refs/remotes/origin/HEAD' contains too many nested subdirectories (max 6)
get_ipython().system('rm -r  rxrx1-utils')
get_ipython().system('ls')


# # To Be Continued ...
