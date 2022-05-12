#!/usr/bin/env python
# coding: utf-8

# *The data, concept, and initial implementation of this notebook was done in Colab by Ross Wightman, the creator of timm. I (Jeremy Howard) did some refactoring, curating, and expanding of the analysis, and added prose.*

# ## timm
# 
# [PyTorch Image Models](https://timm.fast.ai/) (timm) is a wonderful library by Ross Wightman which provides state-of-the-art pre-trained computer vision models. It's like Huggingface Transformers, but for computer vision instead of NLP (and it's not restricted to transformers-based models)!
# 
# Ross has been kind enough to help me understand how to best take advantage of this library by identifying the top models. I'm going to share here so of what I've learned from him, plus some additional ideas.

# ## The data
# 
# Ross regularly benchmarks new models as they are added to timm, and puts the results in a CSV in the project's GitHub repo. To analyse the data, we'll first clone the repo:

# In[ ]:


get_ipython().system(' git clone --depth 1 https://github.com/rwightman/pytorch-image-models.git')
get_ipython().run_line_magic('cd', 'pytorch-image-models/results')


# Using Pandas, we can read the two CSV files we need, and merge them together.
# 
# We'll also add a "family" column that will allow us to group architectures into categories with similar characteristics:

# In[ ]:


import pandas as pd

df = pd.read_csv('benchmark-infer-amp-nchw-pt111-cu113-rtx3090.csv').merge(
     pd.read_csv('results-imagenet.csv'), on='model')
df['secs'] = 1. / df['infer_samples_per_sec']
df['family'] = df.model.str.extract('^([a-z]+?(?:v2)?)(?:\d|_|$)')


# Ross has told me which models he's found the most usable in practice, so I'll limit the charts to just look at these.

# In[ ]:


df2 = df[df.family.str.match('re[sg]net|beit|convnext|levit|efficient|vit')]


# ## The results
# 
# Here's the results. In this chart, the x axis shows how many seconds it takes to process one image (**note**: it's a log scale), and the y axis is the accuracy on Imagenet.
# 
# The size of each bubble is proportional to the size of images used in testing.
# 
# The color shows what "family" the architecture is from.
# 
# Just hover your mouse over a marker to see details about the model.
# 
# **Note**: on my screen, Kaggle cuts off the family selector and some plotly functionality -- to see the whole thing, collapse the table of contents on the right by clicking the little arrow to the right of "*Contents*".

# In[ ]:


import plotly.express as px
w,h = 1000,800

px.scatter(df2, width=w, height=h, size=df2.infer_img_size**2,
    x='secs',  y='top1', log_x=True, color='family',
    hover_name='model', hover_data=['infer_samples_per_sec', 'infer_img_size']
)


# From this, we can see that the *levit* family models are extremely fast for image recognition, and clearly the most accurate amongst the faster models. That's not surprising, since these models are a hybrid of the best ideas from CNNs and transformers, so get the benefit of each. In fact, we see a similar thing even in the middle category of speeds -- the best is the ConvNeXt, which is a pure CNN, but which takes advantage of ideas from the transformers literature.
# 
# For the slowest models, *beit* is the most accurate -- although we need to be a bit careful of interpreting this, since it's trained on a larger dataset (ImageNet-21k, which is also used for *vit* models).
# 
# I'll add one other plot, which is of speed vs parameter count. Often, parameter count is used in papers as a proxy for speed. However, as we see, there is a wide variation in speeds at each level of parameter count, so it's really not a useful proxy.
# 
# (Parameter count may be be useful for identifying how much memory a model needs, but even for that it's not always a great proxy.)

# In[ ]:


px.scatter(df2, width=w, height=h,
    x='param_count_x',  y='secs', log_x=True, log_y=True, color='infer_img_size',
    hover_name='model', hover_data=['infer_samples_per_sec', 'family']
)


# Finally, we should remember that speed depends on hardware. If you're using something other than a modern NVIDIA GPU, your results may be different. In particular, I suspect that transformers-based models might have worse performance in general on CPUs (although I need to study this more to be sure).

# In[ ]:




