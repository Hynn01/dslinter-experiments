#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# Short notebook on how to create word embeddings using FastText module

# In[ ]:


path = "../input/yelp-dataset/yelp_academic_dataset_tip.json"
df = pd.read_json(path,lines = True)
df.head()


# In[ ]:


df_sample = df[:10000]


# In[ ]:


df_sample.shape


# In[ ]:


import gensim


# In[ ]:


# Pre-defined preprocessing function from gensim which will take care of basic data cleaning 
tip_cleaned = df_sample.text.apply(gensim.utils.simple_preprocess)
tip_cleaned


# In[ ]:


from gensim.models import FastText,Word2Vec


# In[ ]:


embedding_size = 300
window_size = 5
min_word = 5
down_sampling = 1e-2 #words with higher frequencies will be downsampled to avoid huge training corpus

fast_text_model = FastText(tip_cleaned,
                          vector_size = embedding_size,
                          window = window_size,
                          min_count = min_word,
                          sample = down_sampling,
                          workers = 4,
                          sg = 1, #skip gram - predicts context from target
                          epochs = 100)


# In[ ]:


fast_text_model.save("ft_model_yelp_tip")

fast_text_yelp = Word2Vec.load("ft_model_yelp_tip")


# In[ ]:


fast_text_yelp.wv['chicken']


# In[ ]:


fast_text_yelp.wv.similarity('beer','drink')

