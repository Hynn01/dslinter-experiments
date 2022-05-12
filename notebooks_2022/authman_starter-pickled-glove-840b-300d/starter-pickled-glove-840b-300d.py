#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Loading Glove from .txt files typically takes +3min on Kernels:

# In[1]:


# GLOVE_EMBEDDING_PATH = '../input/glove840b300dtxt/glove.840B.300d.txt'

# def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

# def load_embeddings(embed_dir):
#     embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in tqdm(open(embed_dir)))
#     return embedding_index

# glove = load_embeddings(GLOVE_EMBEDDING_PATH)

# TOO SLOW!
# 2196018it [03:19, 11010.98it/s]
# Now, you can load from the pickled file directly. It loads the entire embedding, so if you only wish to use a subset, be sure to `del glove` and `gc.collect()` once you're done with it:

# In[6]:


import pickle
from time import time

t = time()
with open('../input/glove.840B.300d.pkl', 'rb') as fp:
    glove = pickle.load(fp)
print(time()-t)


# In[7]:


len(glove)


# In[8]:


list(glove.keys())[0]


# In[9]:


glove[',']


# Happy Kaggling

# In[ ]:




