#!/usr/bin/env python
# coding: utf-8

# # Hello
# 
# I put some popular word embeddings into a single Kaggle dataset in unified gensim format. Models are binarized, so they are quick to load (data is stored in numpy arrays).
# 
# Why:
# * Unified format - single handling function for different embeddings
# * gensim mdoels are nice - a lot of helpful methods, take `most_similar` for example
# * Fast loading!

# In[ ]:


from gensim.models import KeyedVectors

# As of Gensim 3.7.3 it's using some deprecated function and we don't care about it
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


get_ipython().run_line_magic('timeit', '-n 30 model = KeyedVectors.load("../input/glove.twitter.27B.200d.gensim")')


# Glove in 4 seconds! But if you think that's impressive, hold my beer:

# In[ ]:


get_ipython().run_line_magic('timeit', '-n 30 model = KeyedVectors.load("../input/glove.twitter.27B.200d.gensim", mmap="r")')


# This is tiny `mmap="r"` tells gensim/numpy to read bytes from the [disk directly to memory](https://en.wikipedia.org/wiki/Mmap). It just can't get faster than that.
# 
# And now we can play around with vectors:

# In[ ]:


model.most_similar("good")


# In[ ]:


model.most_similar(positive=["woman", "king"], negative=["man"])


# In[ ]:


model["good"]


# ... and that's it. Now you do your NLP from here. Good luck!
