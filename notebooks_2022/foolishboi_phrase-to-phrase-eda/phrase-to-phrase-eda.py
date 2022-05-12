#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.style as style
style.use('fivethirtyeight')
from matplotlib.ticker import FuncFormatter
from nltk.corpus import stopwords
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import os


# In[ ]:


train = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/train.csv")


# In[ ]:


fig = plt.figure(figsize=(12,4))
ax1 = train.groupby('anchor')['score'].mean().sort_values()[:15].plot(kind="barh")
ax1.set_title("Average score w.r.t Anchor Type", fontsize=14, fontweight = 'bold')
ax1.set_xlabel("Average score", fontsize = 10)
ax1.set_ylabel("")


# In[ ]:


fig = plt.figure(figsize=(16,4))
ax2 = train.groupby('context')['score'].mean().sort_values()[:15].plot(kind="barh")
ax2.set_title("Average score w.r.t Context", fontsize=14, fontweight = 'bold')
ax2.set_xlabel("Average score", fontsize = 10)
ax2.set_ylabel("")


# In[ ]:


fig = plt.figure(figsize=(12,8))
av_per_essay = train['anchor'].value_counts(ascending = True).rename_axis('anchor_num').reset_index(name='count')
av_per_essay['perc'] = round((av_per_essay['count'] / train.id.nunique()),3)
av_per_essay = av_per_essay.set_index('anchor_num')
ax = av_per_essay['perc'].sort_values(ascending = False)[:25].plot(kind="barh")
ax.set_title("Anchor Type: Percent present", fontsize=20, fontweight = 'bold')
ax.bar_label(ax.containers[0], label_type="edge")
ax.set_xlabel("Percent")
ax.set_ylabel("")
plt.show()


# In[ ]:


fig = plt.figure(figsize=(12,8))
av_per_essay = train['context'].value_counts(ascending = True).rename_axis('context_num').reset_index(name='count')
av_per_essay['perc'] = round((av_per_essay['count'] / train.id.nunique()),3)
av_per_essay = av_per_essay.set_index('context_num')
ax = av_per_essay['perc'].sort_values(ascending = False)[:30].plot(kind="barh")
ax.set_title("Context Type: Percent present", fontsize=20, fontweight = 'bold')
ax.bar_label(ax.containers[0], label_type="edge")
ax.set_xlabel("Percent")
ax.set_ylabel("")
plt.show()


# In[ ]:


fig = plt.figure(figsize=(12,2))
av_per_essay = train['score'].value_counts(ascending = True).rename_axis('score_num').reset_index(name='count')
av_per_essay['perc'] = round((av_per_essay['count'] / train.id.nunique()),3)
av_per_essay = av_per_essay.set_index('score_num')
ax = av_per_essay['perc'].sort_values(ascending = False).plot(kind="barh")
ax.set_title("Score: Percent distribution", fontsize=20, fontweight = 'bold')
ax.bar_label(ax.containers[0], label_type="edge")
ax.set_xlabel("Percent")
ax.set_ylabel("")
plt.show()


# In[ ]:




