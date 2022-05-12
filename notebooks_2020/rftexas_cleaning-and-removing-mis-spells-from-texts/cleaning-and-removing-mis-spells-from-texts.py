#!/usr/bin/env python
# coding: utf-8

# # Cleaning and removing misspells from texts

# As you know, to enhance the performance of a model you need to make sure you feed it with high-quality data.
# 
# **What does 'high-quality' mean? 
# 
# Well, it means that you don't have misspelling errors, that your text is not polluted by artifacts or other errors.
# 
# In this short notebook, I show you two ways of cleaning your data.
# 
# **Feel free to share your thoughts on my work ;)**
# 
# Sources:  
# - https://www.kaggle.com/shonenkov/hack-with-parallel-corpus
# - https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/147417 (Dave Lorenz's message)

# ## Importing dependencies

# In[ ]:


get_ipython().system('pip install -q pandarallel')
get_ipython().system('pip install -q spacy ')
get_ipython().system('pip install -q spacy_cld')
get_ipython().system('pip install -q pyspellchecker')
get_ipython().system('python -m spacy download xx_ent_wiki_sm > /dev/null')


# In[ ]:


import numpy as np
import pandas as pd

import os
import gc

import spacy
from spacy_cld import LanguageDetector
import xx_ent_wiki_sm

from spellchecker import SpellChecker

import matplotlib.pyplot as plt
import seaborn as sns

import time
import random
from tqdm.notebook import tqdm
tqdm.pandas()

import re
import nltk

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)


# ## Language score

# In[ ]:


nlp = xx_ent_wiki_sm.load()
language_detector = LanguageDetector()
nlp.add_pipe(language_detector)


# In[ ]:


def get_lang_score(text, lang):
    try:
        doc = nlp(str(text))
        language_scores = doc._.language_scores
        return language_scores.get(lang, 0)
    except Exception:
        return 0


# In[ ]:


# Loading data

train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
train1['lang'] = 'en'

train_es = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-es-cleaned.csv')
train_es['lang'] = 'es'

train_fr = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-fr-cleaned.csv')
train_fr['lang'] = 'fr'

train_pt = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-pt-cleaned.csv')
train_pt['lang'] = 'pt'

train_ru = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-ru-cleaned.csv')
train_ru['lang'] = 'ru'

train_it = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-it-cleaned.csv')
train_it['lang'] = 'it'

train_tr = pd.read_csv('/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-tr-cleaned.csv')
train_tr['lang'] = 'tr'

train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
train2.toxic = train2.toxic.round().astype(int)
train2['lang'] = 'en'

train = pd.concat([
    
    train1[['comment_text', 'lang', 'toxic']],
    train_es[['comment_text', 'lang', 'toxic']],
    train_tr[['comment_text', 'lang', 'toxic']],
    train_fr[['comment_text', 'lang', 'toxic']],
    train_pt[['comment_text', 'lang', 'toxic']],
    train_ru[['comment_text', 'lang', 'toxic']],
    train_it[['comment_text', 'lang', 'toxic']],
    train2[['comment_text', 'lang', 'toxic']]
    
]).sample(n=20000).reset_index(drop=True)

del train1, train_es, train_fr, train_pt, train_ru, train_it, train_tr, train2
gc.collect()


# In[ ]:


train['lang_score'] = train.progress_apply(lambda x: get_lang_score(x['comment_text'], x['lang']), axis=1)


# In[ ]:


sns.distplot(train['lang_score'])


# In[ ]:


train = train[train['lang_score'] > 0.8]


# ## Correct misspellings

# In[ ]:


spell = SpellChecker()

# A quick example
misspelled = spell.unknown(['something', 'somegting', 'helo', 'fack', 'here', 'bijour'])


# In[ ]:


# Counting the number of spelling errors

train['mispell_count'] = train['comment_text'].progress_apply(lambda x: len(spell.unknown(x.split())))


# In[ ]:


train[train['mispell_count'] < 100]['mispell_count'].hist(bins=100)


# Since we have a lot of data, we can remove sentences with more than 20 mispells. But before that, let's have a look at the language distribution to make sure we don't remove too many sentences of the same language. Remember that SpellChecker is trained on some languages including Portuguese, Spanish, English, French.

# In[ ]:


sns.countplot(train['lang'])


# In[ ]:


sns.countplot(train[train['mispell_count'] < 20]['lang'])


# As expected, Russian sentences might be a bit more discarded since Spellcheck might not be able to correctly evaluate Russian sentences.
# **Tips: Always remember to check your data distribution before doing any transformation ;)**

# In[ ]:


train = train[train['mispell_count'] < 20]


# Finally, we can quickly replace the small artifacts \n by nothing.

# In[ ]:


train['comment_text'] = train['comment_text'].apply(lambda x: x.replace('\n', ' '))


# In[ ]:


train.head()


# We are done for this quick notebook. Hope it helped you clean your dataset. If so, don't hesitate to upvote the notebook ;)
