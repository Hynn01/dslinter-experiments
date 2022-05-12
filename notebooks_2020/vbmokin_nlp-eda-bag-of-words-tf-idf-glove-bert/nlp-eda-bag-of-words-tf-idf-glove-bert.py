#!/usr/bin/env python
# coding: utf-8

# <a class="anchor" id="0"></a>
# # [Real or Not? NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)
# 
# # NLP:
# * EDA (with WordCloud) 
# * Bag of Words 
# * TF IDF
# * GloVe
# * BERT with TFHub and with Submission
# * PCA visualization for the main models
# * Showing Confusion Matrices for BERT, Simpletransformers with DistilBERT and GloVe

# # Acknowledgements
# 
# This kernel uses such good notebooks: 
# * [Basic EDA,Cleaning and GloVe](https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove)
# * [Spooky NLP and Topic Modelling tutorial](https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial)
# * [NLP Tutorial using Python](https://www.kaggle.com/itratrahman/nlp-tutorial-using-python)
# * [Basic NLP with TensorFlow and WordCloud](https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud)
# * [[QUEST] Bert-base TF2.0](https://www.kaggle.com/akensert/bert-base-tf2-0-minimalistic)
# * [Bert-base TF2.0 (minimalistic) III](https://www.kaggle.com/khoongweihao/bert-base-tf2-0-minimalistic-iii)
# * [Disaster NLP: Keras BERT using TFHub & tuning, PCA](https://www.kaggle.com/vbmokin/disaster-nlp-keras-bert-using-tfhub-tuning)
# * [Bert starter (inference)](https://www.kaggle.com/user123454321/bert-starter-inference)
# * [Disaster NLP: Keras BERT using TFHub](https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub)
# * [Keras BERT using TFHub (modified train data)](https://www.kaggle.com/wrrosa/keras-bert-using-tfhub-modified-train-data)
# * [Text only - BERT - Keras](https://www.kaggle.com/rftexas/text-only-kfold-bert)
# * [Supershort NLP classification notebook](https://www.kaggle.com/vbmokin/supershort-nlp-classification-notebook)
# 
# my dataset [NLP with Disaster Tweets - cleaning data - source of data](https://www.kaggle.com/vbmokin/nlp-with-disaster-tweets-cleaning-data)
# 
# and other resources:
# * https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb
# * https://tfhub.dev/s?q=bert

# <a class="anchor" id="0.1"></a>
# ## Table of Contents
# 
# 1. [My upgrade BERT model](#1)
#     -  [Commit now](#1.1)
#     -  [Previous commits: Dropout = 0.1 or 0.3](#1.2)
#     -  [Previous commits: epochs = 3](#1.3)
#     -  [Previous commits: epochs = 4](#1.4)
#     -  [Previous commits: epochs = 5](#1.5)
#     -  [Previous commits: with training tweets correction](#1.6)
#     -  [Previous commits: parameters and LB scores](#1.7)    
# 1. [Import libraries](#2)
# 1. [Download data](#3)
# 1. [EDA](#4)
# 1. [Data Cleaning](#5)
# 1. [WordCloud](#6)
# 1. [Bag of Words Counts](#7)
# 1. [TF IDF](#8)
# 1. [GloVe](#9)
# 1. [BERT using TFHub](#10)
#    - [Submission](#10.1)
# 1. [Showing Confusion Matrices](#12)

# ## 1. My upgrade BERT model <a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


random_state_split = 42
Dropout_num = 0
learning_rate = 5.95e-6
valid = 0.15
epochs_num = 3
batch_size_num = 16
target_corrected = False
target_big_corrected = False


# ## 1.1. Commit now <a class="anchor" id="1.1"></a>
# 
# [Back to Table of Contents](#0.1)

# * random_state_split = 42
# * without Dropout
# * Adam(lr=5.95e-6)
# * validation_split = 0.15
# * epochs = 3
# * batch_size = 16
# * target_corrected = False
# * target_big_corrected = False

# ## 1.2. Previous commits: Dropout = 0.1 or 0.3 <a class="anchor" id="1.2"></a>
# 
# [Back to Table of Contents](#0.1)

# ### Commit 24
# * Dropout(0.3)
# * Adam(lr=2e-5)
# * validation_split = 0.2,
# * epochs = 4,
# * batch_size = 32
# 
# **LB = 0.80470**

# ### Commit 30
# * Dropout(0.1)
# * Adam(lr=1e-5)
# * validation_split = 0.2,
# * epochs = 4,
# * batch_size = 32
# 
# **LB = 0.83333**

# ## 1.3. Previous commits: epochs = 3 <a class="anchor" id="1.3"></a>
# 
# [Back to Table of Contents](#0.1)

# ### Commit 14
# * Dropout(0.2)
# * Adam(lr=2e-5)
# * validation_split = 0.2,
# * epochs = 3,
# * batch_size = 32
# 
# **LB = 0.83128**

# ### Commit 37
# #### From https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub (commit 2)
# * without Dropout
# * Adam(lr=2e-5)
# * validation_split = 0.2,
# * epochs = 3,
# * batch_size = 32
# 
# **LB = 0.81390** - This is strange, since this original model gave LB = 0.84355

# ### Commit 49
# * without Dropout
# * Adam(lr=5e-6)
# * validation_split = 0.2,
# * epochs = 3,
# * batch_size = 16
# 
# **LB = 0.83537**

# ### Commit 51
# * without Dropout
# * Adam(lr=3e-6)
# * validation_split = 0.2,
# * epochs = 3,
# * batch_size = 16
# * target_corrected = True
# 
# **LB = 0.83231**

# ### Commit 55
# * without Dropout
# * Adam(lr=5e-6)
# * validation_split = 0.3
# * epochs = 3
# * batch_size = 16
# * target_corrected = False
# 
# **LB = 0.82924**

# ### Commit 60 (The best!)
# * without Dropout
# * Adam(lr=6e-6)
# * validation_split = 0.2
# * epochs = 3
# * batch_size = 16
# * target_corrected = False
# 
# **LB = 0.85378**

# ### Commit 62
# * without Dropout
# * Adam(lr=5.9e-6)
# * validation_split = 0.2
# * epochs = 3
# * batch_size = 16
# * target_corrected = False
# 
# **LB = 0.83537**

# ### Commit 73
# * without Dropout
# * Adam(lr=6e-6)
# * validation_split = 0.25
# * epochs = 3
# * batch_size = 16
# * target_corrected = False
# 
# **LB = 0.83640**

# ### Commit 74
# * without Dropout
# * Adam(lr=6e-6)
# * validation_split = 0.15
# * epochs = 3
# * batch_size = 16
# * target_corrected = False
# 
# **LB = 0.83946**

# ### Commit 78
# * random_state_split = 21
# * without Dropout
# * Adam(lr=6e-6)
# * validation_split = 0.2
# * epochs = 3
# * batch_size = 16
# * target_corrected = False
# 
# **LB = 0.84355**

# ### Commit 80
# * random_state_split = 32
# * without Dropout
# * Adam(lr=6e-6)
# * validation_split = 0.2
# * epochs = 3
# * batch_size = 16
# * target_corrected = False
# 
# **LB = 0.83665**

# ## 1.4. Previous commits: epochs = 4 <a class="anchor" id="1.4"></a>
# 
# [Back to Table of Contents](#0.1)

# ### Commit 15
# * Dropout(0.2)
# * Adam(lr=2e-5)
# * validation_split = 0.2,
# * epochs = 4,
# * batch_size = 32
# 
# **LB = 0.83742**

# ### Commit 25
# * Dropout(0.2)
# * Adam(lr=3e-5)
# * validation_split = 0.2,
# * epochs = 4,
# * batch_size = 32
# 
# **LB = 0.83537**

# ### Commit 26
# * Dropout(0.2)
# * Adam(lr=4e-5)
# * validation_split = 0.2,
# * epochs = 4,
# * batch_size = 32
# 
# **LB = 0.82617**

# ### Commit 27
# * Dropout(0.2)
# * Adam(lr=5e-4)
# * validation_split = 0.2,
# * epochs = 4,
# * batch_size = 32
# 
# **LB = 0.57055**

# ### Commit 28
# * Dropout(0.2)
# * Adam(lr=5e-6)
# * validation_split = 0.2,
# * epochs = 4,
# * batch_size = 32
# 
# **LB = 0.82924**

# ### Commit 29
# * Dropout(0.2)
# * Adam(lr=1e-5)
# * validation_split = 0.2,
# * epochs = 4,
# * batch_size = 32
# 
# **LB = 0.83742**

# ### Commit 31
# * Dropout(0.2)
# * Adam(lr=1e-5)
# * validation_split = 0.15,
# * epochs = 4,
# * batch_size = 32
# 
# **LB = 0.79856**

# ### Commit 33
# * Dropout(0.2)
# * Adam(lr=2e-5)
# * validation_split = 0.3,
# * epochs = 4,
# * batch_size = 32
# 
# **LB = 0.83128**

# ### Commit 35
# * Dropout(0.15)
# * Adam(lr=2e-5)
# * validation_split = 0.2,
# * epochs = 4,
# * batch_size = 32
# 
# **LB = 0.81390**

# ### Commit 43
# * without Dropout
# * Adam(lr=5e-6)
# * validation_split = 0.2,
# * epochs = 4,
# * batch_size = 32
# 
# **LB = 0.83844**

# ### Commit 45
# * without Dropout
# * Adam(lr=2e-6)
# * validation_split = 0.2,
# * epochs = 4,
# * batch_size = 32
# 
# **LB = 0.82719**

# ### Commit 46
# * without Dropout
# * Adam(lr=5e-6)
# * validation_split = 0.2,
# * epochs = 4,
# * batch_size = 32
# * target_corrected = True
# * target_big_corrected = True
# 
# **LB = 0.83333**

# ### Commit 48
# * without Dropout
# * Adam(lr=5e-6)
# * validation_split = 0.2,
# * epochs = 4,
# * batch_size = 16
# * target_corrected = False
# * target_big_corrected = False
# 
# **LB = 0.83231**

# ## 1.5. Previous commits: epochs = 5 <a class="anchor" id="1.5"></a>
# 
# [Back to Table of Contents](#0.1)

# ### Commit 23
# * Dropout(0.2)
# * Adam(lr=2e-5)
# * validation_split = 0.2,
# * epochs = 5,
# * batch_size = 32
# 
# **LB = 0.83435**

# ### Commit 39
# * without Dropout
# * Adam(lr=2e-5)
# * validation_split = 0.2,
# * epochs = 4,
# * batch_size = 24
# 
# **LB = 0.83435**

# ### Commit 40
# * without Dropout
# * Adam(lr=2e-5)
# * validation_split = 0.2,
# * epochs = 5,
# * batch_size = 32
# 
# **LB = 0.81186**

# ### Commit 41
# * without Dropout
# * Adam(lr=1e-5)
# * validation_split = 0.2,
# * epochs = 5,
# * batch_size = 32
# 
# **LB = 0.83231**

# ## 1.6. Previous commits: with training tweets correction <a class="anchor" id="1.6"></a>
# 
# [Back to Table of Contents](#0.1)

# **target_corrected = True :**
# 
# From https://www.kaggle.com/wrrosa/keras-bert-using-tfhub-modified-train-data - author of this kernel read tweets in training data and figure out that some of them have errors:
# 
#     ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]
#     train.loc[train['id'].isin(ids_with_target_error),'target'] = 0

# ### Commit 18
# * Dropout(0.2)
# * Adam(lr=2e-5)
# * validation_split = 0.2,
# * epochs = 3,
# * batch_size = 16
# 
# 
# 
# **LB = 0.83537**

# ### Commit 20
# * without Dropout
# * Adam(lr=1e-5)
# * validation_split = 0.2,
# * epochs = 3,
# * batch_size = 16
# 
# **LB = 0.82413**

# ### Commit 22
# * Dropout(0.2)
# * Adam(lr=1.1e-5)
# * validation_split = 0.2,
# * epochs = 3,
# * batch_size = 16
# 
# **LB (for BERT) = 0.80879**

# ### Commit 32
# 
# * Dropout(0.2)
# * Adam(lr=2e-5)
# * validation_split = 0.2,
# * epochs = 4,
# * batch_size = 32
# 
# **LB = 0.82004**

# ## 1.7. Previous commits: parameters and LB scores<a class="anchor" id="1.7"></a>
# 
# [Back to Table of Contents](#0.1)

# ### Only successful commits :

# * Commit 60
# * without Dropout/lr=6e-6/val=0.2/epochs=3/batch=16/target_corrected=False
# * LB = **0.85378**
# * Commit 43
# * without Dropout/lr=5e-6/val=0.2/epochs=4/batch=32/target_corrected=False
# * LB = 0.83844
# * 15
# * Dropout=0.2/lr=2e-5/val=0.2/epochs=4/batch=32/target_corrected=False
# * LB = 0.83742
# * 29
# * Dropout=0.2/lr=1e-5/val=0.2/epochs=4/batch=32/target_corrected=False
# * LB = 0.83742
# * 62
# * without Dropout/lr=5.9e-6/val=0.2/epochs=3/batch=16/target_corrected=False
# * LB = 0.83537
# * 49
# * without Dropout/lr=5e-6/val=0.2/epochs=3/batch=16/target_corrected=False
# * LB = 0.83537
# * 18
# * Dropout=0.2/lr=2e-5/val=0.2/epochs=3/batch=16/target_corrected=True
# * LB = 0.83537
# * 25
# * Dropout=0.2/lr=3e-5/val=0.2/epochs=4/batch=32/target_corrected=False
# * LB = 0.83537
# * 37
# * without Dropout/lr=2e-5/val=0.2/epochs=4/batch=24/target_corrected=False
# * LB = 0.83435
# * 23
# * Dropout=0.2/lr=2e-5/val=0.2/epochs=5/batch=32/target_corrected=False
# * LB = 0.83435
# * 46
# * without Dropout/lr=5e-6/val=0.2/epochs=4/batch=32/target_corrected=True/target_big_corrected = True
# * LB = 0.83333
# * 30
# * Dropout=0.1/lr=1e-5/val=0.2/epochs=4/batch=32/target_corrected=False
# * LB = 0.83333
# * 51
# * without Dropout/lr=3e-6/val=0.2/epochs=3/batch=16/target_corrected=True
# * LB = 0.83231
# * 46
# * without Dropout/lr=5e-6/val=0.2/epochs=4/batch=16/target_corrected=False
# * LB = 0.83231
# * 33
# * Dropout=0.2/lr=2e-5/val=0.3/epochs=4/batch=32/target_corrected=False
# * LB = 0.83128
# * 14
# * Dropout=0.2/lr=2e-5/val=0.2/epochs=3/batch=32/target_corrected=True
# * LB = 0.83128
# * 55
# * without Dropout/lr=5e-6/val=0.3/epochs=3/batch=16/target_corrected=False
# * LB = 0.82924
# * 28
# * Dropout=0.2/lr=5e-6/val=0.2/epochs=4/batch=32/target_corrected=False
# * LB = 0.82924
# * 26
# * Dropout=0.2/lr=4e-5/val=0.2/epochs=4/batch=32/target_corrected=False
# * LB = 0.82617
# * 45
# * without Dropout/lr=2e-6/val=0.2/epochs=4/batch=32/target_corrected=False
# * LB = 0.82719
# * 20
# * without Dropout/lr=1e-5/val=0.2/epochs=3/batch=16/target_corrected=True
# * LB = 0.82413
# * 42
# * without Dropout/lr=1e-5/val=0.2/epochs=5/batch=32/target_corrected=False
# * LB = 0.83231
# * 32
# * Dropout=0.2/lr=2e-5/val=0.2/epochs=4/batch=32/target_corrected=True
# * LB = 0.82004
# * 37
# * without Dropout/lr=2e-5/val=0.2/epochs=3/batch=32/target_corrected=False
# * LB = 0.81390
# * 35
# * Dropout=0.15/lr=2e-5/val=0.2/epochs=4/batch=32/target_corrected=False
# * LB = 0.81390
# * 40
# * without Dropout/lr=2e-5/val=0.2/epochs=5/batch=32/target_corrected=False
# * LB = 0.81186
# * 22
# * Dropout=0.2/lr=1.1e-5/val=0.2/epochs=3/batch=16/target_corrected=True
# * LB = 0.80879
# * 24
# * Dropout=0.3/lr=2e-5/val=0.2/epochs=4/batch=32/target_corrected=False
# * LB = 0.80470
# * 31
# * Dropout=0.2/lr=1e-5/val=0.15/epochs=4/batch=32/target_corrected=False
# * LB = 0.79856
# * 27
# * Dropout=0.2/lr=5e-4/val=0.2/epochs=4/batch=32/target_corrected=False
# * LB = 0.57055

# ## 2. Import libraries <a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from nltk.corpus import stopwords
from nltk.util import ngrams

from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import classification_report,confusion_matrix

from collections import defaultdict
from collections import Counter
plt.style.use('ggplot')
stop=set(stopwords.words('english'))

import re
from nltk.tokenize import word_tokenize
import gensim
import string

from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM,Dense, SpatialDropout1D, Dropout
from keras.initializers import Constant
from keras.optimizers import Adam

import torch

import warnings
warnings.simplefilter('ignore')


# ## 3. Download data <a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


tweet= pd.read_csv('../input/nlp-getting-started/train.csv')
test=pd.read_csv('../input/nlp-getting-started/test.csv')
submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")


# In[ ]:


# # From https://www.kaggle.com/wrrosa/keras-bert-using-tfhub-modified-train-data - 
# # author of this kernel read tweets in training data and figure out that some of them have errors:
# ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]
# tweet.loc[tweet['id'].isin(ids_with_target_error),'target'] = 0
# tweet[tweet['id'].isin(ids_with_target_error)]


# In[ ]:


print('There are {} rows and {} columns in train'.format(tweet.shape[0],tweet.shape[1]))
print('There are {} rows and {} columns in test'.format(test.shape[0],test.shape[1]))


# In[ ]:


tweet.head(10)


# ## 4. EDA <a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# Thanks to:
# * https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove
# * https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial
# * https://www.kaggle.com/itratrahman/nlp-tutorial-using-python

# ### Class distribution

# Before we begin with anything else, let's check the class distribution.

# In[ ]:


# extracting the number of examples of each class
Real_len = tweet[tweet['target'] == 1].shape[0]
Not_len = tweet[tweet['target'] == 0].shape[0]


# In[ ]:


# bar plot of the 3 classes
plt.rcParams['figure.figsize'] = (7, 5)
plt.bar(10,Real_len,3, label="Real", color='blue')
plt.bar(15,Not_len,3, label="Not", color='red')
plt.legend()
plt.ylabel('Number of examples')
plt.title('Propertion of examples')
plt.show()


# ### Number of characters in tweets

# In[ ]:


def length(text):    
    '''a function which returns the length of text'''
    return len(text)


# In[ ]:


tweet['length'] = tweet['text'].apply(length)


# In[ ]:


plt.rcParams['figure.figsize'] = (18.0, 6.0)
bins = 150
plt.hist(tweet[tweet['target'] == 0]['length'], alpha = 0.6, bins=bins, label='Not')
plt.hist(tweet[tweet['target'] == 1]['length'], alpha = 0.8, bins=bins, label='Real')
plt.xlabel('length')
plt.ylabel('numbers')
plt.legend(loc='upper right')
plt.xlim(0,150)
plt.grid()
plt.show()


# In[ ]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
tweet_len=tweet[tweet['target']==1]['text'].str.len()
ax1.hist(tweet_len,color='blue')
ax1.set_title('disaster tweets')
tweet_len=tweet[tweet['target']==0]['text'].str.len()
ax2.hist(tweet_len,color='red')
ax2.set_title('Not disaster tweets')
fig.suptitle('Characters in tweets')
plt.show()


# The distribution of both seems to be almost same.120 t0 140 characters in a tweet are the most common among both.

# ### Number of words in a tweet

# In[ ]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
tweet_len=tweet[tweet['target']==1]['text'].str.split().map(lambda x: len(x))
ax1.hist(tweet_len,color='blue')
ax1.set_title('disaster tweets')
tweet_len=tweet[tweet['target']==0]['text'].str.split().map(lambda x: len(x))
ax2.hist(tweet_len,color='red')
ax2.set_title('Not disaster tweets')
fig.suptitle('Words in a tweet')
plt.show()


# ###  Average word length in a tweet

# In[ ]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
word=tweet[tweet['target']==1]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='blue')
ax1.set_title('disaster')
word=tweet[tweet['target']==0]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='red')
ax2.set_title('Not disaster')
fig.suptitle('Average word length in each tweet')


# In[ ]:


def create_corpus(target):
    corpus=[]
    
    for x in tweet[tweet['target']==target]['text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus


# In[ ]:


def create_corpus_df(tweet, target):
    corpus=[]
    
    for x in tweet[tweet['target']==target]['text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus


# ### Common stopwords in tweets

# First we  will analyze tweets with class 0.

# In[ ]:


corpus=create_corpus(0)

dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1
        
top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10]


# In[ ]:


# displaying the stopwords
np.array(stop)


# In[ ]:


plt.rcParams['figure.figsize'] = (18.0, 6.0)
x,y=zip(*top)
plt.bar(x,y)


# Now,we will analyze tweets with class 1.

# In[ ]:


corpus=create_corpus(1)

dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
    

plt.rcParams['figure.figsize'] = (18.0, 6.0)
x,y=zip(*top)
plt.bar(x,y)


# In both of them,"the" dominates which is followed by "a" in class 0 and "in" in class 1.

# ### Analyzing punctuations

# First let's check tweets indicating real disaster.

# In[ ]:


plt.figure(figsize=(16,5))
corpus=create_corpus(1)

dic=defaultdict(int)
special = string.punctuation
for i in (corpus):
    if i in special:
        dic[i]+=1
        
x,y=zip(*dic.items())
plt.bar(x,y)


# Now,we will move on to class 0.

# In[ ]:


plt.figure(figsize=(16,5))
corpus=create_corpus(0)
dic=defaultdict(int)
special = string.punctuation
for i in (corpus):
    if i in special:
        dic[i]+=1
        
x,y=zip(*dic.items())
plt.bar(x,y,color='green')


# ### Common words

# In[ ]:


plt.figure(figsize=(16,5))
counter=Counter(corpus)
most=counter.most_common()
x=[]
y=[]
for word,count in most[:40]:
    if (word not in stop) :
        x.append(word)
        y.append(count)


# In[ ]:


sns.barplot(x=y,y=x)


# Lot of cleaning needed !

# ### N-gram analysis

# we will do a bigram (n=2) analysis over the tweets. Let's check the most common bigrams in tweets.

# In[ ]:


def get_top_tweet_bigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# In[ ]:


plt.figure(figsize=(16,5))
top_tweet_bigrams=get_top_tweet_bigrams(tweet['text'])[:10]
x,y=map(list,zip(*top_tweet_bigrams))
sns.barplot(x=y,y=x)


# ## 5. Data Cleaning <a class="anchor" id="5"></a>
# 
# [Back to Table of Contents](#0.1)

# Thanks to https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove

# In[ ]:


df=pd.concat([tweet,test])
df.shape


# ### Removing urls

# In[ ]:


example="New competition launched :https://www.kaggle.com/c/nlp-getting-started"


# In[ ]:


def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

remove_URL(example)


# In[ ]:


df['text']=df['text'].apply(lambda x : remove_URL(x))


# ### Removing HTML tags

# In[ ]:


example = """<div>
<h1>Real or Fake</h1>
<p>Kaggle </p>
<a href="https://www.kaggle.com/c/nlp-getting-started">getting started</a>
</div>"""


# In[ ]:


def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
print(remove_html(example))


# In[ ]:


df['text']=df['text'].apply(lambda x : remove_html(x))


# ### Removing Emojis

# In[ ]:


# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

remove_emoji("Omg another Earthquake 😔😔")


# In[ ]:


df['text']=df['text'].apply(lambda x: remove_emoji(x))


# ### Removing punctuations

# In[ ]:


def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

example="I am a #king"
print(remove_punct(example))


# In[ ]:


df['text']=df['text'].apply(lambda x : remove_punct(x))


# ## 6. WordCloud <a class="anchor" id="6"></a>
# 
# [Back to Table of Contents](#0.1)

# Thanks to https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial

# ### Real Disaster

# In[ ]:


corpus_new1=create_corpus_df(df,1)
len(corpus_new1)


# In[ ]:


corpus_new1[:10]


# In[ ]:


# Generating the wordcloud with the values under the category dataframe
plt.figure(figsize=(12,8))
word_cloud = WordCloud(
                          background_color='black',
                          max_font_size = 80
                         ).generate(" ".join(corpus_new1[:50]))
plt.imshow(word_cloud)
plt.axis('off')
plt.show()


# ### Not Disaster

# In[ ]:


corpus_new0=create_corpus_df(df,0)
len(corpus_new0)


# In[ ]:


corpus_new0[:10]


# In[ ]:


# Generating the wordcloud with the values under the category dataframe
plt.figure(figsize=(12,8))
word_cloud = WordCloud(
                          background_color='black',
                          max_font_size = 80
                         ).generate(" ".join(corpus_new0[:50]))
plt.imshow(word_cloud)
plt.axis('off')
plt.show()


# In[ ]:


df.head(10)


# ## 7. Bag of Words Counts <a class="anchor" id="7"></a>
# 
# [Back to Table of Contents](#0.1)

# Thanks to https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb

# In[ ]:


def cv(data):
    count_vectorizer = CountVectorizer()

    emb = count_vectorizer.fit_transform(data)

    return emb, count_vectorizer

list_corpus = df["text"].tolist()
list_labels = df["target"].tolist()

X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2, 
                                                                                random_state=random_state_split)

X_train_counts, count_vectorizer = cv(X_train)
X_test_counts = count_vectorizer.transform(X_test)


# ### Visualizing the embeddings

# In[ ]:


def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):
        lsa = TruncatedSVD(n_components=2)
        lsa.fit(test_data)
        lsa_scores = lsa.transform(test_data)
        color_mapper = {label:idx for idx,label in enumerate(set(test_labels))}
        color_column = [color_mapper[label] for label in test_labels]
        colors = ['orange','blue']
        if plot:
            plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))
            orange_patch = mpatches.Patch(color='orange', label='Not')
            blue_patch = mpatches.Patch(color='blue', label='Real')
            plt.legend(handles=[orange_patch, blue_patch], prop={'size': 30})

fig = plt.figure(figsize=(16, 16))          
plot_LSA(X_train_counts, y_train)
plt.show()


# These embeddings don't look very cleanly separated. Let's see if we can still fit a useful model on them.

# ## 8. TF IDF <a class="anchor" id="8"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()

    train = tfidf_vectorizer.fit_transform(data)

    return train, tfidf_vectorizer

X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# In[ ]:


fig = plt.figure(figsize=(16, 16))          
plot_LSA(X_train_tfidf, y_train)
plt.show()


# ## 9. GloVe <a class="anchor" id="9"></a>
# 
# [Back to Table of Contents](#0.1)

# Thanks to https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove

# Here we will use GloVe pretrained corpus model to represent our words. It is available in 3 varieties : 50D, 100D and 200 Dimentional. We will try 100D here.

# In[ ]:


def create_corpus_new(df):
    corpus=[]
    for tweet in tqdm(df['text']):
        words=[word.lower() for word in word_tokenize(tweet)]
        corpus.append(words)
    return corpus   


# In[ ]:


corpus=create_corpus_new(df)


# In[ ]:


embedding_dict={}
with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r') as f:
    for line in f:
        values=line.split()
        word = values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
f.close()


# In[ ]:


MAX_LEN=50
tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(corpus)
sequences=tokenizer_obj.texts_to_sequences(corpus)

tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')


# In[ ]:


word_index=tokenizer_obj.word_index
print('Number of unique words:',len(word_index))


# In[ ]:


num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,100))

for word,i in tqdm(word_index.items()):
    if i < num_words:
        emb_vec=embedding_dict.get(word)
        if emb_vec is not None:
            embedding_matrix[i]=emb_vec           


# In[ ]:


tweet_pad[0][0:]


# ## Baseline Model with GloVe results

# In[ ]:


model=Sequential()

embedding=Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),
                   input_length=MAX_LEN,trainable=False)

model.add(embedding)
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))


optimzer=Adam(learning_rate=3e-4)

model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


train=tweet_pad[:tweet.shape[0]]
test=tweet_pad[tweet.shape[0]:]


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(train,tweet['target'].values,test_size=0.2)
print('Shape of train',X_train.shape)
print("Shape of Validation ",X_test.shape)


# In[ ]:


fig = plt.figure(figsize=(16, 16))          
plot_LSA(train,tweet['target'])
plt.show()


# In[ ]:


# Recomended 10-20 epochs
history=model.fit(X_train,y_train,batch_size=4,epochs=10,validation_data=(X_test,y_test),verbose=2)


# In[ ]:


train_pred_GloVe = model.predict(train)
train_pred_GloVe_int = train_pred_GloVe.round().astype('int')


# ## 10. BERT using TFHub <a class="anchor" id="10"></a>
# 
# [Back to Table of Contents](#0.1)

# Thanks to very good kernel https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub

# In[ ]:


# We will use the official tokenization script created by the Google team
get_ipython().system('wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py')


# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub

import tokenization


# In[ ]:


# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub
def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


# In[ ]:


# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub
def build_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    
    if Dropout_num == 0:
        # Without Dropout
        out = Dense(1, activation='sigmoid')(clf_output)
    else:
        # With Dropout(Dropout_num), Dropout_num > 0
        x = Dropout(Dropout_num)(clf_output)
        out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# ## Big target correction

# In[ ]:


# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert
def clean_tweets(tweet):
    """Removes links and non-ASCII characters"""
    
    tweet = ''.join([x for x in tweet if x in string.printable])
    
    # Removing URLs
    tweet = re.sub(r"http\S+", "", tweet)
    
    return tweet


# In[ ]:


# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# In[ ]:


# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert
def remove_punctuations(text):
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"
    
    for p in punctuations:
        text = text.replace(p, f' {p} ')

    text = text.replace('...', ' ... ')
    
    if '...' not in text:
        text = text.replace('..', ' ... ')
    
    return text


# In[ ]:


# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert
abbreviations = {
    "$" : " dollar ",
    "€" : " euro ",
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk", 
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart", 
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens", #"que pasa",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet", 
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously", 
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"
}


# In[ ]:


# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert
def convert_abbrev(word):
    return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word


# In[ ]:


# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert
def convert_abbrev_in_text(text):
    tokens = word_tokenize(text)
    tokens = [convert_abbrev(word) for word in tokens]
    text = ' '.join(tokens)
    return text


# ## Download data

# In[ ]:


# Load BERT from the Tensorflow Hub
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)


# In[ ]:


# Load CSV files containing training data
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")


# ## Target correction

# In[ ]:


# Thanks to https://www.kaggle.com/wrrosa/keras-bert-using-tfhub-modified-train-data - 
# author of this kernel read tweets in training data and figure out that some of them have errors:
if target_corrected:
    ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]
    train.loc[train['id'].isin(ids_with_target_error),'target'] = 0
    train[train['id'].isin(ids_with_target_error)]


# In[ ]:


# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert
if target_big_corrected:
    train["text"] = train["text"].apply(lambda x: clean_tweets(x))
    test["text"] = test["text"].apply(lambda x: clean_tweets(x))
    
    train["text"] = train["text"].apply(lambda x: remove_emoji(x))
    test["text"] = test["text"].apply(lambda x: remove_emoji(x))
    
    train["text"] = train["text"].apply(lambda x: remove_punctuations(x))
    test["text"] = test["text"].apply(lambda x: remove_punctuations(x))
    
    train["text"] = train["text"].apply(lambda x: convert_abbrev_in_text(x))
    test["text"] = test["text"].apply(lambda x: convert_abbrev_in_text(x))


# ## Build and train BERT model

# In[ ]:


# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub
# Load tokenizer from the bert layer
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)


# In[ ]:


# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub
# Encode the text into tokens, masks, and segment flags
train_input = bert_encode(train.text.values, tokenizer, max_len=160)
test_input = bert_encode(test.text.values, tokenizer, max_len=160)
train_labels = train.target.values


# In[ ]:


# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub
# Build BERT model with my tuning
model_BERT = build_model(bert_layer, max_len=160)
model_BERT.summary()


# In[ ]:


# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub
# Train BERT model with my tuning
checkpoint = ModelCheckpoint('model_BERT.h5', monitor='val_loss', save_best_only=True)

train_history = model_BERT.fit(
    train_input, train_labels,
    validation_split = valid,
    epochs = epochs_num, # recomended 3-5 epochs
    callbacks=[checkpoint],
    batch_size = batch_size_num
)


# ## Prediction

# In[ ]:


# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub
# Prediction by BERT model with my tuning
model_BERT.load_weights('model_BERT.h5')
test_pred_BERT = model_BERT.predict(test_input)
test_pred_BERT_int = test_pred_BERT.round().astype('int')


# In[ ]:


# Prediction by BERT model with my tuning for the training data - for the Confusion Matrix
train_pred_BERT = model_BERT.predict(train_input)
train_pred_BERT_int = train_pred_BERT.round().astype('int')


# ### 10.1. Submission<a class="anchor" id="10.1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


pred = pd.DataFrame(test_pred_BERT, columns=['preds'])
pred.plot.hist()


# In[ ]:


submission['target'] = test_pred_BERT_int
submission.head(10)


# In[ ]:


submission.to_csv("submission_BERT.csv", index=False, header=True)


# ## 11. Showing Confusion Matrices<a class="anchor" id="11"></a>
# 
# [Back to Table of Contents](#0.1)

# Thanks to https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud

# In[ ]:


# Showing Confusion Matrix
def plot_cm(y_true, y_pred, title, figsize=(5,5)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)


# In[ ]:


# Showing Confusion Matrix for GloVe model
plot_cm(train_pred_GloVe_int, train['target'].values, 'Confusion matrix for GloVe model', figsize=(7,7))


# In[ ]:


# Showing Confusion Matrix for BERT model
plot_cm(train_pred_BERT_int, train['target'].values, 'Confusion matrix for BERT model', figsize=(7,7))


# I hope you find this notebook useful and enjoyable.

# Your comments and feedback are most welcome.

# [Go to Top](#0)
