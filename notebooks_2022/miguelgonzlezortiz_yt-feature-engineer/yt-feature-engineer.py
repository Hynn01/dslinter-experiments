#!/usr/bin/env python
# coding: utf-8

# # YouTube Trending Videos Analysis
# 
# This notebook covers a small part of exploration. Includes a way to merge the csv files with the category json files.   
# With all the data together in the same dataframe, the idea is to prepare the dataset for different tasks like classification, regression, clustering and recommendation.
# 
# ## Table of contents:
#  
# 1. Imports
# 2. Joining files
# 3. Feature Exploration

# ## 1. Prepare Data and Import Libraries

# In[ ]:


import os
import re
import string
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import datetime
import nltk
from collections import Counter


# In[ ]:


get_ipython().system('mkdir json/ videos/')
get_ipython().system('cp ../input/youtube-new/*.json json/')
get_ipython().system('cp ../input/youtube-new/*videos.csv videos/')


# ## 2. Join video and category

# In[ ]:


path='videos/'
videos = pd.DataFrame()
for file in os.listdir(path=path):
    try:
        video = pd.read_csv(path+file,encoding='utf-8',index_col='video_id')
        video['country'] = file[:2]
        videos=videos.append(video)

    except:
        ## Hay csv que no se pueden leer con uft-8
        video = pd.read_csv(path+file,encoding='latin-1',index_col='video_id')
        video['country']=file[:2]
        videos=videos.append(video)

videos['videos_id'] = videos.index.values.reshape(-1,1)


# In[ ]:


path = 'json/'
categories=pd.DataFrame()
for file in os.listdir(path=path):
    with open(path+file) as f:
        data = pd.DataFrame(json.load(f)['items'])
        id = pd.Series(data['id'],dtype=int)
        assignable = data['snippet'].apply(lambda x: x['assignable'])
        title = data['snippet'].apply(lambda x: x['title'])

        category = pd.concat([id,assignable,title],axis=1)
        category.columns = ['category_id', 'assignable', 'snippet']
        category['country'] = file[:2]

        categories = categories.append(category)


# In[ ]:


df = videos.merge(categories,on=['category_id','country'])

df.head()


# ## 3. Feature Exploration

# ### Correlations and Class Balance
# 

# In[ ]:


# duplicated rows
print("Duplicated Rows: {}".format(df.shape[0]-df.drop_duplicates().shape[0]))

df = df.drop_duplicates()


# In[ ]:


unique_videos = df[['title','videos_id']].drop_duplicates()
print("Size of the original dataset: {}".format(len(df)))
print("Number of unique videos in the dataset: {}".format(len(unique_videos)))

#Some videos are repeated, but with different trending dates

titles_count = df['title'].value_counts()
df['trending_time'] = df.apply(lambda x: titles_count[x['title']],axis=1)
# Trending_time is the number of days in trending.


# In[ ]:


df.isna().sum()
# 48325 missing values in 'description'


# In[ ]:


print(df.columns)

columns_floats = ['views', 'likes', 'dislikes','comment_count','comments_disabled', 'ratings_disabled']
columns_dates = ['trending_date','publish_time']


# In[ ]:


#Float variables correlations
sns.set(rc={"figure.figsize":(10, 8)})
sns.heatmap(df[columns_floats].corr(),annot=True)


# In[ ]:


sns.countplot(data=df,x='snippet')
plt.xticks(rotation= 90)


# In[ ]:


# Some categories has low representation, can be eliminated

#categories = df['snippet'].value_counts()[:13].index
#print(categories)

#df = df.query('snippet in @categories')


# ### Feature Engineer

# In[ ]:


def DefineDatetimeVariables(DF):
    DF['Month'] = DF['trending_date'].apply(lambda x:datetime.datetime.strptime(x,'%y.%d.%m').month)
    DF['DifferTime'] = DF.apply(lambda x:(datetime.datetime.strptime(x.trending_date,'%y.%d.%m')-datetime.datetime.strptime(x.publish_time,'%Y-%m-%dT%H:%M:%S.000Z')).days,axis=1)
    DF['Hora'] = DF['publish_time'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%dT%H:%M:%S.000Z').hour)

DefineDatetimeVariables(df)

# Month as a ordinal veriable, not categorical.


# I change likes and dislikes for it's ratio and total the number of 'interactions': likes+dislikes

# In[ ]:


def TransformLikes(DF):
    DF['liked'] = DF.apply(lambda x: x.likes/(x.likes+x.dislikes+1e-10),axis=1)
    DF['ratings']= DF.apply(lambda x: x.likes+x.dislikes,axis=1)

TransformLikes(df)


# In[ ]:


columns_floats = ['views',
 #'likes',
 #'dislikes',
 'liked',
 'ratings',
 'comments_disabled',
 'comment_count',
 'ratings_disabled',
 'Month',
 'DifferTime',
 'Hora']

df[columns_floats].corr()


# ### Tokenizer
# 
# Remove punctuation.
# Remove Stopwords

# In[ ]:


def remove_punc(text):
    text = text.replace('|',' ')
    text = "".join([chart for chart in text if chart not in string.punctuation])

    return text

df['tags_clean'] = df['tags'].apply(lambda x: remove_punc(x))


# In[ ]:


def tokenizer(text):
    tokens = re.split('\W+',text)
    return tokens

df['tokens'] = df['tags_clean'].apply(lambda x: tokenizer(x.lower()))


# #### Remove Stopwords and numbers

# In[ ]:


nltk.download('stopwords')
stopwords_en = nltk.corpus.stopwords.words('english')
stopwords_sp = nltk.corpus.stopwords.words('spanish')

stopwords = stopwords_en + stopwords_sp


# In[ ]:


def remove_stopwords(tokens):
    tokens = [token for token in tokens if token not in stopwords]
    return tokens

df['tokens'] = df['tokens'].apply(lambda x: remove_stopwords(x))


# In[ ]:


def remove_numb(tokens):
    tokens = [token for token in tokens if token.isalpha()]
    return tokens

df['tokens'] = df['tokens'].apply(lambda x: remove_numb(x))


# #### Count words

# In[ ]:


unique_videos = df[['title','videos_id']].drop_duplicates()

c = Counter()
def counter(text):
      c.update(text)
df.loc[unique_videos.index,'tokens'].apply(lambda x: counter(x))

common_words_1 = [word[0] for word in c.most_common()[:300]]


# In[ ]:


mono_words = list()
for word in common_words_1:
    if len(word)==1:
        mono_words.append(word)

for word in mono_words:
    common_words_1.remove(word)


# ### Create new features
# 
# For each most common word, I create a new feature in the dataset, that counts the how many times this word appears in tag. 

# In[ ]:


for word in common_words_1:
    df[word] = df['tokens'].apply(lambda x: x.count(word)) 


# In[ ]:


df.head()

