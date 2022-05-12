#!/usr/bin/env python
# coding: utf-8

# # Don’t miss what’s happening...🐤
# 
# ### People on Twitter are the first to know.
# 
# 
# ![tweet](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTigQWzoYCNiDyrz1BN4WTf2X2k9OZ_yvW-FsmcIMsdS9fppNmh)
# * In this competition, you’re challenged to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t.

# # Summary
# 
# In this notebook I tried to apply what I learned on NLP.
# This notebook includes:
# * Preprocessing the text
# * visualizing the processed data by several methods like
#  * tweet lenghts
#  * word counts
#  * average word lengths
#  * ngrams
# * build baseline models.
# 
# I hope this notebook helps some of you as others helped me a lot.

# # Getting the Text Data Ready

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF


import re
import string

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize



import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')


# In[ ]:


train = pd.read_csv('../input/nlp-getting-started/train.csv')
test = pd.read_csv('../input/nlp-getting-started/test.csv')

# dispaly data
print(f'Train shape: {train.shape}\nTest shape: {test.shape}\n', ' '.join(['-']*70))

print('Train data')
display(train.head())
print('Test Data')
display(test.head())


# # Combine Data For Preprossesing

# In[ ]:


train= pd.concat([train,test]).reset_index()


# # Target Distribution
# 
# When we check our target variables and look at how they disturbuted we can say it not bad. There is no huge difference between classes we can say it's good sign for modelling

# In[ ]:


# Target Distribution

fig, ax = plt.subplots(ncols=2,nrows=1,figsize=(18,6),dpi=100)

sns.countplot(train.target, ax=ax[0])
ax[0].bar_label(ax[0].containers[0])
ax[0].set_xticklabels(['Not Disaster', 'Disaster'])


ax[1].pie(train.target.value_counts(),
          labels=['Not Disaster', 'Disaster'],
          autopct='%.2f%%',
          explode=(0.05, 0),
          #textprops={'fontsize': 12},
          #shadow=True,
          startangle=90)


fig.suptitle('Distribution of the Target', fontsize=24)

plt.show()


# # Cleaning Text
# 
# So basically what we did are:
# * Removed urls, emojis, html tags and punctuations
# * Some basic helper reg compiled functions to clean text by removing urls, emojis, html tags and punctuation

# In[ ]:


# In a loop, it would be better to compile the regular expression first

url_reg = re.compile(r'https?://\S+|www\.\S+')

emoji_reg = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)

html_reg = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

punct_table = str.maketrans('', '', string.punctuation)

# Applying helper reg functions

train['text_clean'] = train['text'].apply(lambda x: url_reg.sub(r'',x))
train['text_clean'] = train['text_clean'].apply(lambda x: emoji_reg.sub(r'', x))
train['text_clean'] = train['text_clean'].apply(lambda x: html_reg.sub(r'', x))
train['text_clean'] = train['text_clean'].apply(lambda x: x.translate(punct_table))
train.head()


# # word lemmatizer
# In many languages, words appear in several inflected forms. For example, in English, the verb 'to walk' may appear as 'walk', 'walked', 'walks' or 'walking'. The base form, 'walk', that one might look up in a dictionary, is called the lemma for the word. The association of the base form with a part of speech is often called a lexeme of the word.
# 
# Lemmatisation is closely related to stemming. The difference is that a stemmer operates on a single word without knowledge of the context, and therefore cannot discriminate between words which have different meanings depending on part of speech. However, stemmers are typically easier to implement and run faster. The reduced "accuracy" may not matter for some applications. In fact, when used within information retrieval systems, stemming improves query recall accuracy, or true positive rate, when compared to lemmatisation. Nonetheless, stemming reduces precision, or the proportion of positively-labeled instances that are actually positive, for such systems.

# In[ ]:


# Tokenizing the tweet base texts.

train['text_clean']=train['text_clean'].str.lower().apply(word_tokenize).apply(nltk.tag.pos_tag) # Applying part of speech tags.


# Removing stopwords.

stop = set(stopwords.words('english'))

train['text_clean'] = train['text_clean'].apply(lambda x: [word for word in x if word not in stop])

# Converting part of speeches to wordnet format.

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


train['text_clean']= train['text_clean'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])

# Applying word lemmatizer.

wnl = WordNetLemmatizer()

train['text_clean']= train['text_clean'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])


train['text_clean']= train['text_clean'].apply(lambda x: ' '.join(x))

train.head()


# # Visualizing the processed data
# ## Tweet lengths
# 
# Let's start with the number of characters per tweet and compare if it's disaster related or not. It seems disaster tweets are longer than non disaster tweets in general. We can assume longer tweets are more likely for disasters but this is only an assumption and might be not true...¶

# In[ ]:


def plot_dist(col, sup_title, xlabel):
    fig, ax = plt.subplots(ncols=2,nrows=1,figsize=(18,6), sharey=True)
    
    for index, x in enumerate(['Non Disaster Tweets', 'Disaster Tweets']):
        sns.distplot(col.loc[train.target==index],ax=ax[index], color='#e74c3c')
        
        ax[index].set_xlabel(xlabel)
        ax[index].set_ylabel('Frequency')
        ax[index].set_title(x)
    
    fig.suptitle(sup_title, fontsize=24, va='baseline')
    fig.tight_layout()


# In[ ]:


plot_dist(col= train.text_clean.str.len(), sup_title= 'Characters Per Tweet', xlabel='Character Count')


# # Word Counts
# Ok let's check number of words per tweet now, they both look somewhat normally distributed, again disaster tweets seems to have slightly more words than non disaster ones. We might dig this deeper to get some more info in next part...

# In[ ]:


plot_dist(col= train.text_clean.str.split().map(lambda x: len(x)), sup_title= 'words Per Tweet', xlabel='Word count')


# # Mean Word Lengths
# This time we're gonna check if word complexity differs from tweet class. It looks like disaster tweets has longer words than non disaster ones in general. It's pretty visible which is good sign, yet again we can only assume at this stage...

# In[ ]:


plot_dist(train.text_clean.str.split().apply(lambda x: np.mean([len(i) for i in x])),'Mean Word Lengths','Word Lengths')


# # Most Common Words
# It's time to move to words themselves instead of their quantitative features. We start with most common words in both classes. I'd say it's pretty obvious if it's from disaster tweets or not. Disaster tweets has words like killed, news, bomb indicating disasters. Meanwhile non disaster ones looks like pretty generic.

# In[ ]:


def plot_ngram(ngram, sup_title):
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(ngram, ngram))
    df = vectorizer.fit_transform(train.text_clean)
    df = pd.DataFrame(df.toarray(), columns=vectorizer.get_feature_names_out())
    
    fig, ax = plt.subplots(ncols=2,nrows=1,figsize=(18,8))
    
    for index, xlabel in enumerate(['Non Disaster Tweets', 'Disaster Tweets']):
        
        df_class = df.loc[train.target==index].sum().sort_values(ascending=False).head(15)

        sns.barplot(x= df_class, y = df_class.index, ax=ax[index], palette='plasma')
        
        ax[index].set_xlabel('Count')
        ax[index].set_ylabel('Word')
        ax[index].set_title(xlabel)
    
    fig.suptitle(sup_title, fontsize=24, va='baseline')
    fig.tight_layout()


# In[ ]:


plot_ngram(1,'Most Common Unigrams')


# # Most Common Bigrams
# Let's have a look for bigrams this time, which they are sequences of adjacent two words. Again it's pretty obvious to seperate two classes if it's disaster related or not. There are some confusing bigrams in non disaster ones like body bag, emergency services etc. which needs deeper research but we'll leave it here since we got what we looking for in general.

# In[ ]:


plot_ngram(2,'Most Common Bigrams')


# # Most Common Trigrams
# Alright! Things are much clearer with sequences of 3 words. The confusing body bags were cross body bags (Who uses them in these days anyways!) which I found it pretty funny when I found the reason of the confusion. Anyways we can see disasters are highly seperable now from non disaster ones, which is great!

# In[ ]:


plot_ngram(3,'Most Common Trigrams')


# # Building Baseline Model

# In[ ]:


import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Dropout
from tensorflow.keras.layers import TextVectorization

from sklearn.model_selection import train_test_split


# In[ ]:


# separate train and test set

test = train.loc[train.target.isna()]
train = train.loc[train.target.notna()]


# In[ ]:


#A preprocessing layer which maps text features to integer sequences.

vectorizer= TextVectorization()
vectorizer.adapt(train.text_clean)
vectorizer(train.text_clean)

# vectorize train & test text data

train_vec= vectorizer(train.text_clean).numpy()
test_vec= vectorizer(test.text_clean).numpy()


# model

model = tf.keras.Sequential([
    Embedding(
        len(vectorizer.get_vocabulary()) + 1, # Integer. Size of the vocabulary, i.e. maximum integer index + 1.
        100,                                  # Integer. Dimension of the dense embedding.
        input_length= train_vec[0].shape[0]), # Length of input sequences
    Dropout(0.2),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')])

model.summary()


# In[ ]:


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy', 'AUC'])


# In[ ]:


# split training set into train test

X_train, X_test, y_train, y_test= train_test_split(train_vec, train.target.astype(int).values)

# classes weights for imbalance data
# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
total = 7613

weight_for_0 = (1 / 4342) * (total / 2.0)
weight_for_1 = (1 / 3271) * (total / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

# fit the model
history=model.fit(X_train, y_train,
                  #batch_size=7,
                  epochs=4,
                  validation_data=(X_test,y_test),
                  class_weight=class_weight)


# In[ ]:


# summarize history for accuracy

fig, ax= plt.subplots(nrows=1,ncols=2,figsize=(18,4),sharex=True)

for i, m in enumerate(['accuracy', 'loss']):
    ax[i].plot(history.history[m])
    ax[i].plot(history.history[f'val_{m}'])
    ax[i].set_title(f'model {m}')
    ax[i].set_ylabel(m)
    ax[i].set_xlabel('epoch')
    ax[i].legend(['train', 'test'], loc='upper left')


fig.show()


# # Submission

# In[ ]:


#Making our submission

sample_sub=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')


sample_sub['target'] = model.predict(test_vec).round().astype(int)

sample_sub.to_csv('submission.csv',index=False)

