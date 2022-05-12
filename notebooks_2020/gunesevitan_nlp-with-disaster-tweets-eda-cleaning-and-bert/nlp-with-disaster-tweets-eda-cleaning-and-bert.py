#!/usr/bin/env python
# coding: utf-8

# ## **0. Introduction and References**
# I decided to write this kernel when I first started learning about NLP. It is basically the things I learned documented in Kaggle Notebook format. It can be helpful for you if you are looking for **data analysis on competition data**, **feature engineering ideas for NLP**, **cleaning and text processing ideas**, **baseline BERT model** or **test set with labels**. If you have any idea that might improve this kernel, please be sure to comment, or fork and experiment as you like. If you don't understand any part, feel free to ask in the comment section.
# 
# This kernel includes codes and ideas from kernels below. If this kernel helps you, please upvote their work as well. 
# * [Simple Exploration Notebook - QIQC](https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc) by [@sudalairajkumar](https://www.kaggle.com/sudalairajkumar)
# * [How to: Preprocessing when using embeddings](https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings) by [@christofhenkel](https://www.kaggle.com/christofhenkel)
# * [Improve your Score with some Text Preprocessing](https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing) by [@theoviel](https://www.kaggle.com/theoviel)
# * [A Real Disaster - Leaked Label](https://www.kaggle.com/szelee/a-real-disaster-leaked-label) by [@szelee](https://www.kaggle.com/szelee)
# * [Disaster NLP: Keras BERT using TFHub](https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub) by [@xhlulu](https://www.kaggle.com/xhlulu)

# In[ ]:


get_ipython().system('wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py')


# In[ ]:


import gc
import re
import string
import operator
from collections import defaultdict

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import matplotlib.pyplot as plt
import seaborn as sns

import tokenization
from wordcloud import STOPWORDS

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

SEED = 1337


# In[ ]:


df_train = pd.read_csv('../input/nlp-getting-started/train.csv', dtype={'id': np.int16, 'target': np.int8})
df_test = pd.read_csv('../input/nlp-getting-started/test.csv', dtype={'id': np.int16})

print('Training Set Shape = {}'.format(df_train.shape))
print('Training Set Memory Usage = {:.2f} MB'.format(df_train.memory_usage().sum() / 1024**2))
print('Test Set Shape = {}'.format(df_test.shape))
print('Test Set Memory Usage = {:.2f} MB'.format(df_test.memory_usage().sum() / 1024**2))


# ## **1. Keyword and Location**

# ### **1.1 Missing Values**
# Both training and test set have same ratio of missing values in `keyword` and `location`.
# * **0.8%** of `keyword` is missing in both training and test set
# * **33%** of `location` is missing in both training and test set
# 
# Since missing value ratios between training and test set are too close, **they are most probably taken from the same sample**. Missing values in those features are filled with `no_keyword` and `no_location` respectively.

# In[ ]:


missing_cols = ['keyword', 'location']

fig, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)

sns.barplot(x=df_train[missing_cols].isnull().sum().index, y=df_train[missing_cols].isnull().sum().values, ax=axes[0])
sns.barplot(x=df_test[missing_cols].isnull().sum().index, y=df_test[missing_cols].isnull().sum().values, ax=axes[1])

axes[0].set_ylabel('Missing Value Count', size=15, labelpad=20)
axes[0].tick_params(axis='x', labelsize=15)
axes[0].tick_params(axis='y', labelsize=15)
axes[1].tick_params(axis='x', labelsize=15)
axes[1].tick_params(axis='y', labelsize=15)

axes[0].set_title('Training Set', fontsize=13)
axes[1].set_title('Test Set', fontsize=13)

plt.show()

for df in [df_train, df_test]:
    for col in ['keyword', 'location']:
        df[col] = df[col].fillna(f'no_{col}')


# ### **1.2 Cardinality and Target Distribution**
# Locations are not automatically generated, they are user inputs. That's why `location` is very dirty and there are too many unique values in it. It shouldn't be used as a feature.
# 
# Fortunately, there is signal in `keyword` because some of those words can only be used in one context. Keywords have very different tweet counts and target means. `keyword` can be used as a feature by itself or as a word added to the text. Every single keyword in training set exists in test set. If training and test set are from the same sample, it is also possible to use target encoding on `keyword`.

# In[ ]:


print(f'Number of unique values in keyword = {df_train["keyword"].nunique()} (Training) - {df_test["keyword"].nunique()} (Test)')
print(f'Number of unique values in location = {df_train["location"].nunique()} (Training) - {df_test["location"].nunique()} (Test)')


# In[ ]:


df_train['target_mean'] = df_train.groupby('keyword')['target'].transform('mean')

fig = plt.figure(figsize=(8, 72), dpi=100)

sns.countplot(y=df_train.sort_values(by='target_mean', ascending=False)['keyword'],
              hue=df_train.sort_values(by='target_mean', ascending=False)['target'])

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=12)
plt.legend(loc=1)
plt.title('Target Distribution in Keywords')

plt.show()

df_train.drop(columns=['target_mean'], inplace=True)


# ## **2. Meta Features**
# Distributions of meta features in classes and datasets can be helpful to identify disaster tweets. It looks like disaster tweets are written in a more formal way with longer words compared to non-disaster tweets because most of them are coming from news agencies. Non-disaster tweets have more typos than disaster tweets because they are coming from individual users. The meta features used for the analysis are;
# * `word_count` number of words in text
# * `unique_word_count` number of unique words in text
# * `stop_word_count` number of stop words in text
# * `url_count` number of urls in text
# * `mean_word_length` average character count in words
# * `char_count` number of characters in text
# * `punctuation_count` number of punctuations in text
# * `hashtag_count` number of hashtags (**#**) in text
# * `mention_count` number of mentions (**@**) in text

# In[ ]:


# word_count
df_train['word_count'] = df_train['text'].apply(lambda x: len(str(x).split()))
df_test['word_count'] = df_test['text'].apply(lambda x: len(str(x).split()))

# unique_word_count
df_train['unique_word_count'] = df_train['text'].apply(lambda x: len(set(str(x).split())))
df_test['unique_word_count'] = df_test['text'].apply(lambda x: len(set(str(x).split())))

# stop_word_count
df_train['stop_word_count'] = df_train['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
df_test['stop_word_count'] = df_test['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

# url_count
df_train['url_count'] = df_train['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))
df_test['url_count'] = df_test['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))

# mean_word_length
df_train['mean_word_length'] = df_train['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
df_test['mean_word_length'] = df_test['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

# char_count
df_train['char_count'] = df_train['text'].apply(lambda x: len(str(x)))
df_test['char_count'] = df_test['text'].apply(lambda x: len(str(x)))

# punctuation_count
df_train['punctuation_count'] = df_train['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
df_test['punctuation_count'] = df_test['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

# hashtag_count
df_train['hashtag_count'] = df_train['text'].apply(lambda x: len([c for c in str(x) if c == '#']))
df_test['hashtag_count'] = df_test['text'].apply(lambda x: len([c for c in str(x) if c == '#']))

# mention_count
df_train['mention_count'] = df_train['text'].apply(lambda x: len([c for c in str(x) if c == '@']))
df_test['mention_count'] = df_test['text'].apply(lambda x: len([c for c in str(x) if c == '@']))


# All of the meta features have very similar distributions in training and test set which also proves that training and test set are taken from the same sample.
# 
# All of the meta features have information about target as well, but some of them are not good enough such as `url_count`, `hashtag_count` and `mention_count`.
# 
# On the other hand, `word_count`, `unique_word_count`, `stop_word_count`, `mean_word_length`, `char_count`, `punctuation_count` have very different distributions for disaster and non-disaster tweets. Those features might be useful in models.

# In[ ]:


METAFEATURES = ['word_count', 'unique_word_count', 'stop_word_count', 'url_count', 'mean_word_length',
                'char_count', 'punctuation_count', 'hashtag_count', 'mention_count']
DISASTER_TWEETS = df_train['target'] == 1

fig, axes = plt.subplots(ncols=2, nrows=len(METAFEATURES), figsize=(20, 50), dpi=100)

for i, feature in enumerate(METAFEATURES):
    sns.distplot(df_train.loc[~DISASTER_TWEETS][feature], label='Not Disaster', ax=axes[i][0], color='green')
    sns.distplot(df_train.loc[DISASTER_TWEETS][feature], label='Disaster', ax=axes[i][0], color='red')

    sns.distplot(df_train[feature], label='Training', ax=axes[i][1])
    sns.distplot(df_test[feature], label='Test', ax=axes[i][1])
    
    for j in range(2):
        axes[i][j].set_xlabel('')
        axes[i][j].tick_params(axis='x', labelsize=12)
        axes[i][j].tick_params(axis='y', labelsize=12)
        axes[i][j].legend()
    
    axes[i][0].set_title(f'{feature} Target Distribution in Training Set', fontsize=13)
    axes[i][1].set_title(f'{feature} Training & Test Set Distribution', fontsize=13)

plt.show()


# ## **3. Target and N-grams**

# ### **3.1 Target**
# Class distributions are **57%** for **0** (Not Disaster) and **43%** for **1** (Disaster). Classes are almost equally separated so they don't require any stratification by `target` in cross-validation.

# In[ ]:


fig, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)
plt.tight_layout()

df_train.groupby('target').count()['id'].plot(kind='pie', ax=axes[0], labels=['Not Disaster (57%)', 'Disaster (43%)'])
sns.countplot(x=df_train['target'], hue=df_train['target'], ax=axes[1])

axes[0].set_ylabel('')
axes[1].set_ylabel('')
axes[1].set_xticklabels(['Not Disaster (4342)', 'Disaster (3271)'])
axes[0].tick_params(axis='x', labelsize=15)
axes[0].tick_params(axis='y', labelsize=15)
axes[1].tick_params(axis='x', labelsize=15)
axes[1].tick_params(axis='y', labelsize=15)

axes[0].set_title('Target Distribution in Training Set', fontsize=13)
axes[1].set_title('Target Count in Training Set', fontsize=13)

plt.show()


# In[ ]:


def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(' ') if token != '' if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [' '.join(ngram) for ngram in ngrams]

N = 100

# Unigrams
disaster_unigrams = defaultdict(int)
nondisaster_unigrams = defaultdict(int)

for tweet in df_train[DISASTER_TWEETS]['text']:
    for word in generate_ngrams(tweet):
        disaster_unigrams[word] += 1
        
for tweet in df_train[~DISASTER_TWEETS]['text']:
    for word in generate_ngrams(tweet):
        nondisaster_unigrams[word] += 1
        
df_disaster_unigrams = pd.DataFrame(sorted(disaster_unigrams.items(), key=lambda x: x[1])[::-1])
df_nondisaster_unigrams = pd.DataFrame(sorted(nondisaster_unigrams.items(), key=lambda x: x[1])[::-1])

# Bigrams
disaster_bigrams = defaultdict(int)
nondisaster_bigrams = defaultdict(int)

for tweet in df_train[DISASTER_TWEETS]['text']:
    for word in generate_ngrams(tweet, n_gram=2):
        disaster_bigrams[word] += 1
        
for tweet in df_train[~DISASTER_TWEETS]['text']:
    for word in generate_ngrams(tweet, n_gram=2):
        nondisaster_bigrams[word] += 1
        
df_disaster_bigrams = pd.DataFrame(sorted(disaster_bigrams.items(), key=lambda x: x[1])[::-1])
df_nondisaster_bigrams = pd.DataFrame(sorted(nondisaster_bigrams.items(), key=lambda x: x[1])[::-1])

# Trigrams
disaster_trigrams = defaultdict(int)
nondisaster_trigrams = defaultdict(int)

for tweet in df_train[DISASTER_TWEETS]['text']:
    for word in generate_ngrams(tweet, n_gram=3):
        disaster_trigrams[word] += 1
        
for tweet in df_train[~DISASTER_TWEETS]['text']:
    for word in generate_ngrams(tweet, n_gram=3):
        nondisaster_trigrams[word] += 1
        
df_disaster_trigrams = pd.DataFrame(sorted(disaster_trigrams.items(), key=lambda x: x[1])[::-1])
df_nondisaster_trigrams = pd.DataFrame(sorted(nondisaster_trigrams.items(), key=lambda x: x[1])[::-1])


# ### **3.2 Unigrams**
# Most common unigrams exist in **both classes** are mostly punctuations, stop words or numbers. It is better to clean them before modelling since they don't give much information about `target`.
# 
# Most common unigrams in **disaster** tweets are already giving information about disasters. It is very hard to use some of those words in other contexts.
# 
# Most common unigrams in **non-disaster** tweets are verbs. This makes sense because most of those sentences have informal active structure since they are coming from individual users.

# In[ ]:


fig, axes = plt.subplots(ncols=2, figsize=(18, 50), dpi=100)
plt.tight_layout()

sns.barplot(y=df_disaster_unigrams[0].values[:N], x=df_disaster_unigrams[1].values[:N], ax=axes[0], color='red')
sns.barplot(y=df_nondisaster_unigrams[0].values[:N], x=df_nondisaster_unigrams[1].values[:N], ax=axes[1], color='green')

for i in range(2):
    axes[i].spines['right'].set_visible(False)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].tick_params(axis='x', labelsize=13)
    axes[i].tick_params(axis='y', labelsize=13)

axes[0].set_title(f'Top {N} most common unigrams in Disaster Tweets', fontsize=15)
axes[1].set_title(f'Top {N} most common unigrams in Non-disaster Tweets', fontsize=15)

plt.show()


# ### **3.3 Bigrams**
# There are no common bigrams exist in **both classes**  because the context is clearer.
# 
# Most common bigrams in **disaster** tweets are giving more information about the disasters than unigrams, but punctuations have to be stripped from words.
# 
# Most common bigrams in **non-disaster** tweets are mostly about reddit or youtube, and they contain lots of punctuations. Those punctuations have to be cleaned out of words as well.

# In[ ]:


fig, axes = plt.subplots(ncols=2, figsize=(18, 50), dpi=100)
plt.tight_layout()

sns.barplot(y=df_disaster_bigrams[0].values[:N], x=df_disaster_bigrams[1].values[:N], ax=axes[0], color='red')
sns.barplot(y=df_nondisaster_bigrams[0].values[:N], x=df_nondisaster_bigrams[1].values[:N], ax=axes[1], color='green')

for i in range(2):
    axes[i].spines['right'].set_visible(False)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].tick_params(axis='x', labelsize=13)
    axes[i].tick_params(axis='y', labelsize=13)

axes[0].set_title(f'Top {N} most common bigrams in Disaster Tweets', fontsize=15)
axes[1].set_title(f'Top {N} most common bigrams in Non-disaster Tweets', fontsize=15)

plt.show()


# ### **3.4 Trigrams**
# There are no common trigrams exist in **both classes**  because the context is clearer.
# 
# Most common trigrams in **disaster** tweets are very similar to bigrams. They give lots of information about disasters, but they may not provide any additional information along with bigrams.
# 
# Most common trigrams in **non-disaster** tweets are also very similar to bigrams, and they contain even more punctuations.

# In[ ]:


fig, axes = plt.subplots(ncols=2, figsize=(20, 50), dpi=100)

sns.barplot(y=df_disaster_trigrams[0].values[:N], x=df_disaster_trigrams[1].values[:N], ax=axes[0], color='red')
sns.barplot(y=df_nondisaster_trigrams[0].values[:N], x=df_nondisaster_trigrams[1].values[:N], ax=axes[1], color='green')

for i in range(2):
    axes[i].spines['right'].set_visible(False)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].tick_params(axis='x', labelsize=13)
    axes[i].tick_params(axis='y', labelsize=11)

axes[0].set_title(f'Top {N} most common trigrams in Disaster Tweets', fontsize=15)
axes[1].set_title(f'Top {N} most common trigrams in Non-disaster Tweets', fontsize=15)

plt.show()


# ## **4. Embeddings and Text Cleaning**

# ### **4.1 Embeddings Coverage**
# When you have pre-trained embeddings, doing standard preprocessing steps  might not be a good idea because some of the valuable information can be lost. It is better to get vocabulary as close to embeddings as possible. In order to do that, train vocab and test vocab are created by counting the words in tweets.
# 
# Text cleaning is based on the embeddings below:
# * GloVe-300d-840B
# * FastText-Crawl-300d-2M

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nglove_embeddings = np.load('../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl', allow_pickle=True)\nfasttext_embeddings = np.load('../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl', allow_pickle=True)")


# Words in the intersection of vocab and embeddings are stored in `covered` along with their counts. Words in vocab that don't exist in embeddings are stored in `oov` along with their counts. `n_covered` and `n_oov` are total number of counts and they are used for calculating coverage percentages.
# 
# Both GloVe and FastText embeddings have more than **50%** vocabulary and **80%** text coverage without cleaning. GloVe and FastText coverage are very close but GloVe has slightly higher coverage.

# In[ ]:


def build_vocab(X):
    
    tweets = X.apply(lambda s: s.split()).values      
    vocab = {}
    
    for tweet in tweets:
        for word in tweet:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1                
    return vocab


def check_embeddings_coverage(X, embeddings):
    
    vocab = build_vocab(X)    
    
    covered = {}
    oov = {}    
    n_covered = 0
    n_oov = 0
    
    for word in vocab:
        try:
            covered[word] = embeddings[word]
            n_covered += vocab[word]
        except:
            oov[word] = vocab[word]
            n_oov += vocab[word]
            
    vocab_coverage = len(covered) / len(vocab)
    text_coverage = (n_covered / (n_covered + n_oov))
    
    sorted_oov = sorted(oov.items(), key=operator.itemgetter(1))[::-1]
    return sorted_oov, vocab_coverage, text_coverage

train_glove_oov, train_glove_vocab_coverage, train_glove_text_coverage = check_embeddings_coverage(df_train['text'], glove_embeddings)
test_glove_oov, test_glove_vocab_coverage, test_glove_text_coverage = check_embeddings_coverage(df_test['text'], glove_embeddings)
print('GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Training Set'.format(train_glove_vocab_coverage, train_glove_text_coverage))
print('GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Test Set'.format(test_glove_vocab_coverage, test_glove_text_coverage))

train_fasttext_oov, train_fasttext_vocab_coverage, train_fasttext_text_coverage = check_embeddings_coverage(df_train['text'], fasttext_embeddings)
test_fasttext_oov, test_fasttext_vocab_coverage, test_fasttext_text_coverage = check_embeddings_coverage(df_test['text'], fasttext_embeddings)
print('FastText Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Training Set'.format(train_fasttext_vocab_coverage, train_fasttext_text_coverage))
print('FastText Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Test Set'.format(test_fasttext_vocab_coverage, test_fasttext_text_coverage))


# ### **4.2 Text Cleaning**
# Tweets require lots of cleaning but it is inefficient to clean every single tweet because that would consume too much time. A general approach must be implemented for cleaning.
# 
# * The most common type of words that require cleaning in `oov` have punctuations at the start or end. Those words doesn't have embeddings because of the trailing punctuations. Punctuations `#`, `@`, `!`, `?`, `+`, `&`, `-`, `$`, `=`, `<`, `>`, `|`, `{`, `}`, `^`, `'`, `(`, `)`,`[`, `]`, `*`, `%`, `...`, `'`, `.`, `:`, `;` are separated from words
# * Special characters that are attached to words are removed completely
# * Contractions are expanded
# * Urls are removed
# * Character entity references are replaced with their actual symbols
# * Typos and slang are corrected, and informal abbreviations are written in their long forms
# * Some words are replaced with their acronyms and some words are grouped into one
# * Finally, hashtags and usernames contain lots of information about the context but they are written without spaces in between words so they don't have embeddings. Informational usernames and hashtags should be expanded but there are too many of them. I expanded as many as I could, but it takes too much time to run `clean` function after adding those replace calls.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef clean(tweet): \n            \n    # Special characters\n    tweet = re.sub(r"\\x89Û_", "", tweet)\n    tweet = re.sub(r"\\x89ÛÒ", "", tweet)\n    tweet = re.sub(r"\\x89ÛÓ", "", tweet)\n    tweet = re.sub(r"\\x89ÛÏWhen", "When", tweet)\n    tweet = re.sub(r"\\x89ÛÏ", "", tweet)\n    tweet = re.sub(r"China\\x89Ûªs", "China\'s", tweet)\n    tweet = re.sub(r"let\\x89Ûªs", "let\'s", tweet)\n    tweet = re.sub(r"\\x89Û÷", "", tweet)\n    tweet = re.sub(r"\\x89Ûª", "", tweet)\n    tweet = re.sub(r"\\x89Û\\x9d", "", tweet)\n    tweet = re.sub(r"å_", "", tweet)\n    tweet = re.sub(r"\\x89Û¢", "", tweet)\n    tweet = re.sub(r"\\x89Û¢åÊ", "", tweet)\n    tweet = re.sub(r"fromåÊwounds", "from wounds", tweet)\n    tweet = re.sub(r"åÊ", "", tweet)\n    tweet = re.sub(r"åÈ", "", tweet)\n    tweet = re.sub(r"JapÌ_n", "Japan", tweet)    \n    tweet = re.sub(r"Ì©", "e", tweet)\n    tweet = re.sub(r"å¨", "", tweet)\n    tweet = re.sub(r"SuruÌ¤", "Suruc", tweet)\n    tweet = re.sub(r"åÇ", "", tweet)\n    tweet = re.sub(r"å£3million", "3 million", tweet)\n    tweet = re.sub(r"åÀ", "", tweet)\n    \n    # Contractions\n    tweet = re.sub(r"he\'s", "he is", tweet)\n    tweet = re.sub(r"there\'s", "there is", tweet)\n    tweet = re.sub(r"We\'re", "We are", tweet)\n    tweet = re.sub(r"That\'s", "That is", tweet)\n    tweet = re.sub(r"won\'t", "will not", tweet)\n    tweet = re.sub(r"they\'re", "they are", tweet)\n    tweet = re.sub(r"Can\'t", "Cannot", tweet)\n    tweet = re.sub(r"wasn\'t", "was not", tweet)\n    tweet = re.sub(r"don\\x89Ûªt", "do not", tweet)\n    tweet = re.sub(r"aren\'t", "are not", tweet)\n    tweet = re.sub(r"isn\'t", "is not", tweet)\n    tweet = re.sub(r"What\'s", "What is", tweet)\n    tweet = re.sub(r"haven\'t", "have not", tweet)\n    tweet = re.sub(r"hasn\'t", "has not", tweet)\n    tweet = re.sub(r"There\'s", "There is", tweet)\n    tweet = re.sub(r"He\'s", "He is", tweet)\n    tweet = re.sub(r"It\'s", "It is", tweet)\n    tweet = re.sub(r"You\'re", "You are", tweet)\n    tweet = re.sub(r"I\'M", "I am", tweet)\n    tweet = re.sub(r"shouldn\'t", "should not", tweet)\n    tweet = re.sub(r"wouldn\'t", "would not", tweet)\n    tweet = re.sub(r"i\'m", "I am", tweet)\n    tweet = re.sub(r"I\\x89Ûªm", "I am", tweet)\n    tweet = re.sub(r"I\'m", "I am", tweet)\n    tweet = re.sub(r"Isn\'t", "is not", tweet)\n    tweet = re.sub(r"Here\'s", "Here is", tweet)\n    tweet = re.sub(r"you\'ve", "you have", tweet)\n    tweet = re.sub(r"you\\x89Ûªve", "you have", tweet)\n    tweet = re.sub(r"we\'re", "we are", tweet)\n    tweet = re.sub(r"what\'s", "what is", tweet)\n    tweet = re.sub(r"couldn\'t", "could not", tweet)\n    tweet = re.sub(r"we\'ve", "we have", tweet)\n    tweet = re.sub(r"it\\x89Ûªs", "it is", tweet)\n    tweet = re.sub(r"doesn\\x89Ûªt", "does not", tweet)\n    tweet = re.sub(r"It\\x89Ûªs", "It is", tweet)\n    tweet = re.sub(r"Here\\x89Ûªs", "Here is", tweet)\n    tweet = re.sub(r"who\'s", "who is", tweet)\n    tweet = re.sub(r"I\\x89Ûªve", "I have", tweet)\n    tweet = re.sub(r"y\'all", "you all", tweet)\n    tweet = re.sub(r"can\\x89Ûªt", "cannot", tweet)\n    tweet = re.sub(r"would\'ve", "would have", tweet)\n    tweet = re.sub(r"it\'ll", "it will", tweet)\n    tweet = re.sub(r"we\'ll", "we will", tweet)\n    tweet = re.sub(r"wouldn\\x89Ûªt", "would not", tweet)\n    tweet = re.sub(r"We\'ve", "We have", tweet)\n    tweet = re.sub(r"he\'ll", "he will", tweet)\n    tweet = re.sub(r"Y\'all", "You all", tweet)\n    tweet = re.sub(r"Weren\'t", "Were not", tweet)\n    tweet = re.sub(r"Didn\'t", "Did not", tweet)\n    tweet = re.sub(r"they\'ll", "they will", tweet)\n    tweet = re.sub(r"they\'d", "they would", tweet)\n    tweet = re.sub(r"DON\'T", "DO NOT", tweet)\n    tweet = re.sub(r"That\\x89Ûªs", "That is", tweet)\n    tweet = re.sub(r"they\'ve", "they have", tweet)\n    tweet = re.sub(r"i\'d", "I would", tweet)\n    tweet = re.sub(r"should\'ve", "should have", tweet)\n    tweet = re.sub(r"You\\x89Ûªre", "You are", tweet)\n    tweet = re.sub(r"where\'s", "where is", tweet)\n    tweet = re.sub(r"Don\\x89Ûªt", "Do not", tweet)\n    tweet = re.sub(r"we\'d", "we would", tweet)\n    tweet = re.sub(r"i\'ll", "I will", tweet)\n    tweet = re.sub(r"weren\'t", "were not", tweet)\n    tweet = re.sub(r"They\'re", "They are", tweet)\n    tweet = re.sub(r"Can\\x89Ûªt", "Cannot", tweet)\n    tweet = re.sub(r"you\\x89Ûªll", "you will", tweet)\n    tweet = re.sub(r"I\\x89Ûªd", "I would", tweet)\n    tweet = re.sub(r"let\'s", "let us", tweet)\n    tweet = re.sub(r"it\'s", "it is", tweet)\n    tweet = re.sub(r"can\'t", "cannot", tweet)\n    tweet = re.sub(r"don\'t", "do not", tweet)\n    tweet = re.sub(r"you\'re", "you are", tweet)\n    tweet = re.sub(r"i\'ve", "I have", tweet)\n    tweet = re.sub(r"that\'s", "that is", tweet)\n    tweet = re.sub(r"i\'ll", "I will", tweet)\n    tweet = re.sub(r"doesn\'t", "does not", tweet)\n    tweet = re.sub(r"i\'d", "I would", tweet)\n    tweet = re.sub(r"didn\'t", "did not", tweet)\n    tweet = re.sub(r"ain\'t", "am not", tweet)\n    tweet = re.sub(r"you\'ll", "you will", tweet)\n    tweet = re.sub(r"I\'ve", "I have", tweet)\n    tweet = re.sub(r"Don\'t", "do not", tweet)\n    tweet = re.sub(r"I\'ll", "I will", tweet)\n    tweet = re.sub(r"I\'d", "I would", tweet)\n    tweet = re.sub(r"Let\'s", "Let us", tweet)\n    tweet = re.sub(r"you\'d", "You would", tweet)\n    tweet = re.sub(r"It\'s", "It is", tweet)\n    tweet = re.sub(r"Ain\'t", "am not", tweet)\n    tweet = re.sub(r"Haven\'t", "Have not", tweet)\n    tweet = re.sub(r"Could\'ve", "Could have", tweet)\n    tweet = re.sub(r"youve", "you have", tweet)  \n    tweet = re.sub(r"donå«t", "do not", tweet)   \n            \n    # Character entity references\n    tweet = re.sub(r"&gt;", ">", tweet)\n    tweet = re.sub(r"&lt;", "<", tweet)\n    tweet = re.sub(r"&amp;", "&", tweet)\n    \n    # Typos, slang and informal abbreviations\n    tweet = re.sub(r"w/e", "whatever", tweet)\n    tweet = re.sub(r"w/", "with", tweet)\n    tweet = re.sub(r"USAgov", "USA government", tweet)\n    tweet = re.sub(r"recentlu", "recently", tweet)\n    tweet = re.sub(r"Ph0tos", "Photos", tweet)\n    tweet = re.sub(r"amirite", "am I right", tweet)\n    tweet = re.sub(r"exp0sed", "exposed", tweet)\n    tweet = re.sub(r"<3", "love", tweet)\n    tweet = re.sub(r"amageddon", "armageddon", tweet)\n    tweet = re.sub(r"Trfc", "Traffic", tweet)\n    tweet = re.sub(r"8/5/2015", "2015-08-05", tweet)\n    tweet = re.sub(r"WindStorm", "Wind Storm", tweet)\n    tweet = re.sub(r"8/6/2015", "2015-08-06", tweet)\n    tweet = re.sub(r"10:38PM", "10:38 PM", tweet)\n    tweet = re.sub(r"10:30pm", "10:30 PM", tweet)\n    tweet = re.sub(r"16yr", "16 year", tweet)\n    tweet = re.sub(r"lmao", "laughing my ass off", tweet)   \n    tweet = re.sub(r"TRAUMATISED", "traumatized", tweet)\n    \n    # Hashtags and usernames\n    tweet = re.sub(r"IranDeal", "Iran Deal", tweet)\n    tweet = re.sub(r"ArianaGrande", "Ariana Grande", tweet)\n    tweet = re.sub(r"camilacabello97", "camila cabello", tweet) \n    tweet = re.sub(r"RondaRousey", "Ronda Rousey", tweet)     \n    tweet = re.sub(r"MTVHottest", "MTV Hottest", tweet)\n    tweet = re.sub(r"TrapMusic", "Trap Music", tweet)\n    tweet = re.sub(r"ProphetMuhammad", "Prophet Muhammad", tweet)\n    tweet = re.sub(r"PantherAttack", "Panther Attack", tweet)\n    tweet = re.sub(r"StrategicPatience", "Strategic Patience", tweet)\n    tweet = re.sub(r"socialnews", "social news", tweet)\n    tweet = re.sub(r"NASAHurricane", "NASA Hurricane", tweet)\n    tweet = re.sub(r"onlinecommunities", "online communities", tweet)\n    tweet = re.sub(r"humanconsumption", "human consumption", tweet)\n    tweet = re.sub(r"Typhoon-Devastated", "Typhoon Devastated", tweet)\n    tweet = re.sub(r"Meat-Loving", "Meat Loving", tweet)\n    tweet = re.sub(r"facialabuse", "facial abuse", tweet)\n    tweet = re.sub(r"LakeCounty", "Lake County", tweet)\n    tweet = re.sub(r"BeingAuthor", "Being Author", tweet)\n    tweet = re.sub(r"withheavenly", "with heavenly", tweet)\n    tweet = re.sub(r"thankU", "thank you", tweet)\n    tweet = re.sub(r"iTunesMusic", "iTunes Music", tweet)\n    tweet = re.sub(r"OffensiveContent", "Offensive Content", tweet)\n    tweet = re.sub(r"WorstSummerJob", "Worst Summer Job", tweet)\n    tweet = re.sub(r"HarryBeCareful", "Harry Be Careful", tweet)\n    tweet = re.sub(r"NASASolarSystem", "NASA Solar System", tweet)\n    tweet = re.sub(r"animalrescue", "animal rescue", tweet)\n    tweet = re.sub(r"KurtSchlichter", "Kurt Schlichter", tweet)\n    tweet = re.sub(r"aRmageddon", "armageddon", tweet)\n    tweet = re.sub(r"Throwingknifes", "Throwing knives", tweet)\n    tweet = re.sub(r"GodsLove", "God\'s Love", tweet)\n    tweet = re.sub(r"bookboost", "book boost", tweet)\n    tweet = re.sub(r"ibooklove", "I book love", tweet)\n    tweet = re.sub(r"NestleIndia", "Nestle India", tweet)\n    tweet = re.sub(r"realDonaldTrump", "Donald Trump", tweet)\n    tweet = re.sub(r"DavidVonderhaar", "David Vonderhaar", tweet)\n    tweet = re.sub(r"CecilTheLion", "Cecil The Lion", tweet)\n    tweet = re.sub(r"weathernetwork", "weather network", tweet)\n    tweet = re.sub(r"withBioterrorism&use", "with Bioterrorism & use", tweet)\n    tweet = re.sub(r"Hostage&2", "Hostage & 2", tweet)\n    tweet = re.sub(r"GOPDebate", "GOP Debate", tweet)\n    tweet = re.sub(r"RickPerry", "Rick Perry", tweet)\n    tweet = re.sub(r"frontpage", "front page", tweet)\n    tweet = re.sub(r"NewsInTweets", "News In Tweets", tweet)\n    tweet = re.sub(r"ViralSpell", "Viral Spell", tweet)\n    tweet = re.sub(r"til_now", "until now", tweet)\n    tweet = re.sub(r"volcanoinRussia", "volcano in Russia", tweet)\n    tweet = re.sub(r"ZippedNews", "Zipped News", tweet)\n    tweet = re.sub(r"MicheleBachman", "Michele Bachman", tweet)\n    tweet = re.sub(r"53inch", "53 inch", tweet)\n    tweet = re.sub(r"KerrickTrial", "Kerrick Trial", tweet)\n    tweet = re.sub(r"abstorm", "Alberta Storm", tweet)\n    tweet = re.sub(r"Beyhive", "Beyonce hive", tweet)\n    tweet = re.sub(r"IDFire", "Idaho Fire", tweet)\n    tweet = re.sub(r"DETECTADO", "Detected", tweet)\n    tweet = re.sub(r"RockyFire", "Rocky Fire", tweet)\n    tweet = re.sub(r"Listen/Buy", "Listen / Buy", tweet)\n    tweet = re.sub(r"NickCannon", "Nick Cannon", tweet)\n    tweet = re.sub(r"FaroeIslands", "Faroe Islands", tweet)\n    tweet = re.sub(r"yycstorm", "Calgary Storm", tweet)\n    tweet = re.sub(r"IDPs:", "Internally Displaced People :", tweet)\n    tweet = re.sub(r"ArtistsUnited", "Artists United", tweet)\n    tweet = re.sub(r"ClaytonBryant", "Clayton Bryant", tweet)\n    tweet = re.sub(r"jimmyfallon", "jimmy fallon", tweet)\n    tweet = re.sub(r"justinbieber", "justin bieber", tweet)  \n    tweet = re.sub(r"UTC2015", "UTC 2015", tweet)\n    tweet = re.sub(r"Time2015", "Time 2015", tweet)\n    tweet = re.sub(r"djicemoon", "dj icemoon", tweet)\n    tweet = re.sub(r"LivingSafely", "Living Safely", tweet)\n    tweet = re.sub(r"FIFA16", "Fifa 2016", tweet)\n    tweet = re.sub(r"thisiswhywecanthavenicethings", "this is why we cannot have nice things", tweet)\n    tweet = re.sub(r"bbcnews", "bbc news", tweet)\n    tweet = re.sub(r"UndergroundRailraod", "Underground Railraod", tweet)\n    tweet = re.sub(r"c4news", "c4 news", tweet)\n    tweet = re.sub(r"OBLITERATION", "obliteration", tweet)\n    tweet = re.sub(r"MUDSLIDE", "mudslide", tweet)\n    tweet = re.sub(r"NoSurrender", "No Surrender", tweet)\n    tweet = re.sub(r"NotExplained", "Not Explained", tweet)\n    tweet = re.sub(r"greatbritishbakeoff", "great british bake off", tweet)\n    tweet = re.sub(r"LondonFire", "London Fire", tweet)\n    tweet = re.sub(r"KOTAWeather", "KOTA Weather", tweet)\n    tweet = re.sub(r"LuchaUnderground", "Lucha Underground", tweet)\n    tweet = re.sub(r"KOIN6News", "KOIN 6 News", tweet)\n    tweet = re.sub(r"LiveOnK2", "Live On K2", tweet)\n    tweet = re.sub(r"9NewsGoldCoast", "9 News Gold Coast", tweet)\n    tweet = re.sub(r"nikeplus", "nike plus", tweet)\n    tweet = re.sub(r"david_cameron", "David Cameron", tweet)\n    tweet = re.sub(r"peterjukes", "Peter Jukes", tweet)\n    tweet = re.sub(r"JamesMelville", "James Melville", tweet)\n    tweet = re.sub(r"megynkelly", "Megyn Kelly", tweet)\n    tweet = re.sub(r"cnewslive", "C News Live", tweet)\n    tweet = re.sub(r"JamaicaObserver", "Jamaica Observer", tweet)\n    tweet = re.sub(r"TweetLikeItsSeptember11th2001", "Tweet like it is september 11th 2001", tweet)\n    tweet = re.sub(r"cbplawyers", "cbp lawyers", tweet)\n    tweet = re.sub(r"fewmoretweets", "few more tweets", tweet)\n    tweet = re.sub(r"BlackLivesMatter", "Black Lives Matter", tweet)\n    tweet = re.sub(r"cjoyner", "Chris Joyner", tweet)\n    tweet = re.sub(r"ENGvAUS", "England vs Australia", tweet)\n    tweet = re.sub(r"ScottWalker", "Scott Walker", tweet)\n    tweet = re.sub(r"MikeParrActor", "Michael Parr", tweet)\n    tweet = re.sub(r"4PlayThursdays", "Foreplay Thursdays", tweet)\n    tweet = re.sub(r"TGF2015", "Tontitown Grape Festival", tweet)\n    tweet = re.sub(r"realmandyrain", "Mandy Rain", tweet)\n    tweet = re.sub(r"GraysonDolan", "Grayson Dolan", tweet)\n    tweet = re.sub(r"ApolloBrown", "Apollo Brown", tweet)\n    tweet = re.sub(r"saddlebrooke", "Saddlebrooke", tweet)\n    tweet = re.sub(r"TontitownGrape", "Tontitown Grape", tweet)\n    tweet = re.sub(r"AbbsWinston", "Abbs Winston", tweet)\n    tweet = re.sub(r"ShaunKing", "Shaun King", tweet)\n    tweet = re.sub(r"MeekMill", "Meek Mill", tweet)\n    tweet = re.sub(r"TornadoGiveaway", "Tornado Giveaway", tweet)\n    tweet = re.sub(r"GRupdates", "GR updates", tweet)\n    tweet = re.sub(r"SouthDowns", "South Downs", tweet)\n    tweet = re.sub(r"braininjury", "brain injury", tweet)\n    tweet = re.sub(r"auspol", "Australian politics", tweet)\n    tweet = re.sub(r"PlannedParenthood", "Planned Parenthood", tweet)\n    tweet = re.sub(r"calgaryweather", "Calgary Weather", tweet)\n    tweet = re.sub(r"weallheartonedirection", "we all heart one direction", tweet)\n    tweet = re.sub(r"edsheeran", "Ed Sheeran", tweet)\n    tweet = re.sub(r"TrueHeroes", "True Heroes", tweet)\n    tweet = re.sub(r"S3XLEAK", "sex leak", tweet)\n    tweet = re.sub(r"ComplexMag", "Complex Magazine", tweet)\n    tweet = re.sub(r"TheAdvocateMag", "The Advocate Magazine", tweet)\n    tweet = re.sub(r"CityofCalgary", "City of Calgary", tweet)\n    tweet = re.sub(r"EbolaOutbreak", "Ebola Outbreak", tweet)\n    tweet = re.sub(r"SummerFate", "Summer Fate", tweet)\n    tweet = re.sub(r"RAmag", "Royal Academy Magazine", tweet)\n    tweet = re.sub(r"offers2go", "offers to go", tweet)\n    tweet = re.sub(r"foodscare", "food scare", tweet)\n    tweet = re.sub(r"MNPDNashville", "Metropolitan Nashville Police Department", tweet)\n    tweet = re.sub(r"TfLBusAlerts", "TfL Bus Alerts", tweet)\n    tweet = re.sub(r"GamerGate", "Gamer Gate", tweet)\n    tweet = re.sub(r"IHHen", "Humanitarian Relief", tweet)\n    tweet = re.sub(r"spinningbot", "spinning bot", tweet)\n    tweet = re.sub(r"ModiMinistry", "Modi Ministry", tweet)\n    tweet = re.sub(r"TAXIWAYS", "taxi ways", tweet)\n    tweet = re.sub(r"Calum5SOS", "Calum Hood", tweet)\n    tweet = re.sub(r"po_st", "po.st", tweet)\n    tweet = re.sub(r"scoopit", "scoop.it", tweet)\n    tweet = re.sub(r"UltimaLucha", "Ultima Lucha", tweet)\n    tweet = re.sub(r"JonathanFerrell", "Jonathan Ferrell", tweet)\n    tweet = re.sub(r"aria_ahrary", "Aria Ahrary", tweet)\n    tweet = re.sub(r"rapidcity", "Rapid City", tweet)\n    tweet = re.sub(r"OutBid", "outbid", tweet)\n    tweet = re.sub(r"lavenderpoetrycafe", "lavender poetry cafe", tweet)\n    tweet = re.sub(r"EudryLantiqua", "Eudry Lantiqua", tweet)\n    tweet = re.sub(r"15PM", "15 PM", tweet)\n    tweet = re.sub(r"OriginalFunko", "Funko", tweet)\n    tweet = re.sub(r"rightwaystan", "Richard Tan", tweet)\n    tweet = re.sub(r"CindyNoonan", "Cindy Noonan", tweet)\n    tweet = re.sub(r"RT_America", "RT America", tweet)\n    tweet = re.sub(r"narendramodi", "Narendra Modi", tweet)\n    tweet = re.sub(r"BakeOffFriends", "Bake Off Friends", tweet)\n    tweet = re.sub(r"TeamHendrick", "Hendrick Motorsports", tweet)\n    tweet = re.sub(r"alexbelloli", "Alex Belloli", tweet)\n    tweet = re.sub(r"itsjustinstuart", "Justin Stuart", tweet)\n    tweet = re.sub(r"gunsense", "gun sense", tweet)\n    tweet = re.sub(r"DebateQuestionsWeWantToHear", "debate questions we want to hear", tweet)\n    tweet = re.sub(r"RoyalCarribean", "Royal Carribean", tweet)\n    tweet = re.sub(r"samanthaturne19", "Samantha Turner", tweet)\n    tweet = re.sub(r"JonVoyage", "Jon Stewart", tweet)\n    tweet = re.sub(r"renew911health", "renew 911 health", tweet)\n    tweet = re.sub(r"SuryaRay", "Surya Ray", tweet)\n    tweet = re.sub(r"pattonoswalt", "Patton Oswalt", tweet)\n    tweet = re.sub(r"minhazmerchant", "Minhaz Merchant", tweet)\n    tweet = re.sub(r"TLVFaces", "Israel Diaspora Coalition", tweet)\n    tweet = re.sub(r"pmarca", "Marc Andreessen", tweet)\n    tweet = re.sub(r"pdx911", "Portland Police", tweet)\n    tweet = re.sub(r"jamaicaplain", "Jamaica Plain", tweet)\n    tweet = re.sub(r"Japton", "Arkansas", tweet)\n    tweet = re.sub(r"RouteComplex", "Route Complex", tweet)\n    tweet = re.sub(r"INSubcontinent", "Indian Subcontinent", tweet)\n    tweet = re.sub(r"NJTurnpike", "New Jersey Turnpike", tweet)\n    tweet = re.sub(r"Politifiact", "PolitiFact", tweet)\n    tweet = re.sub(r"Hiroshima70", "Hiroshima", tweet)\n    tweet = re.sub(r"GMMBC", "Greater Mt Moriah Baptist Church", tweet)\n    tweet = re.sub(r"versethe", "verse the", tweet)\n    tweet = re.sub(r"TubeStrike", "Tube Strike", tweet)\n    tweet = re.sub(r"MissionHills", "Mission Hills", tweet)\n    tweet = re.sub(r"ProtectDenaliWolves", "Protect Denali Wolves", tweet)\n    tweet = re.sub(r"NANKANA", "Nankana", tweet)\n    tweet = re.sub(r"SAHIB", "Sahib", tweet)\n    tweet = re.sub(r"PAKPATTAN", "Pakpattan", tweet)\n    tweet = re.sub(r"Newz_Sacramento", "News Sacramento", tweet)\n    tweet = re.sub(r"gofundme", "go fund me", tweet)\n    tweet = re.sub(r"pmharper", "Stephen Harper", tweet)\n    tweet = re.sub(r"IvanBerroa", "Ivan Berroa", tweet)\n    tweet = re.sub(r"LosDelSonido", "Los Del Sonido", tweet)\n    tweet = re.sub(r"bancodeseries", "banco de series", tweet)\n    tweet = re.sub(r"timkaine", "Tim Kaine", tweet)\n    tweet = re.sub(r"IdentityTheft", "Identity Theft", tweet)\n    tweet = re.sub(r"AllLivesMatter", "All Lives Matter", tweet)\n    tweet = re.sub(r"mishacollins", "Misha Collins", tweet)\n    tweet = re.sub(r"BillNeelyNBC", "Bill Neely", tweet)\n    tweet = re.sub(r"BeClearOnCancer", "be clear on cancer", tweet)\n    tweet = re.sub(r"Kowing", "Knowing", tweet)\n    tweet = re.sub(r"ScreamQueens", "Scream Queens", tweet)\n    tweet = re.sub(r"AskCharley", "Ask Charley", tweet)\n    tweet = re.sub(r"BlizzHeroes", "Heroes of the Storm", tweet)\n    tweet = re.sub(r"BradleyBrad47", "Bradley Brad", tweet)\n    tweet = re.sub(r"HannaPH", "Typhoon Hanna", tweet)\n    tweet = re.sub(r"meinlcymbals", "MEINL Cymbals", tweet)\n    tweet = re.sub(r"Ptbo", "Peterborough", tweet)\n    tweet = re.sub(r"cnnbrk", "CNN Breaking News", tweet)\n    tweet = re.sub(r"IndianNews", "Indian News", tweet)\n    tweet = re.sub(r"savebees", "save bees", tweet)\n    tweet = re.sub(r"GreenHarvard", "Green Harvard", tweet)\n    tweet = re.sub(r"StandwithPP", "Stand with planned parenthood", tweet)\n    tweet = re.sub(r"hermancranston", "Herman Cranston", tweet)\n    tweet = re.sub(r"WMUR9", "WMUR-TV", tweet)\n    tweet = re.sub(r"RockBottomRadFM", "Rock Bottom Radio", tweet)\n    tweet = re.sub(r"ameenshaikh3", "Ameen Shaikh", tweet)\n    tweet = re.sub(r"ProSyn", "Project Syndicate", tweet)\n    tweet = re.sub(r"Daesh", "ISIS", tweet)\n    tweet = re.sub(r"s2g", "swear to god", tweet)\n    tweet = re.sub(r"listenlive", "listen live", tweet)\n    tweet = re.sub(r"CDCgov", "Centers for Disease Control and Prevention", tweet)\n    tweet = re.sub(r"FoxNew", "Fox News", tweet)\n    tweet = re.sub(r"CBSBigBrother", "Big Brother", tweet)\n    tweet = re.sub(r"JulieDiCaro", "Julie DiCaro", tweet)\n    tweet = re.sub(r"theadvocatemag", "The Advocate Magazine", tweet)\n    tweet = re.sub(r"RohnertParkDPS", "Rohnert Park Police Department", tweet)\n    tweet = re.sub(r"THISIZBWRIGHT", "Bonnie Wright", tweet)\n    tweet = re.sub(r"Popularmmos", "Popular MMOs", tweet)\n    tweet = re.sub(r"WildHorses", "Wild Horses", tweet)\n    tweet = re.sub(r"FantasticFour", "Fantastic Four", tweet)\n    tweet = re.sub(r"HORNDALE", "Horndale", tweet)\n    tweet = re.sub(r"PINER", "Piner", tweet)\n    tweet = re.sub(r"BathAndNorthEastSomerset", "Bath and North East Somerset", tweet)\n    tweet = re.sub(r"thatswhatfriendsarefor", "that is what friends are for", tweet)\n    tweet = re.sub(r"residualincome", "residual income", tweet)\n    tweet = re.sub(r"YahooNewsDigest", "Yahoo News Digest", tweet)\n    tweet = re.sub(r"MalaysiaAirlines", "Malaysia Airlines", tweet)\n    tweet = re.sub(r"AmazonDeals", "Amazon Deals", tweet)\n    tweet = re.sub(r"MissCharleyWebb", "Charley Webb", tweet)\n    tweet = re.sub(r"shoalstraffic", "shoals traffic", tweet)\n    tweet = re.sub(r"GeorgeFoster72", "George Foster", tweet)\n    tweet = re.sub(r"pop2015", "pop 2015", tweet)\n    tweet = re.sub(r"_PokemonCards_", "Pokemon Cards", tweet)\n    tweet = re.sub(r"DianneG", "Dianne Gallagher", tweet)\n    tweet = re.sub(r"KashmirConflict", "Kashmir Conflict", tweet)\n    tweet = re.sub(r"BritishBakeOff", "British Bake Off", tweet)\n    tweet = re.sub(r"FreeKashmir", "Free Kashmir", tweet)\n    tweet = re.sub(r"mattmosley", "Matt Mosley", tweet)\n    tweet = re.sub(r"BishopFred", "Bishop Fred", tweet)\n    tweet = re.sub(r"EndConflict", "End Conflict", tweet)\n    tweet = re.sub(r"EndOccupation", "End Occupation", tweet)\n    tweet = re.sub(r"UNHEALED", "unhealed", tweet)\n    tweet = re.sub(r"CharlesDagnall", "Charles Dagnall", tweet)\n    tweet = re.sub(r"Latestnews", "Latest news", tweet)\n    tweet = re.sub(r"KindleCountdown", "Kindle Countdown", tweet)\n    tweet = re.sub(r"NoMoreHandouts", "No More Handouts", tweet)\n    tweet = re.sub(r"datingtips", "dating tips", tweet)\n    tweet = re.sub(r"charlesadler", "Charles Adler", tweet)\n    tweet = re.sub(r"twia", "Texas Windstorm Insurance Association", tweet)\n    tweet = re.sub(r"txlege", "Texas Legislature", tweet)\n    tweet = re.sub(r"WindstormInsurer", "Windstorm Insurer", tweet)\n    tweet = re.sub(r"Newss", "News", tweet)\n    tweet = re.sub(r"hempoil", "hemp oil", tweet)\n    tweet = re.sub(r"CommoditiesAre", "Commodities are", tweet)\n    tweet = re.sub(r"tubestrike", "tube strike", tweet)\n    tweet = re.sub(r"JoeNBC", "Joe Scarborough", tweet)\n    tweet = re.sub(r"LiteraryCakes", "Literary Cakes", tweet)\n    tweet = re.sub(r"TI5", "The International 5", tweet)\n    tweet = re.sub(r"thehill", "the hill", tweet)\n    tweet = re.sub(r"3others", "3 others", tweet)\n    tweet = re.sub(r"stighefootball", "Sam Tighe", tweet)\n    tweet = re.sub(r"whatstheimportantvideo", "what is the important video", tweet)\n    tweet = re.sub(r"ClaudioMeloni", "Claudio Meloni", tweet)\n    tweet = re.sub(r"DukeSkywalker", "Duke Skywalker", tweet)\n    tweet = re.sub(r"carsonmwr", "Fort Carson", tweet)\n    tweet = re.sub(r"offdishduty", "off dish duty", tweet)\n    tweet = re.sub(r"andword", "and word", tweet)\n    tweet = re.sub(r"rhodeisland", "Rhode Island", tweet)\n    tweet = re.sub(r"easternoregon", "Eastern Oregon", tweet)\n    tweet = re.sub(r"WAwildfire", "Washington Wildfire", tweet)\n    tweet = re.sub(r"fingerrockfire", "Finger Rock Fire", tweet)\n    tweet = re.sub(r"57am", "57 am", tweet)\n    tweet = re.sub(r"fingerrockfire", "Finger Rock Fire", tweet)\n    tweet = re.sub(r"JacobHoggard", "Jacob Hoggard", tweet)\n    tweet = re.sub(r"newnewnew", "new new new", tweet)\n    tweet = re.sub(r"under50", "under 50", tweet)\n    tweet = re.sub(r"getitbeforeitsgone", "get it before it is gone", tweet)\n    tweet = re.sub(r"freshoutofthebox", "fresh out of the box", tweet)\n    tweet = re.sub(r"amwriting", "am writing", tweet)\n    tweet = re.sub(r"Bokoharm", "Boko Haram", tweet)\n    tweet = re.sub(r"Nowlike", "Now like", tweet)\n    tweet = re.sub(r"seasonfrom", "season from", tweet)\n    tweet = re.sub(r"epicente", "epicenter", tweet)\n    tweet = re.sub(r"epicenterr", "epicenter", tweet)\n    tweet = re.sub(r"sicklife", "sick life", tweet)\n    tweet = re.sub(r"yycweather", "Calgary Weather", tweet)\n    tweet = re.sub(r"calgarysun", "Calgary Sun", tweet)\n    tweet = re.sub(r"approachng", "approaching", tweet)\n    tweet = re.sub(r"evng", "evening", tweet)\n    tweet = re.sub(r"Sumthng", "something", tweet)\n    tweet = re.sub(r"EllenPompeo", "Ellen Pompeo", tweet)\n    tweet = re.sub(r"shondarhimes", "Shonda Rhimes", tweet)\n    tweet = re.sub(r"ABCNetwork", "ABC Network", tweet)\n    tweet = re.sub(r"SushmaSwaraj", "Sushma Swaraj", tweet)\n    tweet = re.sub(r"pray4japan", "Pray for Japan", tweet)\n    tweet = re.sub(r"hope4japan", "Hope for Japan", tweet)\n    tweet = re.sub(r"Illusionimagess", "Illusion images", tweet)\n    tweet = re.sub(r"SummerUnderTheStars", "Summer Under The Stars", tweet)\n    tweet = re.sub(r"ShallWeDance", "Shall We Dance", tweet)\n    tweet = re.sub(r"TCMParty", "TCM Party", tweet)\n    tweet = re.sub(r"marijuananews", "marijuana news", tweet)\n    tweet = re.sub(r"onbeingwithKristaTippett", "on being with Krista Tippett", tweet)\n    tweet = re.sub(r"Beingtweets", "Being tweets", tweet)\n    tweet = re.sub(r"newauthors", "new authors", tweet)\n    tweet = re.sub(r"remedyyyy", "remedy", tweet)\n    tweet = re.sub(r"44PM", "44 PM", tweet)\n    tweet = re.sub(r"HeadlinesApp", "Headlines App", tweet)\n    tweet = re.sub(r"40PM", "40 PM", tweet)\n    tweet = re.sub(r"myswc", "Severe Weather Center", tweet)\n    tweet = re.sub(r"ithats", "that is", tweet)\n    tweet = re.sub(r"icouldsitinthismomentforever", "I could sit in this moment forever", tweet)\n    tweet = re.sub(r"FatLoss", "Fat Loss", tweet)\n    tweet = re.sub(r"02PM", "02 PM", tweet)\n    tweet = re.sub(r"MetroFmTalk", "Metro Fm Talk", tweet)\n    tweet = re.sub(r"Bstrd", "bastard", tweet)\n    tweet = re.sub(r"bldy", "bloody", tweet)\n    tweet = re.sub(r"MetrofmTalk", "Metro Fm Talk", tweet)\n    tweet = re.sub(r"terrorismturn", "terrorism turn", tweet)\n    tweet = re.sub(r"BBCNewsAsia", "BBC News Asia", tweet)\n    tweet = re.sub(r"BehindTheScenes", "Behind The Scenes", tweet)\n    tweet = re.sub(r"GeorgeTakei", "George Takei", tweet)\n    tweet = re.sub(r"WomensWeeklyMag", "Womens Weekly Magazine", tweet)\n    tweet = re.sub(r"SurvivorsGuidetoEarth", "Survivors Guide to Earth", tweet)\n    tweet = re.sub(r"incubusband", "incubus band", tweet)\n    tweet = re.sub(r"Babypicturethis", "Baby picture this", tweet)\n    tweet = re.sub(r"BombEffects", "Bomb Effects", tweet)\n    tweet = re.sub(r"win10", "Windows 10", tweet)\n    tweet = re.sub(r"idkidk", "I do not know I do not know", tweet)\n    tweet = re.sub(r"TheWalkingDead", "The Walking Dead", tweet)\n    tweet = re.sub(r"amyschumer", "Amy Schumer", tweet)\n    tweet = re.sub(r"crewlist", "crew list", tweet)\n    tweet = re.sub(r"Erdogans", "Erdogan", tweet)\n    tweet = re.sub(r"BBCLive", "BBC Live", tweet)\n    tweet = re.sub(r"TonyAbbottMHR", "Tony Abbott", tweet)\n    tweet = re.sub(r"paulmyerscough", "Paul Myerscough", tweet)\n    tweet = re.sub(r"georgegallagher", "George Gallagher", tweet)\n    tweet = re.sub(r"JimmieJohnson", "Jimmie Johnson", tweet)\n    tweet = re.sub(r"pctool", "pc tool", tweet)\n    tweet = re.sub(r"DoingHashtagsRight", "Doing Hashtags Right", tweet)\n    tweet = re.sub(r"ThrowbackThursday", "Throwback Thursday", tweet)\n    tweet = re.sub(r"SnowBackSunday", "Snowback Sunday", tweet)\n    tweet = re.sub(r"LakeEffect", "Lake Effect", tweet)\n    tweet = re.sub(r"RTphotographyUK", "Richard Thomas Photography UK", tweet)\n    tweet = re.sub(r"BigBang_CBS", "Big Bang CBS", tweet)\n    tweet = re.sub(r"writerslife", "writers life", tweet)\n    tweet = re.sub(r"NaturalBirth", "Natural Birth", tweet)\n    tweet = re.sub(r"UnusualWords", "Unusual Words", tweet)\n    tweet = re.sub(r"wizkhalifa", "Wiz Khalifa", tweet)\n    tweet = re.sub(r"acreativedc", "a creative DC", tweet)\n    tweet = re.sub(r"vscodc", "vsco DC", tweet)\n    tweet = re.sub(r"VSCOcam", "vsco camera", tweet)\n    tweet = re.sub(r"TheBEACHDC", "The beach DC", tweet)\n    tweet = re.sub(r"buildingmuseum", "building museum", tweet)\n    tweet = re.sub(r"WorldOil", "World Oil", tweet)\n    tweet = re.sub(r"redwedding", "red wedding", tweet)\n    tweet = re.sub(r"AmazingRaceCanada", "Amazing Race Canada", tweet)\n    tweet = re.sub(r"WakeUpAmerica", "Wake Up America", tweet)\n    tweet = re.sub(r"\\\\Allahuakbar\\\\", "Allahu Akbar", tweet)\n    tweet = re.sub(r"bleased", "blessed", tweet)\n    tweet = re.sub(r"nigeriantribune", "Nigerian Tribune", tweet)\n    tweet = re.sub(r"HIDEO_KOJIMA_EN", "Hideo Kojima", tweet)\n    tweet = re.sub(r"FusionFestival", "Fusion Festival", tweet)\n    tweet = re.sub(r"50Mixed", "50 Mixed", tweet)\n    tweet = re.sub(r"NoAgenda", "No Agenda", tweet)\n    tweet = re.sub(r"WhiteGenocide", "White Genocide", tweet)\n    tweet = re.sub(r"dirtylying", "dirty lying", tweet)\n    tweet = re.sub(r"SyrianRefugees", "Syrian Refugees", tweet)\n    tweet = re.sub(r"changetheworld", "change the world", tweet)\n    tweet = re.sub(r"Ebolacase", "Ebola case", tweet)\n    tweet = re.sub(r"mcgtech", "mcg technologies", tweet)\n    tweet = re.sub(r"withweapons", "with weapons", tweet)\n    tweet = re.sub(r"advancedwarfare", "advanced warfare", tweet)\n    tweet = re.sub(r"letsFootball", "let us Football", tweet)\n    tweet = re.sub(r"LateNiteMix", "late night mix", tweet)\n    tweet = re.sub(r"PhilCollinsFeed", "Phil Collins", tweet)\n    tweet = re.sub(r"RudyHavenstein", "Rudy Havenstein", tweet)\n    tweet = re.sub(r"22PM", "22 PM", tweet)\n    tweet = re.sub(r"54am", "54 AM", tweet)\n    tweet = re.sub(r"38am", "38 AM", tweet)\n    tweet = re.sub(r"OldFolkExplainStuff", "Old Folk Explain Stuff", tweet)\n    tweet = re.sub(r"BlacklivesMatter", "Black Lives Matter", tweet)\n    tweet = re.sub(r"InsaneLimits", "Insane Limits", tweet)\n    tweet = re.sub(r"youcantsitwithus", "you cannot sit with us", tweet)\n    tweet = re.sub(r"2k15", "2015", tweet)\n    tweet = re.sub(r"TheIran", "Iran", tweet)\n    tweet = re.sub(r"JimmyFallon", "Jimmy Fallon", tweet)\n    tweet = re.sub(r"AlbertBrooks", "Albert Brooks", tweet)\n    tweet = re.sub(r"defense_news", "defense news", tweet)\n    tweet = re.sub(r"nuclearrcSA", "Nuclear Risk Control Self Assessment", tweet)\n    tweet = re.sub(r"Auspol", "Australia Politics", tweet)\n    tweet = re.sub(r"NuclearPower", "Nuclear Power", tweet)\n    tweet = re.sub(r"WhiteTerrorism", "White Terrorism", tweet)\n    tweet = re.sub(r"truthfrequencyradio", "Truth Frequency Radio", tweet)\n    tweet = re.sub(r"ErasureIsNotEquality", "Erasure is not equality", tweet)\n    tweet = re.sub(r"ProBonoNews", "Pro Bono News", tweet)\n    tweet = re.sub(r"JakartaPost", "Jakarta Post", tweet)\n    tweet = re.sub(r"toopainful", "too painful", tweet)\n    tweet = re.sub(r"melindahaunton", "Melinda Haunton", tweet)\n    tweet = re.sub(r"NoNukes", "No Nukes", tweet)\n    tweet = re.sub(r"curryspcworld", "Currys PC World", tweet)\n    tweet = re.sub(r"ineedcake", "I need cake", tweet)\n    tweet = re.sub(r"blackforestgateau", "black forest gateau", tweet)\n    tweet = re.sub(r"BBCOne", "BBC One", tweet)\n    tweet = re.sub(r"AlexxPage", "Alex Page", tweet)\n    tweet = re.sub(r"jonathanserrie", "Jonathan Serrie", tweet)\n    tweet = re.sub(r"SocialJerkBlog", "Social Jerk Blog", tweet)\n    tweet = re.sub(r"ChelseaVPeretti", "Chelsea Peretti", tweet)\n    tweet = re.sub(r"irongiant", "iron giant", tweet)\n    tweet = re.sub(r"RonFunches", "Ron Funches", tweet)\n    tweet = re.sub(r"TimCook", "Tim Cook", tweet)\n    tweet = re.sub(r"sebastianstanisaliveandwell", "Sebastian Stan is alive and well", tweet)\n    tweet = re.sub(r"Madsummer", "Mad summer", tweet)\n    tweet = re.sub(r"NowYouKnow", "Now you know", tweet)\n    tweet = re.sub(r"concertphotography", "concert photography", tweet)\n    tweet = re.sub(r"TomLandry", "Tom Landry", tweet)\n    tweet = re.sub(r"showgirldayoff", "show girl day off", tweet)\n    tweet = re.sub(r"Yougslavia", "Yugoslavia", tweet)\n    tweet = re.sub(r"QuantumDataInformatics", "Quantum Data Informatics", tweet)\n    tweet = re.sub(r"FromTheDesk", "From The Desk", tweet)\n    tweet = re.sub(r"TheaterTrial", "Theater Trial", tweet)\n    tweet = re.sub(r"CatoInstitute", "Cato Institute", tweet)\n    tweet = re.sub(r"EmekaGift", "Emeka Gift", tweet)\n    tweet = re.sub(r"LetsBe_Rational", "Let us be rational", tweet)\n    tweet = re.sub(r"Cynicalreality", "Cynical reality", tweet)\n    tweet = re.sub(r"FredOlsenCruise", "Fred Olsen Cruise", tweet)\n    tweet = re.sub(r"NotSorry", "not sorry", tweet)\n    tweet = re.sub(r"UseYourWords", "use your words", tweet)\n    tweet = re.sub(r"WordoftheDay", "word of the day", tweet)\n    tweet = re.sub(r"Dictionarycom", "Dictionary.com", tweet)\n    tweet = re.sub(r"TheBrooklynLife", "The Brooklyn Life", tweet)\n    tweet = re.sub(r"jokethey", "joke they", tweet)\n    tweet = re.sub(r"nflweek1picks", "NFL week 1 picks", tweet)\n    tweet = re.sub(r"uiseful", "useful", tweet)\n    tweet = re.sub(r"JusticeDotOrg", "The American Association for Justice", tweet)\n    tweet = re.sub(r"autoaccidents", "auto accidents", tweet)\n    tweet = re.sub(r"SteveGursten", "Steve Gursten", tweet)\n    tweet = re.sub(r"MichiganAutoLaw", "Michigan Auto Law", tweet)\n    tweet = re.sub(r"birdgang", "bird gang", tweet)\n    tweet = re.sub(r"nflnetwork", "NFL Network", tweet)\n    tweet = re.sub(r"NYDNSports", "NY Daily News Sports", tweet)\n    tweet = re.sub(r"RVacchianoNYDN", "Ralph Vacchiano NY Daily News", tweet)\n    tweet = re.sub(r"EdmontonEsks", "Edmonton Eskimos", tweet)\n    tweet = re.sub(r"david_brelsford", "David Brelsford", tweet)\n    tweet = re.sub(r"TOI_India", "The Times of India", tweet)\n    tweet = re.sub(r"hegot", "he got", tweet)\n    tweet = re.sub(r"SkinsOn9", "Skins on 9", tweet)\n    tweet = re.sub(r"sothathappened", "so that happened", tweet)\n    tweet = re.sub(r"LCOutOfDoors", "LC Out Of Doors", tweet)\n    tweet = re.sub(r"NationFirst", "Nation First", tweet)\n    tweet = re.sub(r"IndiaToday", "India Today", tweet)\n    tweet = re.sub(r"HLPS", "helps", tweet)\n    tweet = re.sub(r"HOSTAGESTHROSW", "hostages throw", tweet)\n    tweet = re.sub(r"SNCTIONS", "sanctions", tweet)\n    tweet = re.sub(r"BidTime", "Bid Time", tweet)\n    tweet = re.sub(r"crunchysensible", "crunchy sensible", tweet)\n    tweet = re.sub(r"RandomActsOfRomance", "Random acts of romance", tweet)\n    tweet = re.sub(r"MomentsAtHill", "Moments at hill", tweet)\n    tweet = re.sub(r"eatshit", "eat shit", tweet)\n    tweet = re.sub(r"liveleakfun", "live leak fun", tweet)\n    tweet = re.sub(r"SahelNews", "Sahel News", tweet)\n    tweet = re.sub(r"abc7newsbayarea", "ABC 7 News Bay Area", tweet)\n    tweet = re.sub(r"facilitiesmanagement", "facilities management", tweet)\n    tweet = re.sub(r"facilitydude", "facility dude", tweet)\n    tweet = re.sub(r"CampLogistics", "Camp logistics", tweet)\n    tweet = re.sub(r"alaskapublic", "Alaska public", tweet)\n    tweet = re.sub(r"MarketResearch", "Market Research", tweet)\n    tweet = re.sub(r"AccuracyEsports", "Accuracy Esports", tweet)\n    tweet = re.sub(r"TheBodyShopAust", "The Body Shop Australia", tweet)\n    tweet = re.sub(r"yychail", "Calgary hail", tweet)\n    tweet = re.sub(r"yyctraffic", "Calgary traffic", tweet)\n    tweet = re.sub(r"eliotschool", "eliot school", tweet)\n    tweet = re.sub(r"TheBrokenCity", "The Broken City", tweet)\n    tweet = re.sub(r"OldsFireDept", "Olds Fire Department", tweet)\n    tweet = re.sub(r"RiverComplex", "River Complex", tweet)\n    tweet = re.sub(r"fieldworksmells", "field work smells", tweet)\n    tweet = re.sub(r"IranElection", "Iran Election", tweet)\n    tweet = re.sub(r"glowng", "glowing", tweet)\n    tweet = re.sub(r"kindlng", "kindling", tweet)\n    tweet = re.sub(r"riggd", "rigged", tweet)\n    tweet = re.sub(r"slownewsday", "slow news day", tweet)\n    tweet = re.sub(r"MyanmarFlood", "Myanmar Flood", tweet)\n    tweet = re.sub(r"abc7chicago", "ABC 7 Chicago", tweet)\n    tweet = re.sub(r"copolitics", "Colorado Politics", tweet)\n    tweet = re.sub(r"AdilGhumro", "Adil Ghumro", tweet)\n    tweet = re.sub(r"netbots", "net bots", tweet)\n    tweet = re.sub(r"byebyeroad", "bye bye road", tweet)\n    tweet = re.sub(r"massiveflooding", "massive flooding", tweet)\n    tweet = re.sub(r"EndofUS", "End of United States", tweet)\n    tweet = re.sub(r"35PM", "35 PM", tweet)\n    tweet = re.sub(r"greektheatrela", "Greek Theatre Los Angeles", tweet)\n    tweet = re.sub(r"76mins", "76 minutes", tweet)\n    tweet = re.sub(r"publicsafetyfirst", "public safety first", tweet)\n    tweet = re.sub(r"livesmatter", "lives matter", tweet)\n    tweet = re.sub(r"myhometown", "my hometown", tweet)\n    tweet = re.sub(r"tankerfire", "tanker fire", tweet)\n    tweet = re.sub(r"MEMORIALDAY", "memorial day", tweet)\n    tweet = re.sub(r"MEMORIAL_DAY", "memorial day", tweet)\n    tweet = re.sub(r"instaxbooty", "instagram booty", tweet)\n    tweet = re.sub(r"Jerusalem_Post", "Jerusalem Post", tweet)\n    tweet = re.sub(r"WayneRooney_INA", "Wayne Rooney", tweet)\n    tweet = re.sub(r"VirtualReality", "Virtual Reality", tweet)\n    tweet = re.sub(r"OculusRift", "Oculus Rift", tweet)\n    tweet = re.sub(r"OwenJones84", "Owen Jones", tweet)\n    tweet = re.sub(r"jeremycorbyn", "Jeremy Corbyn", tweet)\n    tweet = re.sub(r"paulrogers002", "Paul Rogers", tweet)\n    tweet = re.sub(r"mortalkombatx", "Mortal Kombat X", tweet)\n    tweet = re.sub(r"mortalkombat", "Mortal Kombat", tweet)\n    tweet = re.sub(r"FilipeCoelho92", "Filipe Coelho", tweet)\n    tweet = re.sub(r"OnlyQuakeNews", "Only Quake News", tweet)\n    tweet = re.sub(r"kostumes", "costumes", tweet)\n    tweet = re.sub(r"YEEESSSS", "yes", tweet)\n    tweet = re.sub(r"ToshikazuKatayama", "Toshikazu Katayama", tweet)\n    tweet = re.sub(r"IntlDevelopment", "Intl Development", tweet)\n    tweet = re.sub(r"ExtremeWeather", "Extreme Weather", tweet)\n    tweet = re.sub(r"WereNotGruberVoters", "We are not gruber voters", tweet)\n    tweet = re.sub(r"NewsThousands", "News Thousands", tweet)\n    tweet = re.sub(r"EdmundAdamus", "Edmund Adamus", tweet)\n    tweet = re.sub(r"EyewitnessWV", "Eye witness WV", tweet)\n    tweet = re.sub(r"PhiladelphiaMuseu", "Philadelphia Museum", tweet)\n    tweet = re.sub(r"DublinComicCon", "Dublin Comic Con", tweet)\n    tweet = re.sub(r"NicholasBrendon", "Nicholas Brendon", tweet)\n    tweet = re.sub(r"Alltheway80s", "All the way 80s", tweet)\n    tweet = re.sub(r"FromTheField", "From the field", tweet)\n    tweet = re.sub(r"NorthIowa", "North Iowa", tweet)\n    tweet = re.sub(r"WillowFire", "Willow Fire", tweet)\n    tweet = re.sub(r"MadRiverComplex", "Mad River Complex", tweet)\n    tweet = re.sub(r"feelingmanly", "feeling manly", tweet)\n    tweet = re.sub(r"stillnotoverit", "still not over it", tweet)\n    tweet = re.sub(r"FortitudeValley", "Fortitude Valley", tweet)\n    tweet = re.sub(r"CoastpowerlineTramTr", "Coast powerline", tweet)\n    tweet = re.sub(r"ServicesGold", "Services Gold", tweet)\n    tweet = re.sub(r"NewsbrokenEmergency", "News broken emergency", tweet)\n    tweet = re.sub(r"Evaucation", "evacuation", tweet)\n    tweet = re.sub(r"leaveevacuateexitbe", "leave evacuate exit be", tweet)\n    tweet = re.sub(r"P_EOPLE", "PEOPLE", tweet)\n    tweet = re.sub(r"Tubestrike", "tube strike", tweet)\n    tweet = re.sub(r"CLASS_SICK", "CLASS SICK", tweet)\n    tweet = re.sub(r"localplumber", "local plumber", tweet)\n    tweet = re.sub(r"awesomejobsiri", "awesome job siri", tweet)\n    tweet = re.sub(r"PayForItHow", "Pay for it how", tweet)\n    tweet = re.sub(r"ThisIsAfrica", "This is Africa", tweet)\n    tweet = re.sub(r"crimeairnetwork", "crime air network", tweet)\n    tweet = re.sub(r"KimAcheson", "Kim Acheson", tweet)\n    tweet = re.sub(r"cityofcalgary", "City of Calgary", tweet)\n    tweet = re.sub(r"prosyndicate", "pro syndicate", tweet)\n    tweet = re.sub(r"660NEWS", "660 NEWS", tweet)\n    tweet = re.sub(r"BusInsMagazine", "Business Insurance Magazine", tweet)\n    tweet = re.sub(r"wfocus", "focus", tweet)\n    tweet = re.sub(r"ShastaDam", "Shasta Dam", tweet)\n    tweet = re.sub(r"go2MarkFranco", "Mark Franco", tweet)\n    tweet = re.sub(r"StephGHinojosa", "Steph Hinojosa", tweet)\n    tweet = re.sub(r"Nashgrier", "Nash Grier", tweet)\n    tweet = re.sub(r"NashNewVideo", "Nash new video", tweet)\n    tweet = re.sub(r"IWouldntGetElectedBecause", "I would not get elected because", tweet)\n    tweet = re.sub(r"SHGames", "Sledgehammer Games", tweet)\n    tweet = re.sub(r"bedhair", "bed hair", tweet)\n    tweet = re.sub(r"JoelHeyman", "Joel Heyman", tweet)\n    tweet = re.sub(r"viaYouTube", "via YouTube", tweet)\n           \n    # Urls\n    tweet = re.sub(r"https?:\\/\\/t.co\\/[A-Za-z0-9]+", "", tweet)\n        \n    # Words with punctuations and special characters\n    punctuations = \'@#!?+&*[]-%.:/();$=><|{}^\' + "\'`"\n    for p in punctuations:\n        tweet = tweet.replace(p, f\' {p} \')\n        \n    # ... and ..\n    tweet = tweet.replace(\'...\', \' ... \')\n    if \'...\' not in tweet:\n        tweet = tweet.replace(\'..\', \' ... \')      \n        \n    # Acronyms\n    tweet = re.sub(r"MH370", "Malaysia Airlines Flight 370", tweet)\n    tweet = re.sub(r"mÌ¼sica", "music", tweet)\n    tweet = re.sub(r"okwx", "Oklahoma City Weather", tweet)\n    tweet = re.sub(r"arwx", "Arkansas Weather", tweet)    \n    tweet = re.sub(r"gawx", "Georgia Weather", tweet)  \n    tweet = re.sub(r"scwx", "South Carolina Weather", tweet)  \n    tweet = re.sub(r"cawx", "California Weather", tweet)\n    tweet = re.sub(r"tnwx", "Tennessee Weather", tweet)\n    tweet = re.sub(r"azwx", "Arizona Weather", tweet)  \n    tweet = re.sub(r"alwx", "Alabama Weather", tweet)\n    tweet = re.sub(r"wordpressdotcom", "wordpress", tweet)    \n    tweet = re.sub(r"usNWSgov", "United States National Weather Service", tweet)\n    tweet = re.sub(r"Suruc", "Sanliurfa", tweet)   \n    \n    # Grouping same words without embeddings\n    tweet = re.sub(r"Bestnaijamade", "bestnaijamade", tweet)\n    tweet = re.sub(r"SOUDELOR", "Soudelor", tweet)\n    \n    return tweet\n\ndf_train[\'text_cleaned\'] = df_train[\'text\'].apply(lambda s : clean(s))\ndf_test[\'text_cleaned\'] = df_test[\'text\'].apply(lambda s : clean(s))\n\ntrain_glove_oov, train_glove_vocab_coverage, train_glove_text_coverage = check_embeddings_coverage(df_train[\'text_cleaned\'], glove_embeddings)\ntest_glove_oov, test_glove_vocab_coverage, test_glove_text_coverage = check_embeddings_coverage(df_test[\'text_cleaned\'], glove_embeddings)\nprint(\'GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Training Set\'.format(train_glove_vocab_coverage, train_glove_text_coverage))\nprint(\'GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Test Set\'.format(test_glove_vocab_coverage, test_glove_text_coverage))\n\ntrain_fasttext_oov, train_fasttext_vocab_coverage, train_fasttext_text_coverage = check_embeddings_coverage(df_train[\'text_cleaned\'], fasttext_embeddings)\ntest_fasttext_oov, test_fasttext_vocab_coverage, test_fasttext_text_coverage = check_embeddings_coverage(df_test[\'text_cleaned\'], fasttext_embeddings)\nprint(\'FastText Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Training Set\'.format(train_fasttext_vocab_coverage, train_fasttext_text_coverage))\nprint(\'FastText Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Test Set\'.format(test_fasttext_vocab_coverage, test_fasttext_text_coverage))')


# After cleaning the tweets, `glove_embeddings` and `fasttext_embeddings` are deleted and garbage collected because they consume too much memory (10+ gigabytes).

# In[ ]:


del glove_embeddings, fasttext_embeddings, train_glove_oov, test_glove_oov, train_fasttext_oov, test_fasttext_oov
gc.collect()


# ## **5. Mislabeled Samples**
# There are **18** unique tweets in training set which are labeled differently in their duplicates. Those tweets are probably labeled by different people and they interpreted the meaning differently because some of them are not very clear. Tweets with two unique `target` values are relabeled since they can affect the training score.

# In[ ]:


df_mislabeled = df_train.groupby(['text']).nunique().sort_values(by='target', ascending=False)
df_mislabeled = df_mislabeled[df_mislabeled['target'] > 1]['target']
df_mislabeled.index.tolist()


# In[ ]:


df_train['target_relabeled'] = df_train['target'].copy() 

df_train.loc[df_train['text'] == 'like for the music video I want some real action shit like burning buildings and police chases not some weak ben winston shit', 'target_relabeled'] = 0
df_train.loc[df_train['text'] == 'Hellfire is surrounded by desires so be careful and donÛªt let your desires control you! #Afterlife', 'target_relabeled'] = 0
df_train.loc[df_train['text'] == 'To fight bioterrorism sir.', 'target_relabeled'] = 0
df_train.loc[df_train['text'] == '.POTUS #StrategicPatience is a strategy for #Genocide; refugees; IDP Internally displaced people; horror; etc. https://t.co/rqWuoy1fm4', 'target_relabeled'] = 1
df_train.loc[df_train['text'] == 'CLEARED:incident with injury:I-495  inner loop Exit 31 - MD 97/Georgia Ave Silver Spring', 'target_relabeled'] = 1
df_train.loc[df_train['text'] == '#foodscare #offers2go #NestleIndia slips into loss after #Magginoodle #ban unsafe and hazardous for #humanconsumption', 'target_relabeled'] = 0
df_train.loc[df_train['text'] == 'In #islam saving a person is equal in reward to saving all humans! Islam is the opposite of terrorism!', 'target_relabeled'] = 0
df_train.loc[df_train['text'] == 'Who is bringing the tornadoes and floods. Who is bringing the climate change. God is after America He is plaguing her\n \n#FARRAKHAN #QUOTE', 'target_relabeled'] = 1
df_train.loc[df_train['text'] == 'RT NotExplained: The only known image of infamous hijacker D.B. Cooper. http://t.co/JlzK2HdeTG', 'target_relabeled'] = 1
df_train.loc[df_train['text'] == "Mmmmmm I'm burning.... I'm burning buildings I'm building.... Oooooohhhh oooh ooh...", 'target_relabeled'] = 0
df_train.loc[df_train['text'] == "wowo--=== 12000 Nigerian refugees repatriated from Cameroon", 'target_relabeled'] = 0
df_train.loc[df_train['text'] == "He came to a land which was engulfed in tribal war and turned it into a land of peace i.e. Madinah. #ProphetMuhammad #islam", 'target_relabeled'] = 0
df_train.loc[df_train['text'] == "Hellfire! We donÛªt even want to think about it or mention it so letÛªs not do anything that leads to it #islam!", 'target_relabeled'] = 0
df_train.loc[df_train['text'] == "The Prophet (peace be upon him) said 'Save yourself from Hellfire even if it is by giving half a date in charity.'", 'target_relabeled'] = 0
df_train.loc[df_train['text'] == "Caution: breathing may be hazardous to your health.", 'target_relabeled'] = 1
df_train.loc[df_train['text'] == "I Pledge Allegiance To The P.O.P.E. And The Burning Buildings of Epic City. ??????", 'target_relabeled'] = 0
df_train.loc[df_train['text'] == "#Allah describes piling up #wealth thinking it would last #forever as the description of the people of #Hellfire in Surah Humaza. #Reflect", 'target_relabeled'] = 0
df_train.loc[df_train['text'] == "that horrible sinking feeling when youÛªve been at home on your phone for a while and you realise its been on 3G this whole time", 'target_relabeled'] = 0


# ## **6. Cross-validation**
# First of all, when the training/test sets are concatenated, and tweet counts by `keyword` are computed, it can be seen that training and test set are split inside `keyword` groups. We can also come to that conclusion by looking at `id` feature. This means every `keyword` are stratified while creating training and test set. We can replicate the same split for cross-validation.
# 
# Tweets from every `keyword` group exist in both training and test set and they are from the same sample. In order to replicate the same split technique, `StratifiedKFold` is used and `keyword` is passed as `y`, so stratification is done based on the `keyword` feature. `shuffle` is set to `True` for extra training diversity. Both folds have tweets from every `keyword` group in training and validation sets which can be seen from below.

# In[ ]:


K = 2
skf = StratifiedKFold(n_splits=K, random_state=SEED, shuffle=True)

DISASTER = df_train['target'] == 1
print('Whole Training Set Shape = {}'.format(df_train.shape))
print('Whole Training Set Unique keyword Count = {}'.format(df_train['keyword'].nunique()))
print('Whole Training Set Target Rate (Disaster) {}/{} (Not Disaster)'.format(df_train[DISASTER]['target_relabeled'].count(), df_train[~DISASTER]['target_relabeled'].count()))

for fold, (trn_idx, val_idx) in enumerate(skf.split(df_train['text_cleaned'], df_train['target']), 1):
    print('\nFold {} Training Set Shape = {} - Validation Set Shape = {}'.format(fold, df_train.loc[trn_idx, 'text_cleaned'].shape, df_train.loc[val_idx, 'text_cleaned'].shape))
    print('Fold {} Training Set Unique keyword Count = {} - Validation Set Unique keyword Count = {}'.format(fold, df_train.loc[trn_idx, 'keyword'].nunique(), df_train.loc[val_idx, 'keyword'].nunique()))    


# ## **7. Model**

# ### **7.1 Metric**
# The leaderboard is based on **Mean F-Score** which can be implemented with **Macro Average F1 Score**. However, it won't be very informative without **Accuracy**, **Precision** and **Recall** because classes are almost balanced and it is hard to tell which class is harder to predict.
# 
# * **Accuracy** measures the fraction of the total sample that is correctly identified
# * **Precision** measures that out of all the examples predicted as positive, how many are actually positive
# * **Recall** measures that out of all the actual positives, how many examples were correctly classified as positive by the model
# * **F1 Score** is the harmonic mean of the **Precision** and **Recall**
# 
# Keras has accuracy in its `metrics` module, but doesn't have rest of the metrics stated above. Another crucial point is **Precision**, **Recall** and **F1-Score** are global metrics so they should be calculated on whole training or validation set. Computing them on every batch would be both misleading and ineffective in terms of execution time. `ClassificationReport` which is similar to `sklearn.metrics.classification_report`, computes those metrics after every epoch for the given training and validation set.

# In[ ]:


class ClassificationReport(Callback):
    
    def __init__(self, train_data=(), validation_data=()):
        super(Callback, self).__init__()
        
        self.X_train, self.y_train = train_data
        self.train_precision_scores = []
        self.train_recall_scores = []
        self.train_f1_scores = []
        
        self.X_val, self.y_val = validation_data
        self.val_precision_scores = []
        self.val_recall_scores = []
        self.val_f1_scores = [] 
               
    def on_epoch_end(self, epoch, logs={}):
        train_predictions = np.round(self.model.predict(self.X_train, verbose=0))        
        train_precision = precision_score(self.y_train, train_predictions, average='macro')
        train_recall = recall_score(self.y_train, train_predictions, average='macro')
        train_f1 = f1_score(self.y_train, train_predictions, average='macro')
        self.train_precision_scores.append(train_precision)        
        self.train_recall_scores.append(train_recall)
        self.train_f1_scores.append(train_f1)
        
        val_predictions = np.round(self.model.predict(self.X_val, verbose=0))
        val_precision = precision_score(self.y_val, val_predictions, average='macro')
        val_recall = recall_score(self.y_val, val_predictions, average='macro')
        val_f1 = f1_score(self.y_val, val_predictions, average='macro')
        self.val_precision_scores.append(val_precision)        
        self.val_recall_scores.append(val_recall)        
        self.val_f1_scores.append(val_f1)
        
        print('\nEpoch: {} - Training Precision: {:.6} - Training Recall: {:.6} - Training F1: {:.6}'.format(epoch + 1, train_precision, train_recall, train_f1))
        print('Epoch: {} - Validation Precision: {:.6} - Validation Recall: {:.6} - Validation F1: {:.6}'.format(epoch + 1, val_precision, val_recall, val_f1))  


# ### **7.2 BERT Layer**
# 
# This model uses the implementation of BERT from the TensorFlow Models repository on GitHub at `tensorflow/models/official/nlp/bert`. It uses L=12 hidden layers (Transformer blocks), a hidden size of H=768, and A=12 attention heads.
# 
# This model has been pre-trained for English on the Wikipedia and BooksCorpus. Inputs have been **"uncased"**, meaning that the text has been lower-cased before tokenization into word pieces, and any accent markers have been stripped. In order to download this model, `Internet` must be activated on the kernel.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nbert_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1', trainable=True)")


# ### **7.3 Architecture**
# `DisasterDetector` is a wrapper that incorporates the cross-validation and metrics stated above. 
# 
# The tokenization of input text is performed with the `FullTokenizer` class from `tensorflow/models/official/nlp/bert/tokenization.py`. `max_seq_length` parameter can be used for tuning the sequence length of text.
# 
# Parameters such as `lr`, `epochs` and `batch_size` can be used for controlling the learning process. There are no dense or pooling layers added after last layer of BERT. `SGD` is used as optimizer since others have hard time while converging.
# 
# `plot_learning_curve` plots **Accuracy**, **Precision**, **Recall** and **F1 Score** (for validation set) stored after every epoch alongside with training/validation loss curve. This helps to see which metric fluctuates most while training.

# In[ ]:


class DisasterDetector:
    
    def __init__(self, bert_layer, max_seq_length=128, lr=0.0001, epochs=15, batch_size=32):
        
        # BERT and Tokenization params
        self.bert_layer = bert_layer
        
        self.max_seq_length = max_seq_length        
        vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
        
        # Learning control params
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.models = []
        self.scores = {}
        
        
    def encode(self, texts):
                
        all_tokens = []
        all_masks = []
        all_segments = []

        for text in texts:
            text = self.tokenizer.tokenize(text)
            text = text[:self.max_seq_length - 2]
            input_sequence = ['[CLS]'] + text + ['[SEP]']
            pad_len = self.max_seq_length - len(input_sequence)

            tokens = self.tokenizer.convert_tokens_to_ids(input_sequence)
            tokens += [0] * pad_len
            pad_masks = [1] * len(input_sequence) + [0] * pad_len
            segment_ids = [0] * self.max_seq_length

            all_tokens.append(tokens)
            all_masks.append(pad_masks)
            all_segments.append(segment_ids)

        return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
    
    
    def build_model(self):
        
        input_word_ids = Input(shape=(self.max_seq_length,), dtype=tf.int32, name='input_word_ids')
        input_mask = Input(shape=(self.max_seq_length,), dtype=tf.int32, name='input_mask')
        segment_ids = Input(shape=(self.max_seq_length,), dtype=tf.int32, name='segment_ids')    
        
        pooled_output, sequence_output = self.bert_layer([input_word_ids, input_mask, segment_ids])   
        clf_output = sequence_output[:, 0, :]
        out = Dense(1, activation='sigmoid')(clf_output)
        
        model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
        optimizer = SGD(learning_rate=self.lr, momentum=0.8)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        return model
    
    
    def train(self, X):
        
        for fold, (trn_idx, val_idx) in enumerate(skf.split(X['text_cleaned'], X['keyword'])):
            
            print('\nFold {}\n'.format(fold))
        
            X_trn_encoded = self.encode(X.loc[trn_idx, 'text_cleaned'].str.lower())
            y_trn = X.loc[trn_idx, 'target_relabeled']
            X_val_encoded = self.encode(X.loc[val_idx, 'text_cleaned'].str.lower())
            y_val = X.loc[val_idx, 'target_relabeled']
        
            # Callbacks
            metrics = ClassificationReport(train_data=(X_trn_encoded, y_trn), validation_data=(X_val_encoded, y_val))
            
            # Model
            model = self.build_model()        
            model.fit(X_trn_encoded, y_trn, validation_data=(X_val_encoded, y_val), callbacks=[metrics], epochs=self.epochs, batch_size=self.batch_size)
            
            self.models.append(model)
            self.scores[fold] = {
                'train': {
                    'precision': metrics.train_precision_scores,
                    'recall': metrics.train_recall_scores,
                    'f1': metrics.train_f1_scores                    
                },
                'validation': {
                    'precision': metrics.val_precision_scores,
                    'recall': metrics.val_recall_scores,
                    'f1': metrics.val_f1_scores                    
                }
            }
                    
                
    def plot_learning_curve(self):
        
        fig, axes = plt.subplots(nrows=K, ncols=2, figsize=(20, K * 6), dpi=100)
    
        for i in range(K):
            
            # Classification Report curve
            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.models[i].history.history['val_accuracy'], ax=axes[i][0], label='val_accuracy')
            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.scores[i]['validation']['precision'], ax=axes[i][0], label='val_precision')
            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.scores[i]['validation']['recall'], ax=axes[i][0], label='val_recall')
            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.scores[i]['validation']['f1'], ax=axes[i][0], label='val_f1')        

            axes[i][0].legend() 
            axes[i][0].set_title('Fold {} Validation Classification Report'.format(i), fontsize=14)

            # Loss curve
            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.models[0].history.history['loss'], ax=axes[i][1], label='train_loss')
            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.models[0].history.history['val_loss'], ax=axes[i][1], label='val_loss')

            axes[i][1].legend() 
            axes[i][1].set_title('Fold {} Train / Validation Loss'.format(i), fontsize=14)

            for j in range(2):
                axes[i][j].set_xlabel('Epoch', size=12)
                axes[i][j].tick_params(axis='x', labelsize=12)
                axes[i][j].tick_params(axis='y', labelsize=12)

        plt.show()
        
        
    def predict(self, X):
        
        X_test_encoded = self.encode(X['text_cleaned'].str.lower())
        y_pred = np.zeros((X_test_encoded[0].shape[0], 1))

        for model in self.models:
            y_pred += model.predict(X_test_encoded) / len(self.models)

        return y_pred


# ### **7.4 Training, Evaluation and Prediction**

# In[ ]:


clf = DisasterDetector(bert_layer, max_seq_length=128, lr=0.0001, epochs=10, batch_size=32)

clf.train(df_train)


# In[ ]:


clf.plot_learning_curve()


# In[ ]:


y_pred = clf.predict(df_test)

model_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
model_submission['target'] = np.round(y_pred).astype('int')
model_submission.to_csv('model_submission.csv', index=False)
model_submission.describe()


# ## **8. Test Set Labels**
# Test set labels can be found on [this](https://www.figure-eight.com/data-for-everyone/) website. Dataset is named **Disasters on social media**. This is how people are submitting perfect scores. Other "Getting Started" competitions also have their test labels available. The main point of "Getting Started" competitions is **learning and sharing**, and perfect score doesn't mean anything. 
# 
# > **Phil Culliton wrote:**
# > For the AutoML prize, any use of the label set will result in disqualification.
# 
# According to [@philculliton](https://www.kaggle.com/philculliton) from Kaggle Team, competitors who use test set labels in any way are not eligible to win AutoML prize. There are no other penalties for using them.

# In[ ]:


df_leak = pd.read_csv('../input/nlp-with-disaster-tweets-test-set-with-labels/socialmedia-disaster-tweets-DFE.csv', encoding ='ISO-8859-1')[['choose_one', 'text']]

# Creating target and id
df_leak['target'] = (df_leak['choose_one'] == 'Relevant').astype(np.int8)
df_leak['id'] = df_leak.index.astype(np.int16)
df_leak.drop(columns=['choose_one', 'text'], inplace=True)

# Merging target to test set
df_test = df_test.merge(df_leak, on=['id'], how='left')

print('Leaked Data Set Shape = {}'.format(df_leak.shape))
print('Leaked Data Set Memory Usage = {:.2f} MB'.format(df_leak.memory_usage().sum() / 1024**2))

perfect_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
perfect_submission['target'] = df_test['target'].values
perfect_submission.to_csv('perfect_submission.csv', index=False)
perfect_submission.describe()


# ## **9. Preprocessed Datasets**
# Preprocessed datasets are saved in pickle format for people who don't want to wait for preprocessing. Instead of forking and waiting for all preprocessing operations, it is easier to use this kernel as a data source in your own kernel. It can be done by searching and selecting this kernel after clicking `+ Add Data` button.

# In[ ]:


TRAIN_FEATURES = ['id', 'keyword', 'location', 'text', 'target', 'text_cleaned', 'target_relabeled']
TEST_FEATURES = ['id', 'keyword', 'location', 'text', 'target', 'text_cleaned']

df_train[TRAIN_FEATURES].to_pickle('train.pkl')
df_test[TEST_FEATURES].to_pickle('test.pkl')

print('Training Set Shape = {}'.format(df_train[TRAIN_FEATURES].shape))
print('Training Set Memory Usage = {:.2f} MB'.format(df_train[TRAIN_FEATURES].memory_usage().sum() / 1024**2))
print('Test Set Shape = {}'.format(df_test[TEST_FEATURES].shape))
print('Test Set Memory Usage = {:.2f} MB'.format(df_test[TEST_FEATURES].memory_usage().sum() / 1024**2))

