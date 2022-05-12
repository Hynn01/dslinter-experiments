#!/usr/bin/env python
# coding: utf-8

# # <div align = 'center'><b>NLP with Disaster Tweets</b></div>
# <img align = middle src="https://akm-img-a-in.tosshub.com/aajtak/images/photo_gallery/202105/twitter_final_5.jpg">
# 
# # Installing the Necessary Libraries

# In[ ]:


get_ipython().system('pip install text-hammer')


# # Imports

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from wordcloud import WordCloud

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium 
from folium import plugins 

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from collections import defaultdict
import text_hammer as th

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')


# # Color Scheme
# Setting the color scheme for the notebook

# In[ ]:


custom_colors = ['#000000', '#E31E33', '#4A53E1', '#F5AD02', '#94D5EA', '#F6F8F7']
custom_palette = sns.set_palette(sns.color_palette(custom_colors))
sns.palplot(sns.color_palette(custom_colors), size = 1)
plt.tick_params(axis = 'both', labelsize = 0, length = 0)


# Looking at the input files

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Reading the dataframe

# In[ ]:


df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
df


# In[ ]:


print(df.isna().sum())
print('----------------------------')
print('Total Missing Values: ', df.isna().sum().sum())
print('----------------------------')


# # EDA (Exploratory Data Analysis)

# Visualizing the missing data in the form of a chart

# In[ ]:


plt.figure(figsize = (15, 10))
sns.heatmap(df.isna(), yticklabels = False, cbar = False, cmap = 'afmhot')
plt.title("Visualizing the Missing Data", fontsize = 20)
plt.xticks(rotation = 35, fontsize = 15)
plt.show()


# Bar representation of the missing values

# In[ ]:


msno.bar(df, color = (0, 0, 0), sort = "ascending", figsize = (15, 10))
plt.show()


# Lets take a look at the class distribution of our dataset

# In[ ]:


plt.figure(figsize = (15, 12))
ax = plt.axes()
ax.set_facecolor('black')
ax = sns.countplot(x = 'target', data = df, palette = [custom_colors[2], custom_colors[1]], edgecolor = 'white', linewidth = 1.2)
plt.title('Disaster Count', fontsize = 25)
plt.xlabel('Disaster', fontsize = 20)
plt.ylabel('Count', fontsize = 20)
ax.xaxis.set_tick_params(labelsize = 15)
ax.yaxis.set_tick_params(labelsize = 15)
bbox_args = dict(boxstyle = 'round', fc = '0.9')
for p in ax.patches:
        ax.annotate('{:.0f} = {:.2f}%'.format(p.get_height(), (p.get_height() / len(df['target'])) * 100), (p.get_x() + 0.25, p.get_height() + 60), 
                   color = 'black',
                   bbox = bbox_args,
                   fontsize = 15)
plt.show()


# There is a class imbalance in the dataset, with 4342 non-disaster tweets and 3271 disaster tweets.
# 
# Let's take a look at where most of the tweets in our dataset come from:

# In[ ]:


df['location'].value_counts()[:10]


# In[ ]:


plt.figure(figsize = (15, 13))
ax = plt.axes()
ax.set_facecolor('black')
ax = ((df.location.value_counts())[:10]).plot(kind = 'bar', color = custom_colors[2], linewidth = 2, edgecolor = 'white')
plt.title('Location Count', fontsize = 30)
plt.xlabel('Location', fontsize = 25)
plt.ylabel('Count', fontsize = 25)
ax.xaxis.set_tick_params(labelsize = 15, rotation = 30)
ax.yaxis.set_tick_params(labelsize = 15)
bbox_args = dict(boxstyle = 'round', fc = '0.9')
for p in ax.patches:
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x() + 0.15, p.get_height() + 2),
                   bbox = bbox_args,
                   color = custom_colors[2],
                   fontsize = 15)


# Visualizing the top 10 locations from where most of the tweets originate from

# In[ ]:


new_df = pd.DataFrame()
new_df['location'] = ((df['location'].value_counts())[:10]).index
new_df['count'] = ((df['location'].value_counts())[:10]).values
geolocator = Nominatim(user_agent = 'Rahil')
geocode = RateLimiter(geolocator.geocode, min_delay_seconds = 0.5)
lat = {}
long = {}
for i in new_df['location']:
    location = geocode(i)
    lat[i] = location.latitude
    long[i] = location.longitude
new_df['latitude'] = new_df['location'].map(lat)
new_df['longitude'] = new_df['location'].map(long)
map = folium.Map(location = [10.0, 10.0], tiles = 'CartoDB dark_matter', zoom_start = 1.5)
markers = []
title = '''<h1 align = "center" style = "font-size: 35px"><b>Top 10 Tweet Locations</b></h1>'''
for i, r in new_df.iterrows():
    loss = r['count']
    if r['count'] > 0:
        counts = r['count'] * 0.4
        folium.CircleMarker([float(r['latitude']), float(r['longitude'])], radius = float(counts), color = custom_colors[1], fill = True).add_to(map)
map.get_root().html.add_child(folium.Element(title))
map


# In[ ]:


non_list_stopwords = stopwords.words('english')
stopwords = list(stopwords.words('english'))
stopwords[:10]


# In[ ]:


non_disaster_tweets_length = (df[df['target'] == 0])['text'].str.len()
disaster_tweets_length = (df[df['target'] == 1])['text'].str.len()
print(non_disaster_tweets_length)
print(disaster_tweets_length)


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize = (30, 15))
fig.suptitle('Tweet Character Length', fontsize = 45)

axes[0].set_facecolor('black')
axes[0].hist(non_disaster_tweets_length, color = custom_colors[1], edgecolor = 'white', linewidth = 4)
axes[0].set_title('Non-Disaster Tweets', fontsize = 40)
axes[0].set_xlabel('Character Length', fontsize = 35)
axes[0].set_ylabel('Frequency', fontsize = 35)
axes[0].xaxis.set_tick_params(labelsize = 30)
axes[0].yaxis.set_tick_params(labelsize = 30)

axes[1].set_facecolor('black')
axes[1].hist(disaster_tweets_length, color = custom_colors[2], edgecolor = 'white', linewidth = 4)
axes[1].set_title('Disaster Tweets', fontsize = 40)
axes[1].set_xlabel('Character Length', fontsize = 35)
axes[1].set_ylabel('Frequency', fontsize = 35)
axes[1].xaxis.set_tick_params(labelsize = 30)
axes[1].yaxis.set_tick_params(labelsize = 30)

plt.subplots_adjust(wspace = 0.25, hspace = 0.1)
plt.show()


# In[ ]:


class tweet_functions:
    
    '''Getting the count of different attributes of our tweets'''
    
    def __init__(self, column):
        self.column = column
        
    def count_characters(self):
        return((self.column).apply(lambda word: len(str(word))))
    
    def count_words(self):
        return((self.column).apply(lambda word: len(str(word).split())))
    
    def count_urls(self):
        return((self.column).apply(lambda word: len([url for url in str(word).lower().split() if 'http' in word or 'https' in word])))
    
    def count_hashtags(self):
        return((self.column).apply(lambda word: len([hashtag for hashtag in str(word) if '#' in hashtag])))
    
    def count_tags(self):
        return((self.column).apply(lambda word: len([tag for tag in str(word) if '@' in tag])))
    
    def count_stopwords(self):
        return((self.column).apply(lambda word: len([word for word in str(word).lower().split() if word in stopwords])))


# In[ ]:


fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (30, 30))

axes[0][0].set_facecolor('black')
sns.distplot(tweet_functions((df[df['target'] == 0])['text']).count_characters(), ax = axes[0][0], color = custom_colors[3], label = 'Non-Disaster Tweets', kde_kws = dict(linewidth = 3.5))
sns.distplot(tweet_functions((df[df['target'] == 1])['text']).count_characters(), ax = axes[0][0], color = custom_colors[4], label = 'Disaster Tweets', kde_kws = dict(linewidth = 3.5))
axes[0][0].set_title('Character Count', fontsize = 45)
axes[0][0].set_xlabel('Characters', fontsize = 40)
axes[0][0].set_ylabel('Density', fontsize = 40)
axes[0][0].xaxis.set_tick_params(labelsize = 30)
axes[0][0].yaxis.set_tick_params(labelsize = 30)
axes[0][0].legend(facecolor = 'black', labelcolor = 'white', prop = {'size': 25}).get_frame().set_linewidth(2.5)

axes[0][1].set_facecolor('black')
sns.distplot(tweet_functions((df[df['target'] == 0])['text']).count_words(), ax = axes[0][1], color = custom_colors[3], label = 'Non-Disaster Tweets', kde_kws = dict(linewidth = 3.5))
sns.distplot(tweet_functions((df[df['target'] == 1])['text']).count_words(), ax = axes[0][1], color = custom_colors[4], label = 'Disaster Tweets', kde_kws = dict(linewidth = 3.5))
axes[0][1].set_title('Word Count', fontsize = 45)
axes[0][1].set_xlabel('Words', fontsize = 40)
axes[0][1].set_ylabel('Density', fontsize = 40)
axes[0][1].xaxis.set_tick_params(labelsize = 30)
axes[0][1].yaxis.set_tick_params(labelsize = 30)
axes[0][1].legend(facecolor = 'black', labelcolor = 'white', prop = {'size': 25}).get_frame().set_linewidth(2.5)

axes[1][0].set_facecolor('black')
sns.distplot(tweet_functions((df[df['target'] == 0])['text']).count_urls(), ax = axes[1][0], color = custom_colors[3], label = 'Non-Disaster Tweets', kde_kws = dict(linewidth = 3.5))
sns.distplot(tweet_functions((df[df['target'] == 1])['text']).count_urls(), ax = axes[1][0], color = custom_colors[4], label = 'Disaster Tweets', kde_kws = dict(linewidth = 3.5))
axes[1][0].set_title('URL Count', fontsize = 45)
axes[1][0].set_xlabel('URLs', fontsize = 40)
axes[1][0].set_ylabel('Density', fontsize = 40)
axes[1][0].xaxis.set_tick_params(labelsize = 30)
axes[1][0].yaxis.set_tick_params(labelsize = 30)
axes[1][0].legend(facecolor = 'black', labelcolor = 'white', prop = {'size': 25}).get_frame().set_linewidth(2.5)

axes[1][1].set_facecolor('black')
sns.distplot(tweet_functions((df[df['target'] == 0])['text']).count_hashtags(), ax = axes[1][1], color = custom_colors[3], label = 'Non-Disaster Tweets', kde_kws = dict(linewidth = 3.5))
sns.distplot(tweet_functions((df[df['target'] == 1])['text']).count_hashtags(), ax = axes[1][1], color = custom_colors[4], label = 'Disaster Tweets', kde_kws = dict(linewidth = 3.5))
axes[1][1].set_title('Hashtag Count', fontsize = 45)
axes[1][1].set_xlabel('Hashtags', fontsize = 40)
axes[1][1].set_ylabel('Density', fontsize = 40)
axes[1][1].xaxis.set_tick_params(labelsize = 30)
axes[1][1].yaxis.set_tick_params(labelsize = 30)
axes[1][1].legend(facecolor = 'black', labelcolor = 'white', prop = {'size': 25}).get_frame().set_linewidth(2.5)

axes[2][0].set_facecolor('black')
sns.distplot(tweet_functions((df[df['target'] == 0])['text']).count_tags(), ax = axes[2][0], color = custom_colors[3], label = 'Non-Disaster Tweets', kde_kws = dict(linewidth = 3.5))
sns.distplot(tweet_functions((df[df['target'] == 1])['text']).count_tags(), ax = axes[2][0], color = custom_colors[4], label = 'Disaster Tweets', kde_kws = dict(linewidth = 3.5))
axes[2][0].set_title('Mention Count', fontsize = 45)
axes[2][0].set_xlabel('Mentions', fontsize = 40)
axes[2][0].set_ylabel('Density', fontsize = 40)
axes[2][0].xaxis.set_tick_params(labelsize = 30)
axes[2][0].yaxis.set_tick_params(labelsize = 30)
axes[2][0].legend(facecolor = 'black', labelcolor = 'white', prop = {'size': 25}).get_frame().set_linewidth(2.5)

axes[2][1].set_facecolor('black')
sns.distplot(tweet_functions((df[df['target'] == 0])['text']).count_stopwords(), ax = axes[2][1], color = custom_colors[3], label = 'Non-Disaster Tweets', kde_kws = dict(linewidth = 3.5))
sns.distplot(tweet_functions((df[df['target'] == 1])['text']).count_stopwords(), ax = axes[2][1], color = custom_colors[4], label = 'Disaster Tweets', kde_kws = dict(linewidth = 3.5))
axes[2][1].set_title('Stopword Count', fontsize = 45)
axes[2][1].set_xlabel('Stopwords', fontsize = 40)
axes[2][1].set_ylabel('Density', fontsize = 40)
axes[2][1].xaxis.set_tick_params(labelsize = 30)
axes[2][1].yaxis.set_tick_params(labelsize = 30)
axes[2][1].legend(facecolor = 'black', labelcolor = 'white', prop = {'size': 25}).get_frame().set_linewidth(2.5)

plt.subplots_adjust(hspace = 0.5)
plt.show()


# # Preprocessing the Tweets

# In[ ]:


def remove_urls(text):
    urls = re.compile(r'https?://\S+|www\.\S+')
    return urls.sub(r'', text)

def remove_HTML(text):
    html = re.compile('<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    emoji_pattern = re.compile('['
                           u'\U0001F600-\U0001F64F'
                           u'\U0001F300-\U0001F5FF'
                           u'\U0001F680-\U0001F6FF'
                           u'\U0001F1E0-\U0001F1FF'
                           u'\U00002702-\U000027B0'
                           u'\U000024C2-\U0001F251'
                           ']+', flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_emoticons(text):
    emoticons = {
    u":‑\)":"Happy face or smiley",
    u":\)":"Happy face or smiley",
    u":-\]":"Happy face or smiley",
    u":\]":"Happy face or smiley",
    u":-3":"Happy face smiley",
    u":3":"Happy face smiley",
    u":->":"Happy face smiley",
    u":>":"Happy face smiley",
    u"8-\)":"Happy face smiley",
    u":o\)":"Happy face smiley",
    u":-\}":"Happy face smiley",
    u":\}":"Happy face smiley",
    u":-\)":"Happy face smiley",
    u":c\)":"Happy face smiley",
    u":\^\)":"Happy face smiley",
    u"=\]":"Happy face smiley",
    u"=\)":"Happy face smiley",
    u":‑D":"Laughing, big grin or laugh with glasses",
    u":D":"Laughing, big grin or laugh with glasses",
    u"8‑D":"Laughing, big grin or laugh with glasses",
    u"8D":"Laughing, big grin or laugh with glasses",
    u"X‑D":"Laughing, big grin or laugh with glasses",
    u"XD":"Laughing, big grin or laugh with glasses",
    u"=D":"Laughing, big grin or laugh with glasses",
    u"=3":"Laughing, big grin or laugh with glasses",
    u"B\^D":"Laughing, big grin or laugh with glasses",
    u":-\)\)":"Very happy",
    u":‑\(":"Frown, sad, andry or pouting",
    u":-\(":"Frown, sad, andry or pouting",
    u":\(":"Frown, sad, andry or pouting",
    u":‑c":"Frown, sad, andry or pouting",
    u":c":"Frown, sad, andry or pouting",
    u":‑<":"Frown, sad, andry or pouting",
    u":<":"Frown, sad, andry or pouting",
    u":‑\[":"Frown, sad, andry or pouting",
    u":\[":"Frown, sad, andry or pouting",
    u":-\|\|":"Frown, sad, andry or pouting",
    u">:\[":"Frown, sad, andry or pouting",
    u":\{":"Frown, sad, andry or pouting",
    u":@":"Frown, sad, andry or pouting",
    u">:\(":"Frown, sad, andry or pouting",
    u":'‑\(":"Crying",
    u":'\(":"Crying",
    u":'‑\)":"Tears of happiness",
    u":'\)":"Tears of happiness",
    u"D‑':":"Horror",
    u"D:<":"Disgust",
    u"D:":"Sadness",
    u"D8":"Great dismay",
    u"D;":"Great dismay",
    u"D=":"Great dismay",
    u"DX":"Great dismay",
    u":‑O":"Surprise",
    u":O":"Surprise",
    u":‑o":"Surprise",
    u":o":"Surprise",
    u":-0":"Shock",
    u"8‑0":"Yawn",
    u">:O":"Yawn",
    u":-\*":"Kiss",
    u":\*":"Kiss",
    u":X":"Kiss",
    u";‑\)":"Wink or smirk",
    u";\)":"Wink or smirk",
    u"\*-\)":"Wink or smirk",
    u"\*\)":"Wink or smirk",
    u";‑\]":"Wink or smirk",
    u";\]":"Wink or smirk",
    u";\^\)":"Wink or smirk",
    u":‑,":"Wink or smirk",
    u";D":"Wink or smirk",
    u":‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"X‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"XP":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":‑Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"d:":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"=p":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u">:P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":‑/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":-[.]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u">:[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u">:/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":L":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=L":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":S":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":‑\|":"Straight face",
    u":\|":"Straight face",
    u":$":"Embarrassed or blushing",
    u":‑x":"Sealed lips or wearing braces or tongue-tied",
    u":x":"Sealed lips or wearing braces or tongue-tied",
    u":‑#":"Sealed lips or wearing braces or tongue-tied",
    u":#":"Sealed lips or wearing braces or tongue-tied",
    u":‑&":"Sealed lips or wearing braces or tongue-tied",
    u":&":"Sealed lips or wearing braces or tongue-tied",
    u"O:‑\)":"Angel, saint or innocent",
    u"O:\)":"Angel, saint or innocent",
    u"0:‑3":"Angel, saint or innocent",
    u"0:3":"Angel, saint or innocent",
    u"0:‑\)":"Angel, saint or innocent",
    u"0:\)":"Angel, saint or innocent",
    u":‑b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"0;\^\)":"Angel, saint or innocent",
    u">:‑\)":"Evil or devilish",
    u">:\)":"Evil or devilish",
    u"\}:‑\)":"Evil or devilish",
    u"\}:\)":"Evil or devilish",
    u"3:‑\)":"Evil or devilish",
    u"3:\)":"Evil or devilish",
    u">;\)":"Evil or devilish",
    u"\|;‑\)":"Cool",
    u"\|‑O":"Bored",
    u":‑J":"Tongue-in-cheek",
    u"#‑\)":"Party all night",
    u"%‑\)":"Drunk or confused",
    u"%\)":"Drunk or confused",
    u":-###..":"Being sick",
    u":###..":"Being sick",
    u"<:‑\|":"Dump",
    u"\(>_<\)":"Troubled",
    u"\(>_<\)>":"Troubled",
    u"\(';'\)":"Baby",
    u"\(\^\^>``":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(\^_\^;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(-_-;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(~_~;\) \(・\.・;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(-_-\)zzz":"Sleeping",
    u"\(\^_-\)":"Wink",
    u"\(\(\+_\+\)\)":"Confused",
    u"\(\+o\+\)":"Confused",
    u"\(o\|o\)":"Ultraman",
    u"\^_\^":"Joyful",
    u"\(\^_\^\)/":"Joyful",
    u"\(\^O\^\)／":"Joyful",
    u"\(\^o\^\)／":"Joyful",
    u"\(__\)":"Kowtow as a sign of respect, or dogeza for apology",
    u"_\(\._\.\)_":"Kowtow as a sign of respect, or dogeza for apology",
    u"<\(_ _\)>":"Kowtow as a sign of respect, or dogeza for apology",
    u"<m\(__\)m>":"Kowtow as a sign of respect, or dogeza for apology",
    u"m\(__\)m":"Kowtow as a sign of respect, or dogeza for apology",
    u"m\(_ _\)m":"Kowtow as a sign of respect, or dogeza for apology",
    u"\('_'\)":"Sad or Crying",
    u"\(/_;\)":"Sad or Crying",
    u"\(T_T\) \(;_;\)":"Sad or Crying",
    u"\(;_;":"Sad of Crying",
    u"\(;_:\)":"Sad or Crying",
    u"\(;O;\)":"Sad or Crying",
    u"\(:_;\)":"Sad or Crying",
    u"\(ToT\)":"Sad or Crying",
    u";_;":"Sad or Crying",
    u";-;":"Sad or Crying",
    u";n;":"Sad or Crying",
    u";;":"Sad or Crying",
    u"Q\.Q":"Sad or Crying",
    u"T\.T":"Sad or Crying",
    u"QQ":"Sad or Crying",
    u"Q_Q":"Sad or Crying",
    u"\(-\.-\)":"Shame",
    u"\(-_-\)":"Shame",
    u"\(一一\)":"Shame",
    u"\(；一_一\)":"Shame",
    u"\(=_=\)":"Tired",
    u"\(=\^\·\^=\)":"cat",
    u"\(=\^\·\·\^=\)":"cat",
    u"=_\^=	":"cat",
    u"\(\.\.\)":"Looking down",
    u"\(\._\.\)":"Looking down",
    u"\^m\^":"Giggling with hand covering mouth",
    u"\(\・\・?":"Confusion",
    u"\(?_?\)":"Confusion",
    u">\^_\^<":"Normal Laugh",
    u"<\^!\^>":"Normal Laugh",
    u"\^/\^":"Normal Laugh",
    u"\（\*\^_\^\*）" :"Normal Laugh",
    u"\(\^<\^\) \(\^\.\^\)":"Normal Laugh",
    u"\(^\^\)":"Normal Laugh",
    u"\(\^\.\^\)":"Normal Laugh",
    u"\(\^_\^\.\)":"Normal Laugh",
    u"\(\^_\^\)":"Normal Laugh",
    u"\(\^\^\)":"Normal Laugh",
    u"\(\^J\^\)":"Normal Laugh",
    u"\(\*\^\.\^\*\)":"Normal Laugh",
    u"\(\^—\^\）":"Normal Laugh",
    u"\(#\^\.\^#\)":"Normal Laugh",
    u"\（\^—\^\）":"Waving",
    u"\(;_;\)/~~~":"Waving",
    u"\(\^\.\^\)/~~~":"Waving",
    u"\(-_-\)/~~~ \($\·\·\)/~~~":"Waving",
    u"\(T_T\)/~~~":"Waving",
    u"\(ToT\)/~~~":"Waving",
    u"\(\*\^0\^\*\)":"Excited",
    u"\(\*_\*\)":"Amazed",
    u"\(\*_\*;":"Amazed",
    u"\(\+_\+\) \(@_@\)":"Amazed",
    u"\(\*\^\^\)v":"Laughing,Cheerful",
    u"\(\^_\^\)v":"Laughing,Cheerful",
    u"\(\(d[-_-]b\)\)":"Headphones,Listening to music",
    u'\(-"-\)':"Worried",
    u"\(ーー;\)":"Worried",
    u"\(\^0_0\^\)":"Eyeglasses",
    u"\(\＾ｖ\＾\)":"Happy",
    u"\(\＾ｕ\＾\)":"Happy",
    u"\(\^\)o\(\^\)":"Happy",
    u"\(\^O\^\)":"Happy",
    u"\(\^o\^\)":"Happy",
    u"\)\^o\^\(":"Happy",
    u":O o_O":"Surprised",
    u"o_0":"Surprised",
    u"o\.O":"Surpised",
    u"\(o\.o\)":"Surprised",
    u"oO":"Surprised",
    u"\(\*￣m￣\)":"Dissatisfied",
    u"\(‘A`\)":"Snubbed or Deflated"
    }
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in emoticons) + u')')
    return emoticon_pattern.sub(r'', text)

def remove_mentions(text):
    mentions = re.compile('@[A-Za-z0-9_]+')
    return mentions.sub(r'', text)

def word_lemmatizer(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])


# In[ ]:


df['text'] = df['text'].str.lower() # convert to lowercase
df['text'] = df['text'].apply(lambda text: remove_urls(text)) # remove URLs
df['text'] = df['text'].apply(lambda text: remove_HTML(text)) # remove HTML tags
df['text'] = df['text'].str.translate(str.maketrans('', '', string.punctuation)) # remove punctuations
df['text'] = df['text'].apply(lambda text: ' '.join([word for word in str(text).split() if word not in stopwords])) # remove stopwords
df['text'] = df['text'].apply(lambda text: remove_emoji(text)) # remove emojis
df['text'] = df['text'].apply(lambda text: remove_emoticons(text)) # remove emoticons
df['text'] = df['text'].apply(lambda text: remove_mentions(text)) # remove mentions
df['text'] = df['text'].apply(lambda text: word_lemmatizer(text)) # lemmatize words
df['text'] = df['text'].apply(lambda text: th.cont_exp(text)) # convert i'm to i am, you're to you are, etc
df['text']


# In[ ]:


counter = Counter()
for text in df['text'].values:
    for word in text.split():
        counter[word] += 1
counter.most_common(10)


# In[ ]:


data = dict(sorted(counter.items(), key = lambda x: x[1] ,reverse = True)[:10])
words = list(data.keys())
frequency = list(data.values())

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (15, 15))
ax.set_facecolor('black')
ax = sns.barplot(x = frequency, y = words, color = '#8699A7', edgecolor = 'white', linewidth = 2)
plt.title('Word Frequency', fontsize = 35)
plt.xlabel('Frequency', fontsize = 30)
plt.ylabel('Words', fontsize = 30)
plt.xticks(size = 20)
plt.yticks(size = 20)
bbox_args = dict(boxstyle = 'round', fc = '0.9')
for p in ax.patches:
    width = p.get_width()
    plt.text(9.5 + p.get_width(), p.get_y() + 0.5 * p.get_height(), '{:1.0f}'.format(width), 
             ha = 'center', 
             va = 'center', 
             color = 'black', 
             bbox = bbox_args, 
             fontsize = 15)
plt.show()


# # Creating Ngrams

# In[ ]:


def generate_ngrams(text, n_gram = 0):
    token = [token for token in text.lower().split(' ') if token != '' if token not in non_list_stopwords]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [' '.join(ngram) for ngram in ngrams]

def generate_ngram_plots(n_gram):
    non_disaster_ngrams = defaultdict(int)
    disaster_ngrams = defaultdict(int)

    for tweet in df[df['target'] == 0]['text']:
        for word in generate_ngrams(tweet, n_gram = n_gram):
            non_disaster_ngrams[word] += 1

    for tweet in df[df['target'] == 1]['text']:
        for word in generate_ngrams(tweet, n_gram = n_gram):
            disaster_ngrams[word] += 1

    non_disaster_ngram_data = dict(sorted(non_disaster_ngrams.items(), key = lambda x: x[1], reverse = True)[:10])
    non_disaster_ngram_words = list(non_disaster_ngram_data.keys())
    non_disaster_ngram_frequency = list(non_disaster_ngram_data.values())

    disaster_ngram_data = dict(sorted(disaster_ngrams.items(), key = lambda x: x[1], reverse = True)[:10])
    disaster_ngram_words = list(disaster_ngram_data.keys())
    disaster_ngram_frequency = list(disaster_ngram_data.values())

    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (30, 20))

    axes[0].set_facecolor('black')
    sns.barplot(x = non_disaster_ngram_frequency, y = non_disaster_ngram_words, ax = axes[0], color = '#0057B1', edgecolor = 'white', linewidth = 2)
    if(n_gram == 1):
        axes[0].set_title('Non-Disaster Unigrams', fontsize = 45)
    if(n_gram == 2):
        axes[0].set_title('Non-Disaster Bigrams', fontsize = 45)
    if(n_gram == 3):
        axes[0].set_title('Non-Disaster Trigrams', fontsize = 45)
    axes[0].set_xlabel('Count', fontsize = 40)
    axes[0].set_ylabel('Words', fontsize = 40)
    if(n_gram == 1):
        axes[0].xaxis.set_tick_params(labelsize = 30)
        axes[0].yaxis.set_tick_params(labelsize = 30)
    elif(n_gram == 2):
        axes[0].xaxis.set_tick_params(labelsize = 20)
        axes[0].yaxis.set_tick_params(labelsize = 20)
    else:
        axes[0].xaxis.set_tick_params(labelsize = 18)
        axes[0].yaxis.set_tick_params(labelsize = 18)
    for p in axes[0].patches:
        width = p.get_width()
        if(n_gram == 1 or n_gram == 2):
            axes[0].text(0.75 + p.get_width(), p.get_y() + 0.5 * p.get_height(), '{:1.0f}'.format(width), 
                     ha = 'center', 
                     va = 'center', 
                     color = 'blue', 
                     bbox = bbox_args, 
                     fontsize = 25)
        if(n_gram == 3):
            axes[0].text(0.6 + p.get_width(), p.get_y() + 0.5 * p.get_height(), '{:1.0f}'.format(width), 
                     ha = 'center', 
                     va = 'center', 
                     color = 'blue', 
                     bbox = bbox_args, 
                     fontsize = 22)

    axes[1].set_facecolor('black')
    sns.barplot(x = disaster_ngram_frequency, y = disaster_ngram_words, ax = axes[1], palette = [custom_colors[1]], edgecolor = 'white', linewidth = 2)
    if(n_gram == 1):
        axes[1].set_title('Disaster Unigrams', fontsize = 45)
    if(n_gram == 2):
        axes[1].set_title('Disaster Bigrams', fontsize = 45)
    if(n_gram == 3):
        axes[1].set_title('Disaster Trigrams', fontsize = 45)
    axes[1].set_xlabel('Count', fontsize = 40)
    axes[1].set_ylabel('Words', fontsize = 40)
    if(n_gram == 1):
        axes[1].xaxis.set_tick_params(labelsize = 30)
        axes[1].yaxis.set_tick_params(labelsize = 30)
    elif(n_gram == 2):
        axes[1].xaxis.set_tick_params(labelsize = 20)
        axes[1].yaxis.set_tick_params(labelsize = 20)
    else:
        axes[1].xaxis.set_tick_params(labelsize = 18)
        axes[1].yaxis.set_tick_params(labelsize = 18)
    for p in axes[1].patches:
        width = p.get_width()
        if(n_gram == 1 or n_gram == 2):
            axes[1].text(0.8 + p.get_width(), p.get_y() + 0.5 * p.get_height(), '{:1.0f}'.format(width), 
                     ha = 'center', 
                     va = 'center', 
                     color = 'red', 
                     bbox = bbox_args, 
                     fontsize = 25)
        if(n_gram == 3):
            axes[1].text(0.6 + p.get_width(), p.get_y() + 0.5 * p.get_height(), '{:1.0f}'.format(width), 
                     ha = 'center', 
                     va = 'center', 
                     color = 'red', 
                     bbox = bbox_args, 
                     fontsize = 22)
    if(n_gram == 1 or n_gram == 2):
        plt.subplots_adjust(wspace = 0.4)
    if(n_gram == 3):
        plt.subplots_adjust(wspace = 0.6)


# # Unigrams

# In[ ]:


generate_ngram_plots(1)


# # Bigrams

# In[ ]:


generate_ngram_plots(2)


# # Trigrams

# In[ ]:


generate_ngram_plots(3)


# # Wordcloud of Tweets

# In[ ]:


wordcloud = WordCloud(width = 1400, height = 600, background_color = 'black').generate(''.join(text for text in df['text']))
plt.figure(figsize = (20, 10))
plt.title('Wordcloud Visualization of Tweets', fontsize = 30)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[ ]:


df


# In[ ]:


X = df['text']
y = df['target']
print(X)
print(y)


# # Splitting the Data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 0)


# In[ ]:


print(X_train)
print(X_test)
print(y_train)
print(y_test)


# In[ ]:


model_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'
sentence_encoder_layer = hub.KerasLayer(model_url, input_shape = [], dtype = tf.string, trainable = False)


# # Creating the Model

# In[ ]:


model = keras.Sequential([
    sentence_encoder_layer,
    layers.Dense(128, activation = 'relu'),
    layers.Dense(64, activation = 'relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation = 'sigmoid')
])

model.compile(
    loss = 'binary_crossentropy',
    optimizer = keras.optimizers.Adam(lr = 1e-4),
    metrics = ['accuracy']
)


# In[ ]:


model.summary()


# # Learning Rate Reduction

# In[ ]:


reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', patience = 2, verbose = 1, factor = 0.5, min_lr = 0.00001)


# # Training the Model

# In[ ]:


classifier = model.fit(
    X_train,
    y_train,
    epochs = 50,
    validation_data = (X_test, y_test),
    callbacks = [reduce_lr]
)


# # Model Performance

# In[ ]:


def model_performance_graphs():
    
    fig, axes = plt.subplots(1, 2, figsize = (15, 8))

    axes[0].plot(classifier.epoch, classifier.history['accuracy'], label = 'acc')
    axes[0].plot(classifier.epoch, classifier.history['val_accuracy'], label = 'val_acc')
    axes[0].set_title('Accuracy vs Epochs', fontsize = 20)
    axes[0].set_xlabel('Epochs', fontsize = 15)
    axes[0].set_ylabel('Accuracy', fontsize = 15)
    axes[0].legend()

    axes[1].plot(classifier.epoch, classifier.history['loss'], label = 'loss')
    axes[1].plot(classifier.epoch, classifier.history['val_loss'], label="val_loss")
    axes[1].set_title("Loss Curve",fontsize=18)
    axes[1].set_xlabel("Epochs",fontsize=15)
    axes[1].set_ylabel("Loss",fontsize=15)
    axes[1].legend()

    plt.show()
    
model_performance_graphs()


# In[ ]:


model.evaluate(X_train, y_train)


# In[ ]:


model.evaluate(X_test, y_test)


# # Making the Predictions

# In[ ]:


test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
test_df = test_df[['id', 'text']]
pred = model.predict(test_df['text'])
print(pred)
pred = tf.squeeze(tf.round((pred)))
print(np.array(pred))


# # Creating the Submission File

# In[ ]:


test_df['target'] = pred
test_df['target'] = test_df['target'].astype(int)
test_df = test_df[['id', 'target']]
test_df.to_csv('submission.csv', index = False)
test_df


# # References
# > - https://www.kaggle.com/code/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert#2.-Meta-Features
# > - https://www.kaggle.com/code/shahules/basic-eda-cleaning-and-glove
# > - https://www.kaggle.com/code/sudalairajkumar/getting-started-with-text-preprocessing/notebook
# > - https://www.kaggle.com/code/ratan123/start-from-here-disaster-tweets-eda-basic-model#4.-Exploring-location-column

# <div class="alert alert-warning" role="alert">🚧 Work in Progress 🚧</div>
