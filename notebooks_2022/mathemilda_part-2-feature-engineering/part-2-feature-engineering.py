#!/usr/bin/env python
# coding: utf-8

# My work with Russian Troll Tweets is divided into 3 parts due to Kaggle resources restrictions. Here are the part links:
# #### [Part 1. EDA](https://www.kaggle.com/code/mathemilda/part-i-eda)
# #### [Part 2. Feature Engineering](https://www.kaggle.com/code/mathemilda/part-2-feature-engineering) (this one)
# #### [Part 3. Machine Learning with accuracy 99.6 on a test set.%](https://www.kaggle.com/code/mathemilda/part-3-machine-learning-with-accuracy-99-6)
# 
# I learned in Part 1, EDA and other Kaggle notebooks for Russian Troll dataset that we cannot determine if an account belongs to Russian trolls or not by one post. But we can analyze its activity, like tweet posts times and tweet properties. Now I want to download other data sets and compare them with Russian Troll Tweets to figure out what features could be useful. For this I found on Kaggle [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) and 
# [Raw Twitter Timelines w/ No Retweets](https://www.kaggle.com/datasets/speckledpingu/RawTwitterFeeds). The first one had been harvested before 2009, and I hope that at the time numerous Russian Trolls were not as numerous as afterwards. Unfortunately the tweets are not raw but partially cleaned and in particular emojis are removed. The second set contains American celebrity tweets.
# 
# They are problematic sets because they have too few tweets per account, as you will see below. The reason for this might be that they were harvested for a short period of time. I wish I can get something more suitable and I will appreciate it if somebody can help me with it.
# 
# # Outline
# ## 1) Download, initial cleaning and preparation for Russian Troll data set and 2 more Kaggle datasets.
# ## 2) Counts of different objects in texts
# ## 3) Plots
# ## 4) Feature calculation
# ## 5) Feature selection with Mutual information method and correlations
# ___
#  My findings
#  * The additional datasets which I use do not contain sufficient information comparatively with Russian Troll tweets. As result: 
#    * Combined data are not balanced with respect to troll presence.
#    * There are no proper datetime columns so I cannot use clustering by post time.
#    * Amount of posted texts in the additional datasets cannot be used for proper comparison of n-gram frequencies.
#  * Nevertheless we can detect differences in account behaviors judging by troll propaganda efforts. Below goes a partial list of their tricks.
#    * Trolls actively use URL, Twitter handles and hashtags to spread their information as wide as possible
#    * It appears that trolls have specific guidelines for message length. 
#    * It looks like they tend to use longer words in their messages than normal people do. It could be because they should include particular words and expressions.
#    * I got the same emoji frequency for trolls and non-trolls. The second set was partially cleaned from them, so I believe that usually trolls do not use many emojis. Emoji absence tells us that troll's messages are not personal, just work.
#  * Trolls produce a bit more errors in English than non-trolls, although not in a way I expected and they are much less significant for their detection. Nevertheless their usage of punctuation signs slightly differs from native English speakers and can be used for fine distinction.
#  
# **Conclusion.** An account behavior analysis can yield good predictors for troll detection. 
# 
# ___
# 
# *Remark.* For some reasons a very useful module `ftfy` cannot be found on Kaggle, although it was here a couple of years ago. I have tried to reach the Support team about it and they replied that I should look up the Kaggle forum for help. I installed it here, although as you see I got a message that I should not do it.

# In[ ]:


import pandas as pd
import os
import glob
import numpy as np
import re
from string import punctuation, whitespace
import warnings
warnings.filterwarnings("ignore")
get_ipython().system('pip install ftfy')
import ftfy
from sklearn.utils import shuffle
import gc
import matplotlib.pyplot as plt
import multiprocess as mp
from sklearn.feature_selection import mutual_info_classif


# Here goes the Russian Troll Tweets dataset. 
# I dropped features I do not need, removed a row with missing data, and restricted the data set to only English tweets. 
# Please check https://www.kaggle.com/code/mathemilda/part-i-eda because I explained more there. In addition I removed all texts with Russian letters and anything looking like German. At first I did it to speed up my work and start Machine Learning sooner, but then I discovered that I can have a very high accuracy even without the posts.
# 
# ### 1) Download, initial cleaning and preparation for Russian Troll dataset and a couple of more Kaggle datasets.

# In[ ]:


PATH = "../input/russian-troll-tweets/"
filenames = glob.glob(os.path.join(PATH, "*.csv"))
full_ru_trolls = pd.concat((pd.read_csv(f) for f in filenames))
full_ru_trolls.drop(['external_author_id', 'region', 'harvested_date',
        'updates', 'account_type', 'new_june_2018', 'post_type',
        'account_category', 'following', 'followers', 'retweet'],
        axis=1, inplace=True)
full_ru_trolls = full_ru_trolls[full_ru_trolls.content.notnull()]
full_ru_trolls_en = full_ru_trolls[full_ru_trolls.language == 'English'].copy(deep=True)
full_ru_trolls_en.rename(columns={'author': 'account', 'content': 'tweet'}, inplace=True)
full_ru_trolls_en = full_ru_trolls_en[~full_ru_trolls_en.tweet.str.contains('А-Яа-я')]
german_s = re.compile('(Ich )|(Sie )|(Ihnen )|( sich$)|( [Kk]?eine? )|( [Dd]as )|'+
           '^[Dd]as |^[Ss]ind | bist | und | sind | (?!(van|von|-)) der |' + 
           '[ ^][a-z]*ö|[ ^][a-z]*ä|[ ^][a-z]*ü')
full_ru_trolls_en = full_ru_trolls_en[~full_ru_trolls_en.tweet.str.contains(german_s)].copy(deep=True)
del full_ru_trolls
full_ru_trolls_en.drop(['language', 'publish_date'],
        axis=1, inplace=True)
full_ru_trolls_en['troll'] = 1


# I found here on Kaggle a couple of data sets which seem to be suitable for comparison. I wanted something which was posted by genuine Americans. Apparently celebrity tweets are good candidates. In addition I picked up an old data set for sentiment detection, hoping that at the time it was harvested (before 2009) Russian trolls had not been so abundant. Although the last data set has been cleaned of emojis and may be something else.

# In[ ]:


PATH = "../input/RawTwitterFeeds"
filenames = glob.glob(os.path.join(PATH, "*.csv"))
celebs = pd.concat((pd.read_csv(f) for f in filenames))
celebs.drop(['Unnamed: 0', 'Unnamed: 0.1','id', 'date', 'link', 'retweet'], axis=1,inplace=True)
celebs.rename(columns={'author': 'account', 'text': 'tweet'}, inplace=True)
celebs = celebs[celebs.tweet.notnull()]

sentiment140 = pd.read_csv('../input/sentiment140/training.1600000.processed.noemoticon.csv',    encoding = 'Latin-1', names=('target', 'id', 'date', 'flag', 'username','tweet'))
sentiment140.drop(['target', 'id', 'date', 'flag'], axis=1, inplace=True)
sentiment140.rename(columns={'username': 'account'}, inplace=True)


# Regretfully their columns are different from the ones for Russian Trolls. In particular their `date` may look like "23h23 hours ago" or "Oct 13".  As a result I cannot use a very promising for clustering Timestamp variable. 

# In[ ]:


used_cols = ['account', 'tweet']
combined = pd.concat([celebs[used_cols], sentiment140[used_cols]], axis=0
                     , ignore_index = True)
#shuffle() from sklearn.utils
combined['troll'] = 0
combined = shuffle(combined, random_state=42).reset_index()
del celebs, sentiment140


# Now let us check the new data properties. 

# In[ ]:


print('A number of Twitter accounts in the new data sets is:')
print(combined.account.nunique())
print('The average number of tweets per account is about '+       str(round(len(combined)/combined.account.nunique(), 1)))
combined_tweet_counts = combined.groupby(combined.account).size().reset_index()
combined_tweet_counts.rename(columns={ 0: 'tweet_count'}, inplace=True)
print('The maximal number of posts per account is '+ str(combined_tweet_counts.tweet_count.max()))
print('The minimal number of posts per account is '+ str(combined_tweet_counts.tweet_count.min()))
print("The median number of post per account is " + str(combined_tweet_counts.tweet_count.median()))
combined_tweet_counts.tweet_count.hist(bins=50)
plt.title('Histogram of tweet counts per account in the new data set')


# Let me remind you that with the Russian troll data set our statistics were:
# 
#  * The mean number of tweets per author is about 1044
#  * The maximal number of posts per account is 59652
#  * The minimal number of posts per account is 1
#  * The median number of post per account is 154.5
#  
# As we see statistics are different, in particular at least half of new accounts have only one tweet. One tweet does not have much information about its author. In particular, counting bigram frequency is not very reliable for only one post. Let us keep only accounts which have at least 10 tweets. It is still not sufficient for bigram frequency, so I'm not going to do it.
#  
# I am using that my datasets were checked for duplication and each row represents a unique tweet per account at posting time. Thus for the number of tweets I need to compute a number of rows for each account. As we see a histogram still has the same shape, because we do not have many posts for the vast majority of accounts.

# In[ ]:


def at_least_10_tweets(data, account_col, troll, min_count=10):
    acc_properties = data[[account_col, troll]].groupby(account_col).count()            .reset_index()
    acc_properties.rename(columns={'troll': 'tweet_count'}, inplace=True)
    kept_accs = acc_properties[acc_properties.tweet_count >= min_count]
    restricted = data[data.account.isin(kept_accs.account)].copy(deep=True)
    return restricted

combined = at_least_10_tweets(combined, account_col= 'account', troll = 'troll',  min_count=10)
print('A number of Twitter accounts in the restricted new data sets is: ')
print(combined.account.nunique())
combined_tweet_counts.tweet_count.hist(bins=50)
plt.title('Histogram of tweet counts per account with at least 10 posts')


# Apparently we need to fix our Russian Troll tweets data as well. Of course I can concatenate the sets, but I run into a problem with memory for a cell with plots, and now I am trying to  alleviate all I can. So the datasets will be separate.

# In[ ]:


ru_trolls =  at_least_10_tweets(full_ru_trolls_en, account_col= 'account', troll = 'troll',  min_count=10)


# ### 2) Counts of different objects in texts
# The features are selected for reasons stated below.
# 
# a) **Detection troll activities related to their propaganda.** They need to spread their message as wide as possible, and for this they can use URLs, hashtags and Twitter handles. They want to use emotional shock because distracted people are not good at rational thinking and I thought that I could use the appearance of an exclamation sign for this, but it did not work out the way I expected. There are other ways to measure a message sentiment level, but they are more involved and I already got a high accuracy with my features, so I left it out. Trolls should include a lot of particular words and expressions. While the words may change with time, the frequent appearance of such propaganda terms does not, thus average word lengths are not very random. Paid posters are likely to have particular guidelines about their message length: not too short, while they are in a hurry to produce a required number of posts, and they often go just above the accepted minimum. With many messages it could be detected as well. 
# 
# b) **Picking up typical mistakes of native Russian speakers when they use English.** Russian grammar has different rules for commas and dashes. Russian language does not have determiners, so incorrect usage of them can show up. There are other possible mistakes, but they mostly require more involved NLP. 
# 
# Below is a function which cleans a tweet and returns for each the following values as a tuple:
# 
#  * a cleaned tweet
#  
#  a)
#    * an url count
#    * a hashtag count
#    * a handle count
#    * an exclamation sign count
#    * an emoji and other pictogram count
#    * a cleaned tweet length (meaningful text length)
#    * an average word length in a tweet
#    
#  b)
#    * a comma count
#    * a dash count
#    * a determiner 'a' count
#    * a determiner 'the' count
# 
# Their order is changed due to calculation optimization. 

# In[ ]:


dashes = [chr(int(d, 16)) for d in ['058A', '05BE', '1400', '1806', '2010', '2011',          '2012', '2013', '2014', '2015', '2053', '207B', '208B', '2212', '2E17',           '2E1A', '2E3A', '2E3B', '2E40', '2E5D', '301C', '3030', '30A0', 'FE31',           'FE32', 'FE58', 'FE63', 'FF0D', '10EAD']]
dashes_compiled = re.compile('[' + ''.join(dashes) + ']+', flags = re.UNICODE)

def cleaning_and_counts(s):
    s = ftfy.fix_text(s)
    s = re.sub(dashes_compiled, '-', s)
    url_n = len(re.findall('https?://\\S+\\b', s))
    s = re.sub('https?://\\S+\\b', '', s)
    hasht_n = len(re.findall('#\S+', s))
    s = re.sub('#\S+', '', s)
    handle_n = len(re.findall('@([a-z0-9_]{1,15})\\b', s))
    s = re.sub('@([a-z0-9_]{1,15})\\b', '', s)
    exl_n = len(re.findall('!', s))
    s = re.sub('pic\\.twitter\\.com/\\w+\\b', '', s)
    s = re.sub('\\s+', ' ', s) #reducing multiple whitespaces to one
    s = s.lstrip(whitespace+punctuation+'\xa0'+chr(8230))
    s = s.rstrip(whitespace+'\xa0')
    l=''
    emoji_and_such = 0
    for ch in s:
        if ord(ch) < 8204:
            l += ch
        else:
            emoji_and_such += 1
    comma_n = len(re.findall(',', s))
    dash_n = len(re.findall('-', s)) # dropped spaces around the dash
    a_an_n = len(re.findall(r'\b[Aa]n?\b', s))
    the_n = len(re.findall(r'\b[Tt]he\b', s))
    # reduce a number of repeated symbols to no more than 2 
    l = re.sub(r'(.)\1\1+', r'\1\1', l)
    length = len(l)
    words = [len(w) for w in re.findall(r'\b\w+\b', l)]
    if len(words)==0:
        average_word = 0
    else:
        average_word = np.median(words)
    return l, url_n, hasht_n, handle_n, emoji_and_such        , exl_n, comma_n, dash_n        , a_an_n, the_n, length, average_word


# For the `multiprocess` module to work it must be in a separate cell from the applied function. 

# In[ ]:


get_ipython().run_cell_magic('time', '', "with mp.Pool(processes= mp.cpu_count()) as p:\n    combined['tuple'] = p.map(cleaning_and_counts, combined.tweet)\n\nwith mp.Pool(processes= mp.cpu_count()) as p:\n    ru_trolls['tuple'] = p.map(cleaning_and_counts, ru_trolls.tweet)")


# Now I need to distribute the created features in separate columns. I specified short integer types for numeric columns to reduce troubles with excessive RAM demand.

# In[ ]:


features = ("cleaned_tweet, url_n, hasht_n, handle_n, emoji_and_such, "+            "exl_n, comma_n, dash_n, " +            "a_an_n, the_n, length, average_word").split(', ')

data_list = [combined, ru_trolls]
for df in data_list:
    for i in range(len(features)):
        if i ==0:
            df[features[i]] = df.tuple.apply(lambda t: t[i])
        else:
            df[features[i]] = df.tuple.apply(lambda t: t[i]).astype(np.uint8)


# I want to free some memory and I will need a list of numerical columns.

# In[ ]:


combined.drop(['tuple'], axis=1,inplace=True)
ru_trolls.drop(['tuple'], axis=1,inplace=True)
gc.collect()
print(features)
num_cols = features[1:]


# ### 3) Plots
# Now we can check if the calculated features are helpful. Let us look at their histograms. Two variables are notably different from others and they are the length of the cleaned tweet and the maximal word length in the tweet, so I will plot them separately. At this point you can see the other 9 counts.

# In[ ]:


def nine_plots(data1, data2, cols):
    figure, axis = plt.subplots(3, 3, sharex=False, sharey=False, figsize=(12, 9))
    k=0
    for i in range(3):
        for j in range(3):
            upper = data1[cols[k]].max()
            axis[i,j].hist(data1[cols[k]], bins =20, range=[0, upper],                     label='trolls', density = True, alpha=.75)
            axis[i,j].hist(data2[cols[k]], bins =20, range=[0, upper],
                     label='not trolls', density = True, alpha=.75)
            axis[i,j].set_title(cols[k])
            axis[i,j].legend()
            k +=1
    plt.show

nine_plots(ru_trolls[num_cols], combined[num_cols], num_cols)


# Their histograms differs, although they have something in common. It appears that Russian Trolls have a particular guidance about publishing more items which facilitate spreading their message as wide as possible: url weblinks, Twitter handles, hashtags. The interesting thing is that the frequency of emojis is the same for both data sets, although one of the non-troll datasets was cleaned. Looks like normal people tend to use more emojis. I guess they do not have somebody's rules on their feeling expressions. 
# 
# Let us observe how post lengths and mean word lengths vary between trolls and non-trolls. 

# In[ ]:


figure, axis = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(12, 4))
axis[0].hist(ru_trolls.length.values, bins = 50, label='trolls', density = True, alpha=.75)
axis[0].hist(combined.length.values, bins = 50, label='not trolls', density = True, alpha=.75)
axis[0].set_title("Histograms for 'length' variable")
axis[0].legend()
axis[1].hist(ru_trolls.average_word.values, bins = 50, range= [0,100], label='trolls', density = True, alpha=.75)
axis[1].hist(combined.average_word.values, bins = 50,  range= [0,100], label='not trolls', density = True, alpha=.75)
axis[1].set_title("Histograms for average word length")
axis[1].legend()
plt.show


# We notice similar phenomena here. Histograms are not quite the same, but they overlap a lot. In particular, we see an evident pick for tweet lengths, which are likely to be because trolls have guidelines for their posts: to keep their messages to particular length to count toward their daily quote (around 130 posts per shift). At the same time they may often avoid posting messages longer than necessary because they have to type many messages. Their words in general are longer and it could be due to their instructions to repeat particular words and expressions with slight variations.
# 
# ## 4) Feature calculation
# I decided to compute percentiles for my count variables. This way I can catch differences in distributions. For example, 90 percentile is a data value such that 90% of all entries lie in the histogram to the left of the value. It can be approximated in the same way as median calculations.
# First let us combine the datasets.

# In[ ]:


colnames = ['account', 'troll'] + num_cols
total_data = pd.concat([combined[colnames], ru_trolls[colnames]],                      ignore_index = True)
del ru_trolls, combined, full_ru_trolls_en


# Here go the percentile computations.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'for qu in range(10, 100, 10): \n    percentiles = total_data.groupby(\'account\').quantile(q=qu/100).reset_index()\n    cols_to_change = {col : col +\'_\' + str(qu) for col in num_cols}\n    percentiles.rename(columns=cols_to_change, inplace=True)\n    if qu == 10:\n        all_percentiles = percentiles\n    else:\n        all_percentiles = pd.merge(all_percentiles, percentiles, how = "left", on = [\'account\', \'troll\'])\n    \ndel total_data\nall_percentiles.head()')


# Interestly enough we get no zero columns and not many columns with very few entries.

# In[ ]:


new_features = all_percentiles.columns[2:]
zero_cols = [all_percentiles[col].sum()==0 for col in new_features]
print('A number of zero columns is '+str(sum(zero_cols)))
almost_zero = sum([(all_percentiles[col] != 0).sum()<0.005*len(all_percentiles) for col in new_features])
print('Here are only ' + str(almost_zero) + ' of columns with nonzero entries less than .5%.')


# As we see a vast majority of accounts favor particular things which they tend to mention more often.
# 
# ### 5) Feature selection with mutual information method and Pearson correlation
# For Machine Learning I'd better reduce the number of variables to speed up computations. I decided to use Mutual Information method for this because usual Pearson correlation may fail to detect dependency between random variables when it is not linear. In the same time when the last one is high, then variables are likely to be very similar. The conventional threshold is .7, but it is for variables with normal distribution. We do not have it, so we need to find a suitable maximal value of correlation.
# 
# The scikit-learn module contains feature selection methods which use Mutual Information. There are 2 ways it could be applied: 
#  * Choosing a specified number of features with the highest MI scores
#  * Choosing a specified percentage of features with highest scores.
#  
# I want to choose features with MI higher than a provided threshold and drop highly correlated ones. The correlation will be checked with the Pearson correlation because it is much faster to compute than MI score. In a highly correlated pair I prefer to keep a column with a higher MI score. So both of the scikit-learn methods are unsuitable for me.
# 
# You can see below how some of my variables are correlated, especially when we look right above the matrix main diagonal, and how percentiles of the same count might not correlate much. 

# In[ ]:


all_percentiles[['url_n_10', 'url_n_20', 'url_n_30', 'url_n_40', 'url_n_50', 'url_n_60', 'url_n_70', 'url_n_80', 'url_n_90']].corr()


# I need to thin them keeping the most useful variables. I will use the Mutual Information method for selecting good variables. Constant columns will have 0 MI score, so I will drop them together with low MI score columns if I have any. You see below the list of new features ordered by their MI score with troll status. They are ordered by their MI score in descending order. 

# In[ ]:


mi = mutual_info_classif(all_percentiles[new_features].values, all_percentiles.troll.values, n_neighbors= 19)
cols_mi = list(zip(new_features, mi))
cols_mi.sort(reverse = True, key=lambda x: x[1])
cols_mi = [pair[0] for pair in cols_mi if pair[1] > 0.001]
print(cols_mi)


# As we see the most significant features are the ones for propaganda methods: hashtags, URLs, average word length, Twitter handles. I ran Logistic regression on them and got a good accuracy on a test set: 98%. I was wondering if I can get better accuracy with non linear methods. At the same time I would like to reduce the number of variables. I will use Pearson correlation for this, because it computes quickly and if it is high, the variables definitely depend on each other a lot. (While when it is low, we are not sure.) I could use the MI method again to check for dependency between variables, but it takes a long time to compute.
# 
# The function below drops highly correlated variables from our list, keeping the ones with better MI score. 

# In[ ]:


def drop_correlated(data, sorted_cols, threshold=.8):
    new = [[sorted_cols[0]], [0]]
    corr_matrix = data[sorted_cols].corr().values
    N = len(sorted_cols)
    for i in range(1, N):
        tr = corr_matrix[new[1], i]
        if sum(np.abs(tr) >threshold)==0:
            new[0] += [sorted_cols[i]]
            new[1] += [i]
    return new[0]


# Here is a list of newly computed variables.

# In[ ]:


new_cols =  drop_correlated(all_percentiles[new_features], cols_mi, threshold =.75)
print('A number of selected columns is: ' + str(len(new_cols)))
print(new_cols)


# Let us check histogram plots with the first 18 variables.

# In[ ]:


nine_plots(all_percentiles[all_percentiles.troll==1], all_percentiles[all_percentiles.troll==0], new_cols[:9])
nine_plots(all_percentiles[all_percentiles.troll==1], all_percentiles[all_percentiles.troll==0], new_cols[9:18])


# Well, these variables look more promising.

# In[ ]:




