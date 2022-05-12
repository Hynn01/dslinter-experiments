#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import numpy as np
import pandas as pd
import pandas_profiling as pp

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud

import missingno as ms

import string

# NLTK
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# spaCy
import spacy
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS


# In[ ]:


# Set Plot style
plt.style.use('fivethirtyeight')


# # Getting the data

# In[ ]:


# Load data
df = pd.read_csv('../input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv', index_col=0)


# In[ ]:


# print head of data
df.head()


# In[ ]:


# Info of DataFrame
df.info()


# In[ ]:


df.dtypes.value_counts()


# The above snippet shows that 13 object dtypes and 4 int64 dtypes

# In[ ]:


# Statistical Description
df.describe()


# ## Missing value

# **Missingno** is a visualization tools that highlights the missing value in the entire dataset.

# In[ ]:


ms.matrix(df)
plt.show()


# * White line shows the missing record in the dataset.
# * It looks lost of missing value in the dataset.
# 
# **Let's check out the missing value count**
# 

# In[ ]:


ms.bar(df)
plt.show()


# **The value above bar shows the total count of non-missing value**

# In[ ]:


# counts of missing value for each feature and target
df.isnull().sum()


# **Salary_range feature containt lot's of missing value.**

# In[ ]:


# Drop salary_range
del df['salary_range']


# In[ ]:


# Fill null value
df.fillna("", inplace=True)


# **Count value of flaudulent(target)**

# In[ ]:


fig, ax = plt.subplots(1, 2)

sns.countplot(x='fraudulent', data=df, ax=ax[0])
ax[1].pie(df['fraudulent'].value_counts(), labels=['Real Post', 'Fake Post'], autopct='%1.1f%%')

fig.suptitle('Bar & Pie charts of Fraudulent value count', fontsize=16)
plt.show()


# **Required Experience Real/Fake**

# In[ ]:


fig, ax = plt.subplots(1, 2)

chart = sns.countplot(x = 'required_experience', data=df[df['fraudulent']==0], ax=ax[0])
chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
ax[0].set_title('Real required experience')

chart = sns.countplot(x = 'required_experience', data=df[df['fraudulent']==1], ax=ax[1])
chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
ax[1].set_title('Fake required experience')
plt.show()


# # Basic Feature Extraction
# * Number of characters
# * Number of words
# * Average Word Length
# 

# In[ ]:


# Create features from text columns
text_features = df[["title", "company_profile", "description", "requirements", "benefits","fraudulent"]]


# In[ ]:


# print samples of the text_features
text_features.sample(5)


# In[ ]:


columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']


# 1. Number of characters

# In[ ]:


for col in columns:
    text_features[col+'_len'] = text_features[col].apply(len)


# In[ ]:


real = text_features[text_features['fraudulent']==0]
fake = text_features[text_features['fraudulent']==1]

fig, ax = plt.subplots(5, 2, figsize=(15, 15))
ax[0, 0].set_title('Character Count of Real Post ')
ax[0, 1].set_title('Character Count of Fake Post ')

for i in range(5):
    for j in range(2):
        if j==0:
            ax[i, j].hist(real[columns[i]+'_len'], color='g', bins=15);
            ax[i, j].set_ylabel( columns[i] )
        else:
            ax[i, j].hist(fake[columns[i]+'_len'], color='r', bins=15);

plt.show()


# 2. Number of words

# In[ ]:


for col in columns:
    text_features[col+'_len_word'] = text_features[col].apply(lambda x: len(x.split()))


# In[ ]:


real = text_features[text_features['fraudulent']==0]
fake = text_features[text_features['fraudulent']==1]

fig, ax = plt.subplots(5, 2, figsize=(15, 15))
ax[0, 0].set_title('Word Count of Real Post ')
ax[0, 1].set_title('Word Count of Fake Post ')

for i in range(5):
    for j in range(2):
        if j==0:
            ax[i, j].hist(real[columns[i]+'_len_word'], color='g', bins=15);
            ax[i, j].set_ylabel( columns[i] )
        else:
            ax[i, j].hist(fake[columns[i]+'_len_word'], color='r', bins=15);

plt.show()


# 3. Average Word Length

# In[ ]:


def avg_word_ln(string):
    words = string.split()
    word_len = [len(word) for word in words]
    try:
        return sum(word_len)/len(words)
    except:
        return 0

for col in columns:
    text_features[col+'_avg_word_ln'] = text_features[col].apply(avg_word_ln)


# In[ ]:


real = text_features[text_features['fraudulent']==0]
fake = text_features[text_features['fraudulent']==1]

fig, ax = plt.subplots(5, 2, figsize=(15, 15))
ax[0, 0].set_title('Average Word Count of Real Post ')
ax[0, 1].set_title('Average Word Count of Fake Post ')

for i in range(5):
    for j in range(2):
        if j==0:
            ax[i, j].hist(real[columns[i]+'_avg_word_ln'], color='g', bins=15);
            ax[i, j].set_ylabel( columns[i] )
        else:
            ax[i, j].hist(fake[columns[i]+'_avg_word_ln'], color='r', bins=15);

plt.show()


# **After compare all the basic feature such as word length, character length and avg word length, fake post has less count than the real post.**

# In[ ]:


# delete text_features
del text_features


# # Data Preprocessing
# 1. Converting words into lowercase
# 2. Removing leading white spaces 
# 3. Removing punctuations & stop words
# 4. Lemamtization

# In[ ]:


# Create new feature jd (job description)
df['jd'] = df['title'] + ' ' + df['location'] + ' ' + df['department'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements'] + ' ' + df['benefits'] + ' ' + df['employment_type'] + ' ' + df['required_education'] + ' ' + df['industry'] + ' ' + df['function'] 


# In[ ]:


# drop features
del df['title']
del df['location']
del df['department']
del df['company_profile']
del df['description']
del df['requirements']
del df['benefits']
del df['employment_type']
del df['required_experience']
del df['required_education']
del df['industry']
del df['function']


# In[ ]:


df.head()


# In[ ]:


# Load spacy large model
nlp = spacy.load('en_core_web_lg')


# 1. Converting word into lowercase

# In[ ]:


df['jd'] = df['jd'].apply(str.lower)


# In[ ]:


df['jd'].iloc[0]


# **All characters converted into lower case.**

# 2. Removing extra white spaces, removing Punctuation & Stop words
# 
# **Tokenization**
# * It is a process of splitting a string into the consituent tokens.
# * These tokens may be sentence, words or punctutions and is specific to a percular language(In our case: English)
# 
#   I'm using word_tokenize from **nltk libarary**

# In[ ]:


def remove_punctuation_and_stop_words(s):
    punctuations = list(string.punctuation)
    
    strings = " ".join([token for token in word_tokenize(s) if not token in punctuations+list(STOP_WORDS)])
    return strings
    


# In[ ]:


# Apply above function to the jd feature
df['jd'] = df['jd'].apply(remove_punctuation_and_stop_words)


# In[ ]:


# After removing puctuations and stopwords
df['jd'].iloc[0]


# 3. Lemmatization
# 
#     * It is the process of converting a word lowercase base form or lemma.
#     * This is extremely powerful standarization
#     * Examples:
#             am, are, is --->  be
#             n't        ---->  not
#             've        ---->  have

# In[ ]:


doc = nlp(df['jd'].iloc[0])


# In[ ]:


def lemmatization(s):
    doc = nlp(s)
    return " ".join([token.lemma_ for token in doc])


# In[ ]:


# Apply above function to the jd feature
df['jd'] = df['jd'].apply(lemmatization)


# In[ ]:


df['jd'].iloc[0]


# **NER(Named-entity recognition)**
# 
# 1. An NER is anything that can be denoted with the proper name or a pronoun.
# 2. Indentifying and classifying named entity into predefined category.
# 3. Categorization include **Person**, **Organization**, **Country**, etc.

# **Visualize named entities**

# In[ ]:


# take first record and visualize NER
doc = nlp(df['jd'].iloc[0])

displacy.render(doc, style="ent")


# SpaCy is great library for NER and NER visualization
# Represents: 
# 1. **new york** is GPE(Geopolitical entity, i.e. countries, cities, states.) 
# 2. **mario batali** is Person 
# 3. **twitter** ORG(Organizatio)
# 
# 
# Visualize Dependencies

# In[ ]:


displacy.render(doc, style="dep")


# **WordCloud**

# In[ ]:


# WordCloud Real/Fake post

real = df[df['fraudulent']==0]['jd']
fake = df[df['fraudulent']==1]['jd']


# In[ ]:


# Real WordCloud

plt.figure(figsize = (20,20))
wc = WordCloud(width = 1600 , height = 800 , max_words = 3000).generate(" ".join(real))
plt.imshow(wc , interpolation = 'bilinear')
plt.show()


# In[ ]:


# Fake WordCloud

plt.figure(figsize = (20,20))
wc = WordCloud(width = 1600 , height = 800 , max_words = 3000).generate(" ".join(fake))
plt.imshow(wc , interpolation = 'bilinear')
plt.show()


# In[ ]:




