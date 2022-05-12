#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import nltk

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('../input/amazon-fine-food-reviews/Reviews.csv')
df


# In[ ]:


df = df.sample(500)
df.head()


# In[ ]:


import plotly.express as px

px.histogram(data_frame=df, x='Score', title='Score Histogram')


# # NLTK Basics

# In[ ]:


example = np.random.choice(df.Text)
print(example)    


# In[ ]:


tokens = nltk.word_tokenize(example)
tokens[:10]


# Believe me it is better than splitting on spaces.

# In[ ]:


tags = nltk.pos_tag(tokens)
tags[:10]


# In[ ]:


# named entity Chunking
entities = nltk.chunk.ne_chunk(tags)
entities.pprint()


# # VADER Sentiment analysis
# 
# Might like to take a look at, https://github.com/nltk/nltk/blob/develop/nltk/sentiment/vader.py#L411

# In[ ]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()

sia.polarity_scores('I love coding in Python!')


# In[ ]:


sia.polarity_scores('A war causes only distruction. It is never going to be a solution.')


# In[ ]:


sia.polarity_scores(example)


# In[ ]:


df['polarity'] = df.Text.apply(sia.polarity_scores)
df


# In[ ]:


# Extract pos neg neu and compound scores from polarity and remove polarity column

df['polarity_neg'] = df.polarity.apply(lambda x:x['neg'])
df['polarity_pos'] = df.polarity.apply(lambda x:x['pos'])
df['polarity_neu'] = df.polarity.apply(lambda x:x['neu'])
df['polarity_compound'] = df.polarity.apply(lambda x:x['compound'])

df = df.drop('polarity', axis=1)
df


# In[ ]:


sns.set_style('dark')
sns.barplot(data=df, x='Score', y='polarity_compound')


# In[ ]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,3, figsize=(20,5))

sns.barplot(data=df, x='Score', y='polarity_pos', ax=ax[0])
sns.barplot(data=df, x='Score', y='polarity_neg', ax=ax[1])
sns.barplot(data=df, x='Score', y='polarity_neu', ax=ax[2])

ax[0].set_title('Positive')
ax[1].set_title('Negative')
ax[2].set_title('Neutral')
plt.show()


# Which matches our intution.

# # RoBerta for sentiment Analysis

# In[ ]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from scipy.special import softmax

task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# In[ ]:


def roberta_result(text):
    try:
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        res = {"rob_neg":scores[0],
              "rob_neu":scores[1],
              "rob_pos":scores[2]}
    except:
        res = {"rob_neg":0,
              "rob_neu":0,
              "rob_pos":0}
    return res


# In[ ]:


df['roberta'] = df.Text.apply(roberta_result)


# In[ ]:


df.roberta.values[0]['rob_neg']


# In[ ]:


df['rob_neg'] = df.roberta.apply(lambda x:x['rob_neg'])
df['rob_neu'] = df.roberta.apply(lambda x:x['rob_neu'])
df['rob_pos'] = df.roberta.apply(lambda x:x['rob_pos'])

df = df.drop('roberta', axis=1)
df


# In[ ]:


# let us try to see relation between 'polarity_neg', 'polarity_pos', 'polarity_neu', 'polarity_compound',
       #'rob_neg', 'rob_neu', 'rob_pos' 
sns.pairplot(df, vars=['polarity_neg', 'polarity_pos', 'polarity_neu', 'polarity_compound',
       'rob_neg', 'rob_neu', 'rob_pos'], hue='Score', palette='magma')


# In[ ]:


# Text which could not be processed because they were too lengthy
df[(df.rob_neg == 0) & (df.rob_neu == 0) & (df.rob_pos ==0)].Text.values[0]


# In[ ]:


encoded_input = tokenizer(example, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
res = {"rob_neg":scores[0],
       "rob_neu":scores[1],
       "rob_pos":scores[2]}
res
vader = sia.polarity_scores(example)
print(f'Given statement is : {example}\n\nVader output : {vader}\n\nRoberta output : {res}')


# # Compare Roberta and Vader

# In[ ]:


# Score 1 with highest positive score by roberta
df.query('Score == 1').sort_values(by='rob_pos', ascending=False).Text.values[0]


# In[ ]:


# Score 1 with highest positive score by Vader
df.query('Score == 1').sort_values(by='polarity_pos', ascending=False).Text.values[0]


# In[ ]:


# Score 5 with highest positive score by roberta
df.query('Score == 5').sort_values(by='rob_neg', ascending=False).Text.values[0]


# In[ ]:


# Score 5 with highest positive score by Vader
df.query('Score == 5').sort_values(by='polarity_neg', ascending=False).Text.values[0]


# # Transformer Pipelines

# In[ ]:


# following pipeline would download all the weights and model for the given task : 'sentiment-analysis'
tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
sent_pipeline = pipeline("sentiment-analysis", **tokenizer_kwargs)


# In[ ]:


sent_pipeline('I love Python!')


# In[ ]:


sent_pipeline('I hate fighting')


# In[ ]:


df['Pipeline'] = df.Text.apply(sent_pipeline)


# In[ ]:


df['label'] = df.Pipeline.apply(lambda x:x[0]['label'])
df['Pipeline Score'] = df.Pipeline.apply(lambda x:x[0]['score'])
df = df.drop('Pipeline', axis=1)
df = df[['polarity_neg', 'polarity_pos', 'polarity_neu', 'polarity_compound',
       'rob_neg', 'rob_neu', 'rob_pos','label','Pipeline Score', 'Score']]
df


# In[ ]:


df


# In[ ]:


#s = 0
for i in range(1,6):
    n = df.query(f'Score == {i} and label == "POSITIVE"').shape[0]
    #s += n
    print(f'No. of rows with Rating = {i} and label = POSITIVE is : {n}')
    n = df.query(f'Score == {i} and label == "NEGATIVE"').shape[0]
    print(f'No. of rows with Rating = {i} and label = NEGATIVE is : {n}\n{100*"*"}')
    #s += n
#s was created to check if we cover all rows on df


# In[ ]:




