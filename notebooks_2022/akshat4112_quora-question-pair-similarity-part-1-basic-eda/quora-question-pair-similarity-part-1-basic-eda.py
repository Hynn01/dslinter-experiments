#!/usr/bin/env python
# coding: utf-8

# **Problem Statement**
# * Identify which question asked on Quora are duplicates of question that have already been asked.
# * This could be useful to instantly provide answers to questions that have already been answered
# * We are tasked with predicting whethet a pair of questions are duplicate or not

# **Source/Useful Links**
# * Source: https://www.kaggle.com/competitions/quora-question-pairs/data
# * Blog1: https://quoraengineering.quora.com/Semantic-Question-Matching-with-Deep-Learning
# * Blog2: https://towardsdatascience.com/identifying-duplicate-questions-on-quora-top-12-on-kaggle-4c1cf93f1c30

# **Real World/Business Objectives and Constraints**
# 1. The cost of a mis-classification can be very high.
# 2. You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.
# 3. No strict latency concerns
# 4. Interpretaiblity is partially important

# **Machine Learning Problem** <br>
# **Data Overview**
# - Data will be in Train.csv
# - Train.csv contains: qid1, qid2, question1, question2, is_duplicate
# - Size of Train.csv - 60 Mb
# - Number of rows in Train.csv = 404,290

# **Example Data Point**
# <pre>
# "id","qid1","qid2","question1","question2","is_duplicate"
# "0","1","2","What is the step by step guide to invest in share market in india?","What is the step by step guide to invest in share market?","0"
# "1","3","4","What is the story of Kohinoor (Koh-i-Noor) Diamond?","What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?","0"
# "7","15","16","How can I be a good geologist?","What should I do to be a great geologist?","1"
# "11","23","24","How do I read and find my YouTube comments?","How can I see all my Youtube comments?","1"
# </pre>

# **Mapping Real World problemt o ML problem**<br>
# **Type of Machine Learning Problem**
# It is a binary classification problem, for a given pair of questions we need to predict if they are duplicate or not.
# 
# **Performance Metric**<br>
# Source: https://www.kaggle.com/c/quora-question-pairs#evaluation
# 
# Metric(s): 
# * log-loss : https://www.kaggle.com/wiki/LogarithmicLoss
# * Binary Confusion Matrix
# 
# **Train and Test Construction**
# <p>  </p>
# <p> We build train and test by randomly splitting in the ratio of 70:30 or 80:20 whatever we choose as we have sufficient points to work with. </p>

# **Exploratory Data Analysis**

# In[ ]:


get_ipython().system('pip3 install distance')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import os
import gc
import shutil

import re
from nltk.corpus import stopwords
import distance
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup


# In[ ]:


shutil.unpack_archive('../input/quora-question-pairs/train.csv.zip', '.')
shutil.unpack_archive('../input/quora-question-pairs/test.csv.zip', '.')


# In[ ]:


df = pd.read_csv("./train.csv")

print("Number of datapoints:", df.shape[0])


# In[ ]:


df.head()


# In[ ]:


df.info()


# * We are given a minimal number of data fields here, consisting of:
# * id: Looks like a simple rowID
# * qid{1,2}: The unique ID of each question in pair
# * question(1,2): The actual textual content of the questions.
# * is_duplicate: The label that we are trying to predict - whether the two questions are duplicates of each other

# **Distribution of data points amoung output classes**
# * Numbe of duplicates and non-duplicate questions

# In[ ]:


df.groupby("is_duplicate")['id'].count().plot.bar()


# In[ ]:


print('~> Total number of question pairs for training:\n {}'.format(100 - round(df['is_duplicate'].mean()*100, 2)))


# In[ ]:


print('~> Question pairs are Similar:\n {}'.format(round(df['is_duplicate'].mean()*100, 2)))


# **Number of unique questions**

# In[ ]:


qids = pd.Series(df['qid1'].tolist() + df['qid2'].to_list())
unique_qs = len(np.unique(qids))

qs_morethan_onetime = np.sum(qids.value_counts() > 1)
print('Total number of Unique Questions are: {}\n'.format(unique_qs))

print('Number of unique questions that appear more than once time: {} ({}%)\n'.format(qs_morethan_onetime,qs_morethan_onetime/unique_qs*100))

print('Max number of time a single question is repeated: {}\n'.format(max(qids.value_counts())))


# In[ ]:


x = ["unique_questions", "Repeated Questions"]
y = [unique_qs, qs_morethan_onetime]

plt.figure(figsize=(10,6))
plt.title("Plot representing unique and repeated questions")
sns.barplot(x,y)
plt.show()


# **Checking for Duplicates**

# In[ ]:


#checking whether there aaaare any repeated pair of questions

pair_duplicates = df[['qid1', 'qid2', 'is_duplicate']].groupby(['qid1', 'qid2', ]).count().reset_index()
print("Number of duplicate questions", (pair_duplicates).shape[0] - df.shape[0])


# **Number of occurences of each question**

# In[ ]:


plt.figure(figsize=(20, 10))

plt.hist(qids.value_counts(), bins=160)

plt.yscale('log')

plt.title('Log-Histogram of question appearance counts')

plt.xlabel("Number of occurences of questions")

plt.ylabel("Number of questions")

print('Maximum number of times a single question is repeated:  {}\n'.format(max(qids.value_counts())))


# **Checking for Null values**

# In[ ]:


#Checking whether there are any rows with null values

nan_rows = df[df.isnull().any(1)]
print(nan_rows)


# There are two rows with null values in question2

# In[ ]:


#Fill the null values with ' '

df = df.fillna('')
nan_rows = df[df.isnull().any(1)]
print(nan_rows)


# **Basic Feature Extraction**<br>
# Let us now contruct few features like:
# 
# * freq_qid1: Frequency of qid1's
# * freq_qid2: Frequency of qid2's
# * q1len: Length of q1
# * q2len: Length of q2
# * q1_n_words: Number of words in Questions 1
# * q2_n_words: Number of words in Questions 2
# * word_Common: (Number of common unique words in Question 1 and Questions 2)
# * word_Total: Total num of words in Question1 + Total num of words in Question 2
# * word_share : (word_common)/(word_Total)
# * freq_q1 + freq_q2: sum total of frequency of qid1 and qid2
# * freq_q1 - freq_q2: absolute difference of frequency of qid1 and qid2

# In[ ]:


if os.path.isfile('./df_fe_without_preprocessing_train.csv'):
    df = pd.read_csv('./df_fe_without_preprocessing_train.csv', encoding='latin-1')
else:
    df['freq_qid1'] = df.groupby('qid1')['qid1'].transform('count')
    df['freq_qid2'] = df.groupby('qid2')['qid2'].transform('count')
    df['q1len'] = df['question1'].str.len()
    df['q2len'] = df['question2'].str.len()
    df['q1_n_words'] = df['question1'].apply(lambda row :len(row.split(" ")))
    df['q2_n_words'] = df['question2'].apply(lambda row :len(row.split(" ")))
    
    def normalized_word_Common(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split()))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split()))
        return 1.0* len(w1 & w2)
    df['word_Common'] = df.apply(normalized_word_Common, axis=1)
    
    def normalized_word_Total(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split()))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split()))
        return 1* (len(w1) + len(w2))
    df['word_Total'] = df.apply(normalized_word_Total, axis = 1)
    
    def normalized_word_share(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split()))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split()))
        return 1.0*len(w1 & w2) / (len(w1) + len(w2))
    df['word_share'] = df.apply(normalized_word_share, axis = 1)
    
    df['freq_q1+q2'] = df['freq_qid1'] + df['freq_qid2']
    df['freq_q1-q2'] = abs(df['freq_qid1'] - df['freq_qid2'])
    
    df.to_csv("df_fe_without_preprocessing_train.csv", index= False)
    
df.head()


# **Analysis of some of the extracted features** <br>

# In[ ]:


print("Minimum length of the question in q1:", min(df['q1_n_words']))
print("Minimum length of the question in q2:", min(df['q2_n_words']))

print("Number of Questions with minimum length [question1] :", df[df['q1_n_words']==1].shape[0])
print("Number of Questions with minimum length [question2] :", df[df['q2_n_words']==1].shape[0])


# **Feature word share**

# In[ ]:


plt.figure(figsize=(12,8))

plt.subplot(1,2,1)
sns.violinplot(x = 'is_duplicate', y='word_share', data = df[0:])

plt.subplot(1,2,2)
sns.distplot(df[df['is_duplicate'] == 1.0]['word_share'][0:], label = "1", color="red")
sns.distplot(df[df['is_duplicate'] == 0.0]['word_share'][0:], label = "0", color="blue")
plt.show()


# - The distributions for normalized word_share have some overlap on the far right-hand side, i.e., there are quite a lot of questions with high word similarity
# - The average word share and Common no. of words of qid1 and qid2 is more when they are duplicate(Similar)

# **Feature word common**

# In[ ]:


plt.figure(figsize=(12,8))

plt.subplot(1,2,1)
sns.violinplot(x = 'is_duplicate', y='word_Common', data = df[0:])

plt.subplot(1,2,2)
sns.distplot(df[df['is_duplicate'] == 1.0]['word_Common'][0:], label = "1", color="red")
sns.distplot(df[df['is_duplicate'] == 0.0]['word_Common'][0:], label = "0", color="blue")
plt.show()


# - The distribution of the word Common feature in similar and non similar questions are highly overlapping. 

# ### End of Notebook
# Advance Feature Engineering is in Part 2 [Click Here](https://www.kaggle.com/code/akshat4112/quora-question-pair-similarity-part-2-adv-eda)

# In[ ]:




