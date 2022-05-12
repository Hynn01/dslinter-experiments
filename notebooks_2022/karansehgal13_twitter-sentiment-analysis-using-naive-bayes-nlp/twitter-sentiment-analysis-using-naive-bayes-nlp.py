#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import seaborn as sns #for visualization
import matplotlib.pyplot as plt #for visualization
#from jupyterthemes import jtplot #for styling matplotlib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


train_data = pd.read_csv('../input/twitter-sentiment-analysis-hatred-speech/train.csv')
test_data = pd.read_csv('../input/twitter-sentiment-analysis-hatred-speech/test.csv')


# In[ ]:


#jtplot.style(theme = 'monokai', context = 'notebook', grid = False, ticks = False)


# In[ ]:


train_data


# 0 label = Positive Tweet
# 
# 1 label = Negative Tweet

# In[ ]:


train_data.info()


# In[ ]:


train_data.shape


# In[ ]:


train_data['tweet']


# In[ ]:


train_data = train_data.drop('id', axis = 1)


# In[ ]:


train_data.head()


# Exploring the dataset

# In[ ]:


sns.heatmap(train_data.isnull(), yticklabels = False)
plt.show()


# In[ ]:


sns.countplot(x = train_data['label'])
plt.show()


# In[ ]:


len(train_data)


# In[ ]:


length = list()
for i in range(len(train_data)):
    length.append(len(train_data.iloc[i,1]))


# In[ ]:


train_data['length'] = length


# In[ ]:


train_data.head()


# In[ ]:


plt.hist(train_data['length'], bins = 100)
plt.show()


# In[ ]:


train_data.describe()


# Shortest Tweet -

# In[ ]:


train_data[train_data['length'] == min(train_data['length'])]['tweet'].iloc[0]


# Longest Tweet - 

# In[ ]:


train_data[train_data['length'] == max(train_data['length'])]['tweet'].iloc[0]


# Average Length Tweet - 

# In[ ]:


train_data[train_data['length'] == 84]['tweet'].iloc[0]


# In[ ]:


positive = train_data[train_data['label'] == 0]


# In[ ]:


positive.describe()


# In[ ]:


negative = train_data[train_data['label'] == 1]


# In[ ]:


negative


# In[ ]:


negative.describe()


# In[ ]:


from wordcloud import WordCloud


# In[ ]:


sentences = train_data['tweet'].tolist()


# In[ ]:


combined_sentences = " ".join(sentences)


# In[ ]:


plt.imshow(WordCloud().generate(combined_sentences))
plt.show()


# In[ ]:


negative_sentences = negative['tweet'].tolist()


# In[ ]:


combined_negative_sentences = " ".join(negative_sentences)


# In[ ]:


plt.imshow(WordCloud().generate(combined_negative_sentences))
plt.show()


# Cleaning the data

# In[ ]:


import string
string.punctuation


# In[ ]:


import nltk


# In[ ]:


nltk.download('stopwords')


# In[ ]:


from nltk.corpus import stopwords


# In[ ]:


st = stopwords.words('english')
st.append("and")


# In[ ]:


Test = "Goblin and Ninja FoR the win!!"
test_punc_removed = []
for char in Test:
    if char not in string.punctuation:
        test_punc_removed.append(char)
test_punc_removed = ''.join(test_punc_removed)        


# In[ ]:


test_punc_removed


# In[ ]:


test_punc_st_removed = []
for char in test_punc_removed.split():
    if char.lower() not in st:
        test_punc_st_removed.append(char)
test_punc_st_removed = " ".join(test_punc_st_removed)  


# In[ ]:


test_punc_st_removed


# Tokenization

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


sample_data = ["This is my first paper", "This is the second document", "here is the third thi one"]


# In[ ]:


vectorizer = CountVectorizer()


# In[ ]:


X = vectorizer.fit_transform(sample_data)


# In[ ]:


vectorizer.get_feature_names_out()


# In[ ]:


X.toarray()


# Creating a pipeline which removes punctuation, stopwords and peforms tokenization

# In[ ]:


def message_cleaning(message):
    test_punc_removed = [char for char in message if char not in string.punctuation]
    test_punc_removed = ''.join(test_punc_removed)
    test_punc_st_removed = []
    for char in test_punc_removed.split():
        if char.lower() not in st:
            test_punc_st_removed.append(char)
    test_punc_st_removed = " ".join(test_punc_st_removed)  
    return test_punc_st_removed


# In[ ]:


message_cleaning("Goblin and Ninja fOR thE win!!")


# In[ ]:


train_data_clean = train_data['tweet'].apply(message_cleaning)


# In[ ]:


train_data['tweet'][5]


# In[ ]:


train_data_clean[5]


# In[ ]:


data = train_data_clean.tolist()


# In[ ]:


tweets_countvectorizer = CountVectorizer().fit_transform(data).toarray()


# In[ ]:


tweets_countvectorizer.shape


# In[ ]:


X = tweets_countvectorizer
y = train_data['label']


# Using Naive Bayes to classify

# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB #for discrete values


# In[ ]:


NB_classifier = MultinomialNB()


# In[ ]:


NB_classifier.fit(X_train, y_train)


# In[ ]:


y_pred = NB_classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# In[ ]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:


sns.heatmap(cm, annot = True)


# In[ ]:


ac = accuracy_score(y_test, y_pred)
ac


# In[ ]:


print(classification_report(y_test, y_pred))

