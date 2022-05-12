#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


data = pd.read_csv("../input/million-headlines/abcnews-date-text.csv")["headline_text"]
data


# In[ ]:


data[8]


# In[ ]:


data = data.sample(n=100000, axis=0)


# In[ ]:


from nltk.tokenize import word_tokenize

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language="english")

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

import string
punct = string.punctuation


# In[ ]:


word_tokenize("this is here")


# In[ ]:


stemmer.stem("jubilate")


# In[ ]:


def process(s):
    for p in punct:
        s = s.replace(p, '')
    s = s.lower()
    s = word_tokenize(s)
    s = [w for w in s if not w in stop_words] #optional
    s = [stemmer.stem(word) for word in s]
    return s


# In[ ]:


process("the happy crocodile eats the crying fish")


# In[ ]:


" ".join(['happi', 'crocodil', 'eat', 'cri', 'fish'])


# In[ ]:


from tqdm import tqdm

for i in tqdm(data.index):
    data[i] = process(data[i])
    data[i] = " ".join(data[i])


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[ ]:


vectorizer = CountVectorizer()
bow_data = vectorizer.fit_transform(data)
bow_data.shape


# In[ ]:


vectorizer = TfidfVectorizer(max_features=3000)
tfidf_data = vectorizer.fit_transform(data).toarray()
tfidf_features = vectorizer.get_feature_names_out()
tfidf_data = pd.DataFrame(tfidf_data, columns=tfidf_features)
tfidf_data


# # Topic Modelling with Latent Dirichlet Allocation (LDA)

# In[ ]:


from sklearn.decomposition import LatentDirichletAllocation


# In[ ]:


lda = LatentDirichletAllocation(n_components=15)


# In[ ]:


mat = lda.fit_transform(tfidf_data)
mat.shape


# In[ ]:


mat[5,:]


# In[ ]:


mat[227]


# In[ ]:


lda.components_.shape


# In[ ]:


lda.components_[:,333]


# In[ ]:


topics = np.argmax(mat, axis=1)


# In[ ]:


topics


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.figure(figsize=(20,6))
sns.countplot(x=topics)
plt.show()


# In[ ]:


topic_word = lda.components_
topic_word.shape


# In[ ]:


topic_word[5,:].shape


# In[ ]:


from wordcloud import WordCloud, STOPWORDS


# In[ ]:


[4,5,7,8][-1:-3:-1]


# In[ ]:


def wordcloud(i):
    words_dist = topic_word[i,:]
    ranked_word_indices = words_dist.argsort()
    ranked_words = [tfidf_features[i] for i in ranked_word_indices][-1:-40:-1]
    ranked_words_in_one_text = " ".join(ranked_words)
    cloud = WordCloud(background_color='black',
              colormap="vlag", stopwords=STOPWORDS.add("australia")).generate(ranked_words_in_one_text)
    plt.figure(figsize=(10,8))
    plt.imshow(cloud)
    plt.title(f"Most common words in topic {i}")
    plt.axis("off")
    plt.show()


# In[ ]:


number_of_topics = topic_word.shape[0]
for i in range(number_of_topics):
    wordcloud(i)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




