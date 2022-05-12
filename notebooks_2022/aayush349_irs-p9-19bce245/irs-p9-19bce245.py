#!/usr/bin/env python
# coding: utf-8

# # IRS Practical 9
# > 19BCE245 - Aayush Shah

# In[ ]:


import re
import string
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

def retrieve_docs_and_clean():
  # Untuk mendapatkan link berita populer
  r = requests.get('https://bola.kompas.com/')
  soup = BeautifulSoup(r.content, 'html.parser')

  link = []
  for i in soup.find('div', {'class':'most__wrap'}).find_all('a'):
      i['href'] = i['href'] + '?page=all'
      link.append(i['href'])

  # Retrieve Paragraphs
  documents = []
  for i in link:
      r = requests.get(i)
      soup = BeautifulSoup(r.content, 'html.parser')

      sen = []
      for i in soup.find('div', {'class':'read__content'}).find_all('p'):
          sen.append(i.text)
      documents.append(' '.join(sen))

  # Clean Paragraphs
  documents_clean = []
  for d in documents:
      document_test = re.sub(r'[^\x00-\x7F]+', ' ', d)
      document_test = re.sub(r'@\w+', '', document_test)
      document_test = document_test.lower()
      document_test = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document_test)
      document_test = re.sub(r'[0-9]', '', document_test)
      document_test = re.sub(r'\s{2,}', ' ', document_test)
      documents_clean.append(document_test)

  return documents_clean


# In[ ]:


docs = retrieve_docs_and_clean()

# Create Term-Document Matrix with TF-IDF weighting
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

# Create a DataFrame
df = pd.DataFrame(X.T.toarray(), index=vectorizer.get_feature_names())
print(df.head())
print(df.shape)


# In[ ]:


docs = retrieve_docs_and_clean()
# Create Term-Document Matrix with TF-IDF weighting
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

# Create a DataFrame
df = pd.DataFrame(X.T.toarray(), index=vectorizer.get_feature_names())
df.head()


# In[ ]:


def get_similar_articles(q, df):
  print("query:", q)
  print("Berikut artikel dengan nilai cosine similarity tertinggi: ")
  q = [q]
  q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0],)
  sim = {}
  for i in range(10):
    sim[i] = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)
  
  sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
  
  for k, v in sim_sorted:
    if v != 0.0:
      print("Nilai Similaritas:", v)
      print(docs[k])
      print()


q1 = 'barcelona'
q2 = 'gareth bale'
q3 = 'shin tae yong'

get_similar_articles(q1, df)
print('-'*100)
get_similar_articles(q2, df)
print('-'*100)
get_similar_articles(q3, df)

