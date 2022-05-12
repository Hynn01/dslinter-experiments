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


mv = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv')
credit = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv') 


# In[ ]:


mv.head()


# In[ ]:


mv.shape


# In[ ]:


credit.head()


# In[ ]:


movies = mv.merge(credit,on='title')


# In[ ]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[ ]:


movies.head()


# In[ ]:


import ast


# In[ ]:


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 


# In[ ]:


movies.dropna(inplace=True)


# In[ ]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[ ]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[ ]:


import ast

ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[ ]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 


# In[ ]:


movies['cast'] = movies['cast'].apply(convert)
movies.head()


# In[ ]:


movies['cast'] = movies['cast'].apply(lambda x:x[0:3])


# In[ ]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[ ]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[ ]:


movies.sample(5)


# In[ ]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[ ]:


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[ ]:


movies.head()


# In[ ]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[ ]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[ ]:


new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
#new.head()


# In[ ]:


new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[ ]:


vector = cv.fit_transform(new['tags']).toarray()


# In[ ]:


vector.shape


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


similarity = cosine_similarity(vector)


# In[ ]:


similarity


# In[ ]:


new[new['title'] == 'The Lego Movie'].index[0]


# In[ ]:


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)
        


# In[ ]:


recommend('Gandhi')


# In[ ]:


import pickle


# In[ ]:


pickle.dump(new,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:





# In[ ]:




