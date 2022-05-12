#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import necessary library
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


credits = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv')
movies = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv')


# # Merge the Tables

# In[ ]:


movies = movies.merge(credits,on='title')


# # Remove the columns which are not important

# In[ ]:


movies= movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[ ]:


movies.isnull().sum()


# # Drop blank rows

# In[ ]:


movies.dropna(inplace=True)


# In[ ]:


movies.isnull().sum()


# # For covert a string to integer

# In[ ]:





# In[ ]:


def convert(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i["name"])
    return l


# In[ ]:


movies['genres']  = movies['genres'].apply(convert)


# In[ ]:


movies['genres'].head()


# In[ ]:


#keyword column convert into correct format
movies['keywords'] = movies['keywords'].apply(convert)
movies['keywords'].head()


# In[ ]:


#Cast column
movies['cast'][0]


# Here in cast column we only need the first 3 dictionary which contain the main 3 acctrees name and from the 3 dictionary we only need the name of the acctrees ,lets so it ..
# 

# In[ ]:


def convert3(obj):
    l = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter !=3:
            l.append(i['name'])
            counter+=1
        else:
            break
    return l


# In[ ]:


movies['cast']= movies['cast'].apply(convert3)


# In[ ]:


movies['cast']

#Crew Column
# In[ ]:


movies['crew'][0]

Here we only need the director name for each dictionary.
# In[ ]:


def fetch_dir(obj):
    l = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            l.append(i['name'])
            break
    return l


# In[ ]:


movies['crew'] = movies['crew'].apply(fetch_dir)


# In[ ]:


movies['crew']


# In[ ]:


# we have a problem in overview column. Our overview column is string but we need this column as a list of list .
movies["overview"][0]


# In[ ]:


#Convert overview column as a list
movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[ ]:


movies['overview']


# In[ ]:


#Now we need to remove all spaces from the columns
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[ ]:


# we create the right data format for our recommendation system,last one task left
#Now we concatination all the last 5 col into one columns as a "Tags" column 
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[ ]:


movies['tags']


# In[ ]:


#Now we create new data table that contains only the 'movies_id', 'title' and 'tags' column
movies_update = movies[['movie_id','title','tags']]


# In[ ]:


movies_update.head(2)


# In[ ]:


#now we need to again convert tags column values into string
movies_update['tags']=movies_update['tags'].apply(lambda x:" ".join(x))
movies_update.head(3)


# In[ ]:


#convert into lower case 
movies_update['tags'] =  movies_update['tags'].apply(lambda x:x.lower())


# In[ ]:


movies_update.head(2)


# In[ ]:


#vectorize the tags column with CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words = 'english')


# In[ ]:


vector_tag = cv.fit_transform(movies_update['tags']).toarray()
vector_tag


# In[ ]:


#Now we apply steming to remove duplicate words
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[ ]:


def stem(text):
    y= []
    
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[ ]:


movies_update['tags'] = movies_update['tags'].apply(stem)


# In[ ]:


cv.get_feature_names()


# In[ ]:


#calculate the cosine similarity with one movie to another movie
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


similarity = cosine_similarity(vector_tag)


# In[ ]:


similarity[1]


# In[ ]:


def recommend(movie):
    movie_index = movies_update[movies_update['title'] == movie].index[0]
    distance = similarity[movie_index]
    movies_list = sorted(list(enumerate(distance)),reverse = True, key=lambda x:x[1])[1:11]
    
    for i in movies_list:
        print(movies_update.iloc[i[0]].title)
    return


# In[ ]:


recommend('Batman Begins')

