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


#importing library
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


#Load the data 
movies=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')
credits=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')


# In[ ]:


movies.head()


# In[ ]:


credits.head()


# In[ ]:


#check the shape of the two dataset 
print("movies dataset shape: ",movies.shape)
print("credits dataset shape: ",credits.shape)


# In[ ]:


# merge the two dataset into one dataset
movies=movies.merge(credits,on="title")
movies.head(1)


# In[ ]:


# removing the unwanted columns 
movies=movies[["movie_id","title","overview","genres","keywords","cast","crew"]]


# In[ ]:


movies.head()


# In[ ]:


movies.info()


# In[ ]:


#chacking the missing data
movies.isnull().sum()


# In[ ]:


#removing the missing data
movies.dropna(inplace=True)


# # preproccing and formating some columns
# 

# In[ ]:


#genres col 
movies.iloc[0].genres


# this format is seems to be wrong ,we can fixed it -----we convert the '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]' format into the ["Action","Adventure","Fantasy","Science Fiction"] format....
# 
# 

# In[ ]:


def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i["name"])
    return l     


# In[ ]:


movies["genres"]=movies["genres"].apply(convert)


# In[ ]:


movies["genres"].head()


# In[ ]:


#keyword columns 
movies["keywords"].head()


# In[ ]:


#convert into correct format
movies["keywords"]=movies["keywords"].apply(convert)
movies["keywords"].head()


# In[ ]:


#cast columns 
movies["cast"][0]


# Here in cast column we only need the first 3 dictionary which contain the main 3 acctrees name and from the 3 dict we only need the name of the acctrees ,lets so it ..
# 

# In[ ]:


def convert3(obj):
    l=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter !=3:
            l.append(i["name"])
            counter+=1
        else:
            break
    return l     


# In[ ]:


movies["cast"]=movies["cast"].apply(convert3)
movies["cast"]


# In[ ]:


#crew column 
movies["crew"][0]


# Here we only need the director name for each dict....lets go

# In[ ]:


def featch_dir(obj): 
    l=[]
    for i in ast.literal_eval(obj):
        if i["job"]=="Director":
            l.append(i["name"])
            break
    return l 


# In[ ]:


movies["crew"]=movies["crew"].apply(featch_dir)
movies["crew"]


# In[ ]:


# we have a problem in overview col that is,our overview col are string but we need this col as a list of list .
movies["overview"][0]


# In[ ]:


#let convert it
movies["overview"]=movies["overview"].apply(lambda x:x.split())
movies["overview"]


# In[ ]:


#finally our dataset look like this
movies.head()


# In[ ]:


#removing the space from all the columns values.
movies["genres"]=movies["genres"].apply(lambda x:[i.replace(" ","") for i in x])
movies["keywords"]=movies["keywords"].apply(lambda x:[i.replace(" ","") for i in x])
movies["cast"]=movies["cast"].apply(lambda x:[i.replace(" ","") for i in x])
movies["crew"]=movies["crew"].apply(lambda x:[i.replace(" ","") for i in x])
movies.head()


# In[ ]:


# we create the right data format for our recommendation system,last one task left
#we need to concatination all the last 5 col into one columns 
movies["tags"]=movies["overview"]+movies["genres"]+movies["keywords"]+movies["cast"]+movies["crew"]
movies["tags"][0]


# In[ ]:


#Now we don't need the last 5 col ,so lets remove them.
new_df=movies[["movie_id","title","tags"]]
new_df.head()


# In[ ]:


#now we need to again convert tag col values into string 
new_df["tags"]=new_df["tags"].apply(lambda x:" ".join(x))
new_df.head()


# In[ ]:


#convert into lower case 
new_df["tags"]=new_df["tags"].apply(lambda x:x.lower())


# # Creating the Embadding vector

# In[ ]:


#use stemmer to create the root word
ps=PorterStemmer()

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i)) 
    return " ".join(y) 


# In[ ]:


new_df["tags"]=new_df["tags"].apply(stem)
new_df["tags"][0]


# In[ ]:


#vectorize the tag col with CV
cv=CountVectorizer(max_features=5000,stop_words="english")
vector_tag=cv.fit_transform(new_df["tags"]).toarray()
vector_tag


# In[ ]:


#calculate the cosine similarity one movie to another movie
similarity=cosine_similarity(vector_tag)
similarity[0]


# In[ ]:


similarity.shape


# In[ ]:


# building the movie recommendation function 

def recommend(movie):
    movie_index= new_df[new_df['title']==movie].index[0]
    distance=similarity[movie_index]
    movie_list=sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
    


# In[ ]:


# lets see the recommendation
moviess=["The Dark Knight Rises","Avatar","Spectre","The Fast and the Furious","Batman"]
for i in moviess:
    print("Movie name is :",i)
    print("Recommended movie are: \n")
    recommend(i)
    print("\n")
    print("<------------------------------------------>")
    print("\n")


# # Plz consider a upvote for me ,Thank you ,happy kaggling!!!!!!!!!!!!!!!!!!
