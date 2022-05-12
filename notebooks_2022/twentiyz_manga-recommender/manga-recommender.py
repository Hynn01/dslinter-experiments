#!/usr/bin/env python
# coding: utf-8

# # large parts of this code are inspire from 
# (https://www.kaggle.com/code/yonatanrabinovich/anime-recommendations-project#Preprocessing-and-Data-Analysis-%F0%9F%92%BB)

# In[ ]:


import pandas as pd
import numpy as np
import os


# # Get path to dataframes

# In[ ]:


#list all files under the input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df=pd.read_csv("/kaggle/input/myanimelist-dataset-containingover-5000-manga/mal_manga_df.csv")


# # Clean data

# In[ ]:


df.head()


# In[ ]:


df.info


# In[ ]:


rating=pd.read_csv("/kaggle/input/myanimelist-dataset-containingover-5000-manga/mal_users_manga.csv")


# In[ ]:


rating.head()


# In[ ]:


rating.info


# In[ ]:


print("manga missing values (%):\n")
print(round(df.isnull().sum().sort_values(ascending=False)/len(df.index),4)*100) 
print("\n","*"*50,"\n\nRating missing values (%):\n")
print(round(rating.isnull().sum().sort_values(ascending=False)/len(rating.index),4)*100)


# I scrape the data on sunday and monday i quess Fujoshi were pretty horny during this time

# In[ ]:


print(df['type'].mode())
print(df['genres'].mode())


# # merge
# Our rating dataframe only has mal_id to idenifty manga it would be easier if we have the manga title

# In[ ]:


ratings=rating.merge(df[['title','mal_id']], left_on='manga_id',right_on='mal_id')


# In[ ]:


ratings.drop(['manga_id','mal_id'],axis=1)


# # handling missing values

# In[ ]:


ratings['rating'] = ratings['rating'].apply(lambda x: np.nan if x==-1 else x)
ratings.head(20)


# pivot tables are needed for cosine simmilarty 

# In[ ]:


pivot = ratings.pivot_table(index=['Username'], columns=['title'], values='rating')
pivot.head()


# In[ ]:


import scipy as sp #pivot egineering
from scipy.sparse import csr_matrix


# # Now we will engineer our pivot table in the following steps:
#     Value normalization.
#     Filling Nan values as 0.
#     Transposing the pivot for the next step.
#     Dropping columns with the values of 0 (unrated).
#     Using scipy package to convert to sparse matrix format for the similarity computation.

# In[ ]:


# step 1
pivot_n = pivot.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)

# step 2
pivot_n.fillna(0, inplace=True)

# step 3
pivot_n = pivot_n.T

# step 4
pivot_n = pivot_n.loc[:, (pivot_n != 0).any(axis=0)]

# step 5
piv_sparse = sp.sparse.csr_matrix(pivot_n.values)


# Cosine similarity measures the similarity between two vectors of an inner product space. It is measured by the cosine of the angle between two vectors and determines whether two vectors are pointing in roughly the same direction (https://www.sciencedirect.com/topics/computer-science/cosine-similarity).

# In[ ]:


#ML model
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


manga_similarity = cosine_similarity(piv_sparse)

#Df of anime similarities
manga_sim_df = pd.DataFrame(manga_similarity, index = pivot_n.index, columns = pivot_n.index)


# Function that return manga recommendations note the manga_name has to be the exact same as the title in mal_manga_df.csv

# In[ ]:


def manga_recommendation(manga_name):  
    number = 1
    print('Recommended because you read {}:\n'.format(manga_name))
    for manga in manga_sim_df.sort_values(by = manga_name, ascending = False).index[1:6]:
        print(f'#{number}: {manga}, {round(manga_sim_df[manga][manga_name]*100,2)}% match')
        number +=1  


# In[ ]:


manga_recommendation('Bleach')


# # Manga_recommender but with k-nearstes neightbor

# In[ ]:


import scipy as sp #pivot egineering
import scipy.sparse
pivot2 = ratings.pivot_table(columns='Username', index='title', values="rating")
pivot2.fillna(0, inplace=True)


# In[ ]:


from scipy.sparse import csr_matrix
pivot_sparse = csr_matrix(pivot2.values)


# In[ ]:


from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(metric='cosine',algorithm='brute')


# In[ ]:


model.fit(pivot_sparse)


# return a random manga and recommendation with mangas that are simmilar

# In[ ]:


def random_recommendation(): 
    query_index = np.random.choice(pivot2.shape[0])
    distances, indices = model.kneighbors(pivot2.iloc[query_index,:].values.reshape(1, -1),n_neighbors = 6)
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for {0}:\n'.format(pivot2.index[query_index]))
        else:
            print('{0}: {1}, with distance of {2}:'.format(i, pivot2.index[indices.flatten()[i]], distances.flatten()[i]))


# In[ ]:


random_recommendation()

