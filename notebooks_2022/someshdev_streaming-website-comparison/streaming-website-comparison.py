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


# couple of questions for EDA 
# * take a look at all the different columns 
# * drop the columns that you dont need 
# 
# 
# some business questions 
# * How many tv shows and movies have started appearing on netflix over the years (both cumulative and non cumulative) 
# * Is netflix bringing back old shows/movies? 
# * which countries have seen the highest growth in recent years? (Y-O-Y growth) 
# * focusing on top 5 countries in terms of number of shows, see the breakdown in no. of tv shows and movies (what is the trend in the type of shows?) 
# 
# 
# per data point 
# * type --> overall and over the years breakdown 
# * Country --> see the overall dist of which has the most number of shows/movies (and/or) then see y-y growth 
# * Date added --> breakdown by months and year (when do most shows/movies get released) 
# * release year --> check on how many old shows and movies netflix has been releasing on the platform (compare with date added) 
# * rating --> breakdown of rating 
# * listed_in --> find out the breakdown of genres  
# repeat for each streaming platform 
# 
# 
# question for across streaming platforms 
# * dist of shows and movies across diff platforms
# * growth over time overall 
# * and in terms of top 5 countries per streaming platform (as %?) 
# 
# 
# 
# recommender system 
# * building one from the corpus of all the different datasets combined so that we have more data 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# #### EDA of the netflix dataset

# In[ ]:


#read in the netflix dataset 
netflix=pd.read_csv('../input/netflix-shows/netflix_titles.csv')
netflix.head()


# In[ ]:


netflix.shape


# a total of 8807 shows and movies on netflix currently. For this analysis we wont be looking at the director and cast so we will drop those columns 

# In[ ]:


netflix=netflix.drop(['director','cast'],axis=1)
netflix.head()


# Ok let's start our EDA. First thing is we want to see a breakdown of the number of shows and movies overall. 

# In[ ]:


#use a simple group by function to determine this 
netflix.groupby('type').agg({'type':'count'})


# Clearly alot more movies than tv shows overall. but let's see the breakdown over the years. we will use the date that it was added to netflix. 

# In[ ]:



#convert the date added into datetime and crate new column for the year of the addition of the show on netflix 
netflix['date_added']=pd.to_datetime(netflix['date_added'])
netflix['date_added_year']=pd.DatetimeIndex(netflix['date_added']).year
netflix.head()


# In[ ]:


#create a new column called row 
netflix.groupby(['date_added_year','type']).agg({'type':'count'}).unstack().plot(kind = "bar")


# What's very interesting is that there is no major difference between the growth between the different types. both saw a spike at around 2019 and a slight decrease following that. 

# Lets look at the countries next. 

# In[ ]:


netflix['country'].isna().sum()


# Let;s remove the NAs

# In[ ]:


netflix_country=netflix[netflix['country'].isna()==False]
netflix_country['country'].isna().sum()


# In[ ]:


g=netflix_country.groupby('country').agg({'country':'count'})
g.apply(lambda x: x.sort_values(ascending=False)).head()


# ok clearly there are some dirty data here, in the sense that some shows and movies might have been produced or filmed over several countries.BUt for the sake of this we will keep it as it is since the top few countries are what we want to focus on and that tneo thers will jsut add to this top few count. 
# 
# So as we can see US, india, Uk, japan and sk are the ones where alot of them are proeuced. US will obviously have the highest. But let;s take a closer look in terms of netflix bringing more shows and movies form non english speaking areas. For this. let;s remove the shows procuded in US and UK and see what happens. 
# 

# In[ ]:


netflix_country=netflix_country[netflix_country['country']!='United States']
netflix_country=netflix_country[netflix_country['country']!='United Kingdom']


g_2=netflix_country.groupby('country').agg({'country':'count'})
g_2.apply(lambda x: x.sort_values(ascending=False)).head()




    


# Will not take out Canada as it might contain french movies as well. Ok now let;' see how the trends have been over the years.  

# In[ ]:


netflix_country_final=netflix_country[netflix_country['country'].isin(['India','Japan','South Korea','Canada','Spain'])]
netflix_country_final.groupby(['date_added_year','country']).agg({'country':'count'}).unstack().plot(kind = "line")


# While india saw a boom after 2016, the other 4 countries did see a bit of an incraase too over time. however, not as much as we might have expected. nonetheless it looks like netflix is slowly increasing their number of foriegn filksm over time. 

# Let's now look at the trend of when do shows and movies get released. 

# In[ ]:


import matplotlib.pyplot as plt
#create a new month variable
netflix['month']=pd.DatetimeIndex(netflix['date_added']).month
netflix.head()
#plt.scatter(netflix['date_added_year'], netflix['month'], marker='o')
netflix.groupby(['date_added_year','month']).agg({'month':'count'}).unstack()




# Fomr this we can see that it is quite varied, but the general trend is thatm ost shows and movies thend to get released at the later part of the year. 
# 
# Now let's look the proportion at which netflix brings back old shows. 

# In[ ]:


#we see if there is a difference between the year of the releas of the movie/tv show and the year it was added to netflix 
netflix['diff_year']=netflix['date_added_year']-netflix['release_year']
netflix[netflix['diff_year']>0]


# What's interesting is that there is a huge proportion (more than 5000) of old shows/movies being brought into netflix. most likley this does not incluide the netflix originals either.
# 
# Let's move on to genres. As there are more than one genre per movie/show, we should try and break it down indivudally and see what are the popular ones.  

# In[ ]:


netflix_genre=list(netflix['listed_in'])

empty=[]


for x in netflix_genre:
    temp=x.split(', ')
    for y in temp:
        empty.append(y)

empty=pd.DataFrame(empty)
empty.columns=['genre']
group=empty.groupby('genre').agg({'genre':'count'})
group.apply(lambda x: x.sort_values(ascending=False))


# Let's do a quick comparison across the different straming platforms to see how many shows and movies have been added ot the various platforms and hwo they compare among each other 

# In[ ]:


#import the data of the other streaming platforms 
amazon=pd.read_csv('../input/amazon-prime-movies-and-tv-shows/amazon_prime_titles.csv')
disney=pd.read_csv('../input/disney-movies-and-tv-shows/disney_plus_titles.csv')
hulu=pd.read_csv('../input/hulu-movies-and-tv-shows/hulu_titles.csv')


# In[ ]:


amazon['date_added']=pd.to_datetime(amazon['date_added'])
amazon['date_added_year']=pd.DatetimeIndex(amazon['date_added']).year

disney['date_added']=pd.to_datetime(disney['date_added'])
disney['date_added_year']=pd.DatetimeIndex(disney['date_added']).year

hulu['date_added']=pd.to_datetime(hulu['date_added'])
hulu['date_added_year']=pd.DatetimeIndex(hulu['date_added']).year


# In[ ]:


amazon.groupby(['date_added_year']).agg({'date_added_year':'count'})


# In[ ]:


disney.groupby(['date_added_year']).agg({'date_added_year':'count'})


# In[ ]:


hulu.groupby(['date_added_year']).agg({'date_added_year':'count'})


# Might not be the best idea to do a comparison by the number of shows since the timeline for the different streaming platforms look differerent. LEts move on to building a recommender system for the platforms! 

# #### Recommender System 

# Let's see if we can build one using datra from all the differnt sites. Sometimes when people have a sub to more than one of the platofrms they might want to find something that is across platforms that might be something whihc is in tune to what they watch. 
# 
# We can use the cast,genre and director for this as well, however with the amount of missing data across the differnt datasts, it might make mroe sense to  build the recommender based of the description of the movie instead. 

# In[ ]:


#lets prep the data first 
columns=['title','description']
netflix_rec=pd.DataFrame(netflix,columns=columns)
netflix_rec['source']='netflix'

disney_rec=pd.DataFrame(disney,columns=columns)
disney_rec['source']='disney'

hulu_rec=pd.DataFrame(hulu,columns=columns)
hulu_rec['source']='hulu'


amazon_rec=pd.DataFrame(amazon,columns=columns)
amazon_rec['source']='amazon'


corpus=pd.concat([netflix_rec,disney_rec,hulu_rec,amazon_rec],axis=0)
corpus.tail()


# In[ ]:


#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
corpus['description'] = corpus['description'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(corpus['description'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape


# We will use a cosine similarity score to find the best movies/shows 

# In[ ]:


# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[ ]:


#Construct a reverse map of indices and movie titles
indices = pd.Series(corpus.index, index=corpus['title']).drop_duplicates()


# We now create a function to get the top 5 closest movies based on the similarity score.

# In[ ]:


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar movies
    sim_scores = sim_scores[1:6]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    columns=['title','source']
    new=pd.DataFrame(corpus,columns=columns)
    return new.iloc[movie_indices]


# In[ ]:


get_recommendations('Die Hard')


# Time to watch some new movies! 

# #### Closing thoughts 

# The data was well prepped and the EDA can be interchangle for the various datasets form the differnt streaming platforms. There can definitely be more analysis than can be done in the first few steps but those were the quesitons that i could think of for the time being. Do leave some comments on where i can imporve and if you have any questions about the differnt analysis. Recommender systems have become more and more prevalent and this comes to show how simplfieid it has become over the years.
