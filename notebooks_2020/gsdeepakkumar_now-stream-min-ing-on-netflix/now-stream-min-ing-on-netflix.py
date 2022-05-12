#!/usr/bin/env python
# coding: utf-8

# # Mining Netflix 

# According to [Wikipedia](https://en.m.wikipedia.org/wiki/Netflix), Netflix, Inc. is an American media-services provider and production company headquartered in Los Gatos, California, founded in 1997 by Reed Hastings and Marc Randolph in Scotts Valley, California. The company's primary business is its subscription-based streaming service which offers online streaming of a library of films and television programs, including those produced in-house.As of April 2019, Netflix had over 148 million paid subscriptions worldwide, including 60 million in the United States, and over 154 million subscriptions total including free trials.
# 
# While Netflix was launched in India few years ago,keeping in mind the largest internet and smartphone market,inorder to expand its customer base to tire 1 and tire 2 cities ,it launched its first mobile only subscription plan of Rs.199/USD 2.81 per month.India is the only market where Netflix has launched this mobile only plan.The base netflix plan in India costs around Rs.499/USD 7.04 per month - the most expensive compared to Amazon prime (Rs.129/USD 1.82 per month) and Hotstar(Rs.299/USD 4.22 per month).
# 
# The content library of Netflix is one of the most outstanding and has appeal to all sections of people.Also, the streaming quality,the advantages of binge-watching,ever rising price of movie tickes(and popcorns!!!) are some of the reasons for the people to prefer OTT platforms like Netflix.
# 
# The data provided here lists movies and TV shows found in Netflix along with their cast,rating and other details.We will explore this content and mine interesting insights.
# 
# I have also published a [tableau workbook](https://public.tableau.com/profile/deepak.kumar5649#!/vizhome/NetflixMovieandShowAnalysis/Introduction) accompanying this kernel . Feel feel to check that out and let me know if you like my work.

# Updated :**Version 10** - Reran the data and changed the results of few analysis inline with the update in dataset.

# In[ ]:


get_ipython().run_cell_magic('HTML', '', "\n<div class='tableauPlaceholder' id='viz1588308320617' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ne&#47;NetflixMovieandShowAnalysis&#47;Introduction&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='NetflixMovieandShowAnalysis&#47;Introduction' /><param name='tabs' value='yes' /><param name='toolbar' value='no' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ne&#47;NetflixMovieandShowAnalysis&#47;Introduction&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1588308320617');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# ### Loading the library and  data

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')


# In[ ]:


kaggle=1

if kaggle==0:
    data=pd.read_csv('../data/netflix_titles.csv')
else:
    data=pd.read_csv('../input/netflix-shows/netflix_titles.csv')


# ## Summary of the data

# In[ ]:


data.head()


# In[ ]:


print(f'There are {data.shape[0]} rows and {data.shape[1]} columns')


# In[ ]:


data.dtypes


# In[ ]:


### Checking the null values:

data.isnull().any()


# * From the above datatype description we see that the release_year is of integer type,date added is object type whose datatype should have been date .
# * When we check the null values , we see that there are 5 columns having null values - director,cast,country,date_added and rating.
# 
# 

# In[ ]:


### Changing the date added and release_year to date:

data['date_added']=pd.to_datetime(data['date_added'])
data['release_year']=pd.to_datetime(data['release_year'],format="%Y")
data['release_year']=data['release_year'].dt.year


# In[ ]:


data.head()


# ## Exploratory Data Analysis

# ### Number of titles

# In[ ]:


## Check the number of titles present in the dataset:

print(f'There are {data.show_id.nunique()} shows in the netflix dataset')


# We understand that each row is a unique title .Lets check the categories of titles available.

# In[ ]:


(data['type'].value_counts()/data.shape[0])*100


# 69 % of the titles in this dataset is of Movie type whereas 30 % is of TV show type.

# ### Number of titles by date addded:

# Lets create two columns - year added and month added and understand the trend of titles added in the platform.

# In[ ]:


data['year_added']=data['date_added'].dt.year
data['month_added']=data['date_added'].dt.month


# In[ ]:


data['year_added'].value_counts().sort_index()


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x='year_added',data=data)
plt.ylabel('Count of titles',fontsize=12)
plt.xlabel('Year',fontsize=12)
plt.title("Count of titles over the year",fontsize=15)
plt.show()


# From the plot above , we understand that the number of titles over the years have increased.Lets plot the same with respect to the type of title. - Movie or TV Show.

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x='year_added',data=data,hue='type')
plt.ylabel('Count of titles',fontsize=12)
plt.xlabel('Year',fontsize=12)
plt.title("Count of titles over the year by type",fontsize=15)
plt.show()


# Clearly , we understand that the number of tv shows added to the platform has increased over the years whereas the numer of movies has hit a peak in 2019 and seen a dip in 2020.Lets check the growth of the TV shows over the year and which year has seen the highest growth.

# In[ ]:


tvshow=data[data['type']=='TV Show']['year_added'].value_counts().sort_index()


# In[ ]:


tvshow.pct_change()


# There has been a 5.16 % increase in TV shows in the year 2016.

# ### Number of titles by month added:

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x='month_added',data=data)
plt.ylabel("Count of titles",fontsize=12)
plt.xlabel("Month",fontsize=12)
plt.title("Number of titles by month added",fontsize=15)
plt.show()


# Maximum titles were added between the month of Oct-Dec.Lets check whether the trend is change is different between TV show and movie title.

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x='month_added',data=data,hue='type')
plt.ylabel("Count of titles",fontsize=12)
plt.xlabel("Month",fontsize=12)
plt.title("Number of titles by month added",fontsize=15)
plt.show()


# The trend is the same for TV show and movie titles.Oct-Dec were the months where most of the titles were added.

# ### Movie/TV Show Listing

# From the summary of the data , there is a column called listed_in which specifies what is the genre of each of the movies.There are multiple genres for a single title.Lets check how is the genre distributed in this dataset.

# In[ ]:


data['listed_in']=data['listed_in'].astype('str')
data['num_listing']=data['listed_in'].apply(lambda x:len(x.split(",")))


# In[ ]:


listing=set()
count_listings=Counter()

for listings in data['listed_in']:
    listings=listings.split(",")
    listing.update([lst.strip() for lst in listings])
    for lst in listings:
        count_listings[lst.strip()]+=1


# In[ ]:


listings_title=pd.DataFrame.from_dict(count_listings,orient='index')
listings_title.sort_values(0,ascending=False,inplace=True)
listings_title.rename(columns={0:'count'},inplace=True)


# In[ ]:


data['num_listing'].describe()


# In[ ]:


print(f'There are {len(listings_title)} genres in the database')


# In[ ]:


## Top 10 genre titles available:

plt.figure(figsize=(12,8))
sns.barplot(x=listings_title[:10].index,y='count',data=listings_title[:10])
plt.ylabel("Count of genres",fontsize=12)
plt.xlabel("Genres",fontsize=12)
plt.title("Genre Title Count",fontsize=15)
plt.xticks(rotation=90)


# From the plot of top 10 genre titles available,there are more than 2000+ titles listed as International Movies followed by ~2000 dramas and ~1300 comedies. 

# ### Countries of the titles:

# It is not sure from the data description what exactly the countries column refer to -I am assuming it to be the country of origin of the title.Lets analyse this column in a similar way as that of listed_in column.

# In[ ]:


data['country']=data['country'].astype('str')
data['num_countries']=data['country'].apply(lambda x:len(x.split(",")))

country=set()
count_country=Counter()

for c in data['country']:
    if c not in ['nan','']:
        c=c.split(",")
        country.update([cs.strip() for cs in c])
        for cs in c:
            count_country[cs.strip()]+=1
        
        
countries_df=pd.DataFrame.from_dict(count_country,orient='index')
countries_df.sort_values(0,ascending=False,inplace=True)
countries_df.rename(columns={0:'count'},inplace=True)


# In[ ]:


data['num_countries'].describe()


# In[ ]:


print(f'There are {len(country)} countries titles in the database')


# In[ ]:


## Top 10 countries titles available:

plt.figure(figsize=(12,8))
sns.barplot(x=countries_df[:10].index,y='count',data=countries_df[:10])
plt.ylabel("Count of countries",fontsize=12)
plt.xlabel("Country",fontsize=12)
plt.title("Countries Count",fontsize=15)
plt.xticks(rotation=90)


# * There are 3000+ titles from United States followed by India and UK .But the difference in the number of titles of US and India is very high.

# ### Ratings 

# In[ ]:


data['rating'].value_counts()


# TV-MA and TV-14 are the most common ratings available in this database.Lets check the rating with respect to type.
# 

# ### Rating with respect to type

# In[ ]:


data.groupby('type')['rating'].value_counts()


# There is no difference between the 1st and 2nd movie rating with respect to type.Is there any rating unique to a TV show or to a movie ?Lets check.

# In[ ]:


## check if there exist unique rating type in tv show 
set(data[data['type']=='TV Show'].rating.unique())-set(data[data['type']=='Movie'].rating.unique())


# In[ ]:


#check if there exist unique movie rating type
set(data[data['type']=='Movie'].rating.unique())-set(data[data['type']=='TV Show'].rating.unique())


# From the above output,it is seen that G,NC-17,PG,PG-13 and UR are only certified for movies and no tv show has this rating.

# ### Directors

# In[ ]:


## How many titles have more than one directors:
data['director']=data['director'].astype('str')
data['num_directors']=data['director'].apply(lambda x:len(x.split(",")))


# In[ ]:


multi_director=data.loc[data['num_directors']>1,]


# In[ ]:


multi_director.head()


# In[ ]:


multi_director['num_directors'].value_counts().sort_index()


# The maximum number of directors collaborated for a movie is 13.Let us check the titles where the number of directors is more than 5.

# In[ ]:


multi_director.loc[multi_director['num_directors']>5,['title','director','num_directors']].sort_values(by='num_directors',ascending=False)


# In[ ]:


###Directors with most titles:
director=data.loc[data['director']!='nan','director'].str.split(",",expand=True).stack().reset_index()
director.rename(columns={0:'director'},inplace=True)


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x=director['director'],data=director,order=director['director'].value_counts()[:10].index)
plt.xlabel("Director",fontsize=12)
plt.ylabel("Count of titles",fontsize=12)
plt.title("Top 10 Directors with most titles",fontsize=15)
plt.xticks(rotation=90)


# Raul Campos and Jan Suter are the directors having 18 titles each.Lets check individually by the type of title.

# In[ ]:


director=data.loc[data['director']!='nan',].set_index('type').director.str.split(",",expand=True).stack().reset_index('type')
director.rename(columns={0:'directors'},inplace=True)


# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(121)
sns.countplot(x=director.loc[director['type']=='Movie','directors'],data=director[director['type']=='Movie'],order=director.loc[director['type']=='Movie','directors'].value_counts()[:10].index)
plt.xlabel("Director",fontsize=12)
plt.ylabel("Count of titles",fontsize=12)
plt.title("Top 10 Directors(Movie) with most titles",fontsize=15)
plt.xticks(rotation=90)
plt.subplot(122)
sns.countplot(x=director.loc[director['type']=='TV Show','directors'],data=director[director['type']=='TV Show'],order=director.loc[director['type']=='TV Show','directors'].value_counts()[:10].index)
plt.xlabel("Director",fontsize=12)
plt.ylabel("Count of titles",fontsize=12)
plt.title("Top 10 Directors(TV Show) with most titles",fontsize=15)
plt.xticks(rotation=90)


# Analysing the top 10 list between movie and tv show , it is seen that there is no overlap between the directors.Lets check if there exist any director who have directed both a tv show and a movie.

# In[ ]:


### directors who have both tv show and movie title:
all_dup=director.groupby('directors')['type'].nunique()


# In[ ]:


all_dup[all_dup>1]


# The above directors have both tv show and movie to their credit.
