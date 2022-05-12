#!/usr/bin/env python
# coding: utf-8

# ![](https://media.giphy.com/media/tWRJEZeQeHAqs/giphy.gif)

# <body>
#     <font size=6 color="red">Background checkup</font>
# </body>
#  

# The basic idea of analyzing the Zomato dataset is to get a fair idea about the factors affecting the establishment of different types of restaurant at different places in Bengaluru, aggregate rating of each restaurant, Bengaluru being one such city has more than 12,000 restaurants with restaurants serving dishes from all over the world. With each day new restaurants opening the industry has’nt been saturated yet and the demand is increasing day by day. Inspite of increasing demand it however has become difficult for new restaurants to compete with established restaurants. Most of them serving the same food. Bengaluru being an IT capital of India. Most of the people here are dependent mainly on the restaurant food as they don’t have time to cook for themselves. With such an overwhelming demand of restaurants it has therefore become important to study the demography of a location.
# 
# - What kind of a food is more popular in a locality. 
# - Do the entire locality loves vegetarian food. If yes then is that locality populated by a particular sect of people for eg. Jain, Marwaris, Gujaratis who are mostly vegetarian. These kind of analysis can be done using the data, by studying the factors such as • Location of the restaurant • Approx Price of food  Theme based restaurant or not 
# -  Which locality of that city serves that cuisines with maximum number of restaurants • The needs of people who are striving to get the best cuisine of the neighborhood • Is a particular neighborhood famous for its own kind of food.

# <font size=5 color="violet">If you like my work,please consider <font color='red'>upvoting</font> my kernel.</font>

# <font size=5 color='blue'> What is in this kernel</font>

# 1. [Loading required libraries](#1)
# 2. [Getting basic ideas about the data](#2)
# 3. [Exploratory data analysis](#3)
# 
#      1. [ Which are the top restaurant chains in Bangaluru?](#4)
#      2. [How many of the restuarants do not accept online orders?](#5)
#      3. [What is the ratio b/w restaurants that provide and do not provide table booking ?](#25)
#      4. [ Rating distribution](#6)
#      5. [Is there any difference b/w votes of restaurants accepting and not accepting online orders?](#7)
#      6. [Which are the most common restaurant type in Banglore?](#8)
#      7. [Cost factor ?](#9)
#      8. [Which are the foodie areas?](#10)
#      9. [Which are the most common cuisines in each locations?](#11)
#      10. [Extracting location information using Geopy](#12)
#      11. [Heatmap of restaurant count on each location](13)
#      12. [Which are  the most popular cuisines of Bangalore?](#14)
#      13. [Heatmap of North Indian restaurants](#15)
#      14. [What about South  Indian cuisines?](#16)
#      15. [Analysing Restaurant chains ](#17)
#             16. [Which are the most popular casual dining restaurant chains?](#18)
#                    1. [Where are their outlets located?](#19)
#             17. [Top quick bites restaurant chains in Banglore](#20)
#                    1. [Where are their outlets located?](#21)
#             18. [Top Cafes of Banglore?](#22)
#                    1. [Where are their outlets located?](#23)
#             19. [Wordcloud of dishes liked by cuisines.](#24)
#      20. [Analysing Reviews](#26)
#             21. [Wordcloud of restaurant reviews](#27)
#             22. [Rating distribution](#28)
# 4. [Topic modelling](#33)
#      23. [Topic modelling for positive comments](#34)
#      24. [Topic modelling for positive comments](#35)
#          
#             
# 5. [Sentimental Analysis on Reviews](#29)
#      23. [Data preparation](#30)
#      24. [Building our model](#31)
#      25. [Validation](#32)
#      

# ## [Loading libraries]()<a id="1"></a> <br>
# 

# In[ ]:




import numpy as np 
import pandas as pd
import os
import seaborn as sns
print(os.listdir("../input"))
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=False)
from wordcloud import WordCloud
from geopy.geocoders import Nominatim
from folium.plugins import HeatMap
import folium
from tqdm import tqdm
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from gensim.models import word2vec
import nltk
# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv("../input/zomato.csv")


# ## [Getting Basic Ideas]()<a id="2"></a> <br>
# 

# In[ ]:


print("dataset contains {} rows and {} columns".format(df.shape[0],df.shape[1]))


# In[ ]:


df.info()


# In[ ]:


df.head()


# **Columns description**
# 
# - **url**
# contains the url of the restaurant in the zomato website
# 
# - **address**
# contains the address of the restaurant in Bengaluru
# 
# - **name**
# contains the name of the restaurant
# 
# - **online_order**
# whether online ordering is available in the restaurant or not
# 
# - **book_table**
# table book option available or not
# 
# - **rate**
# contains the overall rating of the restaurant out of 5
# 
# - **votes**
# contains total number of rating for the restaurant as of the above mentioned date
# 
# - **phone**
# contains the phone number of the restaurant
# 
# - **location**
# contains the neighborhood in which the restaurant is located
# 
# - **rest_type**
# restaurant type
# 
# - **dish_liked**
# dishes people liked in the restaurant
# 
# - **cuisines**
# food styles, separated by comma
# 
# - **approx_cost(for two people)**
# contains the approximate cost for meal for two people
# 
# - **reviews_list**
# list of tuples containing reviews for the restaurant, each tuple 
# 
# - **menu_item**
# contains list of menus available in the restaurant
# 
# - **listed_in(type)**
# type of meal
# 
# - **listed_in(city)**
# contains the neighborhood in which the restaurant is listed
# 

# ## [Exploratory data analysis](#3)

# ## [Which are the top restaurant chains in Bangaluru?]()<a id="4"></a> <br>
# 

# In[ ]:


plt.figure(figsize=(10,7))
chains=df['name'].value_counts()[:20]
sns.barplot(x=chains,y=chains.index,palette='deep')
plt.title("Most famous restaurants chains in Bangaluru")
plt.xlabel("Number of outlets")


# - As you can see **Cafe coffee day,Onesta,Just Bake** has the most number of outlets in and around bangalore.
# - This is rather interesting,we will inspect each of them later.

# ## [How many of the restuarants do not accept online orders?]()<a id="5"></a> <br>
# 

# In[ ]:


x=df['online_order'].value_counts()
colors = ['#FEBFB3', '#E1396C']

trace=go.Pie(labels=x.index,values=x,textinfo="value",
            marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))
layout=go.Layout(title="Accepting vs not accepting online orders",width=500,height=500)
fig=go.Figure(data=[trace],layout=layout)
py.iplot(fig, filename='pie_chart_subplots')
    
    
    


# 1. As clearly indicated,almost 60 per cent of restaurants in Banglore accepts online orders.
# 2. Nearly 40 per cent of the restaurants do not accept online orders.
# 3. This might be because of the fact that these restaurants cannot afford to pay commission to zomoto for giving them orders online.
#    zomato may want to consider giving them some more benefits if they want to increse the number of restaurants serving their customers online.
# 

# ## [What is the ratio b/w restaurants that provide and do not provide table booking ?]()<a id="25"></a> <br>
# 
# 

# In[ ]:


x=df['book_table'].value_counts()
colors = ['#96D38C', '#D0F9B1']

trace=go.Pie(labels=x.index,values=x,textinfo="value",
            marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))
layout=go.Layout(title="Table booking",width=500,height=500)
fig=go.Figure(data=[trace],layout=layout)
py.iplot(fig, filename='pie_chart_subplots')
    


# 1. Almost 90 percent of restaurants in Banglore do not provide table booking facility.
# 2. In India you cannot find table booking facility in any average restaurants,usually only five star restaurants provides table booking.
# 3. We will inspect this further.

# ## [Rating distribution]()<a id="6"></a> <br>
# 

# In[ ]:


plt.figure(figsize=(6,5))
rating=df['rate'].dropna().apply(lambda x : float(x.split('/')[0]) if (len(x)>3)  else np.nan ).dropna()
sns.distplot(rating,bins=20)


# 1. Almost more than 50 percent of restaurants has rating between 3 and 4.
# 2. Restaurants having rating more than 4.5 are very rare.

# In[ ]:



cost_dist=df[['rate','approx_cost(for two people)','online_order']].dropna()
cost_dist['rate']=cost_dist['rate'].apply(lambda x: float(x.split('/')[0]) if len(x)>3 else 0)
cost_dist['approx_cost(for two people)']=cost_dist['approx_cost(for two people)'].apply(lambda x: int(x.replace(',','')))


# ## [Cost vs rating]()

# In[ ]:


plt.figure(figsize=(10,7))
sns.scatterplot(x="rate",y='approx_cost(for two people)',hue='online_order',data=cost_dist)
plt.show()


# - 

# ### [Distribution of cost for two people]()

# In[ ]:


plt.figure(figsize=(6,6))
sns.distplot(cost_dist['approx_cost(for two people)'])
plt.show()


# - We can see that the distribution if left skewed.
# - This means almost 90percent of restaurants serve food for budget less than 1000 INR.($15)

# ## [Is there any difference b/w votes of restaurants accepting and not accepting online orders?]()<a id="7"></a> <br>
# 

# In[ ]:




votes_yes=df[df['online_order']=="Yes"]['votes']
trace0=go.Box(y=votes_yes,name="accepting online orders",
              marker = dict(
        color = 'rgb(214, 12, 140)',
    ))

votes_no=df[df['online_order']=="No"]['votes']
trace1=go.Box(y=votes_no,name="Not accepting online orders",
              marker = dict(
        color = 'rgb(0, 128, 128)',
    ))

layout = go.Layout(
    title = "Box Plots of votes",width=800,height=500
)

data=[trace0,trace1]
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)



# 1. Yes,you can observe that median number of votes for both categories vary.
# 2. Restaurants accepting online orders tend to get more votes from customers as there is a rating option poping up after each order through zomato application.

# ## [Which are the most common restaurant type in Banglore?]()<a id="8"></a> <br>
# 

# In[ ]:


plt.figure(figsize=(7,7))
rest=df['rest_type'].value_counts()[:20]
sns.barplot(rest,rest.index)
plt.title("Restaurant types")
plt.xlabel("count")


# 1. No doubt about this as Banglore is known as the tech capital of India,people having busy and modern life will prefer Quick Bites.
# 2. We can observe tha Quick Bites type restaurants dominates.

# ## [Cost factor ?]()<a id="9"></a> <br>
# 

# In[ ]:


trace0=go.Box(y=df['approx_cost(for two people)'],name="accepting online orders",
              marker = dict(
        color = 'rgb(214, 12, 140)',
    ))
data=[trace0]
layout=go.Layout(title="Box plot of approximate cost",width=800,height=500,yaxis=dict(title="Price"))
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


# 1. The median approximate cost for two people is 400 for a single meal.
# 2. 50 percent of restaurants charge between 300 and 650 for single meal for two people.
# 

# ### [Finding Best budget Restaurants in any location]()

# - I have implemented a simple filtering mechanism to find best budget restaurants in any locations in Bangalore.
# - You can pass location and restaurant type as parameteres,function will return name of restaurants.
# 

# In[ ]:


cost_dist=df[['rate','approx_cost(for two people)','location','name','rest_type']].dropna()
cost_dist['rate']=cost_dist['rate'].apply(lambda x: float(x.split('/')[0]) if len(x)>3 else 0)
cost_dist['approx_cost(for two people)']=cost_dist['approx_cost(for two people)'].apply(lambda x: int(x.replace(',','')))


# In[ ]:


def return_budget(location,rest):
    budget=cost_dist[(cost_dist['approx_cost(for two people)']<=400) & (cost_dist['location']==location) & 
                     (cost_dist['rate']>4) & (cost_dist['rest_type']==rest)]
    return(budget['name'].unique())


# In[ ]:


return_budget('BTM',"Quick Bites")


# ## [Which are the foodie areas?]()<a id="10"></a> <br>
# 

# In[ ]:


plt.figure(figsize=(7,7))
Rest_locations=df['location'].value_counts()[:20]
sns.barplot(Rest_locations,Rest_locations.index,palette="rocket")


# 1. We can see that **BTM,HSR and Koranmangala 5th block** has the most number of restaurants.
# 2. BTM dominates the section by having more than 5000 restaurants.

# ## [Which are the most common cuisines in each locations?]()<a id="11"></a> <br>
# 

# In[ ]:


df_1=df.groupby(['location','cuisines']).agg('count')
data=df_1.sort_values(['url'],ascending=False).groupby(['location'],
                as_index=False).apply(lambda x : x.sort_values(by="url",ascending=False).head(3))['url'].reset_index().rename(columns={'url':'count'})


# In[ ]:



data.head(10)


# ## [Extracting location information using Geopy]()<a id="12"></a> <br>
# 

# In[ ]:


locations=pd.DataFrame({"Name":df['location'].unique()})
locations['Name']=locations['Name'].apply(lambda x: "Bangalore " + str(x))
lat_lon=[]
geolocator=Nominatim(user_agent="app")
for location in locations['Name']:
    location = geolocator.geocode(location)
    if location is None:
        lat_lon.append(np.nan)
    else:    
        geo=(location.latitude,location.longitude)
        lat_lon.append(geo)


locations['geo_loc']=lat_lon
locations.to_csv('locations.csv',index=False)


# In[ ]:


locations["Name"]=locations['Name'].apply(lambda x :  x.replace("Bangalore","")[1:])
locations.head()


# - We have found out latitude and longitude of each location listed in the dataset using geopy.
# - This is used to plot maps.

# ## [Heatmap of restaurant count on each location]()<a id="13"></a> <br>
# 

# In[ ]:


Rest_locations=pd.DataFrame(df['location'].value_counts().reset_index())
Rest_locations.columns=['Name','count']
Rest_locations=Rest_locations.merge(locations,on='Name',how="left").dropna()
Rest_locations['count'].max()


# In[ ]:


def generateBaseMap(default_location=[12.97, 77.59], default_zoom_start=12):
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map


# In[ ]:


lat,lon=zip(*np.array(Rest_locations['geo_loc']))
Rest_locations['lat']=lat
Rest_locations['lon']=lon
basemap=generateBaseMap()
HeatMap(Rest_locations[['lat','lon','count']].values.tolist(),zoom=20,radius=15).add_to(basemap)


# In[ ]:


basemap


# 1. It is clear that restaurants tend to concentrate in central bangalore area.
# 2. The clutter of restaurants lowers are we move away from central.
# 3. So,potential restaurant entrepreneurs can refer this and find out good locations for their venture.

# ## [Which are  the most popular cuisines of Bangalore?]()<a id="14"></a> <br>
# 

# In[ ]:


plt.figure(figsize=(7,7))
cuisines=df['cuisines'].value_counts()[:10]
sns.barplot(cuisines,cuisines.index)
plt.xlabel('Count')
plt.title("Most popular cuisines of Bangalore")


# 1. We can observe that **North Indian,chinese,South Indian and Biriyani** are most common.
# 2. Is this imply the fact that Banglore is more influenced by North Indian culture more than South?
# - We will inspect further......

# ## [Heatmap of North Indian restaurants]()<a id="15"></a> <br>
# 

# In[ ]:


def produce_data(col,name):
    data= pd.DataFrame(df[df[col]==name].groupby(['location'],as_index=False)['url'].agg('count'))
    data.columns=['Name','count']
    print(data.head())
    data=data.merge(locations,on="Name",how='left').dropna()
    data['lan'],data['lon']=zip(*data['geo_loc'].values)
    return data.drop(['geo_loc'],axis=1)


# In[ ]:


North_India=produce_data('cuisines','North Indian')


# In[ ]:


basemap=generateBaseMap()
HeatMap(North_India[['lan','lon','count']].values.tolist(),zoom=20,radius=15).add_to(basemap)
basemap


# 1. Interesting to see a cluster of North Indian Restaurants in South Bangalore area !
# 2. This may indicate that these areas are more populated by North Indians.

# ## [What about South  Indian cuisines?]()<a id="16"></a> <br>
# 

# In[ ]:


food=produce_data('cuisines','South Indian')
basemap=generateBaseMap()
HeatMap(food[['lan','lon','count']].values.tolist(),zoom=20,radius=15).add_to(basemap)
basemap


# 1. They tend to span all over Bangalore.
# 2. South Indian cuisines tend to cluster near central Bangalore.

# ## [Analysing Restaurant chains]()<a id="17"></a> <br>
#  

# In[ ]:


def produce_chains(name):
    data_chain=pd.DataFrame(df[df["name"]==name]['location'].value_counts().reset_index())
    data_chain.columns=['Name','count']
    data_chain=data_chain.merge(locations,on="Name",how="left").dropna()
    data_chain['lan'],data_chain['lon']=zip(*data_chain['geo_loc'].values)
    return data_chain[['Name','count','lan','lon']]


# In[ ]:





# In[ ]:


df_1=df.groupby(['rest_type','name']).agg('count')
datas=df_1.sort_values(['url'],ascending=False).groupby(['rest_type'],
                as_index=False).apply(lambda x : x.sort_values(by="url",ascending=False).head(3))['url'].reset_index().rename(columns={'url':'count'})


# In[ ]:


mapbox_access_token="pk.eyJ1Ijoic2hhaHVsZXMiLCJhIjoiY2p4ZTE5NGloMDc2YjNyczBhcDBnZnA5aCJ9.psBECQ2nub0o25PgHcU88w"


# ## [Which are the most popular casual dining restaurant chains?]()<a id="18"></a> <br>
# 

# In[ ]:


casual=datas[datas['rest_type']=='Casual Dining']
casual


# - We can see tht **Empire restaurant,Beijing bites and Mani's dum biriyani** are the most popular casual dining restaurant chains in Bangalore.
# - We will inspect them further...

# ## [Where are their outlets located?]()<a id="19"></a> <br>
# 

# In[ ]:


def produce_trace(data_chain,name):
        data_chain['text']=data_chain['Name']+'<br>'+data_chain['count'].astype(str)
        trace =  go.Scattermapbox(
           
                lat=data_chain['lan'],
                lon=data_chain['lon'],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=data_chain['count']*4
                ),
                text=data_chain['text'],name=name
            )
        
        return trace


# In[ ]:



data=[] 
for row in casual['name']:
    data_chain=produce_chains(row) 
    trace_0=produce_trace(data_chain,row)
    data.append(trace_0)



layout = go.Layout(title="Casual Dining Restaurant chains locations around Banglore",
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,style="streets",
        center=dict(
            lat=12.96,
            lon=77.59
        ),
        pitch=0,
        zoom=10
    ),
)


fig = dict(data=data, layout=layout)
py.iplot(fig, filename='Montreal Mapbox')


# 1. We can see that **Mani's dum biriyani** restaurant has half of their restaurants in central Banglore.
# 2. ** Empire Restaurant** is present in all over Banglore.
# 3. **Mani's dum Biriyani** has 12 outlets in ulsoor,which is the most in a single location.

# ## [Top quick bites restaurant chains in Banglore]()<a id="20"></a> <br>
# 

# In[ ]:


quick=datas[datas['rest_type']=='Quick Bites']
quick 


# - Surprisingly  ** Five star chicken** dominates in quick bites restaurant types overtaking famous **Domino's Pizza and McDonald's**.
# - Five Star Chicken is a branch of Charoen Pokphand Group Foods, a Thai multinational conglomerate with over $12 billion business in Agro and Food       Industries. Five Star Chicken specialties in fried chicken.

# ## [Where are their outlets located?]()<a id="21"></a> <br>
# 

# In[ ]:


data=[]  
for row in quick['name']:
    data_chain=produce_chains(row) 
    trace_0=produce_trace(data_chain,row)
    data.append(trace_0)



layout = go.Layout(title="Quick Bites Restaurant chains locations around Banglore",
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,style="streets",
        center=dict(
            lat=12.96,
            lon=77.59
        ),
        pitch=0,
        zoom=10
    ),
)


fig = dict(data=data, layout=layout)
py.iplot(fig, filename='Montreal Mapbox')


# ## [Top Cafes of Banglore?]()<a id="22"></a> <br>
# 

# In[ ]:


cafe=datas[datas['rest_type']=='Cafe']
cafe


# - No surprises,Cafe coffee day restaurant dominated way over other cafe chains in Banglore.
# - Cafe coffee day has 96 outlets in Banglore.
# - Café Coffee Day was started as a retail restaurant in 1996. The first CCD outlet was set up on July 11, 1996, at Brigade Road, Bengaluru, Karnataka.
# 

# ## [Where are their outlets located?]()<a id="23"></a> <br>
# 

# In[ ]:


data=[]  
for row in cafe['name']:
    data_chain=produce_chains(row) 
    trace_0=produce_trace(data_chain,row)
    data.append(trace_0)



layout = go.Layout(title="Cafe Restaurant chains locations around Banglore",
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,style="streets",
        center=dict(
            lat=12.96,
            lon=77.59
        ),
        pitch=0,
        zoom=10
    ),
)


fig = dict(data=data, layout=layout)
py.iplot(fig, filename='Montreal Mapbox')


# ## [Wordcloud of dishes liked by cuisines.]()<a id="24"></a> <br>
# 

# In[ ]:


df['dish_liked']=df['dish_liked'].apply(lambda x : x.split(',') if type(x)==str else [''])
#x=df.groupby('rest_type',as_index=False)['dish_liked'].agg('sum')
#x['dish_liked']=x['dish_liked'].apply(lambda x : list(filter(lambda a : a!='',x)))


# In[ ]:


rest=df['rest_type'].value_counts()[:9].index
def produce_wordcloud(rest):
    
    plt.figure(figsize=(20,30))
    for i,r in enumerate(rest):
        plt.subplot(3,3,i+1)
        corpus=df[df['rest_type']==r]['dish_liked'].values.tolist()
        corpus=','.join(x  for list_words in corpus for x in list_words)
        wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1500, height=1500).generate(corpus)
        plt.imshow(wordcloud)
        plt.title(r)
        plt.axis("off")
        

        
        
produce_wordcloud(rest)


# ## [Analysing Reviews]()<a id="26"></a><br>

# - In this section we will go on to prepare reviews dataframe.
# - We will extract reviews and ratings of each restaurant and create a dataframe with it.
# 

# In[ ]:


all_ratings = []

for name,ratings in tqdm(zip(df['name'],df['reviews_list'])):
    ratings = eval(ratings)
    for score, doc in ratings:
        if score:
            score = score.strip("Rated").strip()
            doc = doc.strip('RATED').strip()
            score = float(score)
            all_ratings.append([name,score, doc])


# In[ ]:


rating_df=pd.DataFrame(all_ratings,columns=['name','rating','review'])
rating_df['review']=rating_df['review'].apply(lambda x : re.sub('[^a-zA-Z0-9\s]',"",x))


# In[ ]:


rating_df.to_csv("Ratings.csv")


# In[ ]:


rating_df.head()


# ## [WordCloud of Reviews of restaurant chains]()<a id="27"></a><br>

# In[ ]:


rest=df['name'].value_counts()[:9].index
def produce_wordcloud(rest):
    
    plt.figure(figsize=(20,30))
    for i,r in enumerate(rest):
        plt.subplot(3,3,i+1)
        corpus=rating_df[rating_df['name']==r]['review'].values.tolist()
        corpus=' '.join(x  for x in corpus)
        wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1500, height=1500).generate(corpus)
        plt.imshow(wordcloud)
        plt.title(r)
        plt.axis("off")
        

        
        
produce_wordcloud(rest)


# ## [Rating distribution]()<a id="28" ></a><br>

# In[ ]:


plt.figure(figsize=(7,6))
rating=rating_df['rating'].value_counts()
sns.barplot(x=rating.index,y=rating)
plt.xlabel("Ratings")
plt.ylabel('count')


# ## [Topic modelling]()<a id="33" ></a><br>

# We will do topic modelling for postive and negative comments seperately to understand the different between the two types.

# ##  [Topic modeling for positive comments]()<a id="34" ></a><br>

# - As the first step we will divide comments as negative and positive on the basis on rating provided.
# - Comments with rating below 2.5 is classified as negative and greater tham 2.5 as classified as positive.

# In[ ]:


rating_df['sent']=rating_df['rating'].apply(lambda x: 1 if int(x)>2.5 else 0)


# Now,
#    - we will remove stopwords
#    - Lemmatize each word
#    - Create corpus
#    - Tokenize them

# In[ ]:


stops=stopwords.words('english')
lem=WordNetLemmatizer()
corpus=' '.join(lem.lemmatize(x) for x in rating_df[rating_df['sent']==1]['review'][:3000] if x not in stops)
tokens=word_tokenize(corpus)


# > Now we will  use **Termfrequency Inverse doc frequency(Tfidf)** to vectorize the tokens.

# In[ ]:



vect=TfidfVectorizer()
vect_fit=vect.fit(tokens)
    


# **Latent Dirichlet allocation**

# In[ ]:


id_map=dict((v,k) for k,v in vect.vocabulary_.items())
vectorized_data=vect_fit.transform(tokens)
gensim_corpus=gensim.matutils.Sparse2Corpus(vectorized_data,documents_columns=False)
ldamodel = gensim.models.ldamodel.LdaModel(gensim_corpus,id2word=id_map,num_topics=5,random_state=34,passes=25)


# ## [Visualizing output]()<a id="28" ></a><br>

# **Word Counts of Topic Keywords**
# 
# When it comes to the keywords in the topics, the importance (weights) of the keywords matters. Along with that, how frequently the words have appeared in the documents is also interesting to look.
# 
# Let’s plot the word counts and the weights of each keyword in the same chart.
# 
# You want to keep an eye out on the words that occur in multiple topics and the ones whose relative frequency is more than the weight. Often such words turn out to be less important. The chart I’ve drawn below is a result of adding several such words to the stop words list in the beginning and re-running the training process.

# In[ ]:


counter=Counter(corpus)


# In[ ]:


out=[]
topics=ldamodel.show_topics(formatted=False)
for i,topic in topics:
    for word,weight in topic:
        out.append([word,i,weight,counter[word]])

dataframe = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        


# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 2, figsize=(8,6), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=dataframe.loc[dataframe.topic_id==i, :], color=cols[i], width=0.3, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=dataframe.loc[dataframe.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    #ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=8)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(dataframe.loc[dataframe.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=8, y=1.05)    
plt.show()


# ## [Topic modeling for negative comments]()<a id="35" ></a><br>

# In[ ]:


stops=stopwords.words('english')
lem=WordNetLemmatizer()
corpus=' '.join(lem.lemmatize(x) for x in rating_df[rating_df['sent']==0]['review'][:3000] if x not in stops)
tokens=word_tokenize(corpus)


# In[ ]:


vect=TfidfVectorizer()
vect_fit=vect.fit(tokens)
id_map=dict((v,k) for k,v in vect.vocabulary_.items())
vectorized_data=vect_fit.transform(tokens)
gensim_corpus=gensim.matutils.Sparse2Corpus(vectorized_data,documents_columns=False)
ldamodel = gensim.models.ldamodel.LdaModel(gensim_corpus,id2word=id_map,num_topics=5,random_state=34,passes=25)

 


# In[ ]:


counter=Counter(corpus)
out=[]
topics=ldamodel.show_topics(formatted=False)
for i,topic in topics:
    for word,weight in topic:
        out.append([word,i,weight,counter[word]])

dataframe = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        


# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 2, figsize=(8,6), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=dataframe.loc[dataframe.topic_id==i, :], color=cols[i], width=0.3, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=dataframe.loc[dataframe.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    #ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=8)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(dataframe.loc[dataframe.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=8, y=1.05)    
plt.show()


# - We can clearly observe the difference between the two types of comments
# - The words used are clearly distinguishable.
# - The words used in negative comments are clearly critisizing.
# - The word used in positive comments  are clearly appreciating.

# ### T-SNE of Reviews 

# (t-SNE) t-Distributed Stochastic Neighbor Embedding is a non-linear dimensionality reduction algorithm used for exploring high-dimensional data. It maps multi-dimensional data to two or more dimensions suitable for human observation. With help of the t-SNE algorithms, you may have to plot fewer exploratory data analysis plots next time you work with high dimensional data.
# ![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/01/19100947/20131959678_bf1a8e3fcc_b-768x798.jpg)

# In this section we will visualize words used in reviews in a 2 dimensional space.
# - For that we will first lemmatize and tokenize each reviews and build a corpus out of it.

# In[ ]:


stops=set(stopwords.words('english'))
lem=WordNetLemmatizer()
corpus=[]
for review in tqdm(rating_df['review'][:10000]):
    words=[]
    for x in word_tokenize(review):
        x=lem.lemmatize(x.lower())
        if x not in stops:
            words.append(x)
            
    corpus.append(words)


# Now we will use word2vec to represent each word as a vector.

# In[ ]:


model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=200, workers=4)


# In[ ]:


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(10, 10)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


# In[ ]:


tsne_plot(model)


# ### T-SNE of adjectivs used in postive reviews

# In[ ]:


postive=rating_df[rating_df['rating']>3]['review'][:2000]
negative=rating_df[rating_df['rating']<2.5]['review'][:2000]

def return_corpus(df):
    corpus=[]
    for review in df:
        tagged=nltk.pos_tag(word_tokenize(review))
        adj=[]
        for x in tagged:
            if x[1]=='JJ':
                adj.append(x[0])
        corpus.append(adj)
    return corpus


# Wow,we can observe all the adjective used in the postive reviews represented in the 2D space.

# In[ ]:


corpus=return_corpus(postive)
model = word2vec.Word2Vec(corpus, size=100, min_count=10,window=20, workers=4)
tsne_plot(model)


# ### T-SNE of adjectives used in Negative reviews

# In[ ]:


corpus=return_corpus(negative)
model = word2vec.Word2Vec(corpus, size=100, min_count=10,window=20, workers=4)
tsne_plot(model)


# 

# ## [Sentimental Analysis]()<a id="29" ></a><br>

# Sentiment Analysis is the process of computationally determining whether a piece of writing is positive, negative or neutral. It’s also known as opinion mining, deriving the opinion or attitude of a speaker.
# 
# 

# ![](https://www.geeksforgeeks.org/wp-content/uploads/Twitter-Sentiment-Analysis-using-Python.jpg)

# ## [Data preparation]()<a id="30" ></a><br>
# 

# For doing sentimental analysis on reviews provided bt users.We have to prepare our data in appropriate format.
# We will map reviews to positive and negative on the basis of the ratings provided by each user.So,we will map reviews to negative if the rating given is less than 2.5 and positive if rating is greater than 2.5 
# 

# In[ ]:


rating_df['sent']=rating_df['rating'].apply(lambda x: 1 if int(x)>2.5 else 0)


# - Next, we will tokenize the data and vectorize the reviews to feed it to our model.
# 

# In[ ]:


max_features=3000
tokenizer=Tokenizer(num_words=max_features,split=' ')
tokenizer.fit_on_texts(rating_df['review'].values)
X = tokenizer.texts_to_sequences(rating_df['review'].values)
X = pad_sequences(X)


# ## [Building our model]()<a id="31" ></a><br>

# In[ ]:


embed_dim = 32
lstm_out = 32

model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))
#model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


# #### Train and test split

# - We will now encode our target variable. **pd.get_dummies** is used for on-hot encoding.
# - 33 percent of data is reserved for testing our model

# In[ ]:


Y = pd.get_dummies(rating_df['sent'].astype(int)).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[ ]:


batch_size = 3200
model.fit(X_train, Y_train, epochs = 5, batch_size=batch_size)


# ### [Validating our model]()<a id="32" ></a><br>

# We will take 1500 rows to validate our model.We have choosen **accuacy** to be our evaluation criteria.

# In[ ]:


validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))


# <font color='red' size=4>  If you like my kernel please consider upvoting.</font>
# 
# <font color='green' size=3>Comments are most welcomed !</font>    

# In[ ]:




