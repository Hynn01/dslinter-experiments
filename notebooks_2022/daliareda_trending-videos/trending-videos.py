#!/usr/bin/env python
# coding: utf-8

# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#0C2D48;text-align:center;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%">Content</div>
#    
# #### <span style='color:#2E8BC0'>1| </span>[import and clean show data summry](#section-one) 
# #### <span style='color:#2E8BC0'>2| </span> [get the category name into the data](#section-two) 
# #### <span style='color:#2E8BC0'>3| </span>[start answring some questions](#section-three)
# 

# <a id="section-one"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#0C2D48;text-align:center;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%">import and show the data</div>

# - read the data and show the first few rows

# In[ ]:


import pandas as pd 
df = pd.read_csv("../input/trendingvidoes/TredndingVidoes.csv")


# In[ ]:


df.head() 


# - search for the missing values in the data

# In[ ]:


df.isnull().sum()


# - reamove unwanted columns `["description","thumbnail_link","tags","video_id"]` from the data

# In[ ]:


df.drop(columns=["description","thumbnail_link","tags","video_id"],inplace=True)


# - show the datatype of the columns

# In[ ]:


df.dtypes


# - change the datatype of `[tranding_data, puplish_time]` into datatime 

# In[ ]:


df.trending_date=pd.to_datetime(df.trending_date,format="%y.%d.%m")


# In[ ]:


df.publish_time=pd.to_datetime(df.publish_time)


# <a id="section-two"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#0C2D48;text-align:center;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%">get the category name</div>

# - to get the cateory name from category id there is [categoryId.json] file
# - work with it to add new column to the data [ category_name ]  

# In[ ]:


import json


# In[ ]:


f=open("../input/trendingvidoes/Id_to_category.json")


# In[ ]:


category=json.load(f)


# In[ ]:


list_1=[]
for i in category:
    list_1.append(i["id"])


# In[ ]:


list_2=[]
for dic in category:
    list_2.append(dic["snippet"]["title"])


# In[ ]:


dic={list_1[i]:list_2[i] for i in range(len(list_1))}


# In[ ]:


df["category_title"]=df["category_id"].apply(lambda x: dic[str(x)])


# In[ ]:


df.head()


# <a id="section-three"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#0C2D48;text-align:center;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%">Ask questions and asnwers them</div>

# #### searching for rellations

# - show the relation between these column with each other ["likes","dislikes","views","comment_count","category"]
# - use a suitable graph to show the ralations

# In[ ]:


import matplotlib.pyplot as plt
# put these line to make the graphs much better looking 
plt.style.use("ggplot")


# ####  <div style="color:#fff;display:fill;border-radius:10px;background-color:red;text-align:center;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%">Ask questions and asnwers them " modified "</div>

# In[ ]:


import seaborn as sns 
import matplotlib.pyplot as plt 


# In[ ]:


# simply using the pairplot function 
plt.figure(figsize=(8,8))
sns.pairplot(df[["likes","dislikes","views","comment_count"]],kind="reg")


# ### <div style="background-color:red;color:#fff;padding:10px;"> we can simply see the relation between each two variables </div>

# In[ ]:


list_3=["likes","dislikes","views","comment_count"]
plt.figure(figsize=(15,10))
for i in range(4):
    plt.subplot(1,4,i+1)
    df.groupby("category_title")[list_3[i]].sum().plot()
    plt.title(list_3[i])
    plt.xticks(rotation=90)
#     plt.tight_layout(1.5)
plt.show()


# In[ ]:


plt.figure(figsize=(50,50))
df.groupby(df.category_title)["likes","dislikes","views","comment_count"].sum().plot()
                                                                            
plt.xticks(rotation=90)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
f, ax = plt.subplots(nrows=4 , ncols=1,figsize=(15,25))
for i, d in enumerate(list_3):
    sns.lineplot(x= df["category_title"],y=df[d], ax = ax[i])
plt.xticks(rotation=90) 


# - write what you gained from the graph ` What information did you ganied`

# music has the largest values in each on of ["likes","dislikes","views","comment_count"] 

# - show the relation between ["likes','views'] column for each category

# In[ ]:


df.groupby(df.category_title)["likes","views"].sum().plot()
plt.xticks(rotation=90)


# it is a positive correlation

# ### Questions section
# ######  asnwers the question with the plots and graphs as final answer

# - what is the categories that dominate the trending videos?

# In[ ]:


df.groupby(df.category_title)["views"].sum().sort_values(ascending=False).plot(kind="bar")
plt.ylabel("views")


# ####  <div style="color:#fff;display:fill;border-radius:10px;background-color:red;text-align:center;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> the category that dominate trending videos "modified "</div>

# - we can define it as the category that has the highest number of trending videos 

# In[ ]:


plt.figure(figsize=(10,6))
df.category_title.value_counts(normalize=True).plot(kind="bar")
plt.ylabel("percent of trending videos")
# normalize = true  -> to get the percent over total number 


# ***

# - which categrory have highest interaction

# In[ ]:


(df.groupby(df.category_title)["likes"].sum()+df.groupby(df.category_title)["dislikes"].sum()+df.groupby(df.category_title)["comment_count"].sum()).sort_values(ascending=False).plot(kind="bar")
plt.ylabel("all interactions : likes + dislikes + comments")


# - in each category , is there is a channel that dominate the trending vidoes in this category
#      - this one is realy good 😂

# In[ ]:


(df.groupby(["category_title","channel_title"])["category_title"].count()).sort_values(ascending=False)


# - how many vidoes have disabled the comments ? 

# In[ ]:


df.comments_disabled.sum()    #true = 1  &  false = 0


# - how many divoes have disabled the ratings ?

# In[ ]:


df.ratings_disabled.sum()


# - Show the channnels with the most number of treanding videos

# In[ ]:


df.channel_title.value_counts()

