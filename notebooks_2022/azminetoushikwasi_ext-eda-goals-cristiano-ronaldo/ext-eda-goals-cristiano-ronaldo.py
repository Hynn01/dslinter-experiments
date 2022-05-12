#!/usr/bin/env python
# coding: utf-8

# # CR7 - Extensive EDA & Analytics - Cristiano Ronaldo - All Club Goals Stats

# **Cristiano Ronaldo dos Santos Aveiro** is a Portuguese professional footballer who plays as a forward for Premier League club Manchester United and captains the Portugal national team.
#  
# - Current team: Portugal national football team (#7 / Forward) Trending
#                                                
# - Born: February 5, 1985 (age 37 years), Hospital Dr. Nélio Mendonça, Funchal, Portugal
# - Height: 1.87 m
# - Partner: Georgina Rodríguez (2017–)
# - Salary: 26.52 million GBP (2022)
# - Children: Cristiano Ronaldo Jr., Alana Martina dos Santos Aveiro, Eva Maria Dos Santos, Mateo Ronaldo
# 
# ![asaa.jpeg](attachment:38fb37d1-a3c6-417d-aede-c483a670426a.jpeg)

# # Data Loading

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df=pd.read_csv("/kaggle/input/cr7-cristiano-ronaldo-all-club-goals-stats/data.csv")


# # Basic Exploration

# In[ ]:


df.head()


# In[ ]:


df.info()


# # Exploritory Data Analysis

# ## Goals per competition

# In[ ]:


sns.set(rc={'figure.figsize':(15,3)})
plt.xticks(fontsize=15,rotation='vertical')
p=sns.countplot(df['Competition'],hue_order=df.groupby('Competition'))
p.axes.set_title("Goals per competition",fontsize=30)


# ## Goals per season

# In[ ]:


sns.set(rc={'figure.figsize':(10,5)})
plt.xticks(fontsize=15,rotation='vertical')
p=sns.countplot(df['Season'],hue_order=df.groupby('Competition'))
p.axes.set_title("Goals per season",fontsize=30)
("")


# ## Goals per Clubs

# In[ ]:


sns.set(rc={'figure.figsize':(10,5)})
plt.xticks(fontsize=15,rotation='vertical')
p=sns.countplot(df['Team'],hue_order=df.groupby('Competition'))
p.axes.set_title("Goals per Team",fontsize=30)


# ## Goals per playing Position

# In[ ]:


sns.set(rc={'figure.figsize':(10,5)})
plt.xticks(fontsize=15,rotation='vertical')
p=sns.countplot(df['Position'],hue_order=df.groupby('Competition'))
p.axes.set_title("Goals per Position",fontsize=30)


# ## Goals per Game Minute

# In[ ]:


mins=list(map(str, range(1,120)))
for min in df['Minute']:
    if min not in mins:
        mins.append(min)
mins1=mins[:int(len(mins)/5)]
mins2=mins[int(len(mins)/5):int(2*len(mins)/5)]
mins3=mins[2*int(len(mins)/5):int(3*len(mins)/5)]
mins4=mins[3*int(len(mins)/5):int(4*len(mins)/5)]
mins5=mins[int(4*len(mins)/5):]


# In[ ]:


sns.set(rc={'figure.figsize':(20,5)})
plt.xticks(fontsize=10)
p=sns.countplot(df['Minute'],order=mins1)
p.axes.set_title("Goals per Game Minute (1-26)",fontsize=30)


# In[ ]:


p=sns.countplot(df['Minute'],order=mins2)
p.axes.set_title("Goals per Game Minute (27-52)",fontsize=30)


# In[ ]:


p=sns.countplot(df['Minute'],order=mins3)
p.axes.set_title("Goals per Game Minute (53-78)",fontsize=30)


# In[ ]:


p=sns.countplot(df['Minute'],order=mins4)
p.axes.set_title("Goals per Game Minute (79-104)",fontsize=30)


# In[ ]:


p=sns.countplot(df['Minute'],order=mins5)
p.axes.set_title("Goals per Game Minute (104-Extra)",fontsize=30)


# ## Goals per Type_of_goal

# In[ ]:


sns.set(rc={'figure.figsize':(10,4)})
plt.xticks(fontsize=15,rotation='vertical')
p=sns.countplot(df['Type_of_goal'],hue_order=df.groupby('Competition'))
p.axes.set_title("Goals per Type_of_goal",fontsize=30)


# ## Scoreline after goal

# In[ ]:


sns.set(rc={'figure.figsize':(10,5)})
plt.xticks(fontsize=15,rotation='vertical')
p=sns.countplot(df['At_score'],hue_order=df.groupby('Competition'))
p.axes.set_title("Goals per At_score",fontsize=30)


# ## Opponents

# In[ ]:


sns.set(rc={'figure.figsize':(30,5)})
plt.xticks(fontsize=10,rotation='vertical')
p=sns.countplot(df['Opponent'],hue_order=df.groupby('Competition'))
p.axes.set_title("Goals per Opponent",fontsize=30)


# ## Favourite Opponents

# In[ ]:


sns.set(rc={'figure.figsize':(12,5)})
opponents_df=df.groupby('Opponent').size().reset_index(name='count')
fav_opponents_df=opponents_df[opponents_df["count"]>15]
plt.bar(x=fav_opponents_df['Opponent'],height=fav_opponents_df['count'],color="#1a55e7")


# ## Home - Away goals

# In[ ]:


sns.set(rc={'figure.figsize':(3,3)})
plt.xticks(fontsize=15)
p=sns.countplot(df['Venue'],hue_order=df.groupby('Competition'))
p.axes.set_title("Goals per Venue(Home/Away)",fontsize=20)


# # More coming soon
