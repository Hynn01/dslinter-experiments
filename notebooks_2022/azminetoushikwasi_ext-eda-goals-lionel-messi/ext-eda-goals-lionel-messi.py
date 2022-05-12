#!/usr/bin/env python
# coding: utf-8

# # LM10 - Extensive EDA & Analytics - Lionel Messi - All Club Goals Stats

# Lionel Andrés Messi, also known as Leo Messi, is an Argentine professional footballer who plays as a forward for Ligue 1 club Paris Saint-Germain and captains the Argentina national team. 
# 
# - Born: June 24, 1987 (age 34 years), Rosario, Argentina
# - Height: 1.69 m
# - Spouse: Antonela Roccuzzo (m. 2017)
# - Salary: 41 million USD (2022)
# - Current teams: Paris Saint-Germain F.C. (#30 / Forward), Argentina national football team (#10 / Forward)
# - Children: Mateo Messi Roccuzzo, Thiago Messi Roccuzzo, Ciro Messi Roccuzzo
# - Awards: European Golden Shoe, FIFA World Player of the Year, MORE

# ![merlin_153612873_5bb119b9-8972-4087-b4fd-371cab8c5ba2-superJumbo.jpg](attachment:29f89976-cc30-4f33-bd60-5108d468864c.jpg)

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


df=pd.read_csv("/kaggle/input/-lionel-messi-all-club-goals/data.csv")


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


# ## Goals per Clubs

# In[ ]:


sns.set(rc={'figure.figsize':(10,5)})
plt.xticks(fontsize=15,rotation='vertical')
p=sns.countplot(df['Club'],hue_order=df.groupby('Competition'))
p.axes.set_title("Goals per Clubs Played",fontsize=30)


# ## Goals per playing Position

# In[ ]:


sns.set(rc={'figure.figsize':(10,5)})
plt.xticks(fontsize=15,rotation='vertical')
p=sns.countplot(df['Playing_Position'],hue_order=df.groupby('Competition'))
p.axes.set_title("Goals per Playing Position",fontsize=30)


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


# ## Goals per Type_of_goal

# In[ ]:


sns.set(rc={'figure.figsize':(10,4)})
plt.xticks(fontsize=15,rotation='vertical')
p=sns.countplot(df['Type'],hue_order=df.groupby('Competition'))
p.axes.set_title("Goals per Type of goal scored",fontsize=30)


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
plt.xticks(fontsize=10,rotation='vertical')
plt.bar(x=fav_opponents_df['Opponent'],height=fav_opponents_df['count'],color="#1ae5e1")


# ## Home - Away goals

# In[ ]:


sns.set(rc={'figure.figsize':(3,3)})
plt.xticks(fontsize=15)
p=sns.countplot(df['Venue'],hue_order=df.groupby('Competition'))
p.axes.set_title("Goals per Venue(Home/Away)",fontsize=20)


# # More coming soon
