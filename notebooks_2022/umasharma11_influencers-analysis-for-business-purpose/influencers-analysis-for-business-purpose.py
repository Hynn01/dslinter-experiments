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


import matplotlib.pyplot as plt
import seaborn as sns


# Top 1000 Influencers include celebrities with large followings to niche content creators with a loyal following on YouTube, Instagram, Facebook, and Twitter.They are categorized in tiers (mini,mega), based on their number of followers.
# 
# Businesses pursue people who aim to lessen their consumption of advertisements, and are willing to pay their influencers more.
# 
# Market Researchers find that influencer selection extends into product personality. This product and benefit matching is key. For a shampoo, it should use an influencer with good hair.

# In[ ]:


df_t=pd.read_csv('../input/top-1000-social-media-channels/social media influencers - tiktok.csv')
df_i=pd.read_csv('../input/top-1000-social-media-channels/social media influencers - instagram.csv')
df_y=pd.read_csv('../input/top-1000-social-media-channels/social media influencers - youtube.csv')


# # tiktok data set analysis

# In[ ]:


df_t.head(3)


# ## In tiktok dataset there is not category and audience country data  so I didn't find it useful to tell a buisnessmen that which influencer must be approached by them categorywise

# # other conclusions:

# In[ ]:


df_t.isnull().sum()


# In[ ]:


df_t['Subscribers count'].str[-1].value_counts()


# In[ ]:


import re
def convert(x):
    return re.findall('\d+\.?\d*',x)


# In[ ]:


def change(df,list1):
    for i in list1:
        df['new'+i]=df[i].apply(convert)
        df['new'+i]=df['new'+i].apply(lambda x: "".join(x))
        df['new'+i]=pd.to_numeric(df['new'+i])
        df['new'+i]=np.where(['M' in j for j in df[i]],df['new'+i]*1000000,
                             np.where(['K' in j1 for j1 in df[i]],df['new'+i]*1000,df['new'+i]))
    return df
    


# In[ ]:


change(df_t,['Subscribers count'])


# # TOP 10 most followed celebrity on Tiktok

# In[ ]:


df_t.sort_values(by='newSubscribers count',ascending=False,ignore_index=True).iloc[0:10,[1,2]]


# # now analyse instagram and youtube data set

# In[ ]:


df_i.head(2)
# instagram dataset


# In[ ]:


df_y.head(2)
# youtube dataset


# # renaming some columns in instagram dataframe for convenience

# In[ ]:


df_i.rename({'category_1':'Category','Audience country(mostly)':'Audience Country'},axis=1,inplace=True)

df_y.rename({'Subscribers':'Followers',},axis=1,inplace=True)


# In[ ]:


df_i.head(2)


# In[ ]:


df_y.head(2)


# In[ ]:


df_i.isnull().sum()


# In[ ]:


df_y.isnull().sum()


# In[ ]:


df_i.drop_duplicates(subset=['Influencer insta name'],inplace=True)


# In[ ]:


df_i.shape


# In[ ]:


df_i.drop(labels=['Influencer insta name','Authentic engagement\r\n'],axis=1,inplace=True)


# In[ ]:


df_i.head(2)


# In[ ]:


li=['Followers','Engagement avg\r\n']


# In[ ]:


change(df_i,li)


# ## Engagement rate : the Percentage of Followers who really engages with the content posted by Influencers 
# 
# # Why ER  is so Important?
# ## 1-good ER means your content is making an impact on audience(they really  like you)
# ## 2- The higher the engagement ,the more likely it is that the content will be boosted in the newsfeed and attracting more eyes.

# ##### Engagement Rate formula:
#     ER=(Engagement Average/total Followers)*100

# In[ ]:


df_i['Engagement Rate']=np.round((df_i['newEngagement avg\r\n']/df_i['newFollowers'])*100,3)


# In[ ]:


print(df_i['Followers'].str[-1].unique())


# In[ ]:


# for convenice 
df_i['newFollowers']=df_i['newFollowers']/1000000


# In[ ]:


df_i.drop(labels=['Engagement avg\r\n','newEngagement avg\r\n'],axis=1,inplace=True)


# In[ ]:


df_i.head(5)


# # TOP 15 most followed celebrity on  instagram

# In[ ]:


df_i.sort_values(by='newFollowers',ascending=False,ignore_index=True).iloc[0:15,[0,1,3,-1]]


# In[ ]:


plt.title('Top 15 most followed celebrity on instagram')
plt.xlabel('Followers in Million')
sns.barplot(y='instagram name',x='newFollowers',data=df_i.sort_values(by='newFollowers',ascending=False).head(15))


# In[ ]:


pallete=['red','green','yellow','salmon','cyan','blue','orange']


# In[ ]:


def plot(df):
    plt.figure(figsize=(8,6))
    plt.xlabel('number of times category occured')
    plt.ylabel('Category')
    df['Category'].value_counts().sort_values(ascending=True).plot.barh(color=pallete)


# ## TOP  categories followed on instagram(POPULAR CATEGORIES  ON INSTAGRAM)

# In[ ]:


plot(df_i)
    


# ## TOP  categories followed on YOUTUBE(POPULAR CATEGORIES  ON YOUTUBE)

# In[ ]:


df_y.drop_duplicates(subset=['channel name'],inplace=True)


# In[ ]:


plot(df_y)


# # conclusion:
# ### 1-some categories are not on both plateforms 
# ### 2-some categories are more popular on instagram than youtube and vice versa
# ### 3-Example-EDUCATION and Animation is more popular on YOUTUBE the INSTAGRAM

# In[ ]:





# # Decide That where you want to make ads

# In[ ]:


def plot_c(df):
    plt.figure(figsize=(10,8))
    plt.xlabel('number of times category occured')
    df['Audience Country'].value_counts().sort_values().plot.barh(color=pallete)


# ## TOP consumer countries of the influencers content on INSTAGRAM

# In[ ]:


#plot_c(df_i)


# ## TOP consumer countries of the influencers content on YOUTUBE

# In[ ]:


#plot_c(df_y)


# # (TARGET COUNTRY FOR BUISNESS)Checking the demand for categories by Country wise

# ### for understanding that where is the demand of product

# In[ ]:


def demand(data,category):
    return data[data['Category']==category]['Audience Country'].value_counts().sort_values(ascending=True).plot.barh(color=pallete)
    


# In[ ]:


demand(df_y,'Education')


# In[ ]:





# In[ ]:


demand(df_i,'Lifestyle')


# ## 1-for a particular category we can choose the country where a huge audience is liking that particular category .
# ### for example if you want to make an advertisement of educational app then you can choose India

# In[ ]:





# # TOP 15 most followed channels on  youtube

# In[ ]:


df_y.iloc[0:10,[1,2,3]]
# youtube dataset is already sorted


# In[ ]:


ly=['Followers','avg views', 'avg likes', 'avg comments']


# In[ ]:





# ### if you want to go with mini followers for advertisement on instagram

# In[ ]:


df_i['newFollowers'].describe()


# In[ ]:


df_i['newFollowers'].quantile(0.94)


# #### I am taking 60M as a threshold means for instagram celebrity havning above 60M followers are considerd to be mega celebrity

# In[ ]:


df_i.head(2)


# ## if you  want to make ads by mini influencers 

# In[ ]:


def for_mini_followers_instagram(coun,cat):
    df1=df_i[df_i['Audience Country']==coun]
    df1_mini=df1[df1['newFollowers']<60]
    return df1_mini.sort_values(by='Engagement Rate',ascending=False).groupby('Category').get_group(cat).iloc[:,[0,3,-1]]
    
    


# In[ ]:


for_mini_followers_instagram('India','Music')


# ## if you want to make ads by mega influencers

# In[ ]:


def for_mega_followers_instagram(coun,cat):
    df1=df_i[df_i['Audience Country']==coun]
    df1_mini=df1[df1['newFollowers']>60]
    return df1_mini.sort_values(by='Engagement Rate',ascending=False).groupby('Category').get_group(cat).iloc[:,[0,3,-1]]
    
    


# In[ ]:


for_mega_followers_instagram('India','Music')


# In[ ]:


for_mini_followers_instagram('India','Beauty')


# In[ ]:


for_mini_followers_instagram('India','Shows')


# In[ ]:


# category(df_i,'Sports with a ball')


# In[ ]:


for_mini_followers_instagram('India','Sports with a ball')


# In[ ]:


for_mega_followers_instagram('India','Sports with a ball')


# In[ ]:


df_y.head(3)


# In[ ]:


df_y.isnull().sum()


# #### Due to nan values we have to remove some data .but in reality you can collect this data easily

# In[ ]:


df_y.dropna(axis=0,how='any',subset=['avg likes','avg comments']).isnull().sum()


# In[ ]:





# In[ ]:


df_y.dropna(axis=0,how='any',subset=['avg likes','avg comments'],inplace=True)


# In[ ]:


change(df_y,ly)


# In[ ]:


#df_y[df_y['Audience Country']=='Spain']['Category'].value_counts()


# In[ ]:


#df_y[df_y['Audience Country']=='Brazil'].groupby('Category').get_group('Animation')


# In[ ]:


df_y['Engagement rate']=round(((df_y['newavg comments']+df_y['newavg likes']+df_y['newavg views'])/df_y['newFollowers'])*100,3)


# In[ ]:


df_y.head(2)


# In[ ]:


df_y.columns


# In[ ]:


# for convenince
df_y['newFollowers']=df_y['newFollowers']/1000000


# In[ ]:


df_y.drop(labels=['avg views', 'avg likes', 'avg comments','newavg views', 'newavg likes', 'newavg comments',
       ],axis=1,inplace=True)


# In[ ]:


df_y['newFollowers'].describe()


# In[ ]:


df_y['newFollowers'].quantile(0.90)


# ### Threshold can be decided by your choice 
# #### here i am cosidering that who have >30M subscribers that is coming the category of mega celebrity

# In[ ]:


df_y.head(1)


# In[ ]:


def for_mini_followers_youtube(coun,cat):
    df1=df_y[df_y['Audience Country']==coun]
    df1_mini=df1[df1['newFollowers']<=30]
    return df1_mini.sort_values(by='Engagement rate',ascending=False).groupby('Category').get_group(cat).iloc[:,[0,3,-1]]
    
    


# In[ ]:


# df_y.groupby('Category')['Audience Country'].first()


# In[ ]:


for_mini_followers_youtube('United States','Music & Dance')


# In[ ]:


def for_mega_followers_youtube(coun,cat):
    df1=df_y[df_y['Audience Country']==coun]
    df1_mini=df1[df1['newFollowers']>30]
    return df1_mini.sort_values(by='Engagement rate',ascending=False).groupby('Category').get_group(cat).iloc[:,[0,3,-1]]
    
    


# In[ ]:


for_mega_followers_instagram('Brazil','Sports with a ball')


# In[ ]:


# df_y['Category'].value_counts()


# In[ ]:


for_mega_followers_youtube('India','Movies')

