#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv("../input/amazon-top-50-bestselling-books-2009-2019/bestsellers with categories.csv")


# In[ ]:


df.head()


# In[ ]:


df.dtypes


# In[ ]:


df.shape


# In[ ]:


len(df['Name'].value_counts())


# In[ ]:


df.nunique()


# In[ ]:


df[['Name','Author']].duplicated().sum()


# In[ ]:


df[['Name','Author']].drop_duplicates()


# In[ ]:


df.shape


# In[ ]:


df.drop_duplicates(['Name','Author'],inplace=True)


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df['Year'].value_counts()


# In[ ]:


c=df.corr()
sns.heatmap(c,annot=True)


# In[ ]:


sns.pairplot(df)


# In[ ]:


df_non=df[df['Genre'] == 'Non Fiction']
df_fiction=df[df['Genre'] == 'Fiction']


# In[ ]:


df_non.shape


# In[ ]:


df_fiction.shape


# In[ ]:


plt.figure(figsize=(15,8))
sns.boxplot(x=df['User Rating'],y=df['Reviews'])


# In[ ]:


plt.figure(figsize=(15,8))
sns.boxplot(x=df['Year'],y=df['Reviews'])


# In[ ]:


# plt.pie()
x=df['Genre'].value_counts()
q1=pd.DataFrame(x)
q1.reset_index(inplace=True)
plt.pie(q1['Genre'],labels=q1['index'],autopct="%.2f")
plt.xlabel("Gener")
plt.ylabel("Count")
plt.title("Compare between Gener elements")


# In[ ]:


x=df.groupby(['Year'])['Name'].count()
q2=pd.DataFrame(x)
q2.reset_index(inplace=True)
q2['Year']=q2['Year'].astype('object')


# In[ ]:


plt.bar(q2['Year'],q2['Name'])
plt.xlabel("Years")
plt.ylabel("Count of books")
plt.title("Count of Books which Product by each Years")
plt.figure(figsize=(20,10))


# In[ ]:


x=df.groupby('Author')['Reviews'].agg([sum]).sort_values(by=('sum'),ascending=False).head(10)
q3=pd.DataFrame(x)
q3.reset_index(inplace=True)
plt.bar(q3['Author'],q3['sum'],color="blue")
plt.title("Top 5 Auhtor get Rating")
plt.xticks(rotation=90)
plt.figure(figsize=(18,18))


# In[ ]:


x=df.groupby('Name')['Price'].agg([sum]).sort_values(by=('sum'),ascending=False).head(10)
q4=pd.DataFrame(x)
q4.reset_index(inplace=True)
plt.barh(q4['Name'],q4['sum'])
plt.title("Top 10 Price by Name")
# plt.xticks(rotation=90)
plt.figure(figsize=(18,18))

