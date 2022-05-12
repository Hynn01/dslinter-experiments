#!/usr/bin/env python
# coding: utf-8

# # Imports and loading

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


get_ipython().system('ls /kaggle/input/fmi-su-recommender-system-hw-22022/')


# In[ ]:


books_df = pd.read_csv('/kaggle/input/fmi-su-recommender-system-hw-22022/books.csv')
users_df = pd.read_csv('/kaggle/input/fmi-su-recommender-system-hw-22022/users.csv')
ratings_df = pd.read_csv('/kaggle/input/fmi-su-recommender-system-hw-22022/ratings_train.csv')
ratings_to_predict_df = pd.read_csv('/kaggle/input/fmi-su-recommender-system-hw-22022/ratings_to_predict.csv')


# # EDA

# In[ ]:


books_df[:10]


# In[ ]:


print(f"{len(books_df['Book_Id'].value_counts())} different books")
print(f"{len(books_df['Author'].value_counts())} different authors")
print(f"{len(books_df['Publisher'].value_counts())} different publishers")


# In[ ]:


books_df[books_df['Year'] > 1900].hist(column = 'Year', bins = 150, figsize = (20, 3))


# In[ ]:


users_df[:10]


# In[ ]:


print(f"{len(users_df['User_Id'].value_counts())} different users")
print(f"{len(users_df['Country'].value_counts())} different countries")


# In[ ]:


top_countries = users_df['Country'].value_counts()[:20].index.tolist()
users_df.loc[users_df['Country'] == ' usa'] # country names seem to have a leading space for some reason


# In[ ]:


users_df[users_df['Country'].isin(top_countries)].groupby('Country').count().reset_index()[['Country', 'User_Id']].rename(columns={'User_Id': 'Amount of users'})


# In[ ]:


users_df[users_df['Age'] > 0].hist(column = 'Age', bins = 150, figsize = (20, 3))


# In[ ]:


ratings_df[:10]


# In[ ]:


ratings_df['Rating'].value_counts()


# In[ ]:


user_rating_counts = ratings_df['User_Id'].value_counts()
user_rating_counts


# In[ ]:


print(f"{len(user_rating_counts[user_rating_counts == 1])} users who only rated one item")
print(f"{len(user_rating_counts[user_rating_counts == 2])} users who only rated two items")
print(f"{len(user_rating_counts[user_rating_counts == 3])} users who only rated three items")


# In[ ]:


user_rating_counts[user_rating_counts < 10].hist()


# In[ ]:


user_rating_counts[user_rating_counts > 1000].hist()


# In[ ]:


plt.figure(figsize = (20, 3))
plt.plot(range(0, len(user_rating_counts)), user_rating_counts)


# In[ ]:


ratings_to_predict_df[:10]

