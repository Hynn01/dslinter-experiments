#!/usr/bin/env python
# coding: utf-8

# <img src="https://i.pinimg.com/originals/9c/ef/6f/9cef6fef614f43e2ffe6e82e9790512b.jpg" style="width:1000px;height:800px;">

# ** Initially, I have to say something about car. Most popular product is this so that what are they looking like? How fast can it does? It is significiant point at that issue. Everybody wants to drive pretty car like McLaren P1 LM, Bugatti Divo, Rolls Royce Sweptail.The car that you are driven demonstrates who you are or what you have in Turkey.I hope all people drive what they want that car.**

# # Information and Stage of Prepare Data.

# In[ ]:


import numpy as np
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


car_data=pd.read_csv("/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv")
car_data.head()
#It function gives us 5 line at the top in the data.


# In[ ]:


car_data = car_data.drop(columns = ['Unnamed: 0'])


# In[ ]:


car_data.describe().T
#Some statistical explanation. Do you know Box Plot ? 
# It displays %25 is first quartile. 
# %50 is median and gives at the middle value when you ascending data.
# %75 is third quartile.
#min-max value is value that gives at lowest value and at highest value.


# Observations:
# 
# * Minimum price of car is zero. this cannot be true. we need to replace these values with an appropriate value.
# * Maximum price of the car in this dataset is $84990.
# * Oldest model of car is from 1973.
# * First quartile value (25%) lies in the year 2016. This indicated that more than 75% of the cars are new models and are built after 2016.

# In[ ]:


car_data.columns
#Columns that we have in data.


# In[ ]:


car_data.dtypes
#we can see what's the type of columns.


#  # Data

# In[ ]:


#missing data
car_data.isnull().sum().sort_values(ascending=False)


# ## Feature Engineering
# As we saw earlier that few of the cars have price as zero. This isn't practical.So, we need to replace these zero values with a resonably good value. In this case i will replace it with median of price column.

# In[ ]:


median_price = car_data['price'].median()
car_data['price'] = car_data['price'].astype(int)
car_data['price'].replace(0,median_price ,inplace=True)


# # VISUALIZATION

# In[ ]:


brand_of_car = car_data.groupby('brand')['model'].count().reset_index().sort_values('model',ascending = False).head(10)
brand_of_car = brand_of_car.rename(columns = {'model':'count'})
fig = px.bar(brand_of_car, x='brand', y='count', color='count')
fig.show()

#You can reach a lot of information about car brand and their count 


# In[ ]:


expensive_cars = car_data.sort_values('price',ascending = False).head(2)
fig = px.bar(expensive_cars, x='brand', y='price', color='price')
fig.show()
#We saw which car brand is expensive in this vis.


# In[ ]:


data = car_data[['price','year']]
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.regplot(x='year', y="price", data=data)


# In[ ]:


# Pair Plots
# Plot pairwise relationships in a dataset

cont_col= ['year','price','mileage']
sns.pairplot(car_data[cont_col],  kind="reg", diag_kind = "kde"   )
plt.show()


# # What We Did It

# We can see relationship between car price, year, brand like that so if you come here, please you upvote my notebook and feedback to me.Thank you so much for everything :) 
