#!/usr/bin/env python
# coding: utf-8

# ![](https://images4.alphacoders.com/216/thumb-1920-216804.jpg)

# The history of the first steam car dates back to 1769 when Nicholas Joseph Kignot invented the first steam-powered car on full scale,  In 1801 Joseph Trevethick also designed a four-wheeled steam engine.After that, the steam engines were continuously developed and the various systems used were improved,  Until 1885, Karl Baines developed the first gasoline-fueled internal combustion engine, which is still in use today in cars, and has patented it.Later, in 1896, Henry Ford  manufactured the first four-wheel gasoline-powered vehicle; the first air-pressure tires were manufactured in the previous year and were manufactured by Mechelen.Since then, the automotive industry has witnessed a very large development; many different designs have been made for cars.
# At present, in the 20th and 21st centuries, electric cars has been invented in addition to and hybrid cars which powered by fuel and electricity.The first hybrid car manufactured by Ferdinand Porsche was in 1901.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objects as go
import plotly.offline as py
import plotly.express as px
import pycountry
py.init_notebook_mode(connected=True)
import folium 
from folium import plugins
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
plt.rcParams['figure.figsize'] = 8, 5
pd.options.mode.chained_assignment = None 
pd.set_option('display.max_columns',None)
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv')
df.drop('Unnamed: 0',axis = 1, inplace = True)
df.head()


# In[ ]:


df.shape


# So we have 2499 rows and 12 columns in this dataset.

# In[ ]:


df.describe()


# **Observations:**
# * Minimum price of car is zero. this cannot be true. we need to replace these values with an appropriate value.
# * Maximum price of the car in this dataset is $84990.
# * Oldest model of car is from 1973.
# * First quartile value (25%) lies in the year 2016. This indicated that more than 75% of the cars are new models and are built after 2016.

# ### Missing Values

# In[ ]:


#missing data
df.isnull().sum().sort_values(ascending=False)


# We don't have any missing values in this dataset.

# ### Feature Engineering

# As we saw earlier that few of the cars have price as zero. This isn't practical.So, we need to replace these zero values with a resonably good value. In this case i will replace it with median of price column.

# In[ ]:


median_price = df['price'].median()
df['price'] = df['price'].astype(int)
df['price'].replace(0,median_price ,inplace=True)


# ### Number of Cars per Brand

# In[ ]:


cars_per_brand = df.groupby('brand')['model'].count().reset_index().sort_values('model',ascending = False).head(10)
cars_per_brand = cars_per_brand.rename(columns = {'model':'count'})
fig = px.bar(cars_per_brand, x='brand', y='count', color='count')
fig.show()


# Half of the cars in this dataset belong to the Ford Company!

# ### Cars by Model Year

# In[ ]:


cars_by_model_year = df.groupby('year')['model'].count().reset_index().sort_values('model',ascending = False)
cars_by_model_year = cars_by_model_year[cars_by_model_year['year'] >= 2010]
cars_by_model_year = cars_by_model_year.rename(columns = {'model':'count'})
fig = px.bar(cars_by_model_year, x='year', y='count', color='count')
fig.show()


# Majority of the cars in this dataset were manufactured after 2015.

# ### Cars by Color

# In[ ]:


car_colors = df.groupby('color')['model'].count().reset_index().sort_values('model',ascending = False).head(10)
car_colors = car_colors.rename(columns = {'model':'count'})
fig = px.bar(car_colors, x='color', y='count', color='count')
fig.show()


# It looks like most of the people prefer either White or Black color cars.

# ### Cars by State

# In[ ]:


cars_per_state = df.groupby('state')['model'].count().reset_index().sort_values('model',ascending = False).head(10)
cars_per_state = cars_per_state.rename(columns = {'model':'count'})
fig = px.bar(cars_per_state, x='state', y='count', color='count')
fig.show()


# Most of the cars in this dataset are registered in Pennsylvania.

# ### Most Expensive Cars

# In[ ]:


expensive_cars = df.sort_values('price',ascending = False).head(2)
fig = px.bar(expensive_cars, x='brand', y='price', color='price')
fig.show()


# 

# ### Bivariate Analysis

# In[ ]:


sns.distplot(df['price']);


# We can see that the car price column has a positive slope.

# In[ ]:


#skewness and kurtosis
print("Skewness: %f" % df['price'].skew())
print("Kurtosis: %f" % df['price'].kurt())


# Skewness is a measure of symmetry, or more precisely, the lack of symmetry. A distribution, or data set, is symmetric if it looks the same to the left and right of the center point.
# Kurtosis is a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution. That is, data sets with high kurtosis tend to have heavy tails, or outliers. Data sets with low kurtosis tend to have light tails, or lack of outliers. 

# ### Price vs Model Year

# In[ ]:


data = df[['price','year']]
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x='year', y="price", data=data)


# As expected price of the cars and the model year are proportional.i.e. Newer the model of the car, higher the price.

# ### Correlation

# In[ ]:


#correlation matrix
corrmat = df.corr()
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(corrmat,annot = True);


# **Observations:**
# * Price and Model year have poistive correlation,i.e. newer the car, higher the price.
# * Price and Mileage have negatve correlation, i.e. as the  miles traveled by the car increases, it's price decreases.

# ### Pairplot

# In[ ]:


#scatterplot
sns.set()
sns.pairplot(df, size = 2)
plt.show();


# A pairplot plots a pairwise relationships in a dataset. The pairplot function creates a grid of Axes such that each variable in data will by shared in the y-axis across a single row and in the x-axis across a single column. That creates plots as shown above.

# ## Please do upvote if you like my work!
