#!/usr/bin/env python
# coding: utf-8

# # Superstore Analysis އ

# > Context: With growing demands and cut-throat competitions in the market, a Superstore Giant is seeking your knowledge in understanding what works best for them. They would like to understand which products, regions, categories and customer segments they should target or avoid.

# <div class="alert alert-block alert-info" style="font-size:16px; font-family:Helvetica;">
#      📌 In this notebook we are going to analyse the superstore dataset that's contains lot's of informations such as the customer ID his name the city of the custome and so on. The notebook will be organized as above: <br> 
#     <b> Used Library -- First Look to the Data -- Data Cleaning -- Data Visualization -- Profit Per Category --  Most Ordered Category Per Region -- Look into the Date --  City & Profit -- Customers -- Machine Learning   </b>
#     
# </div>
# 
# 
# ![image](https://media-exp1.licdn.com/dms/image/C4E12AQHG1RNjrpiZ-w/article-inline_image-shrink_1500_2232/0/1561471706289?e=2147483647&v=beta&t=w0yQRQSY-ZmSI7KS3vR_nmUQaSbPcXJxWNMvUMHDFcg)

# <font size=4> Metadata </font>
# * Row ID => Unique ID for each row.
# * Order ID => Unique Order ID for each Customer.
# * Order Date => Order Date of the product.
# * Ship Date => Shipping Date of the Product.
# * Ship Mode=> Shipping Mode specified by the Customer.
# * Customer ID => Unique ID to identify each Customer.
# * Customer Name => Name of the Customer.
# * Segment => The segment where the Customer belongs.
# * Country => Country of residence of the Customer.
# * City => City of residence of of the Customer.
# * State => State of residence of the Customer.
# * Postal Code => Postal Code of every Customer.
# * Region => Region where the Customer belong.
# * Product ID => Unique ID of the Product.
# * Category => Category of the product ordered.
# * Sub-Category => Sub-Category of the product ordered.
# * Product Name => Name of the Product
# * Sales => Sales of the Product.
# * Quantity => Quantity of the Product.
# * Discount => Discount provided.
# * Profit => Profit/Loss incurred.

# # Used Library 📖 <br>
# We'll use basic python library well known for data analysis and observation 
# *  Numpy link: https://numpy.org
# *  Pandas link: https://pandas.pydata.org
# *  Matplotlib link: https://matplotlib.org
# *  Plotly link: https://plotly.com
# <hr>

# In[ ]:


import matplotlib.pyplot as plt 
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np 
import math 
import seaborn as sns
from pandas_profiling import ProfileReport
get_ipython().run_line_magic('matplotlib', 'inline')


# # First Look to the Data 👀
# 
# 
# <div class="alert alert-block alert-info" style="font-size:16px; font-family:Helvetica;">
#      📌 As I did in my others notebooks where i try to analyse the dataset,i like to create a function to get the data it can be useful when you got a bug to get the original dataset. In this first step we'll see the columns names,check if there are any missing value, the description and make a profile report, the profile report is very important to have a good understanding of the data it make statistics for you and show the partitions...
# </div>

# In[ ]:


def getData():
    df = pd.read_csv('../input/superstore-dataset-final/Sample - Superstore.csv',encoding='cp1252')
    return df 


# #### Note If you do not precise the encoding you'll get an error like this: <br>
# <code>'utf-8' codec can't decode byte 0xa0 in position 2944: invalid start byte </code>

# In[ ]:


df = getData()
df.head()


# 
# <div class="alert alert-block alert-info" style="font-size:16px; font-family:Helvetica;">
#      📌 We have the row id as colum we can see for the country only United States probably we'll need to make a little bit of data cleaning 
# </div>

# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


profiling = ProfileReport(df)
profiling.to_file("profiling.html")
profiling


# 
# <div class="alert alert-block alert-info" style="font-size:16px; font-family:Helvetica;">
#      📌 Some information that we got from the profile report <br>
#     Country has constant value "United States"	Constant <br>
#     Order ID has a high cardinality: 5009 distinct values	 <b> High cardinality </b><br>
# Order Date has a high cardinality: 1237 distinct values	 <b>High cardinality </b><br>
# Ship Date has a high cardinality: 1334 distinct values	<b>High cardinality</b><br>
# Customer Name has a high cardinality: 793 distinct values	<b>High cardinality</b><br>
# City has a high cardinality: 531 distinct values	<b>High cardinality</b><br>
# Product Name has a high cardinality: 1850 distinct values	<b>High cardinality</b><br>
# Discount is highly correlated with Profit	<b>High correlation</b><br>
# Profit is highly correlated with Sales and 1 other fields	<b>High correlation</b><br>
# </div>
# 
# 

# # Data Cleaning ⏳
# *Data Cleaning represent 90% of data processing* 

# In[ ]:


df.columns


# In[ ]:


df.drop('Row ID',axis=1,inplace=True)
df.drop('Order ID',axis=1,inplace=True)
df.drop('Country',axis=1,inplace=True)


# In[ ]:


df.columns


# In[ ]:


df.isnull().sum()


# # Data Visualization 🔮

# In[ ]:


sns.set(rc = {'figure.figsize':(15,8)})

df['Category'].value_counts().plot(kind='barh',title='Most Popular Category')


# In[ ]:


sns.countplot(x=df['Ship Mode'],palette='rocket')


# ## Profit Per Category 
# 
# 
# <div class="alert alert-block alert-info" style="font-size:16px; font-family:Helvetica;">
#      📌Category<br>
# Furniture           18451.2728 <br>
# Office Supplies    122490.8008<br>
# Technology         145454.9481<br>
# <b> Technology Category is the first one however is also the category that have the less order </b><br>
# </div>

# In[ ]:


df.groupby('Category')['Profit'].sum().plot(kind='barh',title='Category Profit')


# In[ ]:


df.groupby('Category')['Sales'].sum().plot(kind='barh',title='Category Sales')


# In[ ]:


df.columns


# In[ ]:


df.groupby('Category')['Quantity'].sum().plot(kind='barh',title='Quantity that has been sold')


# In[ ]:


sns.countplot(x=df['Segment'],palette='rocket')


# ### Most Ordered Category Per Region 🗺
# <b> Global Plot </b>
# 
# <div class="alert alert-block alert-info" style="font-size:16px; font-family:Helvetica;">
#      📌 We can see that in each category the West is number one 
# </div>

# In[ ]:


df.groupby('Region')['Category'].value_counts().plot(kind='barh',title='Quantity that has been sold')


# ### Repartition of the Profit per Region 💶

# In[ ]:


labels = df.groupby('Region')['Profit'].sum().index
values = df.groupby('Region')['Profit'].sum().values


# In[ ]:


colors = sns.color_palette('pastel')[0:4]
plt.pie(values, labels = labels, colors = colors, autopct='%.0f%%')
plt.title("Repartition of the profit per Region")
plt.show()


# In[ ]:


A = df[df['Category'] == 'Technology']

labels = A.groupby('Region')['Profit'].sum().index
values = A.groupby('Region')['Profit'].sum().values

plt.pie(values, labels = labels, colors = colors, autopct='%.0f%%')
plt.title("Repartition of the Profit for the Technology Category")
plt.show()


# <div class="alert alert-block alert-info" style="font-size:16px; font-family:Helvetica;">
#      📌 Even if the West order more technology product than the East, the East make more profit
# </div>

# # Version  ❶
# <hr>

# # Look into the Date ⏳

# In[ ]:


df.columns


# In[ ]:


def to_integer(dt_time):
    return 10000*dt_time.year + 100*dt_time.month + dt_time.day


# In[ ]:


import math 


# In[ ]:


def category_date(category):
    category_1 = df[df['Category'] == category]
    
    order_date =  pd.to_datetime(category_1["Order Date"])
    ship_date =  pd.to_datetime(category_1["Ship Date"])

    order_date = list(map(to_integer, order_date))
    ship_date = list(map(to_integer, ship_date))

    difference = []
    for date1, date2 in zip(order_date, ship_date):
        difference.append(date2- date1)

    mean = sum(difference) // len(difference)
    maxi = max(difference)
    mini = min(difference)
    
    print("Mean of the Category {} {} ".format(category,mean))
    print("Maximum date of the Category {} {} ".format(category,maxi))
    print("Minimum date of the Category {} {} ".format(category,mini))
    print("\n")
    print("\n")
    
category_date('Technology')

category_date('Furniture')
category_date('Office Supplies')


# # City 🌆 & Profit  💵

# <div class="alert alert-block alert-info" style="font-size:16px; font-family:Helvetica;">
#      📌 531 City there are too many city so the information may not be too interesting to deal with we can see which city order the most technology product or others categories 
# </div>

# In[ ]:


def city_profit(category):
    category_1 = df[df['Category'] == category]
    category_1.groupby('City')['Profit'].sum().sort_values(ascending=False)[:5].plot(kind='bar',title="Top 5 Cities that made the most profit in {}".format(category))
    


# In[ ]:


city_profit('Technology')


# In[ ]:


city_profit('Furniture')


# In[ ]:


city_profit('Office Supplies')


# <div class="alert alert-block alert-info" style="font-size:16px; font-family:Helvetica;">
#      📌 New York City is the heart of united states when we see those values hovewer we cannot see the partitions, the populations of each city can be a useful informations.
# </div>

# In[ ]:


plt.figure(figsize=(18,18))
sns.countplot(x=df['Sub-Category'],palette='rocket')
plt.title("Sub-Category Count")


# # Customers 🧓🏻 👩🏽 👨🏻 🧑 

# In[ ]:


df['Customer Name'].value_counts()[:10].plot(kind='barh',title='Customer Name & Product Ordered')


# In[ ]:


customers = list(df['Customer Name'].value_counts()[:10].index)
filters = df['Customer Name'].isin(customers)
A = df[filters]


# In[ ]:


colors = sns.color_palette('pastel')[0:]


values = A.groupby('Customer Name')['Profit'].sum()
values


# <div class="alert alert-block alert-info" style="font-size:16px; font-family:Helvetica;">
#      📌 Zuschuss Carroll is in the top 10 but her/his profit is negative
# </div>

# In[ ]:


A = A[A['Customer Name'] != 'Zuschuss Carroll']


# In[ ]:


del customers[-2]


# In[ ]:


plt.figure(figsize=(15,15))
values = A.groupby('Customer Name')['Profit'].sum()

plt.pie(values, labels = customers, colors = colors, autopct='%.0f%%')
plt.title("Repartition of the Profit for the top 9 Customer")
plt.show()


# In[ ]:


plt.pie(A['Region'].value_counts().values, labels = A['Region'].value_counts().index, colors = colors, autopct='%.0f%%')
plt.title("Order come from")
plt.show()


# # Version ❷
# <hr>

# # Machine Learning 🧙 
# <div class="alert alert-block alert-info" style="font-size:16px; font-family:Helvetica;">
#      📌 Now let's try to build a machine learning model to predict profit,Instead of predicting the exact Sales we can make this a classification problem and classify the Sales between n number of categories
# </div>

# In[ ]:


df.head()


# # Define Columns 🪦

# In[ ]:


X = df[['Segment','Ship Mode','Category','Profit','Region','Quantity','Sales']]
X.head()


# ### Add Categories of Sales ➕

# In[ ]:


values = X['Sales'].values
mini = min(values)
maxi = max(values)
mean = sum(values) / len(values)
mean,mini,maxi,len(values),mean / 3


# In[ ]:


array = []
value = mini
for i in range(4):
    value += mean // 6
    array.append(value)
    
array


# In[ ]:


def affectColumn(sale):
    index = 1
    i = 0
    while i < len(array) and sale > array[i]:
        index += 1
        i += 1
    return index
    
saleCategory = list(map(affectColumn,X['Sales'].values ))
max(saleCategory),len(saleCategory),min(saleCategory)


# In[ ]:


X['SaleCategory'] = saleCategory


# In[ ]:


X.head()


# In[ ]:


sns.countplot(x=X['SaleCategory'],palette='rocket')

