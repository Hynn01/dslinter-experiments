#!/usr/bin/env python
# coding: utf-8

# # E-Commerce Data
# ### Dataset: https://www.kaggle.com/benroshan/ecommerce-data
# What's inside:
# 1. List of Orders-This dataset contains purchase information. The information includes ID, Date of Purchase and customer details
# 2. Order Details- This dataset contains order ID, with the order price, quantity,profit, category and subcategory of product
# 3. Sales target-This dataset contains sales target amount and date for each product category
# 
# Firstly, we should import the required libraries, and our data.

# In[ ]:


import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen as uReq
import re


# In[ ]:


list_of_orders = pd.read_csv('/kaggle/input/ecommerce-data/List of Orders.csv')
order_details = pd.read_csv('/kaggle/input/ecommerce-data/Order Details.csv')
sales_target = pd.read_csv('/kaggle/input/ecommerce-data/Sales target.csv')


# Before we start to analyse the data we should view that are there empty cells, and are the columns in the expected type. After that, do correction, if needed.

# In[ ]:


list_of_orders.isna().sum()
order_details.isna().sum()
sales_target.isna().sum()

list_of_orders = list_of_orders.dropna()


# In[ ]:


type(sales_target['Month of Order Date'][0])
type(list_of_orders['Order Date'][0])

sales_target['Month of Order Date'] = pd.to_datetime(sales_target['Month of Order Date'], format = '%b-%y')
sales_target['Month of Order Date'] = pd.to_datetime(sales_target['Month of Order Date']).dt.to_period('M')
list_of_orders['Order Date'] = pd.to_datetime(list_of_orders['Order Date'])


# ## Which months did they hit the sales target?
# We have the data about the sales target in our sales_target table, and the data about the sales in the order_details and list_of_orders tables. If we bring them together, we can get the answer.

# In[ ]:


temp = pd.merge(list_of_orders[['Order ID','Order Date']],
                  order_details[['Order ID','Amount','Category']],
                  on = 'Order ID')
temp['Month of Order Date'] = pd.to_datetime(temp['Order Date']).dt.to_period('M')
temp = temp.groupby(['Month of Order Date','Category']).sum().reset_index()


# In[ ]:


target = pd.merge(sales_target, temp, on = 'Month of Order Date')
target = target[target['Category_x'] == target['Category_y']].reset_index(drop=True)
target = target.drop(columns = 'Category_y')
target = target.rename(columns = {'Category_x':'Category'})
target['Month of Order Date'] = target['Month of Order Date'].astype(str)
target.head()


# Now we can find the months, where the sales reached the sales target.

# In[ ]:


target[target['Amount'] >= target['Target']]


# We should make further examination on the months. How big was the difference? For more insight, we will break down the data by the category of products.

# In[ ]:


target['difference'] = target['Amount'] - target['Target']

furniture = target[target['Category'] == 'Furniture'].drop(columns = 'Category').set_index('Month of Order Date')
electronics = target[target['Category'] == 'Electronics'].drop(columns = 'Category').set_index('Month of Order Date')
clothing = target[target['Category'] == 'Clothing'].drop(columns = 'Category').set_index('Month of Order Date')
summation = target.groupby(['Month of Order Date']).sum()


# In[ ]:


plt.figure(figsize = (10,7))

plt.plot(furniture['difference'], marker = '.', label = 'furniture')
plt.plot(electronics['difference'], marker = '.', label = 'electronics')
plt.plot(clothing['difference'], marker = '.', label = 'clothing')
plt.plot(summation['difference'],linewidth = .5, label = 'summation')
plt.legend()

plt.axhline(y = 0, color = 'black', linewidth = .5)
plt.xticks(summation.index[::2])
plt.xlabel('Date', fontsize = 14)
plt.ylabel('($)', fontsize = 14)
plt.title('Difference between the real sales and the target sales', fontdict={'fontweight':'bold', 'fontsize':18})

plt.show()


# Interestingly, we can see a periodicity in the data. After 2018 June, overestimation and underestimation alternate, which can be seen not only by the summation line, but also by the categories. The periodicity is only 2 month, so it can't be due to big holidays. The reason is maybe, that the products are not so lasting, so they have to be replaced after a few weeks. The customers don't have an ID, so we can't examine how frequently do they order.

# ## Online spending habits by state
# We have a lot of geographical data in the list_of_orders table, so we could summarize this type of data.
# Firstly, we construct a new table, called Cities. This will contain the latitude and longitude parameters, which are missing from our database. The source: https://www.latlong.net/category/cities-102-15.html

# In[ ]:


filename = 'Cities.csv'
headers = 'City,Latitude,Longitude\n'

f = open(filename, "w")
f.write(headers)

for j in range(1,9):    
    page_url = f'https://www.latlong.net/category/cities-102-15-{j}.html'

    uClient = uReq(page_url)
    page_soup = soup(uClient.read(), "html.parser")
    uClient.close()

    rows = page_soup.findAll('tr')
    rows = rows[1:]
    
    for row in rows:

        cell = row.findAll('td')

        City = cell[0].text
        Latitude = cell[1].text
        Longitude = cell[2].text

        f.write(City.replace(',', '|') + ',' + Latitude + ',' + Longitude + '\n')

f.close()


# In[ ]:


cities = pd.read_csv('/kaggle/working/Cities.csv')
cities['Latitude'].astype(float)
cities['Longitude'].astype(float)
cities['City'] = cities['City'].str.split('|',expand=False).loc[:]
cities['City'] = cities['City'].str[0]


# After we constructed the Cities table, we should examine, how accurate is it.

# In[ ]:


toiter = pd.DataFrame(list_of_orders['City'].unique())
for index, row in toiter[0].iteritems():
    if row in cities['City'].tolist():
        pass
    else:
        print(row)


# As we can see, there are 5 missing items. We could search for extra sources, or we could make up for by hand. Because 5 is a really small amount of data, we can complete the table with Google Maps.

# In[ ]:


my_cities = pd.DataFrame(list_of_orders.groupby(['City']).count()['Order ID'])
city_map = my_cities.merge(cities[['City','Longitude','Latitude']], how='left', on='City')
city_map.set_index('City',inplace = True)


# In[ ]:


city_map.at['Allahabad','Latitude'] = 25.435113
city_map.at['Allahabad','Longitude'] = 81.845084
city_map.at['Goa','Latitude'] = 15.331737
city_map.at['Goa','Longitude'] = 74.126248
city_map.at['Kashmir','Latitude'] = 33.925603
city_map.at['Kashmir','Longitude'] = 76.137685
city_map.at['Kohima','Latitude'] = 25.674477
city_map.at['Kohima','Longitude'] = 94.108727
city_map.at['Simla','Latitude'] = 31.111610
city_map.at['Simla','Longitude'] = 77.169854


# In[ ]:


city_map = gpd.GeoDataFrame(city_map, 
                            geometry=gpd.points_from_xy(city_map.Longitude, city_map.Latitude))
city_map.drop(columns = ['Longitude', 'Latitude','Order ID'], inplace = True)
city_map.reset_index(inplace = True)


# Secondly, we get the required data for the states. The source: http://www.diva-gis.org/gdata

# In[ ]:


ind_dist = gpd.read_file(r'/kaggle/input/ind-states/IND_adm1.shp')
ind_dist = ind_dist[['NAME_1','geometry']]
ind_dist.rename(columns = {'NAME_1' : 'State'}, inplace = True)


# Check if our states from list_of_orders are compatible with the states from ind_dist.

# In[ ]:


toiter = pd.DataFrame(list_of_orders['State'].unique())
for index, row in toiter[0].iteritems():
    if row in ind_dist['State'].tolist():
        pass
    else:
        print(row)


# As we can see, there is one state in our data, that missing from the imported data about the states. But if we examine by hand, we can find Kerala in both table. Maybe there is a typo. We can see, that Kerala from list_of_orders contains an extra space at the end: 'Kerala '. We should correct it.

# In[ ]:


list_of_orders.replace('Kerala ','Kerala',inplace = True)


# After a few adjustment we can plot our data to a map to observ, where do people spend online the most.

# In[ ]:


orders = list_of_orders.groupby(['State']).count()['Order ID']
ind_dist = ind_dist.merge(orders, how = 'left', on='State')
ind_dist.rename(columns={'Order ID':'Orders'},inplace=True)
ind_dist['Orders'] = ind_dist['Orders'].fillna(0)


# In[ ]:


fig, ax = plt.subplots(figsize = (10,10))

divider = make_axes_locatable(ax)
cax = divider.append_axes(position = 'right', size = '7%', pad = 0.1)

ind_dist.plot(column = 'Orders', legend = True, cmap = 'Oranges_r', ax = ax, cax = cax,
             legend_kwds = {'label':'# of orders'}, alpha=0.8, edgecolor='k', linewidth=0.2)

city_map.plot(ax=ax, marker='o', color='r', markersize=45, edgecolor = 'k')

ax.set_title('# of orders by states in India', fontsize = 18)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.show()


# Based on this dataset, we can see that the most cities are located on the northwest side of the country, but the substantial majority of orders by states came from Madhya Pradesh and Maharashtra.

# In[ ]:


ind_dist[['State','Orders']].sort_values('Orders', ascending = False).head()


# In[ ]:




