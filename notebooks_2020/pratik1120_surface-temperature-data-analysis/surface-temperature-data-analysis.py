#!/usr/bin/env python
# coding: utf-8

# # Data Story 4

# This is the day 4 of telling stories with data. 
# 
# In this data I will try to analyse which countries and cities contribute the most to the world temperature

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Let's first check the land temperatures based on Countries.

# In[ ]:


# reading in GlobalLandTemperaturesByCountry.csv

gltc = pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv')


# In[ ]:


gltc.head(2)


# Now we can see that the data ranges from the year 1743 to 2013(not in all cases though) for each of the months.

# Let's first randomly pick up one country and check it's data over the years. I am picking up the country Zimbabwe

# In[ ]:


#separating zimbabwe data
df = gltc[gltc['Country']=='Zimbabwe']

#dropping rows with NaN values
df.dropna(inplace=True)

# first lets bifurcate the months and year data for the dt
df.loc[:,'dt'] = pd.to_datetime(df['dt'])

df.loc[:,'month'] = [x.month for x in list(df['dt'])]
df.loc[:,'year'] = [x.year for x in list(df['dt'])]


# In[ ]:


plt.plot(df['dt'], df['AverageTemperature'])
plt.show()


# Not much sense can be made out of this plot so it's better to analyse one year data at a time
# 
# Let's then analyze the data for 2012 year(because it contains all the months)

# In[ ]:


fig = plt.figure(figsize=(10,5))
plt.plot(df.loc[df['year']==2012, 'dt'], df.loc[df['year']==2012,'AverageTemperature'])
plt.title('Temperature at Zimbabwe in 2012')
plt.xlabel('Months')
plt.ylabel('Average Temperature')
plt.show()


# From the above plot w can deduce that the temperature in Zimbabwe reaches it's lowest point in the month of July but again starts rising after that which seems pretty odd because being an Indian I expect the least value to be around the months of Dec-Feb

# Now let's check the couuntry that shows the highest temperature value in all these years.
# 
# Then I will analyse that country's data and maybe find out the top 10 contributors

# In[ ]:


#checking highest temperature

gltc[gltc['AverageTemperature']==gltc['AverageTemperature'].max()]


# Okay so the highest temperature reach was by the country Kuwait with an AverageTemperature of 38.842. It reached the highest temperature on July,2012

# In[ ]:


#lets analyse Kuwait daa throughout 2012

df = gltc[gltc['Country']=='Kuwait']
df.dropna(inplace=True)
df.loc[:,'dt'] = pd.to_datetime(df['dt'])
df.loc[:,'month'] = [x.month for x in list(df['dt'])]
df.loc[:,'year'] = [x.year for x in list(df['dt'])]
fig = plt.figure(figsize=(10,5))
plt.plot(df.loc[df['year']==2012, 'dt'], df.loc[df['year']==2012,'AverageTemperature'])
plt.show()


# Lets see what the mean temperature is and then we can compare Kuwait temperature to it

# In[ ]:


mean_temp = gltc['AverageTemperature'].mean()


# In[ ]:


#lets plot again
fig = plt.figure(figsize=(10,5))
plt.plot(df.loc[df['year']==2012, 'dt'], df.loc[df['year']==2012,'AverageTemperature'])
plt.axhline(mean_temp)
plt.show()


# Woh te situation doesn't seem too favourable for Kuwait.
# 
# Lets compare its performance for the last 5 years.

# In[ ]:


fig = plt.figure(figsize=(10,5))
years = [2008,2009,2010,2011,2012]
for year in years:
    plt.plot(df.loc[df['year']==year, 'month'], df.loc[df['year']==year,'AverageTemperature'], label=year)
plt.title('Temperature variation in Kuwait in last 5 years')
plt.xlabel('Months')
plt.ylabel('Average Temperature')
plt.legend(loc='upper left')
plt.axhline(mean_temp)
plt.show()


# Looking at the plot it seems obvious that the temperature of Kuwait remains that way throughout every year
# 
# Lets check the mean for each years

# In[ ]:


df['year'].unique()


# In[ ]:


mean_by_year = []
for year in list(df['year'].unique()):
    df1 = df[df['year']==year]
    mean_by_year.append(df1['AverageTemperature'].mean())
    
fig = plt.figure(figsize=(15,10))
plt.bar(list(df['year'].unique()), mean_by_year)
plt.show()


# This does not look good so let's the data for last 10 years

# In[ ]:


mean_by_year = []
years = list(df['year'].unique()[-10:])
for year in years:
    df1 = df[df['year']==year]
    mean_by_year.append(df1['AverageTemperature'].mean())
    
fig = plt.figure(figsize=(10,5))
barlist = plt.bar(years, mean_by_year)
barlist[0].set_color('r')
barlist[1].set_color('b')
barlist[2].set_color('g')
barlist[3].set_color('r')
barlist[4].set_color('y')
barlist[5].set_color('m')
barlist[6].set_color('c')
barlist[7].set_color('k')
barlist[8].set_color('g')
barlist[9].set_color('y')
plt.xticks(np.arange(2004,2014),labels=years)
plt.title('Temperature in Kuwait in last decade')
plt.show()


# Okay now I thought of checking the top 5 countries who are having the highest average tempeartures in the year 2012

# In[ ]:


gltc['dt'] = pd.to_datetime(gltc['dt'])
df = gltc[gltc['dt'].dt.year==2012]
df = df.sort_values('AverageTemperature', ascending=False)
df.head()


# Since I see that most countries reached their highest values at around the month of July, so let's consider only that month so that I get unique countries

# In[ ]:


df = gltc[(gltc['dt'].dt.year==2012) & (gltc['dt'].dt.month==7)]
df = df.sort_values('AverageTemperature', ascending=False)
top_countries = list(df['Country'].head())
df.head()


# Okay so now I got my top 5 countries from the filtering. I will store them in a list and then check their temperature variation

# In[ ]:


gltc['month'] = gltc['dt'].dt.month
gltc['year'] = gltc['dt'].dt.year


# In[ ]:


fig = plt.figure(figsize=(10,5))
for country in top_countries:
    plt.plot(gltc.loc[(gltc['year']==2012)&(gltc['Country']==country), 'month'], gltc.loc[(gltc['year']==2012)&(gltc['Country']==country),'AverageTemperature'], label=country)
plt.legend(loc="upper left")
plt.title('Top 5 countries with highest Average Soil Temperature for 2012')
plt.xlabel('Months')
plt.ylabel('Average Temperature')
plt.show()


# Okay so this is where I end today's analysis. There is a lot more to explore in this data and it is fun as well. So I will continue again tomorrow.

# # Data Story 5

# Okay so yesterday I ended till the analysis of top 5 countries with the highest average temperature. 
# 
# Today I am thinking of analyzzing the cities. This may provide some useful information.

# In[ ]:


data = pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCity.csv')


# In[ ]:


data.tail(10)


# Let's see the city that reached the highest temperature in the dataset.

# In[ ]:


data[data['AverageTemperature']==data['AverageTemperature'].max()]


# Okay so the city name is Warqla and it is in the country Algeria.
# 
# The highest temperature reached in the year 1761
# 
# Let's now check the highest temperature reached in 2013(the last recorded year in the dataset)

# In[ ]:


data['dt'] = pd.to_datetime(data['dt'])
data['year'] = data['dt'].dt.year
data['month'] = data['dt'].dt.month


# In[ ]:


df = data[data['year']==2013]
df.sort_values('AverageTemperature', ascending=False, inplace=True)
df = df.reset_index()
df.head()


# Okay so I found the top 5 cities with the highest temperature.
# 
# Lets see their variation

# In[ ]:


cities = df['City'][:5].tolist()
fig = plt.figure(figsize=(10,5))
for city in cities:
    df1 = df[df['City']==city]
    df1.sort_values('month', inplace=True)
    plt.plot(df1['month'], df1['AverageTemperature'], label=city)
plt.legend(loc='upper left')
plt.title('Top 5 cities having highest temperature in 2013')
plt.show()


# So now the top 5 cities with the highest temperature are revealed.

# Let's now check how many unique countries data do we have in the dataset

# In[ ]:


data['City'].nunique()


# So we have 3448 unique countries. Since we have geographic data. LEt's try and visualize that.
# 
# A lot of things come up from a geographic data analysis.
# 
# I will take mean of average temperature for each countries

# Creating new dataframe with the mean values

# In[ ]:


def clean_lat_data(lat):
    lat1 = lat.replace('N','')
    lat1 = float(lat1.replace('S',''))
    if lat[-1] == 'S':
        lat1 *= (-1)
    return lat1

def clean_lon_data(lon):
    lon1 = lon.replace('E','')
    lon1 = float(lon1.replace('W',''))
    if lon[-1] == 'W':
        lon1 *= (-1)
    return lon1


# In[ ]:


df = data[['AverageTemperature','City','Latitude','Longitude']]
cities = list(df['City'].unique())
latitudes = []
longitudes = []
temp = []
for city in cities:
    df1 = df[df['City']==city]
    df1.dropna(inplace=True)
    temp.append(df1['AverageTemperature'].mean())
    lats = df1['Latitude'].tolist()
    lons = df1['Longitude'].tolist()
    lat = clean_lat_data(lats[0])
    lon = clean_lon_data(lons[0])
    latitudes.append(lat)
    longitudes.append(lon)


# In[ ]:


len(cities), len(latitudes), len(longitudes), len(temp)


# So I tried to separate the data for each cities but it took to long and crashed the notebook thrice. 
# 
# So now I shifted to the thought of using unique countries to decrease the variability.

# In[ ]:


df = data[['AverageTemperature','Country','Latitude','Longitude']]
countries = list(df['Country'].unique())
latitudes = []
longitudes = []
temp = []
for country in countries:
    df1 = df[df['Country']==country]
    df1.dropna(inplace=True)
    temp.append(df1['AverageTemperature'].mean())
    lats = df1['Latitude'].tolist()
    lons = df1['Longitude'].tolist()
    lat = clean_lat_data(lats[0])
    lon = clean_lon_data(lons[0])
    latitudes.append(lat)
    longitudes.append(lon)


# In[ ]:


len(countries), len(latitudes), len(longitudes), len(temp)


# In[ ]:


new_df = pd.DataFrame({'Country':countries, 'Latitude': latitudes, 'Longitude':longitudes, 'AverageTemperature': temp})
new_df.head()


# In[ ]:


#summoning the libraries needed
from mpl_toolkits.basemap import Basemap

m = Basemap(projection="merc", llcrnrlat=-40, urcrnrlat=60, llcrnrlon=-50, urcrnrlon=150)

#creating instances
x , y = m(new_df["Longitude"].tolist(),new_df["Latitude"].tolist())


# In[ ]:


fig = plt.figure(figsize=(10,7))
plt.title("Temperature of Countries")
m.scatter(x, y, s=1, c='red')
m.drawcoastlines()
plt.show()


# With that I will end my today's data story. Today most time I spent in figuring out the structure.
# 
# Didn't really do much analysis.

# # See you in next data story. Bye.

# In[ ]:




