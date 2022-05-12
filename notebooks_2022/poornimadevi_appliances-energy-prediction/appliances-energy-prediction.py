#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import datetime as dt
from datetime import datetime
from matplotlib import pyplot as plt
import plotly.express as px

import warnings 
warnings.simplefilter('ignore')
get_ipython().run_line_magic('matplotlib', 'Inline')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# #  **Attribute Information:**

# date time year-month-day hour:minute:second \
# Appliances, energy use in Wh \
# lights, energy use of light fixtures in the house in Wh \
# T1, Temperature in kitchen area, in Celsius\
# RH_1, Humidity in kitchen area, in % \
# T2, Temperature in living room area, in Celsius \
# RH_2, Humidity in living room area, in % \
# T3, Temperature in laundry room area \
# RH_3, Humidity in laundry room area, in %\
# T4, Temperature in office room, in Celsius\
# RH_4, Humidity in office room, in %\
# T5, Temperature in bathroom, in Celsius\
# RH_5, Humidity in bathroom, in %\
# T6, Temperature outside the building (north side), in Celsius\
# RH_6, Humidity outside the building (north side), in %\
# T7, Temperature in ironing room , in Celsius\
# RH_7, Humidity in ironing room, in %\
# T8, Temperature in teenager room 2, in Celsius\
# RH_8, Humidity in teenager room 2, in %\
# T9, Temperature in parents room, in Celsius\
# RH_9, Humidity in parents room, in %\
# To, Temperature outside (from Chievres weather station), in Celsius\
# Pressure (from Chievres weather station), in mm Hg\
# RH_out, Humidity outside (from Chievres weather station), in %\
# Wind speed (from Chievres weather station), in m/s\
# Visibility (from Chievres weather station), in km\
# Tdewpoint (from Chievres weather station), Â°C\
# rv1, Random variable 1, nondimensional\
# rv2, Random variable 2, nondimensional

# # Some questions I am trying to answer: 
# * How much energy is consumed in indoor and outdoor appliances?
# * what is the coerration of humidity and temperature in indoor and outdoor environment?
# * Does the environment of sensor affects the way enegry is consumed by appliances?
# * Identify the appliances with low efficiency in home (which room)
# * relationship between appliances and light
# * How weather is coerrated with the energy consumption in homes and why this study is important? 

# # Introduction:
# 
# The data set is at 10 min for about 4.5 months. The house temperature and humidity conditions were monitored with a ZigBee wireless sensor network. Each wireless node transmitted the temperature and humidity conditions around 3.3 min. Then, the wireless data was averaged for 10 minutes periods. The energy data was logged every 10 minutes with m-bus energy meters. Weather from the nearest airport weather station (Chievres Airport, Belgium) was downloaded from a public data set from Reliable Prognosis (rp5.ru), and merged together with the experimental data sets using the date and time column. Two random variables have been included in the data set for testing the regression models and to filter out non predictive attributes (parameters).

# # Data exploration:

# In[ ]:


#reading the dataset

data = pd.read_csv("../input/appliances-energy-prediction/KAG_energydata_complete.csv")


# In[ ]:


#looking at the dataset
data.head(3)


# In[ ]:


data.tail(5)


# In[ ]:


#shape of the date
data.shape


# In[ ]:


data.describe()


# In[ ]:


#looking at null values
data.isnull().sum()


# In[ ]:


#checking datatype
data.dtypes


# In[ ]:


data.info()


# In[ ]:


#converting date into datetime
data['date'] = data['date'].astype('datetime64[ns]')


# In[ ]:


data.head(2)


# In[ ]:


data['Date'] = pd.to_datetime(data['date']).dt.date
data['Time'] = pd.to_datetime(data['date']).dt.time
data['hour'] = data['date'].dt.hour
data['month'] = data['date'].dt.month
data.tail(2)
#data['Dates'] = data['date'].values


# # Data Visualization:
# **- Correlation for all columns**

# In[ ]:


plt.figure(figsize = (20,10))
sns.heatmap(data= data.corr(), cmap="YlGnBu", annot= True)
plt.title("Pairwise correlation of all the columns in the dataframe ")
plt.show()


# **Analyzing Humdity :**

# In[ ]:


#dropping columns to see humidity indoor and outdoor and how its correlated over time period 
new_data = data.loc[:, data.columns.drop(['date','Appliances', 'lights','T1','T2','T3','T4','T5','T6','T7','T8','T9','T_out','Press_mm_hg','Windspeed','Visibility','Tdewpoint','rv1','rv2'])]
print(new_data)


# **-Correlation matrix for humdity indoor and outdoor**

# In[ ]:


corr_data=new_data.corr()


# In[ ]:


sns.clustermap(corr_data,  cmap="BuPu", annot=True, linecolor="black", linewidth=3)


# In[ ]:


sns.pairplot(new_data, kind='scatter')
plt.show()


# RH_1, Humidity in kitchen area, in %
# RH_2, Humidity in living room area, in %
# RH_3, Humidity in laundry room area, in %
# RH_4, Humidity in office room, in %
# RH_5, Humidity in bathroom, in %
# RH_6, Humidity outside the building (north side), in %
# RH_7, Humidity in ironing room, in %
# RH_8, Humidity in teenager room 2, in %
# RH_9, Humidity in parents room, in %
# RH_out, Humidity outside (from Chievres weather station), in %

# **Energy consumption of appliances:(time series) For First Month**

# In[ ]:


data.head(3)


# In[ ]:


data['Date'] = pd.to_datetime(data['Date'])
energy = data.loc[(data['Date'] >= '2016-01-11')
                     & (data['Date'] < '2016-01-30')]


# In[ ]:


energy = data[[ 'Date','Appliances']]
energy.tail(2)


# In[ ]:


energy.dtypes


# In[ ]:


energy['Date'] = pd.to_datetime(energy['Date'])
# Set the date as index 
energy = energy.set_index('Date')
energy.head()


# In[ ]:


import warnings
y = energy['Appliances']
fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(y,marker='.', linestyle='-', linewidth=0.5, label='Weekly')
ax.plot(y.resample('M').mean(),marker='o', markersize=8, linestyle='-', label='Monthly Mean Resample')
ax.set_ylabel('Appliances')
ax.legend();


# **energy consumption for whole time period:**

# In[ ]:


energywhole = data[['Date','Appliances']]
energywhole.tail(2)


# In[ ]:


energywhole['Date'] = pd.to_datetime(energywhole['Date'])
# Set the date as index 
energywhole = energywhole.set_index('Date')
energywhole.head(3)


# In[ ]:


y = energywhole['Appliances']
fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(y,marker='.', linestyle='-', linewidth=0.5, label='Monthly')
ax.plot(y.resample('M').mean(),marker='o', markersize=8, linestyle='-', label='Monthly Mean Resample')
ax.set_ylabel('Appliances')
ax.legend();


# feb month has the highest energy consumption of appliances in home. This could mean people stayed more inside in the month feb. To find out if they stayed inside more because of weather conditions further analysis is made.

# **Energy distribution of light (Wh) in appliances :**

# In[ ]:


energy1 = data[['Appliances','lights']]


# In[ ]:


energy1.plot( x='Appliances', y='lights', figsize=(20, 5),kind='hist')
plt.xlabel("Appliances in Wh ")
plt.ylabel("Lights (wh), frequency")
plt.title("energy consumption distribution of light in wh as a appliance ")
plt.show()


# The most used light is around 10 Watt-hour and the less used light bulb is 40 wh   

# In[ ]:


x= data["Appliances"]
y= data["lights"]
size=(20)
plt.scatter(x,y,color=["r"], alpha=0.5)


# 

#  **Appliances energy consumption with regards to weather outside:** 

# In[ ]:


data.head(2)


# In[ ]:


energy2 = data[['Appliances','Windspeed','Press_mm_hg']]


# In[ ]:


corr= energy2.corr()
corr


# In[ ]:


plt.figure(figsize=(7,4))
sns.heatmap(corr, annot=True, linecolor="black", linewidths=1, fmt='.1f')


# In[ ]:


fig, ax = plt.subplots(1, figsize=(7,5))
sns.kdeplot(energy2.Appliances,energy2.Press_mm_hg,cmap='Blues',
           shade=True, shade_lowest=False)
plt.scatter(energy2.Appliances, energy2.Press_mm_hg,color='orangered')


# In[ ]:


energyweather = data[['Date','Press_mm_hg']]
energyweather.tail(2)


# In[ ]:


energyweather['Date'] = pd.to_datetime(energyweather['Date'])
# Set the date as index 
energyweather = energyweather.set_index('Date')
energyweather.head(3)


# In[ ]:


#ploting atmospheric pressure
date = data["Date"]
Pressure = data["Press_mm_hg"]
plt.figure(figsize=(18, 5))
plt.plot(date, Pressure, label="low")
plt.title("Pressure")
plt.xlabel("date")
plt.ylabel("Pressure")
plt.show()

#ploting appliance
date = data["Date"]
Appliance = data["Appliances"]
plt.figure(figsize=(18, 5))
plt.plot(date, Appliance, label="low")
plt.title("Applliance energy consumption")
plt.xlabel("date")
plt.ylabel("Appliance")
plt.show()

#ploting appliance
date = data["Date"]
light = data["lights"]
plt.figure(figsize=(18, 5))
plt.plot(date,light, label="low")
plt.title("light energy consumption")
plt.xlabel("date")
plt.ylabel("light")
plt.show()


# when the atomospheric pressure is low it means it is raining, when the pressure is high it is warm weather. Jan month has the most energy consumption by appliances. So, it means peope used appliances more when its hot outside.  

# In[ ]:


#plotting pressure and appliances
fig,ax = plt.subplots(figsize=(15,7))


chart=sns.lineplot(x='Appliances', y='Press_mm_hg', data= data)
sns.despine(left=True)


# In[ ]:


#plotting windspeed and appliances
fig,ax = plt.subplots(figsize=(15,7))


chart=sns.lineplot(x='Appliances', y='Windspeed', data= data)
sns.despine(left=True)


# In[ ]:


# plotting atomospheric pressure
y = energyweather['Press_mm_hg']
fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(y,marker='.', linestyle='-', linewidth=0.5, label='Monthly')
ax.plot(y.resample('M').mean(),marker='o', markersize=8, linestyle='-', label='Monthly Mean Resample')
ax.set_ylabel('Pressure')
ax.legend();


# **Predicting energy consumption of appliances in future regarding to weather conditions:**
# 
# * **Change in energy consumption:**

# In[ ]:


# calculating the change in energy
data['change_in_energy'] = data['Appliances'].diff()
data.head(5)


# In[ ]:


data['Return'] = data['change_in_energy'].apply(lambda x: 1 if x>0 else -1 if x==0 else 0)
data.head(5)


# # Conclusion:
# This analysis shows that how much impact weather conditions have in our day-today energy consumption in home. Whenever the atomospheric pressure is low or high, there's a evident change in the way enegry is consumed. 
#     
