#!/usr/bin/env python
# coding: utf-8

# # Throughout history, nothing has killed more human beings than infectious disease. Covid-19 is climbing up the same ladder.
# 
# The following is an analysis of covid-19. This analysis will address some of the questions which everyone should be aware of.
# 
# The summary of this notebook/kernel:
# 
# * A graph showing confirmed cases with time across all the countries.
# * Countries with maximum deaths.
# * Countries which are successfully fighting against covid-19.(comparing the ratios of confirmed cases by recovered cases.)
# * Pattern of rising cases in India.
# * Comparision of confirmed, recovered and death curves.

# In[ ]:


# Importing the necessary libraries.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[ ]:


# reading the data from the repository. 

location = "../input/covid19dataset/COVID_Data_Basic.csv"
data = pd.read_csv(location)
data


# # Data Cleaning: 

# In[ ]:


# Looking at the key features of our dataset.

data.describe()


# In[ ]:


#Checking the datatypes of each column.

data.dtypes


# Since the "Date" is in object datatype, we need to convert it. 

# In[ ]:


# Converting the datatype of Date to datatime format. 

data["Date"] = pd.to_datetime(data["Date"])


# In[ ]:


# Cross-checking whether we got the required datatype.

data.dtypes


# In[ ]:


# Checking for null values.

data.isnull().any()


# In[ ]:


# Checking for india's data. 

data["Country"].unique()


# Yes, India is there. Few countries seem to be missing from the dataset. So, it's good to cross-check. 

# We will do country specific analysis on India.
# 
# But before that we will do some global analysis.

# # Global_analysis on confirmed cases with time. 

# In[ ]:


# Creating a dataframe with required features.

data_date = data.groupby("Date", as_index=False).Confirmed.sum()
data_date


# In[ ]:


# Looking at the datatypes.(in case they need to be astyped)

data_date.dtypes


# In[ ]:


# Creating a line-graph with above data.

plt.figure(figsize=(10,10))
sns.lineplot(x=data_date["Date"], y=data_date["Confirmed"])
plt.title("No of confirmed cases with time.")
plt.xticks(rotation=45)


# We see a steep rise in no of cases with time across all countries.

# # Countries with maximum death cases.

# In[ ]:


# Creating a dataframe with the neccesary features/columns.

data_death = data.groupby("Country", as_index=False).Death.max()
data_death


# In[ ]:


# plotting a bargraph with the above dataframe. 

plt.figure(figsize=(15,40))
sns.barplot(x=data_death["Death"], y=data_death["Country"])
plt.yticks(size=10)
plt.xticks(size=20)


# US is seen as the country with the highest death cases. Followed by Italy and Spain.

# To make thing a bit easier to visualize, we consider the top 10 affected countries and plot the values ina similar graph.

# In[ ]:


# Creating a dataframe of top 10 countries.

death = data_death.sort_values(by="Death", ascending=False)
death_top10 = death.head(10)
death_top10


# So, here we get the top countries with the most death cases.

# In[ ]:


# plotting a bargraph with the above dataframe.

plt.figure(figsize=(10,5))
sns.barplot(x=death_top10["Death"], y=death_top10["Country"])
plt.title("top countries with the most death cases")


# Now, things are more clean and clear.

# #  Countries who are successfully fighting against covid-19:

# We need to create a dataframe with the necessary features/columns.

# In[ ]:


# Creating a dataframe of countries vs recovered cases.

data_recover = data.groupby("Country", as_index=False).Recovered.max()
data_recover


# In[ ]:


# Creating a dataframe of countries vs confirmed cases.

data_confirmed = data.groupby("Country", as_index=False).Confirmed.max()
data_confirmed


# To know which country is performing better, we need to compare their ratios of deaths is to recoveries.

# In[ ]:


# Merging the above 2 Dataframes.
data_fight = pd.merge(data_recover, data_confirmed, on='Country')
data_fight


# In[ ]:


# Creating the ratios by dividing confirmed cases/ recovered cases.

ratio = data_fight["Confirmed"]/data_fight["Recovered"]
ratio


# In[ ]:


# Adding the above ration to the existing dataframe for easier evaluation.

data_fight["Ratio"] = ratio
data_fight


# So, this ratio tells us which country is fighting covid-19 successfully.
# 
# Lesser the ratio the more successfull they have been.

# In[ ]:


# checking the datatype of ratio column. 

data_fight.dtypes


# In[ ]:


# plotting a bargraph with the above dataframe.

plt.figure(figsize=(15,40))
sns.barplot(x=data_fight["Ratio"], y=data_fight["Country"])
plt.yticks(size=10)
plt.xticks(size=20)


# In[ ]:


# sorting the dataframe with the least ratios.

data_top10 = data_fight.sort_values(by="Ratio", ascending=True)
data_top10.head(10)


# In[ ]:


# Here, we need to drop "Diamond Princess" beacause that's a cruising ship and and not a country. 

data_top10 = data_top10.drop(index=[49])
data_top10 = data_top10.head(10)


# In[ ]:


plt.figure(figsize=(10,10))
sns.barplot(x=data_top10["Country"], y=data_top10["Ratio"])
plt.title("Countries vs ratio of confirmed cases to recoveries")


# These countries have maximium of their patients recovered.
# 
# Saint Lucia leading the list. Followed by China and Cambodia. 

# # Analysing India's data: 

# In[ ]:


# Extracting country specific data.

data_india = data[data['Country']=="India"]
data_india


# In[ ]:


# Confirmed cases in India in each day from 2019-12-31 to 2020-4-22.

india_confirmed = data_india.groupby("Date", as_index=False).Confirmed.sum()
india_confirmed


# In[ ]:


# Plotting the above dataframe in a line graph.

plt.figure(figsize=(10,10))
sns.lineplot(x=india_confirmed["Date"], y=india_confirmed["Confirmed"])
plt.xticks(rotation=45)
plt.title("India's rising covid-19 cases")


# In[ ]:


# # Recovered cases in India in each day from 2019-12-31 to 2020-4-22.

india_recover = data_india.groupby("Date", as_index=False).Recovered.sum()
india_recover


# In[ ]:


india_death = data_india.groupby("Date", as_index=False).Death.sum()
india_death


# In[ ]:


# plotting the above dataframes in a line graph. 

plt.figure(figsize=(10,10))
sns.lineplot(x=india_recover["Date"], y=india_recover["Recovered"], label="recovered cases")
sns.lineplot(x=india_confirmed["Date"], y=india_confirmed["Confirmed"], label="confirmed cases")
sns.lineplot(x=india_death["Date"], y=india_death["Death"], label="Death cases")
plt.xticks(rotation=45)
plt.ylabel("no of cases")


# We see a steep rise in confirmed cases from 2020-03-15.
# 
# It seems like recovery rate is low but we need to consider the fact that a patient needs time to recover.
# 
# Also, the death rate is a lot low than the recovery rate.
