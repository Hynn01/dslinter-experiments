#!/usr/bin/env python
# coding: utf-8

# # According to WHO, One person every 40 seconds commit suicide.
# 
# The following is an analysis of suicides across the globe. The facts driven from these analysis can be used to understand specific patterns in the dataset. They can further be used for taking relevant measures to prevent suicide rates.
# 
# Summary of this notebook/kernel:
# 
# * Suicide cases across each age group.
# * Pattern of suicide cases in the past years.
# * Comparision between male and female.
# * Pattern of suicide cases in each generation.
# * Suicide cases across each country.
# * Suicides per 100k population.
# * Relation between GDP and suicides.

# In[ ]:


#importing the necessary libraries.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#reading the dataset from its directory.

location = "../input/suicide-rates-overview-1985-to-2016/master.csv"
df = pd.read_csv(location)


# In[ ]:


#showing the first 5 rows of the dataframe. 

df.head()


# In[ ]:


#checking if the dataset requires some cleaning (fixing trash data).

df.isnull().any()


# Most of the columns/feature doesn't contain NaN (Not a number).

# In[ ]:


df = df.drop(columns="HDI for year")


# In[ ]:


df.dtypes


# In[ ]:


df.columns


# In[ ]:



df = df.drop(columns=' gdp_for_year ($) ')


# In[ ]:


#Creating a heatmap of the dataset.

plt.figure(figsize=(15,10))
cor = df.corr()
sns.heatmap(data=cor, annot=True)
plt.yticks(rotation=0)


# visualizing the relationship between the various features. 

# #  Finding out the suicide_cases in each age group:

# In[ ]:


#Here we create a separate dataframe of "age" and "suicides_no".

data_age = df.groupby("age", as_index=False).suicides_no.sum()
data_age


# In[ ]:


#Here we plot the above dataframe. 

plt.figure(figsize=(10,5))
sns.barplot(x=data_age["suicides_no"], y=data_age["age"])
plt.title("showing the relation between age group and no of suicides cases")


# The age group of (35-54) has the highest number of cases.
# 
# This might be due to (marital/economic/unemployment) reasons.

# # Finding out which year had the highest record of suicide cases:

# In[ ]:


#Firstly, we create a dataframe with the necessary features(columns) required for the operation.

data_year = df.groupby("year", as_index=False).suicides_no.sum()
data_year.head()# A total of 31 rows are there. we are showing just the first 5 rows to prevent complexcity.


# In[ ]:


# now we plot the above dataframe.

plt.figure(figsize=(10,5))
sns.barplot(x=data_year["year"], y=data_year["suicides_no"])
plt.title("showing the relation between year and no of suicides cases")
plt.xticks(rotation=45)


# Here we find that from 1998 to 2003 the suicide cases are highest.
# 
# This may be due to economic crisis or recessions.
# 
# 2016 shows a suprising drop but that is due to missing values in the dateset.

# In[ ]:


# Here we cross-checked the data of 2016.
a = df[df['year']==2016]
a.describe


# Here we find just 160 rows for the year 2016.(unlike for year 2000 with 10000 rows)
# 
# So, there are clearly some missing values.

# # Finding out which gender is more likely to commit suicide based on the historical data:

# In[ ]:


data_gender = df.groupby("sex", as_index=False).suicides_no.sum()
data_gender

From this data it is clear that males are more likely to commit suicide.
# In[ ]:


# To visualize the above data we use barplot.

plt.figure(figsize=(10,5))
sns.barplot(x=data_gender["sex"], y=data_gender["suicides_no"])
plt.title("Relation between gender and no of suicide cases")


# Male suicide cases are 3 times that of female cases.

# # Finding out the relation between generation and no of suicide cases.

# In[ ]:


# Creating a dataframe of generation and suicides_no.
data_gen = df.groupby("generation", as_index=False).suicides_no.sum()
data_gen


# In[ ]:


plt.figure(figsize=(10,5))
sns.barplot(x=data_gen["generation"], y=data_gen["suicides_no"])
plt.title("No of Suicide cases in each generation.")


# The Boomers generation commited the most suicides.(Generation range=1946-1964)
# 
# Generation Z commited the least suicide.(Generation range=1992-2010)

# # Finding out which country has the highest suicide cases.

# In[ ]:


# we create a dataframe with the necessary features.
data_country = df.groupby("country", as_index=False).suicides_no.sum()
data_country


# In[ ]:


# To make things easy, we plot the above values in a bargraph.
plt.figure(figsize=(100,100))
sns.barplot(x=data_country["suicides_no"], y=data_country["country"])
plt.title("Showing the no of suicides in every country")
plt.xticks(size = 50)
plt.yticks(size = 50)
#plt.xticks(rotation=80)


# The above representation shows that the russian federation had the highest number of suicide cases.
# Followed by US and Japan.

# # Again cross-checking the suicide rates/100k in every country.

# In[ ]:


data_rate = df.groupby("country", as_index=False)["suicides/100k pop"].mean()
data_rate


# In[ ]:


# To make things easy, we plot the above values in a bargraph.
plt.figure(figsize=(100,100))
sns.barplot(x=data_rate["suicides/100k pop"], y=data_country["country"])
plt.title("Showing the suicides rates in every country")
plt.xticks(size = 50)
plt.yticks(size = 50)


# So, according to this data analysis, we find the countries with the most suicide cases/100k population.
# 
# Lithunia seems to have the highest suicide rate. Followed by Russian Federation and Sri lanka.
# 
# Basically, this pictorial representation (ratio) shows which countries were mostly affected by suicides rate.

# 
# # Relation between GDP per capita and no of suicide cases

# In[ ]:


# creating a dataframe.

data_gdp=df.groupby("country", as_index=False)["gdp_per_capita ($)"].mean()
data_gdp


# In[ ]:


data_s=df.groupby("country", as_index=False)["suicides_no"].sum()
data_s


# In[ ]:


#data_gdp.append(data_s, ignore_index=True)
concat_df = pd.concat([data_gdp, data_s], axis=1)
concat_df


# In[ ]:


plt.figure(figsize=(10,10))
sns.regplot(x=concat_df["gdp_per_capita ($)"], y=concat_df["suicides_no"]) 
plt.title("Relation between GDP per capita and no of suicide cases")


# From this pictorial representation we find there is a slight increase in suicide cases with the increase in gdp per capita.
