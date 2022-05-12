#!/usr/bin/env python
# coding: utf-8

# # Comments
# Hi Mercy, nice job working on this project. 
# - My suggestions are related to formulating questions that will lead to deeper data exploration and more informative insights. 
# It would have been a lot more insightful if you compared trends across various countries, e.g finding the countries that have the most cases, or deaths, or recoveries.
# You could also calculate the death rate or recovery rates for the different countries. 
# 
# ### Additionally, if you notice how the covid data is being structured, you'll see that more records are being added on a daily basis. So one way to work with only the most recent data is by doing this:
# 
# ```
# recent_data = df[df['ObservationDate'] == df['ObservationDate'].max()]
# ```
# 
# i.e creating a new data frame that holds the most recent data (the `.max()` selects the most recent data which also happens to be the maximum date.
# 
# Doing this helps you compare the current status of the cases across different countries and regions.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
cd = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')


# In[ ]:


cd = cd.rename(columns={'Country/Region': 'Country', 'ObservationDate':'Date'})


# In[ ]:


cd.head()


# In[ ]:


cd.tail()


# In[ ]:


cd.loc[:, ['Date', 'Country/Region', 'Confrimed', 'Deaths', 'Recovered']]


# In[ ]:


#Exploring data on confirmed cases


# In[ ]:


#Scatter diagram showing the distribution of confirmed cases
plt.scatter(cd['Confirmed'] , cd['Date'])
plt.xlabel('Date')
plt.ylabel('Confrimed cases')
plt.title('Distribution of confirmed cases')
plt.show


# In[ ]:


#Plotting a scatter diagram to show the size of the growth of each confirmed case
plt.scatter(cd['Confirmed'] , cd['Date'] , s = cd['Confirmed'] / 100)
plt.xscale('log') 
plt.xlabel('Date')
plt.ylabel('Confrimed cases')
plt.title('Distribution of confirmed cases')
plt.xticks([1000, 10000, 100000],['1k', '10k', '100k'])
plt.show


# In[ ]:


#Determining the mean value of the sum of confirmed cases at the time
cd['Confirmed'].mean()


# In[ ]:


#Creating a variable that carried the mean value of confirmed cases
top_confirmed_cases = cd['Confirmed'] > 3133.506542958275


# In[ ]:


#All data entries above the mean value of confirmed cases
cd[top_confirmed_cases]


# In[ ]:


#Creating a new column that contains the mean value of confirmed cases
cd['Mean_C'] = np.mean(cd['Confirmed'])
cd.groupby('Date')
cd_mean_c = cd.describe()['Confirmed'][1]
cd1 = cd[ (cd['Deaths'] < cd_mean_c * 10000) ]
#Plotting a graph showing the distribution of confirmed cases above the mean value
cd1['Confirmed'].plot()
plt.title('Distribution of confirmed cases above the mean value')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.show()


# In[ ]:


#Exploring data on recovery cases


# In[ ]:


#A scatter diagram showing the distribution of recovered cases across a period of time
plt.scatter(cd['Recovered'] , cd['Date'])
plt.xlabel('Date')
plt.ylabel('Recovered cases')
plt.title('Distribution of Recovered cases')
plt.show


# In[ ]:


#Determining the mean value of the recovered cases at the time
cd['Recovered'].mean()


# In[ ]:


#Creating a variable that carried the mean value of recovered cases
top_recovered_cases = cd['Recovered'] > 795.2522256784297


# In[ ]:


#printing out the dataframe containing all cases above the mean value of recovered cases
cd[top_recovered_cases]


# In[ ]:


#Creating a new column that contains the mean value of recovered cases
cd['Mean_R'] = np.mean(cd['Recovered'])
cd.groupby('Date')
cd_mean_r = cd.describe()['Recovered'][1]
cd2 = cd[ (cd['Deaths'] < cd_mean_r * 10000) ]
#Plotting a graph showing the distribution of recovered cases above the mean value
cd2['Recovered'].plot()
plt.xlabel('Date')
plt.ylabel('Recovered Cases')
plt.title('Distribution of confirmed cases')
plt.show()


# In[ ]:


#Exploring data on the death cases


# In[ ]:


#Determining the mean value of all death cases
cd['Deaths'].mean()


# In[ ]:


top_death_cases = cd['Deaths'] > 196.18834695491697


# In[ ]:


cd.loc[cd['Deaths'] > 196.18834695491697]


# In[ ]:


#Creating a new column that contains the mean value of confirmed cases
cd['Mean_d'] = np.mean(cd['Deaths'])
cd.groupby('Date')
cd_mean_d = cd.describe()['Deaths'][1]
cd3 = cd[ (cd['Deaths'] < cd_mean_d * 10000) ]
#Plotting a graph showing the distribution of death cases above the its mean value
cd3['Deaths'].plot()
plt.xlabel('Date')
plt.ylabel('Recovered cases')
plt.title('Distribution of Recovered Cases above mean valuse')
plt.show()


# In[ ]:


#A scatter daigram d=showing the distribution of death cases across time
plt.scatter(cd['Date'] , cd['Deaths'] * 10000)
plt.xlabel('Date')
plt.ylabel('Death cases')
plt.title('Distribution of Death cases')
plt.show


# In[ ]:


#Exploring data on Mainland China 
cty = cd.groupby('Country') 
print(type(cty))
# Finding the values contained in the "Mainland China" group 
chn = cty.get_group('Mainland China') 
#Generating the standard deviation of the confirm cases in china 
chn['STD_CHN_c'] = chn.std(1, 5)
#Generating the standard deviation of the death cases in china 
chn['STD_CHN_d'] = chn.std(1, 6)
#Generating the standard deviation of the death cases in china 
chn['STD_CHN_r'] = chn.std(1, 7)
chn.head()
#graph showing the standard deviation of all columns in china
chn.plot.line('STD_CHN_c')
plt.xlabel('Date')
plt.ylabel('No. of Cases')
plt.title('Distribution of the change in Standard deviation for all cases in china')
plt.show()

#graph showing the distribution of the standard deviation of confirmed cases in China
chn['STD_CHN_c'].plot()
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.title('Distribution of the change in Standard deviation for Confirmed Cases in China')
plt.show()

#graph shoing the distribution of the standard deviation of recovered cases in China
chn['STD_CHN_r'].plot()
plt.xlabel('Date')
plt.ylabel('Recovered Cases')
plt.title('Distribution of the change in Standard deviation for Recovered Cases in China')
plt.show()

#graph shoing the distribution of the standard deviation of death cases in China
chn['STD_CHN_d'].plot()
plt.xlabel('Date')
plt.ylabel('Death Cases')
plt.title('Distribution of the change in Standard deviation for Death Cases in China')
plt.show()


# In[ ]:


#Exploring data on Nigeria 
cty = cd.groupby('Country') 
# Finding the values contained in the "Nigeria" group 
chn = cty.get_group('Nigeria') 
#Generating the standard deviation of the confirm cases in Nigeria 
chn['STD_NIG_c'] = chn.std(1, 5)
#Generating the standard deviation of the death cases in Nigeria 
chn['STD_NIG_d'] = chn.std(1, 6)
#Generating the standard deviation of the death cases in Nigeria 
chn['STD_NIG_r'] = chn.std(1, 7)
chn.head()
#graph showing the standard deviation of confirmed cases in Nigeria
chn.plot.line('STD_NIG_c')
plt.xlabel('Date')
plt.ylabel('No. of Cases')
plt.title('Distribution of the change in Standard deviation for all cases in Nigeria')
plt.show()

#graph showing the distribution of the standard deviation of confirmed cases in Nigeria
chn['STD_NIG_c'].plot()
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.title('Distribution of the change in Standard deviation for Confirmed Cases in Nigeria')
plt.show()

#graph showing the distribution of the standard deviation of Recovered cases in Nigeria
chn['STD_NIG_r'].plot()
plt.xlabel('Date')
plt.ylabel('Recovered  Cases')
plt.title('Distribution of the change in Standard deviation for Confirmed Cases in Nigeria')
plt.show()

#graph showing the distribution of the standard deviation of death cases in Nigeria
chn['STD_NIG_d'].plot()
plt.xlabel('Date')
plt.xlabel('Death Cases')
plt.title('Distribution of the change in Standard deviation for Death Cases in Nigeria')
plt.show()

