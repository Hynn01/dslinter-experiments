#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns
import time
import datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Import the dataset
covid = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')


# In[ ]:


covid.info()


# In[ ]:


print ('Last Updated: ' + str(covid.ObservationDate.max()))


# Total cases

# In[ ]:


import plotly.express as px
choro_map=px.choropleth(covid, 
                    locations="Country/Region", 
                    locationmode = "country names",
                    color="Confirmed", 
                    hover_name="Country/Region", 
                    animation_frame="ObservationDate"
                   )

choro_map.update_layout(
    title_text = 'Global Spread of Coronavirus',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
    
choro_map.show()


# Convert float to integer

# In[ ]:


covid[["Confirmed","Deaths","Recovered"]] =covid[["Confirmed","Deaths","Recovered"]].astype(int)


# In[ ]:


# Convert 'Last Update' & 'ObservationDate' column to datetime object
covid['Last Update'] = covid['Last Update'].apply(pd.to_datetime)
covid['ObservationDate'] =covid['ObservationDate'].apply(pd.to_datetime)


# In[ ]:


# Also drop the 'SNo' and the 'Province/State' columns
covid.drop(['SNo'], axis=1, inplace=True)
covid.drop(['Province/State'], axis = 1, inplace =True)


# In[ ]:


# Lets rename the columns 
covid.rename(columns={'Last Update': 'LastUpdate','Country/Region': 'Country', 'ObservationDate': 'Date'}, inplace=True)


# In[ ]:


covid['Active_case'] = covid['Confirmed'] - covid['Deaths'] - covid['Recovered']
covid.head()


# In[ ]:


# Group dataset by 'Date' with sum parameter and analyse the 'Confirmed','Deaths' values.
cases = covid.groupby('Date').sum()[['Confirmed', 'Recovered', 'Deaths']]
sns.set(style = 'whitegrid')
cases.plot(kind='line', figsize = (15,7) , marker='o',linewidth=2)
plt.bar(cases.index, cases['Confirmed'],alpha=0.3,color='g')
plt.xlabel('Days', fontsize=15)
plt.ylabel('Number of cases', fontsize=15)
plt.title('Worldwide Covid-19 cases - Confirmed, Recovered & Deaths',fontsize=20)
plt.legend()
plt.show()


# In[ ]:


df_India = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')


# In[ ]:


df_India.head()


# In[ ]:


df_age = pd.read_csv('/kaggle/input/covid19-in-india/AgeGroupDetails.csv')


# Distribution of age group

# In[ ]:


plt.figure(figsize = (12,10))
sns.barplot(data = df_age, x = 'AgeGroup', y = 'TotalCases', color = 'red')
plt.title('Distribution of age in India')
plt.xlabel('Age Group')
plt.ylabel('Total Cases')
plt.show()


# From above graph we note that most of cases occure in India age group of 20-50 years.

# Case in India

# In[ ]:


df_India['Date'] = pd.to_datetime(df_India['Date'],dayfirst=True)
df_India.at[1431,'Deaths']=119
df_India.at[1431,'State/UnionTerritory']='Madhya Pradesh'
df_India['Deaths']=df_India['Deaths'].astype(int)
df=df_India.groupby('Date').sum()
df.reset_index(inplace=True)


# In[ ]:


plt.figure(figsize= (12,8))
plt.xticks(rotation = 100 ,fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel("Dates",fontsize = 20)
plt.ylabel('Total cases',fontsize = 20)
plt.title("Total Confirmed, Active, Death in India" , fontsize = 20)
ax1 = plt.plot_date(data=df,y= 'Confirmed',x= 'Date',label = 'Confirmed',linestyle ='-',color = 'b')
ax2 = plt.plot_date(data=df,y= 'Cured',x= 'Date',label = 'Cured',linestyle ='-',color = 'g')
ax3 = plt.plot_date(data=df,y= 'Deaths',x= 'Date',label = 'Death',linestyle ='-',color = 'r')
plt.legend()
plt.show()


# Statewise cases in India

# In[ ]:


state_cases=df_India.groupby('State/UnionTerritory')['Confirmed','Deaths','Cured'].max().reset_index()
state_cases['Active'] = state_cases['Confirmed'] - abs((state_cases['Deaths']- state_cases['Cured']))
state_cases["Death Rate (per 100)"] = np.round(100*state_cases["Deaths"]/state_cases["Confirmed"],2)
state_cases["Cure Rate (per 100)"] = np.round(100*state_cases["Cured"]/state_cases["Confirmed"],2)
state_cases.sort_values('Confirmed', ascending= False).fillna(0).style.background_gradient(cmap='Reds',subset=["Confirmed"])                        .background_gradient(cmap='Blues',subset=["Deaths"])                        .background_gradient(cmap='Greens',subset=["Cured"])                        .background_gradient(cmap='Purples',subset=["Active"])                        .background_gradient(cmap='Greys',subset=["Death Rate (per 100)"])                        .background_gradient(cmap='Oranges',subset=["Cure Rate (per 100)"])


# In[ ]:


covid_India = covid [(covid['Country'] == 'India') ].reset_index(drop=True)


# In[ ]:


covid_India.info()


# In[ ]:


covid_India.head()


# In[ ]:


# Group dataset by 'Date' with sum parameter and analyse the 'Confirmed','Deaths' values.
cases = covid_India.groupby('Date').sum()[['Confirmed', 'Recovered', 'Deaths']]
sns.set(style = 'whitegrid')
cases.plot(kind='line', figsize = (15,7) , marker='o',linewidth=2)
plt.bar(cases.index, cases['Confirmed'],alpha=0.3,color='c')
plt.xlabel('Days', fontsize=15)
plt.ylabel('Number of cases', fontsize=15)
plt.title('India Covid-19 cases - Confirmed, Recovered & Deaths',fontsize=20)
plt.legend()
plt.show()


# In[ ]:


# Group dataset by 'Date' with sum parameter and analyse the Active_case values.
cases = covid_India.groupby('Date').sum()[['Active_case']]
sns.set(style = 'whitegrid')
cases.plot(kind='line', figsize = (15,7) , marker='o',linewidth=2)
plt.bar(cases.index, cases['Active_case'],alpha=0.3,color='c')
plt.xlabel('Days', fontsize=15)
plt.ylabel('Number of cases', fontsize=15)
plt.title('India Covid-19 cases - Active_case',fontsize=20)
plt.legend()
plt.show()


# Simple prediction by polynomial regression

# In[ ]:


dates = covid_India['Date'] 


# In[ ]:


days_since_1_30 = np.array([i for i in range(len(dates))]).reshape(-1, 1)


# In[ ]:


x = days_since_1_30
y = covid_India['Confirmed']


# In[ ]:


x.shape, y.shape


# In[ ]:


# Fitting Polynomial Regression to the dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)


# In[ ]:


# Visualising the Polynomial Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('India Covid-19 cases - Confirmed')
plt.xlabel('Days')
plt.ylabel('Confirmed')
plt.show()


# In[ ]:


days_in_future = 18
future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[132:150]


# In[ ]:


#Predict the future result of India
lin_reg.predict(poly_reg.fit_transform(adjusted_dates))


# In[ ]:


# Visualising the future prediction results
sns.set(style = 'whitegrid')
plt.plot(lin_reg.predict(poly_reg.fit_transform(adjusted_dates)),color='purple')
plt.xlabel('Days', fontsize=20)
plt.ylabel('Number of cases', fontsize=20)
plt.title('India Covid-19 cases - Confirmed',fontsize=10)
plt.legend()
plt.show()

