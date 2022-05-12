#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# Let's try to fit a logistic curve over the cases of COVID-19 in different countries. Idea: https://www.youtube.com/watch?v=Kas0tIxDvrg

# ## Data preparation
# 
# Adding the needed packages, reading the data and creating some basic aggregations.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Read dataset
all_df = pd.read_csv('/kaggle/input/coronavirus-2019ncov/covid-19-all.csv', names=['Country', 'Province', 'Lat', 'Long', 'Confirmed', 'Recovered', 'Deaths', 'Date'], header=0)
all_df.drop(['Lat', 'Long'], inplace = True, axis=1)

# Convert Date to datetime object
all_df['Date'] = all_df['Date'].apply(pd.Timestamp)
# Sum all provinces
all_df = all_df.groupby(['Country', 'Date']).sum().reset_index()


# In[ ]:


all_df.sample(5)


# ### Defining the logistic function

# In[ ]:


def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0))) + 1


# ## Fitting the logistic function on the data
# We used some initialisation for the parameters as follows:
# * L (the maximum number of confirmed cases) = 80000 taken from the China example
# * k (growth rate) = 0.2 approximated value from most of the countries
# * x0 (the day of the inflexion) = 50 approximated

# ## Top countries by number of cases

# # US

# In[ ]:


country = 'US'

df = all_df[all_df['Country'] == country]

plt.title("Number of cases in " + country + " by day")
plt.plot(df['Date'], df['Confirmed'], 'b-', label='data')


# In[ ]:


p0 = [800000, 0.25, 80] 

popt, pcov = curve_fit(logistic, range(len(df)), df['Confirmed'], p0, method = "dogbox")
print("Last day number of cases: " + str(int(df['Confirmed'][-1:])))
print("Number of cases aproximated for the next day: " + str(int(int(df['Confirmed'][-1:] + logistic(len(df) , *popt) - logistic(len(df)-1 , *popt)))))

plt.title("Number of cases (blue) and prediction (red) for " + str(len(df)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(len(df) + 1), logistic(range(len(df) + 1), *popt), 'r-', label='fit')


# In[ ]:


print("Predicted L (the maximum number of confirmed cases): " + str(int(popt[0])))
print("Predicted k (growth rate): " + str(float(popt[1])))
print("Predicted x0 (the day of the inflexion): " + str(int(popt[2])) + "")


# In[ ]:


plt.title("Number of cases (blue) and prediction (red) for " + str(int(popt[-1]*2)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(int(popt[-1]*2)), logistic(range(int(popt[-1]*2)), *popt), 'r-', label='fit')


# In[ ]:


plt.title("Log scale of cases (blue) and prediction (red) for " + str(len(df)+10) + " days in " + country)
plt.plot(range(len(df)), np.log(df["Confirmed"]), 'b-')
plt.plot(range(len(df)+10), np.log(logistic(range(len(df)+10), *popt)), 'r-')


# In[ ]:


plt.title("New cases per day in " + country)
plt.plot(range(len(df)), df["Confirmed"].diff(), 'b-')


# In[ ]:


plt.title("Log scale of cumulative cases by log scale of new cases in " + country)
plt.plot(np.log(df["Confirmed"]), np.log(df["Confirmed"].diff().replace(0, 1)), 'b-')


# # China

# In[ ]:


country = 'China'

df = all_df[all_df['Country'] == country]

plt.title("Number of cases in " + country + " by day")
plt.plot(df['Date'], df['Confirmed'], 'b-', label='data')


# In[ ]:


p0 = [80000, 0.2, 30] 

popt, pcov = curve_fit(logistic, range(len(df)), df['Confirmed'], p0, method = "dogbox")
print("Last day number of cases: " + str(int(df['Confirmed'][-1:])))
print("Number of cases aproximated for the next day: " + str(int(int(df['Confirmed'][-1:] + logistic(len(df) , *popt) - logistic(len(df)-1 , *popt)))))

plt.title("Number of cases (blue) and prediction (red) for " + str(len(df)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(len(df) + 1), logistic(range(len(df) + 1), *popt), 'r-', label='fit')


# In[ ]:


print("Predicted L (the maximum number of confirmed cases): " + str(int(popt[0])))
print("Predicted k (growth rate): " + str(float(popt[1])))
print("Predicted x0 (the day of the inflexion): " + str(int(popt[2])) + "")


# In[ ]:


plt.title("Number of cases (blue) and prediction (red) for " + str(int(popt[-1]*2)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(int(popt[-1]*2)), logistic(range(int(popt[-1]*2)), *popt), 'r-', label='fit')


# In[ ]:


plt.title("Log scale of cases (blue) and prediction (red) for " + str(len(df)+10) + " days in " + country)
plt.plot(range(len(df)), np.log(df["Confirmed"]), 'b-')
plt.plot(range(len(df)+10), np.log(logistic(range(len(df)+10), *popt)), 'r-')


# In[ ]:


plt.title("New cases per day in " + country)
plt.plot(range(len(df)), df["Confirmed"].diff(), 'b-')


# In[ ]:


plt.title("Log scale of cumulative cases by log scale of new cases in " + country)
plt.plot(np.log(df["Confirmed"]), np.log(df["Confirmed"].diff().replace(0, 1)), 'b-')


# # Italy

# In[ ]:


country = 'Italy'

df = all_df[all_df['Country'] == country]

plt.title("Number of cases in " + country + " by day")
plt.plot(df['Date'], df['Confirmed'], 'b-', label='data')


# In[ ]:


p0 = [80000, 0.2, 50] 

popt, pcov = curve_fit(logistic, range(len(df)), df['Confirmed'], p0, method = "dogbox")
print("Last day number of cases: " + str(int(df['Confirmed'][-1:])))
print("Number of cases aproximated for the next day: " + str(int(int(df['Confirmed'][-1:] + logistic(len(df) , *popt) - logistic(len(df)-1 , *popt)))))

plt.title("Number of cases (blue) and prediction (red) for " + str(len(df)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(len(df) + 1), logistic(range(len(df) + 1), *popt), 'r-', label='fit')


# In[ ]:


print("Predicted L (the maximum number of confirmed cases): " + str(int(popt[0])))
print("Predicted k (growth rate): " + str(float(popt[1])))
print("Predicted x0 (the day of the inflexion): " + str(int(popt[2])) + "")


# In[ ]:


plt.title("Number of cases (blue) and prediction (red) for " + str(int(popt[-1]*2)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(int(popt[-1]*2)), logistic(range(int(popt[-1]*2)), *popt), 'r-', label='fit')


# In[ ]:


plt.title("Log scale of cases (blue) and prediction (red) for " + str(len(df)+10) + " days in " + country)
plt.plot(range(len(df)), np.log(df["Confirmed"]), 'b-')
plt.plot(range(len(df)+10), np.log(logistic(range(len(df)+10), *popt)), 'r-')


# In[ ]:


plt.title("New cases per day in " + country)
plt.plot(range(len(df)), df["Confirmed"].diff(), 'b-')


# In[ ]:


plt.title("Log scale of cumulative cases by log scale of new cases in " + country)
plt.plot(np.log(df["Confirmed"]), np.log(df["Confirmed"].diff().replace(0, 1)), 'b-')


# # Spain

# In[ ]:


country = 'Spain'

df = all_df[all_df['Country'] == country]

plt.title("Number of cases in " + country + " by day")
plt.plot(df['Date'], df['Confirmed'], 'b-', label='data')


# In[ ]:


p0 = [80000, 0.2, 50] 

popt, pcov = curve_fit(logistic, range(len(df)), df['Confirmed'], p0, method = "dogbox")
print("Last day number of cases: " + str(int(df['Confirmed'][-1:])))
print("Number of cases aproximated for the next day: " + str(int(int(df['Confirmed'][-1:] + logistic(len(df) , *popt) - logistic(len(df)-1 , *popt)))))

plt.title("Number of cases (blue) and prediction (red) for " + str(len(df)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(len(df) + 1), logistic(range(len(df) + 1), *popt), 'r-', label='fit')


# In[ ]:


print("Predicted L (the maximum number of confirmed cases): " + str(int(popt[0])))
print("Predicted k (growth rate): " + str(float(popt[1])))
print("Predicted x0 (the day of the inflexion): " + str(int(popt[2])) + "")


# In[ ]:


plt.title("Number of cases (blue) and prediction (red) for " + str(int(popt[-1]*2)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(int(popt[-1]*2)), logistic(range(int(popt[-1]*2)), *popt), 'r-', label='fit')


# In[ ]:


plt.title("Log scale of cases (blue) and prediction (red) for " + str(len(df)+10) + " days in " + country)
plt.plot(range(len(df)), np.log(df["Confirmed"]), 'b-')
plt.plot(range(len(df)+10), np.log(logistic(range(len(df)+10), *popt)), 'r-')


# In[ ]:


plt.title("New cases per day in " + country)
plt.plot(range(len(df)), df["Confirmed"].diff(), 'b-')


# In[ ]:


plt.title("Log scale of cumulative cases by log scale of new cases in " + country)
plt.plot(np.log(df["Confirmed"]), np.log(df["Confirmed"].diff().replace(0, 1)), 'b-')


# # Germany

# In[ ]:


country = 'Germany'

df = all_df[all_df['Country'] == country]

plt.title("Number of cases in " + country + " by day")
plt.plot(df['Date'], df['Confirmed'], 'b-', label='data')


# In[ ]:


p0 = [80000, 0.2, 50] 

popt, pcov = curve_fit(logistic, range(len(df)), df['Confirmed'], p0, method = "dogbox")
print("Last day number of cases: " + str(int(df['Confirmed'][-1:])))
print("Number of cases aproximated for the next day: " + str(int(int(df['Confirmed'][-1:] + logistic(len(df) , *popt) - logistic(len(df)-1 , *popt)))))

plt.title("Number of cases (blue) and prediction (red) for " + str(len(df)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(len(df) + 1), logistic(range(len(df) + 1), *popt), 'r-', label='fit')


# In[ ]:


print("Predicted L (the maximum number of confirmed cases): " + str(int(popt[0])))
print("Predicted k (growth rate): " + str(float(popt[1])))
print("Predicted x0 (the day of the inflexion): " + str(int(popt[2])) + "")


# In[ ]:


plt.title("Number of cases (blue) and prediction (red) for " + str(int(popt[-1]*2)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(int(popt[-1]*2)), logistic(range(int(popt[-1]*2)), *popt), 'r-', label='fit')


# In[ ]:


plt.title("Log scale of cases (blue) and prediction (red) for " + str(len(df)+10) + " days in " + country)
plt.plot(range(len(df)), np.log(df["Confirmed"]), 'b-')
plt.plot(range(len(df)+10), np.log(logistic(range(len(df)+10), *popt)), 'r-')


# In[ ]:


plt.title("New cases per day in " + country)
plt.plot(range(len(df)), df["Confirmed"].diff(), 'b-')


# In[ ]:


plt.title("Log scale of cumulative cases by log scale of new cases in " + country)
plt.plot(np.log(df["Confirmed"]), np.log(df["Confirmed"].diff().replace(0, 1)), 'b-')


# # Iran

# In[ ]:


country = 'Iran'

df = all_df[all_df['Country'] == country]

plt.title("Number of cases in " + country + " by day")
plt.plot(df['Date'], df['Confirmed'], 'b-', label='data')


# In[ ]:


p0 = [800000, 0.2, 50] 

popt, pcov = curve_fit(logistic, range(len(df)), df['Confirmed'], p0, method = "dogbox")
print("Last day number of cases: " + str(int(df['Confirmed'][-1:])))
print("Number of cases aproximated for the next day: " + str(int(int(df['Confirmed'][-1:] + logistic(len(df) , *popt) - logistic(len(df)-1 , *popt)))))

plt.title("Number of cases (blue) and prediction (red) for " + str(len(df)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(len(df) + 1), logistic(range(len(df) + 1), *popt), 'r-', label='fit')


# In[ ]:


print("Predicted L (the maximum number of confirmed cases): " + str(int(popt[0])))
print("Predicted k (growth rate): " + str(float(popt[1])))
print("Predicted x0 (the day of the inflexion): " + str(int(popt[2])) + "")


# In[ ]:


plt.title("Number of cases (blue) and prediction (red) for " + str(int(popt[-1]*2)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(int(popt[-1]*2)), logistic(range(int(popt[-1]*2)), *popt), 'r-', label='fit')


# In[ ]:


plt.title("Log scale of cases (blue) and prediction (red) for " + str(len(df)+10) + " days in " + country)
plt.plot(range(len(df)), np.log(df["Confirmed"]), 'b-')
plt.plot(range(len(df)+10), np.log(logistic(range(len(df)+10), *popt)), 'r-')


# In[ ]:


plt.title("New cases per day in " + country)
plt.plot(range(len(df)), df["Confirmed"].diff(), 'b-')


# In[ ]:


plt.title("Log scale of cumulative cases by log scale of new cases in " + country)
plt.plot(np.log(df["Confirmed"]), np.log(df["Confirmed"].diff().replace(0, 1)), 'b-')


# # France

# In[ ]:


country = 'France'

df = all_df[all_df['Country'] == country]

plt.title("Number of cases in " + country + " by day")
plt.plot(df['Date'], df['Confirmed'], 'b-', label='data')


# In[ ]:


p0 = [800000, 0.2, 70] 

popt, pcov = curve_fit(logistic, range(len(df)), df['Confirmed'], p0, method = "dogbox")
print("Last day number of cases: " + str(int(df['Confirmed'][-1:])))
print("Number of cases aproximated for the next day: " + str(int(int(df['Confirmed'][-1:] + logistic(len(df) , *popt) - logistic(len(df)-1 , *popt)))))

plt.title("Number of cases (blue) and prediction (red) for " + str(len(df)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(len(df) + 1), logistic(range(len(df) + 1), *popt), 'r-', label='fit')


# In[ ]:


print("Predicted L (the maximum number of confirmed cases): " + str(int(popt[0])))
print("Predicted k (growth rate): " + str(float(popt[1])))
print("Predicted x0 (the day of the inflexion): " + str(int(popt[2])) + "")


# In[ ]:


plt.title("Number of cases (blue) and prediction (red) for " + str(int(popt[-1]*2)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(int(popt[-1]*2)), logistic(range(int(popt[-1]*2)), *popt), 'r-', label='fit')


# In[ ]:


plt.title("Log scale of cases (blue) and prediction (red) for " + str(len(df)+10) + " days in " + country)
plt.plot(range(len(df)), np.log(df["Confirmed"]), 'b-')
plt.plot(range(len(df)+10), np.log(logistic(range(len(df)+10), *popt)), 'r-')


# In[ ]:


plt.title("New cases per day in " + country)
plt.plot(range(len(df)), df["Confirmed"].diff(), 'b-')


# In[ ]:


plt.title("Log scale of cumulative cases by log scale of new cases in " + country)
plt.plot(np.log(df["Confirmed"]), np.log(df["Confirmed"].diff().replace(0, 1)), 'b-')


# # South Korea

# In[ ]:


country = 'South Korea'

df = all_df[all_df['Country'] == country]

plt.title("Number of cases in " + country + " by day")
plt.plot(df['Date'], df['Confirmed'], 'b-', label='data')


# In[ ]:


p0 = [80000, 0.2, 50] 

popt, pcov = curve_fit(logistic, range(len(df)), df['Confirmed'], p0, method = "dogbox")
print("Last day number of cases: " + str(int(df['Confirmed'][-1:])))
print("Number of cases aproximated for the next day: " + str(int(int(df['Confirmed'][-1:] + logistic(len(df) , *popt) - logistic(len(df)-1 , *popt)))))

plt.title("Number of cases (blue) and prediction (red) for " + str(len(df)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(len(df) + 1), logistic(range(len(df) + 1), *popt), 'r-', label='fit')


# In[ ]:


print("Predicted L (the maximum number of confirmed cases): " + str(int(popt[0])))
print("Predicted k (growth rate): " + str(float(popt[1])))
print("Predicted x0 (the day of the inflexion): " + str(int(popt[2])) + "")


# In[ ]:


plt.title("Number of cases (blue) and prediction (red) for " + str(int(popt[-1]*2)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(int(popt[-1]*2)), logistic(range(int(popt[-1]*2)), *popt), 'r-', label='fit')


# In[ ]:


plt.title("Log scale of cases (blue) and prediction (red) for " + str(len(df)+10) + " days in " + country)
plt.plot(range(len(df)), np.log(df["Confirmed"]), 'b-')
plt.plot(range(len(df)+10), np.log(logistic(range(len(df)+10), *popt)), 'r-')


# In[ ]:


plt.title("New cases per day in " + country)
plt.plot(range(len(df)), df["Confirmed"].diff(), 'b-')


# In[ ]:


plt.title("Log scale of cumulative cases by log scale of new cases in " + country)
plt.plot(np.log(df["Confirmed"]), np.log(df["Confirmed"].diff().replace(0, 1)), 'b-')


# # United Kingdom

# In[ ]:


country = 'United Kingdom'

df = all_df[all_df['Country'] == country]

plt.title("Number of cases in " + country + " by day")
plt.plot(df['Date'], df['Confirmed'], 'b-', label='data')


# In[ ]:


p0 = [800000, 0.2, 70] 

popt, pcov = curve_fit(logistic, range(len(df)), df['Confirmed'], p0, method = "dogbox")
print("Last day number of cases: " + str(int(df['Confirmed'][-1:])))
print("Number of cases aproximated for the next day: " + str(int(int(df['Confirmed'][-1:] + logistic(len(df) , *popt) - logistic(len(df)-1 , *popt)))))

plt.title("Number of cases (blue) and prediction (red) for " + str(len(df)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(len(df) + 1), logistic(range(len(df) + 1), *popt), 'r-', label='fit')


# In[ ]:


print("Predicted L (the maximum number of confirmed cases): " + str(int(popt[0])))
print("Predicted k (growth rate): " + str(float(popt[1])))
print("Predicted x0 (the day of the inflexion): " + str(int(popt[2])) + "")


# In[ ]:


plt.title("Number of cases (blue) and prediction (red) for " + str(int(popt[-1]*2)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(int(popt[-1]*2)), logistic(range(int(popt[-1]*2)), *popt), 'r-', label='fit')


# In[ ]:


plt.title("Log scale of cases (blue) and prediction (red) for " + str(len(df)+10) + " days in " + country)
plt.plot(range(len(df)), np.log(df["Confirmed"]), 'b-')
plt.plot(range(len(df)+10), np.log(logistic(range(len(df)+10), *popt)), 'r-')


# In[ ]:


plt.title("New cases per day in " + country)
plt.plot(range(len(df)), df["Confirmed"].diff(), 'b-')


# In[ ]:


plt.title("Log scale of cumulative cases by log scale of new cases in " + country)
plt.plot(np.log(df["Confirmed"]), np.log(df["Confirmed"].diff().replace(0, 1)), 'b-')


# # Russia

# In[ ]:


country = 'Russia'

df = all_df[all_df['Country'] == country]

plt.title("Number of cases in " + country + " by day")
plt.plot(df['Date'], df['Confirmed'], 'b-', label='data')


# In[ ]:


p0 = [80000, 0.2, 70] 

popt, pcov = curve_fit(logistic, range(len(df)), df['Confirmed'], p0, method = "dogbox")
print("Last day number of cases: " + str(int(df['Confirmed'][-1:])))
print("Number of cases aproximated for the next day: " + str(int(int(df['Confirmed'][-1:] + logistic(len(df) , *popt) - logistic(len(df)-1 , *popt)))))

plt.title("Number of cases (blue) and prediction (red) for " + str(len(df)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(len(df) + 1), logistic(range(len(df) + 1), *popt), 'r-', label='fit')


# In[ ]:


print("Predicted L (the maximum number of confirmed cases): " + str(int(popt[0])))
print("Predicted k (growth rate): " + str(float(popt[1])))
print("Predicted x0 (the day of the inflexion): " + str(int(popt[2])) + "")


# In[ ]:


plt.title("Number of cases (blue) and prediction (red) for " + str(int(popt[-1]*2)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(int(popt[-1]*2)), logistic(range(int(popt[-1]*2)), *popt), 'r-', label='fit')


# In[ ]:


plt.title("Log scale of cases (blue) and prediction (red) for " + str(len(df)+10) + " days in " + country)
plt.plot(range(len(df)), np.log(df["Confirmed"]), 'b-')
plt.plot(range(len(df)+10), np.log(logistic(range(len(df)+10), *popt)), 'r-')


# In[ ]:


plt.title("New cases per day in " + country)
plt.plot(range(len(df)), df["Confirmed"].diff(), 'b-')


# In[ ]:


plt.title("Log scale of cumulative cases by log scale of new cases in " + country)
plt.plot(np.log(df["Confirmed"]), np.log(df["Confirmed"].diff().replace(0, 1)), 'b-')


# # Romania

# In[ ]:


country = 'Romania'

df = all_df[all_df['Country'] == country]

plt.title("Number of cases in " + country + " by day")
plt.plot(df['Date'], df['Confirmed'], 'b-', label='data')


# In[ ]:


p0 = [80000, 0.2, 50] 

popt, pcov = curve_fit(logistic, range(len(df)), df['Confirmed'], p0, method = "dogbox")
print("Last day number of cases: " + str(int(df['Confirmed'][-1:])))
print("Number of cases aproximated for the next day: " + str(int(int(df['Confirmed'][-1:] + logistic(len(df) , *popt) - logistic(len(df)-1 , *popt)))))

plt.title("Number of cases (blue) and prediction (red) for " + str(len(df)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(len(df) + 1), logistic(range(len(df) + 1), *popt), 'r-', label='fit')


# In[ ]:


print("Predicted L (the maximum number of confirmed cases): " + str(int(popt[0])))
print("Predicted k (growth rate): " + str(float(popt[1])))
print("Predicted x0 (the day of the inflexion): " + str(int(popt[2])) + "")


# In[ ]:


plt.title("Number of cases (blue) and prediction (red) for " + str(int(popt[-1]*2)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(int(popt[-1]*2)), logistic(range(int(popt[-1]*2)), *popt), 'r-', label='fit')


# In[ ]:


plt.title("Log scale of cases (blue) and prediction (red) for " + str(len(df)+10) + " days in " + country)
plt.plot(range(len(df)), np.log(df["Confirmed"]), 'b-')
plt.plot(range(len(df)+10), np.log(logistic(range(len(df)+10), *popt)), 'r-')


# In[ ]:


plt.title("New cases per day in " + country)
plt.plot(range(len(df)), df["Confirmed"].diff(), 'b-')


# In[ ]:


plt.title("Log scale of cumulative cases by log scale of new cases in " + country)
plt.plot(np.log(df["Confirmed"]), np.log(df["Confirmed"].diff().replace(0, 1)), 'b-')


# # Moldova

# In[ ]:


country = 'Moldova'

df = all_df[all_df['Country'] == country]

plt.title("Number of cases in " + country + " by day")
plt.plot(df['Date'], df['Confirmed'], 'b-', label='data')


# In[ ]:


p0 = [80000, 0.2, 50] 

popt, pcov = curve_fit(logistic, range(len(df)), df['Confirmed'], p0, method = "dogbox")
print("Last day number of cases: " + str(int(df['Confirmed'][-1:])))
print("Number of cases aproximated for the next day: " + str(int(int(df['Confirmed'][-1:] + logistic(len(df) , *popt) - logistic(len(df)-1 , *popt)))))

plt.title("Number of cases (blue) and prediction (red) for " + str(len(df)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(len(df) + 1), logistic(range(len(df) + 1), *popt), 'r-', label='fit')


# In[ ]:


print("Predicted L (the maximum number of confirmed cases): " + str(int(popt[0])))
print("Predicted k (growth rate): " + str(float(popt[1])))
print("Predicted x0 (the day of the inflexion): " + str(int(popt[2])) + "")


# In[ ]:


plt.title("Number of cases (blue) and prediction (red) for " + str(int(popt[-1]*2)) + " days for " + country)
plt.plot(range(len(df)), df['Confirmed'], 'b-', label='data')
plt.plot(range(int(popt[-1]*2)), logistic(range(int(popt[-1]*2)), *popt), 'r-', label='fit')


# In[ ]:


plt.title("Log scale of cases (blue) and prediction (red) for " + str(len(df)+10) + " days in " + country)
plt.plot(range(len(df)), np.log(df["Confirmed"]), 'b-')
plt.plot(range(len(df)+10), np.log(logistic(range(len(df)+10), *popt)), 'r-')


# In[ ]:


plt.title("New cases per day in " + country)
plt.plot(range(len(df)), df["Confirmed"].diff(), 'b-')


# In[ ]:


plt.title("Log scale of cumulative cases by log scale of new cases in " + country)
plt.plot(np.log(df["Confirmed"]), np.log(df["Confirmed"].diff().replace(0, 1)), 'b-')

