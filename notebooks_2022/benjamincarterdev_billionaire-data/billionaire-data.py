#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# 

# In[ ]:


df = pd.read_csv("/kaggle/input/the-top-billionaires/The Top Billionaires.csv")
df = df.head(5) # Lets grab just the top 5.
df


# In[ ]:


year_list = ['2018','2019','2020','2021','2022']


# In[ ]:


nice_data = {
    'NAME' : ["Elon Musk"] * 5 + ["Jeff Bezos"] * 5 + ["Bernard Arnault"] * 5 + ['Bill Gates'] * 5 + ["Warren Buffet"] * 5, 
    "Year" : year_list * 5,
    "Wealth": [19.9, 22.3, 24.6, 151.0, 219.0] + [112.0,131.0,113.0,177.0,171.0] + [72.0,76.0,76.0,150,158] + [90,96.5,98,124,129] + [84,82.5,67.5,96,118]
}
nice_df = pd.DataFrame(nice_data)


# In[ ]:


plt.rc('font', size=10)
plt.rcParams["figure.figsize"] = (15,8.5)
sns.set_theme(style="darkgrid")
sns.lineplot(data=nice_df, x = "Year", y= "Wealth", hue = 'NAME', linewidth = 5)
plt.plot
plt.ylabel('Billions')
plt.show()


# # lets Talk for a moment about what we see here.
# In 2021 the average individual income for an american was $63,214 USD. Given this average it would take a person more than 3 million years to match Elons current wealth. [source](https://www.statista.com/chart/27194/time-needed-working-at-average-annual-wage-to-match-countrys-richest-billionaire/#:~:text=To%20reach%20Musk%20levels%20of,OECD%20for%20the%20year%202020.)
# 
# Although its difficult to see here, that slight uptick in Elons wealth from 2018 to 2020, is just under 5 Billion dollars. In order for the average american to earn that sum (5 billion) they would have to work just shy of 80,000 years.
# 
# Most of us cannot wrap our heads around just how much this is. Here is a good study to help you understand the [numbers](https://ucmp.berkeley.edu/education/explorations/tours/geotime/guide/billion.html).

# ## In an Ideal world,
# If we were to chart the next 5 years of wealth for each of these oligarchs we would get a chart like this. 

# In[ ]:


year_list_2 = ['2023','2024','2025','2026','2027']
nice_data_2 = {
    'NAME' : ["Elon Musk"] * 5 + ["Jeff Bezos"] * 5 + ["Bernard Arnault"] * 5 + ['Bill Gates'] * 5 + ["Warren Buffet"] * 5, 
    "Year" : year_list_2 * 5,
    "Wealth": [151.0, 24.6, 22.3, 19.9, 11.2] + [177.0, 113.0, 131.0, 112.0, 42.0] + [150, 76.0, 76.0, 72.0, 52.5] + [124, 98,96.5, 90, 82] + [96, 67.5, 82.5, 84, 71]
}
nicer_df = pd.DataFrame(nice_data_2)
plt.rc('font', size=10)
plt.rcParams["figure.figsize"] = (15,8.5)
sns.set_theme(style="darkgrid")
sns.lineplot(data=nicer_df, x = "Year", y= "Wealth", hue = 'NAME', linewidth = 5)
plt.plot
plt.ylabel('Billions')
plt.show()

