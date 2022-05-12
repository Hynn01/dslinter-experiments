#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import matplotlib.pyplot as plt
from matplotlib import style
#%matplotlib inline


# Any results you write to the current directory are saved as output.


# In[ ]:


MCR_DF = pd.read_csv('../input/multipleChoiceResponses.csv',low_memory =False)
FFR_DF = pd.read_csv('../input/freeFormResponses.csv',low_memory =False)
SS_DF = pd.read_csv('../input/SurveySchema.csv',low_memory =False)
cols = ['Time from Start to Finish (seconds)','Q1']
#col = ['Time from Start to Finish (seconds)']


# In[ ]:


gender = MCR_DF.loc[1:,cols]
Female = gender[gender.Q1 == 'Female']
# print('Female count is ')
# print(Female.count()) 
# Male = gender[gender.Q1 == 'Male']
# print('Male count is ')
# print(Male.count())
# print('Total count is ')
# print(gender.count()) ##23859


# There are more number of the male kagglers than the female  kagglers who took the survey

# In[ ]:


gender['Q1'].value_counts().plot.pie(shadow=True,explode=(0,0.1,0,0))


# There are more number of kagglers from USA and India

# In[ ]:


ax = MCR_DF.loc[1:,'Q3'].value_counts().head(10).plot.barh(figsize = (12,6),fontsize =16)
plt.xlabel("Number of kaggle Survey responders",fontsize =16)
#plt.title("Kaggle Survey responders Residence country")
ax.set_title("Kaggle Survey responders Residence country",fontsize=20)


# Formal education of the kagglers survey responders

# In[ ]:


ax=MCR_DF.loc[1:,'Q4'].value_counts().plot.barh(figsize=(12,6),fontsize=16)
plt.xlabel("Number of kaggle Survey responders",fontsize=16)
#plt.title("Formal education of kaggle Survey responders")
ax.set_title("Formal education of kaggle Survey responders",fontsize=20)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


# Top  10 Undergraduate education degrees of the kagglers

# In[ ]:


ax=MCR_DF.loc[1:,'Q5'].value_counts().head(10).plot.barh(figsize=(12,6),fontsize=16)
plt.xlabel("Number of kaggle survey responders",fontsize=16)
#plt.title("Undergraduate degree of kaggle survey responders")
ax.set_title("Undergraduate degree of kaggle survey responders",fontsize=20)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


# Current Role of the kagglers who took the survey

# In[ ]:


ax=MCR_DF.loc[1:,'Q6'].value_counts().head(10).plot.barh(figsize=(12,6),fontsize=16)
plt.xlabel("Number of kaggle survey responders",fontsize=16)
#plt.title("Current role of kaggle survey responders")
ax.set_title("Current role of kaggle survey responders",fontsize=20)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


# Industry current employer

# In[ ]:


ax=MCR_DF.loc[1:,'Q7'].value_counts().head(10).plot.barh(figsize=(12,6),fontsize=16)
plt.xlabel("Number of kaggle survey responders",fontsize=20)
#plt.title("Industry current employer of kaggle survey responders")
ax.set_title("Industry current employer of kaggle survey responders",fontsize=20)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


# Experience in number of years

# In[ ]:



ax=MCR_DF.loc[1:,'Q8'].value_counts().head(10).plot.barh(figsize=(12,6),fontsize=16)
plt.xlabel("Number of kaggle survey responders",fontsize=16)
plt.ylabel("years of experience",fontsize=16)
#plt.title("Experience in number of years of kaggle survey responders")
ax.set_title("Experience in number of years of kaggle survey responders",fontsize=20)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


# Compensation

# In[ ]:


ax=MCR_DF.loc[1:,'Q9'].value_counts().head(10).plot.barh(figsize=(12,6),fontsize=16)
plt.xlabel("Number of kaggle survey responders",fontsize=16)
plt.ylabel("Compensation in USD",fontsize=16)
#plt.title("Compensation of kaggle survey reponders")
ax.set_title("Compensation of kaggle survey reponders",fontsize=20)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


# Impementing Machine learning concepts

# In[ ]:


ax=MCR_DF.loc[1:,'Q10'].value_counts().head(10).plot.barh(figsize=(12,6),fontsize=16)
plt.xlabel("Number of kaggle survey responders",fontsize=16)
#plt.title("Use of machine learning methods by kaggle survey responders")
ax.set_title("Use of machine learning methods by kaggle survey responders",fontsize=20)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


# Primary tools used to analyze
# 

# In[ ]:


ax=MCR_DF.loc[1:,'Q12_MULTIPLE_CHOICE'].value_counts().plot.barh(figsize=(12,6),fontsize=16)
plt.xlabel("Number of kaggle survey responders",fontsize=16)
#plt.title("Primary tools used to analyze data by kaggle survey responders")
ax.set_title("Primary tools used to analyze data by kaggle survey responders",fontsize=20)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


# Most  used programming languages by kaggle survey  takers 

# In[ ]:


ax=MCR_DF.loc[1:,'Q17'].value_counts().plot.bar(figsize=(12,6),fontsize=16)
plt.ylabel("Number of kaggle survey responders",fontsize=16)
plt.xlabel("Programming languages",fontsize=16)
#plt.title("Programming languages used by kaggle survey responders")
ax.set_title("Programming languages used by kaggle survey responders",fontsize=20)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


# programming language would you recommend an aspiring data scientist to learn first
# 

# In[ ]:


ax=MCR_DF.loc[1:,'Q18'].value_counts().plot.bar(figsize=(12,6),fontsize=16)
plt.ylabel("Number of kaggle survey responders",fontsize=16)
plt.xlabel("Programming language recommended by aspiring data scientist",fontsize=16)
#plt.title("Recommendation for Aspiring data scientist")
ax.set_title("Recommendation for Aspiring data scientist",fontsize=20)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


# Most used ML library  used  by the kaggle survey responders

# In[ ]:


ax=MCR_DF.loc[1:,'Q20'].value_counts().plot.bar(figsize=(12,6),fontsize=16)
plt.ylabel("Number of kaggle survey responders",fontsize=16)
plt.xlabel("ML library used by kaggle survey responders",fontsize=16)
#plt.title("ML libraries used")
ax.set_title("ML libraries used",fontsize=20)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


# ###Data visualization libraries or tools  kaggle survey responders have  used in the past 5 years

# In[ ]:


# num_dict  = {
#     'ggplot2'   :(MCR_DF['Q21_Part_1'].count()),
#     'Matplotlib':(MCR_DF['Q21_Part_2'].count()),
#     'Altair'    :(MCR_DF['Q21_Part_3'].count()),
#     'Shiny'     :(MCR_DF['Q21_Part_4'].count()),
#     'D3'        :(MCR_DF['Q21_Part_5'].count()),
#     'Plotly'    :(MCR_DF['Q21_Part_6'].count()),
#      'Bokeh'    :(MCR_DF['Q21_Part_7'].count()),
#     'Seaborn'   :(MCR_DF['Q21_Part_8'].count()),
#     'Geoplotlib':(MCR_DF['Q21_Part_9'].count()),
#     'Leaflet'   :(MCR_DF['Q21_Part_10'].count()),
#     'Lattice'   :(MCR_DF['Q21_Part_11'].count()),
#     'None'      :(MCR_DF['Q21_Part_12'].count()),
# }
# num_src = pd.Series(num_dict)

# num_src.plot.bar()
# plt.xlabel("visualization libraries used")
# plt.ylabel("Number of kaggle survey takers")
# plt.title("Data visualization libraries")


# Most used data visualization library or tool by kaggle survey responders

# In[ ]:


ax=MCR_DF.loc[1:,'Q22'].value_counts().plot.bar(figsize=(12,6),fontsize=16)
plt.ylabel("Number of kaggle survey responders",fontsize=16)
#plt.title("Data visualization library or tools used by kaggle survey responders")
plt.xlabel("Data visualization library or tool",fontsize=16)
ax.set_title("Data visualization library or tools used by kaggle survey responders",fontsize=20)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


# Kaggle survey responders consider themselves as "Data Scientist" ?

# In[ ]:


ax=MCR_DF.loc[1:,'Q26'].value_counts().plot.bar(figsize=(12,6),fontsize=16)
plt.ylabel("Number of kaggle survey responders",fontsize=16)
#plt.title("Data Scientists")
plt.xlabel("Data Scientist yes or no",fontsize=16)
ax.set_title("Data Scientists",fontsize=20)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


# Data that kaggle survey responders  currently interact with most often at work 

# In[ ]:


ax=MCR_DF.loc[1:,'Q32'].value_counts().plot.bar(figsize=(12,6),fontsize=16)
plt.ylabel("Number of kaggle survey responders",fontsize=16)
#plt.title("Type of data they interact with")
plt.xlabel("Type of data",fontsize=16)
ax.set_title("Type of data they interact with",fontsize=20)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


# Online platform kaggle survey responders spent the most amount of time

# In[ ]:


ax=MCR_DF.loc[1:,'Q37'].value_counts().plot.bar(figsize=(12,6),fontsize=16)
plt.ylabel("Number of kaggle survey responders",fontsize=16)
#plt.title("Online learning portals")
plt.xlabel("Online platforms",fontsize=16)
ax.set_title("Online learning portals",fontsize=20)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


# how many years kaggle survey responders have  used machine learning methods

# In[ ]:


ax=MCR_DF.loc[1:,'Q25'].value_counts().plot.barh(figsize=(12,6),fontsize=16)
plt.xlabel("Number of kaggle survey responders",fontsize=16)
plt.ylabel("Number of years",fontsize=16)
#plt.title("Number of years survey responders used machine learning methods")
ax.set_title("Number of years survey responders used machine learning methods",fontsize=20)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

