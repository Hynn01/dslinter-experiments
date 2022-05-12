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
import plotly.express as px
import plotly.graph_objs as go

import plotly
plotly.offline.init_notebook_mode(connected=True)

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <center style="font-family:verdana;"><h1 style="font-size:200%; padding: 10px; background: #E9967A;"><b style="color:white;">Homeless Deaths Capture-Recapture Technique</b></h1></center>
# 
# Measuring Homeless People Deaths
# 
# "Homelessness is an important problem affecting some of the most vulnerable people in society, but which is difficult to measure as well as to solve."
# 
# "A statistical modelling technique known as capture-recapture to estimate a total figure, which allows for the likelihood of more deaths of homeless people being present in the data but not identified."
# 
# Five search strategies were used:
# 
# Search criteria one
# 
# "The recorded place of residence contained any of a list of text expressions such as “no fixed abode”, “homeless” and “night shelter” or the name or address of a known homeless hostel or project. An extensive list of addresses was compiled from publicly available sources. While this list was necessarily incomplete, the statistical model was found to be robust against even a substantial number of omissions." 
# 
# https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/bulletins/deathsofhomelesspeopleinenglandandwales/2019registrations#causes-of-death-among-homeless-people
# 
# 
# International Classification of Diseases ICD 10 https://www.cdc.gov/nchs/icd/icd10.htm

# ![](https://www.researchgate.net/profile/Lise-Grout/publication/263709081/figure/fig1/AS:271350283632645@1441706357059/The-capture-recapture-method-applied-to-the-CMDR-and-CepiDc-databases.png)https://www.researchgate.net/figure/The-capture-recapture-method-applied-to-the-CMDR-and-CepiDc-databases_fig1_263709081

# In[ ]:


df = pd.read_csv("/kaggle/input/cusersmarildownloadshomelecsv/homele.csv", delimiter=';', encoding='utf8')
pd.set_option('display.max_columns', None)
df.tail()


# <h1><span class="label label-default" style="background-color:#E9967A;border-radius:100px 100px; font-weight: bold; font-family:Garamond; font-size:20px; color:white; padding:10px">Search Criterias</span></h1><br>

# Search criteria two
# 
# "Similarly, the recorded place of death containing any of a list of text expressions such as “no fixed abode”, “homeless” and “night shelter” or the name or address of a known homeless hostel or project."
# 
# Search criteria three
# 
# "The death had been investigated by a coroner, and the details received by the ONS after the inquest included any of the text expressions or addresses outlined previously. The information provided by coroners is broader and may be more precise than for deaths that do not require an inquest."
# 
# Search criteria four
# 
# "The record contained a “communal establishment code”, which specified a homeless hostel or shelter. These codes are assigned by the ONS during the initial processing of a death registration, based on a periodically updated list of known postcodes of institutions of all kinds, ranging from hospitals to prisons."
# 
# Search criteria five
# 
# "The death occurred in hospital or in a hostel or similar location, and the recorded postcode of the place of residence was identical to the postcode of the place of death. This search ensured the inclusion of homeless people who had been found in need of medical attention in the street and subsequently died in hospital, or certain other possible scenarios."
# 
# https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/bulletins/deathsofhomelesspeopleinenglandandwales/2019registrations#causes-of-death-among-homeless-people

# In[ ]:


df.isnull().sum()


# In[ ]:


#Code by tpmeli  https://www.kaggle.com/tpmeli/barplots-of-all-questions-exploratory/notebook

def graph_barplots(df, colname, title, one_col = False, sort_vals = True, tall = False):
    
    if one_col:
        series = df[colname].value_counts()
    else:
        series = df.filter(like = colname).sum()
    
    if sort_vals:
        series = series.sort_values()
        
    # Get rid of colname in front
    # Capitalize
    
    ax = series.plot(kind = "barh")
    plt.title(title)
    if tall:
        plt.gcf().set_figheight(20)
        
    plt.show()


# "Most of the deaths in 2019 were among men (687 estimated deaths; 88.3% of the total)."
# 
# "Among men, the highest proportion and number of deaths were observed in those aged 45 to 49 years (117 deaths; 17.0% of all male deaths). Women aged 40 to 44 years had the highest number of deaths (21 deaths; 23.1% of all female deaths). The age group 50 to 54 years had the second highest number of estimated deaths both for men (112 deaths; 16.3% of all male deaths) and for women (14 deaths; 15.4% of all female deaths)."
# 
# https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/bulletins/deathsofhomelesspeopleinenglandandwales/2019registrations#causes-of-death-among-homeless-people

# In[ ]:


graph_barplots(df, "sex", "Homeless People Gender", one_col = True)


# "The mean age at death for the identified homeless deaths was 45.9 years for males and 43.4 years for females. In the general population of the same age (15 to 74 years), the mean age at death was 61.5 years and 62.4 years respectively. In contrast, when looking at all deaths registered in England and Wales in 2019, the mean age at death was 76.1 years for males and 80.9 years for females."
# 
# "Mean age at death is not the same as life expectancy."
# 
# https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/bulletins/deathsofhomelesspeopleinenglandandwales/2019registrations#causes-of-death-among-homeless-people

# In[ ]:


graph_barplots(df, "age_group", "Homeless People Age Group", one_col = True)


# In[ ]:


graph_barplots(df, "type_of_death", "Homeless People Types of Deaths", one_col = True)


# In[ ]:


graph_barplots(df, "2013", "Homeless People Deceased in 2013", one_col = True)


# #As far as I can see those bars increased on 2020.

# In[ ]:


graph_barplots(df, "2020", "Homeless People Deceased in 2020", one_col = True)


# In[ ]:


# Count Plot by Nittin Datta
plt.style.use("classic")
plt.figure(figsize=(10, 8))
sns.countplot(df['sex'], palette='flag', **{'hatch':'/','linewidth':3})
plt.xlabel("Gender")
plt.ylabel("Count")
plt.title("Homeless People Deaths by Gender")
plt.xticks(rotation=45, fontsize=8)
plt.show()


# In[ ]:


plt.figure(figsize=(20,4))
plt.subplot(131)
sns.countplot(x= 'sex', data = df, palette="twilight_shifted",edgecolor="black")
plt.xticks(rotation=45)
plt.subplot(132)
sns.countplot(x= 'age_group', data = df, palette="spring",edgecolor="black")
plt.xticks(rotation=45)
plt.show()


# In[ ]:


#Code by Sudhir Kumar https://www.kaggle.com/sudhirnl7/logistic-regression-with-stratifiedkfold

fig ,ax = plt.subplots(2,2,figsize=(16,16))
ax1,ax2,ax3,ax4 = ax.flatten()
sns.countplot(data=df,x='sex',hue='type_of_death',palette='flag',ax=ax1)
sns.countplot(data=df,x='age_group',hue='type_of_death',palette='mako',ax=ax2)
sns.countplot(data=df,x='2020',hue='type_of_death',palette='ocean',ax=ax3)
sns.countplot(data=df,x='2013',hue='type_of_death',palette='twilight',ax=ax4)
plt.xticks(rotation=45); #only the last chart has 45 rotation


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud

plt.rcParams['figure.figsize'] = (15, 15)
wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 100).generate(str(df["type_of_death"]))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Homeless People Type of Deaths',fontsize = 20)
plt.show()


# #Next step: Coronavirus and deaths of homeless people, England and Wales: deaths registered up to 26 June 2020
# 
# https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/articles/coronavirusanddeathsofhomelesspeopleenglandandwalesdeathsregisteredupto26june2020/2020-07-10
