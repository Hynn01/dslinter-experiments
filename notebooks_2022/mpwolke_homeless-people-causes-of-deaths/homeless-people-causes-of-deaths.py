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


# <center style="font-family:verdana;"><h1 style="font-size:200%; padding: 10px; background: #9400D3;"><b style="color:white;">Homeless People Causes of Deaths</b></h1></center>
# 
# 
# "As in the general population, homeless people die from a broad range of causes such as accidents, diseases of the liver, ischaemic heart diseases, cancers, and influenza and pneumonia. Most deaths among homeless people are captured in our National Statistics definitions of drug-related poisoning, suicide, and alcohol-specific deaths."
# 
# "289 estimated deaths of homeless people in 2019 were related to drug poisoning, that is, 37.1% of all estimated deaths. Suicide and alcohol-specific causes accounted for 14.4% (112 deaths) and 9.8% (76 deaths) of estimated deaths of homeless people in 2019 respectively."
# 
# 
# https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/bulletins/deathsofhomelesspeopleinenglandandwales/2019registrations#causes-of-death-among-homeless-people

# In[ ]:


df = pd.read_csv("/kaggle/input/cusersmarildownloadsicdtencsv/icdten.csv", delimiter=';', encoding='utf8')
pd.set_option('display.max_columns', None)
df.tail()


# In[ ]:


df.isnull().sum()


# <h1><span class="label label-default" style="background-color:#ADD8E6;border-radius:100px 100px; font-weight: bold; font-family:Garamond; font-size:20px; color:black; padding:10px">No fixed abode. Night Shelter. Homeless Addresses.</span></h1><br>

# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT-Lh6oKk3H559hT3nsuXC4bjtfwnyG-MzbCQ&usqp=CAU)https://www.citizensadvice.org.uk/Global/CitizensAdvice/Post%20and%20Telecoms/Homelessness%20report%20-%20Final.pdf

# "The recorded place of residence contained any of a list of text expressions such as “no fixed abode”, “homeless” and “night shelter” or the name or address of a known homeless hostel or project. An extensive list of addresses was compiled from publicly available sources. While this list was necessarily incomplete, the statistical model was found to be robust against even a substantial number of omissions."
# 
# https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/bulletins/deathsofhomelesspeopleinenglandandwales/2019registrations#causes-of-death-among-homeless-people

# In[ ]:


##Code by Taha07  https://www.kaggle.com/taha07/data-scientists-jobs-analysis-visualization/notebook

from wordcloud import WordCloud
from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'Red',
                      height =2000,
                      width = 2000
                     ).generate(str(df["ICD10_sub_chapter_ grouping"]))
plt.rcParams['figure.figsize'] = (12,12)
plt.axis("off")
plt.imshow(wordcloud)
plt.title("International Classification of Diseases, Tenth Revision (ICD-10)")
plt.show()


# In[ ]:


##Code by Taha07  https://www.kaggle.com/taha07/data-scientists-jobs-analysis-visualization/notebook

from wordcloud import WordCloud
from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'Purple',
                      height =2000,
                      width = 2000
                     ).generate(str(df["type_of_death"]))
plt.rcParams['figure.figsize'] = (12,12)
plt.axis("off")
plt.imshow(wordcloud)
plt.title("Homeless People Types of Deaths")
plt.show()


# #Since there are many diseases classifications, I added just ten [:] to avoid overlapping the names.

# In[ ]:


#Code by Lucas Abrahão https://www.kaggle.com/lucasabrahao/trabalho-manufatura-an-lise-de-dados-no-brasil

plt.figure(figsize=(6,4)) #Figure size should come 1st otherwise it won't change the size 
df["ICD10_sub_chapter_ grouping"].value_counts()[:10].plot.barh(color=['blue', '#f5005a', '#7FFF00'], title='International Classification of Diseases, Tenth Revision (ICD-10)');


# In[ ]:


#Code by Lucas Abrahão https://www.kaggle.com/lucasabrahao/trabalho-manufatura-an-lise-de-dados-no-brasil

plt.figure(figsize=(6,4))
df["type_of_death"].value_counts().plot.barh(color=['blue', '#f5005a', '#7FFF00', '#DC143C'], title='Homeless People Type of Deaths');


# In[ ]:


#Code by Gabriel Preda

def plot_count(feature, title, df, size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    total = float(len(df))
    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='summer')
    g.set_title("Number and percentage of {}".format(title))
    if(size > 2):
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center")
    plt.xticks(rotation=45)    
    plt.show()


# In[ ]:


plot_count("ICD10_sub_chapter_ grouping", "International Classification of Diseases, Tenth Revision", df,4)#xticks shoul be on the snippet above


# In[ ]:


labels = 'Estimated deaths', 'Identified deaths'
sizes = [337, 149]  #must have same number labels, sizes and explode
explode = (0, 0.2)  # only "explode" the 2nd slice 

fig1, ax1 = plt.subplots(figsize=(6,6))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[ ]:


#Code by Taha07  https://www.kaggle.com/taha07/data-scientists-jobs-analysis-visualization/notebook

color = plt.cm.Greens(np.linspace(0,1,20))
df["ICD10_sub_chapter_ grouping"].value_counts().sort_values(ascending=False).head(20).plot.pie(y="ICD-10 Codes",colors=color,autopct="%0.1f%%")
plt.title("International Classification of Diseases, 10th Revision")
plt.axis("off")
plt.show()


# In[ ]:


fig = px.bar(df, 
             x='ICD10_sub_chapter_ grouping', y='2020',color_discrete_sequence=['blue'],
             title='International Classification of Diseases', text='type_of_death')
fig.show()


# In[ ]:


fig = px.bar(df, 
             x='type_of_death', y='2013',color_discrete_sequence=['#DC143C'],
             title='International Classification of Diseases', text='ICD10_sub_chapter_ grouping')
fig.show()


# In[ ]:


fig = px.parallel_categories(df, color="2020", color_continuous_scale=px.colors.sequential.Blackbody)
fig.show()


# #Homeless People Deaths seven years before Covid-19 (2013)

# In[ ]:


fig = px.parallel_categories(df, color="2013", color_continuous_scale=px.colors.sequential.haline)
fig.show()


# In[ ]:


fig = px.bar(df, 
             x='2020', y='type_of_death',color_discrete_sequence=['#2B3A67'],
             title='Homeless People 2020 Types of Deaths', text='ICD10_sub_chapter_ grouping')
fig.show()


# #Homeless People Deaths by Accidents

# In[ ]:


acc = df[(df['ICD10_sub_chapter_ grouping']=='Accidents')].reset_index(drop=True)
acc.head()


# #Transpose

# In[ ]:


#Code by Arun P R https://www.kaggle.com/arunprathap/unicef-immunization-data-preliminary-eda

acc.head().T


# #Drop columns with na values

# In[ ]:


#Code by Arun P R https://www.kaggle.com/arunprathap/unicef-immunization-data-preliminary-eda

acc_clean = acc.dropna(axis=1)
acc_clean.head().T


# In[ ]:


#Code by Arun P R https://www.kaggle.com/arunprathap/unicef-immunization-data-preliminary-eda

pd.DataFrame(acc_clean[['ICD10_sub_chapter_ grouping','2020', '2019', '2014','2013']].value_counts()).T


# In[ ]:


acc_clean.columns.tolist()


# In[ ]:


#Code by Olga Belitskaya https://www.kaggle.com/olgabelitskaya/parts-of-speech
plt.figure(figsize=(10,5))
sns.countplot(y="2020",data=acc_clean,
             facecolor=(0,0,0,0),linewidth=5,
             edgecolor=sns.color_palette("Purples_r"))
plt.title('Homeless People Deaths in 2020 by Accidents',
         fontsize=15);


# In[ ]:


# Count Plot
plt.style.use("classic")
plt.figure(figsize=(6, 4))
sns.countplot(df['type_of_death'], palette='Greens', **{'hatch':'/','linewidth':3})
plt.xlabel("type_of_death")
plt.ylabel("Count")
plt.title("Homeless People Types of Deaths")
plt.xticks(rotation=45, fontsize=8)
plt.show()


# #Why the columns above have the same size? Identified 30,7 and Estimated is 69,3.
# 
# Something went wrong in that chart above.

# <h1><span class="label label-default" style="background-color:#ADD8E6;border-radius:100px 100px; font-weight: bold; font-family:Garamond; font-size:20px; color:black; padding:10px">Addressing Homelessness with Data Analytics</span></h1><br>
# 
# Authors: Mahesh Kelkar, Nagen Suriya, Rachel Frey, Shane Engel.
# 
# 
# ![](https://www2.deloitte.com/content/dam/insights/us/articles/5228_addressing-homelessness/figures/5228_fig3.png)
# 
# "While traditional approaches have helped policymakers reduce homelessness to an extent, technology and data analytics can help them go a step further in dealing with the country’s homelessness problem."
# 
# https://www2.deloitte.com/xe/en/insights/industry/public-sector/homelessness-data.html
