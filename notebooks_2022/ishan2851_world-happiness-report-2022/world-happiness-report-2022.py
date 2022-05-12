#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
sns.set_style('dark')
warnings.filterwarnings('ignore')
font={'family':'sans-serif',
      'weight':'500',
      'size':14}
plt.rc('font',**font)


# In[ ]:


happiness=pd.read_csv("../input/world-happiness-report-2022/World Happiness Report 2022.csv")
happiness.head()


# In[ ]:


country_name_dict = {"Luxembourg*": "Luxembourg", 
                    "Guatemala*": "Guatemala",
                    "Kuwait*": "Kuwait",
                    "Belarus*": "Belarus",
                    "Turkmenistan*": "Turkmenistan",
                    "North Cyprus*": "North Cyprus",
                    "Libya*": "Libya",
                    "Azerbaijan*": "Azerbaijan",
                    "Gambia*": "Gambia",
                    "Liberia*": "Liberia",
                    "Niger*": "Niger",
                    "Comoros*": "Comoros",
                    "Palestinian Territories*": "Palestinian Territories",
                    "Eswatini, Kingdom of*": "Eswatini, Kingdom of",
                    "Madagascar*": "Madagascar",
                    "Chad*": "Chad",
                    "Yemen*": "Yemen",
                    "Mauritania*": "Mauritania",
                    "Lesotho*": "Lesotho",
                    "Botswana*": "Botswana",
                    "Rwanda*": "Rwanda",}
happiness["Country"] = happiness["Country"].replace(country_name_dict)
happiness


# In[ ]:


continents_df = pd.read_csv("../input/world-happiness-report-2021/world-happiness-report-2021.csv")
continents_dict = d = {k:v for k,v in zip(continents_df["Country name"],continents_df["Regional indicator"])}
continents_dict


# In[ ]:


happiness["Continents"] = happiness["Country"].map(continents_dict)
happiness.isnull().sum()


# In[ ]:


continents_name_dict = {
    "Sub-Saharan Africa":"Africa",
    "Western Europe":"Europe",
    "Latin America and Caribbean":"South America",
    "Middle East and North Africa":"Africa",
    "Central and Eastern Europe":"Europe",
    "Commonwealth of Independent States":"Asia",
    "Southeast Asia":"Asia",
    "South Asia":"Asia",
    "East Asia":"Asia",
    "North America and ANZ":"North America"
    }
happiness["Continents"] = happiness["Continents"].replace(continents_name_dict)
happiness=happiness.rename(columns={'Explained by: GDP per capita':'GDP','Explained by: Healthy life expectancy':'Life Expectancy','Explained by: Freedom to make life choices':'Life choices','Explained by: Perceptions of corruption':'Corruption','Explained by: Social support':'Social Support','Explained by: Generosity':'Generosity'})
happiness


# In[ ]:


Africa = happiness[happiness["Continents"]=="Africa"]
Europe = happiness[happiness["Continents"]=="Europe"]
Asia = happiness[happiness["Continents"]=="Asia"]
South_America = happiness[happiness["Continents"]=="South America"]
North_America = happiness[happiness["Continents"]=="North America"]


# # Happiness Score For Different Continents

# **Asia Continent**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(33)]
color_map[0] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(Asia["Country"], Asia['Happiness score'],color=color_map,zorder=3)
plt.ylabel("Happiness Score")
plt.title("Asia Continent")


# **Taiwan Province Of China has highest Happiness Score**

# **Africa Continent**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(50)]
color_map[0] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(Africa["Country"], Africa['Happiness score'],color=color_map,zorder=3)
plt.ylabel("Happiness Score")
plt.title("Africa Continent")


# **Israel has highest Happiness Score**

# **Europe Continent**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(37)]
color_map[0] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(Europe["Country"], Europe['Happiness score'],color=color_map,zorder=3)
plt.ylabel("Happiness Score")
plt.title("Europe Continent")


# **Finland has highest Happiness Score**

# **South America Continent**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(19)]
color_map[0] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(South_America["Country"], South_America['Happiness score'],color=color_map,zorder=3)
plt.ylabel("Happiness Score")
plt.title("South America Continent")


# **Costa Rica has highest Happiness Score**

# **North America Continent**

# In[ ]:


plt.figure(figsize=(15,5))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(37)]
color_map[0] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(North_America["Country"], North_America['Happiness score'],color=color_map,zorder=3)
plt.ylabel("Happiness Score")
plt.title("South America Continent")


# **New Zealand has highest Happiness Score**

# # GDP Per Capita For Different Countries

# **Asia**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(33)]
color_map[1] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(Asia["Country"], Asia['GDP'],color=color_map,zorder=3)
plt.ylabel("GDP Per Capita")
plt.title("Asia Continent")


# **Singapore has highest GDP per capita**

# **Africa**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(50)]
color_map[2] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(Africa["Country"], Africa['GDP'],color=color_map,zorder=3)
plt.ylabel("GDP Per Capita")
plt.title("Africa Continent")


# **United Arab Emirates has highest GDP per capita**

# **Europe**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(37)]
color_map[5] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(Europe["Country"], Europe['GDP'],color=color_map,zorder=3)
plt.ylabel("GDP Per Capita")
plt.title("Europe Continent")


# **Luxembourg has highest GDP per capita**

# **North America**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(4)]
color_map[3] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(North_America["Country"], North_America['GDP'],color=color_map,zorder=3)
plt.ylabel("GDP Per Capita")
plt.title("North America Continent")


# **United States has highest GDP per capita**

# **South America**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(19)]
color_map[2] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(South_America["Country"], South_America['GDP'],color=color_map,zorder=3)
plt.ylabel("GDP Per Capita")
plt.title("South America Continent")


# **Panama has highest GDP per capita**

# # Social Support For Different Countries

# **Asia**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(33)]
color_map[15] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(Asia["Country"], Asia['Social Support'],color=color_map,zorder=3)
plt.ylabel("Social Support")
plt.title("Asia Continent")


# **Turkmenistan has highest Social Support**

# **Africa**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(50)]
color_map[0] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(Africa["Country"], Africa['Social Support'],color=color_map,zorder=3)
plt.ylabel("Social Support")
plt.title("Africa Continent")


# **Israel has highest Social Support**

# **Europe**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(37)]
color_map[2] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(Europe["Country"], Europe['Social Support'],color=color_map,zorder=3)
plt.ylabel("Social Support")
plt.title("Europe Continent")


# **Iceland has highest Social Support**

# **North America**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(33)]
color_map[0] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(North_America["Country"], North_America['Social Support'],color=color_map,zorder=3)
plt.ylabel("Social Support")
plt.title("North America Continent")


# **New Zealand has highest Social Support**

# **South America**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(19)]
color_map[1] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(South_America["Country"], South_America['Social Support'],color=color_map,zorder=3)
plt.ylabel("Social Support")
plt.title("South America Continent")


# **Uruguay has highest Social Support**

# # Healthy Life Expectancy For Different Countries

# **Asia**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(33)]
color_map[17] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(Asia["Country"], Asia['Life Expectancy'],color=color_map,zorder=3)
plt.ylabel("Healthy Life Expectancy")
plt.title("Asia Continent")


# **Hong Kong S.A.R. of China has highest Healthy life expectancy**

# **Africa**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(50)]
color_map[0] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(Africa["Country"], Africa['Life Expectancy'],color=color_map,zorder=3)
plt.ylabel("Healthy Life Expectancy")
plt.title("Africa Continent")


# **Israel has highest Healthy life expectancy**

# **Europe**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(37)]
color_map[3] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(Europe["Country"], Europe['Life Expectancy'],color=color_map,zorder=3)
plt.ylabel("Healthy Life Expectancy")
plt.title("Europe Continent")


# **Switzerland has highest Healthy life expectancy**

# **North America**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(4)]
color_map[2] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(North_America["Country"], North_America['Life Expectancy'],color=color_map,zorder=3)
plt.ylabel("Healthy Life Expectancy")
plt.title("North America Continent")


# **Canada has highest Healthy life expectancy**

# **South America**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(19)]
color_map[5] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(South_America["Country"], South_America['Life Expectancy'],color=color_map,zorder=3)
plt.ylabel("Healthy Life Expectancy")
plt.title("South America Continent")


# **Chile has highest Healthy life expectancy**

# # Life Choices For Different Continents

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(33)]
color_map[27] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(Asia["Country"], Asia['Life choices'],color=color_map,zorder=3)
plt.ylabel("Life Choices")
plt.title("Asia Continent")


# **Cambodia has Highest Freedom to make life choices**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(50)]
color_map[2] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(Africa["Country"], Africa['Life choices'],color=color_map,zorder=3)
plt.ylabel("Life Choices")
plt.title("Africa Continent")


# **U.A.E. has Highest Freedom to make life choices**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(37)]
color_map[0] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(Europe["Country"], Europe['Life choices'],color=color_map,zorder=3)
plt.ylabel("Life Choices")
plt.title("Europe Continent")


# **Finland has Highest Freedom to make life choices**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(19)]
color_map[0] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(North_America["Country"], North_America['Life choices'],color=color_map,zorder=3)
plt.ylabel("Life Choices")
plt.title("North America Continent")


# **New Zealand has Highest Freedom to make life choices**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(19)]
color_map[1] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(South_America["Country"], South_America['Life choices'],color=color_map,zorder=3)
plt.ylabel("Life Choices")
plt.title("South America Continent")


# **Uruguay has Highest Freedom to make life choices**

# # Generosity For Different Continents

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(33)]
color_map[21] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(Asia["Country"], Asia['Generosity'],color=color_map,zorder=3)
plt.ylabel("Generoisty")
plt.title("Asia Continent")


# **Indonesia has Highest Generoisty**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(50)]
color_map[9] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(Africa["Country"], Africa['Generosity'],color=color_map,zorder=3)
plt.ylabel("Generoisty")
plt.title("Africa Continent")


# **Gambia has Highest Generoisty**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(37)]
color_map[18] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(Europe["Country"], Europe['Generosity'],color=color_map,zorder=3)
plt.ylabel("Generoisty")
plt.title("Europe Continent")


# **Kosovo has Highest Generoisty**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(4)]
color_map[1] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(North_America["Country"], North_America['Generosity'],color=color_map,zorder=3)
plt.ylabel("Generoisty")
plt.title("North America Continent")


# **Australia has Highest Generoisty**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(19)]
color_map[18] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(South_America["Country"], South_America['Generosity'],color=color_map,zorder=3)
plt.ylabel("Generoisty")
plt.title("South America Continent")


# **Venezuela has Highest Generoisty**

# # Corruption For Different Continents

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(33)]
color_map[1] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(Asia["Country"], Asia['Corruption'],color=color_map,zorder=3)
plt.ylabel("Corruption")
plt.title("Asia Continent")


# **Singapore has Highest Perceptions Of Corruption**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(50)]
color_map[47] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(Africa["Country"], Africa['Corruption'],color=color_map,zorder=3)
plt.ylabel("Corruption")
plt.title("Africa Continent")


# **Rwanda has Highest Perceptions Of Corruption**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(37)]
color_map[0] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(Europe["Country"], Europe['Corruption'],color=color_map,zorder=3)
plt.ylabel("Corruption")
plt.title("Europe Continent")


# **Finland has Highest Perceptions Of Corruption**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(4)]
color_map[0] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(North_America["Country"], North_America['Corruption'],color=color_map,zorder=3)
plt.ylabel("Corruption")
plt.title("North America Continent")


# **New Zealand has Highest Perceptions Of Corruption**

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
color_map = ["#4A4655" for _ in range(19)]
color_map[1] = "#4898EF"
plt.grid(color='black', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
plt.bar(South_America["Country"], South_America['Corruption'],color=color_map,zorder=3)
plt.ylabel("Corruption")
plt.title("South America Continent")


# **Uruguay has Highest Perceptions Of Corruption**
