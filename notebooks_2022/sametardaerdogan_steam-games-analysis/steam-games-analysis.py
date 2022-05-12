#!/usr/bin/env python
# coding: utf-8

# <h2 style="background-color:#166088;font-family:newtimeroman;font-size:250%;text-align:center;border-radius: 10px 10px;color:#dbe9ee">Welcome</h2>
# 
# <h4><center>The purpose of this notebook is on the analysis of steam games. It will be a short notebook as there are only 2 variables in the dataset that can be analyzed. Includes topics in entry-level data manipulation and visualization</center></h4>
# 
# 

# <h3 style="background-color:#4F6D7A;font-family:newtimeroman;font-size:175%;text-align:left;border-radius: 5px 5px;color:#dbe9ee">Imports</h3><a id="1"></a>

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


# <h3 style="background-color:#4F6D7A;font-family:newtimeroman;font-size:175%;text-align:left;border-radius: 5px 5px;color:#dbe9ee">Data Manipulation</h3><a id="2"></a>

# In[ ]:


df = pd.read_csv("../input/steam-games-hours-played-and-peak-no-of-players/Book 1.csv")


# In[ ]:


df.info()


# #### >> There are 200 rows and no missing values
# #### >> "Peak No. of Players" and "Hours Played" columns are object. We need to convert dtypes to float for analyze. 

# In[ ]:


df.head()


# #### >> Columns names are irregular. We need to fix them.
# #### >> ID column is unnecessary. We will drop it.

# In[ ]:


df.rename(columns={"\nName":"name",
                   "Peak No. of Players ":"Peak_no_of_players",
                   "Hours Played":"hours_played"},inplace=True)


# In[ ]:


df.drop("ID",axis=1,inplace=True)


# In[ ]:


print("Unique values")
for i in df.columns:
    print(f"{i}: {len(df[i].unique())}")


# #### >> All values are unique. That means all games are unique. There are not same games in dataset.

# In[ ]:


df[["Peak_no_of_players","hours_played"]] = df[["Peak_no_of_players","hours_played"]].apply(lambda x: x.str.replace(',',''))


# In[ ]:


df = df.astype({'Peak_no_of_players': 'float64',
          "hours_played":"float64"})


# #### >> As I said, we need to convert values to float for analyze. We did it here.

# # 
# 
# <h3 style="background-color:#4F6D7A;font-family:newtimeroman;font-size:175%;text-align:left;border-radius: 5px 5px;color:#dbe9ee">Visualization</h3><a id="7"></a>
# 

# #### >> To find Max 5 / Min 5 games, I used 2 different methods. You can use one of them but I just wanted to show you another method.

# In[ ]:


# Method One (to find top 5 games: nlargest  // to find min played 5 games: nsmallest)

topPeakGamesIndex = df.name[df.Peak_no_of_players.nlargest(5).index].values
topPeakGamesValues = df.Peak_no_of_players.nlargest(5)

topHoursPlayedIndex = df.name[df.hours_played.nlargest(5).index].values
topHoursPlayedValues = df.hours_played.nlargest(5)


# In[ ]:


ax = sns.barplot(x=topPeakGamesIndex,y=topPeakGamesValues)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 60)
ax.set_title('TOP 5 "Peak_no_of_players"');


# In[ ]:


ax = sns.barplot(x=topHoursPlayedIndex,y=topHoursPlayedValues)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 60)
ax.set_title('TOP 5 "Hours Played Games"');


# In[ ]:


# Method Two (df.column.sort_values | ascending = True: smallest to largest / False is opposite.)

minPeakGamesIndex = df.name[df.Peak_no_of_players.sort_values(ascending=True)[:5].index].values
minPeakGamesValues = df.Peak_no_of_players.sort_values(ascending=True)[:5].values

minHoursPlayedIndex = df.name[df.hours_played.sort_values(ascending=True)[:5].index].values
minHoursPlayedValues = df.hours_played.sort_values(ascending=True)[:5].values


# In[ ]:


ax = sns.barplot(x=minHoursPlayedIndex,y=minHoursPlayedValues)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 60)
ax.set_title('Min 5 "Hours Played Games"');


# In[ ]:


ax = sns.barplot(x=minPeakGamesIndex,y=minPeakGamesValues)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 60)
ax.set_title('Min 5 "Peak_no_of_players"');


# <h2 style="background-color:#166088;font-family:newtimeroman;font-size:250%;text-align:center;border-radius: 10px 10px;color:#dbe9ee">Thank you for reading</h2>
