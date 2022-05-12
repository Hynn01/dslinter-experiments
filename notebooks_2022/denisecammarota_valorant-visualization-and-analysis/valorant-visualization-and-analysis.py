#!/usr/bin/env python
# coding: utf-8

# # Visualising Valorant Data
# 
# In this notebook I will be visualising and doing some preliminary data analysis on the Valorant Statistics provided by this dataset. This data have been scraped from [thespike](http://thespike.gg) by the dataset owner.
# 
# * [What is Valorant?](#section1)
# 
# * [Map Analysis](#section2)
#     - [Which are the most played maps?](#subsection1)
#     - [Are maps balanced?](#subsection2)
#         
#         
# * [Agent Analysis](#section3)
#     - [Which are the most played? What about each category?](#subsection3)
#     - [Which is the best agent? Are they the most played?](#subsection4)
#         
#         
# * [Pro Player Analysis](#section4)
#     - [How many Pro Players by Country/Region?](#subsection5)
#     - [How many Pro Players by Team?](#subsection6)
#     - [What makes a good Valorant Player?](#subsection7)
# 
# 
# <a id="section1"></a>
# # What is Valorant?
# 
# ![Valorant Official Image](https://images.contentstack.io/v3/assets/bltb6530b271fddd0b1/blt54240895e72cd3bb/61e70d1a79b5bd5a1ef7b956/Chamber_Jett_1920x1080.jpg)
# 
# **Valorant** is a first person shooter game, released by RiotGames on June 2020, for free. Following its release, the game has become increasingly popular worldwide.
# 
# In each match, two teams of 5 players compete by rounds of fixed duration, being the first to win 13 rounds considered the winner team. Each player can choose from a set of agents that have distinct habilities and they can purchase different guns throughout the rounds with credits won according to round result, kills and other game considerations. 
# 
# One team will be considered the defender and other the attacker for the first 12 rounds. After that, the roles get inverted and they take on opposite roles. The attacker team in a round has a spike (which is sort of like a bomb) that they need to plant and wait until it detonates to win. If they do not plant the spike, this team can win if they eliminate all the other team's players. On the other hand, the defenders' team need to either defuse the spike or prevent the other players from planting it before the round time runs out. The sets where it all can take place are called maps. Each one has specific places where the spike could be planted, as well as attacker and defender spawn sites.
# 
# First, we need to retrieve the location of our datasets, and see how many of them we have. 

# In[ ]:


#Importing all relevant libraries that we will use
import pandas as pd
import numpy as np
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.express as px
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go


# In[ ]:


#Seeing the location of our data
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <a id="section2"></a>
# # Map Analysis 
# 
# Valorant currently has 7 different maps to play a match: Breeze, Fracture, Split, Icebox, Haven, Ascent and Bind. All of them, except for Haven, have exactly two spots to plant the spike, called A and B. However, Haven has three, named respectively A, B and C. The latest map ever released was Fracture.
# 
# In this section, we will analyze map statistics for the seven of them. 
# 
# For each one, we have 10 attributes:
# 
# - map, which indicates the map name
# - played, which tells us the percentage of games played on that map
# - avg_spike_plant, which I'm not really sure what it is and [thespike](http://thespike.gg) doesn't clarify
# - rounds, which is the amount of rounds played per map 
# - atk_win_rate, which is the win rate when attacking
# - def_win_rate, which is the win rate when defending
# - pistol_atk_win_rate, which is the win rate when attacking on a pistol round
# - pistol_def_win_rate, which is the win rate when defending on a pistol round
# - second_round_conversion_atk, which is the second round conversion after having won a pistol round on attack
# - second_round_conversion_def, which is the second round conversion after having won a pistol round on defense
# 
# Before starting, I performed some basic data cleaning. Below, you can see some entries of the original and modified datasets to assess the changes yourself if you'd like. 

# In[ ]:


maps_data = pd.read_csv('/kaggle/input/valorant/maps.csv')
maps_data.head()


# In[ ]:


def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""


# In[ ]:


def find_between_nolast( s, last ):
    try:
        end = s.index(last)
        return s[:end]
    except ValueError:
        return ""


# In[ ]:


for i in range(maps_data.shape[0]):
    s_aux = maps_data['played'][i]
    s_aux = find_between(s_aux, '\n', '%')
    maps_data.loc[i,'played'] = float(s_aux)


# In[ ]:


list_tomodify = ['atk_win_rate','def_win_rate','pistol_atk_win_rate','pistol_def_win_rate','second_round_conversion_atk','second_round_conversion_def']
for i in range(len(list_tomodify)):
    for j in range(maps_data.shape[0]):
        s_aux = maps_data[list_tomodify[i]][j]
        s_aux = find_between_nolast(s_aux,'%')
        maps_data.loc[j,list_tomodify[i]] = float(s_aux)


# In[ ]:


maps_data


# <a id="subsection1"></a>
# ## Which are the most played maps? 
# To do this, we are going to first load the data and perform some data cleaning to just get the percentages of played matched in each match. The three most played maps are (in descending order): Ascent, Haven and Icebox. This indicators are influenced by the frequency of each map coming up when a match starts, as well as selective queue dodging. 

# In[ ]:


labels = ['Ascent','Haven','Icebox','Bind','Breeze','Split','Fracture']
fig = {
  "data": [
    {
      "values": maps_data['played'],
      "labels": labels,
      "domain": {"x": [0, .6]},
      "name": "Number Of Students Rates",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },],
  "layout": {
        "title":"Map playing Rates"
    }
}
iplot(fig)


# <a id="subsection2"></a>
# ## Are maps balanced?
# 
# For this, we will analyze the attack and defense general win rates first, which are general and include all the rounds in a Valorant match. 

# ### Average Attack and Defense Win Rates 

# In[ ]:


trace1 = go.Bar(
                x = maps_data.map,
                y = maps_data.atk_win_rate,name='Attack Win Rate',
                marker = dict(color = 'rgba(255, 0, 0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))
trace2 = go.Bar(
                x = maps_data.map,
                y = maps_data.def_win_rate,name='Defense Win Rate',
                marker = dict(color = 'rgba(0, 0, 255, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)))
data = [trace1,trace2]
layout = go.Layout(title= 'Attack and Defense Win Rates by Map',barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# If we compute the diference and plot it on a similar graph, we can see more clearly which are the unbalanced maps. As we can see, Fracture and Bind are the less unbalanced maps, while maps like Split and Ascent are the most unbalanced ones. On these cases, these two maps are unbalanced in benefit of the defenders. 

# In[ ]:


trace3 = go.Bar(
                x = maps_data.map,
                y = maps_data.atk_win_rate - maps_data.def_win_rate,
                marker = dict(color = 'rgba(0, 255, 0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))

data = [trace3]
layout = go.Layout(title= 'Attack and Defense Win Rates by Map',barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# ### Pistol Rounds Attack and Defense Win Rates 

# We can do the same process for the pistol round attack and defense win rates. Rounds 1 and 13 of a game are considered to be pistol rounds, since players have no money to but other types of weapons. 
# 
# From these plots, we can observe that both Split and Ascent remain unbalanced in benefit of the defenders on Pistol Rounds. Plus, Icebox seems to be quite biased as well. More generaly, all maps seem biased to the same side on pistol rounds. That is, is the map generally benefits the attackers, it will also do so on Pistol Rounds and viceversa. 

# In[ ]:


trace1 = go.Bar(
                x = maps_data.map,
                y = maps_data.pistol_atk_win_rate,name='Attack Win Rate on Pistol',
                marker = dict(color = 'rgba(255, 0, 0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))
trace2 = go.Bar(
                x = maps_data.map,
                y = maps_data.pistol_def_win_rate,name='Defense Win Rate on Pistol',
                marker = dict(color = 'rgba(0, 0, 255, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)))
data = [trace1,trace2]
layout = go.Layout(title= 'Attack and Defense Win Rates by Map on Pistol Rounds',barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# In[ ]:


trace3 = go.Bar(
                x = maps_data.map,
                y = maps_data.pistol_atk_win_rate - maps_data.pistol_def_win_rate,
                marker = dict(color = 'rgba(0,255,0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))

data = [trace3]
layout = go.Layout(title= 'Attack and Defense Win Rates by Map on Pistol Rounds',barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# ### Second Round Conversion Rates 
# 
# Finally, the same analysis can be carried out for the second round conversion rates. If we remind ourselves of the definition, this quantity tells us how likely a team that is attacking or defending is to win the round after the pistol round. Since rounds 1 and 13 are considered pistol rounds, this quantity assesses the rounds 2 and 14.
# 
# From our analysis, we observe that all rates are pretty high. Meaning that, in general, a team who has one the pistol round is pretty likely to win the following one as well. However, it is more likely for attacker teams to do this, in all maps. Icebox is the map for which this difference is greater, being 10% more likely for an attacker team who won a pistol round to win the following one. 

# In[ ]:


trace1 = go.Bar(
                x = maps_data.map,
                y = maps_data.second_round_conversion_atk,name='Attack Win Rate on Pistol',
                marker = dict(color = 'rgba(255, 0, 0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))
trace2 = go.Bar(
                x = maps_data.map,
                y = maps_data.second_round_conversion_def,name='Defense Win Rate on Pistol',
                marker = dict(color = 'rgba(0, 0, 255, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)))
data = [trace1,trace2]
layout = go.Layout(title= 'Attack and Defense Win Rates by Map on Pistol Rounds',barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# In[ ]:


trace3 = go.Bar(
                x = maps_data.map,
                y = maps_data.second_round_conversion_atk - maps_data.second_round_conversion_def,
                marker = dict(color = 'rgba(0,255,0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))

data = [trace3]
layout = go.Layout(title= 'Attack and Defense Win Rates by Map on Pistol Rounds',barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# <a id="section3"></a>
# # Agent Analysis 
# 
# In this section, we are going to analyze the performance of Valorant Agents from which players can choose from. In this dataset, we have 18 possible agents, which were the only ones available up until a few weeks, when an agent called Fase was added. Valorant divides its agents in 5 categories, which we list below as well as the corresponding agents for each one:
# 
# - **Duelist**: Phoenix, Reyna, Jett, Raze, Yoru and Neon.
# - **Controller**: Brimstone, Viper, Omen and Astra. 
# - **Sentinel**: Sage, Cypher, Killjoy and Chamber.
# - **Initiator**: Sova, Breach, Skye and KAY/O.
# 
# For each agent, the dataset provides us with several attributes such as:
# - agent, the name of the agent
# - pick_rate, the pick rate of an agent in matches
# - round, the amount of total rounds played with an agent
# - rating, which I'm not really sure how it is computed
# - ACS, average combat score, a measure of how a player performs in match 
# - K/D, which is the kills-deaths ratio
# - ADR, the average damage per round
# - KPR, the average kills per round
# - DPR, the average damage caused per round
# - APR, the average amount of assistences per round 
# 
# There are also two extra columns I did not see in the dataset's description and I don't understand, even after looking at the source. Therefore, I won't use them in my analysis. 
# 
# As I did with the maps, I am first going to do some very simple data cleaning. 

# In[ ]:


agents_data = pd.read_csv('/kaggle/input/valorant/agents.csv')
agents_data.head()


# In[ ]:


list_tomodify = ['pick_rate']
for i in range(len(list_tomodify)):
    for j in range(agents_data.shape[0]):
        s_aux = agents_data[list_tomodify[i]][j]
        s_aux = find_between_nolast(s_aux,'%')
        agents_data.loc[j,list_tomodify[i]] = float(s_aux)


# In[ ]:


agents_data.head()


# <a id="subsection3"></a>
# ## Which are the most played agents? What about each category?
# 
# First, I want to find out which are the most played agents in general as well as for each agent category. For that, I am just going to make a simple bar plot of agent pick rate, sorted in descending order so we can easily see which are the most and least played agents. 
# 
# To sum up, we have that Jett, Sova, Astra and Viper are the top 4 most played agents. 
# 
# Separating agents into categories and ordering them from most to least played: 
# - **Duelist**: **Jett**, Raze, Reyna, Neon, Phoenix and Yoru.
# - **Controller** : **Astra**, Viper, Omen and Brimstone.
# - **Sentinel**: **Killjoy**, Chamber, Sage and Cypher.
# - **Initiator**: **Sova**, Skye, KAY/O and Breach. 

# In[ ]:


trace3 = go.Bar(
                x = agents_data.agent,
                y = agents_data.pick_rate,
                marker = dict(color = 'rgba(0,255,0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))

data = [trace3]
layout = go.Layout(title= 'Agent Pick Rate',barmode = "group",xaxis={'categoryorder':'total descending'})
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# <a id="subsection4"></a>
# ## Which is the best agent? Are they the most played?
# 
# Then, we have several other columns that may help us answer the question of the best agent, according to several criteria. 
# 
# ### Average Combat Score (ACS)
# 
# Firstly, we are going to start analysing the Average Combat Score (ACS). This is something that is given to each player at the end of each match, and takes into account kills, deaths, assistances as well as spike plants and first bloods, to name a few criteria.
# 
# From this plot, we can make several interesting observations: 
# - The ACS is not hugely different among agents. 
# - There seems to be no clear relationship between agent pick rate and the ACS. Meaning, people do not necessarily choose more frequently the agents that end up having higher values of ACS, despite ACS determining the MVP for the match and for each team. For example, Phoenix has a high ACS but it is not picked that often. 
# - Duelists have the highest ACS, possibly because kills are extremly important when computing this quantity and the role of duelists is more associated with having to kill enemies. 

# In[ ]:


trace3 = go.Bar(
                x = agents_data.agent,
                y = agents_data.ACS,
                marker = dict(color = 'rgba(0,255,0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))

data = [trace3]
layout = go.Layout(title= 'Average Combat Score',barmode = "group",xaxis={'categoryorder':'total descending'})
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# ### Average Damage per Round (ADR)
# 
# We can reproduce the same analysis but using another indicator, the Average Damage per Round (ADR). 
# 
# - Again, this quantity doesn't seem closely related with agent pick rates. For example, Raze has the highest ADR but doesn't have a high pick rate. 
# - It is interesting to note that most duelists score higher in ADR. This is consistent with our last observation about ACS.

# In[ ]:


trace3 = go.Bar(
                x = agents_data.agent,
                y = agents_data.ADR,
                marker = dict(color = 'rgba(0,255,0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))

data = [trace3]
layout = go.Layout(title= 'Average Damage per Round',barmode = "group",xaxis={'categoryorder':'total descending'})
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# ### Kills/Deaths Ratio (K/D)
# 
# The Kills/Deaths Ratio (K/D) is another extremely popular indicator for Valorant performances. It indicated how many enemies a person killed for each one of their deaths. Meaning, if I kill 2 enemies and I only die one time, my K/D will be 2. We will plot it in the next cell, and order agents by K/D on descending order. 
# 
# From the plot, it is interesting to remark that: 
# 
# - K/D rates are usually near one and don't vary that much between agents. Probably it compensates like that because killing a lot of people requires players to go into enemy dominated territory and make more dangerous plays. 
# - Jett, Chamber, Phoenix, Raze and Reyna have the highests K/D ratios. This is interesting because, in my experience, this agents have ultimate habilities that gives a player the chance to kill off several enemies.

# In[ ]:


trace3 = go.Bar(
                x = agents_data.agent,
                y = agents_data['K/D'],
                marker = dict(color = 'rgba(0,255,0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))

data = [trace3]
layout = go.Layout(title= 'Agent K/D Ratio',barmode = "group",xaxis={'categoryorder':'total descending'})
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# ### Average Kills, Deaths and Assistences per Round
# 
# Finally, we will plot the average kills (KPR), deaths (DPR) and assistences (APR) per round for each agent.
# 
# - In general, all duelists have high KPR. The top 4 agents in terms of KPR are Jett, Raze, Reyna and Phoenix. However, these agents do not have a high APR.
# - The agents with the highest APR values, in descending order, are KAY/O, Brimstone, Sage and Omen. KAY/O has abilities that can leave enemies vulnerable, Brimstone can deploy combat stimulants as well smokes, Sage can heal herself and other allies and Omen can blind enemies with smokes.
# - Agents that can heal others, meaning Sage and Skye, have higher APR. However, they do not have the highest values.
# - DPR is similar among all agents, but it is slightly higher for Duelists, probably as a consequence of their roles in-game, like we have mentioned before. 

# In[ ]:


trace1 = go.Bar(
                x = agents_data.agent,
                y = agents_data.KPR,name='Kills per Round',
                marker = dict(color = 'rgba(255, 0, 0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))
trace2 = go.Bar(
                x = agents_data.agent,
                y = agents_data.DPR,name='Deaths per Round',
                marker = dict(color = 'rgba(0, 0, 255, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)))
trace3 = go.Bar(
                x = agents_data.agent,
                y = agents_data.APR,name='Assistences per Round',
                marker = dict(color = 'rgba(0, 255, 0, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)))
data = [trace1,trace2,trace3]
layout = go.Layout(title= 'Kills, Deaths and Assistences per Round',barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# <a id="section4"></a>
# # Pro Players Analysis 
# 
# In this section, we will work with a dataset that provides us with data on Valorant Pro Players to do some short analysis on what makes are the characteristics of great Valorant players. In this case, the attributes provided are:
# 
# - player, which indicated the player's name
# - country, which indicates a player's country. Sometimes, when country is not available, we only have region or 'world'. 
# - team, which is the players team. Sometimes there is no team value, I will asume it is because they have no team. 
# - rounds, the amount of rounds played.
# - rating, which is not clearly described. I guess it is a measure of how good a player is.
# - ACS, the average combat score
# - ADR, the average damage per round
# - KPR, which is kills per round
# - DPR, the kills per round
# - APR, the assistences per round
# 
# There are a few other attributes, which I'm not sure what it is. Therefore, I won't be using them for this analysis. 
# 
# Before getting any results, I did first a bit of data cleaning. The before and after of the dataset can be visualised below. 

# In[ ]:


players_data = pd.read_csv('/kaggle/input/valorant/players.csv')
players_data.head()


# In[ ]:


players_data.loc[players_data['team'].isnull(),'team'] = 'no team'
players_data = players_data.iloc[:,:11]


# In[ ]:


players_data.head()


# <a id="subsection5"></a>
# ## How many Pro Players by Country/Region?
# 
# First, we will sort and plot the 10 regions, which have the most Valorant Pro Players in two plots: a bar chart and a pie chart. From there, we can see that Canada, USA, Brazil and Turkey are the leading countries in terms of Pro Players, having more than 200 players each. Canada, the one with the most, has approximately 500 Pro Players.  

# In[ ]:


players_data_country = players_data.groupby(by=['country'])['country'].count().sort_values(ascending=False)


# In[ ]:


labels_2 = ['Canada','USA','Brazil', 'Turkey', 'Europe', 'Indonesia', 'Russia', 'Thailand', 'France', 'South Korea']
trace3 = go.Bar(
                x = labels_2,
                y = players_data_country.values,
                marker = dict(color = 'rgba(0,255,0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))

data = [trace3]
layout = go.Layout(title= 'Pro Players by Country',barmode = "group",xaxis={'categoryorder':'total descending'})
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# In[ ]:


fig = {
  "data": [
    {
      "values": list(players_data_country.values)[:10],
      "labels": labels_2,
      "domain": {"x": [0, .6]},
      "name": "Number Of Students Rates",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },],
  "layout": {
        "title":"Percentage of Pro Players by Country"
    }
}
iplot(fig)


# <a id="subsection6"></a>
# ## How many Pro Players by Team? 
# 
# Next, we can also group Pro Players by team. In the following bar charts, we present the 10 teams around the world with the highest amount of these players. For the first one, we consider pro players that have no team where as, in the second plot, we only take into account well known teams. We can conclude that:
# 
# - Despite in game teams being composed by 5 players, a lot of teams have more than only 5 people. 
# - Most of these Pro Players do not have a known team. 

# In[ ]:


players_data_team = players_data.groupby(by=['team'])['team'].count().sort_values(ascending=False)


# In[ ]:


trace3 = go.Bar(
                x = players_data_team.index[:10],
                y = players_data_team.values[:10],
                marker = dict(color = 'rgba(0,255,0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))

data = [trace3]
layout = go.Layout(title= 'Pro Players by Team',barmode = "group",xaxis={'categoryorder':'total descending'})
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# In[ ]:


trace3 = go.Bar(
                x = players_data_team.index[1:11],
                y = players_data_team.values[1:11],
                marker = dict(color = 'rgba(0,255,0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))

data = [trace3]
layout = go.Layout(title= 'Pro Players by Team (excluding no team)',barmode = "group",xaxis={'categoryorder':'total descending'})
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# <a id="subsection7"></a>
# ## What makes a good Valorant Player? 
# 
# In this section, we will try to see what makes a good Valorant Player using some of the provided attributes to evaluare performance, such as rating, rounds, ACS, K/D, ADR, KPR, DPR and APR.
# 
# ### Summary Statistics of Attributes
# First, we calculated some relevant statistics for each one of these characteristics, which can be observed in the table below. There, we can observe indicators such as the mean damage per round or ACS of Pro Players (and perhaps comparing them to our own statistics? Riot website lets you see your own in game statistics after signing in). From these statistics, besides simply obtaining expected values for the different attributes, we can observe some peculiarities such as:
# - There is a player that played around 8500 rounds when the mean is about 900
# - There is a huge variation in ACS, being 126 the minimum and 322 the maximum. Not necessarily huge ACS is necessary to be a Pro Player, apparently. It is remarkable that the same happens for K/D, another important indicator of performance. Maybe this is due to the fact that there are different roles in the game, and a good sentinel or a good initiator maybe don't kill as many people as a good duelist does (it would be nice to know the usual roles for each player to analyze this). 
# - ADR is mostly near one, indicating that Pro Players kill at least one person every round. 
# - DPR variate from 0.5 to 0.9, indicating that pro players have a range of dying every two rounds, or each round. 
# - APR is pretty low, at around 0.3, and with a small standard deviation. 

# In[ ]:


players_data.describe()


# ### Correlations 
# Also, to investigated what makes a good Valorant Player, we have calculated the correlations among all the attributes provided for the performance of a player, and we plot them in the HeatMap below. From here, we can gather a few observations:
# - All the autocorrelations are 1, as one would expect.
# - The amount of rounds played seems to bear no relationship with any other indicators of performance. That means, a higher amount of rounds played doesn't imply someone is better or worse. 
# - Indicators of performance such as ACS and rating hold a positive correlation with ADR, KPR and K/D, and a negative correlation with DPR. This makes sense because killing more enemies is an indicator of good performance. On the other hand, dying more could be considered as performing worse. 
# - It is remarkable that assistances have a null or negative correlation with ACS, ADR, KPR and K/D. That means, a lot of assistences per round don't seem to make a pro player. If anything, it even seems to influence negatively on quantities like their scores and the amount of enemies they kill. However, the correlation between APR and DPR is negative, which is probably due to the fact that an assist may mean a person did some damage on an enemy and didn't die but, at the same time, didn't do sufficient damage to kill that player. 

# In[ ]:


corr = players_data[3:].corr(method='pearson')
fig = go.Figure(data=go.Heatmap(
        z=corr.values.tolist()[::-1],
        x = list(corr.index),
        y = list(corr.index)[::-1],
        colorscale='viridis'))
#fig.update_layout(margin = dict(t=0,r=550,b=0,l=0))
iplot(fig)

