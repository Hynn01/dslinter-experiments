#!/usr/bin/env python
# coding: utf-8

# ## NFL Big Data Bowl

# <img src="https://i.ibb.co/prCZrWf/Dv-C9-Btp-Ws-AACt-AP.jpg" width="800"> 

# **Content**
# 1. [Importing Libraries and Loading Dataset](#1)
# 1. [Understanding Dataset](#2)
# 1. [Data Visualization](#3)
# 1. [Ending](#4)

# <a id="1"></a> <br>
# # Importing Libraries and Loading Dataset

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) 

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df_train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)


# <a id="2"></a> <br>
# # Understanding Dataset

# NFL Big Data Bowl dataset includes location, speed, acceleration, and velocity for all players on the field along with many other features such as weather conditions, stadium etc.
# 
# Let's begin.

# In[ ]:


print("Total Row Number: {0} \nTotal Col Number: {1}".format(df_train.shape[0], df_train.shape[1]))


# In[ ]:


df_train.head()


# In[ ]:


df_train.info()


# In[ ]:


df_train.columns


# There are 49 columns and 11 columns have null values. Total null values of these are below:

# In[ ]:


nullvalues = df_train.loc[:, df_train.isnull().any()].isnull().sum().sort_values(ascending=False)

print(nullvalues)


# In[ ]:


#i'll ignore to ids.
f,ax = plt.subplots(figsize=(12,10))
sns.heatmap(df_train.iloc[:,2:].corr(),annot=True, linewidths=.1, fmt='.1f', ax=ax)

plt.show()


# There are a close relationship between S and Dis. There are also a relationship between JerseyNumber and PlayerWeight. 

# <a id="3"></a> <br>
# # Data Visualization

# In[ ]:


unique_plays = df_train.groupby('Season').agg({'PlayId': 'nunique', 'GameId': 'nunique', 'NflId':'nunique'})
unique_plays.columns = ["PlayCount","GameCount", "PlayerCount"]

print(unique_plays)


# In[ ]:


f,ax=plt.subplots(1,3,figsize=(11,4))
unique_plays["PlayCount"].plot(ax=ax[0],color='crimson', kind='bar')

ax[0].set_title('Count of Plays')
unique_plays["GameCount"].plot(ax=ax[1],color='darkmagenta', kind='bar')

ax[1].set_title('Count of Games')
unique_plays["PlayerCount"].plot(ax=ax[2],color='slateblue', kind='bar')

ax[2].set_title('Count of Player')

plt.show()


# In[ ]:


df_team = df_train.loc[:,['PlayId', 'Team']].drop_duplicates()
teamcount = df_team["Team"].value_counts()
df_teamcount = pd.DataFrame({'Team': teamcount.index, 'Count': teamcount.values})

fig = px.bar(df_teamcount, x='Team', y='Count')
fig.update_layout(
    title_text="Team (Away or Home)", height=300, width=300)

fig.show()


# It's reasonable, "home" and "away" are equal to each other

# In[ ]:


df_player = df_train.loc[:,['NflId', 'DisplayName', 'PlayerBirthDate', 'PlayerWeight', 'PlayerHeight', 'PlayerCollegeName']].drop_duplicates()

df_player["HeightFt"] = df_player["PlayerHeight"].str.split('-', expand=True)[0].astype(int)
df_player["HeightIn"] = df_player["PlayerHeight"].str.split('-', expand=True)[1].astype(int)
df_player["HeightCm"] = df_player["HeightFt"]*30.48 + df_player["HeightIn"]*2.54

df_player["WeightKg"] = df_player["PlayerWeight"]*0.45359237

df_height = df_player.groupby(['PlayerHeight','HeightFt','HeightIn']).size().reset_index().sort_values(["HeightFt", "HeightIn"])

df_height.columns = ["PlayerHeight","HeightFt","HeightIn","Count"]


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(13,5))
df_height.loc[:,["PlayerHeight","Count"]].set_index("PlayerHeight").plot(ax=ax[0],color='slateblue', kind='bar')

ax[0].set_xlabel("Height") 
ax[0].set_ylabel("Count") 
ax[0].get_legend().remove()
ax[0].set_title('Player Height (ft-in)')

df_player.PlayerWeight.plot.hist(ax=ax[1],bins=30,edgecolor='black',color='steelblue')

ax[1].set_xlabel("Weight") 
ax[1].set_title('Player Weight (lbs)')

plt.show()


# In[ ]:


fig = go.Figure(data=go.Scattergl
(
    x = df_player["HeightCm"], 
    y = df_player["WeightKg"],
    mode='markers',
    text = df_player['DisplayName'],
    marker=dict(line_width=0.4),
    marker_color='rgb(0,145,119)'))

fig.update_layout(title='Weight Height Correlation', height=600, width=800)
fig.update_xaxes(title_text="Height (Cm)")
fig.update_yaxes(title_text="Weight (Kg)")

fig.show()


# In[ ]:


Collage = df_player["PlayerCollegeName"].value_counts()
df_CollageCount = pd.DataFrame({'College Name':Collage.index, 'Count':Collage.values}).sort_values("Count", ascending = False).head(50)

fig = px.bar(df_CollageCount, x='College Name', y='Count', title="The 50 Top Colleges With The Most Players", height=700, width=800)

fig.update_traces(marker_color='rgb(239, 117, 100)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.7)

fig.show()


# In[ ]:


df_player['BirthDate'] = df_player['PlayerBirthDate'].astype('datetime64[ns]')

#df_train["TimeSnap"].astype('datetime64[ns]').sort_values(ascending=False)
#max date in TimeSnap cloumn is 2018-12-31, so i'm not using today function

df_player['MaxDate'] = pd.to_datetime('2018-12-31')

df_player['Age'] = round((df_player['MaxDate'] - df_player['BirthDate'])/np.timedelta64(1,'D')/365.25,1)


# In[ ]:


plt.figure(figsize=(6,6))

plt.hist(df_player["Age"], bins=20 ,edgecolor='black', color='crimson')
plt.title("Age of Players in 2018 Season")

plt.show()


# In[ ]:


df_stadium = df_train.loc[:,["PlayId","Stadium", "StadiumType"]].drop_duplicates()
stadium = df_stadium["Stadium"].value_counts()
df_stadiumCount = pd.DataFrame({'Stadium':stadium.index, 'Count':stadium.values}).sort_values("Stadium")

fig = px.bar(df_stadiumCount, x='Stadium', y='Count', title="Games by Stadium", height=700, width=800)

fig.update_traces(marker_color='darkorange', marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.7)

fig.show()


# In[ ]:


StadiumType = df_stadium["StadiumType"].value_counts()

df_StadiumType = pd.DataFrame({'StadiumType':StadiumType.index, 'Count':StadiumType.values}).sort_values("StadiumType")

fig = px.bar(df_StadiumType, x='StadiumType', y='Count', title="Games by Stadium Type", height=600, width=800)

fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)

fig.show()


# In[ ]:


stadium_loc = df_train.loc[:,['GameId', 'Location']].drop_duplicates()
stadium_loc = stadium_loc['Location'].value_counts()

df_stadium_loc = pd.DataFrame({'Location':stadium_loc.index, 'Count':stadium_loc.values})

df_stadium_loc["State"] = df_stadium_loc["Location"].str.split(', ', expand=True)[0]
df_stadium_loc["StateAbb"] = df_stadium_loc["Location"].str.split(', ', expand=True)[1]


fig = go.Figure(data=go.Choropleth(
    locations=df_stadium_loc["StateAbb"], 
    z = df_stadium_loc['Count'], 
    locationmode = 'USA-states', 
    text = df_stadium_loc['State'],
    colorscale=
            [[0.0, "rgb(251, 237, 235)"],
            [0.09, "rgb(245, 211, 206)"],
            [0.12, "rgb(239, 179, 171)"],
            [0.15, "rgb(236, 148, 136)"],
            [0.22, "rgb(239, 117, 100)"],
            [0.35, "rgb(235, 90, 70)"],
            [0.45, "rgb(207, 81, 61)"],
            [0.65, "rgb(176, 70, 50)"],
            [0.85, "rgb(147, 59, 39)"],
            [1.00, "rgb(110, 47, 26)"]],
    autocolorscale = False))

fig.update_layout(
    geo_scope='usa',
    title_text = 'Total Number of Games by States in 2017 and 2018', height=500, width=700)

fig.show()


# There are a lot of version of stadium types. It must be cleaned up.

# In[ ]:


df_Def = df_train.loc[:,["PlayId","DefensePersonnel"]].drop_duplicates()
DefenseCount = df_Def["DefensePersonnel"].value_counts()

dic_Defense = {'DL': [0],'LB': [0],'DB': [0],'OL': [0]}
df_Defense = pd.DataFrame.from_dict(dic_Defense, orient='index')
df_Defense.columns = ["Count"]

for i in DefenseCount.index:
    for j in i.split(", "):
        df_Defense.loc[ j.split(" ")[1] : j.split(" ")[1],:] += int(j.split(" ")[0]) * DefenseCount[i]


# In[ ]:


colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']

fig = go.Figure(data=[go.Pie(labels=df_Defense.index.values.tolist(), 
                             values=df_Defense["Count"].tolist())])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=18,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(
    title_text="Defensive Players Ratio in All Plays", height=500, width=700
)

fig.show()


# OLs are the minority group among the defensive players with 15 total. So they aren't visible on the chart.  

# In[ ]:


df_Off = df_train.loc[:,["PlayId","OffensePersonnel"]].drop_duplicates()
df_Off["OffensePersonnel"] = df_Off["OffensePersonnel"].str.replace(",1", ", 1")
df_Off["OffensePersonnel"] = df_Off["OffensePersonnel"].str.replace(",2", ", 2")
OffenseCount = df_Off["OffensePersonnel"].value_counts()

dic_Offense = {'DL': [0],'LB': [0],'DB': [0],'OL': [0],'RB': [0],'TE': [0],'WR': [0],'QB': [0]}
df_Offense = pd.DataFrame.from_dict(dic_Offense, orient='index')
df_Offense.columns = ["Count"]

for i in OffenseCount.index:
    for j in i.split(", "):
        df_Offense.loc[ j.split(" ")[1] : j.split(" ")[1],:] += int(j.split(" ")[0]) * OffenseCount[i]


# In[ ]:


colors = ['g', 'b', 'p', 'plum', 'gold', 'mediumturquoise', 'darkorange', 'lightgreen']

fig = go.Figure(data=[go.Pie(labels=df_Offense.index.values.tolist(), 
                             values=df_Offense["Count"].tolist() , hole=.3)])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=18,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(
    title_text="Offensive Players Ratio in All Plays", height=500, width=700)

fig.show()


# In[ ]:


df_speed = df_train.loc[:,['DisplayName', 'S']].groupby('DisplayName').mean()
df_speed.columns = ["Average Speed"]
df_speed = df_speed.sort_values("Average Speed", ascending = False)


# In[ ]:


fig = px.histogram(df_speed, x="Average Speed",
                   title='Average Speed Distribution of Players',
                   opacity=0.8,
                   color_discrete_sequence=['indianred']
                   )

fig.update_layout(
    yaxis_title_text='Count',
    height=500, width=800)

fig.update_traces(marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.8
                 )

fig.show()


# It seems to, most players have 2 - 2.3 yards/second speed.

# In[ ]:


df_speed_yards = df_train[df_train["NflIdRusher"] == df_train["NflId"]].loc[:,['GameId', 'DisplayName', 'S', 'Yards']]


fig = go.Figure(data=go.Scattergl
(
    x = df_speed_yards["Yards"], 
    y = df_speed_yards["S"],
    text = df_speed_yards['DisplayName'],
    mode='markers',
    marker=dict(line_width=0.4),
    marker_color='darkorange'))

fig.update_layout(title='Speed to Yards', height=700, width=800)

fig.update_xaxes(title_text="Yards")
fig.update_yaxes(title_text="Speed (yards/second)")

fig.show()


# The graph above shows the speed of players on their way to target.

# In[ ]:


df_MaxSpeed = df_train.loc[:,['DisplayName', 'PlayerBirthDate', 'TimeSnap', 'S', 'Orientation']].groupby(['DisplayName', 'PlayerBirthDate'], as_index=False).max()

df_MaxSpeed["Age"] = round((df_MaxSpeed['TimeSnap'].astype('datetime64[ns]') - df_MaxSpeed['PlayerBirthDate'].astype('datetime64[ns]'))/np.timedelta64(1,'D')/365.25,1)


# I have calcalculated real time age with game played time.

# In[ ]:


fig = go.Figure(data=go.Scattergl
(
    x = df_MaxSpeed["S"], 
    y = df_MaxSpeed["Age"],
    text = df_MaxSpeed['DisplayName'],
    mode='markers',
    marker=dict(line_width=0.4),
    marker_color='mediumorchid'
)
               ) 

fig.update_layout(title='Age of The Players When They Make Their Maximum Speed', height=700, width=800)
fig.update_traces(marker_line_color='rgb(8,48,107)', opacity=0.7)

fig.update_xaxes(title_text="Speed of Player (yards/second)")
fig.update_yaxes(title_text="Age")

fig.show()


# In[ ]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=df_MaxSpeed["S"], y=df_MaxSpeed["Orientation"],
                    mode='markers',
                    text = df_MaxSpeed['DisplayName'],     
                    name='Speed',
                    marker=dict(line_width=0.4)
                    )
             )

fig.add_trace(go.Scatter(x=df_MaxSpeed["Age"], y=df_MaxSpeed["Orientation"],
                    mode='markers',
                    text = df_MaxSpeed['DisplayName'],     
                    name='Age',
                    marker=dict(line_width=0.4)
                    )
             )

fig.update_layout(title='Orientation & Speed +  Orientation & Age Correlation', height=500, width=800)

fig.update_traces(marker_line_color='rgb(8,48,107)', opacity=0.8)
fig.update_yaxes(title_text="Orientation")

fig.show()


# In[ ]:


df_defenders = df_train.loc[:,['DefendersInTheBox', 'Yards']]

d1 = df_defenders[df_defenders.DefendersInTheBox==1]['Yards']
d2 = df_defenders[df_defenders.DefendersInTheBox==2]['Yards']
d3 = df_defenders[df_defenders.DefendersInTheBox==3]['Yards']
d4 = df_defenders[df_defenders.DefendersInTheBox==4]['Yards']
d5 = df_defenders[df_defenders.DefendersInTheBox==5]['Yards']
d6 = df_defenders[df_defenders.DefendersInTheBox==6]['Yards']
d7 = df_defenders[df_defenders.DefendersInTheBox==7]['Yards']
d8 = df_defenders[df_defenders.DefendersInTheBox==8]['Yards']
d9 = df_defenders[df_defenders.DefendersInTheBox==9]['Yards']
d10 = df_defenders[df_defenders.DefendersInTheBox==10]['Yards']
d11 = df_defenders[df_defenders.DefendersInTheBox==11]['Yards']


# In[ ]:


fig = go.Figure()
yards = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11]
tags = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]

for index, category in enumerate(yards):
    fig.add_trace(go.Box(y=category, name=tags[index]))
    
fig.update_layout(autosize=False, width=800,height=500,showlegend=False,
                  title="Yards and Number of Defenders in the Box Correlation", 
                  yaxis=dict(title="Yards"), 
                  xaxis=dict(title="Number of Players"),
                  paper_bgcolor='rgb(243, 243, 243)', 
                  plot_bgcolor='rgb(243, 243, 243)', 
                  margin=dict(l=40,r=30,b=80,t=100,)
                 )

fig.show()


# If we zoom in on the chart, a regular distribution appears as follows.

# <img src="https://i.ibb.co/WfCg2Ny/kaggle-NFL.png" width="800"> 

# In[ ]:


df_OffenseForm = df_train.loc[:,['PlayId', 'OffenseFormation', 'Season']].drop_duplicates()
OffenseForm2017 = df_OffenseForm[df_train["Season"] == 2017]["OffenseFormation"].value_counts()
OffenseForm2018 = df_OffenseForm[df_train["Season"] == 2018]["OffenseFormation"].value_counts()

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(
    go.Pie(labels=OffenseForm2017.index, values=OffenseForm2017.values, name="2017"),
              1, 1)

fig.add_trace(
    go.Pie(labels=OffenseForm2018.index, values=OffenseForm2018.values, name="2018"),
              1, 2)

fig.update_traces(hole=.4, hoverinfo="label+value")

fig.update_layout(
    title_text="Favorite Offense Formation in 2017 and 2018",
    height=500, width=800,
    annotations=[dict(text='2017', x=0.18, y=0.5, font_size=20, showarrow=False),
                 dict(text='2018', x=0.82, y=0.5, font_size=20, showarrow=False)]
)

fig.show()


# In[ ]:


df_xy = df_train[(df_train["NflIdRusher"] == df_train["NflId"])].loc[:,["Yards", "DefendersInTheBox", "X", "Y" ]].sort_values("X" )


fig = go.Figure(data=go.Scattergl
(
    x = df_xy["X"], 
    y = df_xy["Y"],
    mode='markers',
    marker=dict(line_width=0.8),
    marker_color='tan'))

fig.update_layout(title='X and Y Positions of Players', height=700, width=800)
fig.update_traces(marker_line_color='rgb(8,48,107)', opacity=0.7)

fig.update_xaxes(title_text="X")
fig.update_yaxes(title_text="Y")

fig.show()


# We can see vital points as two lines. We can also see field guidelines and outliers.

# <a id="4"></a> <br>
# # Ending

# #### To be continued... If you like the kernel, Please upvote.
