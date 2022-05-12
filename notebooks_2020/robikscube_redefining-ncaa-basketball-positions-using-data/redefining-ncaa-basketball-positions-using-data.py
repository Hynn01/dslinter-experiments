#!/usr/bin/env python
# coding: utf-8

# # Redefining the Roster
# *Using unsupervised machine learning to cluster NCAA players by their stats.*
# 
# Most people who watch March Madness are familiar with the standard positions of basketball: guards, forwards, and centers. But as basketball continues to evolve, the lines that define players have become more blurred. In a 2013 article, Myron Medcalf questioned if the traditional positions were even still relevant. Metcalf said, "The descriptors that made sense years ago feel inadequate today. Point guards, shooting guards, small forwards, power forwards and centers -- in some cases -- have been replaced by young men who simply call themselves players. [1]"
# 
# My question, then is: What does the data show us about player positions? Would it be possible to categorize players' positions based solely on the data? After defining these "positions" what types of rosters fare well in the March Madness tournament?
# 
# Using an unsupervised machine learning technique called clustering, I've identified 5 distinct types of college basketball players. They are: Ball Distributors, Elite Bigs, Shot Takers, 3 Point Specialists, and Paint Dominators. In this notebook, I'll go through the process used to identify these groups, and go into more detail about what makes each unique. So let's let the data tell the story of the players who rule March Madness!

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from tqdm.notebook import tqdm

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

import lightgbm as lgb

from scipy.spatial.distance import cdist 

import time

import matplotlib.pylab as plt
import matplotlib as mpl
from matplotlib.patches import Circle, Rectangle, Arc

import plotly.express as px

from IPython.display import display, HTML

import warnings
warnings.filterwarnings('ignore')

sns.set(rc={'figure.figsize':(15, 5)})
palette = sns.color_palette("bright", 10)
sns.set_style("whitegrid")

pd.set_option('max_columns', 500)
pd.set_option('max_rows', 500)
pd.options.display.float_format = '{:.2f}'.format
# plt.rcParams['figure.figsize'] = (15.0, 5.0)


# In[ ]:


def create_ncaa_half_court(ax=None, three_line='mens', court_color='#dfbb85',
                           lw=3, lines_color='black', lines_alpha=0.5,
                           paint_fill='blue', paint_alpha=0.4,
                          inner_arc=False):
    """
    Version 2020.2.19

    Creates NCAA Basketball Half Court
    Dimensions are in feet (Court is 97x50 ft)
    Created by: Rob Mulla / https://github.com/RobMulla

    * Note that this function uses "feet" as the unit of measure.
    * NCAA Data is provided on a x range: 0, 100 and y-range 0 to 100
    * To plot X/Y positions first convert to feet like this:
    ```
    Events['X_'] = (Events['X'] * (94/100))
    Events['Y_'] = (Events['Y'] * (50/100))
    ```
    ax: matplotlib axes if None gets current axes using `plt.gca`
    
    three_line: 'mens', 'womens' or 'both' defines 3 point line plotted
    court_color : (hex) Color of the court
    lw : line width
    lines_color : Color of the lines
    lines_alpha : transparency of lines
    paint_fill : Color inside the paint
    paint_alpha : transparency of the "paint"
    inner_arc : paint the dotted inner arc
    """
    if ax is None:
        ax = plt.gca()

    # Create Pathes for Court Lines
    center_circle = Circle((50/2, 94/2), 6,
                           linewidth=lw, color=lines_color, lw=lw,
                           fill=False, alpha=lines_alpha)
    hoop = Circle((50/2, 5.25), 1.5 / 2,
                       linewidth=lw, color=lines_color, lw=lw,
                       fill=False, alpha=lines_alpha)

    # Paint - 18 Feet 10 inches which converts to 18.833333 feet - gross!
    paint = Rectangle(((50/2)-6, 0), 12, 18.833333,
                           fill=paint_fill, alpha=paint_alpha,
                           lw=lw, edgecolor=None)
    
    paint_boarder = Rectangle(((50/2)-6, 0), 12, 18.833333,
                           fill=False, alpha=lines_alpha,
                           lw=lw, edgecolor=lines_color)
    
    arc = Arc((50/2, 18.833333), 12, 12, theta1=-
                   0, theta2=180, color=lines_color, lw=lw,
                   alpha=lines_alpha)
    
    block1 = Rectangle(((50/2)-6-0.666, 7), 0.666, 1, 
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    block2 = Rectangle(((50/2)+6, 7), 0.666, 1, 
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    ax.add_patch(block1)
    ax.add_patch(block2)
    
    l1 = Rectangle(((50/2)-6-0.666, 11), 0.666, 0.166,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    l2 = Rectangle(((50/2)-6-0.666, 14), 0.666, 0.166,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    l3 = Rectangle(((50/2)-6-0.666, 17), 0.666, 0.166,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    ax.add_patch(l1)
    ax.add_patch(l2)
    ax.add_patch(l3)
    l4 = Rectangle(((50/2)+6, 11), 0.666, 0.166,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    l5 = Rectangle(((50/2)+6, 14), 0.666, 0.166,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    l6 = Rectangle(((50/2)+6, 17), 0.666, 0.166,
                           fill=True, alpha=lines_alpha,
                           lw=0, edgecolor=lines_color,
                           facecolor=lines_color)
    ax.add_patch(l4)
    ax.add_patch(l5)
    ax.add_patch(l6)
    
    # 3 Point Line
    if (three_line == 'mens') | (three_line == 'both'):
        # 22' 1.75" distance to center of hoop
        three_pt = Arc((50/2, 6.25), 44.291, 44.291, theta1=12,
                            theta2=168, color=lines_color, lw=lw,
                            alpha=lines_alpha)

        # 4.25 feet max to sideline for mens
        ax.plot((3.34, 3.34), (0, 11.20),
                color=lines_color, lw=lw, alpha=lines_alpha)
        ax.plot((50-3.34, 50-3.34), (0, 11.20),
                color=lines_color, lw=lw, alpha=lines_alpha)
        ax.add_patch(three_pt)

    if (three_line == 'womens') | (three_line == 'both'):
        # womens 3
        three_pt_w = Arc((50/2, 6.25), 20.75 * 2, 20.75 * 2, theta1=5,
                              theta2=175, color=lines_color, lw=lw, alpha=lines_alpha)
        # 4.25 inches max to sideline for mens
        ax.plot( (4.25, 4.25), (0, 8), color=lines_color,
                lw=lw, alpha=lines_alpha)
        ax.plot((50-4.25, 50-4.25), (0, 8.1),
                color=lines_color, lw=lw, alpha=lines_alpha)

        ax.add_patch(three_pt_w)

    # Add Patches
    ax.add_patch(paint)
    ax.add_patch(paint_boarder)
    ax.add_patch(center_circle)
    ax.add_patch(hoop)
    ax.add_patch(arc)
    
    if inner_arc:
        inner_arc = Arc((50/2, 18.833333), 12, 12, theta1=180,
                             theta2=0, color=lines_color, lw=lw,
                       alpha=lines_alpha, ls='--')
        ax.add_patch(inner_arc)

    # Restricted Area Marker
    restricted_area = Arc((50/2, 6.25), 8, 8, theta1=0,
                        theta2=180, color=lines_color, lw=lw,
                        alpha=lines_alpha)
    ax.add_patch(restricted_area)
    
    # Backboard
    ax.plot(((50/2) - 3, (50/2) + 3), (4, 4),
            color=lines_color, lw=lw*1.5, alpha=lines_alpha)
    ax.plot( (50/2, 50/2), (4.3, 4), color=lines_color,
            lw=lw, alpha=lines_alpha)

    # Half Court Line
    ax.axhline(94/2, color=lines_color, lw=lw, alpha=lines_alpha)

    
    # Plot Limit
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 94/2 + 2)
    ax.set_facecolor(court_color)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    return ax


# In[ ]:


###########
# Data Prep
###########

MENS_PBP_DIR = '../input/march-madness-analytics-2020/MPlayByPlay_Stage2'

MPlayers = pd.read_csv(f'{MENS_PBP_DIR}/MPlayers.csv', error_bad_lines=False)
MTeamSpelling = pd.read_csv('../input/march-madness-analytics-2020/MDataFiles_Stage2/MTeamSpellings.csv',
                            engine='python')
mens_events = []
for year in [2015, 2016, 2017, 2018, 2019]:
    mens_events.append(pd.read_csv(f'{MENS_PBP_DIR}/MEvents{year}.csv'))
MEvents = pd.concat(mens_events)

MTeams = pd.read_csv('../input/march-madness-analytics-2020/MDataFiles_Stage2/MTeams.csv')
MPlayers = MPlayers.merge(MTeams[['TeamID','TeamName']], on='TeamID', how='left')

area_mapping = {0: np.nan,
                1: 'under basket',
                2: 'in the paint',
                3: 'inside right wing',
                4: 'inside right',
                5: 'inside center',
                6: 'inside left',
                7: 'inside left wing',
                8: 'outside right wing',
                9: 'outside right',
                10: 'outside center',
                11: 'outside left',
                12: 'outside left wing',
                13: 'backcourt'}

MEvents['Area_Name'] = MEvents['Area'].map(area_mapping)

# Normalize X, Y positions for court dimentions
# Court is 50 feet wide and 94 feet end to end.
MEvents['X_'] = (MEvents['X'] * (94/100))
MEvents['Y_'] = (MEvents['Y'] * (50/100))

MEvents['X_half'] = MEvents['X']
MEvents.loc[MEvents['X'] > 50, 'X_half'] = (100 - MEvents['X'].loc[MEvents['X'] > 50])
MEvents['Y_half'] = MEvents['Y']
MEvents.loc[MEvents['X'] > 50, 'Y_half'] = (100 - MEvents['Y'].loc[MEvents['X'] > 50])

MEvents['X_half_'] = (MEvents['X_half'] * (94/100))
MEvents['Y_half_'] = (MEvents['Y_half'] * (50/100))

# Merge Player name onto events
MEvents = MEvents.merge(MPlayers,
              how='left',
              left_on='EventPlayerID',
              right_on='PlayerID')

# Only Look at Events with Player assoicated and X, Y position
MPlayerEvents = MEvents.query('EventPlayerID != 0 and X > 0')

# Create GameId
MPlayerEvents['GameId'] =     MPlayerEvents['Season'].astype('str') + '_' +     MPlayerEvents['DayNum'].astype('str') + '_' +     MPlayerEvents['WTeamID'].astype('str') + '_' +     MPlayerEvents['LTeamID'].astype('str')

EventPlayerSeasonCombo = MPlayerEvents[['EventPlayerID','Season']].drop_duplicates().reset_index(drop=True)

# Expand MPlayers to have a row for each player
MPlayerSeason = MPlayers.merge(EventPlayerSeasonCombo,
               left_on=['PlayerID'],
               right_on=['EventPlayerID'],
              validate='1:m') \
    .drop('EventPlayerID', axis=1)

MPlayerSeason['PlayerID_Season'] = MPlayerSeason['PlayerID'].astype('int').astype('str') + '_' + MPlayerSeason['Season'].astype('str')
MPlayerEvents['PlayerID_Season'] = MPlayerEvents['PlayerID'].astype('int').astype('str') + '_' + MPlayerEvents['Season'].astype('str')

MPlayerSeasonStats = pd.read_csv('../input/ncaa-mplayerseasonstats/MPlayerSeasonStats.csv')

MPlayerSeasonStats.loc[MPlayerSeasonStats['Weight'] < 150, 'Weight'] = 150 # Fix for zero weights

MPlayerSeasonStats['Position'] = MPlayerSeasonStats['Pos'].map({'F':'Forward',
                                                      'G':'Guard',
                                                      'C':'Center'})
MPlayerSeasonStats['RSCI Top 100 Number'] = MPlayerSeasonStats['RSCI Top 100'].str.split(' ', expand=True)[0].fillna(1000).astype('int')

KAGGLE_PER_GAME_FEATURES = [c for c in MPlayerSeasonStats if 'per_game' in c and 'ShotCount' in c]


# In[ ]:


f_name_dict = { 'MP' : 'Minutes Played',
                 'FG' : 'Field Goals',
                 'FGA':'Field Goal Attempts',
                 'FG%' : 'Field Goal Percentage',
                 '2P' : '2-Point Field Goals',
                 '2PA': '2-Point Field Goal Attempts',
                 '2P%': '2-Point Field Goal Percentage',
                 '3P' : '3-Point Field Goals',
                 '3PA' : '3-Point Field Goal Attempts',
                 '3P%' : '3-Point Field Goal Percentage',
                 'FT' : 'Free Throws',
                 'FTA' : 'Free Throw Attempts',
                 'FT%' : 'Free Throw Percentage',
                 'ORB' : 'Offensive Rebounds',
                 'DRB':'Defensive Rebounds',
                 'TRB' : 'Total Rebounds',
                 'AST' : 'Assists',
                 'STL' : 'Steals',
                 'BLK' : 'Blocks',
                 'TOV' : 'Turnovers',
                 'PF' : 'Personal Fouls',
                 'PTS' : 'Points'}

# Make Features Per Minute Played
for f in ['FG', 'FGA', '2P', '2PA', '3P', '3PA', 'FTA',
          'ORB', 'DRB',  'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'] + KAGGLE_PER_GAME_FEATURES:
    if f in f_name_dict:
        MPlayerSeasonStats[f_name_dict[f] + ' Per Minute'] = MPlayerSeasonStats[f] / MPlayerSeasonStats['MP']
    else:
        MPlayerSeasonStats[f] = MPlayerSeasonStats[f] / MPlayerSeasonStats['MP']


# In[ ]:


# Features
KAGGLE_PER_GAME_FEATURES = [c for c in MPlayerSeasonStats if 'per_game' in c and 'ShotCount' in c]
SR_PER_GAME_FEATURES = ['MP', 'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P',
                        '3PA', '3P%', 'FT', 'FTA',
                        'FT%', 'ORB', 'DRB', 'TRB', 'AST',
                        'STL', 'BLK', 'TOV', 'PF',
                        'PTS', 'HeightFeet','Weight']
PLAYER_FEATS = ['Position', 'PlayerID_Season','FirstName',
                'LastName','HeightFeet','TeamName','Season',
                'G', 'GS','PTS','TRB','HeightFeet','FTA','STL']

FEATURES = ['Field Goal Attempts Per Minute',
            '2-Point Field Goals Per Minute',
            '3-Point Field Goals Per Minute',
            'Free Throw Attempts Per Minute',
            'Offensive Rebounds Per Minute',
            'Defensive Rebounds Per Minute',
            'Assists Per Minute',
            'Blocks Per Minute',
            'Turnovers Per Minute',
            'Personal Fouls Per Minute',
           ]

MPlayerSeasonStats_ =     MPlayerSeasonStats     .query('GS > 20 and EventCount >= 0')     [FEATURES + PLAYER_FEATS]     .dropna()     .copy()

X = MPlayerSeasonStats_[FEATURES].values
ss = StandardScaler()
X = ss.fit_transform(X)
y = MPlayerSeasonStats_['PlayerID_Season'].values

# Create 5 Player Clusters
model = KMeans(n_clusters=5, random_state=529)
kmeans_pred = model.fit_predict(X)
MPlayerSeasonStats_['kmeans_Cluster'] = kmeans_pred
MPlayerSeasonStats_['Data_Position'] = MPlayerSeasonStats_['kmeans_Cluster'] + 1

cluster_names_dict = {1 : 'Shot Taker',
                      2 : 'Skilled Big',
                      3 : '3 Point Specialist',
                      4 : 'Ball Distributor',
                      5 : 'Paint Dominator'}

MPlayerSeasonStats_['Cluster_Name'] = MPlayerSeasonStats_['Data_Position'].map(cluster_names_dict)


# This plot shows only two dimensions (Points vs Rebounds) for the 5 roster positions that I've created. It looks a little bit messy, but its helpful to see how the shape of these clusters are formed by the features used. Keep in mind this only shows two features, and the clusters are formed using 10 features!

# In[ ]:


data = MPlayerSeasonStats_.merge(MPlayerSeasonStats[['PlayerID_Season','TRB','FGA']])
from matplotlib.patches import Ellipse

x_feat = 'TRB'
y_feat = 'PTS'

fig, ax = plt.subplots(figsize=(15, 15))
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]

idx = 0
for i, d in data.query('GS > 20').groupby('Cluster_Name'):
    d.plot(x='PTS', y='TRB', kind='scatter', ax=ax,
           color=color_pal[idx], alpha=0.5, label=f'{i}', s=25)
    idx += 1
idx = 0

ax.annotate('Zion Williamson 2019 (Skilled Big)',
            xy=(22.60, 8.90), xycoords='data',
            xytext=(0.99, 0.65), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top')

ax.annotate('Lonzo Ball 2017 (Ball Distributor)',
            xy=(14.60, 6), xycoords='data',
            xytext=(0.98, 0.51), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top')

ax.annotate('Aaron Harrison 2015 (3Pt Specialist)',
            xy=(11.00, 2.6), xycoords='data',
            xytext=(0.5, 0.05), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top')

ax.annotate('Kaleb Tarczewski 2016 (Paint Dominator)',
            xy=(9.40, 9.30), xycoords='data',
            xytext=(0.32, 0.72), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top')

ax.annotate('Romeo Langford 2019 (Shot Taker)',
            xy=(16.50, 5.40), xycoords='data',
            xytext=(0.95, 0.10), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top')

plt.title('Data Driven Positions: Rebounds vs Points', fontsize=25)
plt.xlabel('Points / Game')
plt.ylabel('Total Rebounds / Game')
plt.legend()
plt.show()


# ## Features and Approach
# 
# To pair down the data, I only focused on the Men's numbers. I pulled various stats for each player/season. These statistics were normalized by the number of minutes played by each player- I didn't want to have clusters forming around the amount of time a player was in the game.  Rather, I wanted the statistics to be about what the player did with the amount of time he was on the court. Also noteable: I filtered down to include only those players who started at least 20 games in a given season.
# 
# **Features Used:**
# - Field Goal Attempts
# - 2-Point Field Goals
# - 3-Point Field Goals
# - Free Throw Attempts
# - Offensive Rebounds
# - Defensive Rebounds
# - Assists
# - Blocks
# - Turnovers
# - Personal Fouls
# 
# I found that KMeans clustering worked the best for this case, as it tended to do a better job of balancing the clusters. I had to choose the *number* of clusters I wanted the algorithm to produce- after many tests, I found that five produced the best result. I've hidden the code, but you can expand the cells below to look in more detail at how the algorithm was set up.

# ## Comparing the Groups in Detail
# By plotting out the average stat for each cluster, it's immediately clear how they differ. Notice that some clusters tend to lean more defensive, with high numbers of blocks and rebounds- while others lean more offensive. I used these analyases to name the groups. 

# In[ ]:


MPlayerSeasonStats_[FEATURES + ['Cluster_Name']].groupby('Cluster_Name')     .mean().sort_index(ascending=False).T     .plot(kind='barh', figsize=(15, 12),
          title='Average Stats by Group')
plt.show()


# Another way to compare the groups is by using boxplots. This is allows me to see the standard deviation and outliers for each statistic. For instance, notice that the players at the extreme high end of 2-Point field goals are in the "Skilled Bigs" group.

# In[ ]:


clus_name_short ={'3 Point Specialist': '3PT Spec',
                  'Shot Taker':'Shot Taker',
                  'Ball Distributor' : 'Distrib',
                  'Skilled Big':'SBig',
                  'Paint Dominator': 'PaintDom'}

MPlayerSeasonStats_['Cluster_Name_Short'] = MPlayerSeasonStats_['Cluster_Name'].map(clus_name_short)

fig, axs = plt.subplots(3, 3, figsize=(15, 10))
axs = axs.flatten()
idx = 0
for f in FEATURES[:9]:
    sns.boxplot(x='Cluster_Name_Short', y=f, data=MPlayerSeasonStats_, ax=axs[idx])
    axs[idx].set_title(f)
    axs[idx].set_xlabel('')
    axs[idx].set_ylabel('')
    idx += 1
plt.tight_layout()
plt.show()


# 
# # The Lineup!
# Below I'll take a deeper dive into each group to explore what makes them unique. Before I do that it's helpful to take a quick look at each positions' shot locations side by side.

# In[ ]:


fig, axs = plt.subplots(1, 5, figsize=(18, 3.6))
axs = axs.flatten()
idx = 0
for cluster in MPlayerSeasonStats_['Cluster_Name'].unique():
    playerid_seasons = MPlayerSeasonStats_.loc[MPlayerSeasonStats_['Cluster_Name'] == cluster]['PlayerID_Season'].unique()
    Made = MPlayerEvents.loc[MPlayerEvents['PlayerID_Season'].isin(playerid_seasons)]         .query('EventType == "made2" or EventType == "made3" or EventType == "miss2" or EventType == "miss3"')
    axs[idx] = create_ncaa_half_court(ax=axs[idx],
                                 three_line='mens',
                                 court_color='black',
                                      lines_color='white',
                                 paint_alpha=0,
                                 inner_arc=True)
    #     hb1 = axs[idx].plot(x=Turnover.query('Y_ != 0')['Y_half_'],
    #                      y=Turnover.query('Y_ != 0')['X_half_'],
    #                      gridsize=10, bins='log', cmap='inferno')
    Made.query('Y_ != 0').plot(x='Y_half_',
                               y='X_half_',
                               kind='scatter',
                               ax=axs[idx],
                               alpha=0.1)
    axs[idx].set_title(f'{cluster}', size=15)
    axs[idx].set_xlabel('')
    axs[idx].set_ylabel('')
    axs[idx].set_ylim(0, 38)
    idx+= 1
fig.suptitle('Shot Selection by Group', y=1.05, fontsize=15)
plt.tight_layout()
plt.show()


# # The Skilled Big
# Players in this group tend to be the tallest on the court. They take the second most shots behind the "Shot Taker" group. They are more likely to take 2 point shots over 3-pointers, and they also have high average foul stats. They tend to make the most free throws, as they are often fouled.

# In[ ]:


MPlayerSeasonStats_ = MPlayerSeasonStats_.loc[:,~MPlayerSeasonStats_.columns.duplicated()]
ax = MPlayerSeasonStats_.groupby('Cluster_Name')['HeightFeet'].mean().sort_values()     .plot(kind='barh',
          xlim=(6, 6.8),
          title='Skilled bigs are the tallest of all groups')
ax.patches[4].set_color('orange')
ax.set_xlabel('Player Height (Feet)')
ax.set_ylabel('')
plt.show()


# In[ ]:


cluster = 'Skilled Big'

fig, axs = plt.subplots(1, 2, figsize=(14, 7))
axs = axs.flatten()

idx = 0
playerid_seasons = MPlayerSeasonStats_.loc[MPlayerSeasonStats_['Cluster_Name'] == cluster]['PlayerID_Season'].unique()
Shots = MPlayerEvents.loc[MPlayerEvents['PlayerID_Season'].isin(playerid_seasons)]     .query('EventType == "made2" or EventType == "made3" or EventType == "miss2" or EventType == "miss3"')
axs[idx] = create_ncaa_half_court(ax=axs[idx],
                             three_line='mens',
                             court_color='white',
                             paint_alpha=0,
                             inner_arc=True)
hb1 = axs[idx].hexbin(x=Shots.query('Y_ != 0')['Y_half_'],
                 y=Shots.query('Y_ != 0')['X_half_'],
                 gridsize=20, bins='log', cmap='inferno')
axs[idx].set_title(f'Shot Attempts: {cluster}', size=15)

axs[idx].set_ylim(0, 40)
axs[idx].set_xlim(1, 49)
plt.tight_layout()

playerid_seasons = MPlayerSeasonStats_.loc[MPlayerSeasonStats_['Cluster_Name'] == cluster]['PlayerID_Season'].unique()
Made = MPlayerEvents.loc[MPlayerEvents['PlayerID_Season'].isin(playerid_seasons)]     .query('EventType == "made2" or EventType == "made3"')
axs[idx+1] = create_ncaa_half_court(ax=axs[idx+1],
                             three_line='mens',
                             court_color='black',
                                  lines_color='white',
                             paint_alpha=0,
                             inner_arc=True)
#     hb1 = axs[idx].plot(x=Turnover.query('Y_ != 0')['Y_half_'],
#                      y=Turnover.query('Y_ != 0')['X_half_'],
#                      gridsize=10, bins='log', cmap='inferno')
Made.query('Y_ != 0').plot(x='Y_half_',
                 y='X_half_',
              kind='scatter', ax=axs[idx+1], alpha=0.3)
axs[idx+1].set_title(f'Made Shots: {cluster}', size=15)

axs[idx+1].set_ylim(0, 40)
axs[idx+1].set_xlim(1, 49)
axs[idx+1].set_xlabel('')
axs[idx+1].set_ylabel('')
plt.show()


# The Skilled Bigs tend to be highly ranked coming out of high school. The number 1 high school recruit in 2014, 2015, 2016, and 2017 all fall into this category.

# ## Notable "Skilled Bigs"
# | FirstName    | LastName             | TeamName   | Season | Position | RSCI Top 100 |
# |--------------|----------------------|------------|--------|----------|--------------|
# | Jahlil       | Okafor               | Duke       | 2015   | Center   | 1 (2014)     |
# | Marvin       | Bagley III           | Duke       | 2018   | Forward  | 1 (2017)     |
# | Josh         | Jackson              | Kansas     | 2017   | Guard    | 1 (2016)     |
# | Ben          | Simmons              | LSU        | 2016   | Forward  | 1 (2015)     |
# | Jaylen       | Brown                | California | 2016   | Forward  | 3 (2015)     |
# | Deandre      | Ayton                | Arizona    | 2018   | Forward  | 3 (2017)     |
# | Mohamed      | Bamba                | Texas      | 2018   | Forward  | 4 (2017)     |
# | Zion         | Williamson           | Duke       | 2019   | Forward  | 4 (2018)     |
# | Ivan         | Rabb                 | California | 2016   | Forward  | 5 (2015)     |
# | Karl-Anthony | Towns                | Kentucky   | 2015   | Forward  | 5 (2014)     |
# 
# (Expand next cell to show data source)

# In[ ]:


Cluster_Name = 'Skilled Big'
group_players = MPlayerSeasonStats_.query('Cluster_Name == @Cluster_Name')['PlayerID_Season'].unique()
display(MPlayerSeasonStats.loc[MPlayerSeasonStats['PlayerID_Season']                        .isin(group_players)].sort_values('RSCI Top 100 Number')     [['FirstName','LastName','TeamName',
      'Season','Position','RSCI Top 100']].drop_duplicates(subset=['FirstName','LastName']).head(10).style.hide_index())


# # The Shot Taker
# These players take more shots than any other group by a good margin. Unlike the 3 point specalists, they take a balance of both 2 and 3-pointers. Interestingly, while this player takes more shots than all other groups, they are typically still in the middle of the pack with regard to their shooting percentage.

# In[ ]:


fig, (ax, ax1) = plt.subplots(1, 2, figsize=(15, 4))
fig.suptitle('High shot count but average shooting %', fontsize=15, y=1.03)
MPlayerSeasonStats[['PlayerID_Season','FGA']].merge(MPlayerSeasonStats_[['PlayerID_Season','Cluster_Name']])     .groupby('Cluster_Name')['FGA'].mean()     .sort_values()     .plot(kind='barh', rot=0, ax=ax)
# ax.set_title('Almost 2x more assists than the next group', fontsize=15)
ax.set_xlabel('Field Goal Attempts Per Game')
ax.set_ylabel('')
ax.patches[4].set_color('orange')


MPlayerSeasonStats[['PlayerID_Season','FG%']].merge(MPlayerSeasonStats_[['PlayerID_Season','Cluster_Name']])     .groupby('Cluster_Name')['FG%'].mean()     .sort_values()     .plot(kind='barh', rot=0, ax=ax1)
# ax1.set_title('Almost 2x more assists than the next group', fontsize=15)
ax1.set_xlabel('Field Goal %')
ax1.set_ylabel('')
ax1.patches[2].set_color('orange')
plt.tight_layout()
plt.show()


# In[ ]:


cluster = 'Shot Taker'

fig, axs = plt.subplots(1, 2, figsize=(14, 7))
axs = axs.flatten()

idx = 0
playerid_seasons = MPlayerSeasonStats_.loc[MPlayerSeasonStats_['Cluster_Name'] == cluster]['PlayerID_Season'].unique()
Shots = MPlayerEvents.loc[MPlayerEvents['PlayerID_Season'].isin(playerid_seasons)]     .query('EventType == "made2" or EventType == "made3" or EventType == "miss2" or EventType == "miss3"')
axs[idx] = create_ncaa_half_court(ax=axs[idx],
                             three_line='mens',
                             court_color='white',
                             paint_alpha=0,
                             inner_arc=True)
hb1 = axs[idx].hexbin(x=Shots.query('Y_ != 0')['Y_half_'],
                 y=Shots.query('Y_ != 0')['X_half_'],
                 gridsize=20, bins='log', cmap='inferno')
axs[idx].set_title(f'Shot Attempts: {cluster}', size=15)

axs[idx].set_ylim(0, 40)
axs[idx].set_xlim(1, 49)
plt.tight_layout()

playerid_seasons = MPlayerSeasonStats_.loc[MPlayerSeasonStats_['Cluster_Name'] == cluster]['PlayerID_Season'].unique()
Made = MPlayerEvents.loc[MPlayerEvents['PlayerID_Season'].isin(playerid_seasons)]     .query('EventType == "made2" or EventType == "made3"')
axs[idx+1] = create_ncaa_half_court(ax=axs[idx+1],
                             three_line='mens',
                             court_color='black',
                                  lines_color='white',
                             paint_alpha=0,
                             inner_arc=True)
#     hb1 = axs[idx].plot(x=Turnover.query('Y_ != 0')['Y_half_'],
#                      y=Turnover.query('Y_ != 0')['X_half_'],
#                      gridsize=10, bins='log', cmap='inferno')
Made.query('Y_ != 0').plot(x='Y_half_',
                 y='X_half_',
              kind='scatter', ax=axs[idx+1], alpha=0.3)
axs[idx+1].set_title(f'Made Shots: {cluster}', size=15)

axs[idx+1].set_ylim(0, 40)
axs[idx+1].set_xlim(1, 49)
axs[idx+1].set_xlabel('')
axs[idx+1].set_ylabel('')
plt.show()


# When looking at some highly ranked players in this group, I notice that many of them end up landing at Duke. Maybe Mike Krzyzewski prefers these types of players? 

# ## Notable "Shot Takers"
# | FirstName | LastName | TeamName   | Season | Position | RSCI Top 100 |
# |-----------|----------|------------|--------|----------|--------------|
# | RJ        | Barrett  | Duke       | 2019   | Forward  | 1 (2018)     |
# | Cam       | Reddish  | Duke       | 2019   | Forward  | 2 (2018)     |
# | Jayson    | Tatum    | Duke       | 2017   | Forward  | 3 (2016)     |
# | Stanley   | Johnson  | Arizona    | 2015   | Forward  | 3 (2014)     |
# | Brandon   | Ingram   | Duke       | 2016   | Guard    | 4 (2015)     |
# | Romeo     | Langford | Indiana    | 2019   | Guard    | 5 (2018)     |
# | Markelle  | Fultz    | Washington | 2017   | Guard    | 5 (2016)     |
# | Collin    | Sexton   | Alabama    | 2018   | Guard    | 6 (2017)     |
# | De'Aaron  | Fox      | Kentucky   | 2017   | Guard    | 6 (2016)     |
# | Kelly     | Oubre    | Kansas     | 2015   | Guard    | 8 (2014)     |

# In[ ]:


Cluster_Name = 'Shot Taker'
group_players = MPlayerSeasonStats_.query('Cluster_Name == @Cluster_Name')['PlayerID_Season'].unique()
display(MPlayerSeasonStats.loc[MPlayerSeasonStats['PlayerID_Season']                        .isin(group_players)].sort_values('RSCI Top 100 Number')     [['FirstName','LastName','TeamName',
      'Season','Position','RSCI Top 100']] \
        .drop_duplicates(subset=['FirstName','LastName']).head(10).style.hide_index())


# # The Ball Distributor
# 
# Ball Distributors are unique in that they have many more assists than the other four groups. They tend to have lower average stats for defensive actions like rebounds and blocks.

# In[ ]:


ax = MPlayerSeasonStats[['PlayerID_Season','AST']]     .merge(MPlayerSeasonStats_[['PlayerID_Season','Cluster_Name']])     .groupby('Cluster_Name')['AST'].mean()     .sort_values()     .plot(kind='bar', rot=0)
ax.set_title('1.5x more assists than the next group', fontsize=15)
ax.set_ylabel('Assists Per Game')
ax.set_xlabel('')
ax.patches[4].set_color('orange')
plt.show()


# In[ ]:


cluster = 'Ball Distributor'

fig, axs = plt.subplots(1, 2, figsize=(14, 7))
axs = axs.flatten()

idx = 0
playerid_seasons = MPlayerSeasonStats_.loc[MPlayerSeasonStats_['Cluster_Name'] == cluster]['PlayerID_Season'].unique()
Shots = MPlayerEvents.loc[MPlayerEvents['PlayerID_Season'].isin(playerid_seasons)]     .query('EventType == "made2" or EventType == "made3" or EventType == "miss2" or EventType == "miss3"')
axs[idx] = create_ncaa_half_court(ax=axs[idx],
                             three_line='mens',
                             court_color='white',
                             paint_alpha=0,
                             inner_arc=True)
hb1 = axs[idx].hexbin(x=Shots.query('Y_ != 0')['Y_half_'],
                 y=Shots.query('Y_ != 0')['X_half_'],
                 gridsize=20, bins='log', cmap='inferno')
axs[idx].set_title(f'Shot Attempts: {cluster}', size=15)

axs[idx].set_ylim(0, 40)
axs[idx].set_xlim(1, 49)
plt.tight_layout()

playerid_seasons = MPlayerSeasonStats_.loc[MPlayerSeasonStats_['Cluster_Name'] == cluster]['PlayerID_Season'].unique()
Made = MPlayerEvents.loc[MPlayerEvents['PlayerID_Season'].isin(playerid_seasons)]     .query('EventType == "made2" or EventType == "made3"')
axs[idx+1] = create_ncaa_half_court(ax=axs[idx+1],
                             three_line='mens',
                             court_color='black',
                                  lines_color='white',
                             paint_alpha=0,
                             inner_arc=True)
#     hb1 = axs[idx].plot(x=Turnover.query('Y_ != 0')['Y_half_'],
#                      y=Turnover.query('Y_ != 0')['X_half_'],
#                      gridsize=10, bins='log', cmap='inferno')
Made.query('Y_ != 0').plot(x='Y_half_',
                 y='X_half_',
              kind='scatter', ax=axs[idx+1], alpha=0.3)
axs[idx+1].set_title(f'Made Shots: {cluster}', size=15)

axs[idx+1].set_ylim(0, 40)
axs[idx+1].set_xlim(1, 49)
axs[idx+1].set_xlabel('')
axs[idx+1].set_ylabel('')
plt.show()


# ## Notable "Ball Distributors"
# 
# | FirstName | LastName   | TeamName | Season | Position | RSCI Top 100 |
# |-----------|------------|----------|--------|----------|--------------|
# | Lonzo     | Ball       | UCLA     | 2017   | Guard    | 3 (2016)     |
# | Trevon    | Duval      | Duke     | 2018   | Guard    | 5 (2017)     |
# | Andrew    | Harrison   | Kentucky | 2015   | Guard    | 5 (2013)     |
# | Tyus      | Jones      | Duke     | 2015   | Guard    | 7 (2014)     |
# | Isaiah    | Briscoe    | Kentucky | 2016   | Guard    | 11 (2015)    |
# | Kasey     | Hill       | Florida  | 2015   | Guard    | 11 (2013)    |
# | Troy      | Brown      | Oregon   | 2018   | Forward  | 12 (2017)    |
# | Derryck   | Thornton   | USC      | 2019   | Guard    | 13 (2015)    |
# | Wayne     | Selden Jr. | Kansas   | 2015   | Guard    | 13 (2013)    |
# | Tre       | Jones      | Duke     | 2019   | Guard    | 13 (2018)    |

# In[ ]:


Cluster_Name = 'Ball Distributor'
group_players = MPlayerSeasonStats_     .query('Cluster_Name == @Cluster_Name')['PlayerID_Season'].unique()
display(MPlayerSeasonStats.loc[MPlayerSeasonStats['PlayerID_Season']                        .isin(group_players)].sort_values('RSCI Top 100 Number')     [['FirstName','LastName','TeamName',
      'Season','Position','RSCI Top 100']] \
            .drop_duplicates(subset=['FirstName','LastName']).head(10).style.hide_index())


# # The Three Point Specialist
# 
# The name says it all. These players love to take three point shots. One of the most notable characteristics of a "Three Point Specialist" is that he attempts far fewer 2-pointers than does a "Shot Taker".

# In[ ]:


fig, (ax, ax1) = plt.subplots(1, 2, figsize=(15, 4))
fig.suptitle('Large disparity between 3 and 2 point shots', fontsize=15, y=1.03)
MPlayerSeasonStats[['PlayerID_Season','3P']].merge(MPlayerSeasonStats_[['PlayerID_Season','Cluster_Name']])     .groupby('Cluster_Name')['3P'].mean()     .sort_values()     .plot(kind='barh', rot=0, ax=ax)
# ax.set_title('Almost 2x more assists than the next group', fontsize=15)
ax.set_xlabel('3 Point Shots Per Game')
ax.set_ylabel('')
ax.patches[4].set_color('orange')


MPlayerSeasonStats[['PlayerID_Season','2P']].merge(MPlayerSeasonStats_[['PlayerID_Season','Cluster_Name']])     .groupby('Cluster_Name')['2P'].mean()     .sort_values()     .plot(kind='barh', rot=0, ax=ax1)
# ax1.set_title('Almost 2x more assists than the next group', fontsize=15)
ax1.set_xlabel('2 Point Shots Per Game')
ax1.set_ylabel('')
ax1.patches[0].set_color('orange')
plt.tight_layout()
plt.show()


# In[ ]:


cluster = '3 Point Specialist'

fig, axs = plt.subplots(1, 2, figsize=(14, 7))
axs = axs.flatten()

idx = 0
playerid_seasons = MPlayerSeasonStats_.loc[MPlayerSeasonStats_['Cluster_Name'] == cluster]['PlayerID_Season'].unique()
Shots = MPlayerEvents.loc[MPlayerEvents['PlayerID_Season'].isin(playerid_seasons)]     .query('EventType == "made2" or EventType == "made3" or EventType == "miss2" or EventType == "miss3"')
axs[idx] = create_ncaa_half_court(ax=axs[idx],
                             three_line='mens',
                             court_color='white',
                             paint_alpha=0,
                             inner_arc=True)
hb1 = axs[idx].hexbin(x=Shots.query('Y_ != 0')['Y_half_'],
                 y=Shots.query('Y_ != 0')['X_half_'],
                 gridsize=20, bins='log', cmap='inferno')
axs[idx].set_title(f'Shot Attempts: {cluster}', size=15)

axs[idx].set_ylim(0, 40)
axs[idx].set_xlim(1, 49)
plt.tight_layout()

playerid_seasons = MPlayerSeasonStats_.loc[MPlayerSeasonStats_['Cluster_Name'] == cluster]['PlayerID_Season'].unique()
Made = MPlayerEvents.loc[MPlayerEvents['PlayerID_Season'].isin(playerid_seasons)]     .query('EventType == "made2" or EventType == "made3"')
axs[idx+1] = create_ncaa_half_court(ax=axs[idx+1],
                             three_line='mens',
                             court_color='black',
                                  lines_color='white',
                             paint_alpha=0,
                             inner_arc=True)
#     hb1 = axs[idx].plot(x=Turnover.query('Y_ != 0')['Y_half_'],
#                      y=Turnover.query('Y_ != 0')['X_half_'],
#                      gridsize=10, bins='log', cmap='inferno')
Made.query('Y_ != 0').plot(x='Y_half_',
                 y='X_half_',
              kind='scatter', ax=axs[idx+1], alpha=0.3)
axs[idx+1].set_title(f'Made Shots: {cluster}', size=15)

axs[idx+1].set_ylim(0, 40)
axs[idx+1].set_xlim(1, 49)
axs[idx+1].set_xlabel('')
axs[idx+1].set_ylabel('')
plt.show()


# ## Notable "3 Point Specialists"
# | FirstName | LastName  | TeamName    | Season | Position | RSCI Top 100 |
# |-----------|-----------|-------------|--------|----------|--------------|
# | Aaron     | Harrison  | Kentucky    | 2015   | Guard    | 6 (2013)     |
# | Malik     | Newman    | Kansas      | 2018   | Guard    | 8 (2015)     |
# | Quentin   | Grimes    | Kansas      | 2019   | Guard    | 8 (2018)     |
# | Rasheed   | Sulaimon  | Maryland    | 2016   | Guard    | 12 (2012)    |
# | Gary      | Trent Jr. | Duke        | 2018   | Guard    | 14 (2017)    |
# | Antonio   | Blakeney  | LSU         | 2016   | Guard    | 15 (2015)    |
# | Rodney    | Purvis    | Connecticut | 2017   | Guard    | 17 (2012)    |
# | Sam       | Dekker    | Wisconsin   | 2015   | Forward  | 19 (2012)    |
# | Jabari    | Bird      | California  | 2016   | Guard    | 20 (2013)    |
# | Joshua    | Langford  | Michigan St | 2017   | Guard    | 20 (2016)    |

# In[ ]:


Cluster_Name = '3 Point Specialist'
group_players = MPlayerSeasonStats_.query('Cluster_Name == @Cluster_Name')['PlayerID_Season'].unique()
display(MPlayerSeasonStats.loc[MPlayerSeasonStats['PlayerID_Season']                        .isin(group_players)].sort_values('RSCI Top 100 Number')     [['FirstName','LastName','TeamName',
      'Season','Position','RSCI Top 100']].drop_duplicates(subset=['FirstName','LastName']).head(10).style.hide_index())


# # The Paint Dominator
# You won't see these players putting up a lot of 3-pointers, but what they do well is control the paint. They come in right behing the "Skilled Bigs" in rebounds and blocks.

# In[ ]:


fig, (ax, ax1) = plt.subplots(1, 2, figsize=(15, 4))
fig.suptitle('Similar to the Skilled Big - with less shooting', fontsize=15, y=1.03)
MPlayerSeasonStats[['PlayerID_Season','BLK']].merge(MPlayerSeasonStats_[['PlayerID_Season','Cluster_Name']])     .groupby('Cluster_Name')['BLK'].mean()     .sort_values()     .plot(kind='barh', rot=0, ax=ax)
ax.set_xlabel('Blocks Per Game')
ax.set_ylabel('')
ax.patches[3].set_color('orange')


MPlayerSeasonStats[['PlayerID_Season','2P']].merge(MPlayerSeasonStats_[['PlayerID_Season','Cluster_Name']])     .groupby('Cluster_Name')['2P'].mean()     .sort_values()     .plot(kind='barh', rot=0, ax=ax1)
ax1.set_xlabel('2 Point Shots Per Game')
ax1.set_ylabel('')
ax1.patches[2].set_color('orange')
plt.tight_layout()
plt.show()


# In[ ]:


cluster = 'Paint Dominator'

fig, axs = plt.subplots(1, 2, figsize=(14, 7))
axs = axs.flatten()

idx = 0
playerid_seasons = MPlayerSeasonStats_.loc[MPlayerSeasonStats_['Cluster_Name'] == cluster]['PlayerID_Season'].unique()
Shots = MPlayerEvents.loc[MPlayerEvents['PlayerID_Season'].isin(playerid_seasons)]     .query('EventType == "made2" or EventType == "made3" or EventType == "miss2" or EventType == "miss3"')
axs[idx] = create_ncaa_half_court(ax=axs[idx],
                             three_line='mens',
                             court_color='white',
                             paint_alpha=0,
                             inner_arc=True)
hb1 = axs[idx].hexbin(x=Shots.query('Y_ != 0')['Y_half_'],
                 y=Shots.query('Y_ != 0')['X_half_'],
                 gridsize=20, bins='log', cmap='inferno')
axs[idx].set_title(f'Shot Attempts: {cluster}', size=15)

axs[idx].set_ylim(0, 40)
axs[idx].set_xlim(1, 49)
plt.tight_layout()

playerid_seasons = MPlayerSeasonStats_.loc[MPlayerSeasonStats_['Cluster_Name'] == cluster]['PlayerID_Season'].unique()
Made = MPlayerEvents.loc[MPlayerEvents['PlayerID_Season'].isin(playerid_seasons)]     .query('EventType == "made2" or EventType == "made3"')
axs[idx+1] = create_ncaa_half_court(ax=axs[idx+1],
                             three_line='mens',
                             court_color='black',
                                  lines_color='white',
                             paint_alpha=0,
                             inner_arc=True)
Made.query('Y_ != 0').plot(x='Y_half_',
                 y='X_half_',
              kind='scatter', ax=axs[idx+1], alpha=0.3)
axs[idx+1].set_title(f'Made Shots: {cluster}', size=15)

axs[idx+1].set_ylim(0, 40)
axs[idx+1].set_xlim(1, 49)
axs[idx+1].set_xlabel('')
axs[idx+1].set_ylabel('')
plt.show()


# ## Notable "Paint Dominators"
# 
# | FirstName | LastName   | TeamName  | Season | Position | RSCI Top 100 |
# |-----------|------------|-----------|--------|----------|--------------|
# | Kaleb     | Tarczewski | Arizona   | 2015   | Center   | 7 (2012)     |
# | Alex      | Poythress  | Kentucky  | 2016   | Forward  | 8 (2012)     |
# | Marques   | Bolden     | Duke      | 2019   | Center   | 11 (2016)    |
# | Kevon     | Looney     | UCLA      | 2015   | Forward  | 11 (2014)    |
# | Trey      | Lyles      | Kentucky  | 2015   | Forward  | 12 (2014)    |
# | Wenyen    | Gabriel    | Kentucky  | 2017   | Forward  | 13 (2016)    |
# | Omari     | Spellman   | Villanova | 2018   | Forward  | 17 (2016)    |
# | Dajuan    | Coleman    | Syracuse  | 2016   | Forward  | 18 (2012)    |
# | Nick      | Richards   | Kentucky  | 2018   | Forward  | 18 (2017)    |
# | Devin     | Robinson   | Florida   | 2017   | Forward  | 20 (2014)    |

# In[ ]:


Cluster_Name = 'Paint Dominator'
group_players = MPlayerSeasonStats_.query('Cluster_Name == @Cluster_Name')['PlayerID_Season'].unique()
display(MPlayerSeasonStats.loc[MPlayerSeasonStats['PlayerID_Season']                        .isin(group_players)].sort_values('RSCI Top 100 Number')     [['FirstName','LastName','TeamName',
      'Season','Position','RSCI Top 100']] \
        .drop_duplicates(subset=['FirstName','LastName']).head(10).style.hide_index())


# # Schools and Player Types
# Taking these clusters, lets take a look at some of the top basketball schools and which types of players their rosters have consisted of in recent seasons. The data below shows the numbers per teams since the 2015 season. It looks like certain schools may have a preference for specific types of players.

# In[ ]:


cmap=sns.light_palette((210, 90, 60), input="husl", as_cmap=True)
TOP_BBALL_SCHOOLS = ['Kentucky','North Carolina','Duke','Villanova','Kansas','UCLA','Connecticut','Indiana',
               'Louisville','Virginia','Florida']
d = MPlayerSeasonStats_.query('TeamName in @TOP_BBALL_SCHOOLS')     .groupby(['TeamName','Cluster_Name'])['Position'].count()     .unstack('TeamName')    .T.sort_values(['3 Point Specialist','Ball Distributor','Paint Dominator','Shot Taker','Skilled Big']).T
d.index.name = ''
d.style.background_gradient(cmap='BuGn', axis=0)


# ## What Types of Rosters Fare well in March Madness?
# It's also interesting to see if there are any differences in rosters amongst winning and losing tournament teams. To do this I simply count the number of players of each position for winning and losing teams and plot the difference. The results show that winning teams have more "Skilled Big" and "Ball Distributor" players. Losing teams however have much more "Shot Taker" players. This makes sense because shot takers don't tend to have high shooting percentage.

# In[ ]:


# Add Team ID
MTourneyResults = pd.read_csv('../input/march-madness-analytics-2020/MDataFiles_Stage2/MNCAATourneyCompactResults.csv')
MPlayerSeasonStats_ = MPlayerSeasonStats_.merge(MTeams[['TeamID','TeamName']], how='left')
TourneyWinTeams = MPlayerSeasonStats_[['TeamID','Season','Cluster_Name']]     .merge(MTourneyResults, left_on=['Season','TeamID'], right_on=['Season','WTeamID'])
TourneyLossTeams = MPlayerSeasonStats_[['TeamID','Season','Cluster_Name']]     .merge(MTourneyResults, left_on=['Season','TeamID'], right_on=['Season','LTeamID'])
TourneyTeams = TourneyWinTeams.groupby('Cluster_Name')[['TeamID']].count()     .rename(columns={'TeamID':'Winning Team Rosters'})
TourneyTeams['Losing Team Rosters'] = TourneyLossTeams.groupby('Cluster_Name')[['TeamID']].count().values
TourneyTeams['Diff'] = TourneyTeams['Winning Team Rosters'] - TourneyTeams['Losing Team Rosters']
TourneyTeams['Winning Team Makeup'] = TourneyTeams['Winning Team Rosters'] / TourneyTeams['Winning Team Rosters'].sum()
TourneyTeams['Losing Team Makeup'] = TourneyTeams['Losing Team Rosters'] / TourneyTeams['Losing Team Rosters'].sum()
TourneyTeams['Makeup Diff'] = TourneyTeams['Winning Team Makeup'] - TourneyTeams['Losing Team Makeup']
ax = TourneyTeams['Makeup Diff'].plot(figsize=(15, 5), kind='barh', rot=0, title='Tourney Winning vs. Losing Team Roster Differential')
ax.axvline(0, color='black')
ax.grid(False)
ax.set_xlabel('<----- Losing Tourney Teams have more | Winning Tourney Teams have more ---->')
ax.set_ylabel('')
ax.patches[0].set_color('lightcoral')
ax.patches[1].set_color('seagreen')
ax.patches[2].set_color('lightcoral')
ax.patches[3].set_color('lightcoral')
ax.patches[4].set_color('seagreen')
plt.show()


# # Interactive 3D Visualization of Clusters
# The last visualization is a fun way to explore these clusters in 3D: Each dot in this plot represents a player/season. I used an algorithm called tSNE (t-distributed Stochastic Neighbor Embedding) - to reduce the data into three dimentions. You can use your mouse to rotate and zoom into the plot.  Hover over a dot to see the player's name and team.

# In[ ]:


sns.set_style("whitegrid")

tsne = TSNE(n_components=3, verbose=0, perplexity=2, n_iter=250, random_state=529)
X_tsne = MPlayerSeasonStats_[FEATURES + ['Data_Position']].values

tsne_results = tsne.fit_transform(X_tsne)
MPlayerSeasonStats_['tsne-3d-one'] = tsne_results[:,0]
MPlayerSeasonStats_['tsne-3d-two'] = tsne_results[:,1]
MPlayerSeasonStats_['tsne-3d-three'] = tsne_results[:,2]


# In[ ]:


fig = px.scatter_3d(MPlayerSeasonStats_.merge(MPlayerSeasonStats[['PlayerID_Season','Height']]),
                    x='tsne-3d-one',
                    y='tsne-3d-two',
                    z='tsne-3d-three',
                    size_max=5,
                    color='Cluster_Name',
                    hover_data=["FirstName", "LastName","TeamName","Height"],
                    opacity=0.7)
# fig.update_layout(hovermode="FirstName")
fig.show()


# # Thanks!
# Thanks for taking the time to read my report. I hope you had as much fun reading it as I did making it!

# # References
# [1] https://www.espn.com/mens-college-basketball/story/_/id/9799167/traditional-positions-being-eliminated-college-basketball
# 
# Data Prep Notebooks:
# - https://www.kaggle.com/robikscube/march-madness-2020-sports-reference-com-scrape
# - https://www.kaggle.com/robikscube/rob-s-march-madness-2020-data-prep
