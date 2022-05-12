#!/usr/bin/env python
# coding: utf-8

# <a id='0'></a>
# ## <p style="background-color:#fdb913; font-family:Computer Modern;src: url('http://mirrors.ctan.org/fonts/cm-unicode/fonts/otf/cmunss.otf'); font-size:100%; text-align:center"> Time Series Anomaly/Outliers Detection on IPL cricket game data</p>
# `
# <div align="center">
#     <img src="https://user-images.githubusercontent.com/48846576/166569679-1faac713-2775-4cc8-8ab2-36509f5e5d6c.jpg"> 
# </div>

# <h2>Table of contents</h2>
# 
# <a id='0'></a>
# 
# * [Overview](#overview)
# * [Methodology](#method)        
# * [Isolation Forest](#iforest)        
# * [Load Data](#data)
# * [Sample IPL Game Time Series](#sample_ts)
# * [Outlier Detection using PyCaret](#pycaret)
# * [Outliers Detection Excercise for all IPL 2022 Games](#outlier-detect)
#     * [Chennai Super Kings](#csk)
#     * [Delhi Capitals](#dc)
#     * [Gujarat Titans](#gt)
#     * [Kolkata Knight Riders](#kkr)
#     * [Lucknow Super Giants](#lsg)
#     * [Mumbai Indians](#mi)
#     * [Punjab Kings](#pbks)
#     * [Rajasthan Royals](#rr)
#     * [Royal Challengers Bangalore](#rcb)
#     * [Sunrisers Hyderabad](#srh)
# * [Conclusion](#conclusion)    

# <a id='overview'></a>
# # Overview
# [back to top](#0)
# 
# This notebook performs anomaly/outliers detection on Indian Premier League T20 (IPL T20) Cricket Matches. A cricket game can be viewed as a time series data. There are two time series in a given cricket game (one for each team). Anomaly detection is a process to identify data points that deviate from the normal tendency of the dataset. The anomalous data in the context of a cricket game represent critical events in the game such as a boundary scored (4 or 6 runs), catch dropped, misfielding, no balls/wide that result in extra delivery, DRS reviews, etc.
# 
# <a id='method'></a>
# # Methodology
# [back to top](#0)
# 
# <div align="left">
#     <img src="https://user-images.githubusercontent.com/48846576/166571986-74a8b0b2-fcc0-413b-8a8d-43345cc40ee6.jpg"> 
# </div>
# 
# * Each T20 cricket game has two innings. The game starts with coin toss where the team that wins the toss chooses to bat first or bat second (bowl first)
# * A typical game will have around 120 deliveries (divided into 20 overs with each over consists of 6 legal balls/deliveries that a bowler has to bowl). The number will go up depending on the extra deliveries that bowlers have to bowl due to  illegal deliveries judged by the umpiers. The number may come down if all the batsman in the batting team get out before the completion of their innings
# * Each delivery/ball bowled can be viewed as a unit of time with the result of the delivery as the outcome. Hence every delivery will have many attributes associated to it like the batsman, bowler, delivery number, over number, innings, venue, legal delivery or not and many others. There will also be commentary associated to each delivery which is a textual description of the play.
# * The goal is to pre-process this time series data, create features, identify clusters from commentary text and feed the data to PyCaret's Isolation Forest algorithm and Visualize the outcome. 
# * This is a unsupervised algorithm where there are no labeld outliers available. Hence the important parameter is the contamination parameter to the Isoaltion Forest model which indicates the % of outliers that we expect from the given data
# 

# <a id='iforest'></a>
# # Isolation Forest
# [back to top](#0)
# 
# 
# Isolation forest is an ensemble learning algorithm, effectively used in time series anomaly detection. It can be used a unsupervised setting which does not rely on prior labeling of the outliers. Its based on the fact that the anomalous data are "few and different" data points in the given dataset. In this algorithm a random sub-sample of the data is obtained from the original dataset and a decision tree is built based on random cuts in values of randomly selected features of the data. The samples that take many branches to reach the leaf node are less likely to be outliers while the samples that take shorter branches may indicate outliers.
# 
# The figure below shows how the cuts are done by the Isolation Forest algorithm. Fig (a) shows that the red point (anomalous data) is isolated in three random cuts where as Fig (b) shows the same process for the normal data point. Since the normal point is deep inside teh data it takes many cuts to isolate that data point. The numbers on the lines represnent the order of tree branching process.
# 
# <div align="left">
#     <img src="https://user-images.githubusercontent.com/48846576/166577211-396c5e0f-63e5-4f17-884f-18183e78ee6b.png"><br> 
#     Source: <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8888179">IEEE - Extended Isolation Forest</a>
# </div>
# 

# References
# 
# * [Time Series Anomaly Detection with PyCaret](https://towardsdatascience.com/time-series-anomaly-detection-with-pycaret-706a6e2b2427)
# * [Anomaly detection using Isolation Forest – A Complete Guide](https://www.analyticsvidhya.com/blog/2021/07/anomaly-detection-using-isolation-forest-a-complete-guide/)
# * [IEEE - Extended Isolation Forest](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8888179)
# 

# In[ ]:


import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)
import plotly_express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.io as pio
from plotly.subplots import make_subplots
# setting default template to plotly_white for all visualizations
pio.templates.default = "plotly_dark"
get_ipython().run_line_magic('matplotlib', 'inline')
import gc

from colorama import Fore, Back, Style

y_ = Fore.YELLOW
r_ = Fore.RED
g_ = Fore.GREEN
b_ = Fore.BLUE
m_ = Fore.MAGENTA
c_ = Fore.CYAN
res = Style.RESET_ALL

import warnings
warnings.filterwarnings('ignore')
import folium
import matplotlib.dates as mdates
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText
import matplotlib
YELLOVE = '#fdb913'   


# ***Install pycaret***

# In[ ]:


get_ipython().system('pip install pycaret')


# In[ ]:


#Define custom plotly template
pio.templates["my_plotly_template"] = go.layout.Template(
    # LAYOUT
    layout = {'annotationdefaults': {'arrowcolor': '#f2f5fa', 'arrowhead': 0, 'arrowwidth': 1},
               'autotypenumbers': 'strict',
               'coloraxis': {'colorbar': {'outlinewidth': 0, 'ticks': ''}},
               'colorscale': {'diverging': [[0, '#8e0152'], [0.1, '#c51b7d'],
                                            [0.2, '#de77ae'], [0.3, '#f1b6da'],
                                            [0.4, '#fde0ef'], [0.5, '#f7f7f7'],
                                            [0.6, '#e6f5d0'], [0.7, '#b8e186'],
                                            [0.8, '#7fbc41'], [0.9, '#4d9221'], [1,
                                            '#276419']],
                              'sequential': [[0.0, '#0d0887'],
                                             [0.1111111111111111, '#46039f'],
                                             [0.2222222222222222, '#7201a8'],
                                             [0.3333333333333333, '#9c179e'],
                                             [0.4444444444444444, '#bd3786'],
                                             [0.5555555555555556, '#d8576b'],
                                             [0.6666666666666666, '#ed7953'],
                                             [0.7777777777777778, '#fb9f3a'],
                                             [0.8888888888888888, '#fdca26'], [1.0,
                                             '#f0f921']],
                              'sequentialminus': [[0.0, '#0d0887'],
                                                  [0.1111111111111111, '#46039f'],
                                                  [0.2222222222222222, '#7201a8'],
                                                  [0.3333333333333333, '#9c179e'],
                                                  [0.4444444444444444, '#bd3786'],
                                                  [0.5555555555555556, '#d8576b'],
                                                  [0.6666666666666666, '#ed7953'],
                                                  [0.7777777777777778, '#fb9f3a'],
                                                  [0.8888888888888888, '#fdca26'],
                                                  [1.0, '#f0f921']]},
               'colorway': ['#636efa', '#EF553B', '#00cc96', '#ab63fa', '#FFA15A', '#19d3f3',
                            '#FF6692', '#B6E880', '#FF97FF', '#FECB52'],
               'font': {'color': '#f2f5fa'},
               'geo': {'bgcolor': 'rgb(17,17,17)',
                       'lakecolor': 'rgb(17,17,17)',
                       'landcolor': 'rgb(17,17,17)',
                       'showlakes': True,
                       'showland': True,
                       'subunitcolor': '#506784'},
               'hoverlabel': {'align': 'left'},
               'hovermode': 'closest',
               'mapbox': {'style': 'dark'},
               'paper_bgcolor': '#666666',
               'plot_bgcolor': '#666666',
               'polar': {'angularaxis': {'gridcolor': '#506784', 'linecolor': '#506784', 'ticks': ''},
                         'bgcolor': 'rgb(17,17,17)',
                         'radialaxis': {'gridcolor': '#506784', 'linecolor': '#506784', 'ticks': ''}},
               'scene': {'xaxis': {'backgroundcolor': 'rgb(17,17,17)',
                                   'gridcolor': '#506784',
                                   'gridwidth': 2,
                                   'linecolor': '#506784',
                                   'showbackground': True,
                                   'ticks': '',
                                   'zerolinecolor': '#C8D4E3'},
                         'yaxis': {'backgroundcolor': 'rgb(17,17,17)',
                                   'gridcolor': '#506784',
                                   'gridwidth': 2,
                                   'linecolor': '#506784',
                                   'showbackground': True,
                                   'ticks': '',
                                   'zerolinecolor': '#C8D4E3'},
                         'zaxis': {'backgroundcolor': 'rgb(17,17,17)',
                                   'gridcolor': '#506784',
                                   'gridwidth': 2,
                                   'linecolor': '#506784',
                                   'showbackground': True,
                                   'ticks': '',
                                   'zerolinecolor': '#C8D4E3'}},
               'shapedefaults': {'line': {'color': '#f2f5fa'}},
               'sliderdefaults': {'bgcolor': '#C8D4E3', 'bordercolor': 'rgb(17,17,17)', 'borderwidth': 1, 'tickwidth': 0},
               'ternary': {'aaxis': {'gridcolor': '#506784', 'linecolor': '#506784', 'ticks': ''},
                           'baxis': {'gridcolor': '#506784', 'linecolor': '#506784', 'ticks': ''},
                           'bgcolor': 'rgb(17,17,17)',
                           'caxis': {'gridcolor': '#506784', 'linecolor': '#506784', 'ticks': ''}},
               'title': {'x': 0.05},
               'updatemenudefaults': {'bgcolor': '#506784', 'borderwidth': 0},
               'xaxis': {'automargin': True,
                         'gridcolor': '#283442',
                         'linecolor': '#506784',
                         'ticks': '',
                         'title': {'standoff': 15},
                         'zerolinecolor': '#283442',
                         'zerolinewidth': 2},
               'yaxis': {'automargin': True,
                         'gridcolor': '#283442',
                         'linecolor': '#506784',
                         'ticks': '',
                         'title': {'standoff': 15},
                         'zerolinecolor': '#283442',
                         'zerolinewidth': 2}},
    data = {'bar': [{'error_x': {'color': '#f2f5fa'},
                      'error_y': {'color': '#f2f5fa'},
                      'marker': {'line': {'color': 'rgb(17,17,17)', 'width': 0.5},
                                 'pattern': {'fillmode': 'overlay', 'size': 10, 'solidity': 0.2}},
                      'type': 'bar'}],
             'barpolar': [{'marker': {'line': {'color': 'rgb(17,17,17)', 'width': 0.5},
                                      'pattern': {'fillmode': 'overlay', 'size': 10, 'solidity': 0.2}},
                           'type': 'barpolar'}],
             'carpet': [{'aaxis': {'endlinecolor': '#A2B1C6',
                                   'gridcolor': '#506784',
                                   'linecolor': '#506784',
                                   'minorgridcolor': '#506784',
                                   'startlinecolor': '#A2B1C6'},
                         'baxis': {'endlinecolor': '#A2B1C6',
                                   'gridcolor': '#506784',
                                   'linecolor': '#506784',
                                   'minorgridcolor': '#506784',
                                   'startlinecolor': '#A2B1C6'},
                         'type': 'carpet'}],
             'choropleth': [{'colorbar': {'outlinewidth': 0, 'ticks': ''}, 'type': 'choropleth'}],
             'contour': [{'colorbar': {'outlinewidth': 0, 'ticks': ''},
                          'colorscale': [[0.0, '#0d0887'], [0.1111111111111111,
                                         '#46039f'], [0.2222222222222222,
                                         '#7201a8'], [0.3333333333333333,
                                         '#9c179e'], [0.4444444444444444,
                                         '#bd3786'], [0.5555555555555556,
                                         '#d8576b'], [0.6666666666666666,
                                         '#ed7953'], [0.7777777777777778,
                                         '#fb9f3a'], [0.8888888888888888,
                                         '#fdca26'], [1.0, '#f0f921']],
                          'type': 'contour'}],
             'contourcarpet': [{'colorbar': {'outlinewidth': 0, 'ticks': ''}, 'type': 'contourcarpet'}],
             'heatmap': [{'colorbar': {'outlinewidth': 0, 'ticks': ''},
                          'colorscale': [[0.0, '#0d0887'], [0.1111111111111111,
                                         '#46039f'], [0.2222222222222222,
                                         '#7201a8'], [0.3333333333333333,
                                         '#9c179e'], [0.4444444444444444,
                                         '#bd3786'], [0.5555555555555556,
                                         '#d8576b'], [0.6666666666666666,
                                         '#ed7953'], [0.7777777777777778,
                                         '#fb9f3a'], [0.8888888888888888,
                                         '#fdca26'], [1.0, '#f0f921']],
                          'type': 'heatmap'}],
             'heatmapgl': [{'colorbar': {'outlinewidth': 0, 'ticks': ''},
                            'colorscale': [[0.0, '#0d0887'], [0.1111111111111111,
                                           '#46039f'], [0.2222222222222222,
                                           '#7201a8'], [0.3333333333333333,
                                           '#9c179e'], [0.4444444444444444,
                                           '#bd3786'], [0.5555555555555556,
                                           '#d8576b'], [0.6666666666666666,
                                           '#ed7953'], [0.7777777777777778,
                                           '#fb9f3a'], [0.8888888888888888,
                                           '#fdca26'], [1.0, '#f0f921']],
                            'type': 'heatmapgl'}],
             'histogram': [{'marker': {'pattern': {'fillmode': 'overlay', 'size': 10, 'solidity': 0.2}},
                            'type': 'histogram'}],
             'histogram2d': [{'colorbar': {'outlinewidth': 0, 'ticks': ''},
                              'colorscale': [[0.0, '#0d0887'],
                                             [0.1111111111111111, '#46039f'],
                                             [0.2222222222222222, '#7201a8'],
                                             [0.3333333333333333, '#9c179e'],
                                             [0.4444444444444444, '#bd3786'],
                                             [0.5555555555555556, '#d8576b'],
                                             [0.6666666666666666, '#ed7953'],
                                             [0.7777777777777778, '#fb9f3a'],
                                             [0.8888888888888888, '#fdca26'], [1.0,
                                             '#f0f921']],
                              'type': 'histogram2d'}],
             'histogram2dcontour': [{'colorbar': {'outlinewidth': 0, 'ticks': ''},
                                     'colorscale': [[0.0, '#0d0887'],
                                                    [0.1111111111111111,
                                                    '#46039f'],
                                                    [0.2222222222222222,
                                                    '#7201a8'],
                                                    [0.3333333333333333,
                                                    '#9c179e'],
                                                    [0.4444444444444444,
                                                    '#bd3786'],
                                                    [0.5555555555555556,
                                                    '#d8576b'],
                                                    [0.6666666666666666,
                                                    '#ed7953'],
                                                    [0.7777777777777778,
                                                    '#fb9f3a'],
                                                    [0.8888888888888888,
                                                    '#fdca26'], [1.0, '#f0f921']],
                                     'type': 'histogram2dcontour'}],
             'mesh3d': [{'colorbar': {'outlinewidth': 0, 'ticks': ''}, 'type': 'mesh3d'}],
             'parcoords': [{'line': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}, 'type': 'parcoords'}],
             'pie': [{'automargin': True, 'type': 'pie'}],
             'scatter': [{'marker': {'line': {'color': '#283442'}}, 'type': 'scatter'}],
             'scatter3d': [{'line': {'colorbar': {'outlinewidth': 0, 'ticks': ''}},
                            'marker': {'colorbar': {'outlinewidth': 0, 'ticks': ''}},
                            'type': 'scatter3d'}],
             'scattercarpet': [{'marker': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}, 'type': 'scattercarpet'}],
             'scattergeo': [{'marker': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}, 'type': 'scattergeo'}],
             'scattergl': [{'marker': {'line': {'color': '#283442'}}, 'type': 'scattergl'}],
             'scattermapbox': [{'marker': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}, 'type': 'scattermapbox'}],
             'scatterpolar': [{'marker': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}, 'type': 'scatterpolar'}],
             'scatterpolargl': [{'marker': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}, 'type': 'scatterpolargl'}],
             'scatterternary': [{'marker': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}, 'type': 'scatterternary'}],
             'surface': [{'colorbar': {'outlinewidth': 0, 'ticks': ''},
                          'colorscale': [[0.0, '#0d0887'], [0.1111111111111111,
                                         '#46039f'], [0.2222222222222222,
                                         '#7201a8'], [0.3333333333333333,
                                         '#9c179e'], [0.4444444444444444,
                                         '#bd3786'], [0.5555555555555556,
                                         '#d8576b'], [0.6666666666666666,
                                         '#ed7953'], [0.7777777777777778,
                                         '#fb9f3a'], [0.8888888888888888,
                                         '#fdca26'], [1.0, '#f0f921']],
                          'type': 'surface'}],
             'table': [{'cells': {'fill': {'color': '#506784'}, 'line': {'color': 'rgb(17,17,17)'}},
                        'header': {'fill': {'color': '#2a3f5f'}, 'line': {'color': 'rgb(17,17,17)'}},
                        'type': 'table'}]}
)


# <a id='data'></a>
# # Load Data
# [back to top](#0)

# In[ ]:


summary_22 = pd.read_csv('/kaggle/input/indian-premier-league-ipl-all-seasons/2022/season_summary.csv', index_col=None)
details_22 = pd.read_csv('/kaggle/input/indian-premier-league-ipl-all-seasons/2022/season_details.csv', index_col=None)
details_22['match_id'] = details_22['match_id'].astype(int)
details_22['comment_id'] = details_22['comment_id'].astype(int)


# In[ ]:


#Ball by Ball data
details_22[['home_team','away_team','current_innings', 'innings_id','over','ball','runs','batsman1_name','bowler1_name','isBoundary','isWide','isNoball','batsman1_runs', 'batsman1_balls','bowler1_overs','bowler1_runs']].head(20)


# In[ ]:


def get_innings_df(match_id, innings):
    innings_df = details_22.loc[(details_22['match_id'] == match_id) & (details_22['current_innings'] == innings)].reset_index(drop=True).sort_values(by="comment_id").reset_index(drop=True)
    innings_df['text'] = innings_df['text'].fillna('')
    innings_df['preText'] = innings_df['preText'].fillna('')    
    innings_df['postText'] = innings_df['postText'].fillna('')   
    innings_df['wicket_id'] = innings_df['wicket_id'].fillna(0)
    innings_df['wicket_id'] = innings_df['wicket_id'].astype(int)
    return innings_df

def ipl_teams_color_palette(custom_colors):
    customPalette = sns.set_palette(sns.color_palette(custom_colors))
    sns.palplot(sns.color_palette(custom_colors),size=0.8)
    plt.tick_params(axis='both', labelsize=0, length = 0)

team_colors = { 'CSK' : '#fdb913',
'MI' : '#0722ab',
'DC' : '#2561AE',
'SRH' : '#fb653f',
'KKR' : '#391F5C',
'PBKS' : '#dd1212',
'RR' : '#FF4081',
'RCB' : '#F23B14',
'GT' : '#004587',
'LSG' : '#02ccbf'}
ipl_teams_color_palette(team_colors.values())


# In[ ]:


from string import punctuation
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from pycaret.anomaly import *
import textwrap
from plotly.subplots import make_subplots

def draw_time_series(match_id, match_name, toss_won, decision):
    
    COLS = ['home_team','away_team','current_innings', 'innings_id','over','ball','runs','batsman1_name','bowler1_name','isBoundary','isWide','isNoball','batsman1_runs', 'batsman1_balls','bowler1_overs','bowler1_runs', 'text_cluster','isOut']    
    output = {}
    background_color = '#757575'
    match_id = match_id
    innings = match_name.split('v')
    pio.templates.default = "my_plotly_template"
    
    team1 = innings[0].strip()
    team2 = innings[1].strip()
    first_inning = None
    second_inning = None    
    innings = []
    if toss_won == team1 and decision == 'BOWL FIRST':
        first_inning = team2
        second_inning = team1
    elif toss_won == team1 and decision == 'BAT FIRST':    
        first_inning = team1
        second_inning = team2
    elif toss_won == team2 and decision == 'BAT FIRST':    
        first_inning = team2
        second_inning = team1
    else:
        first_inning = team1
        second_inning = team2
        
    innings = [first_inning, second_inning]
    fig = make_subplots(rows=2, cols=1,
                   subplot_titles = innings,
                    shared_xaxes=True,
                       vertical_spacing = 0.05)
    row_count = 1
    for current_innings in innings:
        
        #fig = go.Figure()
        current_innings = current_innings.strip()
        df = get_innings_df(match_id, current_innings)
        df['over'] = df['over'].apply(lambda x: x-1)

        hover_data = df[['current_innings','over','ball','runs','batsman1_name','bowler1_name','batsman1_runs', 'batsman1_balls', 'text','isBoundary', 'isWide', 'isNoball']]
        hover_data['text'] = hover_data['text'].apply(lambda text: textwrap.fill(text, width =25))
        hover_data['text'] = hover_data['text'].apply(lambda text: text.replace("\n","<br>"))
        hover_data = hover_data.to_numpy()

        hover_template = '<i>Innings</i>: %{customdata[0]} <br>'+ '<i>Over</i>: %{customdata[1]}.%{customdata[2]}<br>' + '<i>Runs</i>: %{customdata[3]} <br>'+ '<i>Batsman</i>: %{customdata[4]} <br>'+ '<i>Bowler</i>: %{customdata[5]} <br>'+'<i>Batsman Score</i>: %{customdata[6]}/%{customdata[7]} <br>'+ '<i>Text</i>: %{customdata[8]} <br>' + '<i>Is Boundary?</i>: %{customdata[9]} <br>' + '<i>Is wide?</i>: %{customdata[10]} <br>' + '<i>Is no ball?</i>: %{customdata[11]} <br>' 
        
        fig.append_trace(go.Scatter(x=df.index, 
                                 y=df['runs'], 
                                 mode = 'markers+lines', 
                                 line=dict(color=team_colors[current_innings], width=4), 
                                  name = f"{current_innings} innings", hovertemplate=hover_template, customdata=hover_data), 
                         row=row_count, col=1
                    )
    
        fig.update_layout(title=match_name, height=900)#, width=1200)
        row_count +=1
        #fig.show() 
    fig.update_xaxes(tickfont=dict(size=18), row=1, col=1)
    fig.update_yaxes(tickfont=dict(size=18), row=1, col=1,title='Runs', titlefont=dict(size=18),tick0=0, dtick=1)    
    fig.update_xaxes(tickfont=dict(size=18), row=2, col=1,title='Balls', titlefont=dict(size=18))
    fig.update_yaxes(tickfont=dict(size=18), row=2, col=1,title='Runs', titlefont=dict(size=18),tick0=0, dtick=1)    
        
    fig.show()


# <a id='sample_ts'></a>
# # Sample IPL Game Time Series
# [back to top](#0)

# The chart below represents a single IPL game in a time series format. This game was betwee Chennai Super Kings (CSK) and Mumbai Indians (MI) played on April 21, 2002 at DY Patil Stadium, Mumbai in which CSK won the game by 3 wickets

# In[ ]:


draw_time_series(1304079, "CSK v MI", "CSK", "BOWL FIRST")


# <a id='pycaret'></a>
# # Outlier Detection using PyCaret
# [back to top](#0)

# Let's clean identify features, perform KMeans clustering on the commentary text and perform outlier detection using PyCaret library

# In[ ]:


#set(punctuation)
stop_words = stopwords.words('english')
remove_words = ['off','on','over','out', 'in', 'through']
for word in remove_words:
    stop_words.remove(word)

def pre_process_text(txt):
    table = str.maketrans('', '', string.punctuation)
    #stop_words = set(stopwords.words('english'))    
    stripped = [w.translate(table) for w in word_tokenize(BeautifulSoup(txt, 'html.parser').get_text())]

    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    tagged_sentence = pos_tag(words)

    #remove nouns
    edited_sentence = [word for word,tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS']
    words = ' '.join(edited_sentence).lower()
    words = [w for w in words.split() if not w in stop_words]
    
    return ' '.join(words)

def cluster_commentary_text(df, num_clusters):
    unique_text = df["clean_text"].unique()
    n_features = None
    vectorizer = TfidfVectorizer(
        max_df=0.95, min_df=2, max_features=n_features, stop_words=stop_words
    )
    vectorizer.fit(unique_text)
    tfidf_output = vectorizer.transform(df["clean_text"])
   
    model = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
    model.fit(tfidf_output)
    predicted = model.predict(tfidf_output)
    df['text_cluster'] = predicted   
    return df

def is_out(wkid):
    '''
    If the wicket id is non zero then that particular delivery is resulted in batsman getting out!
    '''
    if wkid !=0 :
        return True
    else:
        return False
    
def detect_anamoly_v3(row):
    
    COLS = ['home_team','away_team','current_innings', 'innings_id','over','ball','runs','batsman1_name','bowler1_name','isBoundary','isWide','isNoball','batsman1_runs', 'batsman1_balls','bowler1_overs','bowler1_runs', 'text_cluster','isOut']    
    output = {}
    background_color = '#757575'
    match_id = row['id']
    innings = row['short_name'].split('v')
    pio.templates.default = "my_plotly_template"
    
    team1 = innings[0].strip()
    team2 = innings[1].strip()
    first_inning = None
    second_inning = None    
    innings = []
    if row['toss_won'] == team1 and row['decision'] == 'BOWL FIRST':
        first_inning = team2
        second_inning = team1
    elif row['toss_won'] == team1 and row['decision'] == 'BAT FIRST':    
        first_inning = team1
        second_inning = team2
    elif row['toss_won'] == team2 and row['decision'] == 'BAT FIRST':    
        first_inning = team2
        second_inning = team1
    else:
        first_inning = team1
        second_inning = team2
        
    innings = [first_inning, second_inning]
    fig = make_subplots(rows=2, cols=1,
                   subplot_titles = innings,
                    shared_xaxes=True,
                       vertical_spacing = 0.05)
    row_count = 1
    for current_innings in innings:
        
        #fig = go.Figure()
        current_innings = current_innings.strip()
        df = get_innings_df(match_id, current_innings)
        df['over'] = df['over'].apply(lambda x: x-1)
        df['clean_text'] = df["text"].apply(lambda text: pre_process_text(text))
        df['isOut'] = df['wicket_id'].apply(lambda wkid : is_out(wkid))
        df = cluster_commentary_text(df, 10)

        s = setup(df[COLS],
              session_id=42,
             #categorical_features = categorical_features,
             #numeric_features = numerical_features,
              silent = True
              #ordinal_features = ordinal_features
             )
    
        iforest = create_model('iforest', fraction = 0.15, random_state=42)
        iforest_results = assign_model(iforest)
        fig.append_trace(go.Scatter(x=iforest_results.index, 
                                 y=iforest_results['runs'], 
                                 mode = 'markers+lines', 
                                 line=dict(color=team_colors[current_innings], width=4), 
                                  name = f"{current_innings} innings"),
                         row=row_count, col=1
                    )
    
        # create list of outlier_data
        outlier_data = iforest_results[iforest_results['Anomaly'] == 1].index
        # obtain y value of anomalies to plot
        y_values = [iforest_results.loc[i]['runs'] for i in outlier_data]
   
        hover_data = df.loc[iforest_results.loc[iforest_results['Anomaly'] == 1].index][['current_innings','over','ball','runs','batsman1_name','bowler1_name','batsman1_runs', 'batsman1_balls', 'text','isBoundary', 'isWide', 'isNoball','isOut']]
        hover_data['text'] = hover_data['text'].apply(lambda text: textwrap.fill(text, width =25))
        hover_data['text'] = hover_data['text'].apply(lambda text: text.replace("\n","<br>"))
        hover_data = hover_data.to_numpy()
        #print(hover_data)

        hover_template = '<i>Innings</i>: %{customdata[0]} <br>'+ '<i>Over</i>: %{customdata[1]}.%{customdata[2]}<br>' + '<i>Runs</i>: %{customdata[3]} <br>'+ '<i>Batsman</i>: %{customdata[4]} <br>'+ '<i>Bowler</i>: %{customdata[5]} <br>'+'<i>Batsman Score</i>: %{customdata[6]}/%{customdata[7]} <br>'+ '<i>Text</i>: %{customdata[8]} <br>' + '<i>Is Boundary?</i>: %{customdata[9]} <br>' + '<i>Is wide?</i>: %{customdata[10]} <br>' + '<i>Is no ball?</i>: %{customdata[11]} <br>' + '<i>Is out?</i>: %{customdata[12]} <br>'
    
        #Add outlier data
        fig.append_trace(go.Scatter(x=outlier_data, y=y_values, mode = 'markers', 
                    name = 'Outliers', hovertemplate=hover_template, customdata=hover_data,#text=['batsman1_name','bowler1_name','over','ball','runs','text'], 
                    marker=dict(color='red',size=10)),
                                                 row=row_count, col=1)
        fig.update_layout(title=f"{row['short_name']}", height=900)#, width=1200)
        result = {'model':iforest, 'result': iforest_results}
        output[current_innings] = result
        row_count +=1
        
    fig.update_xaxes(tickfont=dict(size=18), row=1, col=1)
    fig.update_yaxes(tickfont=dict(size=18), row=1, col=1,title='Runs', titlefont=dict(size=18),tick0=0, dtick=1)    
    fig.update_xaxes(tickfont=dict(size=18), row=2, col=1,title='Balls', titlefont=dict(size=18))
    fig.update_yaxes(tickfont=dict(size=18), row=2, col=1,title='Runs', titlefont=dict(size=18),tick0=0, dtick=1)    
        
    output['fig'] = fig
    return output


# In[ ]:


def perform_outlier_analysis(team):
    sum_22 = summary_22.loc[~summary_22['season'].isna()].reset_index(drop=True)
    team_summary = sum_22.loc[sum_22['short_name'].str.contains(team)].sort_values(by='id', ascending=True)
    outputs = {}
    for index, row in team_summary.iterrows():
        print(f"{c_}Processing {row['short_name']}{res}")
        output = detect_anamoly_v3(row)
        outputs[row['short_name']] = output
        
    for index, row in team_summary.iterrows():    
        print(f"\n{c_}{row['short_name']}{res}")
        print(f"{c_}{row['description']}{res}")    
        print(f"{c_}{row['venue_name']}{res}")
        print(f"{c_}Toss won by : {row['toss_won']} and decided to {row['decision']}{res}")
        print(f"{m_}{row['result']}{res}\n")  
        outputs[row['short_name']]['fig'].show()


# <a id='outlier-detect'></a>
# # Outliers Detection Excercise for all IPL 2022 Games
# [back to top](#0)
# 

# <a id='csk'></a>
# ## Chennai Super Kings
# [back to top](#0)
# 

# In[ ]:


perform_outlier_analysis('CSK')


# <a id='dc'></a>
# # Delhi Capitals
# [back to top](#0)
# 

# In[ ]:


perform_outlier_analysis('DC')


# <a id='gt'></a>
# # Gujarat Titans
# [back to top](#0)
# 

# In[ ]:


perform_outlier_analysis('GT')


# <a id='kkr'></a>
# # Kolkata Knight Riders
# [back to top](#0)
# 

# In[ ]:


perform_outlier_analysis('KKR')


# <a id='lsg'></a>
# # Lucknow Super Giants
# [back to top](#0)
# 

# In[ ]:


perform_outlier_analysis('LSG')


# <a id='mi'></a>
# # Mumbai Indians
# [back to top](#0)
# 

# In[ ]:


perform_outlier_analysis('MI')


# <a id='pbks'></a>
# # Punjab Kings
# [back to top](#0)
# 

# In[ ]:


perform_outlier_analysis('PBKS')


# <a id='rr'></a>
# # Rajasthan Royals
# [back to top](#0)

# In[ ]:


perform_outlier_analysis('RR')


# <a id='rcb'></a>
# # Royal Challengers Bangalore
# [back to top](#0)
# 

# In[ ]:


perform_outlier_analysis('RCB')


# <a id='srh'></a>
# # Sunrisers Hyderabad
# [back to top](#0)
# 

# In[ ]:


perform_outlier_analysis('SRH')


# <a id='conclusion'></a>
# # Conclusion
# [back to top](#0)
# 
# * As we can observe from the visualizations, most of the key outlier events from the game have been properly identified this is inline with the fact that there only few outliers in the given dataset. 
# * However the Isolation Forest has a bias for certain data points because of the way the cuts are made. This results in identifying rather normal data points as outliers.

# In[ ]:




