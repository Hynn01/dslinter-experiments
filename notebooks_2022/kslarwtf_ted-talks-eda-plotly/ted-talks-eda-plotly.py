#!/usr/bin/env python
# coding: utf-8

# <div style="font-weight: bold;font-size:40px">Introduction</div>
# 
# ><div style="background-color: #F4F7FF;">
# ><ul style="font-size:16px;">
# >    
# ><b>About TED talk</b>
# >
# >TED talk is a recorded public-speaking presentation that was originally given at the main TED (technology, entertainment and design) annual event or one of its many satellite events around the world. TED is a nonprofit devoted to spreading ideas, usually in the form of short, powerful talks, often called "TED talks ([see here more.](https://www.techtarget.com/whatis/definition/TED-talk))
# >
# > <b>Target is:</b>
# >to investigate collected data, explore such questions like 'Who is the most popular TED talks Speaker?'. 
# >Dataset contains 6 different features of each talk available on TED's website:</b>
# >    
# >* title - Title of the Talk
# >* author - Author of Talk
# >* date - Date when the talk took place
# >* views - Number of views of the Talk
# >* likes - Number of likes of the Talk
# >* link - Link of the talk from ted.com
# >  
# >
# <center><img src="https://repeconomy.info/wp-content/uploads/2018/09/TED.jpg" width=1800></center>
# 
# <br><div style="font-weight: bold;font-size:30px">Table of Contents</div>
# 
# >[Step 1: Examining Data ](#section-one)
# ><br>[Step 2: EDA](#section-two)     
# >[Step 3: Overall conclusion](#section-three) 

# <a id="#section-two"></a>
# <br><div style="font-weight: bold;font-size:30px">Step 1: Examining Data</div>
# 
# <a id="sub-1"></a>
# ><div style="font-weight: bold;font-size:20px">1.1 Basic information & Data preproccessing</div>
# ><div style="background-color: #F4F7FF;">
# >Importing libraries and reading data
# >
# >Data preproccessing: missing values, duplicates, outliers

# In[ ]:


#Importing Requierd Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# for Interactive Shells
from IPython.display import display
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#removing warnings
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# plotly express
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)

#secrets
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("notebook_secret")
df = pd.read_csv('/kaggle/input/ted-talks/data.csv')
#display(df.info());

# some options
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5});

pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 500);


# In[ ]:


# taking a look at the data
print(f'There are {df.shape[0]} entries and {df.shape[1]} columns.')
print(f'There are {df.duplicated().sum()} duplicates.')
print(f'There are {df.isna().sum()[1]} missing values in the author column.', end='\n\n')
display(df[df['author'].isna() == True])

# dropping nans
df.dropna(inplace = True)

# converting to correct dtypes
df['date2'] = pd.to_datetime(df['date'], format='%B %Y')
df['year'] = pd. DatetimeIndex(df['date2']).year
df['month'] = pd. DatetimeIndex(df['date2']). month
df['ratio'] = df['likes'] / df['views'] * 100
numeric = ['views','likes']
# take a look at the data
display(df.info())
#df.describe()

# check data
df.head(2)


# ><div style="background-color: #F4F7FF;">
# ><b>Conclusion</b>
# ><br>
# ><br>1. We see one missing value. This isn't an error, the talk "Year In Ideas 2015" was presented at an official TED conference, 8-minute highlight reel packed with different excerpts of conferences 2015 y. We've dropped this entry, as the amount of missing values is less than 0.1%.
# ><br>2. There are no duplicated values.
# ><br>3. Also we've converted the column 'date' to datetime format, as it's more comfortable to work with this (maybe it's my personal taste).
# ><br>4. The new column 'ratio' equals to 'likes' / 'views' * 100. It can be usefull to find out the most interesting topics for audience.

# <a id="sub-2"></a>
# ><div style="font-weight: bold;font-size:20px">1.2 Checking outliers</div>

# In[ ]:


def outliers_plotly(row, col, data, columns, title, chart_kind='hist'):
    fig = make_subplots(rows=row, cols=col)

    i, j = 1, 1

    for k in range(row*col):
        if chart_kind == 'hist':
            fig.add_trace(
                go.Histogram(x=data[columns[k]], name=columns[k],opacity=0.75),
                row=i, col=j)
            
        elif chart_kind == 'box':
            fig.add_trace(
                go.Box(y=data[columns[k]], name=columns[k],opacity=0.75),
                row=i, col=j)
        else:
            print('Error. Please, check function arguments')
        if j // col == 1:
            i += 1
            j = 1
        else:
            j += 1
    fig.update_layout(title={'text': title,'y':0.98,'x':0.5,'xanchor': 'center','yanchor': 'top'}, height=450,
                    margin=dict(l=120, r=80, t=50, b=10), paper_bgcolor="#fcfeff", plot_bgcolor='#F1F1F1' )

    fig.show()
    
outliers_plotly(row=1, col=2, data=df, columns=numeric, title="Histograms before removing outliers", chart_kind='hist')
outliers_plotly(row=1, col=2, data=df, columns=numeric, title="", chart_kind='box')


# ><div style="background-color: #F4F7FF;">
# ><b>Interquartile Range Method to remove the outliers</b>
# >
# >Data isn't normal or normal enough to treat it as a Gaussian distribution. Views and likes distributions are skewed to right, there are a lot of outliers.
# A good statistic for summarizing a non-Gaussian distribution sample of data is the <i>Interquartile Range, or IQR for short.</i>
# >
# >The IQR is calculated as the difference between <b>the 75th and the 25th percentiles</b> of the data and defines the box in a box and whisker plot.
# >
# >Remember that percentiles can be calculated by sorting the observations and selecting values at specific indices. The 50th percentile is the middle value, or the average of the two middle values for an even number of examples. If we had 10,000 samples, then the 50th percentile would be the average of the 5000th and 5001st values.
# >
# >We refer to the percentiles as quartiles (“quart” meaning 4) because the data is divided into four groups via the 25th, 50th and 75th values.

# In[ ]:


# removing outliers with the IQO method
Q1 = df[numeric].quantile(0.25)
Q3 = df[numeric].quantile(0.75)  
IQR = Q3 - Q1
#print('Here we will get IQR for each column\n',IQR)

df_filtered = df[~((df[numeric] < (Q1 - 1.5 * IQR)) |(df[numeric] > (Q3 + 1.5 * IQR))).any(axis=1)]
# plot the boxes

outliers_plotly(row=1, col=2, data=df_filtered, columns=numeric, title="Box Plots after removing outliers", chart_kind='box')


# ><div style="background-color: #F4F7FF;">
# ><b>Studying distributions after deleting outliers.</b>
# ><br>- There min of views and likes are 1200 and 37 respectively. It's 'Post-Pandemic Paradise in Rapa Nui' by Far Flung, was released in 2020;
# ><br>- The median value of views after filtering data is 1.2M! Indeed, it's really extremelly popular platform to share with ideas and information;
# ><br>- The max value of views after filtering data is 4.2M. That speech is worthy of our attention.

# <a id="#section-two"></a>
# <br><div style="font-weight: bold;font-size:30px">Step 2: EDA</div>

# ><div style="background-color: #F4F7FF;">
# ><div style="font-weight: bold;font-size:20px">2.1 Total views and likes and amount of TED Talks grouped by years and month</div>
# ><br>Now we'are going to find out top 10 the most popular authors and plot line plots with the dynamic of views \ likes by years.

# In[ ]:


# Creating df

g1 = df_filtered.query('year > 2008').groupby(['year'])['views'].sum().reset_index()
g2 = df_filtered.query('year > 2008').groupby(['year'])['likes'].sum().reset_index()

# Plotting Pie charts
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]])
pull = [0]*len(g1['year'])
pull[g1['views'].tolist().index(g1['views'].max())] = 0.2

fig.add_trace(go.Pie(values=g1['views'], labels=g1['year'], pull=pull, hole=0.8), row=1, col=1)
fig.add_trace(go.Pie(values=g2['likes'], labels=g2['year'], pull=pull, hole=0.8), row=1, col=2)

fig.update_layout(
    title='Total views and likes for 2009 - 2022 y.',
    title_x = 0.5,
    margin=dict(l=0, r=0, t=30, b=0),
    legend_orientation='h',
    annotations=[dict(text='Views', x=0.19, y=0.5, font_size=20, showarrow=False),
                 dict(text='Likes', x=0.8, y=0.5, font_size=20, showarrow=False)])
fig.show()


# In[ ]:


# TITLES YEARS
var = 'title'
colors_list = sns.color_palette("deep", n_colors=30).as_hex()

# group by all years
g1 = df_filtered.groupby(['year'])[var].count().reset_index()

# top 10 authors by count
author_list_count = df_filtered.groupby(['author'])[var].count().reset_index().                                    sort_values(var, ascending=False).head(10)['author'].to_list()


df_top_authors = df_filtered[df_filtered['author'].isin(author_list_count)].reset_index(drop=True)
g2 = df_top_authors.groupby(['year', 'author'])[var].count().reset_index()

# plotting line charts
fig = make_subplots(rows=2, cols=2)

fig.add_trace(go.Scatter(x=g1["year"], y=g1[var], name='All authors', mode='lines+markers'), row=1, col=1)
for j, i in enumerate(author_list_count):
    fig.add_trace(go.Scatter(x=g2.query('author == @i')["year"], 
                             y=g2.query('author == @i')[var],
                             line_color=colors_list[j] , name=i,mode='lines'), row=1, col=2)
#fig['layout']['xaxis'].update(title_text='x')   
fig.update_xaxes(title_text='Years', range=[2008, 2022], row=1, col=1)   
fig.update_xaxes(title_text='Years', range=[2008, 2022], row=1, col=2) 

# TITLES MONTH
# group by all month
g1 = df_filtered.groupby(['month'])[var].count().reset_index()

# top 10 authors by count
g2 = df_top_authors.groupby(['month', 'author'])[var].count().reset_index()

# plotting line charts
fig.add_trace(go.Scatter(x=g1["month"], y=g1[var], name='All authors', mode='lines+markers'), row=2, col=1)
for j, i in enumerate(author_list_count):
    fig.add_trace(go.Scatter(x=g2.query('author == @i')["month"], 
                             y=g2.query('author == @i')[var],
                             line_color=colors_list[j] , name=i,mode='lines'), row=2, col=2)

fig.update_xaxes(title_text='Month', row=2, col=1)   
fig.update_xaxes(title_text='Month' , row=2, col=2)     
fig.update_layout(legend_orientation="h",title={'text': 'The dynamic of TED talks amount by years and months',
                         'y':0.98,'x':0.5,'xanchor': 'center','yanchor': 'top'}, height=700,
                         margin=dict(l=120, r=80, t=50, b=10), paper_bgcolor="#fcfeff", plot_bgcolor='#F1F1F1')

fig.show()


# ><div style="background-color: #F4F7FF;">
# ><b>Observations.</b>
# >
# > 1. Since 2008, TED Talks  has been gaining popularity, with public interest peaking in 2019. We see a sharp decline in 2020 - 2021 period, theese years were  filled with various events that could reduce public interest.
# >
# > 2. Alex Gendler and Iseult Gillespie they've performed the most times in 2020 and 2019 respectively, which is related to the huge number of likes and views (see below).
# >
# > 3. In January, August the least amount of TED Talks were produced. Perhaps this is due to holidays and vacations.

# ><div style="background-color: #F4F7FF;">
# ><div style="font-weight: bold;font-size:20px">2.2 Finding top-10 authors.</b></div>
# ><br> Now we'are going to find out top 10 the most popular authors and plot line plots with the dynamic of views \ likes by years.

# In[ ]:


# VIEWS
var = 'views'
# group by all years
g1 = df_filtered.groupby(['year'])[var].sum().reset_index()

# top 10 popular authors by views
author_list_views = df_filtered.groupby(['author']).sum().reset_index().                                    sort_values(var, ascending=False).head(10)['author'].to_list()
colors_list = sns.color_palette("deep", n_colors=10).as_hex()

df_top_authors_v = df_filtered[df_filtered['author'].isin(author_list_views)].reset_index(drop=True)
g2 = df_top_authors_v.groupby(['year', 'author'])[var].sum().reset_index()

# plotting line charts
fig = make_subplots(rows=2, cols=2)

fig.add_trace(go.Scatter(x=g1["year"], y=g1[var], name='All authors', mode='lines+markers'), row=2, col=1)
for j, i in enumerate(author_list_views):
    fig.add_trace(go.Scatter(x=g2.query('author == @i')["year"], 
                             y=g2.query('author == @i')[var],
                             line_color=colors_list[j] , name=i,mode='lines'), row=1, col=1)

    
# LIKES    
# group by all years
var = 'likes'
# group by all years
g1 = df_filtered.groupby(['year'])[var].sum().reset_index()

# top 10 popular authors by likes
author_list_likes = df_filtered.groupby(['author']).sum().reset_index().                                    sort_values(var, ascending=False).head(10)['author'].to_list()
colors_list = sns.color_palette("magma", n_colors=10).as_hex()

df_top_authors_l = df_filtered[df_filtered['author'].isin(author_list_likes)].reset_index(drop=True)
g2 = df_top_authors_l.groupby(['year', 'author'])[var].sum().reset_index()

# plotting line charts
fig.add_trace(go.Scatter(x=g1["year"], y=g1[var], name='All authors', mode='lines+markers'), row=2, col=2)
for j, i in enumerate(author_list_likes):
    fig.add_trace(go.Scatter(x=g2.query('author == @i')["year"], 
                             y=g2.query('author == @i')[var],
                             line_color=colors_list[j] , name=i,mode='lines'), row=1, col=2)
    
fig.update_xaxes(range=[2000, 2022])
fig.update_layout(legend_orientation="h",title={'text': 'The dynamic of views and likes for the most 10 popular authors',
                         'y':0.98,'x':0.5,'xanchor': 'center','yanchor': 'top'}, height=900,
                         margin=dict(l=120, r=80, t=50, b=10), paper_bgcolor="#fcfeff", plot_bgcolor='#F1F1F1')

fig.show()


# In[ ]:


df_filtered.query('author == "Alex Gendler" & year == 2020').sort_values('views', ascending=False).head()


# ><div style="background-color: #F4F7FF;">
# ><b>Observations.</b>
# > 
# >1. The list of ten most popular authors is in the chart above. Alex Gendler leads in the number of views and likes, his "How the world's longest underwater tunnel was built" has 2800,000 views! And now it's more (2.9M). [See the talk](https://www.ted.com/talks/alex_gendler_how_the_world_s_longest_underwater_tunnel_was_built).
# >
# >     "Flanked by two powerful nations, the English Channel has long been one of the world's most important maritime passages. Yet for most of its history, crossing was a dangerous prospect. Engineers proposed numerous plans for spanning the gap, including a design for an underwater passage more than twice the length of any existing tunnel. Alex Gendler details the creation of the Channel Tunnel." - Sounds interesting!
# > 
# >2. The lineplots of views and likes are almost simular! It seems like there’s a linear correlation between these two variables.
# >
# >3. Below I show two other options to view similar information: pie charts and bar plots.

# In[ ]:


fig = go.Figure()

for i in range(2017, 2022):
    g = df_top_authors_v.query('year==@i').groupby(['author'])['views'].sum()                                          .sort_values(ascending=False).reset_index()
    fig.add_trace(go.Bar(x=g['author'], y=g['views'], name=i))

    
fig.update_layout(title='Total views by authors for 2017 - 2021 y.',
                  title_x = 0.5,
                  barmode='stack', paper_bgcolor="#fcfeff", plot_bgcolor='#F1F1F1',
                  xaxis={'categoryorder':'category ascending'})
fig.show()


# ><div style="background-color: #F4F7FF;">
# ><b>Observations. </b>
# >
# > Comparing the period from 2017 to 2021: 
# >    
# > 1. In 2018, many authors were quite popular: Elizabeth Cox, Daniel and Dan Finkel, Emma Bryce, Alex Gendler, Iseult Gillespie and Juan Enriquez.
# > 
# > 2. In 2019 Alex Gendler, Iseult Gillespie  and Danial Finkel are almost equally popular!
# > 
# > 3. 2020 was a very productive year for Alex Gendler and Matt Walker. We can presume that this is due to a pandemic, and some of their work was quite timely.

# In[ ]:


g1 = df_top_authors_v.groupby(['year', 'author'])['views'].sum().reset_index()
g2 = df_top_authors_l.groupby(['year','author'])['likes'].sum().reset_index()

# Plotting Pie charts
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]])
pull1 = [0]*len(g1['author'])
pull1[g1['views'].tolist().index(g1['views'].max())] = 0.2

pull2 = [0]*len(g1['author'])
pull2[g2['likes'].tolist().index(g2['likes'].max())] = 0.2


fig.add_trace(go.Pie(values=g1['views'], labels=g1['author'], 
                     pull=pull1, hole=0.8, marker_colors=px.colors.sequential.RdBu), row=1, col=1)

fig.add_trace(go.Pie(values=g2['likes'], labels=g2['author'], pull=pull2, hole=0.8), row=1, col=2)

fig.update_layout(
    title='Total views and likes for authors',
    title_x = 0.5,
    margin=dict(l=0, r=0, t=30, b=0),
    legend_orientation='h',paper_bgcolor="#fcfeff",
    annotations=[dict(text='Views', x=0.19, y=0.5, font_size=20, showarrow=False),
                 dict(text='Likes', x=0.8, y=0.5, font_size=20, showarrow=False)])
fig.show()


# ><div style="background-color: #F4F7FF;">
# ><b>Observations.</b>
# > 
# > Again we see Alex Gendler at the top, 22.6% likes \ views belong to his TED talks! 
# >

# ><div style="background-color: #F4F7FF;">
# ><div style="font-weight: bold;font-size:20px">2.3 Finding top-5 titles (non-filtered df)</b></div>
# ><br> Finding top 5 the most popular TED Talks by views \ likes and percentage (likes to views).

# In[ ]:


fig = make_subplots(rows=2, cols=2, specs=[[{"rowspan": 2, 'type':'domain'}, {'type':'domain'}],
                                            [None, {'type':'domain'}]])


g1 = df.sort_values('views', ascending=False).head(5)
fig.add_trace(go.Pie(labels=g1['title'], values=g1['views'], name="Views"),1, 1)


g2 = df.sort_values('likes', ascending=False).head(5)
fig.add_trace(go.Pie(labels=g2['title'], values=g2['likes'], name="Likes",  marker_colors=colors_list),1, 2)

g3 = df.sort_values('views', ascending=False).head(1000).sort_values(['ratio'], ascending=False).head(5)
fig.add_trace(go.Pie(labels=g3['title'], values=g3['ratio'], name="Views"),2, 2)

fig.update_layout(title='Top TED Talks by views, likes and percentage likes to views',
                  title_x = 0.5,
                  barmode='stack', paper_bgcolor="#fcfeff", plot_bgcolor='#F1F1F1',
                  xaxis={'categoryorder':'category ascending'})
fig.show()


# ><div style="background-color: #F4F7FF;">
# ><b>Observations.</b>
# > 
# >1. The most popular TED Talks is 'Do schools kill creativity?' by Sir Ken Robinson, 2006 according to the amount of views and likes. At this top-5 we also see 'Your body language may shape who you are' by Amy Cuddy, 'Inside the mind of a master procrastinator' by Tim Urban, 'How great leaders inspire action' by Simon Sinek and 'The power of vulnerability' by Brené Brown. 
# >   
# >2. The most percentage of likes to views belongs to 'A brie(f) history of cheese' by Paul S. Kindstedt, 2018. The second place belongs to 'Why you should define your fears instead of your goals' by Tim Ferriss, then 'There's more to life than being happy' by Emily Esfahani Smith, 'Why do cats act so weird?' by Tony Buffington and 'How to control someone else's arm with your brain' by Greg Gage.
# >
# > As a hypothesis (that's needed to be tested more): more serious topics seem to cover more points of view, but entertainment content gets a little more likes as a percentage.

# <a id="#section-three"></a>
# <br><div style="font-weight: bold;font-size:30px">Step 3: Overall conclusion</div>

# ><div style="background-color: #F4F7FF;">
# ><b>The following has been found:</b>
# > 
# > 1. TED Talks has attracted more viewers since 2008, which may be partly due to digitalization, the advent of accessible Internet and devices.
# >    
# > 2. The median value of views is 1.2M - 1.3M! Indeed, it's really extremelly popular platform to share with ideas and information;
# >
# > 3. The dataset contains large outliers, which identifies some of the videos as the most trendy and popular. Seems like these performances are very topical and vital for the public.
# >
# > 4. The most popular TED Talks is 'Do schools kill creativity?' by Sir Ken Robinson, 2006 according to the amount of views and likes. At this top-5 we also see 'Your body language may shape who you are' by Amy Cuddy, 'Inside the mind of a master procrastinator' by Tim Urban, 'How great leaders inspire action' by Simon Sinek and 'The power of vulnerability' by Brené Brown. 
# >   
# >5. The most percentage of likes to views belongs to 'A brie(f) history of cheese' by Paul S. Kindstedt, 2018. The second place belongs to 'Why you should define your fears instead of your goals' by Tim Ferriss, then 'There's more to life than being happy' by Emily Esfahani Smith, 'Why do cats act so weird?' by Tony Buffington and 'How to control someone else's arm with your brain' by Greg Gage.
# > 
# >6. In January, August the least amount of TED Talks were produced. Perhaps this is due to holidays and vacations.
# >    
# >7. Comparing the period from 2017 to 2021: 
# >    
# > * In 2018, many authors were quite popular: Elizabeth Cox, Daniel and Dan Finkel, Emma Bryce, Alex Gendler, Iseult Gillespie and Juan Enriquez.
# > 
# > *  In 2019 Alex Gendler, Iseult Gillespie  and Danial Finkel are almost equally popular!
# > 
# > *  2020 was a very productive year for Alex Gendler and Matt Walker. We can presume that this is due to a pandemic, and some of their work was quite timely.

# ><div style="background-color: #F4F7FF;">

# <a id="section-end"></a>
# ><div>
# ><ul style="font-size:20px;">
#  <b>Thank you</b> so much for reading my project. 
#  <br>Please, UPvote, if you like it or find usefull!😄
# >    
# > Many thanks to the сreator of this dataset.   
# ></ul>
# ></div>
