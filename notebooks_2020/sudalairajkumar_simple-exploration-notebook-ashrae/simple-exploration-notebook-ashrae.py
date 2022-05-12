#!/usr/bin/env python
# coding: utf-8

# ### Competition Objective:
# Energy savings is one of the important area of focus our current world. Energy savings has two key elements: 
# * Forecasting future energy usage without improvements
# * Forecasting energy use after a specific set of improvements have been implemented
# 
# Once we have implemented a set of improvements, the value of energy efficiency improvements can be challenging as there's no way to truly know how much energy a building would have used without the improvements. The best we can do is to build counterfactual models. his competition challenges you to build these counterfactual models across four energy types (chilled water, electricity, hot water, and steam) based on historic usage rates and observed weather. 
# 
# ![EnergyForecast](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRkkgt96WRGblQj9jG_28YXs6bLzBVmtiRKDAxQUT92ZZ8GH2dG)
# Picture Source: [wur.nl](https://www.wur.nl/en/project/Energy-demand-forecasting-for-urban-systems-1.htm)
# 
# ### Notebook Objective:
# The objective of this notebook is to explore the data and make some inferences on the way.
# 
# ### Dataset:
# The dataset includes three years of hourly meter readings from over one thousand buildings at several different sites around the world.
# 
# We are given two main files - `train.csv` and `test.csv` just like other competitions. In addition to these files, we also have couple of more files
# * building_metadata.csv - meta data about the buildings
# * weather.csv - weather information
# 
# First let us load the train file and look at the top few rows to get an idea about the data.
# 

# In[ ]:


import numpy as np
import pandas as pd
import os
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
pd.set_option('max_columns', 100)

data_path = "/kaggle/input/ashrae-energy-prediction/"

train_df = pd.read_csv(data_path + "train.csv")
train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], format='%Y-%m-%d %H:%M:%S')
train_df.head()


# We have details about building id, meter type {0: electricity, 1: chilledwater, 2: steam, hotwater: 3}, timestamp and the meter reading. Meter reading is our target variable. 
# 
# Now let us get some generic information about the dataset.

# In[ ]:


from IPython.core.display import display, HTML

nrows = train_df.shape[0]
nbuildings = train_df["building_id"].nunique()
max_rows_building = train_df["building_id"].value_counts().values[0]
min_rows_building = train_df["building_id"].value_counts().values[-1]
min_date = train_df["timestamp"].min()
max_date = train_df["timestamp"].max()
display(HTML(f"""<br>Number of rows in the dataset: {nrows:,}</br>
             <br>Number of buildings in the dataset: {nbuildings:,}</br>
             <br>Maximum of {max_rows_building:,} rows is available for a building</br>
             <br>Minimum of {min_rows_building:,} rows is available for a building</br>
             <br>Min date value in train data is {min_date}</br>
             <br>Max date value in train data is {max_date}</br>
             """))


# ## Meter Type Count
# 
# Let us get the count of rows for each of the 4 meter type.

# In[ ]:


cnt_srs = train_df["meter"].value_counts()
cnt_srs = cnt_srs.sort_index()
cnt_srs.index = ["Electricity", "ChilledWater", "Steam", "HotWater"]
trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color="#1E90FF",
    ),
)

layout = go.Layout(
    title=go.layout.Title(
        text="Number of rows for each meter type",
        x=0.5
    ),
    font=dict(size=14),
    width=800,
    height=500,
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="meter")


# * Electricty meter type has the most number of rows (~12M) 
# 
# ## Meter Reading Distribution
# 
# Now let us look at the meter reading distribution for each of the meter types. Since the error metric is Root Mean Squared Logarithmic Error (RMSLE), let us plot the log of the meter reading.

# In[ ]:


from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook

output_notebook()
def make_plot(title, hist, edges, xlabel):
    p = figure(title=title, tools='', background_fill_color="#fafafa")
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="#1E90FF", line_color="white", alpha=0.5)

    p.y_range.start = 0
    p.xaxis.axis_label = f'Log of {xlabel} meter reading'
    p.yaxis.axis_label = 'Probability'
    p.grid.grid_line_color="white"
    return p

temp_df = train_df[train_df["meter"]==0]
hist, edges = np.histogram(np.log1p(temp_df["meter_reading"].values), density=True, bins=50)
p1 = make_plot("Meter Reading Distribution for Electricity meter", hist, edges, "electricity")

temp_df = train_df[train_df["meter"]==1]
hist, edges = np.histogram(np.log1p(temp_df["meter_reading"].values), density=True, bins=50)
p2 = make_plot("Meter Reading Distribution for Chilled Water meter", hist, edges, 'chill water')

temp_df = train_df[train_df["meter"]==2]
hist, edges = np.histogram(np.log1p(temp_df["meter_reading"].values), density=True, bins=50)
p3 = make_plot("Meter Reading Distribution for Steam meter", hist, edges, 'steam')

temp_df = train_df[train_df["meter"]==3]
hist, edges = np.histogram(np.log1p(temp_df["meter_reading"].values), density=True, bins=50)
p4 = make_plot("Meter Reading Distribution for Hot Water meter", hist, edges, 'hot water')

show(gridplot([p1,p2,p3,p4], ncols=2, plot_width=400, plot_height=400, toolbar_location=None))


# ## Distribution of Meter Readings over time
# 
# Now let us take some individual buildings and see how the meter readings has changed over time. First let us take the building with most number of readings in the training data `1298`

# In[ ]:


temp_df = train_df[train_df["building_id"]==1298].reset_index(drop=True)

import datetime

def scatter_plot(cnt_srs, color):
    trace = go.Scatter(
        x=cnt_srs.index[::-1],
        y=cnt_srs.values[::-1],
        showlegend=False,
        marker=dict(
            color=color,
        ),
    )
    return trace

tdf = temp_df[temp_df["meter"]==0]
cnt_srs = tdf["meter_reading"]
cnt_srs.index = tdf["timestamp"]
trace1 = scatter_plot(cnt_srs, 'red')

tdf = temp_df[temp_df["meter"]==1]
cnt_srs = tdf["meter_reading"]
cnt_srs.index = tdf["timestamp"]
trace2 = scatter_plot(cnt_srs, 'blue')

tdf = temp_df[temp_df["meter"]==2]
cnt_srs = tdf["meter_reading"]
cnt_srs.index = tdf["timestamp"]
trace3 = scatter_plot(cnt_srs, 'green')

tdf = temp_df[temp_df["meter"]==3]
cnt_srs = tdf["meter_reading"]
cnt_srs.index = tdf["timestamp"]
trace4 = scatter_plot(cnt_srs, 'purple')

subtitles = ["Meter reading over time for electricity meter for building 1298",
             "Meter reading over time for chill water meter for building 1298",
             "Meter reading over time for steam meter for building 1298",
             "Meter reading over time for hot water meter for building 1298"
            ]
fig = subplots.make_subplots(rows=4, cols=1, vertical_spacing=0.08,
                          subplot_titles=subtitles)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 3, 1)
fig.append_trace(trace4, 4, 1)
fig['layout'].update(height=1200, width=1000, paper_bgcolor='rgb(233,233,233)')
py.iplot(fig, filename='meter-plots')


# * The training time frame is from January to December 2016 for this building
# * The electricity meter readings are generally in the range of 400 to 600 but becomes 0 at times in between which . 
# * We can see an increase in the chill water meter from May to October of 2016 for this building and probably this might be summer time
# * Alternatively 2e can see a dip in the steam meter from May till October
# * we can see a spike in the hot water meter from November till February
# * 28th September to 20th October values are 0 for both electricity and steam meters. 
# 
# Now let us take one more building id `184` and see the plots

# In[ ]:


temp_df = train_df[train_df["building_id"]==184].reset_index(drop=True)

tdf = temp_df[temp_df["meter"]==0]
cnt_srs = tdf["meter_reading"]
cnt_srs.index = tdf["timestamp"]
trace1 = scatter_plot(cnt_srs, 'red')

tdf = temp_df[temp_df["meter"]==1]
cnt_srs = tdf["meter_reading"]
cnt_srs.index = tdf["timestamp"]
trace2 = scatter_plot(cnt_srs, 'blue')

tdf = temp_df[temp_df["meter"]==2]
cnt_srs = tdf["meter_reading"]
cnt_srs.index = tdf["timestamp"]
trace3 = scatter_plot(cnt_srs, 'green')

tdf = temp_df[temp_df["meter"]==3]
cnt_srs = tdf["meter_reading"]
cnt_srs.index = tdf["timestamp"]
trace4 = scatter_plot(cnt_srs, 'purple')

subtitles = ["Meter reading over time for electricity meter for building 184",
             "Meter reading over time for chill water meter for building 184",
             "Meter reading over time for steam meter for building 184",
             "Meter reading over time for hot water meter for building 184"
            ]
fig = subplots.make_subplots(rows=4, cols=1, vertical_spacing=0.08,
                          subplot_titles=subtitles)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 3, 1)
fig.append_trace(trace4, 4, 1)
fig['layout'].update(height=1200, width=1000, paper_bgcolor='rgb(233,233,233)')
py.iplot(fig, filename='meter-plots')


# * Steam meter is not available for this building
# * Electricity meter is between 20 to 70 most of the times with a slight dip in the last 15 days of september
# 
# Please feel free to fork the notebook and check the distribution for other buildings as well.
# 
# ## Building Metadata
# 
# Now let us explore the building metadata to get some idea. This data has 1449 buildings which is the same number as that of train data. Let us look at the top few rows.

# In[ ]:


building_df = pd.read_csv(data_path + "building_metadata.csv")
building_df.head()


# In[ ]:


cnt_srs = building_df["primary_use"].value_counts()
#cnt_srs = cnt_srs.sort_index()
trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color="#1E90FF",
    ),
)

layout = go.Layout(
    title=go.layout.Title(
        text="Primary use of Buildings - Count",
        x=0.5
    ),
    font=dict(size=14),
    width=1000,
    height=500,
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="meter")


# * Education is the one with most number of primary usage followed by office adn entertainment
# 
# ### Building floor count

# In[ ]:


cnt_srs = building_df["floor_count"].value_counts()
#cnt_srs = cnt_srs.sort_index()
trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color="#1E90FF",
    ),
)

layout = go.Layout(
    title=go.layout.Title(
        text="Floor count in the buildings - Count",
        x=0.5
    ),
    font=dict(size=14),
    width=1000,
    height=500,
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="meter")


# * Looks like as the floor count increases, the number of buildings count decreases.
# 
# ### Year Built of the buildings

# In[ ]:


cnt_srs = building_df["year_built"].value_counts()
#cnt_srs = cnt_srs.sort_index()
trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color="#1E90FF",
    ),
)

layout = go.Layout(
    title=go.layout.Title(
        text="Year built of the buildings - Count",
        x=0.5
    ),
    font=dict(size=14),
    width=1000,
    height=500,
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="meter")


# * We have buildings built from 1900 all the way upto 2017
# * Number of buildings built in 1976 seem to be of higher representation
# 
# ### Square feet of the buildings

# In[ ]:


fig = px.scatter(building_df, y="square_feet", x="year_built", color="primary_use", size="square_feet")
fig.update_layout(showlegend=True)
fig.show()


# ## Weather Data
# 
# Now let us explore the weather data 

# In[ ]:


weather_df = pd.read_csv(data_path + "weather_train.csv", parse_dates=["timestamp"])
weather_df.head()


# ### Air temperature distribution in train

# In[ ]:


fig = px.line(weather_df, x='timestamp', y='air_temperature', color='site_id')
fig.show()


# Wow looks very colorful :D
# 
# Overall it seems that the temperature increases in all the sites towards the middle of the year and decreases at the end of year. 
# 
# ### Site level air temperature distribution
# 
# Now let us look at the plot individually at site level. Each represent the change in temperature over time in each of the site ids.

# In[ ]:


from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, show
from bokeh.plotting import figure

output_notebook()

def get_plots(df):
    p = figure(plot_width=1000, plot_height=350, x_axis_type="datetime", title="Air temperature distribution over time")
    p.line(df['timestamp'], df['air_temperature'], color='navy', alpha=0.5)
    return p

tab_list = []
for site in range(16):
    temp_df = weather_df[weather_df["site_id"]==site]
    p = get_plots(temp_df)
    tab = Panel(child=p, title=f"Site:{site}")
    tab_list.append(tab)

tabs = Tabs(tabs=tab_list)
show(tabs)


# Please change the tabs to see the temperature changes in each site. 
# 
# ### Site level weather variables distribution
# 
# Now let us plot the other weather variables as well in tab format. Now each main tab represents the variable and the internal tabs represent the site.

# In[ ]:


from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, show
from bokeh.plotting import figure

output_notebook()

def get_plots(df, col, color):
    p = figure(plot_width=1000, plot_height=350, x_axis_type="datetime", title=f"{col} distribution over time")
    p.line(df['timestamp'], df[col], color=color, alpha=0.5)
    return p

col_map = {
    "dew_temperature": "Dew Temperature",
    "sea_level_pressure": "Sea Level Pressure",
    "wind_speed": "Wind Speed",
    "cloud_coverage": "Cloud Coverage",
}

color_map = {
    "dew_temperature": "brown",
    "sea_level_pressure": "green",
    "wind_speed": "red",
    "cloud_coverage": "blue",
}

main_tabs_list = []
cols = ["dew_temperature", "sea_level_pressure", "wind_speed", "cloud_coverage"]
for col in cols:
    tab_list = []
    for site in range(16):
        temp_df = weather_df[weather_df["site_id"]==site]
        p = get_plots(temp_df, col, color_map[col])
        tab = Panel(child=p, title=f"Site:{site}")
        tab_list.append(tab)
    tabs = Tabs(tabs=tab_list)
    panel = Panel(child=tabs, title=col_map[col])
    main_tabs_list.append(panel)

tabs = Tabs(tabs=main_tabs_list)
show(tabs)


# **More to come. Stay tuned!**
