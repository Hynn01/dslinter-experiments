#!/usr/bin/env python
# coding: utf-8

# ## Chennai & its water sources
# 
# **Chennai** also known as Madras, is the capital of the Indian state of Tamil Nadu. Located on the Coromandel Coast off the Bay of Bengal, it is the biggest cultural, economic and educational centre of south India. Population of Chennai is close to 9 million and is the 36th largest urban area by population in the world - Wikipedia
# 
# ![Chennai](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Chennai_Montage.jpg/500px-Chennai_Montage.jpg)
# 
# Chennai is entirely dependent on ground water resources to meet its water needs. Ground water resources in Chennai are replenished by rain water and the city's average rainfall is 1,276 mm[1](https://en.wikipedia.org/wiki/Water_management_in_Chennai). 
# 
# Following are the major sources of water supply for Chennai city.
# 1. Four major reservoirs in Red Hills, Cholavaram, Poondi and Chembarambakkam
# 2. Cauvery water from Veeranam lake
# 3. Desalination plants at Nemelli and Minjur
# 4. Aquifers in Neyveli, Minjur and Panchetty
# 5. Tamaraipakkam, Poondi and Minjur Agriculture wells 
# 6. CMWSSB Boreweels
# 7. Retteri lake
# 
# The above one is also roughly the descending order in which the contribution is made to overall fresh water requirements of the city. In addition to this, people make use of borewells and private tankers for their water needs. 
# 
# Currently (July 2019), Chennai is facing an acute water shortage due to shortage of rainfall for the past three years (and we had one of the worst floods in history the year before that!). As a result, the water in these resources are depleting along with the groundwater level. This [video](https://youtu.be/iaG7kRcSxwA) will give an idea about the current state.  
# 
# In this notebook, let us explore the data of different water resources available.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
from plotly import tools
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go


# Firstly, we have data about the water availability in four major reservoirs that supply water to Chennai. This data spans from 2004 to 2019. All the measurements are in mcft (million cubic feet). Let us look at the top few lines.

# In[ ]:


df = pd.read_csv("../input/chennai_reservoir_levels.csv")
df["Date"] = pd.to_datetime(df["Date"], format='%d-%m-%Y')
df.head()


# So we have data at daily level and each column represents the water level in the given reservoir at the given date.
# 
# 
# ## Water Levels in Four Major Reservoirs

# In[ ]:


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

cnt_srs = df["POONDI"]
cnt_srs.index = df["Date"]
trace1 = scatter_plot(cnt_srs, 'red')

cnt_srs = df["CHOLAVARAM"]
cnt_srs.index = df["Date"]
trace2 = scatter_plot(cnt_srs, 'blue')

cnt_srs = df["REDHILLS"]
cnt_srs.index = df["Date"]
trace3 = scatter_plot(cnt_srs, 'green')

cnt_srs = df["CHEMBARAMBAKKAM"]
cnt_srs.index = df["Date"]
trace4 = scatter_plot(cnt_srs, 'purple')

subtitles = ["Water Availability in Poondi reservoir - in mcft",
             "Water Availability in Cholavaram reservoir - in mcft",
             "Water Availability in Redhills reservoir - in mcft",
             "Water Availability in Chembarambakkam reservoir - in mcft"
            ]
fig = tools.make_subplots(rows=4, cols=1, vertical_spacing=0.08,
                          subplot_titles=subtitles)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 3, 1)
fig.append_trace(trace4, 4, 1)
fig['layout'].update(height=1200, width=800, paper_bgcolor='rgb(233,233,233)')
py.iplot(fig, filename='h2o-plots')


# **Inference:**
# 
# * We could clearly see that evey year there is a decremental phase and a replenishment phase (mainly during october to december)
# * There was a very bad water scarcity phase seen during 2004.
# * We can also see a bad phase during 2014-15 but there was to water availability in two reservoirs (Redhills and Chembarambakkam) and so it was a savior.
# * Now coming to recent times, the data shows that there is no water availability in any of the four major reservoirs.
# 
# ## Combined Water Levels in Four Major Reservoirs
# 
# For better understanding, let us sum up the water availability in all four reservoirs and then make the plot.

# In[ ]:


df["total"] = df["POONDI"] + df["CHOLAVARAM"] + df["REDHILLS"] + df["CHEMBARAMBAKKAM"]
df["total"] = df["POONDI"] + df["CHOLAVARAM"] + df["REDHILLS"] + df["CHEMBARAMBAKKAM"]

cnt_srs = df["total"]
cnt_srs.index = df["Date"]
trace5 = scatter_plot(cnt_srs, 'red')

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.08,
                          subplot_titles=["Total water availability from all four reservoirs - in mcft"])
fig.append_trace(trace5, 1, 1)


fig['layout'].update(height=400, width=800, paper_bgcolor='rgb(233,233,233)')
py.iplot(fig, filename='h2o-plots')


# **Inference:**
# 
# * Summing up all the water availability from four reservoirs, we can see that the water levels reached almost zero thrice (2004, 2017 and 2019)
# * Generally after the rainfall, the reservoirs used to get replenished to about 10K Mcft until 2012 which is not the case afterwards due to the lack of rainfall.
# * Only during the (in)famous Chennai floods of 2015, it has reached the 10K level after 2012. 
# * 2017 is similar to 2019 in terms of depletion of water availability but the levels reached close to 0 during end of august unlike now where the levels reached in the beginning of June itself. 
# 
# ## Rainfall Levels in Reservoir Regions
# 
# Now there are two clear facts:
# 1. There is no water in any of the major reservoirs
# 2. Reservoirs depend on rain for their replenishment. 
# 
# Next we can look at the rainfall data in these reservoir regions to analyze the rainfall months. Let us take the total monthly rainfall in these reservoir regions and plot the same.
# 

# In[ ]:


rain_df = pd.read_csv("../input/chennai_reservoir_rainfall.csv")
rain_df["Date"] = pd.to_datetime(rain_df["Date"], format='%d-%m-%Y')

rain_df["total"] = rain_df["POONDI"] + rain_df["CHOLAVARAM"] + rain_df["REDHILLS"] + rain_df["CHEMBARAMBAKKAM"]
rain_df["total"] = rain_df["POONDI"] + rain_df["CHOLAVARAM"] + rain_df["REDHILLS"] + rain_df["CHEMBARAMBAKKAM"]

def bar_plot(cnt_srs, color):
    trace = go.Bar(
        x=cnt_srs.index[::-1],
        y=cnt_srs.values[::-1],
        showlegend=False,
        marker=dict(
            color=color,
        ),
    )
    return trace

rain_df["YearMonth"] = pd.to_datetime(rain_df["Date"].dt.year.astype(str) + rain_df["Date"].dt.month.astype(str), format='%Y%m')

cnt_srs = rain_df.groupby("YearMonth")["total"].sum()
trace5 = bar_plot(cnt_srs, 'red')

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.08,
                          subplot_titles=["Total rainfall in all four reservoir regions - in mm"])
fig.append_trace(trace5, 1, 1)


fig['layout'].update(height=400, width=800, paper_bgcolor='rgb(233,233,233)')
py.iplot(fig, filename='h2o-plots')


# **Inferences:**
# 
# * Looks like the city gets some rains in the month of June, July, August and September due to south west monsoon. 
# * Major rainfall happens during October and November of every year which is due to North-east monsoon.
# * During the initial years rain from north-east monsoon is much higher than south-west monsoon. But seems like last few years, they both are similar (reduction in rains from north-east monsoon).
# * We have got some good rains in August and September 2019, but the water reservoir levels are yet to go up.
# 
# 
# 

# In[ ]:


rain_df["Year"] = pd.to_datetime(rain_df["Date"].dt.year.astype(str), format='%Y')

cnt_srs = rain_df.groupby("Year")["total"].sum()
trace5 = bar_plot(cnt_srs, 'red')

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.08,
                          subplot_titles=["Total yearly rainfall in all four reservoir regions - in mm"])
fig.append_trace(trace5, 1, 1)


fig['layout'].update(height=400, width=800, paper_bgcolor='rgb(233,233,233)')
py.iplot(fig, filename='h2o-plots')


# The amount of rainfall in 2018 is the lowest of all the years from 2004.
# 
# We are getting some good rains so far in 2019. Hopefully this continues.

# ## Water shortage estimation
# 
# Since all the data is available in the public domain, would it be possible to do some analysis and see whether we can estimate this water shortage ahead of time so as to plan for it?
# 
# First let us just take a simple step to compare the sum of water levels at the beginning of summer (Let us take February 1st of every year). This is because there will not be any replenishment till the next monsson and the amount of water stored in the four reservoirs itself will be clear indicator of how long can the water be managed during summer and whether there should be some backup plans.

# In[ ]:


temp_df = df[(df["Date"].dt.month==2) & (df["Date"].dt.day==1)]

cnt_srs = temp_df["total"]
cnt_srs.index = temp_df["Date"]
trace5 = bar_plot(cnt_srs, 'red')

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.08,
                          subplot_titles=["Availability of total reservoir water (4 major ones) at the beginning of summer"])
fig.append_trace(trace5, 1, 1)


fig['layout'].update(height=400, width=800, paper_bgcolor='rgb(233,233,233)')
py.iplot(fig, filename='h2o-plots')


# This clearly indicates that there is not enough water in the reservoirs at the beginning of summer 2019 to cope up with the needs of the city. Infact this is the second worst level after 2004 (Also it is important to note that city has grown a lot bigger from 2004 to 2019). 
# 
# The city has just had 1000 mcft of water at the beginning of the summer which is much worser than the 2017 levels of 1500 mcft. So just by looking at the very low water level, the water scarcity could have been forecasted without even computing the consumption level per day. 
# 
# ## Current Sources of Water for the City
# 
# With the major source of water for the city fully dry, the water needs is being *partially* met by other sources mentioned above. Of the estimated need of 830 MLD (Million Liters per Day) of water, 500 MLD is being met by following resources.
# 
# ![Current Water Sources](https://www.thehindu.com/news/cities/chennai/szuw7e/article28023480.ece/ALTERNATES/FREE_460/TH18-city-reservoirscol)
# 

# Thanks to [The Hindu](https://www.thehindu.com/news/cities/chennai/water-supply-in-city-cut-to-525-from-830-mld-hc-told/article28023481.ece) for the above picture. 
# 
# Of the current sources of water supply, [Veeranam lake supply](http://123.63.203.150/veeranam.htm) is having some water and is helping to cope up with the water needs of the city.
# 
# 
# ## Conclusion:
# 
# The water scarcity of 2004 has brought Veeranam lake as the new means of water supply for the city. 
# 
# Hopefully, this current scarcity (July 2019) will bring more additional sources of water for the ailing city. The city has grown a lot in the last 15 years and so need additional water resources to manage the needs. 
# 
# The city needs to devise better scarcity control methods by estimating the needs ahead of time. And for now,
# 
# <p style="text-align: center;"> <strong> "Only RAIN can save the city!" </strong></p>

# P.S: Ironically I work for Water (H2O) currently :D
