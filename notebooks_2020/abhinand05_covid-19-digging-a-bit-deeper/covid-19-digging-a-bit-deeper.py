#!/usr/bin/env python
# coding: utf-8

# ## Introduction (Although it doesn't need any) 
# 
# > <font size=4 color='red'>(The Visualizations are updated every 24 Hours)</font>
# 
# <img align="left" src="https://gcn.com/-/media/GIG/GCN/Redesign/Articles/2020/February/covid19.jpg"></img>

# ### COVID-19 is wreaking havoc across the globe!!!
# > #### Well, that's something everyone probably already knows as long as they aren't living under the cave for some reason. 
# #### But is it really China who's the most affected right now? 
# #### Or one of those European countries that are literally on fire at the moment because of the virus? 
# #### How good is the US doing?
# #### How on Earth did it spread so far? (See for yourself)
# #### Will a public lockdown work? 
# #### Lessons to learn from China?
# #### Should we be worried? 
# > #### This notebook provides reason with data to shed some light on all of the above questions. Lets find out.

# ## References and Acknowledgements
# **Data:**
# * [Novel Coronavirus (COVID-19) Cases, provided by JHU CSSE](https://github.com/CSSEGISandData/COVID-19)
# 
# * [COVID19 Global Forecasting (Week 1)](https://www.kaggle.com/c/covid19-global-forecasting-week-1)
# 
# * [Novel Corona Virus 2019 Dataset](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset)
# 
# * [COVID-19 Complete Dataset (Updated every 24hrs)](https://www.kaggle.com/imdevskp/corona-virus-report)
# 
# **Notebooks:**
# * [covid-19-analysis-visualization-comparisons](https://www.kaggle.com/imdevskp/covid-19-analysis-visualization-comparisons)
# 
# * [Creating a good Analytics Report](https://www.kaggle.com/jpmiller/creating-a-good-analytics-report)
# 
# **Visual Dashboards:**
# 
# * [JHU Dashboard Web](https://www.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6)
# 
# * [JHU Dashbiard Mobile](http://www.arcgis.com/apps/opsdashboard/index.html#/85320e2ea5424dfaaa75ae62e5c06e61)
# 
# * [Bing Live Dashboard](https://www.bing.com/covid)

# ## Previous Reports:
# 
# [**21-03-2020**  (Universal Time) --> Notebook Version 18](https://www.kaggle.com/abhinand05/covid-19-digging-a-bit-deeper?scriptVersionId=30665745)
# 
# [**19-03-2020**  (Universal Time) --> Notebook Version 16](https://www.kaggle.com/abhinand05/covid-19-digging-a-bit-deeper?scriptVersionId=30501389)

# <font size=5 style="color:red"> Please give an UPVOTE if you like this notebook</font>

# In[ ]:


import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_dark"
from plotly.subplots import make_subplots

from pathlib import Path
data_dir = Path('../input/covid19-global-forecasting-week-1')

import os
os.listdir(data_dir)


# The data for this competition...
# 
# (Won't be using much of it for Analysis)

# In[ ]:


data = pd.read_csv(data_dir/'train.csv', parse_dates=['Date'])
data.head()


# In[ ]:


data.info()


# In[ ]:


data.rename(columns={'Date': 'date', 
                     'Id': 'id',
                     'Province/State':'state',
                     'Country/Region':'country',
                     'Lat':'lat',
                     'Long': 'long',
                     'ConfirmedCases': 'confirmed',
                     'Fatalities':'deaths',
                    }, inplace=True)
data.head()


# The cleaned data from [COVID-19 Complete Dataset (Updated every 24hrs)](https://www.kaggle.com/imdevskp/corona-virus-report) is used for visualizations.

# In[ ]:


cleaned_data = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])
cleaned_data.head()


# In[ ]:


cleaned_data.rename(columns={'ObservationDate': 'date', 
                     'Province/State':'state',
                     'Country/Region':'country',
                     'Last Update':'last_updated',
                     'Confirmed': 'confirmed',
                     'Deaths':'deaths',
                     'Recovered':'recovered'
                    }, inplace=True)

# cases 
cases = ['confirmed', 'deaths', 'recovered', 'active']

# Active Case = confirmed - deaths - recovered
cleaned_data['active'] = cleaned_data['confirmed'] - cleaned_data['deaths'] - cleaned_data['recovered']

# replacing Mainland china with just China
cleaned_data['country'] = cleaned_data['country'].replace('Mainland China', 'China')

# filling missing values 
cleaned_data[['state']] = cleaned_data[['state']].fillna('')
cleaned_data[cases] = cleaned_data[cases].fillna(0)
cleaned_data.rename(columns={'Date':'date'}, inplace=True)

data = cleaned_data


# In[ ]:


print("External Data")
print(f"Earliest Entry: {data['date'].min()}")
print(f"Last Entry:     {data['date'].max()}")
print(f"Total Days:     {data['date'].max() - data['date'].min()}")


# ## Data Analysis - COVID-19

# ## 1. Confirmed Cases Over Time

# Now lets take a look at the confirmed cases across the globe.

# In[ ]:


grouped = data.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

fig = px.line(grouped, x="date", y="confirmed", 
              title="Worldwide Confirmed Cases Over Time")
fig.show()

fig = px.line(grouped, x="date", y="confirmed", 
              title="Worldwide Confirmed Cases (Logarithmic Scale) Over Time", 
              log_y=True)
fig.show()


# 1. Looks like the exponential growth of the pandemic is still in it's peaks and that is not good at all.
# 
# 2. The slope of the line at the latest time frame is very high making matters even worse.
# 
# 3. Looking at the same graph in Logarithmic scale reveals the matter is very very serious all over the World maybe because the disease has just started to grow outside of China.
# 
# 4. At the current rate anything may happen. Maybe even a million cases in just a weeks time. Who knows.

# In[ ]:


grouped_china = data[data['country'] == "China"].reset_index()
grouped_china_date = grouped_china.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

grouped_italy = data[data['country'] == "Italy"].reset_index()
grouped_italy_date = grouped_italy.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

grouped_us = data[data['country'] == "US"].reset_index()
grouped_us_date = grouped_us.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

grouped_rest = data[~data['country'].isin(['China', 'Italy', 'US'])].reset_index()
grouped_rest_date = grouped_rest.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()


# In[ ]:


plot_titles = ['China', 'Italy', 'USA', 'Rest of the World']

fig = px.line(grouped_china_date, x="date", y="confirmed", 
              title=f"Confirmed Cases in {plot_titles[0].upper()} Over Time", 
              color_discrete_sequence=['#F61067'],
              height=500
             )
fig.show()

fig = px.line(grouped_italy_date, x="date", y="confirmed", 
              title=f"Confirmed Cases in {plot_titles[1].upper()} Over Time", 
              color_discrete_sequence=['#91C4F2'],
              height=500
             )
fig.show()

fig = px.line(grouped_us_date, x="date", y="confirmed", 
              title=f"Confirmed Cases in {plot_titles[2].upper()} Over Time", 
              color_discrete_sequence=['#6F2DBD'],
              height=500
             )
fig.show()

fig = px.line(grouped_rest_date, x="date", y="confirmed", 
              title=f"Confirmed Cases in {plot_titles[3].upper()} Over Time", 
              color_discrete_sequence=['#FFDF64'],
              height=500
             )
fig.show()


# 1. Looking at the plot of China's cases it is pretty clear that the disease has not been at dire levels since the turn of March. WHICH IS REALLY GOOD NEWS FOR CHINA.
# 
# 2. Well not so much for Italy by the looks of it. They are getting affected very badly. 
# 
# 3. Italy's steep rise is concerning and the new few days are really crucial.
# 
# 4. The clear spike in USA's graph might be the result of more cases getting testing for the first time.
# 
# 5. USA's situation is also very concerning. That increase in the past week or so is really significant. 
# 
# 6. The rest of the World combined is also seeing a steady increase in confirmed cases over time. 

# In[ ]:


data['state'] = data['state'].fillna('')
temp = data[[col for col in data.columns if col != 'state']]

latest = temp[temp['date'] == max(temp['date'])].reset_index()
latest_grouped = latest.groupby('country')['confirmed', 'deaths'].sum().reset_index()


# In[ ]:


fig = px.choropleth(latest_grouped, locations="country", 
                    locationmode='country names', color="confirmed", 
                    hover_name="country", range_color=[1,5000], 
                    color_continuous_scale="peach", 
                    title='Countries with Confirmed Cases')
# fig.update(layout_coloraxis_showscale=False)
fig.show()


# > #### Feel free to zoom into the interactive maps.
# 
# The above graph is just an illustration of how the virus is spread out across the globe.
# 
# **I think, looking at Europe, it's worth having a closer look.**

# In[ ]:


europe = list(['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czechia','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Ireland',
               'Italy', 'Latvia','Luxembourg','Lithuania','Malta','Norway','Netherlands','Poland','Portugal','Romania','Slovakia','Slovenia',
               'Spain', 'Sweden', 'United Kingdom', 'Iceland', 'Russia', 'Switzerland', 'Serbia', 'Ukraine', 'Belarus',
               'Albania', 'Bosnia and Herzegovina', 'Kosovo', 'Moldova', 'Montenegro', 'North Macedonia'])

europe_grouped_latest = latest_grouped[latest_grouped['country'].isin(europe)]


# In[ ]:


fig = px.choropleth(europe_grouped_latest, locations="country", 
                    locationmode='country names', color="confirmed", 
                    hover_name="country", range_color=[1,2000], 
                    color_continuous_scale='portland', 
                    title='European Countries with Confirmed Cases', scope='europe', height=800)
# fig.update(layout_coloraxis_showscale=False)
fig.show()


# **Looks like the COVID-19 has the strongest of holds in Western Europe right now.**
# 
# Cases in Most European countries have rapidly increased. 

# In[ ]:


fig = px.bar(latest_grouped.sort_values('confirmed', ascending=False)[:20][::-1], 
             x='confirmed', y='country',
             title='Confirmed Cases Worldwide', text='confirmed', height=1000, orientation='h')
fig.show()


# Looking at the numbers it's worth noting that Europe is more affected than China and it's neighbors. 
# 
# Iran being the most affected Asian country other than China

# In[ ]:


fig = px.bar(europe_grouped_latest.sort_values('confirmed', ascending=False)[:10][::-1], 
             x='confirmed', y='country', color_discrete_sequence=['#84DCC6'],
             title='Confirmed Cases in Europe', text='confirmed', orientation='h')
fig.show()


# Once again it's Western Europe dominating the number of confirmed cases with Italy more than double over the second most affected Spain and Germany not far off behind with Frace cathing up too.

# In[ ]:


usa = cleaned_data[cleaned_data['country'] == "US"]
usa_latest = usa[usa['date'] == max(usa['date'])]
usa_latest = usa_latest.groupby('state')['confirmed', 'deaths'].max().reset_index()

fig = px.bar(usa_latest.sort_values('confirmed', ascending=False)[:10][::-1], 
             x='confirmed', y='state', color_discrete_sequence=['#D63230'],
             title='Confirmed Cases in USA', text='confirmed', orientation='h')
fig.show()


# Looking at the American states, New York being the clear and outright most affected state with the numbers being almost 8 times as much as Washington State.
# 
# Here in India the total confirmed cases are much less than any of these top affected American States which is really concerning considering the fact that India shares a border with China while the US an Ocean apart.

# ## 2. Confirmed Deaths Over Time

# In[ ]:


fig = px.line(grouped, x="date", y="deaths", title="Worldwide Deaths Over Time",
             color_discrete_sequence=['#F42272'])
fig.show()

fig = px.line(grouped, x="date", y="deaths", title="Worldwide Deaths (Logarithmic Scale) Over Time", 
              log_y=True, color_discrete_sequence=['#F42272'])
fig.show()


# Global Death tolls have began to rise sharply ever since the turn of March.

# In[ ]:


plot_titles = ['China', 'Italy', 'USA', 'Rest of the World']

fig = px.line(grouped_china_date, x="date", y="deaths", 
              title=f"Deaths in {plot_titles[0].upper()} Over Time", 
              color_discrete_sequence=['#F61067'],
              height=500
             )
fig.show()

fig = px.line(grouped_italy_date, x="date", y="deaths", 
              title=f"Deaths in {plot_titles[1].upper()} Over Time", 
              color_discrete_sequence=['#91C4F2'],
              height=500
             )
fig.show()

fig = px.line(grouped_us_date, x="date", y="deaths", 
              title=f"Deaths in {plot_titles[2].upper()} Over Time", 
              color_discrete_sequence=['#6F2DBD'],
              height=500
             )
fig.show()

fig = px.line(grouped_rest_date, x="date", y="deaths", 
              title=f"Deaths in {plot_titles[3].upper()} Over Time", 
              color_discrete_sequence=['#FFDF64'],
              height=500
             )
fig.show()


# But deaths in China somehow seem to be decreased since March which is interesting.
# 
# Looking at the Deaths elsewhere is a saddening graph in itself. Terrible! 
# 
# Look at the graph of USA it is almost as if the curve is going into a straight like to the skies above. 

# In[ ]:


fig = px.choropleth(latest_grouped, locations="country", 
                    locationmode='country names', color="deaths", 
                    hover_name="deaths", range_color=[1,100], 
                    color_continuous_scale="peach", 
                    title='Countries with Reported Deaths')
# fig.update(layout_coloraxis_showscale=False)
fig.show()


# The Deaths however seems to come more from cetain parts of the World...China being the most significant of ones without a doubt, Europe and the Americas aren't doing any good I'm afraid.
# 
# Yet again it always feels like it's good to have a deeper look at Europe.

# In[ ]:


fig = px.choropleth(europe_grouped_latest, locations="country", 
                    locationmode='country names', color="deaths", 
                    hover_name="country", range_color=[1,100], 
                    color_continuous_scale='portland',
                    title='Reported Deaths in EUROPE', scope='europe', height=800)
# fig.update(layout_coloraxis_showscale=False)
fig.show()


# Looks like Spain, Italy and France are topping everyone again, this time for a very sad reason with UK close behind.
# 
# Iceland, Portugal and Ireland seem to be having much fewer deaths in comparision to their surrounding nations.

# In[ ]:


fig = px.bar(latest_grouped.sort_values('deaths', ascending=False)[:10][::-1], 
             x='deaths', y='country',
             title='Confirmed Deaths Worldwide', text='deaths', orientation='h')
fig.show()


# Italy so far have suffered the most. I remember 2 days back I was saying deaths in Italy has just crossed China's mark but look at that now. Overshadowing China and the rest of the world all by itself. Devastating to say the least!

# In[ ]:


fig = px.bar(europe_grouped_latest.sort_values('deaths', ascending=False)[:5][::-1], 
             x='deaths', y='country', color_discrete_sequence=['#84DCC6'],
             title='Deaths in Europe', text='deaths', orientation='h')
fig.show()


# In[ ]:


fig = px.bar(usa_latest.sort_values('deaths', ascending=False)[:5][::-1], 
             x='deaths', y='state', color_discrete_sequence=['#D63230'],
             title='Deaths in USA', text='deaths', orientation='h')
fig.show()


# Not a good time to be a New Yorkian I guess. Nightmare of a week for them.
# 
# Alarming signs for the US?

# ## 3. Active Cases Over Time

# Now lets analyze an important thing - ACTIVE CASES over time. It is the number of people who are affected by the virus excluding the dead and the recovered. This might have a bigger say on what is to come.

# In[ ]:


cleaned_data.rename(columns={'Date':'date'}, inplace=True)

grouped_china = cleaned_data[cleaned_data['country'] == "China"].reset_index()
grouped_china_date = grouped_china.groupby('date')['date', 'confirmed', 'deaths', 'active', 'recovered'].sum().reset_index()

grouped_italy = cleaned_data[cleaned_data['country'] == "Italy"].reset_index()
grouped_italy_date = grouped_italy.groupby('date')['date', 'confirmed', 'deaths', 'active', 'recovered'].sum().reset_index()

grouped_us = cleaned_data[cleaned_data['country'] == "US"].reset_index()
grouped_us_date = grouped_us.groupby('date')['date', 'confirmed', 'deaths', 'active', 'recovered'].sum().reset_index()

grouped_rest = cleaned_data[~cleaned_data['country'].isin(['China', 'Italy', 'US'])].reset_index()
grouped_rest_date = grouped_rest.groupby('date')['date', 'confirmed', 'deaths', 'active', 'recovered'].sum().reset_index()


# In[ ]:


plot_titles = ['China', 'Italy', 'USA', 'Rest of the World']

fig = px.line(grouped_china_date, x="date", y="active", 
              title=f"Active Cases in {plot_titles[0].upper()} Over Time", 
              color_discrete_sequence=['#F61067'],
              height=500
             )
fig.show()

fig = px.line(grouped_italy_date, x="date", y="active", 
              title=f"Active Cases in {plot_titles[1].upper()} Over Time", 
              color_discrete_sequence=['#91C4F2'],
              height=500
             )
fig.show()

fig = px.line(grouped_us_date, x="date", y="active", 
              title=f"Active Cases in {plot_titles[2].upper()} Over Time", 
              color_discrete_sequence=['#6F2DBD'],
              height=500
             )
fig.show()

fig = px.line(grouped_rest_date, x="date", y="active", 
              title=f"Active Cases in {plot_titles[3].upper()} Over Time", 
              color_discrete_sequence=['#FFDF64'],
              height=500
             )
fig.show()


# * Active cases in China have plummeted multifold over the past few days making it a much better sign for the country and its people. Something is working out there. Maybe the lockdown and the strict rules implied? Anyway there's a lot to learn from China this time.
# 
# * Active cases elsewhere have skyrocketed to be frank, which is deeply worrying. Especially USA again sporting an almost straight line graph to the skies. 

# In[ ]:


cleaned_data['state'] = cleaned_data['state'].fillna('')
temp = cleaned_data[[col for col in cleaned_data.columns if col != 'state']]

latest = temp[temp['date'] == max(temp['date'])].reset_index()
latest_grouped = latest.groupby('country')['confirmed', 'deaths', 'active', 'recovered'].sum().reset_index()


# In[ ]:


fig = px.choropleth(latest_grouped, locations="country", 
                    locationmode='country names', color="active", 
                    hover_name="active", range_color=[1,1000], 
                    color_continuous_scale="peach", 
                    title='Active Cases Worldwide')
# fig.update(layout_coloraxis_showscale=False)
fig.show()


# * Don't think China is safe just yet, it still looks to have more active cases than many of the World nations. And it's still a red hot region in terms of the virus.
# 
# * An interesting find from all these plots is that sub-saharan Africa, parts of Eastern Europe, Russia and India seem to have fewer cases than the rest of the world. 
# 
# * However things might get a lot worse when it hits Africa, it is better to keep it as far away as possible. 

# In[ ]:


europe = list(['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czechia','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Ireland',
               'Italy', 'Latvia','Luxembourg','Lithuania','Malta','Norway','Netherlands','Poland','Portugal','Romania','Slovakia','Slovenia',
               'Spain', 'Sweden', 'United Kingdom', 'Iceland', 'Russia', 'Switzerland', 'Serbia', 'Ukraine', 'Belarus',
               'Albania', 'Bosnia and Herzegovina', 'Kosovo', 'Moldova', 'Montenegro', 'North Macedonia'])

europe_grouped_latest = latest_grouped[latest_grouped['country'].isin(europe)]


# In[ ]:


fig = px.choropleth(europe_grouped_latest, locations="country", 
                    locationmode='country names', color="active", 
                    hover_name="country", range_color=[1,2000], 
                    color_continuous_scale='portland',
                    title='Active Cases European Countries', scope='europe', height=800)
# fig.update(layout_coloraxis_showscale=False)
fig.show()


# Western Europe again!

# In[ ]:


fig = px.bar(latest_grouped.sort_values('active', ascending=False)[:10][::-1], 
             x='active', y='country',
             title='Active Cases Worldwide', text='active', orientation='h')
fig.show()


# * It is very much possible to see things escalate in these few countries here even more with the rise in infected people.
# 
# * Much of what is to come over the next few days depend on what is going to happen here 
# 
# * Spain, US, France and Germany could possibly be as affected as Italy over the next few days if things don't change.

# In[ ]:


fig = px.bar(europe_grouped_latest.sort_values('active', ascending=False)[:10][::-1], 
             x='active', y='country',
             title='Active Cases EUROPE', text='active', orientation='h')
fig.show()


# In[ ]:


usa = cleaned_data[cleaned_data['country'] == "US"]
usa_latest = usa[usa['date'] == max(usa['date'])]
usa_latest = usa_latest.groupby('state')['confirmed', 'deaths', 'active', 'recovered'].max().reset_index()

fig = px.bar(usa_latest.sort_values('active', ascending=False)[:10][::-1], 
             x='active', y='state', color_discrete_sequence=['#D63230'],
             title='Active Cases in USA', text='active', orientation='h')
fig.show()


# New York the well known American state has more cases than many countries as a whole which is unbelievable. 

# ## 4. Recovered Cases 

# In[ ]:


fig = px.bar(latest_grouped.sort_values('recovered', ascending=False)[:10][::-1], 
             x='recovered', y='country',
             title='Recovered Cases Worldwide', text='recovered', orientation='h')
fig.show()


# * Wow this graph gives me some home at last, China has recovered a staggering number of patients somehow which might be the reason why they're doing relatively better at the moment.
# 
# * Even the ships have more recoveries than Japan and Singapore?
# 
# * Italy must do a lot better to neutralize the situation. In fact they must try to redo a China before its too late. The aging population might be a huge problem but lets hope things can change soon. 

# In[ ]:


fig = px.bar(europe_grouped_latest.sort_values('recovered', ascending=False)[:10][::-1], 
             x='recovered', y='country',
             title='Recovered Cases in EUROPE', text='recovered', orientation='h', color_discrete_sequence=['cyan'])
fig.show()


# Recoveries in France, UK and Germany got to be a lot better.

# ## 5. Comparisions
# 
# How about comparing the cases to better assess the situation

# In[ ]:


temp = cleaned_data.groupby('date')['recovered', 'deaths', 'active'].sum().reset_index()
temp = temp.melt(id_vars="date", value_vars=['recovered', 'deaths', 'active'],
                 var_name='case', value_name='count')


fig = px.line(temp, x="date", y="count", color='case',
             title='Cases over time: Line Plot', color_discrete_sequence = ['cyan', 'red', 'orange'])
fig.show()


fig = px.area(temp, x="date", y="count", color='case',
             title='Cases over time: Area Plot', color_discrete_sequence = ['cyan', 'red', 'orange'])
fig.show()


# * Active cases rising up completely all together leaving recoveries way behind and deaths also staring to increase and might see a big rise if the trend continues.
# 
# * China's recent recovery waves might be the reason for so much from this graph. 

# Taking China out of the equation to see the effects elsewhere.

# In[ ]:


rest = cleaned_data[cleaned_data['country'] != 'China']
rest_grouped = rest.groupby('date')['recovered', 'deaths', 'active'].sum().reset_index()

temp = rest_grouped.melt(id_vars="date", value_vars=['recovered', 'deaths', 'active'],
                 var_name='case', value_name='count')


fig = px.line(temp, x="date", y="count", color='case',
             title='Cases - Rest of the World: Line Plot', color_discrete_sequence = ['cyan', 'red', 'orange'])
fig.show()


fig = px.area(temp, x="date", y="count", color='case',
             title='Cases - Rest of the World: Area Plot', color_discrete_sequence = ['cyan', 'red', 'orange'])
fig.show()


# * Active cases skyrocketing!!! This is really bad news. It is almost as if the virus has just started for the rest of the World 
# 
# * Recoveries around the World are rock bottom if we take China out of the equation which is very interesting. What are others doing wrong?

# ## 6. Mortality and Recovery Rates

# It is worth seeing these stats as well. It might have a story for sure.

# In[ ]:


cleaned_latest = cleaned_data[cleaned_data['date'] == max(cleaned_data['date'])]
flg = cleaned_latest.groupby('country')['confirmed', 'deaths', 'recovered', 'active'].sum().reset_index()

flg['mortalityRate'] = round((flg['deaths']/flg['confirmed'])*100, 2)
temp = flg[flg['confirmed']>100]
temp = temp.sort_values('mortalityRate', ascending=False)

fig = px.bar(temp.sort_values(by="mortalityRate", ascending=False)[:10][::-1],
             x = 'mortalityRate', y = 'country', 
             title='Deaths per 100 Confirmed Cases', text='mortalityRate', height=800, orientation='h',
             color_discrete_sequence=['darkred']
            )
fig.show()


# * San Marino with the most brutal of them all. Almost 13 people for every 100 that get affected die!
# 
# * Indonesia seems to be very dangerous places to get affected in Asia.
# 
# * The drop of mortality rates in Phillipes from previous days in encouraging. 
# 
# * Death tolls in Italy have stormed up very high in the past few days. It is really worrying to see Italy suffering so much. 
# 
# * UK and Spain being the most notable of ones too.
# 
# > How many of you are surprised to see high mortality rates from well developed countries?

# In[ ]:


print("Countries with Lowest Mortality Rates")
temp = flg[flg['confirmed']>100]
temp = temp.sort_values('mortalityRate', ascending=True)[['country', 'confirmed','deaths']][:16]
temp.sort_values('confirmed', ascending=False)[['country', 'confirmed','deaths']][:20].style.background_gradient(cmap='Greens')


# * Well that's the Czech Republic with the lowest mortality rate despite having 1100+ cases.
# 
# * Israel, Finland, Thailand, Saudi, Qatar and Chile have some impressive numbers but it's worth remembering COVID-19 has an estimated mortality rate of only about 3%.
# 
# * Things are still worse even if deaths are low, the only thing that matters is the number of cases. Stop the Spread!

# In[ ]:


flg['recoveryRate'] = round((flg['recovered']/flg['confirmed'])*100, 2)
temp = flg[flg['confirmed']>100]
temp = temp.sort_values('recoveryRate', ascending=False)

fig = px.bar(temp.sort_values(by="recoveryRate", ascending=False)[:10][::-1],
             x = 'recoveryRate', y = 'country', 
             title='Recoveries per 100 Confirmed Cases', text='recoveryRate', height=800, orientation='h',
             color_discrete_sequence=['#2ca02c']
            )
fig.show()


# * China can recover an estimated 88 out of every 100 that get affected. That's great and the numbers seem to increase for them everyday.
# 
# * Surprised to see Ships here by the way. As pointed out by [Stanwar](https://www.kaggle.com/stanwar) in the comments down below - Diamond Princess Ship outbreak was a while ago now which may explain the relatively high recovery rate.
# 
# * Bahrain as we can see are doing really well in terms of recovering.
# 
# * Italy, Spain, US, France, UK and Germany... Where are they? Is that why they are in trouble now?

# In[ ]:


print("Countries with Worst Recovery Rates")
temp = flg[flg['confirmed']>100]
temp = temp.sort_values('recoveryRate', ascending=True)[['country', 'confirmed','recovered']][:20]
temp.sort_values('confirmed', ascending=False)[['country', 'confirmed','recovered']][:20].style.background_gradient(cmap='Reds')


# United States with 0 recoveries despite having 33k+ confirmed cases is staggering. What on Earth is that stat by the way for one of the most developed countries in the world. 

# ## How did it happen? 

# ### 1. Worldwide Analysis

# In[ ]:


formated_gdf = data.groupby(['date', 'country'])['confirmed', 'deaths'].max()
formated_gdf = formated_gdf.reset_index()
formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])
formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')
formated_gdf['size'] = formated_gdf['confirmed'].pow(0.3)

fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names', 
                     color="confirmed", size='size', hover_name="country", 
                     range_color= [0, 1500], 
                     projection="natural earth", animation_frame="date", 
                     title='COVID-19: Spread Over Time', color_continuous_scale="portland")
# fig.update(layout_coloraxis_showscale=False)
fig.show()


# * At the earliest point (from the data available) the disease seems to be only around China and its neighboring countries.
# 
# * However it quickly spread off to Europe, Autralia and even the US which is very interesting. 
# 
# * Things seem to be in fairly good light even in mid February for European countries.
# 
# * West Asia especially Iran and Iraq begins to catch fire at the end of February along with Italy showing signs of the dread to come. South Korea and China peaking at the moment.
# 
# * By March 5 look at Europe. They could've have locked down right at that moment. 
# 
# * The disease has taken away Africa and Americas too by early March with alarm bells ringing loudly for the US with just over 500 cases.
# 
# * Needless to say how it ended.
# 
# * According to the data so far, USA, UK, Spain, Italy, Germany, France and the UK are in deep trouble. Next few days are crucial for how the disease develops around the world.
# 
# 

# In[ ]:


formated_gdf = data.groupby(['date', 'country'])['confirmed', 'deaths'].max()
formated_gdf = formated_gdf.reset_index()
formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])
formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')
formated_gdf['size'] = formated_gdf['deaths'].pow(0.3)

fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names', 
                     color="deaths", size='size', hover_name="country", 
                     range_color= [0, 100], 
                     projection="natural earth", animation_frame="date", 
                     title='COVID-19: Deaths Over Time', color_continuous_scale="peach")
# fig.update(layout_coloraxis_showscale=False)
fig.show()


# The most interesting thing here is China's relative steadiness since March and Europe's severe spreads.  

# In[ ]:


formated_gdf = cleaned_data.groupby(['date', 'country'])['confirmed', 'deaths', 'active', 'recovered'].max()
formated_gdf = formated_gdf.reset_index()
formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])
formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')
formated_gdf['size'] = formated_gdf['active'].pow(0.3)

fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names', 
                     color="active", size='size', hover_name="country", 
                     range_color= [0, 1000], 
                     projection="natural earth", animation_frame="date", 
                     title='COVID-19: Active Cases Over Time', color_continuous_scale="portland")
fig.update(layout_coloraxis_showscale=False)
fig.show()


# Look at China's significant decrease and Europe's deadly increase at the same time.

# In[ ]:


formated_gdf = cleaned_data.groupby(['date', 'country'])['confirmed', 'deaths', 'active', 'recovered'].max()
formated_gdf = formated_gdf.reset_index()
formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])
formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')
formated_gdf['size'] = formated_gdf['recovered'].pow(0.3)

fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names', 
                     color="recovered", size='size', hover_name="country", 
                     range_color= [0, 100], 
                     projection="natural earth", animation_frame="date", 
                     title='COVID-19: Recoveries Over Time', color_continuous_scale="greens")
fig.update(layout_coloraxis_showscale=False)
fig.show()


# China's recoveries so far are excellent. Can Italy, Spain and Iran follow? 

# ### 2. What happened in Europe

# In[ ]:


formated_gdf = cleaned_data.groupby(['date', 'country'])['confirmed', 'deaths', 'active', 'recovered'].max()
formated_gdf = formated_gdf.reset_index()
formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])
formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')
formated_gdf['size'] = formated_gdf['confirmed'].pow(0.3) * 5

fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names', 
                     color="confirmed", size='size', hover_name="country", 
                     range_color= [0, 5000], 
                     projection="natural earth", animation_frame="date", scope="europe",
                     title='COVID-19: Spread Over Time in EUROPE', color_continuous_scale="portland", height=800)
# fig.update(layout_coloraxis_showscale=False)
fig.show()


# * What a suprise it looks to be France who are first affected by the virus.
# 
# * Nothing bad has happened in Europe as long as mid February. End of February, Italy inflates considerably. 
# 
# * Entire West Europe overshadowed by the virus by the mid of March. 

# In[ ]:


formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])
formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')
formated_gdf['size'] = formated_gdf['deaths'].pow(0.3)

fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names', 
                     color="deaths", size='size', hover_name="country", 
                     range_color= [0, 500], 
                     projection="natural earth", animation_frame="date", scope="europe",
                     title='COVID-19: Deaths Over Time in EUROPE', color_continuous_scale="peach", height=800)
# fig.update(layout_coloraxis_showscale=False)
fig.show()


# * First death again in France by Feb 17. Italy follows suite.
# 
# * The rest is history!

# In[ ]:


formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])
formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')
formated_gdf['size'] = formated_gdf['active'].pow(0.3) * 3.5

fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names', 
                     color="active", size='size', hover_name="country", 
                     range_color= [0, 3000], 
                     projection="natural earth", animation_frame="date", scope="europe",
                     title='COVID-19: Active Cases Over Time in EUROPE', color_continuous_scale="portland", height=800)
# fig.update(layout_coloraxis_showscale=False)
fig.show()


# * At one point in Feb most of the other countries seem to have no active cases except Italy.
# * But then there is no shortage of active cases in Italy and the lack of locking down public events causing the spread to climb so high and even affecting the rest of the Europe possibly.

# In[ ]:


formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])
formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')
formated_gdf['size'] = formated_gdf['recovered'].pow(0.3) * 3.5

fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names', 
                     color="recovered", size='size', hover_name="country", 
                     range_color= [0, 100], 
                     projection="natural earth", animation_frame="date", scope="europe",
                     title='COVID-19: Deaths Over Time in EUROPE', color_continuous_scale="greens", height=800)
# fig.update(layout_coloraxis_showscale=False)
fig.show()


# ## Conclusion
# 
# It is better to go for not spreading than to have someone recovered as we know how hard that is for new diseases so I think the message is clear for these countries. Social Distancing!!!
# 
# Now it matters more than ever!
# 
# I hope everything goes well in the next few days. Keep the virus at bay. Wash your hands, stay home, eat healthy and stay healthy.
# 
# **20-03-2020 | Happy International Happiness Day!** 
# 
# May things change for good very soon.
# 
# I really hope this notebook was useful. **Please give this notebook an upvote if you liked.** Thanks for reading all the way. 
# 
# <br>
# 
# > #### **Written By:** 
# [Abhinand Balachandran](www.kaggle.com/abhinand05)
