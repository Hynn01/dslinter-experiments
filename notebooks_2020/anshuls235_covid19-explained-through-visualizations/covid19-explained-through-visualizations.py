#!/usr/bin/env python
# coding: utf-8

# For a more detailed analysis on India refer to [India vs Coronavirus 🦠](https://www.kaggle.com/anshuls235/uncovering-covid19-india)

# # COVID19 Predictions using XGBOOST
# ![COVID19](https://media.giphy.com/media/gkF9T5vjfdoH0539ch/giphy.gif)
# 
# In this Project I'll be using past three months data to predict the Confirmed Cases and Fatalities for the month of April. The model used for training will be an **XGBOOST** model.
# ## <a id='main'>Table of Contents</a>
# - [Let's Explore the Data](#exp)
# - [Exploratory Data Analysis(EDA)](#eda)
#     1. [Universal growth of COVID19 over time](#world)
#     2. [Trend of COVID19 in top 10 affected countries](#top10)
#     3. [Mortality Rate](#dr)
#     4. [Country Specific growth of COVID19](#country)
#         - [United States of America](#us)
#         - [India](#in)
#         - [Italy](#it)
#         - [Spain](#sp)
#         - [China](#ch)
#         - [South Korea](#sk)
# - [Feature Engineering](#fe)
# - [Preprocessing](#pp)
# - [Training and Prediction](#tp)
# - [Forecast Visualizations](#for)

# # <a id='exp'>Lets explore the Data</a>

# In[ ]:


pip install pycountry_convert


# In[ ]:


#Libraries to import
import pandas as pd
import numpy as np
import datetime as dt
import requests
import sys
from itertools import chain
import pycountry
import pycountry_convert as pc
import plotly_express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import OrdinalEncoder
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv') 
df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')


# In[ ]:


df.rename(columns={'ObservationDate':'Date','Province/State':'Province_State',
                   'Country/Region':'Country_Region','Confirmed':'ConfirmedCases',
                   'Deaths':'Fatalities'},inplace=True)
df.loc[df['Country_Region']=='Mainland China','Country_Region']='China'
df['Date'] = pd.to_datetime(df['Date'],format='%m/%d/%Y')
df['Day'] = df.Date.dt.dayofyear
df['cases_lag_1'] = df.groupby(['Country_Region','Province_State'])['ConfirmedCases'].shift(1)
df['deaths_lag_1'] = df.groupby(['Country_Region','Province_State'])['Fatalities'].shift(1)
df['Daily Cases'] = df['ConfirmedCases'] - df['cases_lag_1']
df['Daily Deaths'] = df['Fatalities'] - df['deaths_lag_1']


# In[ ]:


display(df_train.head())
display(df_train.describe())
display(df_train.info())


# Currenty, the date is coming as a string. Lets convert it into datetime format so that EDA on the data becomes easier.

# In[ ]:


df_train['Date'] = pd.to_datetime(df_train['Date'], format = '%Y-%m-%d')
df_test['Date'] = pd.to_datetime(df_test['Date'], format = '%Y-%m-%d')


# In[ ]:


train_date_min = df_train['Date'].min()
train_date_max = df_train['Date'].max()
print('Minimum date from training set: {}'.format(train_date_min))
print('Maximum date from training set: {}'.format(train_date_max))


# In[ ]:


test_date_min = df_test['Date'].min()
test_date_max = df_test['Date'].max()
print('Minimum date from test set: {}'.format(test_date_min))
print('Maximum date from test set: {}'.format(test_date_max))


# # <a id='eda'>Exploratory Data Analysis(EDA)</a>
# [Go back to the main page](#main)
# 
# After exploring the data and its datatypes, let's perform some EDA on the data in order to get a better understanding of the data and how COVID19 is affecting all of us.

# # <a id='world'>Universal growth of COVID19 over time</a>
# In this section, I'll have a look at how COVID19 has been growing throughout the world from 22nd january 2020. I'll be using tree maps to show the share of COVID19 Cases worldwide and chloropleth maps with a time slider to show the daily impact of virus.  

# In[ ]:


class country_utils():
    def __init__(self):
        self.d = {}
    
    def get_dic(self):
        return self.d
    
    def get_country_details(self,country):
        """Returns country code(alpha_3) and continent"""
        try:
            country_obj = pycountry.countries.get(name=country)
            if country_obj is None:
                c = pycountry.countries.search_fuzzy(country)
                country_obj = c[0]
            continent_code = pc.country_alpha2_to_continent_code(country_obj.alpha_2)
            continent = pc.convert_continent_code_to_continent_name(continent_code)
            return country_obj.alpha_3, continent
        except:
            if 'Congo' in country:
                country = 'Congo'
            elif country == 'Diamond Princess' or country == 'Laos' or country == 'MS Zaandam'            or country == 'Holy See' or country == 'Timor-Leste':
                return country, country
            elif country == 'Korea, South' or country == 'South Korea':
                country = 'Korea, Republic of'
            elif country == 'Taiwan*':
                country = 'Taiwan'
            elif country == 'Burma':
                country = 'Myanmar'
            elif country == 'West Bank and Gaza':
                country = 'Gaza'
            else:
                return country, country
            country_obj = pycountry.countries.search_fuzzy(country)
            continent_code = pc.country_alpha2_to_continent_code(country_obj[0].alpha_2)
            continent = pc.convert_continent_code_to_continent_name(continent_code)
            return country_obj[0].alpha_3, continent
    
    def get_iso3(self, country):
        return self.d[country]['code']
    
    def get_continent(self,country):
        return self.d[country]['continent']
    
    def add_values(self,country):
        self.d[country] = {}
        self.d[country]['code'],self.d[country]['continent'] = self.get_country_details(country)
    
    def fetch_iso3(self,country):
        if country in self.d.keys():
            return self.get_iso3(country)
        else:
            self.add_values(country)
            return self.get_iso3(country)
        
    def fetch_continent(self,country):
        if country in self.d.keys():
            return self.get_continent(country)
        else:
            self.add_values(country)
            return self.get_continent(country)


# In[ ]:


df.ConfirmedCases = np.abs(df.ConfirmedCases)
df_tm = df.copy()
date = df_tm.Date.max()#get current date
df_tm = df_tm[df_tm['Date']==date]
obj = country_utils()
df_tm.Province_State.fillna('',inplace=True)
df_tm['continent'] = df_tm.apply(lambda x: obj.fetch_continent(x['Country_Region']), axis=1)
df_tm["world"] = "World" # in order to have a single root node
fig = px.treemap(df_tm, path=['world', 'continent', 'Country_Region'], values='ConfirmedCases',
                  color='ConfirmedCases', hover_data=['Country_Region'],
                  color_continuous_scale='dense', title='Current share of Worldwide COVID19 Cases')
fig.update_layout(width=700,template='seaborn')
fig.show()


# In[ ]:


fig = px.treemap(df_tm, path=['world', 'continent', 'Country_Region'], values='Fatalities',
                  color='Fatalities', hover_data=['Country_Region'],
                  color_continuous_scale='matter', title='Current share of Worldwide COVID19 Deaths')
fig.update_layout(width=700,template='seaborn')
fig.show()


# Confirmed Cases and Fatalities are cummulative sums of all the previous days. In order to understand the daily trend, I'll create a column for daily cases and deaths that will be the difference between the current value and the previous day's value.

# In[ ]:


def add_daily_measures(df):
    df.loc[0,'Daily Cases'] = df.loc[0,'ConfirmedCases']
    df.loc[0,'Daily Deaths'] = df.loc[0,'Fatalities']
    for i in range(1,len(df)):
        df.loc[i,'Daily Cases'] = df.loc[i,'ConfirmedCases'] - df.loc[i-1,'ConfirmedCases']
        df.loc[i,'Daily Deaths'] = df.loc[i,'Fatalities'] - df.loc[i-1,'Fatalities']
    #Make the first row as 0 because we don't know the previous value
    df.loc[0,'Daily Cases'] = 0
    df.loc[0,'Daily Deaths'] = 0
    return df


# In[ ]:


df_world = df.copy()
df_world = df_world.groupby('Date',as_index=False)['ConfirmedCases','Fatalities','Daily Cases','Daily Deaths'].sum()
df_world = add_daily_measures(df_world)


# In[ ]:


def draw_graph(df,x,y1,y2,title,days=7):
    colors = dict(case='#4285F4',death='#EA4335')
    df['cases_roll_avg'] = df[y1].rolling(days).mean()
    df['deaths_roll_avg'] = df[y2].rolling(days).mean()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(name='Daily Cases',x=df[x],y=df[y1],mode='lines',
                             line=dict(width=0.5,color=colors['case'])),
                 secondary_y=False)
    fig.add_trace(go.Scatter(name='Daily Deaths',x=df[x],y=df[y2],mode='lines',
                             line=dict(width=0.5,color=colors['death'])),
                 secondary_y=True)
    fig.add_trace(go.Scatter(name='Cases: <br>'+str(days)+'-Day Rolling average',
                             x=df[x],y=df['cases_roll_avg'],mode='lines',
                             line=dict(width=3,color=colors['case'])),
                 secondary_y=False)
    fig.add_trace(go.Scatter(name='Deaths: <br>'+str(days)+'-Day rolling average',
                             x=df[x],y=df['deaths_roll_avg'],mode='lines',
                             line=dict(width=3,color=colors['death'])),
                 secondary_y=True)
    
    fig.update_yaxes(title_text='Cases',title_font=dict(color=colors['case']),secondary_y=False,nticks=5,
                     tickfont=dict(color=colors['case']),linewidth=2,linecolor='black',gridcolor='darkgray',
                    zeroline=False)
    fig.update_yaxes(title_text='Deaths',title_font=dict(color=colors['death']),secondary_y=True,nticks=5,
                     tickfont=dict(color=colors['death']),linewidth=2,linecolor='black',gridcolor='darkgray',
                    zeroline=False)

    fig.update_layout(title=title,height=400,width=700,
                      margin=dict(l=0,r=0,t=60,b=30),hovermode='x',
                      legend=dict(x=0.01,y=0.99,bordercolor='black',borderwidth=1,bgcolor='#EED8E4',
                                  font=dict(family='arial',size=10)),
                     xaxis=dict(mirror=True,linewidth=2,linecolor='black',gridcolor='darkgray'),
                     plot_bgcolor='rgb(255,255,255)')
    return fig


# In[ ]:


fig = draw_graph(
    df_world,
    'Date',
    'Daily Cases',
    'Daily Deaths',
    '<b>Worldwide: Daily Cases & Deaths</b><br>   With 7-Day Rolling averages')
fig.show()


# In[ ]:


# fig = go.Figure(data=[
#     go.Bar(name='Cases', x=df_world['Date'], y=df_world['Daily Cases']),
#     go.Bar(name='Deaths', x=df_world['Date'], y=df_world['Daily Deaths'])
# ])

# fig.add_trace(go.Scatter(name='Cases:7-day rolling average',x=df_world['Date'],y=df_world['Cases:7-day rolling average'],marker_color='black'))
# fig.add_trace(go.Scatter(name='Deaths:7-day rolling average',x=df_world['Date'],y=df_world['Deaths:7-day rolling average'],marker_color='darkred'))

# # Change the bar mode
# fig.update_layout(barmode='overlay', title='Worldwide daily Case and Death count',hovermode='x',
#                   template='seaborn',width=700,xaxis=dict(mirror=True,linewidth=2,linecolor='black',showgrid=False),
#                  yaxis=dict(mirror=True,linewidth=2,linecolor='black'),legend=dict(orientation='h',x=0.1,y=-0.1))
# fig.show()


# In[ ]:


df_map = df.copy()
df_map['Date'] = df_map['Date'].astype(str)
df_map = df_map.groupby(['Date','Country_Region'], as_index=False)['ConfirmedCases','Fatalities'].sum()


# In[ ]:


df_map['iso_alpha'] = df_map.apply(lambda x: obj.fetch_iso3(x['Country_Region']), axis=1)


# In[ ]:


df_map['log(ConfirmedCases)'] = np.log(df_map.ConfirmedCases + 1)
df_map['log(Fatalities)'] = np.log(df_map.Fatalities + 1)


# >Since, cases and fatalities have grown exponentially over the last two months and countries like China, Italy, USA,and Spain, I have plotted the choropleth map on logarithmic scale. You can hover on the country to know the total confirmed cases or fatalities.

# In[ ]:


px.choropleth(df_map, 
              locations="iso_alpha", 
              color="log(ConfirmedCases)", 
              hover_name="Country_Region", 
              hover_data=["ConfirmedCases"] ,
              animation_frame="Date",
              color_continuous_scale=px.colors.sequential.dense, 
              title='Total Confirmed Cases growth(Logarithmic Scale)')


# In[ ]:


px.choropleth(df_map, 
              locations="iso_alpha", 
              color="log(Fatalities)", 
              hover_name="Country_Region",
              hover_data=["Fatalities"],
              animation_frame="Date",
              color_continuous_scale=px.colors.sequential.OrRd,
              title = 'Total Deaths growth(Logarithmic Scale)')


# Below are my finding from the above plots:-
# - China was the first country to experience the onset of virus.
# - US and Italy, which are the worst affected countries currently didn't recond many cases in january. This shows that how fast the virus spreads.
# - Majority of the cases are in the northern hemisphere, which is relatively cooler at this time of the year. Maybe the virus is temperature sensitive and as the summer progresses, we may see a fall in the growth of the cases in the northern hemisphere. Meaning not a good winter season for the southern hemisphere.
# - Western Europe is the worst affected. Hence, it can be adjudged as the new epicenter of COVID19. USA is also in the reckoning.
# - Lockdown has seem to have worked in China's favour as the growth rate has plummeted.

# # <a id='top10'>Trend of COVID19 in top 10 affected countries</a>
# I need to find the Top 10 affected countries. Since, the Confirmed cases and Fatalities are the cummulative sums till date, I'll find the top 10 countries by using the country data of the last date for which the training data is available.

# In[ ]:


#Get the top 10 countries
last_date = df.Date.max()
df_countries = df[df['Date']==last_date]
df_countries = df_countries.groupby('Country_Region', as_index=False)['ConfirmedCases','Fatalities'].sum()
df_countries = df_countries.nlargest(10,'ConfirmedCases')
#Get the trend for top 10 countries
df_trend = df.groupby(['Date','Country_Region'], as_index=False)['ConfirmedCases','Fatalities'].sum()
df_trend = df_trend.merge(df_countries, on='Country_Region')
df_trend.drop(['ConfirmedCases_y','Fatalities_y'],axis=1, inplace=True)
df_trend.rename(columns={'Country_Region':'Country', 'ConfirmedCases_x':'Cases', 'Fatalities_x':'Deaths'}, inplace=True)
#Add columns for studying logarithmic trends
df_trend['log(Cases)'] = np.log(df_trend['Cases']+1)# Added 1 to remove error due to log(0).
df_trend['log(Deaths)'] = np.log(df_trend['Deaths']+1)


# In[ ]:


fig = px.line(df_trend, x='Date', y='Cases', color='Country', title='COVID19 Total Cases growth for top 10 worst affected countries')
fig.update_layout(hovermode='closest',template='seaborn',width=700,xaxis=dict(mirror=True,linewidth=2,linecolor='black',showgrid=False),
                 yaxis=dict(mirror=True,linewidth=2,linecolor='black'))
fig.show()


# In[ ]:


fig = px.line(df_trend, x='Date', y='Deaths', color='Country', title='COVID19 Total Deaths growth for top 10 worst affected countries')
fig.update_layout(hovermode='closest',template='seaborn',width=700,xaxis=dict(mirror=True,linewidth=2,linecolor='black',showgrid=False),
                 yaxis=dict(mirror=True,linewidth=2,linecolor='black'))
fig.show()


# Below are my analysis from the above line plots for the top 10 affected countries:
# - Cases and Deaths for China have stagnated over time.
# - The cases and deaths are monotonically increasing(almost exponentially) for rest of the countries.
# - US has shown the greatest rise in the number of Confirmed Cases. Italy, on the other hand having the highest rise in deaths has to bear the brunt of the virus. Spain is a close second to Italy.
# - 6 out of the top 10 affected countries are Western European countries.

# Below, I have also plotted the daily variation of Confirmed Cases and Deaths for Top 10 affected countries on a logarithmic scale.

# In[ ]:


fig = px.line(df_trend, x='Date', y='log(Cases)', color='Country', title='COVID19 Total Cases growth for top 10 worst affected countries(Logarithmic Scale)')
fig.update_layout(hovermode='closest',template='seaborn',width=700,xaxis=dict(mirror=True,linewidth=2,linecolor='black',showgrid=False),
                 yaxis=dict(mirror=True,linewidth=2,linecolor='black'))
fig.show()


# In[ ]:


fig = px.line(df_trend, x='Date', y='log(Deaths)', color='Country', title='COVID19 Total Deaths growth for top 10 worst affected countries(Logarithmic Scale)')
fig.update_layout(hovermode='closest',template='seaborn',width=700,xaxis=dict(mirror=True,linewidth=2,linecolor='black',showgrid=False),
                 yaxis=dict(mirror=True,linewidth=2,linecolor='black'))
fig.show()


# # <a id='dr'>Mortality Rate</a>
# I'm calculating the mortality rate as the number of fatalities divided by the number of confirmed cases. Through the choropleth map I'll showcase the daily death rate for all the countries facing the COVID19 outbreak.

# In[ ]:


df_map['Mortality Rate%'] = round((df_map.Fatalities/df_map.ConfirmedCases)*100,2)


# In[ ]:


px.choropleth(df_map, 
                    locations="iso_alpha", 
                    color="Mortality Rate%", 
                    hover_name="Country_Region",
                    hover_data=["ConfirmedCases","Fatalities"],
                    animation_frame="Date",
                    color_continuous_scale=px.colors.sequential.Magma_r,
                    title = 'Worldwide Daily Variation of Mortality Rate%')


# In[ ]:


df_trend['Mortality Rate%'] = round((df_trend.Deaths/df_trend.Cases)*100,2)
fig = px.line(df_trend, x='Date', y='Mortality Rate%', color='Country', title='Variation of Mortality Rate% \n(Top 10 worst affected countries)')
fig.update_layout(hovermode='closest',template='seaborn',width=700,xaxis=dict(mirror=True,linewidth=2,linecolor='black',showgrid=False),
                 yaxis=dict(mirror=True,linewidth=2,linecolor='black'))
fig.show()


# > **Tip**: Click on Iran in the legends tab to hide it and get a better understanding of the plot.

# # <a id='country'>Country Specific growth of COVID19</a>

# # <a id='us'>United States of America</a>
# As can be seen through the below graphs: 
# - The COVID19 outbreak started from Washington state on the west coast and later on picked up pace in New york on the east coast . 
# - Now, New York itself has around 40% of the total cases in USA.
# - New Jersey is the second in the list of worst affected states.
# - Cases and Fatalities in the East Coast are more than that of West Coast's.

# In[ ]:


# Dictionary to get the state codes from state names for US
us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}


# In[ ]:


df_us = df[df['Country_Region']=='US']
df_us['Date'] = df_us['Date'].astype(str)
df_us['state_code'] = df_us.apply(lambda x: us_state_abbrev.get(x.Province_State,float('nan')), axis=1)
df_us['log(ConfirmedCases)'] = np.log(df_us.ConfirmedCases + 1)
df_us['log(Fatalities)'] = np.log(df_us.Fatalities + 1)


# In[ ]:


px.choropleth(df_us,
              locationmode="USA-states",
              scope="usa",
              locations="state_code",
              color="log(ConfirmedCases)",
              hover_name="Province_State",
              hover_data=["ConfirmedCases"],
              animation_frame="Date",
              color_continuous_scale=px.colors.sequential.Darkmint,
              title = 'Total Cases growth for USA(Logarithmic Scale)')


# In[ ]:


px.choropleth(df_us,
              locationmode="USA-states",
              scope="usa",
              locations="state_code",
              color="log(Fatalities)",
              hover_name="Province_State",
              hover_data=["Fatalities"],
              animation_frame="Date",
              color_continuous_scale=px.colors.sequential.OrRd,
              title = 'Total deaths growth for USA(Logarithmic Scale)')


# In[ ]:


df_usa = df.query("Country_Region=='US'")
df_usa = df_usa.groupby('Date',as_index=False)['ConfirmedCases','Fatalities','Daily Cases','Daily Deaths'].sum()
# #df_usa = add_daily_measures(df_usa)
# fig = go.Figure(data=[
#     go.Bar(name='Cases', x=df_usa['Date'], y=df_usa['Daily Cases']),
#     go.Bar(name='Deaths', x=df_usa['Date'], y=df_usa['Daily Deaths'])
# ])
# # Change the bar mode
# fig.update_layout(barmode='overlay', title='Daily Case and Death count(USA)')
# fig.update_layout(hovermode='closest',template='seaborn',width=700,xaxis=dict(mirror=True,linewidth=2,linecolor='black',showgrid=False),
#                  yaxis=dict(mirror=True,linewidth=2,linecolor='black'))
# fig.show()


# In[ ]:


fig = draw_graph(
    df_usa,
    'Date',
    'Daily Cases',
    'Daily Deaths',
    '<b>USA: Daily Cases & Deaths</b><br>   With 7-Day Rolling averages')
fig.show()


# # <a id='in'>India</a>
# COVID19 outbreak has started a bit late in India as compared to other countries. But, it has started to pick up pace. With limited testing and not a well funded healthcare system, India is surely up for a challenge. Let's hope that the 21 day lockdown helps to stop or atleast slower down the spread of this dreaded virus.

# In[ ]:


df.Province_State.fillna('NaN', inplace=True)
df_plot = df.groupby(['Date','Country_Region','Province_State'], as_index=False)['ConfirmedCases','Fatalities'].sum()


# In[ ]:


def add_daily_measures_country(df,country):
    df = df[df.Country_Region==country]
    df = df.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()
    df['Daily Cases'] = df['ConfirmedCases'] - df['ConfirmedCases'].shift(1)
    df['Daily Deaths'] = df['Fatalities'] - df['Fatalities'].shift(1)
    return df


# In[ ]:


df_ind = add_daily_measures_country(df_plot,'India')


# In[ ]:


fig = draw_graph(
    df_ind,
    'Date',
    'Daily Cases',
    'Daily Deaths',
    '<b>India: Daily Cases & Deaths</b><br>   With 7-Day Rolling averages')
fig.show()


# In[ ]:


df_ind_cases = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')
df_ind_cases.dropna(how='all',inplace=True)
df_ind_cases['DateTime'] = pd.to_datetime(df_ind_cases['Date'], format = '%d/%m/%y')


# In[ ]:


r = requests.get(url='https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson')
geojson = r.json()


# In[ ]:


def change_state_name(state):
    if state == 'Odisha':
        return 'Orissa'
    elif state == 'Telengana':
        return 'Telangana'
    return state


# In[ ]:


df_ind_cases['State/UnionTerritory'] = df_ind_cases.apply(lambda x: change_state_name(x['State/UnionTerritory']), axis=1)
last_date = df_ind_cases.DateTime.max()
df_ind_states = df_ind_cases.copy()
df_ind_cases = df_ind_cases[df_ind_cases['DateTime']==last_date]


# **Here's a state wise breakdown of cases, deaths and recoveries.**

# In[ ]:


columns = ['State/UnionTerritory', 'Cured', 'Deaths','Confirmed']
df_ind_cases = df_ind_cases[columns]
df_ind_cases.sort_values('Confirmed',inplace=True, ascending=False)
df_ind_cases.reset_index(drop=True,inplace=True)
df_ind_cases.style.background_gradient(cmap='viridis')


# In[ ]:


fig = px.choropleth(df_ind_cases, geojson=geojson, color="Confirmed",
                    locations="State/UnionTerritory", featureidkey="properties.NAME_1",
                    hover_data=['Cured','Deaths'],
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title='India: Total Current cases per state'
                   )
fig.update_geos(fitbounds="locations", visible=True)
fig.update_geos(projection_type="orthographic")
fig.update_layout(height=600,margin={"r":0,"t":30,"l":0,"b":30})
fig.show()


# > Hover on the map to view Recoveries and Deaths. 

# In[ ]:


fig = px.line(df_ind_states, x='DateTime', y='Confirmed', color='State/UnionTerritory', title='India: State-wise cases')
fig.update_layout(hovermode='closest',template='seaborn',width=700,xaxis=dict(mirror=True,linewidth=2,linecolor='black',showgrid=False),
                 yaxis=dict(mirror=True,linewidth=2,linecolor='black'))
fig.show()


# # <a id='it'>Italy</a>
# This is the country that has had to bear the brunt of the virus the most. It has reported the maximum number of deaths out of all the countries till now. The whole country lockdown was imposed in italy on 9th of march. Let's see how italy is fairing now.

# In[ ]:


df_ita = add_daily_measures_country(df_plot,'Italy')
fig = go.Figure(data=[
    go.Bar(name='Cases', x=df_ita['Date'], y=df_ita['Daily Cases']),
    go.Bar(name='Deaths', x=df_ita['Date'], y=df_ita['Daily Deaths'])
])
# Change the bar mode
fig.update_layout(barmode='overlay', title='Daily Case and Death count(Italy)',
                 annotations=[dict(x='2020-03-09', y=1797, xref="x", yref="y", text="Lockdown<br>Imposed<br>(9th March)", showarrow=True, arrowhead=1, ax=-50, ay=-50)])
fig.update_layout(hovermode='closest',template='seaborn',width=700,xaxis=dict(mirror=True,linewidth=2,linecolor='black',showgrid=False),
                 yaxis=dict(mirror=True,linewidth=2,linecolor='black'))
fig.show()


# Here's a Region wise data for Italy. By the look of it, **Lombardy** region in Northern Italy has been the hardest hit. 

# In[ ]:


df_ita_cases = pd.read_csv('/kaggle/input/covid19-in-italy/covid19_italy_region.csv')
df_ita_cases['Date'] = pd.to_datetime(df_ita_cases['Date'], format='%Y-%m-%d')
df_ita_cases['Date'] = [d.date() for d in df_ita_cases['Date']]


# In[ ]:


df_ita_group = df_ita_cases.groupby(['Date','RegionCode','RegionName','Latitude','Longitude'],as_index=False)['HospitalizedPatients','IntensiveCarePatients','TotalHospitalizedPatients','HomeConfinement','CurrentPositiveCases' ,'NewPositiveCases','Recovered','Deaths','TotalPositiveCases','TestsPerformed'].sum()


# In[ ]:


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("mapboxtoken")


# In[ ]:


curr_date = df_ita_group.Date.max()
df_ita_curr = df_ita_cases[df_ita_cases['Date']==curr_date]


# In[ ]:


columns=['RegionName','TotalHospitalizedPatients','Recovered', 'Deaths', 'TotalPositiveCases','TestsPerformed']
df_ita_temp = df_ita_curr[columns]
df_ita_temp.sort_values('TotalPositiveCases',inplace=True, ascending=False)
df_ita_temp.reset_index(drop=True,inplace=True)
df_ita_temp.style.background_gradient(cmap='Blues_r')


# Below plot is based on the above table. Colorscale has also been kept same to avoid confusion. Hover on the marker to know rest of the metrics. The plot shows Total Positive cases region wise(Represented through size of the marker) and Total tests performed till now(Represented by the 'Blue_r' colorscale).

# In[ ]:


px.set_mapbox_access_token(secret_value_0)
df_ita_cases['Date'] = df_ita_cases['Date'].astype('str')
fig = px.scatter_mapbox(df_ita_cases,
                        lat="Latitude",
                        lon="Longitude",
                        size="TotalPositiveCases",
                        color='TestsPerformed',
                        mapbox_style='streets',
                        color_continuous_scale=px.colors.sequential.Blues_r,
                        animation_frame='Date',
                        hover_name='RegionName',
                        hover_data=['HospitalizedPatients','IntensiveCarePatients','TotalHospitalizedPatients','HomeConfinement','CurrentPositiveCases'\
 ,'NewPositiveCases','Recovered','Deaths'],
                        zoom=4,
                        size_max=50,
                        title= 'Italy:Daily COVID19 Cases and Test performed')
fig.show()


# In[ ]:


region_colors = {
    'Abruzzo': 'skyblue', 
    'Basilicata': 'gold', 
    'P.A. Bolzano': 'lightseagreen', 
    'Calabria': 'black', 
    'Campania': 'crimson',
    'Emilia-Romagna': 'darkred', 
    'Friuli Venezia Giulia': 'mistyrose', 
    'Lazio': 'lavender', 
    'Liguria': 'wheat',
    'Lombardia': 'red', 
    'Marche': 'green', 
    'Molise': 'yellow', 
    'Piemonte': 'blue', 
    'Puglia': 'purple', 
    'Sardegna': 'peachpuff',
    'Sicilia': 'lightcoral', 
    'Toscana': 'lightpink', 
    'P.A. Trento': 'fuchsia', 
    'Umbria': 'firebrick', 
    "Valle d'Aosta": 'cyan',
    'Veneto': 'darkblue'
} 


# In[ ]:


fig = make_subplots(
    rows=5, cols=1, 
    shared_xaxes=True,
    vertical_spacing=0.009,
    x_title='Date'
)

for region in df_ita_group.RegionName.unique():
    df_region = df_ita_group[df_ita_group['RegionName']==region]
    #Add Traces
    fig.add_trace(go.Scatter(name=region,x=df_region['Date'], y=df_region['TotalHospitalizedPatients'], legendgroup=region, line=dict(color=region_colors[region])), row=1, col=1)
    fig.add_trace(go.Scatter(name=region,x=df_region['Date'], y=df_region['Recovered'], legendgroup=region, showlegend=False, line=dict(color=region_colors[region])), row=2, col=1)
    fig.add_trace(go.Scatter(name=region,x=df_region['Date'], y=df_region['Deaths'], legendgroup=region, showlegend=False, line=dict(color=region_colors[region])), row=3, col=1)
    fig.add_trace(go.Scatter(name=region,x=df_region['Date'], y=df_region['TotalPositiveCases'], legendgroup=region, showlegend=False, line=dict(color=region_colors[region])), row=4, col=1)
    fig.add_trace(go.Scatter(name=region,x=df_region['Date'], y=df_region['TestsPerformed'], legendgroup=region, showlegend=False, line=dict(color=region_colors[region])), row=5, col=1)

# Update yaxis properties
fig.update_yaxes(title_text="Hospitalizations", row=1, col=1)
fig.update_yaxes(title_text="Recoveries", row=2, col=1)
fig.update_yaxes(title_text="Deaths", row=3, col=1)
fig.update_yaxes(title_text="Positive Cases", row=4, col=1)
fig.update_yaxes(title_text="Tests Performed", row=5, col=1)

# Update title and height
fig.update_layout(title_text="Italy: Daily Metrics per Region", height=1000, showlegend=True)

fig.show()


# # <a id='sp'>Spain</a>
# The second worst affetced country in Europe after Italy. It has reported the maximum number of cases out of all the countries in Europe till now and secondmost worldwise. The whole country lockdown was imposed in Spain on 15th of march. Let's see how Spain is fairing now.

# In[ ]:


df_esp = add_daily_measures_country(df_plot,'Spain')
fig = go.Figure(data=[
    go.Bar(name='Cases', x=df_esp['Date'], y=df_esp['Daily Cases']),
    go.Bar(name='Deaths', x=df_esp['Date'], y=df_esp['Daily Deaths'])
])
# Change the bar mode
fig.update_layout(barmode='overlay', title='Daily Case and Death count(Spain)',
                 annotations=[dict(x='2020-03-15', y=1407, xref="x", yref="y", text="Lockdown<br>Imposed(15th March)", showarrow=True, arrowhead=1, ax=-50, ay=-50)])
fig.update_layout(hovermode='closest',template='seaborn',width=700,xaxis=dict(mirror=True,linewidth=2,linecolor='black',showgrid=False),
                 yaxis=dict(mirror=True,linewidth=2,linecolor='black'))
fig.show()


# #### <a id='ch'>China</a>
# - This is where it all started! By looking at the graph it can be seen that China has been able to almost stop the spread of COVID19 substantially.
# - Almost all the cases are from the Hubei Province which can be attributed to the fact that the outbreak started from its capital, Wuhan.
# 
# > In order to get a better understanding of the cases/fatalities growth from other provinces, you can click on Hubei in the legend so that it gets hidden and the scale will autoscale.

# In[ ]:


df_ch = df_plot.query("Country_Region=='China'")
fig = px.line(df_ch, x='Date', y='ConfirmedCases', color='Province_State', title='Total Cases growth for China')
fig.update_layout(hovermode='closest',template='seaborn',width=700,xaxis=dict(mirror=True,linewidth=2,linecolor='black',showgrid=False),
                 yaxis=dict(mirror=True,linewidth=2,linecolor='black'))
fig.show()


# In[ ]:


fig = px.line(df_ch, x='Date', y='Fatalities', color='Province_State', title='Total Deaths growth for China')
fig.update_layout(hovermode='closest',template='seaborn',width=700,xaxis=dict(mirror=True,linewidth=2,linecolor='black',showgrid=False),
                 yaxis=dict(mirror=True,linewidth=2,linecolor='black'))
fig.show()


# In[ ]:


r = requests.get(url='https://raw.githubusercontent.com/deldersveld/topojson/master/countries/china/china-provinces.json')
topology = r.json()


# In[ ]:


#Convert topology json into geojson
#The code is from https://gist.github.com/perrygeo/1e767e42e8bc54ad7262
def rel2abs(arc, scale=None, translate=None):
    """Yields absolute coordinate tuples from a delta-encoded arc.
    If either the scale or translate parameter evaluate to False, yield the
    arc coordinates with no transformation."""
    if scale and translate:
        a, b = 0, 0
        for ax, bx in arc:
            a += ax
            b += bx
            yield scale[0]*a + translate[0], scale[1]*b + translate[1]
    else:
        for x, y in arc:
            yield x, y

def coordinates(arcs, topology_arcs, scale=None, translate=None):
    """Return GeoJSON coordinates for the sequence(s) of arcs.
    
    The arcs parameter may be a sequence of ints, each the index of a
    coordinate sequence within topology_arcs
    within the entire topology -- describing a line string, a sequence of 
    such sequences -- describing a polygon, or a sequence of polygon arcs.
    
    The topology_arcs parameter is a list of the shared, absolute or
    delta-encoded arcs in the dataset.
    The scale and translate parameters are used to convert from delta-encoded
    to absolute coordinates. They are 2-tuples and are usually provided by
    a TopoJSON dataset. 
    """
    if isinstance(arcs[0], int):
        coords = [
            list(
                rel2abs(
                    topology_arcs[arc if arc >= 0 else ~arc],
                    scale, 
                    translate )
                 )[::arc >= 0 or -1][i > 0:] \
            for i, arc in enumerate(arcs) ]
        return list(chain.from_iterable(coords))
    elif isinstance(arcs[0], (list, tuple)):
        return list(
            coordinates(arc, topology_arcs, scale, translate) for arc in arcs)
    else:
        raise ValueError("Invalid input %s", arcs)

def geometry(obj, topology_arcs, scale=None, translate=None):
    """Converts a topology object to a geometry object.
    
    The topology object is a dict with 'type' and 'arcs' items, such as
    {'type': "LineString", 'arcs': [0, 1, 2]}.
    See the coordinates() function for a description of the other three
    parameters.
    """
    return {
        "type": obj['type'], 
        "coordinates": coordinates(
            obj['arcs'], topology_arcs, scale, translate )}

from shapely.geometry import asShape

topojson_path = sys.argv[1]
geojson_path = sys.argv[2]


# file can be renamed, the first 'object' is more reliable
layername = list(topology['objects'].keys())[0]  

features = topology['objects'][layername]['geometries']
scale = topology['transform']['scale']
trans = topology['transform']['translate']

fc = {'type': "FeatureCollection", 'features': []}

for id, tf in enumerate(features):
    f = {'id': id, 'type': "Feature"}
    f['properties'] = tf['properties'].copy()

    geommap = geometry(tf, topology['arcs'], scale, trans)
    geom = asShape(geommap).buffer(0)
    assert geom.is_valid
    f['geometry'] = geom.__geo_interface__

    fc['features'].append(f) 


# In[ ]:


df_ch = df_ch[df_ch['Date']==df.Date.max()]


# In[ ]:


fig = px.choropleth(df_ch,
                    geojson=fc,
                    locations='Province_State',
                    featureidkey="properties.NAME_1",
                    color_continuous_scale=px.colors.sequential.Darkmint,
                    hover_name='Province_State',
                    range_color=(0, df_ch['ConfirmedCases'].max()),
                    color='ConfirmedCases', 
                    title='China: Total Current Cases per Province'
                   )

fig.update_geos(fitbounds="locations", visible=True)
fig.update_geos(projection_type="orthographic")
fig.update_layout(height=600, margin={"r":0,"t":30,"l":0,"b":0})
fig.show()


# In[ ]:


fig = px.choropleth(df_ch,
                    geojson=fc,
                    locations='Province_State',
                    featureidkey="properties.NAME_1",
                    color_continuous_scale=px.colors.sequential.OrRd,
                    hover_name='Province_State',
                    range_color=(0, df_ch['Fatalities'].max()),
                    color='Fatalities', 
                    title='China: Total Current Deaths per Province'
                   )

fig.update_geos(fitbounds="locations", visible=True)
fig.update_geos(projection_type="orthographic")
fig.update_layout(height=600, margin={"r":0,"t":30,"l":0,"b":0})
fig.show()


# In[ ]:


df_ch = add_daily_measures_country(df_plot,'China')
fig = go.Figure(data=[
    go.Bar(name='Cases', x=df_ch['Date'], y=df_ch['Daily Cases']),
    go.Bar(name='Deaths', x=df_ch['Date'], y=df_ch['Daily Deaths'])
])
# Change the bar mode
fig.update_layout(barmode='overlay', title='Daily Case and Death count(China)')
fig.update_layout(hovermode='closest',template='seaborn',width=700,xaxis=dict(mirror=True,linewidth=2,linecolor='black',showgrid=False),
                 yaxis=dict(mirror=True,linewidth=2,linecolor='black'))
fig.show()


# # <a id='sk'>South Korea</a>

# In[ ]:


df_sk = add_daily_measures_country(df_plot,'South Korea')
fig = go.Figure(data=[
    go.Bar(name='Cases', x=df_sk['Date'], y=df_sk['Daily Cases']),
    go.Bar(name='Deaths', x=df_sk['Date'], y=df_sk['Daily Deaths'])
])
# Change the bar mode
fig.update_layout(barmode='overlay', title='Daily Case and Death count(South Korea)')
fig.update_layout(hovermode='closest',template='seaborn',width=700,xaxis=dict(mirror=True,linewidth=2,linecolor='black',showgrid=False),
                 yaxis=dict(mirror=True,linewidth=2,linecolor='black'))
fig.show()


# In[ ]:


df_sk_cases = pd.read_csv('/kaggle/input/coronavirusdataset/Case.csv')


# In[ ]:


df_hotspots = df_sk_cases[(df_sk_cases['group']==True)&(df_sk_cases['city']!='from other city')]
df_hotspots['latitude'] = df_hotspots.apply(lambda x: float(x['latitude'] if x['latitude']!='-' else float('nan'))
                                            , axis=1)
df_hotspots['longitude'] = df_hotspots.apply(lambda x: float(x['longitude'] if x['longitude']!='-' else float('nan'))
                                             , axis=1)


# In[ ]:


px.set_mapbox_access_token(secret_value_0)
fig = px.scatter_mapbox(df_hotspots,
                        lat="latitude",
                        lon="longitude",
                        size="confirmed",
                        hover_data=['infection_case','city','province'],
                        mapbox_style='streets',
                        zoom=5,
                        size_max=50,
                        title= 'COVID19 Hotspots in South Korea')
fig.show()


# In[ ]:


df_sk = pd.read_csv('/kaggle/input/coronavirusdataset/PatientInfo.csv')
df_sk.age.replace('66s','60s', inplace=True)


# In[ ]:


df_sk_age = df_sk.groupby(['age','sex','country','province','city','infection_case'], as_index=False)['patient_id'].count()


# In[ ]:


# Initialize figure with subplots
fig = make_subplots(
    rows=2, cols=2, subplot_titles=("Age-wise distribution of Cases", "Province-wise distribution of Cases"\
                                    , "Infection Origin", "Gender-wise distribution of Patient Statuses"),
    specs=[[{"type": "bar"}, {"type": "pie"}],
           [{"type": "pie"}, {"type": "bar"}]],
)

# Add traces
# trace-1
df_males = df_sk_age[df_sk_age['sex']=='male'].groupby('age',as_index=False)['patient_id'].sum()
df_females = df_sk_age[df_sk_age['sex']=='female'].groupby('age',as_index=False)['patient_id'].sum()
fig.add_trace(go.Bar(name='Males', x=df_males['age'], y=df_males['patient_id']), row=1, col=1)
fig.add_trace(go.Bar(name='Females', x=df_females['age'], y=df_females['patient_id']), row=1, col=1)
# trace-2
df_province = df_sk_age.groupby('province',as_index=False)['patient_id'].sum()
fig.add_trace(go.Pie(labels=df_province['province'], values=df_province['patient_id'], hole=0.3), row=1, col=2)
# trace-3
df_inf_case = df_sk_age.groupby('infection_case', as_index=False)['patient_id'].sum()
df_inf_case.loc[df_inf_case['patient_id'] < 50, 'infection_case'] = 'etc'
fig.add_trace(go.Pie(labels=df_inf_case['infection_case'],values=df_inf_case['patient_id'], hole=0.3), row=2, col=1)
# trace-4
df_males = df_sk[df_sk['sex']=='male'].groupby('state',as_index=False)['patient_id'].sum()
df_females = df_sk[df_sk['sex']=='female'].groupby('state',as_index=False)['patient_id'].sum()
fig.add_trace(go.Bar(name='Males', x=df_males['state'], y=df_males['patient_id']), row=2, col=2)
fig.add_trace(go.Bar(name='Females', x=df_females['state'], y=df_females['patient_id']), row=2, col=2)

# Update xaxis properties
fig.update_xaxes(title_text="Age", row=1, col=1)
fig.update_xaxes(title_text="Status", row=2, col=2)

# Update yaxis properties
fig.update_yaxes(title_text="Confirmed Cases", row=1, col=1)
fig.update_yaxes(title_text="Total number of People", row=2, col=2)

# Update title and height
fig.update_layout(title_text="South Korea: Some more visualizations", height=700, showlegend=False)

fig.show()


# # <a id='fe'>Feature Engineering</a>
# [Go back to the main page](#main)

# In[ ]:


df_pd = pd.read_csv('/kaggle/input/countries-dataset-2020/Pupulation density by countries.csv') 
df_pd['iso_code3'] = df_pd.apply(lambda x: obj.fetch_iso3(x['Country (or dependent territory)'].strip()), axis=1)
df = df_train[df_train['Date']==train_date_max]
#df = df_train.copy()
df = df.groupby(['Date','Country_Region'], as_index=False)['ConfirmedCases','Fatalities'].sum()
df['iso_code3'] = df.apply(lambda x:obj.fetch_iso3(x['Country_Region']), axis=1)
df = df.merge(df_pd, how='left', on='iso_code3')


# In[ ]:


def convert(pop):
    if pop == float('nan'):
        return 0.0
    return float(pop.replace(',',''))

df['Population'].fillna('0', inplace=True)
df['Population'] = df.apply(lambda x: convert(x['Population']),axis=1)
df['Density pop./km2'].fillna('0', inplace=True)
df['Density pop./km2'] = df.apply(lambda x: convert(x['Density pop./km2']),axis=1)


# In[ ]:


q3 = np.percentile(df.ConfirmedCases,75)
q1 = np.percentile(df.ConfirmedCases,25)
IQR = q3-q1
low = q1 - 1.5*IQR
high = q3 + 1.3*IQR
df = df[(df['ConfirmedCases']>low) & (df['ConfirmedCases']<high)]
df['continent'] = df.apply(lambda x: obj.fetch_continent(x['Country_Region']), axis=1)


# In[ ]:


df['Date_x'] = df['Date_x'].astype(str)


# In[ ]:


px.scatter(df,x='ConfirmedCases',y='Density pop./km2', size = 'Population', color='continent',hover_data=['Country_Region'], title='Variation of Population density wrt Confirmed Cases',range_y=[0,1500])


# In[ ]:


px.scatter(df,x='Fatalities',y='Density pop./km2', size = 'Population', color='continent',hover_data=['Country_Region'],title='Variation of Population density wrt Fatalities',range_y=[0,1500])


# In[ ]:


df.corr()


# # <a id='pp'>Preprocessing</a>
# [Go back to the main page](#main)
# 
# Convert Categorical variables: **Province_State** & **Country_Region**, into integers for training the model.
# 
# Province_State contains Null values. As I need null values as well as this feature is needed for training, I'll convert Null values to string 'NaN'. Now, OrdinalEncoder() can be easily applied on it.

# In[ ]:


#Add continent column to training set
df_train['Continent'] = df_train.apply(lambda X: obj.fetch_continent(X['Country_Region']), axis=1)


# In[ ]:


def categoricalToInteger(df):
    #convert NaN Province State values to a string
    df.Province_State.fillna('NaN', inplace=True)
    #Define Ordinal Encoder Model
    oe = OrdinalEncoder()
    df[['Province_State','Country_Region','Continent']] = oe.fit_transform(df.loc[:,['Province_State','Country_Region','Continent']])
    return df


# Extract useful features from date.

# In[ ]:


def create_features(df):
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['quarter'] = df['Date'].dt.quarter
    df['weekofyear'] = df['Date'].dt.weekofyear
    return df


# Split the training data into train and dev set for cross-validation.

# In[ ]:


def train_dev_split(df, days):
    #Last days data as dev set
    date = df['Date'].max() - dt.timedelta(days=days)
    return df[df['Date'] <= date], df[df['Date'] > date]


# In order to avoid data leakage, there should be no overlap between the data in the training and test set. Therefore, I'll remove the data from training set having dates that are already present in the test set.

# In[ ]:


def avoid_data_leakage(df, date=test_date_min):
    return df[df['Date']<date]


# In[ ]:


df_train = avoid_data_leakage(df_train)
df_train = categoricalToInteger(df_train)
df_train = create_features(df_train)


# In[ ]:


df_train, df_dev = train_dev_split(df_train,0)


# Select all the columns that are needed for training the model.

# In[ ]:


columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear','Province_State', 'Country_Region','Continent','ConfirmedCases','Fatalities']
df_train = df_train[columns]
df_dev = df_dev[columns]


# # <a id='tp'>Training and Prediction</a>
# [Go back to the main page](#main)
# 
# In this section, I'll training the data on an XGBOOST model . Since, I have to predict both: **Confirmed Cases** & **Fatalities**, I'll be using 2 separate models.
# 
# Apply the same transformation to the test set that were applied to the training set.

# In[ ]:


#Apply the same transformation to test set that were applied to the training set
df_test['Continent'] = df_test.apply(lambda X: obj.fetch_continent(X['Country_Region']), axis=1)
df_test = categoricalToInteger(df_test)
df_test = create_features(df_test)
#Columns to select
columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear','Province_State', 'Country_Region','Continent']


# In[ ]:


submission = []
#Loop through all the unique countries
for country in df_train.Country_Region.unique():
    #Filter on the basis of country
    df_train1 = df_train[df_train["Country_Region"]==country]
    #Loop through all the States of the selected country
    for state in df_train1.Province_State.unique():
        #Filter on the basis of state
        df_train2 = df_train1[df_train1["Province_State"]==state]
        #Convert to numpy array for training
        train = df_train2.values
        #Separate the features and labels
        X_train, y_train = train[:,:-2], train[:,-2:]
        #model1 for predicting Confirmed Cases
        model1 = XGBRegressor(n_estimators=1000)
        model1.fit(X_train, y_train[:,0])
        #model2 for predicting Fatalities
        model2 = XGBRegressor(n_estimators=1000)
        model2.fit(X_train, y_train[:,1])
        #Get the test data for that particular country and state
        df_test1 = df_test[(df_test["Country_Region"]==country) & (df_test["Province_State"] == state)]
        #Store the ForecastId separately
        ForecastId = df_test1.ForecastId.values
        #Remove the unwanted columns
        df_test2 = df_test1[columns]
        #Get the predictions
        y_pred1 = model1.predict(df_test2.values)
        y_pred2 = model2.predict(df_test2.values)
        #Append the predicted values to submission list
        for i in range(len(y_pred1)):
            d = {'ForecastId':ForecastId[i], 'ConfirmedCases':y_pred1[i], 'Fatalities':y_pred2[i]}
            submission.append(d)


# Convert the submission list to DataFrame and save it as csv for submission

# In[ ]:


df_submit = pd.DataFrame(submission)


# In[ ]:


df_submit.to_csv(r'submission.csv', index=False)


# # <a id='for'>Forecast Vizualizations</a>
# [Go back to the main page](#main)

# In[ ]:


df_forcast = pd.concat([df_test,df_submit.iloc[:,1:]], axis=1)
df_world_f = df_forcast.copy()
df_world_f = df_world_f.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()
df_world_f = add_daily_measures(df_world_f)


# In[ ]:


df_world = avoid_data_leakage(df_world)


# In[ ]:


fig = go.Figure(data=[
    go.Bar(name='Total Cases', x=df_world['Date'], y=df_world['ConfirmedCases']),
    go.Bar(name='Total Cases Forecasted', x=df_world_f['Date'], y=df_world_f['ConfirmedCases'])
])
# Change the bar mode
fig.update_layout(barmode='group', title='Worldwide Confirmed Cases + Forecasted Cases')
fig.show()


# In[ ]:


fig = go.Figure(data=[
    go.Bar(name='Total Deaths', x=df_world['Date'], y=df_world['Fatalities']),
    go.Bar(name='Total Deaths Forecasted', x=df_world_f['Date'], y=df_world_f['Fatalities'])
])
# Change the bar mode
fig.update_layout(barmode='group', title='Worldwide Deaths + Forecasted Deaths')
fig.show()

