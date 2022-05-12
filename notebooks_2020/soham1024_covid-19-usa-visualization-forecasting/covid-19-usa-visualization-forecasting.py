#!/usr/bin/env python
# coding: utf-8

# # Introduction

# ![](https://dlg7f0e93aole.cloudfront.net/wp-content/uploads/coronavirus-image-750x500.jpg)

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from scipy.integrate import odeint

from plotly.offline import iplot, init_notebook_mode
import math
import bokeh 
import matplotlib.pyplot as plt
import plotly.express as px
from urllib.request import urlopen
import json
from dateutil import parser
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import row, column
from bokeh.resources import INLINE
from bokeh.io import output_notebook
from bokeh.models import Span
import warnings
warnings.filterwarnings("ignore")
output_notebook(resources=INLINE)


# # Global

# ![](https://www.washingtonpost.com/graphics/2020/health/coronavirus-how-epidemics-spread-and-end/img/sigsim2.gif)

# In[ ]:


import plotly as py
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=False)


# # Time plot 

# In[ ]:


corona_data=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
choro_map=px.choropleth(corona_data, 
                    locations="Country/Region", 
                    locationmode = "country names",
                    color="Confirmed", 
                    hover_name="Country/Region", 
                    animation_frame="ObservationDate"
                   )

choro_map.update_layout(
    title_text = 'Global Spread of Coronavirus',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
    
choro_map.show()


# # The Story of COVID-19 in USA

# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/COVID-19_outbreak_USA_per_capita_cases_map.svg/800px-COVID-19_outbreak_USA_per_capita_cases_map.svg.png)

# # logarithmic scale

# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/CoViD-19_US.svg/709px-CoViD-19_US.svg.png)
# 
#                    Number of cases (blue) and number of deaths (red)

# # Tree plot USA

# In[ ]:


pd.options.mode.chained_assignment = None

# Read the data
us_data_path = "/kaggle/input/covid19-in-usa/"
us_df = pd.read_csv(us_data_path + "us_covid19_daily.csv")
us_states_df = pd.read_csv(us_data_path + "us_states_covid19_daily.csv")
us_df["date"] = pd.to_datetime(us_df["date"], format="%Y%m%d")
us_states_df = us_states_df.reindex(index=us_states_df.index[::-1])
us_states_df["date"] = pd.to_datetime(us_states_df["date"], format="%Y%m%d").dt.date.astype(str)
#us_states_df.head()

# US state code to name mapping
state_map_dict = {'AL': 'Alabama',
 'AK': 'Alaska',
 'AS': 'American Samoa',
 'AZ': 'Arizona',
 'AR': 'Arkansas',
 'CA': 'California',
 'CO': 'Colorado',
 'CT': 'Connecticut',
 'DE': 'Delaware',
 'DC': 'District of Columbia',
 'D.C.': 'District of Columbia',
 'FM': 'Federated States of Micronesia',
 'FL': 'Florida',
 'GA': 'Georgia',
 'GU': 'Guam',
 'HI': 'Hawaii',
 'ID': 'Idaho',
 'IL': 'Illinois',
 'IN': 'Indiana',
 'IA': 'Iowa',
 'KS': 'Kansas',
 'KY': 'Kentucky',
 'LA': 'Louisiana',
 'ME': 'Maine',
 'MH': 'Marshall Islands',
 'MD': 'Maryland',
 'MA': 'Massachusetts',
 'MI': 'Michigan',
 'MN': 'Minnesota',
 'MS': 'Mississippi',
 'MO': 'Missouri',
 'MT': 'Montana',
 'NE': 'Nebraska',
 'NV': 'Nevada',
 'NH': 'New Hampshire',
 'NJ': 'New Jersey',
 'NM': 'New Mexico',
 'NY': 'New York',
 'NC': 'North Carolina',
 'ND': 'North Dakota',
 'MP': 'Northern Mariana Islands',
 'OH': 'Ohio',
 'OK': 'Oklahoma',
 'OR': 'Oregon',
 'PW': 'Palau',
 'PA': 'Pennsylvania',
 'PR': 'Puerto Rico',
 'RI': 'Rhode Island',
 'SC': 'South Carolina',
 'SD': 'South Dakota',
 'TN': 'Tennessee',
 'TX': 'Texas',
 'UT': 'Utah',
 'VT': 'Vermont',
 'VI': 'Virgin Islands',
 'VA': 'Virginia',
 'WA': 'Washington',
 'WV': 'West Virginia',
 'WI': 'Wisconsin',
 'WY': 'Wyoming'}

state_code_dict = {v:k for k, v in state_map_dict.items()}
state_code_dict["Chicago"] = 'Illinois'

def correct_state_names(x):
    try:
        return state_map_dict[x.split(",")[-1].strip()]
    except:
        return x.strip()
    
def get_state_codes(x):
    try:
        return state_code_dict[x]
    except:
        return "Others"

covid_19_df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
us_covid_df = covid_19_df[covid_19_df["Country/Region"]=="US"]
us_covid_df["Province/State"] = us_covid_df["Province/State"].apply(correct_state_names)
us_covid_df["StateCode"] = us_covid_df["Province/State"].apply(lambda x: get_state_codes(x))


# In[ ]:


statewise_cases = pd.DataFrame(us_covid_df.groupby(['Province/State'])['Confirmed', 'Deaths', 'Recovered'].max().reset_index())
statewise_cases["Country"] = "US" # in order to have a single root node
fig = px.treemap(statewise_cases, path=['Country','Province/State'], values='Confirmed',
                  color='Confirmed', hover_data=['Province/State'],
                  color_continuous_scale='matter')
fig.show()


# In[ ]:


global_df = pd.read_csv('/kaggle/input/global-hospital-beds-capacity-for-covid19/hospital_beds_global_regional_v1.csv')
usa_df = pd.read_csv('/kaggle/input/global-hospital-beds-capacity-for-covid19/hospital_beds_USA_v1.csv')
us_dff = pd.read_csv('/kaggle/input/us-counties-covid-19-dataset/us-counties.csv')


# In[ ]:


import folium
center = [37.0902405, -95.7128906]
m = folium.Map(location = center, zoom_start = 3)

for lat, lng in zip(usa_df.lat, usa_df.lng):
    folium.CircleMarker(
        [lat, lng],
        radius = 5,
        fill = True,
        color = 'Black',
        fill_color = 'red',
        fill_opacity = 0.6
        ).add_to(m)
    
m


# # Heat Map

# In[ ]:


from folium.plugins import HeatMap
m = folium.Map(location = center, zoom_start = 3)

heat_data = [[row['lat'],row['lng']] for index, row in usa_df.iterrows()]
HeatMap(heat_data,radius=16.5, blur = 6.5).add_to(m)

m


# In[ ]:


#statewise_cases
state_details = pd.pivot_table(us_covid_df, values=['Confirmed','Deaths','Recovered'], index='Province/State', aggfunc='max')
state_details['Recovery Rate'] = round(state_details['Recovered'] / state_details['Confirmed'],2)
state_details['Death Rate'] = round(state_details['Deaths'] /state_details['Confirmed'], 2)
state_details = state_details.sort_values(by='Confirmed', ascending= False)
state_details.style.background_gradient(cmap='YlOrRd')


# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
from plotly import tools
py.init_notebook_mode(connected=False)
import plotly.graph_objs as go
import plotly.express as px



cumulative_df = us_covid_df.groupby("ObservationDate")["Confirmed", "Deaths", "Recovered"].sum().reset_index()

### Plot for number of cumulative covid cases over time
fig = px.bar(cumulative_df, x="ObservationDate", y="Confirmed")
layout = go.Layout(
    title=go.layout.Title(
        text="Daily cumulative count of confirmed COVID-19 cases in US",
        x=0.5
    ),
    font=dict(size=14),
    width=800,
    height=500,
    xaxis_title = "Date of observation",
    yaxis_title = "Number of confirmed cases"
)

fig.update_layout(layout)
fig.show()

### Plot for number of cumulative covid cases over time
fig = px.bar(cumulative_df, x="ObservationDate", y="Deaths")
layout = go.Layout(
    title=go.layout.Title(
        text="Daily cumulative count of deaths due to COVID-19 in US",
        x=0.5
    ),
    font=dict(size=14),
    width=800,
    height=500,
    xaxis_title = "Date of observation",
    yaxis_title = "Number of death cases"
)

fig.update_layout(layout)
fig.show()

### Plot for number of cumulative covid cases over time
cumulative_df["ConfirmedNew"] = cumulative_df["Confirmed"].diff() 
fig = px.bar(cumulative_df, x="ObservationDate", y="ConfirmedNew")
layout = go.Layout(
    title=go.layout.Title(
        text="Daily count of new confirmed COVID-19 cases in US",
        x=0.5
    ),
    font=dict(size=14),
    width=800,
    height=500,
    xaxis_title = "Date of observation",
    yaxis_title = "Number of confirmed cases"
)

fig.update_layout(layout)
fig.show()


# # Map View

# In[ ]:


import datetime

cumulative_df = us_covid_df.groupby(["StateCode", "ObservationDate"])["Confirmed", "Deaths", "Recovered"].sum().reset_index()
cumulative_df["ObservationDate"] = pd.to_datetime(cumulative_df["ObservationDate"] , format="%m/%d/%Y").dt.date
cumulative_df = cumulative_df.sort_values(by="ObservationDate").reset_index(drop=True)
start_date = datetime.date(2020, 2, 25)
cumulative_df = cumulative_df[cumulative_df["ObservationDate"]>=start_date]
cumulative_df["ObservationDate"] = cumulative_df["ObservationDate"].astype(str)

fig = px.choropleth(locations=cumulative_df["StateCode"],
                    color=cumulative_df["Confirmed"], 
                    locationmode="USA-states",
                    scope="usa",
                    animation_frame=cumulative_df["ObservationDate"],
                    color_continuous_scale='Reds',
                    range_color=[0,550000]
                    #autocolorscale=False,
                   )

layout = go.Layout(
    title=go.layout.Title(
        text="Cumulative count of COVID-19 cases in US states",
        x=0.5
    ),
    font=dict(size=14),
)

fig.update_layout(layout)
fig.show()


# In[ ]:


cumulative_df = us_covid_df.groupby(["StateCode", "ObservationDate"])["Confirmed", "Deaths", "Recovered"].sum().reset_index()
cumulative_df["ObservationDate"] = pd.to_datetime(cumulative_df["ObservationDate"] , format="%m/%d/%Y").dt.date
cumulative_df = cumulative_df.sort_values(by="ObservationDate").reset_index(drop=True)
start_date = datetime.date(2020, 3, 10)
cumulative_df = cumulative_df[cumulative_df["ObservationDate"]>=start_date]
cumulative_df["ObservationDate"] = cumulative_df["ObservationDate"].astype(str)

fig = px.scatter(cumulative_df, x="Confirmed", y="Deaths", animation_frame="ObservationDate", animation_group="StateCode",
           size="Confirmed", color="StateCode", hover_name="StateCode",
           log_x=False, size_max=55, range_x=[0,550000], range_y=[-20,100000])

layout = go.Layout(
    title=go.layout.Title(
        text="Changes in number of confirmed & death cases over time in US states",
        x=0.5
    ),
    font=dict(size=14),
    xaxis_title = "Total number of confirmed cases",
    yaxis_title = "Total number of death cases"
)

fig.update_layout(layout)

fig.show()


# # Testings

# In[ ]:


fig = px.bar(us_df, x="date", y="total")

layout = go.Layout(
    title=go.layout.Title(
        text="Cumulative number of Total COVID-19 testing over time in US",
        x=0.5
    ),
    font=dict(size=14),
    width=800,
    height=500,
    xaxis_title = "Date of observation",
    yaxis_title = "Number of covid-19 testings"
)

fig.update_layout(layout)
fig.show()


# In[ ]:


import plotly.express as px

us_states_df["positive"] = us_states_df["positive"].fillna(0)

fig = px.scatter(us_states_df, x="total", y="positive", animation_frame="date", animation_group="state",
           size="positive", color="state", hover_name="state",
           log_x=False, size_max=55, range_x=[0,200000], range_y=[0,55000])

layout = go.Layout(
    title=go.layout.Title(
        text="Total testing counts Vs Positive Counts over time",
        x=0.5
    ),
    font=dict(size=14),
#     width=800,
#     height=500,
    xaxis_title = "Total number of tests",
    yaxis_title = "Number of positive tests"
)

fig.update_layout(layout)
fig.show()


# # USA COVID19 Forecasting

# # Sigmoid model

# I thought of a sigmoidal function because China's data resembled a sigmoidal shape. Therefore, I try to fit sigmoid functions onto USA's.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# Any results you write to the current directory are saved as output.
train=pd.read_csv('/kaggle/input/coronavirus-2019ncov/covid-19-all.csv')


# In[ ]:


train.head(5)


# In[ ]:


country_df = train[train['Country/Region']=='US'].groupby('Date')['Confirmed','Deaths'].sum()
country_df['day_count'] = list(range(1,len(country_df)+1))
ydata = country_df.Confirmed
xdata = country_df.day_count
country_df['rate'] = (country_df.Confirmed-country_df.Confirmed.shift(1))/country_df.Confirmed
country_df['increase'] = (country_df.Confirmed-country_df.Confirmed.shift(1))

plt.plot(xdata, ydata, 'o')
plt.title("USA")
plt.ylabel("Population infected")
plt.xlabel("Days")
plt.show()


# 
# ****Sigmoid function,
# 
# Here is a snap of how I learnt to fit Sigmoid Function - y = c/(1+np.exp(-a*(x-b))) and 3 coefficients [c, a, b]:
# 
# * c - the maximum value (eventual maximum infected people, the sigmoid scales to this value eventually)
# * a - the sigmoidal shape (how the infection progress. The smaller, the softer the sigmoidal shape is)
# * b - the point where sigmoid start to flatten from steepening (the midpoint of sigmoid, when the rate of increase start to slow down)
# 
# 

# In[ ]:


us_df = train[train['Country/Region']=='US'].groupby('Date')['Confirmed','Deaths','Recovered'].sum()
us_df = us_df[us_df.Confirmed>=100]


# In[ ]:


from scipy.optimize import curve_fit
import pylab
from datetime import timedelta

us_df['day_count'] = list(range(1,len(us_df)+1))
us_df['increase'] = (us_df.Confirmed-us_df.Confirmed.shift(1))
us_df['rate'] = (us_df.Confirmed-us_df.Confirmed.shift(1))/us_df.Confirmed
us_df['Active']=us_df['Confirmed']-us_df['Deaths']-us_df['Recovered']


def sigmoid(x,c,a,b):
     y = c*1 / (1 + np.exp(-a*(x-b)))
     return y

xdata = np.array(list(us_df.day_count)[::2])
ydata = np.array(list(us_df.Active)[::2])

population=1.332*10**9
#popt, pcov = curve_fit(sigmoid, xdata, ydata, method='dogbox',bounds=([0.,0., 0.],[population,6, 100.]))
#print(popt)


# In[ ]:


est_a = 4200000
est_b = 0.03
est_c = 123
x = np.linspace(-1, us_df.day_count.max()+50, 50)
y = sigmoid(x,est_a,est_b,est_c)
pylab.plot(xdata, ydata, 'o', label='data')
pylab.plot(x,y, label='fit',alpha = 0.6)
pylab.ylim(-0.05, est_a*1.05)
pylab.xlim(-0.05, est_c*2.05)
pylab.legend(loc='best')
plt.xlabel('days from day 1')
plt.ylabel('confirmed cases')
plt.title('USA')
pylab.show()


print('model start date:',us_df[us_df.day_count==1].index[0])
print('model start infection:',int(us_df[us_df.day_count==1].Confirmed[0]))
print('model fitted max Active at:',int(est_a))
print('model sigmoidal coefficient is:',round(est_b,3))
print('model curve stop steepening, start flattening by day:',int(est_c))
print('model curve flattens by day:',int(est_c)*2)
display(us_df.head(3))
display(us_df.tail(3))


# 
# # From this, its seen that in case of USA if the graph goes like that:
# 
# *     **max Active case: approx 4200000 ☠️**
# *     **curve stop steepening, start flattening by day: 44 ,which is: 15/04/2020**
# *     **curve flattens by day: 88 which is: 29/05/2020**
# 
# 

# Some Images are been taken from [Here](http://)

# ![](https://media1.tenor.com/images/2a870cf48abd043938ca70ed0a233b2f/tenor.gif?itemid=16698747)

# # Hope you guys find this notebook useful. 
# # If you like this notebook, please upvote.***👍    
# # Thanks in advance**
