#!/usr/bin/env python
# coding: utf-8

# # COVID19 Real time Updates INDIA:
# @author: Neeraj Tiwari

# In[ ]:


import IPython
IPython.display.HTML('<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1977187" data-url="https://flo.uri.sh/visualisation/1977187/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>')


# In[ ]:


import pandas as pd
import plotly.express as px


# ## Symptoms of Coronavirus

# In[ ]:


symptoms={'symptom':['Fever',
        'Dry cough',
        'Fatigue',
        'Sputum production',
        'Shortness of breath',
        'Muscle pain',
        'Sore throat',
        'Headache',
        'Chills',
        'Nausea or vomiting',
        'Nasal congestion',
        'Diarrhoea',
        'Haemoptysis',
        'Conjunctival congestion'],'percentage':[87.9,67.7,38.1,33.4,18.6,14.8,13.9,13.6,11.4,5.0,4.8,3.7,0.9,0.8]}

symptoms=pd.DataFrame(data=symptoms,index=range(14))
#symptoms


# In[ ]:


fig = px.pie(symptoms,
             values="percentage",
             names="symptom",
             title="Symtoms of Coronavirus",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="value+percent+label")
fig.show()


# ## COVID-19 Statewise Update in INDIA

# In[ ]:


link = "https://api.covid19india.org/csv/latest/state_wise.csv"
df = pd.read_csv(link)
df = df.drop(df.index[0])
#df


# In[ ]:


Date = df['Last_Updated_Time'].values.tolist()
State = df['State'].values.tolist()
Confirmed = df['Confirmed'].values.tolist()
Recovered = df['Recovered'].values.tolist()
Active = df['Active'].values.tolist()
Deaths = df['Deaths'].values.tolist()


# In[ ]:


import plotly.graph_objects as go

fig = go.Figure(data=[
    go.Bar(name='Active', x=State, y=Active),
    go.Bar(name='Recovered', x=State, y=Recovered),
    go.Bar(name='Deaths', x=State, y=Deaths)
])
fig.update_layout(
    autosize=False,
    width=950,
    height=700,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
    paper_bgcolor="LightSteelBlue",title='Statewise Covid19 Case on '+ str(Date[0][0:10]) + " Last Update at"+ str(Date[0][10:])
)
# Change the bar mode
fig.update_layout(barmode='stack')
#fig.update_layout(barmode='group')

fig.show()


# ## Statewise Confirmed Case

# In[ ]:


fig = px.pie(df, values=Confirmed, names=State, title='Statewise Confirmed Case')
fig.show()


# ## Statewise Active Case

# In[ ]:


import plotly.express as px
fig = px.pie(df, values=Active, names=State, title='Statewise Active Case')
fig.show()


# ## Statewise Death Case

# In[ ]:


import plotly.express as px
fig = px.pie(df, values=Deaths, names=State, title='Statewise Deaths Case')
fig.show()


# ## Statewise Recovered Case

# In[ ]:


import plotly.express as px
fig = px.pie(df, values=Recovered, names=State, title='Statewise Recovered Case')
#fig.update_traces(rotation=60, pull=0.01)
fig.show()


# ## Daily Update

# In[ ]:


import requests
import re 

link2 = 'https://api.covid19india.org/data.json'
r = requests.get(link2)
india_Data = r.json()
#india_Data.keys()
#india_Data['cases_time_series'][-1]['date']


# In[ ]:


india_Confirmed = []
india_Recovered = []
india_Deseased = []
timeStamp = []
for index in range(len(india_Data['cases_time_series'])):
    india_Confirmed.append(int(re.sub(',','',india_Data['cases_time_series'][index]['totalconfirmed'])))
    india_Recovered.append(int(re.sub(',','',india_Data['cases_time_series'][index]['totalrecovered'])))
    india_Deseased.append(int(re.sub(',','',india_Data['cases_time_series'][index]['totaldeceased'])))
    
    timeStamp.append(india_Data['cases_time_series'][index]['date'])
    

fig = go.Figure()
#fig = px.scatter(testingHistory,x="time_stamp", y="testing_no")

fig = fig.add_trace(go.Scatter(x=timeStamp, y=india_Confirmed,
                    mode='lines+markers',
                    name='Confirmed Cases'))
fig = fig.add_trace(go.Scatter(x=timeStamp, y=india_Recovered,
                    mode='lines+markers',
                    name='Recoverd Patients'))
fig = fig.add_trace(go.Scatter(x=timeStamp, y=india_Deseased,
                    mode='lines+markers',
                    name='Deseased Patients'))

fig = fig.update_layout(
    title="India COVID-19 cases on  " + str(india_Data['cases_time_series'][-1]['date']) + "2020",
    xaxis_title="Date",
    yaxis_title="Cases",
    
)


fig.show()


# ## Telangana Districtwise COVID-19 Updates

# In[ ]:


link3 = "https://api.covid19india.org/v2/state_district_wise.json"
r = requests.get(link3)
states_Data = r.json()
for i in range(len(states_Data[:])):
    print(states_Data[i]['state'], '>>>', i)
#telangana = 27


# In[ ]:


telangana = 27
district = []
district_Confirmed = []
district_Recovered = []
district_Deseased = []
district_Active = []
for index in range(len(states_Data[telangana]['districtData'])):
    district.append(str(re.sub(',','',states_Data[telangana]['districtData'][index]['district'])))
    district_Confirmed.append(int(states_Data[telangana]['districtData'][index]['confirmed']))
    district_Recovered.append(int(states_Data[telangana]['districtData'][index]['recovered']))
    district_Deseased.append(int(states_Data[telangana]['districtData'][index]['deceased']))
    district_Active.append(int(states_Data[telangana]['districtData'][index]['active']))


# In[ ]:


import plotly.graph_objects as go

fig = go.Figure(data=[
    go.Bar(name='Active', x=district, y=district_Active),
    go.Bar(name='Recovered', x=district, y=district_Recovered),
    go.Bar(name='Deaths', x=district, y=district_Deseased)
])
fig.update_layout(
    autosize=False,
    width=950,
    height=700,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
    paper_bgcolor="LightSteelBlue",title='Statewise Covid19 Case on '+ str(Date[0][0:10])
)
# Change the bar mode
fig.update_layout(barmode='stack')
#fig.update_layout(barmode='group')

fig.show()


# ## Gujrat Districtwise COVID-19 Updates

# In[ ]:


link3 = "https://api.covid19india.org/v2/state_district_wise.json"
r = requests.get(link3)
states_Data = r.json()
#states_Data[9]
gujrat = 13


# In[ ]:


district = []
district_Confirmed = []
district_Recovered = []
district_Deseased = []
district_Active = []
for index in range(len(states_Data[gujrat]['districtData'])):
    district.append(str(re.sub(',','',states_Data[gujrat]['districtData'][index]['district'])))
    district_Confirmed.append(int(states_Data[gujrat]['districtData'][index]['confirmed']))
    district_Recovered.append(int(states_Data[gujrat]['districtData'][index]['recovered']))
    district_Deseased.append(int(states_Data[gujrat]['districtData'][index]['deceased']))
    district_Active.append(int(states_Data[gujrat]['districtData'][index]['active']))


# In[ ]:


import plotly.graph_objects as go

fig = go.Figure(data=[
    go.Bar(name='Active', x=district, y=district_Active),
    go.Bar(name='Recovered', x=district, y=district_Recovered),
    go.Bar(name='Deaths', x=district, y=district_Deseased)
])
fig.update_layout(
    autosize=False,
    width=950,
    height=700,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
    paper_bgcolor="LightSteelBlue",title='Gujrat Covid19 statewise Case on '+ str(Date[1][0:10])
)
# Change the bar mode
fig.update_layout(barmode='stack')
#fig.update_layout(barmode='group')

fig.show()


# ## Derivatives

# In[ ]:


# 1st Derivative
import numpy as np
table2 = (np.gradient(india_Confirmed))


# In[ ]:


fig = go.Figure()
fig = fig.add_trace(go.Scatter(x=timeStamp, y=table2,
                    mode='lines+markers',
                    name='Curve rate'))

fig = fig.update_layout(
    title="India COVID-19 confirm Rate on  " + str(india_Data['cases_time_series'][-1]['date']) + "2020",
    xaxis_title="Date",
    yaxis_title="Rate",
    
)


fig.show()


# In[ ]:


# 2nd Derivative
import numpy as np
table3 = np.gradient(np.gradient(india_Confirmed))


# In[ ]:


fig = go.Figure()
fig = fig.add_trace(go.Scatter(x=timeStamp, y=table3,
                    mode='lines+markers',
                    name='Curve rate'))

fig = fig.update_layout(
    title="India COVID-19 confirm Rate on  " + str(india_Data['cases_time_series'][-1]['date']) + "2020",
    xaxis_title="Date",
    yaxis_title="Rate",
    
)


fig.show()


# In[ ]:




