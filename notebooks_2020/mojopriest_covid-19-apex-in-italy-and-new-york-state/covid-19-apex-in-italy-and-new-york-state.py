#!/usr/bin/env python
# coding: utf-8

# **Disclaimer**
# 
# *This notebook is first and foremost a part of my personal learning project during which I have tried to find relevant factors concerning the apex phase during the first wave of the COVID-19 pandemic. There is no model building included in this notebook, and all analysis conclusions are subjective.*
# 
# *This notebook was first made public on April 7th, 2020. Special thanks to **Sudalai Rajikumar (Kaggle user: SRK)** for his effort in making the Italy COVID-19 dataset available on Kaggle.*
# 
# ***The analysis part of this notebook has now been concluded. The plot graphics etc. will however update themselves barring any modifications to the original datasets. With data from New York state, the plot graphics timeframe has since been readjusted to show more of the 'second wave' situation.***
# 
# ***November 5th 2020,***
# 
# ***Jari Peltola***
# 
# 

# ## COVID-19: the apex phase in Italy and New York state 

# **TABLE OF CONTENTS**
# 1. Introduction <br>
# 1.1 *Summary of conclusions* <br>
# 2. Italy <br>
# 2.1 *Lombardy* <br>
# 2.2 *Emilia-Romagna*<br>
# 2.3 *Piemonte* <br>
# 2.4 *Veneto* <br>
# 3. New York State <br>

# ### 1. Introduction

# Studying the apex phase of the current COVID-19 pandemic is very much a work in progress, since no previous empirical data on subject (global pandemic in modern era) exists. In this notebook, the following things are taken as grounding hypotheses:
# 
# - **No universal COVID-19 test or testing method exists**, which means the number of new COVID-19 cases in different regions may vary greatly depending on test, method and regional testing policy. 
# - **No generally accepted single model or method on predicting COVID-19 spread exists**, because they all are currently being tested and validated in real-time.
# - **The COVID-19 confirmed case data is not collected the same way in different places**, which leaves plenty of room for subjective interpretation.
# - **The COVID-19 death rate data includes mostly diagnosed hospitalized patients**, often leaving other possible covid-caused death incidents outside the data.
# - **Recent COVID-19 studies in the U.S. suggest that COVID-19 pandemic started there earlier than suspected**, which may be true also in other areas such as Northern Italy.
# - **COVID-19 infection rate is - according to recent research - higher than suspected**, meaning the actual number of people infected may be 7-10 times higher than in datasets currently in use.
# - **As most viruses are, COVID-19 is suspected to be more infectious in winter compared to summer.** This is notable when for example outbreaks in South America (most of the continent is located below equator) are studied.
# 
# In this notebook, only two aspects related to COVID-19 are analyzed: the reported number of hospitalizations and the reported number of recovered patients. The term 'recovered' is interpreted in this notebook the same as 'discharged from hospital'. In reality, COVID-19 patients may have a recovery period lasting several weeks or even months after being discharged from hospital.
# 
# With a simple comparison, in this notebook a closer look is taken at **how and when the number of recovered patients correlate with the total number of hospitalized patients in different regions.**
# 
# The regions included in this notebook are Lombardy, Emilia-Romagna , Piemonte and Veneto in Italy as well as the state of New York in United States.

# #### 1.1 Summary of conclusions

# Below is a summary of primary conclusions in this notebook:
# 
# - **In regions included in this notebook, the 'first wave' of COVID-19 as a whole lasted about three months.**
# - **The incline of the curve was about one month long, and it took about two months for hospitalizations to decline on the same level as before the outbreak.**  
# - **The apex plateau phase lasted for 9-18 days.**
# - **The decreasing demand for ICU beds was the first sign of downcurrent.**
# - **The decreasing demand for hospital care does not mean that new positive confirmed COVID-19 decrease at the same rate.**
# - **After three months, the demand for hospitalization had settled on about 5-15 percent compared to peak demand during the apex.**

# ### 2. Italy

# In[ ]:


# import modules
import math
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# In[ ]:


#load dataframe
df_it = pd.read_csv('../input/covid19-in-italy/covid19_italy_region.csv')
df_it['Date'] =  pd.to_datetime(df_it['Date'], infer_datetime_format=True)
df_it['NewPositiveCases'] = df_it['NewPositiveCases'].abs()


# In many ways Italy is a combination of two stories: the story of north and the story of south. This applies also to COVID-19, because in northern Italy the outbreak has been much more severe than in the south. Conversely, beginning from May 4th many regions in Northern Italy began to ease their lockdown measures, which is an interesting factor considering the future development of COVID-19 outbreak in Italy.
# 
# In northern Italy, the 'flattening the curve' i.e. apex phase of the COVID-19 pandemic was reached at the beginning of April. The slowly stabilized number of hospitalized patients in different regions support this hypothesis.

# In[ ]:


# new dataframe
df_regioni = df_it.loc[df_it['RegionName'].isin(['Lombardia','Emilia-Romagna','Piemonte','Veneto', 'Toscana'])]

# plot figure
fig = px.line(df_regioni, x="Date", y="TotalHospitalizedPatients", color='RegionName')

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

fig.update_layout(
    yaxis=dict(
        showline=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

annotations = []

# source
annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.10,
                              xanchor='center', yanchor='top',
                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',
                              font=dict(family='arial narrow',
                                        size=8,
                                        color='rgb(96,96,96)'),
                              showarrow=False))

fig.update_layout(annotations=annotations)
fig.update_layout(title_text='<b>COVID-19</b>:<br>hospitalized patients in Italy by region',              
    font=dict(family='calibri',
        size=12,
        color='rgb(64,64,64)'),
     legend=dict(
        x=0.75,
        y=0.97,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
)

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_yaxes(title_text='Patients')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_font=dict(size=14))
fig.update_xaxes(title_font=dict(size=14))
fig.update_layout(xaxis_showgrid=False)
   
fig.show()


# The one thing no one really knows is the level of post-apex hospitalizations will remain on. Because new COVID-19 cases are to emerge also in the future, this means that patients still need to be treated in hospitals also after the 'first wave' of COVID-19.
# 
# Hospitalization downcurves in Northern Italy may even give a false sense of security, This is because **during the most severe apex phase in March, hospitalization capacity in Northern Italy was overwhelmed by the number of patients and special temporary units such as army hospitals were set up.** This means that ***the actual 100-percent level of available hospital capacity in Northern Italy regions is lower than the hospitalization curve apex in the plot graphics.***
# 
# During the apex of the first wave, the need for intensive care units (ICU) decreased in Northern Italy faster than regular hospital bed need. Of course, this is largely because of the unfortunate fact that a majority of patients put on ventilators will not survive their illness.

# In[ ]:


# new dataframe
df_regions = df_it.loc[df_it['RegionName'].isin(['Lombardia','Emilia-Romagna','Piemonte','Veneto', 'Toscana'])]

# plot figure
fig = px.line(df_regions, x="Date", y="IntensiveCarePatients", color='RegionName')

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

fig.update_layout(
    yaxis=dict(
        showline=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

annotations = []

# source
annotations.append(dict(xref='paper', yref='paper', x=0.85, y=-0.12,
                              xanchor='center', yanchor='top',
                              text='source: Sito del Dipartimento della Protezione Civile:<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',
                              font=dict(family='arial narrow',
                                        size=8,
                                        color='rgb(96,96,96)'),
                              showarrow=False))
fig.update_layout(annotations=annotations)


fig.update_layout(title_text='<b>COVID-19</b>:<br>intensive care patients in Italy by region',
                  
      font=dict(family='calibri',
                                size=12,
                                color='rgb(64,64,64)'),
     legend=dict(
        x=0.75,
        y=0.97,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
)


fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_yaxes(title_text='Patients')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_font=dict(size=14))
fig.update_xaxes(title_font=dict(size=14))
fig.update_layout(xaxis_showgrid=False)
   
   
fig.show()


# In[ ]:


# new dataframe
df_regions = df_it.loc[df_it['RegionName'].isin(['Lombardia','Emilia-Romagna','Piemonte','Veneto', 'Toscana'])]

# plot figure
fig = px.line(df_regions, x="Date", y="HospitalizedPatients", color='RegionName')

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

fig.update_layout(
    yaxis=dict(
        showline=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

annotations = []

# source
annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.10,
                              xanchor='center', yanchor='top',
                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',
                              font=dict(family='arial narrow',
                                        size=8,
                                        color='rgb(96,96,96)'),
                              showarrow=False))

fig.update_layout(annotations=annotations)



fig.update_layout(title_text='<b>COVID-19</b>:<br>regular ward care patients in Italy by region',
                  
      font=dict(family='calibri',
                                size=12,
                                color='rgb(64,64,64)'),
     legend=dict(
        x=0.75,
        y=0.97,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
)


fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_yaxes(title_text='Patients')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_font=dict(size=14))
fig.update_xaxes(title_font=dict(size=14))
fig.update_layout(xaxis_showgrid=False)
   
fig.show()


# As mentioned before, testing COVID-19 is such a fragmented subject - with numerous different methods and practices - that new cases will not be analyzed further in this notebook. What is notable however is the number of positive confirmed COVID-19 cases after the hospitalization peak (around April 1st).
# 
# As one can see below, there has been a steady flow of new positive cases in all four Italian regions although the hospitalization rate has at the same time decreased. In Lombardy the number of new cases was around 45K on April 1st, when the hospitalization apex was reached. Yet a month later, the cumulative number of new cases in the region was over 76K despite of the downcurrent in hospitalizations.
# 
# This means that **in April the number of new monthly COVID-19 cases in Lombardy decreased only 20 percent compared to March (about 10K cases, from 45K new cases found in March to over 35K new cases found in April). However, the number of total hospitalizations in Lombardy decreased from 13K to 7K - almost 50 percent - during the same time.** 
# 
# Four regions in Italy is of course a small sample and changes in testing rate have their own effect on the numbers, but based on that data it can be argued that ***hospitalizations and new COVID-19 cases don't actually correlate with each other as absolute values***. More likely **it is all about who gets infected and when**. Many of those infected by the first (and most destructive) COVID-19 wave in Italy were the most vulnerable people, which lead to sharp increase in hospitalizations, overload of unprepared regional health case systems and eventually also to more victims.

# In[ ]:


# plot figure
fig = px.line(df_regions, x="Date", y="TotalPositiveCases", color='RegionName')

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

fig.update_layout(
    yaxis=dict(
        showline=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

fig.update_layout(title_text='<b>COVID-19</b>:<br>number of confirmed positive cases in Italy',
                                font=dict(family='calibri',
                                size=12,
                                color='rgb(64,64,64)'))

annotations = []

# Source
annotations.append(dict(xref='paper', yref='paper', x=0.88, y=-0.12,
                              xanchor='center', yanchor='top',
                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',
                              font=dict(family='arial narrow',
                                        size=8,
                                        color='rgb(96,96,96)'),
                              showarrow=False))

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_yaxes(title_text='Confirmed cases')
fig.update_layout(legend_title='<b> Region </b>')
fig.update_layout(annotations=annotations)
fig.update_layout(xaxis_showgrid=False)

fig.show()


# Likely because of testing system or the weekend (with more people often outside), the number of new positive cases in different regions seems to peak about once a week. Therefore looking at new cases with for example 5-day average would be one way of avoiding these 'waves' or 'steps' in new cases. Since the downcurve of a modern era pandemic is by and large an unknown research subject as for now, it may well be possible that - instead of a steady downcurve - new COVID-19 positive cases will gradually decrease in wave-like pattern and eventually fade away like high frequency harmonics in bass sound waveform.
# 
# It's good to keep in mind though that in every region, only a small fraction of all people has been tested for COVID-19. In Italy testing has concentrated - largely by necessity - on patients showing clear COVID-19 symptoms. The number of unsymptomatic infected people has so far been a rough estimate, but drawing on clinical studies carried out in New York state, in urban areas the total number of COVID-19 positive people may be some 7-10 times higher than actual testing suggests.   

# In[ ]:


# mask dataframe
start_date = '2020-04-01 17:00:00'
mask = (df_it['Date'] > start_date)
df_cases = df_it.loc[mask]

df_masked_lo  = df_cases.loc[df_cases['RegionName'] == 'Lombardia']
df_masked_er  = df_cases.loc[df_cases['RegionName'] == 'Emilia-Romagna']
df_masked_pi  = df_cases.loc[df_cases['RegionName'] == 'Piemonte']
df_masked_ve  = df_cases.loc[df_cases['RegionName'] == 'Veneto']


#plot figure
fig = go.Figure()

fig.add_trace(go.Bar(
    x=df_masked_lo['Date'],
    y=df_masked_lo['NewPositiveCases'],
    name='Lombardy',
    marker_color='red'
))
fig.add_trace(go.Bar(
    x=df_masked_er['Date'],
    y=df_masked_er['NewPositiveCases'],
    name='Emilia-Romagna',
    marker_color='orange'
))

fig.add_trace(go.Bar(
    x=df_masked_pi['Date'],
    y=df_masked_pi['NewPositiveCases'],
    name='Piemonte',
    marker_color='mediumblue'
))
fig.add_trace(go.Bar(
    x=df_masked_ve['Date'],
    y=df_masked_ve['NewPositiveCases'],
    name='Veneto',
    marker_color='green'
))


fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

fig.update_layout(
    yaxis=dict(
        showline=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

annotations = []

# source
annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.14,
                              xanchor='center', yanchor='top',
                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',
                              font=dict(family='arial narrow',
                                        size=8,
                                        color='rgb(96,96,96)'),
                              showarrow=False))

fig.update_layout(annotations=annotations)

fig.update_layout(barmode='group', xaxis_tickangle=-25)

fig.update_layout(title_text='<b>COVID-19</b>:<br>new daily positive cases in Italy after April 1st',
                  
     font=dict(family='calibri',
        size=12,
        color='rgb(64,64,64)'),
     legend=dict(
        x=0.7,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.2,
    bargroupgap=0.2
)


fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_yaxes(title_text='Positive cases')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_font=dict(size=14))
fig.update_xaxes(title_font=dict(size=14))
fig.update_layout(xaxis_showgrid=False)
   
fig.show()


# #### 2.1 Lombardy

# Next in this notebook, we will be taking a closer look at specific regions in Italy. In Lombardy, the maximum need for hospital beds during the apex was about 13000 beds.
# 
# After reaching the peak, that number stabilized on the same level for a lengthy period (18 days). After that there was a sharp fall, just as in for example Veneto region, as we can see later. Whereas in Veneto the drop basically happened in a day, in Lombardy the same took place over three days, most likely because of larger total number of population in Lombardy. 

# In[ ]:


# new dataframe
df_lo = df_it[df_it.RegionName == 'Lombardia']

# plot figure
fig = px.bar(df_lo, y='TotalHospitalizedPatients', x='Date', text='TotalHospitalizedPatients', color ='TotalHospitalizedPatients')
fig.data[0].marker.line.width = 0.5
fig.data[0].marker.line.color = "black"

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_mode='hide')

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

fig.update_layout(
    yaxis=dict(
        showline=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

annotations = []

# source
annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.12,
                              xanchor='center', yanchor='top',
                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',
                              font=dict(family='arial narrow',
                                        size=8,
                                        color='rgb(96,96,96)'),
                              showarrow=False))

fig.update_layout(annotations=annotations)
fig.update_layout(title_text='<b>COVID-19</b>:<br>total number of hospitalized patients in Lombardy region',
                  
 font=dict(family='calibri',
            size=12,
            color='rgb(64,64,64)'))

fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_yaxes(title_text='Hospitalized patients')
fig.update_yaxes(title_font=dict(size=14))
fig.update_xaxes(title_font=dict(size=14))
fig.update_layout(xaxis_showgrid=False)

fig.show()


# As noted before, it took about month for hospitalizations in Lombardy to decrease by 50 percent from peak apex level. After that the decrease has been relatively slow, most likely because of the large number of overall population in Lombardy region.
# 
# The need for ICU care in Lombardy started to decrease more rapidly compared to regular hospital beds. For example, on June 16th there were 69 ICU patients and 1971 regular ward patients in Lombardy. This means that only 3,5 percent of all hospitalized patients in Lombardy required ICU care on that day. For comparison, on the day when ICU care need peaked (April 3rd), the same percentage was 10,5.

# In[ ]:


# new dataframe
data_lombardia = df_it[df_it.RegionName == 'Lombardia']

# plot figure
fig = px.bar(data_lombardia, x='Date', y='IntensiveCarePatients', color='IntensiveCarePatients', text ='IntensiveCarePatients')
fig.data[0].marker.line.width = 0.5
fig.data[0].marker.line.color = "black"


fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_mode='hide')

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

fig.update_layout(
    yaxis=dict(
        showline=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

annotations = []

# source
annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.10,
                              xanchor='center', yanchor='top',
                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',
                              font=dict(family='arial narrow',
                                        size=8,
                                        color='rgb(96,96,96)'),
                              showarrow=False))

fig.update_layout(annotations=annotations)

fig.update_layout(title_text='<b>COVID-19</b>:<br>number of intensive care patients in Lombardy region',

 font=dict(family='calibri',
                                size=12,
                                color='rgb(64,64,64)'))

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_yaxes(title_text='Intensive care patients')
fig.update_yaxes(title_font=dict(size=14))
fig.update_xaxes(title_font=dict(size=14))
fig.update_layout(xaxis_showgrid=False)

fig.show()


# The plot below shows the timeline when the number of hospitalized patients reached 'half-apex' i.e. 50 percent of the maximum number (6500) compared to the number of recovered patients.

# In[ ]:


# define start date
start_date = '2020-03-24 17:00:00'
mask = (df_lo['Date'] > start_date)
df_lo = df_lo.loc[mask]

# plot figure
fig = go.Figure()
fig.add_trace(go.Bar(
    x=df_lo['Date'],
    y=df_lo['TotalHospitalizedPatients'],
    name='Hospitalized',
    marker_color='indianred'
))
fig.add_trace(go.Bar(
    x=df_lo['Date'],
    y=df_lo['Recovered'],
    name='Recovered',
    marker_color='lightsalmon'
))

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

fig.update_layout(
    yaxis=dict(
        showline=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

annotations = []

# source
annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.14,
                              xanchor='center', yanchor='top',
                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',
                              font=dict(family='arial narrow',
                                        size=8,
                                        color='rgb(96,96,96)'),
                              showarrow=False))

fig.update_layout(annotations=annotations)
fig.update_layout(barmode='group', xaxis_tickangle=-25)
fig.update_layout(title_text='<b>COVID-19</b>:<br>level of hospitalization and recovery in Lombardy region',
                  
      font=dict(family='calibri',
        size=12,
        color='rgb(64,64,64)'),
      legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1 
)

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_yaxes(title_text='Patients')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_font=dict(size=14))
fig.update_xaxes(title_font=dict(size=14))
fig.update_layout(xaxis_showgrid=False)
   
fig.show()


# On April 5th, the number of recovered patients was at the first time larger than the number of hospitalized patients. This recovery rate is steadily increasing although the overall need for hospital care in Lombardy was still steady in the beginning of April. In the latter half of April, also the total number of hospitalized patients in Lombardy has started to come down, mostly because the decreasing need for regular ward beds.   

# #### 2.2 Emilia-Romagna

# Next the same method will be applied to data from Emilia-Romagna region, where the apex stabilized around 4000 hospitalized patients.

# In[ ]:


# new dataframe
df_er = df_it.loc[df_it['RegionName'] == "Emilia-Romagna"]
start_date = '2020-04-04 17:00:00'
mask = (df_er['Date'] > start_date)
df_er = df_er.loc[mask]


# plot figure
fig = go.Figure()
fig.add_trace(go.Bar(
    x=df_er['Date'],
    y=df_er['TotalHospitalizedPatients'],
    name='Hospitalized',
    marker_color='indianred'
))
fig.add_trace(go.Bar(
    x=df_er['Date'],
    y=df_er['Recovered'],
    name='Recovered',
    marker_color='lightsalmon'
))

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

fig.update_layout(
    yaxis=dict(
        showline=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

annotations = []

# source
annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.14,
                              xanchor='center', yanchor='top',
                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',
                              font=dict(family='arial narrow',
                                        size=8,
                                        color='rgb(96,96,96)'),
                              showarrow=False))

fig.update_layout(annotations=annotations)
fig.update_layout(barmode='group', xaxis_tickangle=-25)
fig.update_layout(title_text='<b>COVID-19</b>:<br>level of hospitalization and recovery in Emilia-Romagna region',          
      font=dict(family='calibri',
        size=12,
        color='rgb(64,64,64)'),
      legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1 
)

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_yaxes(title_text='Patients')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_font=dict(size=14))
fig.update_xaxes(title_font=dict(size=14))
fig.update_layout(xaxis_showgrid=False)
   
fig.show()


# The total number of hospitalizations in Emilia-Romanga region has been about a third of that in Lombardy, where there was a constant urgency to free beds for incoming flux of new patients. Also, the recovery rate in Emilia-Romagna surpassed the rate of hospitalization only on April 12th, which means the two regions are not directly comparable with each other.

# #### 2.3 Piemonte

# In Piemonte, the COVID-19 peaked some time later than in Lombardy or Emilia-Romagna. This is why the per capita rate of confirmed cases (not included in this notebook) has been rising in Piemonte after the same rate had already stabilized in other analyzed regions in Italy. 
# 
# However also in Piemonte, the first signs of the downcurrent could be found from decreasing demand of ICU care. The total hospitalizations in Piemonte have also started to decrease, leaving 4000 as the peak in Piemonte hospitalizations. 

# In[ ]:


# new dataframe
data_piemonte = df_it[df_it.RegionName == 'Piemonte']


# plot figure
fig = px.bar(data_piemonte, x='Date', y='IntensiveCarePatients', color='IntensiveCarePatients', text ='IntensiveCarePatients')
fig.data[0].marker.line.width = 0.5
fig.data[0].marker.line.color = "black"

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_mode='hide')

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

fig.update_layout(
    yaxis=dict(
        showline=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

annotations = []

# source
annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.10,
                              xanchor='center', yanchor='top',
                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',
                              font=dict(family='arial narrow',
                                        size=8,
                                        color='rgb(96,96,96)'),
                              showarrow=False))

fig.update_layout(annotations=annotations)




fig.update_layout(title_text='<b>COVID-19</b>:<br>number of intensive care patients in Piemonte region',

 font=dict(family='calibri',
                                size=12,
                                color='rgb(64,64,64)'))


fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_yaxes(title_text='Intensive care patients')
fig.update_yaxes(title_font=dict(size=14))
fig.update_xaxes(title_font=dict(size=14))
fig.update_layout(xaxis_showgrid=False)

fig.show()


# In[ ]:


# new dataframe
df_pi = df_it.loc[df_it['RegionName'] == "Piemonte"]

# plot figure
fig = px.bar(df_pi, y='TotalHospitalizedPatients', x='Date', text='TotalHospitalizedPatients', color ='TotalHospitalizedPatients')
fig.data[0].marker.line.width = 0.5
fig.data[0].marker.line.color = "black"

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_mode='hide')

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

fig.update_layout(
    yaxis=dict(
        showline=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

annotations = []

# source
annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.12,
                              xanchor='center', yanchor='top',
                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',
                              font=dict(family='arial narrow',
                                        size=8,
                                        color='rgb(96,96,96)'),
                              showarrow=False))

fig.update_layout(annotations=annotations)
fig.update_layout(title_text='<b>COVID-19</b>:<br>total number of hospitalized patients in Piemonte region',
                  
 font=dict(family='calibri',
            size=12,
            color='rgb(64,64,64)'))

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_yaxes(title_text='Hospitalized patients')
fig.update_yaxes(title_font=dict(size=14))
fig.update_xaxes(title_font=dict(size=14))
fig.update_layout(xaxis_showgrid=False)

fig.show()


# In[ ]:


# new dataframe
df_pi = df_it.loc[df_it['RegionName'] == "Piemonte"]

start_date = '2020-04-02 17:00:00'
mask = (df_pi['Date'] > start_date)
df_pi = df_pi.loc[mask]

# plot figure
fig = go.Figure()
fig.add_trace(go.Bar(
    x=df_pi['Date'],
    y=df_pi['TotalHospitalizedPatients'],
    name='Hospitalized',
    marker_color='indianred'
))
fig.add_trace(go.Bar(
    x=df_pi['Date'],
    y=df_pi['Recovered'],
    name='Recovered',
    marker_color='lightsalmon'
))

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

fig.update_layout(
    yaxis=dict(
        showline=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

annotations = []

# source
annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.14,
                              xanchor='center', yanchor='top',
                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',
                              font=dict(family='arial narrow',
                                        size=8,
                                        color='rgb(96,96,96)'),
                              showarrow=False))

fig.update_layout(annotations=annotations)
fig.update_layout(barmode='group', xaxis_tickangle=-25)
fig.update_layout(title_text='<b>COVID-19</b>:<br>level of hospitalization and recovery in Piemonte region',          
      font=dict(family='calibri',
        size=12,
        color='rgb(64,64,64)'),
      legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1 
)

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_yaxes(title_text='Patients')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_font=dict(size=14))
fig.update_xaxes(title_font=dict(size=14))
fig.update_layout(xaxis_showgrid=False)
   
fig.show()


# The recovery level first surpassed hospitalizations on April 17th, which again indicates that the COVID-19 community spread in Piemonte was some days behind compared to other three regions in Italy included in this notebook. 

# #### 2.4 Veneto

# Finally, we take a look at Veneto region. There both the need for ICU and regular hospital ward beds decreased fastest of all regions.

# In[ ]:


# new dataframe
data_er = df_it[df_it.RegionName == 'Veneto']


# plot figure
fig = px.bar(data_er, x='Date', y='IntensiveCarePatients', color='IntensiveCarePatients', text='IntensiveCarePatients')
fig.data[0].marker.line.width = 0.5
fig.data[0].marker.line.color = "black"

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_minsize=8)

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

fig.update_layout(
    yaxis=dict(
        showline=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

annotations = []

# source
annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.10,
                              xanchor='center', yanchor='top',
                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',
                              font=dict(family='arial narrow',
                                        size=8,
                                        color='rgb(96,96,96)'),
                              showarrow=False))


fig.update_layout(annotations=annotations)

fig.update_layout(title_text='<b>COVID-19</b>:<br>number of intensive care patients in Veneto region',
                  
 font=dict(family='calibri',
            size=12,
            color='rgb(64,64,64)'))

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_yaxes(title_text='Intensive care patients')
fig.update_yaxes(title_font=dict(size=14))
fig.update_xaxes(title_font=dict(size=14))
fig.update_layout(xaxis_showgrid=False)

fig.show()


# In[ ]:


# new dataframe
df_vene = df_it[df_it.RegionName == 'Veneto']

# plot figure
fig = px.bar(df_vene, y='TotalHospitalizedPatients', x='Date', text='TotalHospitalizedPatients', color ='TotalHospitalizedPatients')
fig.data[0].marker.line.width = 0.5
fig.data[0].marker.line.color = "black"

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_mode='hide')

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

fig.update_layout(
    yaxis=dict(
        showline=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

annotations = []

# source
annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.10,
                              xanchor='center', yanchor='top',
                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',
                              font=dict(family='arial narrow',
                                        size=8,
                                        color='rgb(96,96,96)'),
                              showarrow=False))

fig.update_layout(annotations=annotations)

fig.update_layout(title_text='<b>COVID-19</b>:<br>total number of hospitalized patients in Veneto region',
                  
 font=dict(family='calibri',
            size=12,
            color='rgb(64,64,64)'))

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_yaxes(title_text='Hospitalized patients')
fig.update_yaxes(title_font=dict(size=14))
fig.update_xaxes(title_font=dict(size=14))
fig.update_layout(xaxis_showgrid=False)

fig.show()


# In[ ]:


# new dataframe
df_ve = df_it.loc[df_it['RegionName'] == "Veneto"]

start_date = '2020-04-02 17:00:00'
mask = (df_ve['Date'] > start_date)
df_ve = df_ve.loc[mask]

# plot figure
fig = go.Figure()
fig.add_trace(go.Bar(
    x=df_ve['Date'],
    y=df_ve['TotalHospitalizedPatients'],
    name='Hospitalized',
    marker_color='indianred'
))
fig.add_trace(go.Bar(
    x=df_ve['Date'],
    y=df_ve['Recovered'],
    name='Recovered',
    marker_color='lightsalmon'
))

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

fig.update_layout(
    yaxis=dict(
        showline=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

annotations = []

# source
annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.15,
                              xanchor='center', yanchor='top',
                              text='source: Sito del Dipartimento della Protezione Civile<br>Emergenza Coronavirus: la risposta nazionale<br>original dataset: https://github.com/pcm-dpc/COVID-19',
                              font=dict(family='arial narrow',
                                        size=8,
                                        color='rgb(96,96,96)'),
                              showarrow=False))

fig.update_layout(annotations=annotations)
fig.update_layout(barmode='group', xaxis_tickangle=-25)
fig.update_layout(title_text='<b>COVID-19</b>:<br>level of hospitalization and recovery in Veneto region',
                  
      font=dict(family='calibri',
        size=12,
        color='rgb(64,64,64)'),
      legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
)

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_yaxes(title_text='Patients')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_font=dict(size=14))
fig.update_xaxes(title_font=dict(size=14))
fig.update_layout(xaxis_showgrid=False)
   
fig.show()


# The recovery rate in Italy has increased faster than in other three regions, possbily because the relatively small overall number of patients in Veneto. In fact, in Veneto region the peak apex phase lasted for 9 days, which is half of the one in Lombardy.
# 
# From May 10th onwards, there seems to have been a regional update on collected data, which lowered both regular ward and ICU patient numbers considerably. What the numbers tell us is that on May 10th, the hospitalization rate in Veneto region was on the same level as in the first week of March, some two months ago before the upward curve started to rise steeply. 
# 
# Generally speaking, in Italy the need for ICU has started to decrease before overall need for hospital beds. Of course there is the unfortunate loss of lives included in this, but when indicators for 'bending the curve' are generally being looked at, consistently reduced need for ICU might be one of the first factors.
# 
# Concluding the analysis, the four regions in Italy all had the following characteristics:
# 
# - **The upward curve lasted for about a month**, keeping in mind that the regions included in the analysis all suffered a rapid outbreak. 
# - **The apex plateau phase lasted for 9-18 days**, likely depending on the total number of population in a specific region.
# - **The downcurrent of the curve is more gentle than the upward curve**, because the upward phase in many regions in Italy was more or less uncontained i.e. very steep.
# - **The COVID-19 spread upward curves in Northern Italy may be misleadingly steep**, because it is possible that COVID-19 began spreading there earlier than thought.
# - **The decreasing demand for ICU beds is the first sign of downcurrent**, partly because of the high mortality rate of patients intubated and put on ventilators.
# - **In four weeks after reaching the apex, the need for hospitalization in all four Italian regions had reduced about 50 percent. Six weeks after the apex, the same number was some 75 percent lower compared to apex. Thus the post-apex hospitalization came down about 25 percent from apex level every two weeks (or 12.5 percent/week).**
# - **The decreasing demand for hospital care does not mean that new positive confirmed COVID-19 decrease at the same rate**, which underlines the importance of caution during the regional reopening process.
# 
# 

# ### 3. New York State

# In New York state, the COVID-19 pandemic started some time later than in Italy. However, **as in Italy, it has been suggested that COVID-19 actually began spreading in New York well before the pandemic was officially recognized there. Therefore the steep rise of new cases in the pre-apex phase of the outbreak was likely affected by notable backlog in testing, and in reality the upward curve of new positive COVID-19 cases was more gentle.**
# 
# Just like Italy, New York state is basically a story of two different COVID-19 outbreaks. The city of New York soon became the most followed COVID-19 hotspot, since the hospitalization numbers there were off-scale compared to other regions. The upstate regions have less dense population, with many counties being basically rural areas. This has also affected the reopening process in New York state. Because of less dense population, the northern upstate areas are bound to meet the state reopening criteria much sooner than the metropolis region in the south. This will inevitably create challenges on how to prevent new COVID-19 cases spreading from southern regions to upstate during the summer months. 
# 
# Below is the plot of the *hospitalizedCurrently* column from New York state, presenting hospitalizations after May 1st. 

# In[ ]:


# new dataframe
url="https://covidtracking.com/api/v1/states/daily.csv"
df_us=pd.read_csv(url)

df_us = df_us.rename(columns = {'dateChecked':'Date'})
df_us.drop(df_us.index[df_us['Date'] == 'Invalid DateTime'], inplace = True)

df_us['Date'] =  pd.to_datetime(df_us['Date'], infer_datetime_format=True, errors = 'coerce')

df_la = df_us

#df_us.head(20)


#df_us_two = df_us[df_us['dateChecked'].notna()]
#pd.set_option('display.max_columns', None)
#df_us_two.Date.apply(str)
#df_us.head(10)
#df_us['Date'] =  pd.to_datetime(df_us['Date'], format = '%d-%b-%y %H.%M.%S.%f %p')
#df_us['Date']= pd.to_datetime(df_us['Date'],dayfirst=True)
#df_us['NewDate'] = df_us['Date'].dt.date
#df_us['NewDate'] =  pd.to_datetime(df_us['NewDate'], infer_datetime_format=True)


# drop column
dropped = ['hash']
df_us = df_us.drop(dropped, axis=1)

# mask dataframe
df_la = df_us[df_us.state == 'NY']

df_la['NewDate'] = df_la['Date'].dt.date
df_la['NewDate'] =  pd.to_datetime(df_la['NewDate'], infer_datetime_format=True)

start_date = '2020-05-01'
mask = (df_la['NewDate'] > start_date)
df_la = df_la.loc[mask]

df_la = df_la.drop_duplicates(subset='hospitalizedCurrently', keep="first")
df_la = df_la.drop_duplicates(subset='NewDate', keep="first")


# plot figure
fig = px.bar(df_la, y='hospitalizedCurrently', x='NewDate', text='hospitalizedCurrently', color ='hospitalizedCurrently')
fig.data[0].marker.line.width = 0.5
fig.data[0].marker.line.color = "black"

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_mode='hide')

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
       tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

fig.update_layout(
    yaxis=dict(
        showline=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

annotations = []

# source
annotations.append(dict(xref='paper', yref='paper', x=0.88, y=-0.10,
                              xanchor='center', yanchor='top',
                              text='Source: COVID-19 Tracking Project<br>original dataset: https://covidtracking.com',
                              font=dict(family='arial narrow',
                                        size=8,
                                        color='rgb(96,96,96)'),
                              showarrow=False))


fig.update_layout(annotations=annotations)

fig.update_layout(title_text='<b>COVID-19</b>:<br>number of current hospitalizations in New York state',
                  
                font=dict(family='calibri',
                                size=12,
                                color='rgb(64,64,64)'))

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_yaxes(title_text='Patients')
fig.update_yaxes(title_font=dict(size=14))
fig.update_xaxes(title_font=dict(size=14))
fig.update_layout(xaxis_showgrid=False)

fig.show()


# Given the fact that the COVID-19 apex was reached in New York state around April 15th, it is notable that in the latter part of April hospitals in New York state still took in some 20K new COVID-19 patients. In May, that number reduced by 50 percent to some 10K new hospitalizations.
# 
# 

# In Italy it could be seen that - instead of a steady curve - the new positive case number was more like a harmonic audio waveform. As noted, this may be caused by the testing and reporting system as well as the fact that on weekends people are more likely to leave their house compared to weekdays.
# 
# In any case - because of the possible 14-day incubation period - the **COVID-19 case numbers always show the recent past of the community spread.** Most virologists treat COVID-19 case numbers in month-long periods, consisting of two-week incubation period added with another two weeks of possible illness, which eventually creates the antibodies detected by testing.

# In[ ]:


# new dataframe
url="https://covidtracking.com/api/v1/states/daily.csv"
df_us=pd.read_csv(url)

df_us = df_us.rename(columns = {'dateChecked':'Date'})
df_us.drop(df_us.index[df_us['Date'] == 'Invalid DateTime'], inplace = True)

df_us['Date'] =  pd.to_datetime(df_us['Date'], infer_datetime_format=True, errors = 'coerce')

#df_la = df_us
# drop column
dropped = ['hash']
df_us = df_us.drop(dropped, axis=1)


df_us['NewDate'] = df_us['Date'].dt.date
df_us['NewDate'] =  pd.to_datetime(df_us['NewDate'], infer_datetime_format=True)

#df['time'] = df['full_date'].dt.time


# mask dataframe
start_date = '2020-07-01'
mask = (df_us['NewDate'] > start_date)
df_us_cases = df_us.loc[mask]

df_masked_nys  = df_us_cases.loc[df_us_cases['state'] == 'NY']
df_masked_ca  = df_us_cases.loc[df_us_cases['state'] == 'CA']
df_masked_fl  = df_us_cases.loc[df_us_cases['state'] == 'FL']

df_masked_nys = df_masked_nys.drop_duplicates(subset='positiveIncrease', keep="first")
df_masked_nys = df_masked_nys.drop_duplicates(subset='NewDate', keep="first")

df_masked_ca = df_masked_ca.drop_duplicates(subset='positiveIncrease', keep="first")
df_masked_ca = df_masked_ca.drop_duplicates(subset='NewDate', keep="first")

df_masked_fl = df_masked_fl.drop_duplicates(subset='positiveIncrease', keep="first")
df_masked_fl = df_masked_fl.drop_duplicates(subset='NewDate', keep="first")

# plot figure
fig = go.Figure()

fig.add_trace(go.Bar(
    x=df_masked_nys['NewDate'],
    y=df_masked_nys['positiveIncrease'],
    name='New York',
    marker_color='red'
))
fig.add_trace(go.Bar(
    x=df_masked_ca['NewDate'],
    y=df_masked_ca['positiveIncrease'],
    name='California',
    marker_color='orange'
))


fig.add_trace(go.Bar(
    x=df_masked_fl['NewDate'],
    y=df_masked_fl['positiveIncrease'],
    name='Florida',
    marker_color='green'
))


fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

fig.update_layout(
    yaxis=dict(
        showline=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

annotations = []

# source
annotations.append(dict(xref='paper', yref='paper', x=0.9, y=-0.14,
                              xanchor='center', yanchor='top',
                              text='Source: COVID-19 Tracking Project<br>original dataset: https://covidtracking.com',
                              font=dict(family='arial narrow',
                                        size=8,
                                        color='rgb(96,96,96)'),
                              showarrow=False))

fig.update_layout(annotations=annotations)

fig.update_layout(barmode='group', xaxis_tickangle=-25)

fig.update_layout(title_text='<b>COVID-19</b>:<br>new daily positive cases in some U.S. states after July 1st',
                  
     font=dict(family='calibri',
        size=12,
        color='rgb(64,64,64)'),
     legend=dict(
        x=0.85,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
)

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_yaxes(title_text='Positive cases')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_font=dict(size=14))
fig.update_xaxes(title_font=dict(size=14))
fig.update_layout(xaxis_showgrid=False)
   
fig.show()


# The plot below shows the development of total new confirmed COVID-19 cases in four U.S. states. For example in New York state, the number of new cases in two weeks leading to the apex peak (from April 1st to April 15th) grew from 83K (April 1st) to 214K (April 15th). This means that **in the two weeks leading to apex, New York state saw a total of 131K new confirmed COVID-19 cases (214K - 83K).** 
# 
# In the two weeks after the apex (the latter half of April), new confirmed COVID-19 cases grew from 214K (April 15th) to 304K (April 30th). In other words, **in two weeks after reaching the apex, New York state saw a total of 90K new confirmed COVID-19 cases (304K - 214K).**
# 
# **In New York state, the decrease of new confirmed cases in four-week timeline (two weeks pre-apex, two weeks post-apex) was about 30 percent (from 131K to 90K).** The relative decrease of new positive cases in New York state was larger than for example in Lombardy, but it is still notable that as a virus, *COVID-19 is just as infectious after the apex than before it*.

# In[ ]:


# new dataframe
url="https://covidtracking.com/api/v1/states/daily.csv"
df_us=pd.read_csv(url)

df_us = df_us.rename(columns = {'dateChecked':'Date'})
df_us.drop(df_us.index[df_us['Date'] == 'Invalid DateTime'], inplace = True)

df_us['Date'] =  pd.to_datetime(df_us['Date'], infer_datetime_format=True, errors = 'coerce')

#df_la = df_us
# drop column
dropped = ['hash']
df_us = df_us.drop(dropped, axis=1)


df_us['NewDate'] = df_us['Date'].dt.date
df_us['NewDate'] =  pd.to_datetime(df_us['NewDate'], infer_datetime_format=True, errors = 'coerce')

df_states = df_us.loc[df_us['state'].isin(['TX', 'NY', 'CA', 'FL'])]

# mask dataframe
start_date = '2020-03-15'
mask = (df_states['NewDate'] > start_date)
df_states = df_states.loc[mask]

# plot figure
fig = px.line(df_states, x="NewDate", y="positive",color='state')

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

fig.update_layout(
    yaxis=dict(
        showline=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        )))

annotations = []

# Source
annotations.append(dict(xref='paper', yref='paper', x=0.88, y=-0.10,
                              xanchor='center', yanchor='top',
                              text='Source: COVID-19 Tracking Project<br>original dataset: https://covidtracking.com',
                              font=dict(family='arial narrow',
                                        size=8,
                                        color='rgb(96,96,96)'),
                              showarrow=False))


fig.update_layout(annotations=annotations)

fig.update_layout(title_text='<b>COVID-19</b>:<br>number of confirmed positive cases in some U.S. states',
                  
                  
                font=dict(family='calibri',
                                size=12,
                                color='rgb(64,64,64)'))

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_yaxes(title_text='Confirmed cases')
fig.update_yaxes(title_font=dict(size=14))
fig.update_xaxes(title_font=dict(size=14))
fig.update_layout(xaxis_showgrid=False)

fig.show()


# Often the increase in positive confirmed COVID-19 cases is directly linked to testing rate. As we can see above, this is only partially true. For example New York state has launched an extensive testing and tracing program of late, but the new cases curve has more or less remained steady. Thus, from this viewpoint, **COVID-19 testing equals to monitoring or observing the virus by using scientific method. The results of that observation (in this case positive/negative test results) do not directly affect the phenomenon observed.**
# 
# ***The only way testing process can affect future COVID-19 spread is what happens based on the test results (isolating patients, contact tracing, regional restrictions etc). Considering COVID-19 testing rate only as a final goal or an intrinsic value will not render any significant improvement in limiting future virus spread without further action based on test results.***
# 
# Finally, ***the act of testing itself does not cure COVID-19 or increase a single infection to the total number of positive COVID-19 cases: the person tested positive was already ill before the test. Testing only makes the total number more accurate and relevant considering the actual community spread as well as reducing potential future cases by giving the chance to isolate people tested positive in time.***
# 
# This is why collecting more data on post-apex COVID-19 infections is essential. ***Why so many people still get infected even in a lockdown? What are the demographics (age etc.)?***
# 
# So far New York state has collected a couple of thousand cases on this very matter. As the state was under stay-at-home order during the study, home unsurprisingly shows up as the primary source of COVID-19 infection. *How did the virus enter those homes is however the question that should be answered before any restrictions can be safely lifted.*
# 
# If the data from Lombardy region is used as comparison (it is the largest of the four Italian regions analyzed), **the downcurve in New York will likely be a relatively long period with lingering demand for regular ward hospital care**. In the four regions in Italy, the demand for hospital care dropped 50 percent during the first month following the apex (in Italy's case April). However, it took only a month (in Italy's case March) for hospitalization care to increase from practically zero to apex. Also, **the hospitalization demand is likely to settle on a certain level at some point, since new COVID-19 cases won't totally stop from emerging.**
# 
# Recent studies both from Europe and United States suggest that **COVID-19 existed in both Europe and United States for several weeks before the outbreak was actually detected.** Drawing on this, it can be argued that in every region there can be a certain, manageable "safety level" of individual COVID-19 cases that do not cause a regional outbreak ('outbreak' defined here as consistent 1.0+ infection rate (Rt) ). *Finding out what that level is will be crucial in living with COVID-19 in the near future.*  
# 
# Another notion - not based on any plots in this notebook but still provable - is quite an obvious one. Both in Italy and in New York state, ***the most reliable factor in predicting the number of deaths seems to be the number of new confirmed COVID-19 cases***. Although these two are not plotted in this particular notebook, I have done so elsewhere, and the two are basically identical by shape in all analyzed regions. Also, ***in Veneto the first day of the post-apex curve with zero deaths (June 10) was the same exact day when no new positive COVID-19 cases were recorded in the region (it was also the first day with no ICU patients).***
# 
# Thus **the only effective way to reduce COVID-19 deaths seems to be reducing the number of new infections**, which of course is pretty much common sense (now backed up by data). Of course there is a delay between COVID-19 deaths and new cases, but as long as the overall COVID-19 mortality rate stays somewhere between 1-4 percent, the number of new future deaths is actually quite predictable if the number of new cases is known.
# 
# At the same time, **the reducing need for hospital care does not mean that there would be no new positive COVID-19 cases**. Although social distancing has slowed down the outbreak, COVID-19 has continued to spread among people both in Italy and in New York state. ***The lockdown measures have been successful in slowing down the current infection rate. Yet many of those "curbed infections" are still likely to take place at some point in the future, as the virus is still as infectious as before, no cure for it exists and there is no herd immunity.***
# 
# However - with the hospitalization rate on downfall - the data seems to suggest that as for now ***the recent COVID-19 cases are mostly found among people less vulnerable to the virus, thus not in need for hospital care.***
# 
# In the latter half of April (first two weeks after apex), New York state had about 30 percent less new COVID-19 cases than in two weeks leading to apex (first half of April). At the same time, the number of daily hospitalizations in New York state decreased about 40 percent (from 18.5K to 11.5K). Losing lives is a sad part of this difference, but it seems that the ***more recent COVID-19 cases do not require hospitalization or ICU care as often as those found at the beginning of outbreak. The situation changes only if these less vulnerable people cause a rapid community outbreak by infecting the elderly etc. (which was exactly what happened for example in Lombardy in March).***
# 
# One question (left strictly to professional virologists) is whether COVID-19 - while transforming itself into less vulnerable people - has gone through significant mutations in this process. Viruses are not very good at the process of multiplication, leading to gradually mutating forms. The key is however that *all viruses primarily aim at multiplying themselves, not destroying themselves by causing their host to perish.* 
# 
# ***So far the generally accepted rule of thumb has been that the more infectious a virus becomes, the less lethal it transforms to. Thus stating that a more infectious COVID-19 mutation is automatically 'worse' can be misleading.*** 
# 
# If and when all this has an effect to COVID-19 mortality rate in the future, it remains to be seen. It is also possible that with future mutations, COVID-19 may cause different non-respiratory illnesses among demographics previously considered mostly unaffected by the virus (small children, adolescents), as recent studies around the world seem to suggest. For better or worse, this would eventually render COVID-19 more of a general health scare than a primarily respiratory illness.
# 
# Concluding the notebook, if New York state follows the timeline set by the four regions in Italy (especially Lombardy), the following can be stated:
# 
# - **Since the COVID-19 outbreak really took off in New York state halfway into March, the upward part of the curve happened roughly from March 15th to April 15th. It is likely however that the COVID-19 spread in New York city and other metro areas in the state began earlier than that.**
# 
# - **Because of the large number of hospitalized COVID-19 patients in New York state, the exact timeline or shape of the downcurrent is hard to predict.**
# 
# - **During gradual reopening, everything depends on fast containment of possible new positive COVID-19 clusters.**
# 
# - **When the number of COVID-19 related deaths in New York state (especially densely populated downstate metro areas) fall on the same level as for example combined daily driver, cyclist and pedestrian traffic deaths, COVID-19 will eventually be considered by general public as an acceptable risk and an inevitable part of everyday life.**
