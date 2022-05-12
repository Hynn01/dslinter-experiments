#!/usr/bin/env python
# coding: utf-8

# ### Loading of Notebook might take some time because of Plotly visualizations, Kindly be patient!
# 
# #### After understanding the outbreak of COVID-19 across the World with help of different visualizations and performing Machine Learning model building for prediction and Time Series Forecasting models to understand future outbreak and how the numbers will be like in near future. Here I'm taking a step forward to understand the COVID-19 outbreak in India.
# 
# ### Before that thank you for such a nice response to my previous work, I really appreciate that.
# ### You can check out my previous work here for Analysis and Forecasts related to COVID-19 for World: 
# ### Click Here: [COVID-19 Visualizations, Predictions, Forecasting](http://www.kaggle.com/neelkudu28/covid-19-visualizations-predictions-forecasting)

# ## India's Timeline in fight against the Pandemic

# * **On 5 March** Amidst a surge in fresh cases being confirmed in Delhi NCR, the **Government of Delhi announced that all primary schools across Delhi would be shut until 31 March as a precaution.**
# 
# * **On 7 March:** primary schools in Jammu district and Samba district were closed down until 31 March after two suspected cases with "high viral load" were reported in Jammu.
# 
# * **On 9 March**, collector and district magistrate of Pathanamthitta district of Kerala declared three days long holidays for all educational institutions in the district. Karnataka declared indefinite holiday for all kindergarten and pre-primary schools in Bangalore. The holiday was extended to all primary schools up to fifth grade after a confirmed case was reported in the city.
# 
# * **On 10 March**, Kerala announced closure of all schools and colleges across the state until 31 March, with effect from 11 March.
# 
# * **On 12 March**, the Chief Minister of Delhi Arvind Kejriwal announced that all schools, colleges and cinema halls in New Delhi would be closed till the end of March and all public places disinfected as a precautionary measure. The Chief Minister of Karnataka B. S. Yediyurappa announced that all educational institutions, malls, cinema halls and pubs would be shut for a week across the state. He also issued prohibitory orders on public events such as parties and weddings.The Government of Odisha, declaring the outbreak a "disaster", announced the closure of educational institutions and cinema halls until the end of the month, as well as prohibition on non-essential official gatherings. The Government of Maharashtra declared the outbreak to be an epidemic in the cities of Mumbai, Navi Mumbai, Nagpur, Pune and Pimpri Chinchwad. It announced that all malls, cinema halls, swimming pools and gyms in these cities will be closed until 31 March.
# 
# * **On 13 March**, the Punjab and Chhattisgarh governments declared holidays in all schools and colleges till 31 March. Manipur government announced that all schools in the state, where examination are not being held would remain closed till 31 March.
# 
# * **On 14 March**, the Himachal Pradesh chief minister Jai Ram Thakur declared that all educational institutions and theatres would remain closed until 31 March as a precautionary measure in view of the threat of the coronavirus. Also, in the West Bengal government announced that all educational institutions will be closed till 31 March, however the board examinations will be conducted. Maharashtra government closed shopping malls, swimming pools, movie theatres, gyms and asked all schools and colleges in the state's urban areas to remain close till 31 March 2020. Government of Rajasthan announced to close all educational institutions, gyms, and cinema halls till 30 March, however ongoing school and college exams will continue.
# 
# * **On 15 March**, In Goa chief minister Pramod Sawant declared that all educational institutions would remain closed until 31 March. While the examinations of the 10th and 12th Goa Board of Secondary and Higher Secondary Education will be held as per schedule.The Gujarat government announced that all schools, colleges, cinema halls will be closed till 31 March, however the board examinations will be conducted.Vaishno Devi Shrine Board issued an advisory asking non-resident Indians and foreigners not to visit the temple for 28 days after landing in India.The Tamil Nadu and Telangana governments declared closure of schools, malls and theatres till 31 March. Ministry of Culture shut down all monuments and museums under Archaeological Survey of India till 31 March.
# 
# * **On 17 March**, schools, colleges and multiplexes in Uttar Pradesh were shut down till 2 April and on-going examinations were postponed. BMC ordered private firms in Mumbai to function "only at 50% of their staff capacity or face action under section 188 of the IPC". In Maharashtra, government offices were closed down for seven days. Census work was postponed. The GoM has also directed that no more than 50 people are allowed to gather at any place other than weddings. Pondicherry shut down schools, colleges, cinemas and gyms till 31 March.Mumbai Police ordered the closure of pubs, bars and discotheques till 31 March.
# 
# * **On 18 March**, district magistrate and deputy commissioner of Srinagar district in Jammu and Kashmir said that the entry of all foreign tourists has been banned in the entire union territory. Government of Andhra Pradesh announced closure of all educational institutions till 31 March.
# 
# * **On 23 March**, Chief Minister of Maharashtra announced that borders of all the districts will be closed, and a strict curfew will be implemented statewide.
# 
# * **On 22 March, the Government of India decided to completely lockdown 82 districts in 22 states and Union Territories of country where confirmed cases have been reported till 31 March.** At 6 am on 23 March Delhi was put under lockdown till at least 31 March. Essential services and commodities were to continue.80 cities including major cities such as Bengaluru, Chennai, Mumbai, Chandigarh and Kolkata were also put under lockdown. Inter-state movements are allowed during the lockdown period. However some states have closed their borders.
# 
# * **On 23 March**, union and state governments announced the lockdown of 75 districts where cases were reported.
# 
# * **On 24 March**, PM Narendra Modi announced a complete nationwide lockdown, starting from midnight for 21 days.
# 
# * **On 6 April**, In Telangana, CM suggested that the Lockdown shall continue after April 14 till June 3.
# 
# * **On 8 April**, The Yogi government has ordered the hot spot areas of 15 districts of Uttar Pradesh to be completely sealed till April 15.
# 
# * **On 9 April**, the Government of Odisha extended the Lockdown in the state till April 30.
# 
# * **On 10 April**, the Government of Punjab extended lockdown in the state till April 30.
# 
# * **On 11 April**, the Government of Maharashtra extended lockdown in the state minimum till April 30.
# 
# * **On 14 April**, PM Narendra Modi extended nationwide lockdown till 3 May, and he also announced that after 20 April a conditional relaxation in lockdown will be given in some areas of the country where spread have been prevented or contained.
# 
# * **On 29 April**, Punjab government announced for extension of curfew till 17 May.
# 
# * **On 1st May**, Central Government extended the Lockdown till 17th of May across the country. With conditional relaxation in Orange and Green Zones
# 
# * **On 5 May**, Telangana government announced for extension of lockdown till 29 May in their state.
# 
# * **On 16 May**, Punjab government announced for extension of lockdown till 31 May.
# 
# * **On 17 May**, NDMA extended the lockdown till 31 May in all indian states.
# 
# * **On 30 May**, the MHA announced that the ongoing lockdown would be further extended till 30 June in containment zones, with services resuming in a phased manner, starting from 8 June, in other zones. It is termed as "Unlock 1" and is stated to "have an economic focus".

# **Related videos to India COVID-19**
# 
# Gravitas: Wuhan Coronavirus: The big hurdles before India: https://www.youtube.com/watch?v=m4yzdqsKRKI
# 
# Gravitas: India reports it deadliest day | Coronavirus Outbreak: https://www.youtube.com/watch?v=9O28tlNXD_Q
# 
# Wuhan Coronavirus: 30 countries request India for medicine supplies | Gravitas: https://www.youtube.com/watch?v=keFNYV0NaxE
# 
# Containing the virus spread by using Bhilwara Model | Coronavirus News | India: https://www.youtube.com/watch?v=tLF0c0ztTsQ

# ## Objective of the Notebook
# The objective of this notebook is to understand outbreak of COVID-19 in India, Comparison of India with the neighbouring countries of India, Comparison with worst affected countries in this pandemic and try and build Machine Learning and Time Series Forecasting models to understand what are these numbers going to be like in near future days.

# ## Let's get started

# ### Importing required Python Packages and Libraries

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt
from datetime import timedelta
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing
from sklearn.metrics import mean_squared_error,r2_score
import statsmodels.api as sm
from fbprophet import Prophet
get_ipython().system('pip install plotly')
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
get_ipython().system('pip install pyramid-arima')
from pyramid.arima import auto_arima


# In[ ]:


covid=pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
covid.head()


# In[ ]:


#Extracting India's data 
covid_india=covid[covid['Country/Region']=="India"]

#Extracting other countries for comparison of worst affected countries
covid_spain=covid[covid['Country/Region']=="Spain"]
covid_us=covid[covid['Country/Region']=="US"]
covid_italy=covid[covid['Country/Region']=="Italy"]
covid_iran=covid[covid['Country/Region']=="Iran"]
covid_france=covid[covid['Country/Region']=="France"]
covid_uk=covid[covid['Country/Region']=="UK"]
covid_br=covid[covid['Country/Region']=="Brazil"]
covid_russia=covid[covid['Country/Region']=="Russia"]

#Extracting data of neighbouring countries
covid_pak=covid[covid['Country/Region']=="Pakistan"]
covid_china=covid[covid['Country/Region']=="Mainland China"]
covid_afg=covid[covid['Country/Region']=="Afghanistan"]
covid_nepal=covid[covid['Country/Region']=="Nepal"]
covid_bhutan=covid[covid['Country/Region']=="Bhutan"]
covid_lanka=covid[covid["Country/Region"]=="Sri Lanka"]
covid_ban=covid[covid["Country/Region"]=="Bangladesh"]


# In[ ]:


#Converting the date into Datetime format
covid_india["ObservationDate"]=pd.to_datetime(covid_india["ObservationDate"])
covid_spain["ObservationDate"]=pd.to_datetime(covid_spain["ObservationDate"])
covid_us["ObservationDate"]=pd.to_datetime(covid_us["ObservationDate"])
covid_italy["ObservationDate"]=pd.to_datetime(covid_italy["ObservationDate"])
covid_iran["ObservationDate"]=pd.to_datetime(covid_iran["ObservationDate"])
covid_france["ObservationDate"]=pd.to_datetime(covid_france["ObservationDate"])
covid_uk["ObservationDate"]=pd.to_datetime(covid_uk["ObservationDate"])
covid_br["ObservationDate"]=pd.to_datetime(covid_br["ObservationDate"])
covid_russia["ObservationDate"]=pd.to_datetime(covid_russia["ObservationDate"])

covid_pak["ObservationDate"]=pd.to_datetime(covid_pak["ObservationDate"])
covid_china["ObservationDate"]=pd.to_datetime(covid_china["ObservationDate"])
covid_afg["ObservationDate"]=pd.to_datetime(covid_afg["ObservationDate"])
covid_nepal["ObservationDate"]=pd.to_datetime(covid_nepal["ObservationDate"])
covid_bhutan["ObservationDate"]=pd.to_datetime(covid_bhutan["ObservationDate"])
covid_lanka["ObservationDate"]=pd.to_datetime(covid_lanka["ObservationDate"])
covid_ban["ObservationDate"]=pd.to_datetime(covid_ban["ObservationDate"])


# In[ ]:


#Grouping the data based on the Date 
india_datewise=covid_india.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
spain_datewise=covid_spain.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
us_datewise=covid_us.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
italy_datewise=covid_italy.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
iran_datewise=covid_iran.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
france_datewise=covid_france.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
uk_datewise=covid_uk.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
brazil_datewise=covid_br.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
russia_datewise=covid_russia.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

pak_datewise=covid_pak.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
china_datewise=covid_china.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
afg_datewise=covid_afg.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
nepal_datewise=covid_nepal.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
bhutan_datewise=covid_bhutan.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
lanka_datewise=covid_lanka.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
ban_datewise=covid_ban.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})


# In[ ]:


#Adding week column to perfom weekly analysis further ahead
india_datewise["WeekofYear"]=india_datewise.index.weekofyear
spain_datewise["WeekofYear"]=spain_datewise.index.weekofyear
us_datewise["WeekofYear"]=us_datewise.index.weekofyear
italy_datewise["WeekofYear"]=italy_datewise.index.weekofyear
iran_datewise["WeekofYear"]=iran_datewise.index.weekofyear
france_datewise["WeekofYear"]=france_datewise.index.weekofyear
uk_datewise["WeekofYear"]=uk_datewise.index.weekofyear
brazil_datewise["WeekofYear"]=brazil_datewise.index.weekofyear
russia_datewise["WeekofYear"]=russia_datewise.index.weekofyear

pak_datewise["WeekofYear"]=pak_datewise.index.weekofyear
china_datewise["WeekofYear"]=china_datewise.index.weekofyear
afg_datewise["WeekofYear"]=afg_datewise.index.weekofyear
nepal_datewise["WeekofYear"]=nepal_datewise.index.weekofyear
bhutan_datewise["WeekofYear"]=bhutan_datewise.index.weekofyear
lanka_datewise["WeekofYear"]=lanka_datewise.index.weekofyear
ban_datewise["WeekofYear"]=ban_datewise.index.weekofyear


# In[ ]:


india_datewise["Days Since"]=(india_datewise.index-india_datewise.index[0])
india_datewise["Days Since"]=india_datewise["Days Since"].dt.days


# In[ ]:


No_Lockdown=covid_india[covid_india["ObservationDate"]<pd.to_datetime("2020-03-21")]
Lockdown_1=covid_india[(covid_india["ObservationDate"]>=pd.to_datetime("2020-03-21"))&(covid_india["ObservationDate"]<pd.to_datetime("2020-04-15"))]
Lockdown_2=covid_india[(covid_india["ObservationDate"]>=pd.to_datetime("2020-04-15"))&(covid_india["ObservationDate"]<pd.to_datetime("2020-05-04"))]
Lockdown_3=covid_india[(covid_india["ObservationDate"]>=pd.to_datetime("2020-05-04"))&(covid_india["ObservationDate"]<pd.to_datetime("2020-05-19"))]
Lockdown_4=covid_india[(covid_india["ObservationDate"]>=pd.to_datetime("2020-05-19"))&(covid_india["ObservationDate"]<=pd.to_datetime("2020-05-31"))]
Unlock_1=covid_india[(covid_india["ObservationDate"]>=pd.to_datetime("2020-06-01"))&(covid_india["ObservationDate"]<=pd.to_datetime("2020-06-30"))]
Unlock_2=covid_india[(covid_india["ObservationDate"]>=pd.to_datetime("2020-07-01"))]

No_Lockdown_datewise=No_Lockdown.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
Lockdown_1_datewise=Lockdown_1.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
Lockdown_2_datewise=Lockdown_2.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
Lockdown_3_datewise=Lockdown_3.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
Lockdown_4_datewise=Lockdown_4.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
Unlock_1_datewise=Unlock_1.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
Unlock_2_datewise=Unlock_2.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})


# In[ ]:


covid["ObservationDate"]=pd.to_datetime(covid["ObservationDate"])
grouped_country=covid.groupby(["Country/Region","ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})


# In[ ]:


grouped_country["Active Cases"]=grouped_country["Confirmed"]-grouped_country["Recovered"]-grouped_country["Deaths"]
grouped_country["log_confirmed"]=np.log(grouped_country["Confirmed"])
grouped_country["log_active"]=np.log(grouped_country["Active Cases"])


# ## Exploratory Data Analysis for India

# In[ ]:


print("Number of Confirmed Cases",india_datewise["Confirmed"].iloc[-1])
print("Number of Recovered Cases",india_datewise["Recovered"].iloc[-1])
print("Number of Death Cases",india_datewise["Deaths"].iloc[-1])
print("Number of Active Cases",india_datewise["Confirmed"].iloc[-1]-india_datewise["Recovered"].iloc[-1]-india_datewise["Deaths"].iloc[-1])
print("Number of Closed Cases",india_datewise["Recovered"].iloc[-1]+india_datewise["Deaths"].iloc[-1])
print("Approximate Number of Confirmed Cases per day",round(india_datewise["Confirmed"].iloc[-1]/india_datewise.shape[0]))
print("Approximate Number of Recovered Cases per day",round(india_datewise["Recovered"].iloc[-1]/india_datewise.shape[0]))
print("Approximate Number of Death Cases per day",round(india_datewise["Deaths"].iloc[-1]/india_datewise.shape[0]))
print("Number of New Cofirmed Cases in last 24 hours are",india_datewise["Confirmed"].iloc[-1]-india_datewise["Confirmed"].iloc[-2])
print("Number of New Recoverd Cases in last 24 hours are",india_datewise["Recovered"].iloc[-1]-india_datewise["Recovered"].iloc[-2])
print("Number of New Death Cases in last 24 hours are",india_datewise["Deaths"].iloc[-1]-india_datewise["Deaths"].iloc[-2])


# #### Active Cases = Number of Confirmed Cases - Number of Recovered Cases - Number of Death Cases
# #### Increase in number of Active Cases is probable an indication of Recovered case or Death case number is dropping in comparison to number of Confirmed Cases drastically. Will look for the conclusive evidence for the same in the notebook ahead.

# In[ ]:


fig=px.bar(x=india_datewise.index,y=india_datewise["Confirmed"]-india_datewise["Recovered"]-india_datewise["Deaths"])
fig.update_layout(title="Distribution of Number of Active Cases",
                  xaxis_title="Date",yaxis_title="Number of Cases",)
fig.show()


# #### Closed Cases = Number of Recovered Cases + Number of Death Cases 
# #### Increase in number of Closed classes imply either more patients are getting recovered from the disease or more pepole are dying because of COVID-19. Will look for conclusive evidence ahead.

# In[ ]:


fig=px.bar(x=india_datewise.index,y=india_datewise["Recovered"]+india_datewise["Deaths"])
fig.update_layout(title="Distribution of Number of Closed Cases",
                  xaxis_title="Date",yaxis_title="Number of Cases")
fig.show()


# #### Growth Rate of Confirmed, Recoverd and Death Cases

# In[ ]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Confirmed"],
                    mode='lines+markers',
                    name='Confirmed Cases'))
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Recovered"],
                    mode='lines+markers',
                    name='Recovered Cases'))
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Deaths"],
                    mode='lines+markers',
                    name='Death Cases'))
fig.update_layout(title="Growth of different types of cases in India",
                 xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# Higher Exponential growth of Confirmed Cases in comparison to Recovered and Death Cases is a conclusive evidence why there is increase in number of Active Cases.

# #### Recovery and Mortality Rate

# In[ ]:


print('Mean Recovery Rate: ',((india_datewise["Recovered"]/india_datewise["Confirmed"])*100).mean())
print('Mean Mortality Rate: ',((india_datewise["Deaths"]/india_datewise["Confirmed"])*100).mean())
print('Median Recovery Rate: ',((india_datewise["Recovered"]/india_datewise["Confirmed"])*100).median())
print('Median Mortality Rate: ',((india_datewise["Deaths"]/india_datewise["Confirmed"])*100).median())

fig = make_subplots(rows=2, cols=1,
                   subplot_titles=("Recovery Rate", "Mortatlity Rate"))
fig.add_trace(
    go.Scatter(x=india_datewise.index, y=(india_datewise["Recovered"]/india_datewise["Confirmed"])*100,
              name="Recovery Rate"),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=india_datewise.index, y=(india_datewise["Deaths"]/india_datewise["Confirmed"])*100,
              name="Mortality Rate"),
    row=2, col=1
)
fig.update_layout(height=1000,legend=dict(x=-0.1,y=1.2,traceorder="normal"))
fig.update_xaxes(title_text="Date", row=1, col=1)
fig.update_yaxes(title_text="Recovery Rate", row=1, col=1)
fig.update_xaxes(title_text="Date", row=1, col=2)
fig.update_yaxes(title_text="Mortality Rate", row=1, col=2)
fig.show()


# #### Mortality rate = (Number of Death Cases / Number of Confirmed Cases) x 100
# #### Recovery Rate= (Number of Recoverd Cases / Number of Confirmed Cases) x 100
# Recovery Rate was initially very high when the number of positive (Confirmed) cases were low and showed a drastic drop with increasing number of cases. Increasing Mortality rate and dropped Recovery Rate is worrying sign for India.
# 
# Increasing Mortality Rate and very slowly increasing Recovery Rate is conclusive evidence for increase in number of Closed Cases
# 
# Recovery Rate is showing an upward trend which is a really good sign. Mortality Rate is showing a slight dips but with occasional upward trends.

# ### Growth Factor for different types of Cases 
# Growth factor is the factor by which a quantity multiplies itself over time. The formula used is:
# 
# **Formula: Every day's new (Confirmed,Recovered,Deaths) / new (Confirmed,Recovered,Deaths) on the previous day.**
# 
# A growth factor **above 1 indicates an increase correspoding cases**.
# 
# A growth factor **above 1 but trending downward** is a positive sign, whereas a **growth factor constantly above 1 is the sign of exponential growth**.
# 
# A growth factor **constant at 1 indicates there is no change in any kind of cases**.

# In[ ]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Confirmed"]/india_datewise["Confirmed"].shift(),
                    mode='lines',
                    name='Growth Factor of Confirmed Cases'))
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Recovered"]/india_datewise["Recovered"].shift(),
                    mode='lines',
                    name='Growth Factor of Recovered Cases'))
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Deaths"]/india_datewise["Deaths"].shift(),
                    mode='lines',
                    name='Growth Factor of Death Cases'))
fig.update_layout(title="Datewise Growth Factor of different types of cases in India",
                 xaxis_title="Date",yaxis_title="Growth Factor",
                 legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# Growth Factor of Recoverd Cases is constantly very close to 1 indicating the Recovery Rate very low which was high intially as discussed earlier, with Growth Factor of Confirmed and Death Cases well above 1 is an indication of considerable growth in both types of Cases.

# ### Growth Factor for Active and Closed Cases
# Growth factor is the factor by which a quantity multiplies itself over time. The formula used is:
# 
# **Formula: Every day's new (Active and Closed Cases) / new (Active and Closed Cases) on the previous day.**
# 
# A growth factor **above 1 indicates an increase correspoding cases**.
# 
# A growth factor **above 1 but trending downward** is a positive sign.
# 
# A growth factor **constant at 1 indicates there is no change in any kind of cases**.
# 
# A growth factor **below 1 indicates real positive sign implying more patients are getting recovered or dying as compared to the Confirmed Cases**.

# In[ ]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=india_datewise.index, 
                        y=(india_datewise["Confirmed"]-india_datewise["Recovered"]-india_datewise["Deaths"])/(india_datewise["Confirmed"]-india_datewise["Recovered"]-india_datewise["Deaths"]).shift(),
                    mode='lines',
                    name='Growth Factor of Active Cases'))
fig.add_trace(go.Scatter(x=india_datewise.index, y=(india_datewise["Recovered"]+india_datewise["Deaths"])/(india_datewise["Recovered"]+india_datewise["Deaths"]).shift(),
                    mode='lines',
                    name='Growth Factor of Closed Cases'))
fig.update_layout(title="Datewise Growth Factor of Active and Closed cases in India",
                 xaxis_title="Date",yaxis_title="Growth Factor",
                 legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[ ]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Confirmed"].diff().fillna(0),
                    mode='lines+markers',
                    name='Confirmed Cases'))
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Recovered"].diff().fillna(0),
                    mode='lines+markers',
                    name='Recovered Cases'))
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Deaths"].diff().fillna(0),
                    mode='lines+markers',
                    name='Death Cases'))
fig.update_layout(title="Daily increase in different types of cases in India",
                 xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[ ]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Confirmed"].diff().rolling(window=7).mean(),
                    mode='lines+markers',
                    name='Confirmed Cases'))
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Recovered"].diff().rolling(window=7).mean(),
                    mode='lines+markers',
                    name='Recovered Cases'))
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Deaths"].diff().rolling(window=7).mean().diff(),
                    mode='lines+markers',
                    name='Death Cases'))
fig.update_layout(title="7 Days Rolling mean of Confirmed, Recovered and Death Cases",
                 xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[ ]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=india_datewise.index, y=(india_datewise["Confirmed"]-india_datewise["Recovered"]-india_datewise["Deaths"]).diff().rolling(window=7).mean(),
                    mode='lines+markers',
                    name='Active Cases'))
fig.add_trace(go.Scatter(x=india_datewise.index, y=(india_datewise["Recovered"]+india_datewise["Deaths"]).diff().rolling(window=7).mean(),
                    mode='lines+markers',
                    name='Closed Cases'))
fig.update_layout(title="7 Days Rolling mean of Active and Closed Cases",
                 xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[ ]:


week_num_india=[]
india_weekwise_confirmed=[]
india_weekwise_recovered=[]
india_weekwise_deaths=[]
w=1
for i in list(india_datewise["WeekofYear"].unique()):
    india_weekwise_confirmed.append(india_datewise[india_datewise["WeekofYear"]==i]["Confirmed"].iloc[-1])
    india_weekwise_recovered.append(india_datewise[india_datewise["WeekofYear"]==i]["Recovered"].iloc[-1])
    india_weekwise_deaths.append(india_datewise[india_datewise["WeekofYear"]==i]["Deaths"].iloc[-1])
    week_num_india.append(w)
    w=w+1


# In[ ]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=week_num_india, y=india_weekwise_confirmed,
                    mode='lines+markers',
                    name='Weekly Growth of Confirmed Cases'))
fig.add_trace(go.Scatter(x=week_num_india, y=india_weekwise_recovered,
                    mode='lines+markers',
                    name='Weekly Growth of Recovered Cases'))
fig.add_trace(go.Scatter(x=week_num_india, y=india_weekwise_deaths,
                    mode='lines+markers',
                    name='Weekly Growth of Death Cases'))
fig.update_layout(title="Weekly Growth of different types of Cases in India",
                 xaxis_title="Week Number",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[ ]:


print("Average weekly increase in number of Confirmed Cases",round(pd.Series(india_weekwise_confirmed).diff().fillna(0).mean()))
print("Average weekly increase in number of Recovered Cases",round(pd.Series(india_weekwise_recovered).diff().fillna(0).mean()))
print("Average weekly increase in number of Death Cases",round(pd.Series(india_weekwise_deaths).diff().fillna(0).mean()))

fig = make_subplots(rows=1, cols=2)
fig.add_trace(
    go.Bar(x=week_num_india, y=pd.Series(india_weekwise_confirmed).diff().fillna(0),
          name="Weekly rise in number of Confirmed Cases"),
    row=1, col=1
)
fig.add_trace(
    go.Bar(x=week_num_india, y=pd.Series(india_weekwise_deaths).diff().fillna(0),
          name="Weekly rise in number of Death Cases"),
    row=1, col=2
)
fig.update_layout(title="India's Weekly increas in Number of Confirmed and Death Cases",
    font=dict(
        size=10,
    )
)
fig.update_layout(width=900,legend=dict(x=0,y=-0.5,traceorder="normal"))
fig.update_xaxes(title_text="Date", row=1, col=1)
fig.update_yaxes(title_text="Number of Cases", row=1, col=1)
fig.update_xaxes(title_text="Date", row=1, col=2)
fig.update_yaxes(title_text="Number of Cases", row=1, col=2)
fig.show()


# #### Week 33rd is currently going on.
# 
# Confirmed Cases are showing upward trend every week, recording highest number of Confirmed cases in 28th week (400k+ Confirmed Cases).
# 
# Death cases showed a very slight dip in 16th week. The 22nd week recorded less number of deaths against the trend which is a good sign. 23rd week showed an increase number of deaths yet again and the same trend is being followed in weeks ahead.
# 
# Infection and Death rate both are high and showing upward trend every week continuously.

# ### Doubling Rate of COVID-19 Confirmed Cases

# In[ ]:


cases=65
double_days=[]
C=[]
while(1):
    double_days.append(int(india_datewise[india_datewise["Confirmed"]<=cases].iloc[[-1]]["Days Since"]))
    C.append(cases)
    cases=cases*2
    if(cases<india_datewise["Confirmed"].max()):
        continue
    else:
        break
        
cases=65
tipling_days=[]
C1=[]
while(1):
    tipling_days.append(int(india_datewise[india_datewise["Confirmed"]<=cases].iloc[[-1]]["Days Since"]))
    C1.append(cases)
    cases=cases*3
    if(cases<india_datewise["Confirmed"].max()):
        continue
    else:
        break
        
india_doubling=pd.DataFrame(list(zip(C,double_days)),columns=["No. of cases","Days since first case"])
india_doubling["Number of days required to Double the cases"]=india_doubling["Days since first case"].diff().fillna(india_doubling["Days since first case"].iloc[0])

india_tripling=pd.DataFrame(list(zip(C1,tipling_days)),columns=["No. of cases","Days since first case"])
india_tripling["Number of days required to Triple the cases"]=india_tripling["Days since first case"].diff().fillna(india_tripling["Days since first case"].iloc[0])

india_doubling.style.background_gradient(cmap='Reds')


# ### Tripling Rate of COVID-19 Confirmed Cases

# In[ ]:


india_tripling.style.background_gradient(cmap='Reds')


# In[ ]:


case_100k=5000
rise_100k=[]
C1=[]
while(1):
    rise_100k.append(int(india_datewise[india_datewise["Confirmed"]<=case_100k].iloc[[-1]]["Days Since"]))
    C1.append(case_100k)
    case_100k=case_100k+100000
    if(case_100k<india_datewise["Confirmed"].max()):
        continue
    else:
        break
rate_100k=pd.DataFrame(list(zip(C1,rise_100k)),columns=["No. of Cases","Days Since first Case"])
rate_100k["Days required for increase by 100K"]=rate_100k["Days Since first Case"].diff().fillna(rate_100k["Days Since first Case"].iloc[0])


# In[ ]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=rate_100k["No. of Cases"], y=rate_100k["Days required for increase by 100K"],
                    mode='lines+markers',
                    name='Weekly Growth of Confirmed Cases'))
fig.update_layout(title="Number of Days required for increase in number of cases by 100K",
                 xaxis_title="Number of Cases",yaxis_title="Number of Days")
fig.show()


# ## Lockdown Analysis for India

# In[ ]:


No_Lockdown_datewise["Active Cases"]=No_Lockdown_datewise["Confirmed"]-No_Lockdown_datewise["Recovered"]-No_Lockdown_datewise["Deaths"]
Lockdown_1_datewise["Active Cases"]=Lockdown_1_datewise["Confirmed"]-Lockdown_1_datewise["Recovered"]-Lockdown_1_datewise["Deaths"]
Lockdown_2_datewise["Active Cases"]=Lockdown_2_datewise["Confirmed"]-Lockdown_2_datewise["Recovered"]-Lockdown_2_datewise["Deaths"]
Lockdown_3_datewise["Active Cases"]=Lockdown_3_datewise["Confirmed"]-Lockdown_3_datewise["Recovered"]-Lockdown_3_datewise["Deaths"]
Lockdown_4_datewise["Active Cases"]=Lockdown_4_datewise["Confirmed"]-Lockdown_4_datewise["Recovered"]-Lockdown_4_datewise["Deaths"]
Unlock_1_datewise["Active Cases"]=Unlock_1_datewise["Confirmed"]-Unlock_1_datewise["Recovered"]-Unlock_1_datewise["Deaths"]
Unlock_2_datewise["Active Cases"]=Unlock_2_datewise["Confirmed"]-Unlock_2_datewise["Recovered"]-Unlock_2_datewise["Deaths"]


No_Lockdown_datewise["Days Since"]=(No_Lockdown_datewise.index-No_Lockdown_datewise.index.min()).days
Lockdown_1_datewise["Days Since"]=(Lockdown_1_datewise.index-Lockdown_1_datewise.index.min()).days
Lockdown_2_datewise["Days Since"]=(Lockdown_2_datewise.index-Lockdown_2_datewise.index.min()).days
Lockdown_3_datewise["Days Since"]=(Lockdown_3_datewise.index-Lockdown_3_datewise.index.min()).days
Lockdown_4_datewise["Days Since"]=(Lockdown_4_datewise.index-Lockdown_4_datewise.index.min()).days
Unlock_1_datewise["Days Since"]=(Unlock_1_datewise.index-Unlock_1_datewise.index.min()).days
Unlock_2_datewise["Days Since"]=(Unlock_2_datewise.index-Unlock_2_datewise.index.min()).days


cases=1
NL_doubling=[]
C=[]
while(1):
    NL_doubling.append(int(No_Lockdown_datewise[No_Lockdown_datewise["Confirmed"]<=cases].iloc[[-1]]["Days Since"]))
    C.append(cases)
    cases=cases*2
    if(cases<No_Lockdown_datewise["Confirmed"].max()):
        continue
    else:
        break
NL_Double_rate=pd.DataFrame(list(zip(C,NL_doubling)),columns=["No. of Cases","Days Since First Case"])
NL_Double_rate["Days required for Doubling"]=NL_Double_rate["Days Since First Case"].diff().fillna(NL_Double_rate["Days Since First Case"].iloc[0])

cases=Lockdown_1_datewise["Confirmed"].min()
L1_doubling=[]
C=[]
while(1):
    L1_doubling.append(int(Lockdown_1_datewise[Lockdown_1_datewise["Confirmed"]<=cases].iloc[[-1]]["Days Since"]))
    C.append(cases)
    cases=cases*2
    if(cases<Lockdown_1_datewise["Confirmed"].max()):
        continue
    else:
        break
L1_Double_rate=pd.DataFrame(list(zip(C,L1_doubling)),columns=["No. of Cases","Days Since Lockdown 1.0"])
L1_Double_rate["Days required for Doubling"]=L1_Double_rate["Days Since Lockdown 1.0"].diff().fillna(L1_Double_rate["Days Since Lockdown 1.0"].iloc[0])

cases=Lockdown_2_datewise["Confirmed"].min()
L2_doubling=[]
C=[]
while(1):
    L2_doubling.append(int(Lockdown_2_datewise[Lockdown_2_datewise["Confirmed"]<=cases].iloc[[-1]]["Days Since"]))
    C.append(cases)
    cases=cases*2
    if(cases<Lockdown_2_datewise["Confirmed"].max()):
        continue
    else:
        break
L2_Double_rate=pd.DataFrame(list(zip(C,L2_doubling)),columns=["No. of Cases","Days Since Lockdown 2.0"])
L2_Double_rate["Days required for Doubling"]=L2_Double_rate["Days Since Lockdown 2.0"].diff().fillna(L2_Double_rate["Days Since Lockdown 2.0"].iloc[0])

cases=Lockdown_3_datewise["Confirmed"].min()
L3_doubling=[]
C=[]
while(1):
    L3_doubling.append(int(Lockdown_3_datewise[Lockdown_3_datewise["Confirmed"]<=cases].iloc[[-1]]["Days Since"]))
    C.append(cases)
    cases=cases*2
    if(cases<Lockdown_3_datewise["Confirmed"].max()):
        continue
    else:
        break
L3_Double_rate=pd.DataFrame(list(zip(C,L3_doubling)),columns=["No. of Cases","Days Since Lockdown 3.0"])
L3_Double_rate["Days required for Doubling"]=L3_Double_rate["Days Since Lockdown 3.0"].diff().fillna(L3_Double_rate["Days Since Lockdown 3.0"].iloc[0])

cases=Lockdown_4_datewise["Confirmed"].min()
L4_doubling=[]
C=[]
while(1):
    L4_doubling.append(int(Lockdown_4_datewise[Lockdown_4_datewise["Confirmed"]<=cases].iloc[[-1]]["Days Since"]))
    C.append(cases)
    cases=cases*2
    if(cases<Lockdown_4_datewise["Confirmed"].max()):
        continue
    else:
        break
L4_Double_rate=pd.DataFrame(list(zip(C,L4_doubling)),columns=["No. of Cases","Days Since Lockdown 4.0"])
L4_Double_rate["Days required for Doubling"]=L4_Double_rate["Days Since Lockdown 4.0"].diff().fillna(L4_Double_rate["Days Since Lockdown 4.0"].iloc[0])

cases=Unlock_1_datewise["Confirmed"].min()
UL1_doubling=[]
C=[]
while(1):
    UL1_doubling.append(int(Unlock_1_datewise[Unlock_1_datewise["Confirmed"]<=cases].iloc[[-1]]["Days Since"]))
    C.append(cases)
    cases=cases*2
    if(cases<Unlock_1_datewise["Confirmed"].max()):
        continue
    else:
        break
UL1_Double_rate=pd.DataFrame(list(zip(C,UL1_doubling)),columns=["No. of Cases","Days Since Lockdown 4.0"])
UL1_Double_rate["Days required for Doubling"]=UL1_Double_rate["Days Since Lockdown 4.0"].diff().fillna(UL1_Double_rate["Days Since Lockdown 4.0"].iloc[0])


# In[ ]:


print("Average Active Cases growth rate in Lockdown 1.0: ",(Lockdown_1_datewise["Active Cases"]/Lockdown_1_datewise["Active Cases"].shift()).mean())
print("Median Active Cases growth rate in Lockdown 1.0: ",(Lockdown_1_datewise["Active Cases"]/Lockdown_1_datewise["Active Cases"].shift()).median())
print("Average Active Cases growth rate in Lockdown 2.0: ",(Lockdown_2_datewise["Active Cases"]/Lockdown_2_datewise["Active Cases"].shift()).mean())
print("Median Active Cases growth rate in Lockdown 2.0: ",(Lockdown_2_datewise["Active Cases"]/Lockdown_2_datewise["Active Cases"].shift()).median())
print("Average Active Cases growth rate in Lockdown 3.0: ",(Lockdown_3_datewise["Active Cases"]/Lockdown_3_datewise["Active Cases"].shift()).mean())
print("Median Active Cases growth rate in Lockdown 3.0: ",(Lockdown_3_datewise["Active Cases"]/Lockdown_3_datewise["Active Cases"].shift()).median())
print("Average Active Cases growth rate in Lockdown 4.0: ",(Lockdown_4_datewise["Active Cases"]/Lockdown_4_datewise["Active Cases"].shift()).mean())
print("Median Active Cases growth rate in Lockdown 4.0: ",(Lockdown_4_datewise["Active Cases"]/Lockdown_4_datewise["Active Cases"].shift()).median())
print("Average Active Cases growth rate in Unlock 1.0: ",(Unlock_1_datewise["Active Cases"]/Unlock_1_datewise["Active Cases"].shift()).mean())
print("Median Active Cases growth rate in Unlock 1.0: ",(Unlock_1_datewise["Active Cases"]/Unlock_1_datewise["Active Cases"].shift()).median())


fig=go.Figure()
fig.add_trace(go.Scatter(y=list(Lockdown_1_datewise["Active Cases"]/Lockdown_1_datewise["Active Cases"].shift()),
                    mode='lines+markers',
                    name='Growth Factor of Lockdown 1.0 Active Cases'))
fig.add_trace(go.Scatter(y=list(Lockdown_2_datewise["Active Cases"]/Lockdown_2_datewise["Active Cases"].shift()),
                    mode='lines+markers',
                    name='Growth Factor of Lockdown 2.0 Active Cases'))
fig.add_trace(go.Scatter(y=list(Lockdown_3_datewise["Active Cases"]/Lockdown_3_datewise["Active Cases"].shift()),
                    mode='lines+markers',
                    name='Growth Factor of Lockdown 3.0 Active Cases'))
fig.add_trace(go.Scatter(y=list(Lockdown_4_datewise["Active Cases"]/Lockdown_4_datewise["Active Cases"].shift()),
                    mode='lines+markers',
                    name='Growth Factor of Lockdown 4.0 Active Cases'))
fig.add_trace(go.Scatter(y=list(Unlock_1_datewise["Active Cases"]/Unlock_1_datewise["Active Cases"].shift()),
                    mode='lines+markers',
                    name='Growth Factor of Unlock 1.0 Active Cases'))
# fig.add_trace(go.Scatter(y=list(Unlock_2_datewise["Active Cases"]/Unlock_2_datewise["Active Cases"].shift()),
#                     mode='lines+markers',
#                     name='Growth Factor of Unlock 2.0 Active Cases'))
fig.update_layout(title="Lockdownwise Growth Factor of Active Cases in India",
                 xaxis_title="Date",yaxis_title="Growth Factor",
                 legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# ### Doubling Rate in No Lockdown Period

# In[ ]:


NL_Double_rate.style.background_gradient(cmap='Reds')


# ### Doubling Rate in Lockdown 1.0

# In[ ]:


L1_Double_rate.style.background_gradient(cmap='Reds')


# ### Doubling Rate in Lockdown 2.0

# In[ ]:


L2_Double_rate.style.background_gradient(cmap='Reds')


# ### Doubling Rate in Lockdown 3.0

# In[ ]:


L3_Double_rate.style.background_gradient(cmap='Reds')


# #### All Lockdowns seems to have shown a slight effect of the Growth Rate of Active Cases implying the COVID-19 controlling practices are working well but can be improved.
# #### The Growth rate of Active Cases has slowed down during each Lockdown.
# #### Doubling Rate of Cases seems to have improved significantly during each Lockdown period which is a good sign
# #### Growth of Active Cases is showing a increasing trend in Lockdown 4.0, probably because Lockdown 4.0 is much more lenient as compared to previous Lockdown versions.

# ## Comparison of India with Neighbouring Countries 

# In[ ]:


n_countries=["Pakistan","Mainland China","Afghanistan","Nepal","Bhutan","Sri Lanka","Bangladesh","India"]
comp_data=pd.concat([pak_datewise.iloc[[-1]],china_datewise.iloc[[-1]],afg_datewise.iloc[[-1]],nepal_datewise.iloc[[-1]],
          bhutan_datewise.iloc[[-1]],lanka_datewise.iloc[[-1]],ban_datewise.iloc[[-1]],india_datewise.iloc[[-1]]])
comp_data.drop(["Days Since","WeekofYear"],1,inplace=True)
comp_data.index=n_countries
comp_data["Mortality"]=(comp_data["Deaths"]/comp_data["Confirmed"])*100
comp_data["Recovery"]=(comp_data["Recovered"]/comp_data["Confirmed"])*100
comp_data["Survival Probability"]=(1-(comp_data["Deaths"]/comp_data["Confirmed"]))*100
comp_data.sort_values(["Confirmed"],ascending=False)
comp_data.style.background_gradient(cmap='Reds').format("{:.2f}")


# In[ ]:


print("Pakistan reported it's first confirm case on: ",pak_datewise.index[0].date())
print("China reported it's first confirm case on: ",china_datewise.index[0].date())
print("Afghanistan reported it's first confirm case on: ",afg_datewise.index[0].date())
print("Nepal reported it's first confirm case on: ",nepal_datewise.index[0].date())
print("Bhutan reported it's first confirm case on: ",bhutan_datewise.index[0].date())
print("Sri Lanka reported it's first confirm case on: ",lanka_datewise.index[0].date())
print("Bangladesh reported it's first confirm case on: ",ban_datewise.index[0].date())
print("India reported it's first confirm case on: ",india_datewise.index[0].date())


# In[ ]:


print("Pakistan reported it's first death case on: ",pak_datewise[pak_datewise["Deaths"]>0].index[0].date())
print("China reported it's first death case on: ",china_datewise[china_datewise["Deaths"]>0].index[0].date())
print("Afghanistan reported it's first death case on: ",afg_datewise[afg_datewise["Deaths"]>0].index[0].date())
print("Nepal reported it's first death case on: ",nepal_datewise[nepal_datewise["Deaths"]>0].index[0].date())
print("Sri Lanka reported it's first death case on: ",lanka_datewise[lanka_datewise["Deaths"]>0].index[0].date())
print("Bangladesh reported it's first death case on: ",lanka_datewise[lanka_datewise["Deaths"]>0].index[0].date())
print("India reported it's first death case on: ",india_datewise[india_datewise["Deaths"]>0].index[0].date())


# In[ ]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=pak_datewise.index, y=np.log(pak_datewise["Confirmed"]),
                    mode='lines',name="Pakistan"))
fig.add_trace(go.Scatter(x=china_datewise.index, y=np.log(china_datewise["Confirmed"]),
                    mode='lines',name="China"))
fig.add_trace(go.Scatter(x=afg_datewise.index, y=np.log(afg_datewise["Confirmed"]),
                    mode='lines',name="Afghanistan"))
fig.add_trace(go.Scatter(x=nepal_datewise.index, y=np.log(nepal_datewise["Confirmed"]),
                    mode='lines',name="Nepal"))
fig.add_trace(go.Scatter(x=bhutan_datewise.index, y=np.log(bhutan_datewise["Confirmed"]),
                    mode='lines',name="Bhutan"))
fig.add_trace(go.Scatter(x=lanka_datewise.index, y=np.log(lanka_datewise["Confirmed"]),
                    mode='lines',name="Sri-Lanka"))
fig.add_trace(go.Scatter(x=ban_datewise.index, y=np.log(ban_datewise["Confirmed"]),
                    mode='lines',name="Bangladesh"))
fig.add_trace(go.Scatter(x=india_datewise.index, y=np.log(india_datewise["Confirmed"]),
                    mode='lines',name="India"))
fig.update_layout(title="Confirmed Cases plot for Neighbouring Countries of India (Logarithmic Scale)",
                  xaxis_title="Date",yaxis_title="Number of Cases (Log scale)",
                 legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# China is the worst affected countries among the neighbouring countries of India, as we all are well aware that COVID-19 orginated in China. The flat line after a certain period is clear indication that China has been very much successful in containing the COVID-19. 
# 
# India seems to be second badly affected among all neighbouring countries followed by Pakistan, Bangladesh and Afghanistan
# 
# Bangladesh Confirmed Cases graph is taking off really fast.
# 
# Sri Lanka's Confirmed Cases plot is showing a bit of flattening of the curve, but now it's steadly increasing a bit.

# In[ ]:


mean_mortality=[((pak_datewise["Deaths"]/pak_datewise["Confirmed"])*100).mean(),((china_datewise["Deaths"]/china_datewise["Confirmed"])*100).mean(),
               ((afg_datewise["Deaths"]/afg_datewise["Confirmed"])*100).mean(),((nepal_datewise["Deaths"]/nepal_datewise["Confirmed"])*100).mean(),
               ((bhutan_datewise["Deaths"]/bhutan_datewise["Confirmed"])*100).mean(),((lanka_datewise["Deaths"]/lanka_datewise["Confirmed"])*100).mean(),
               ((ban_datewise["Deaths"]/ban_datewise["Confirmed"])*100).mean(),((india_datewise["Deaths"]/india_datewise["Confirmed"])*100).mean()]
mean_recovery=[((pak_datewise["Recovered"]/pak_datewise["Confirmed"])*100).mean(),((china_datewise["Recovered"]/china_datewise["Confirmed"])*100).mean(),
               ((afg_datewise["Recovered"]/afg_datewise["Confirmed"])*100).mean(),((nepal_datewise["Recovered"]/nepal_datewise["Confirmed"])*100).mean(),
               ((bhutan_datewise["Recovered"]/bhutan_datewise["Confirmed"])*100).mean(),((lanka_datewise["Recovered"]/lanka_datewise["Confirmed"])*100).mean(),
               ((ban_datewise["Recovered"]/ban_datewise["Confirmed"])*100).mean(),((india_datewise["Recovered"]/india_datewise["Confirmed"])*100).mean()]

comp_data["Mean Mortality Rate"]=mean_mortality
comp_data["Mean Recovery Rate"]=mean_recovery


# In[ ]:


fig = make_subplots(rows=2, cols=1)
fig.add_trace(
    go.Bar(y=comp_data.index, x=comp_data["Mean Mortality Rate"],orientation='h'),
    row=1, col=1
)
fig.add_trace(
    go.Bar(y=comp_data.index, x=comp_data["Mean Recovery Rate"],orientation='h'),
    row=2, col=1
)
fig.update_layout(title="Mean Mortality and Recovery Rate of Neighbouring countries",
    font=dict(
        size=10,
    )
)
fig.update_layout(height=800)
fig.update_yaxes(title_text="Country Name", row=1, col=1)
fig.update_xaxes(title_text="Mortality Rate", row=1, col=1)
fig.update_yaxes(title_text="Country Name", row=2, col=1)
fig.update_xaxes(title_text="Recovery Rate", row=2, col=1)
fig.show()


# Mean Recovery Rate graph is a conclusive evidence of China has been able to flatten the curve. 
# 
# Mean Mortality Rate graph is indication that China and Bangladesh are the worst affected among the neighbours followed by Afghanistan and India.

# In[ ]:


print("Mean Mortality Rate of all Neighbouring Countries: ",comp_data["Mortality"].drop(comp_data.index[1],0).mean())
print("Median Mortality Rate of all Neighbouring Countries: ",comp_data["Mortality"].drop(comp_data.index[1],0).median())
print("Mortality Rate in India: ",comp_data.ix[1]["Mortality"])
print("Mean Mortality Rate in India: ",(india_datewise["Deaths"]/india_datewise["Confirmed"]).mean()*100)
print("Median Mortality Rate in India: ",(india_datewise["Deaths"]/india_datewise["Confirmed"]).median()*100)

fig = make_subplots(rows=3, cols=1)
fig.add_trace(
    go.Bar(y=comp_data.index, x=comp_data["Mortality"],orientation='h'),
    row=1, col=1
)
fig.add_trace(
    go.Bar(y=comp_data.index, x=comp_data["Recovery"],orientation='h'),
    row=2, col=1
)
fig.add_trace(
    go.Bar(y=comp_data.index, x=comp_data["Survival Probability"],orientation='h'),
    row=3, col=1
)
fig.update_layout(title="Mortality, Recovery and Survival Probability of Neighbouring countries",
    font=dict(
        size=10,
    )
)
fig.update_layout(height=900)
fig.update_yaxes(title_text="Country Name", row=1, col=1)
fig.update_xaxes(title_text="Mortality", row=1, col=1)
fig.update_yaxes(title_text="Country Name", row=2, col=1)
fig.update_xaxes(title_text="Recovery", row=2, col=1)
fig.update_yaxes(title_text="Country Name", row=3, col=1)
fig.update_xaxes(title_text="Survival Probability", row=3, col=1)
fig.show()


# 
# Survival Probablity of all the neighbours looks Great as it is well above 90%
# 
# Nepal and Bhutan have not reported any Death Case till date.
# Bhutan has really low number of infected people, hopefully the will be able to come out of it possibly in few days, good Recovery Rate is clear indication of it. Nepal is showing slight increase in the infection
# 
# Except China, all the neighbouring countries including India seems to face tough time containing COVID-19. Bhutan and Nepal are exception as they have comparitively low number of Confired Cases.

# Median Age of Neighbouring Countries (2020) Source: https://ourworldindata.org/age-structure
# 
# ![Age.png](attachment:Age.png)

# #### Median age, Tourist Data and International Students data has some interesting story to tell
# 
# High Median age is an clear indication, that majority of the population belong to old age group. China and India are the most densely populated countries with high median age indicating high share of population belonging to old group compared to other countries and these the countires which are badly affected by COVID-19 among all neighbours.
# 
# Tourist Data: https://worldpopulationreview.com/countries/most-visited-countries/
# 
# International Students: https://www.easyuni.com/advice/top-countries-with-most-international-students-1184/
# 
# Also Tourist data is clear indication of China and India having maximum number of foreign visitors among all neighbour countries of India also having really high number of International Students. 
# 
# #### Percentage GDP has interesting thing about Recoverd Number of Cases
# Countrywise GDP Data: https://www.worldometers.info/gdp/gdp-by-country/
# 

# #### Let's try and find Correlation among Median Age and Number of visiting Tourists and Percentage GDP of each country with different types of Cases

# In[ ]:


n_median_age=[23.5,38.7,18.6,25,28.6,34.1,27.5, 28.2]
n_tourist=[907000,59270000,0,753000,210000,2051000,303000,14570000]
n_gdp=[0.38,15.12,0.02,0.03,0.00,0.11,0.31,3.28]
area=[907132,9596961,652230,147181,38394,65610,147570,3287263]
population_density=[286.5,148,59.63,204.430,21.188,341.5,1265.036,450.419]
avg_weight=[58.976,60.555,56.935,50.476,51.142,50.421,49.591,52.943]
comp_data["Median Age"]=n_median_age
comp_data["Tourists"]=n_tourist
comp_data["GDP"]=n_gdp
comp_data["Area (square km)"]=area
comp_data["Population Density (per sq km)"]=population_density
comp_data["Average Weight"]=avg_weight
comp_data.style.background_gradient(cmap='Reds').format("{:.2f}")


# In[ ]:


req=comp_data[["Confirmed","Deaths","Recovered","Median Age","Tourists","GDP",
               "Area (square km)","Population Density (per sq km)","Average Weight"]]
plt.figure(figsize=(12,6))
mask = np.triu(np.ones_like(req.corr(), dtype=np.bool))
sns.heatmap(req.corr(),annot=True, mask=mask)


# ### Initially:
# #### When we see daily news reports on COVID-19 it's really hard to interpret what's actually happening, since the numbers are changing so rapidly but that's something expected from Exponential growth. Since almost all the pandemics tend to grow exponentially it's really hard to understand for someone from a non-mathematical or non-statistical background.
# 
# #### We are more concerned about how we are doing and where we are heading in this pandemic rather than just looking at those exponentially growing numbers. The growth won't be exponentially forever, at some point of time the curve will become flat because probably all the people on the planet are infected or we human have been able to control the disease.
# 
# #### When we are in the middle of the exponential growth it's almost impossible to tell where are we heading.
# 
# #### Here, I am trying to show how we can interpret the exponential growth which is the common trend among all the countries
# 
# ### Now:
# #### The inital correlation analysis doesn't seems to hold true with new updated data.

# In[ ]:


fig=go.Figure()
for country in n_countries:
    fig.add_trace(go.Scatter(x=grouped_country.ix[country]["log_confirmed"], y=grouped_country.ix[country]["log_active"],
                    mode='lines',name=country))
fig.update_layout(height=600,title="COVID-19 Journey of India's Neighbouring countries",
                 xaxis_title="Confirmed Cases (Logrithmic Scale)",yaxis_title="Active Cases (Logarithmic Scale)",
                 legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# China is the only country who has been able to get control over the COVID-19 pandemic, the drop in the graph clearly shows that.
# 
# Rest of the countries will follow the trajectory of China if the growth of cases is exponential, as soon as there is a drop in the graph that's clear indication that the particular country has got the control over the exponential growth of the pandemic.
# 
# Bangladesh showed that drop initially, that probably because the number of cases were low, the second wave of COVID-19 has struck Bangladesh really bad as they are catching up with China's initial trajetory.

# In[ ]:


fig=go.Figure()
for country in n_countries:
    fig.add_trace(go.Scatter(x=grouped_country.ix[country].index, y=grouped_country.ix[country]["Confirmed"].rolling(window=7).mean().diff(),
                    mode='lines',name=country))
fig.update_layout(title="7 Days Rolling Average of Daily increase of Confirmed Cases for Neighbouring Countries",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",
                 legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[ ]:


fig=go.Figure()
for country in n_countries:
    fig.add_trace(go.Scatter(x=grouped_country.ix[country].index, y=grouped_country.ix[country]["Deaths"].rolling(window=7).mean().diff(),
                    mode='lines',name=country))
fig.update_layout(title="7 Days Rolling Average of Daily increase of Death Cases for Neighbouring Countries",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",
                 legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[ ]:


fig = px.pie(comp_data, values='Confirmed', names=comp_data.index, 
             title='Proportion of Confirmed Cases in India and among Neighbouring countries ')
fig.show()


# In[ ]:


fig = px.pie(comp_data, values='Recovered', names=comp_data.index, 
             title='Proportion of Recovered Cases in India and among Neighbouring countries ')
fig.show()


# In[ ]:


fig = px.pie(comp_data, values='Deaths', names=comp_data.index, 
             title='Proportion of Death Cases in India and among Neighbouring countries ')
fig.show()


# ## Comparison of India with other countries badly affected by the Pandemic 

# In[ ]:


pd.set_option('float_format', '{:f}'.format)
country_names=["Spain","US","Italy","Iran","France","UK","Brazil","Russia","India"]
country_data=pd.concat([spain_datewise.iloc[[-1]],us_datewise.iloc[[-1]],italy_datewise.iloc[[-1]],iran_datewise.iloc[[-1]],
                        france_datewise.iloc[[-1]],uk_datewise.iloc[[-1]],brazil_datewise.iloc[[-1]],russia_datewise.iloc[[-1]],
                        india_datewise.iloc[[-1]]])
country_data=country_data.drop(["Days Since","WeekofYear"],1)
country_data["Mortality"]=(country_data["Deaths"]/country_data["Confirmed"])*100
country_data["Recovery"]=(country_data["Recovered"]/country_data["Confirmed"])*100
country_data.index=country_names
country_data.style.background_gradient(cmap='Blues').format("{:.2f}")


# In[ ]:


max_confirm_india=india_datewise["Confirmed"].iloc[-1]
print("It took",spain_datewise[(spain_datewise["Confirmed"]>0)&(spain_datewise["Confirmed"]<=max_confirm_india)].shape[0],"days in Spain to reach number of Confirmed Cases equivalent to India")
print("It took",us_datewise[(us_datewise["Confirmed"]>0)&(us_datewise["Confirmed"]<=max_confirm_india)].shape[0],"days in USA to reach number of Confirmed Cases equivalent to India")
print("It took",italy_datewise[(italy_datewise["Confirmed"]>0)&(italy_datewise["Confirmed"]<=max_confirm_india)].shape[0],"days in Italy to reach number of Confirmed Cases equivalent to India")
print("It took",iran_datewise[(iran_datewise["Confirmed"]>0)&(iran_datewise["Confirmed"]<=max_confirm_india)].shape[0],"days in Iran to reach number of Confirmed Cases equivalent to India")
print("It took",france_datewise[(france_datewise["Confirmed"]>0)&(france_datewise["Confirmed"]<=max_confirm_india)].shape[0],"days in France to reach number of Confirmed Cases equivalent to India")
print("It took",uk_datewise[(uk_datewise["Confirmed"]>0)&(uk_datewise["Confirmed"]<=max_confirm_india)].shape[0],"days in United Kingdom to reach number of Confirmed Cases equivalent to India")
print("It took",brazil_datewise[(brazil_datewise["Confirmed"]>0)&(brazil_datewise["Confirmed"]<=max_confirm_india)].shape[0],"days in Brazil to reach number of Confirmed Cases equivalent to India")
print("It took",russia_datewise[(russia_datewise["Confirmed"]>0)&(russia_datewise["Confirmed"]<=max_confirm_india)].shape[0],"days in Russia to reach number of Confirmed Cases equivalent to India")
print("It took",india_datewise[india_datewise["Confirmed"]>0].shape[0],"days in India to reach",max_confirm_india,"Confirmed Cases")

fig=go.Figure()
fig.add_trace(go.Scatter(x=spain_datewise[spain_datewise["Confirmed"]<=max_confirm_india].index, y=spain_datewise[spain_datewise["Confirmed"]<=max_confirm_india]["Confirmed"],
                    mode='lines',name="Spain"))
fig.add_trace(go.Scatter(x=us_datewise[us_datewise["Confirmed"]<=max_confirm_india].index, y=us_datewise[us_datewise["Confirmed"]<=max_confirm_india]["Confirmed"],
                    mode='lines',name="USA"))
fig.add_trace(go.Scatter(x=italy_datewise[italy_datewise["Confirmed"]<=max_confirm_india].index, y=italy_datewise[italy_datewise["Confirmed"]<=max_confirm_india]["Confirmed"],
                    mode='lines',name="Italy"))
fig.add_trace(go.Scatter(x=iran_datewise[iran_datewise["Confirmed"]<=max_confirm_india].index, y=iran_datewise[iran_datewise["Confirmed"]<=max_confirm_india]["Confirmed"],
                    mode='lines',name="Iran"))
fig.add_trace(go.Scatter(x=france_datewise[france_datewise["Confirmed"]<=max_confirm_india].index, y=france_datewise[france_datewise["Confirmed"]<=max_confirm_india]["Confirmed"],
                    mode='lines',name="France"))
fig.add_trace(go.Scatter(x=uk_datewise[uk_datewise["Confirmed"]<=max_confirm_india].index, y=uk_datewise[uk_datewise["Confirmed"]<=max_confirm_india]["Confirmed"],
                    mode='lines',name="United Kingdom"))
fig.add_trace(go.Scatter(x=brazil_datewise[brazil_datewise["Confirmed"]<=max_confirm_india].index, y=brazil_datewise[brazil_datewise["Confirmed"]<=max_confirm_india]["Confirmed"],
                    mode='lines',name="Brazil"))
fig.add_trace(go.Scatter(x=russia_datewise[russia_datewise["Confirmed"]<=max_confirm_india].index, y=russia_datewise[russia_datewise["Confirmed"]<=max_confirm_india]["Confirmed"],
                    mode='lines',name="Russia"))
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Confirmed"],
                    mode='lines',name="India"))
fig.update_layout(title="Growth of Confirmed Cases with respect to India",
                 xaxis_title="Date",yaxis_title="Number of Confirmed Cases",
                 legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[ ]:


max_deaths_india=india_datewise["Deaths"].iloc[-1]
print("It took",spain_datewise[(spain_datewise["Deaths"]<=max_deaths_india)].shape[0],"days in Spain to reach number of Deaths Cases equivalent to India")
print("It took",us_datewise[(us_datewise["Deaths"]<=max_deaths_india)].shape[0],"days in USA to reach number of Deaths Cases equivalent to India")
print("It took",italy_datewise[(italy_datewise["Deaths"]<=max_deaths_india)].shape[0],"days in Italy to reach number of Deaths Cases equivalent to India")
print("It took",iran_datewise[(iran_datewise["Deaths"]<=max_deaths_india)].shape[0],"days in Iran to reach number of Deaths Cases equivalent to India")
print("It took",france_datewise[(france_datewise["Deaths"]<=max_deaths_india)].shape[0],"days in France to reach number of Deaths Cases equivalent to India")
print("It took",uk_datewise[(uk_datewise["Deaths"]<=max_deaths_india)].shape[0],"days in UK to reach number of Deaths Cases equivalent to India")
print("It took",brazil_datewise[(brazil_datewise["Deaths"]<=max_deaths_india)].shape[0],"days in Brazil to reach number of Deaths Cases equivalent to India")
print("It took",russia_datewise[(russia_datewise["Deaths"]<=max_deaths_india)].shape[0],"days in Russia to reach number of Deaths Cases equivalent to India")
print("It took",india_datewise.shape[0],"days in India to reach",max_deaths_india,"Deaths Cases")

fig=go.Figure()
fig.add_trace(go.Scatter(x=spain_datewise[spain_datewise["Deaths"]<=max_deaths_india].index, y=spain_datewise[spain_datewise["Deaths"]<=max_deaths_india]["Deaths"],
                    mode='lines',name="Spain"))
fig.add_trace(go.Scatter(x=us_datewise[us_datewise["Deaths"]<=max_deaths_india].index, y=us_datewise[us_datewise["Deaths"]<=max_deaths_india]["Deaths"],
                    mode='lines',name="USA"))
fig.add_trace(go.Scatter(x=italy_datewise[italy_datewise["Deaths"]<=max_deaths_india].index, y=italy_datewise[italy_datewise["Deaths"]<=max_deaths_india]["Deaths"],
                    mode='lines',name="Italy"))
fig.add_trace(go.Scatter(x=iran_datewise[iran_datewise["Deaths"]<=max_deaths_india].index, y=iran_datewise[iran_datewise["Deaths"]<=max_deaths_india]["Deaths"],
                    mode='lines',name="Iran"))
fig.add_trace(go.Scatter(x=france_datewise[france_datewise["Deaths"]<=max_deaths_india].index, y=france_datewise[france_datewise["Deaths"]<=max_deaths_india]["Deaths"],
                    mode='lines',name="France"))
fig.add_trace(go.Scatter(x=uk_datewise[uk_datewise["Deaths"]<=max_deaths_india].index, y=uk_datewise[uk_datewise["Deaths"]<=max_deaths_india]["Deaths"],
                    mode='lines',name="United Kingdom"))
fig.add_trace(go.Scatter(x=brazil_datewise[brazil_datewise["Deaths"]<=max_deaths_india].index, y=brazil_datewise[brazil_datewise["Deaths"]<=max_deaths_india]["Deaths"],
                    mode='lines',name="Brazil"))
fig.add_trace(go.Scatter(x=russia_datewise[russia_datewise["Deaths"]<=max_deaths_india].index, y=russia_datewise[russia_datewise["Deaths"]<=max_deaths_india]["Deaths"],
                    mode='lines',name="Russia"))
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Deaths"],
                    mode='lines',name="India"))
fig.update_layout(title="Growth of Death Cases with respect to India",
                 xaxis_title="Date",yaxis_title="Number of Death Cases",
                 legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[ ]:


max_recovered_india=india_datewise["Recovered"].iloc[-1]
print("It took",spain_datewise[(spain_datewise["Recovered"]<=max_recovered_india)].shape[0],"days in Spain to reach number of Recovered Cases equivalent to India")
print("It took",us_datewise[(us_datewise["Recovered"]<=max_recovered_india)].shape[0],"days in USA to reach number of Recovered Cases equivalent to India")
print("It took",italy_datewise[(italy_datewise["Recovered"]<=max_recovered_india)].shape[0],"days in Italy to reach number of Recovered Cases equivalent to India")
print("It took",iran_datewise[(iran_datewise["Recovered"]<=max_recovered_india)].shape[0],"days in Iran to reach number of Recovered Cases equivalent to India")
print("It took",france_datewise[(france_datewise["Recovered"]<=max_recovered_india)].shape[0],"days in France to reach number of Recovered Cases equivalent to India")
print("It took",uk_datewise[(uk_datewise["Recovered"]<=max_recovered_india)].shape[0],"days in UK to reach number of Recovered Cases equivalent to India")
print("It took",brazil_datewise[(brazil_datewise["Recovered"]<=max_recovered_india)].shape[0],"days in Brazil to reach number of Recovered Cases equivalent to India")
print("It took",russia_datewise[(russia_datewise["Recovered"]<=max_recovered_india)].shape[0],"days in Russia to reach number of Recovered Cases equivalent to India")
print("It took",india_datewise.shape[0],"days in India to reach",max_recovered_india,"Recovered Cases")

fig=go.Figure()
fig.add_trace(go.Scatter(x=spain_datewise[spain_datewise["Recovered"]<=max_recovered_india].index, y=spain_datewise[spain_datewise["Recovered"]<=max_recovered_india]["Recovered"],
                    mode='lines',name="Spain"))
fig.add_trace(go.Scatter(x=us_datewise[us_datewise["Recovered"]<=max_recovered_india].index, y=us_datewise[us_datewise["Recovered"]<=max_recovered_india]["Recovered"],
                    mode='lines',name="USA"))
fig.add_trace(go.Scatter(x=italy_datewise[italy_datewise["Recovered"]<=max_recovered_india].index, y=italy_datewise[italy_datewise["Recovered"]<=max_recovered_india]["Recovered"],
                    mode='lines',name="Italy"))
fig.add_trace(go.Scatter(x=iran_datewise[iran_datewise["Recovered"]<=max_recovered_india].index, y=iran_datewise[iran_datewise["Recovered"]<=max_recovered_india]["Recovered"],
                    mode='lines',name="Iran"))
fig.add_trace(go.Scatter(x=france_datewise[france_datewise["Recovered"]<=max_recovered_india].index, y=france_datewise[france_datewise["Recovered"]<=max_recovered_india]["Recovered"],
                    mode='lines',name="France"))
fig.add_trace(go.Scatter(x=uk_datewise[uk_datewise["Recovered"]<=max_recovered_india].index, y=uk_datewise[uk_datewise["Recovered"]<=max_recovered_india]["Recovered"],
                    mode='lines',name="United Kingdom"))
fig.add_trace(go.Scatter(x=brazil_datewise[brazil_datewise["Recovered"]<=max_recovered_india].index, y=brazil_datewise[brazil_datewise["Recovered"]<=max_recovered_india]["Recovered"],
                    mode='lines',name="Brazil"))
fig.add_trace(go.Scatter(x=russia_datewise[russia_datewise["Recovered"]<=max_recovered_india].index, y=russia_datewise[russia_datewise["Recovered"]<=max_recovered_india]["Recovered"],
                    mode='lines',name="Russia"))
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Recovered"],
                    mode='lines',name="India"))
fig.update_layout(title="Growth of Recovered Cases with respect to India",
                 xaxis_title="Date",yaxis_title="Number of Recovered Cases",
                 legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[ ]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=spain_datewise[spain_datewise["Confirmed"]<=max_confirm_india].index, y=spain_datewise[spain_datewise["Confirmed"]<=max_confirm_india]["Confirmed"].diff().fillna(0),
                    mode='lines',name="Spain"))
fig.add_trace(go.Scatter(x=us_datewise[us_datewise["Confirmed"]<=max_confirm_india].index, y=us_datewise[us_datewise["Confirmed"]<=max_confirm_india]["Confirmed"].diff().fillna(0),
                    mode='lines',name="USA"))
fig.add_trace(go.Scatter(x=italy_datewise[italy_datewise["Confirmed"]<=max_confirm_india].index, y=italy_datewise[italy_datewise["Confirmed"]<=max_confirm_india]["Confirmed"].diff().fillna(0),
                    mode='lines',name="Italy"))
fig.add_trace(go.Scatter(x=iran_datewise[iran_datewise["Confirmed"]<=max_confirm_india].index, y=iran_datewise[iran_datewise["Confirmed"]<=max_confirm_india]["Confirmed"].diff().fillna(0),
                    mode='lines',name="Iran"))
fig.add_trace(go.Scatter(x=france_datewise[france_datewise["Confirmed"]<=max_confirm_india].index, y=france_datewise[france_datewise["Confirmed"]<=max_confirm_india]["Confirmed"].diff().fillna(0),
                    mode='lines',name="France"))
fig.add_trace(go.Scatter(x=uk_datewise[uk_datewise["Confirmed"]<=max_confirm_india].index, y=uk_datewise[uk_datewise["Confirmed"]<=max_confirm_india]["Confirmed"].diff().fillna(0),
                    mode='lines',name="United Kingdom"))
fig.add_trace(go.Scatter(x=brazil_datewise[brazil_datewise["Confirmed"]<=max_confirm_india].index, y=brazil_datewise[brazil_datewise["Confirmed"]<=max_confirm_india]["Confirmed"].diff().fillna(0),
                    mode='lines',name="Brazil"))
fig.add_trace(go.Scatter(x=russia_datewise[russia_datewise["Confirmed"]<=max_confirm_india].index, y=russia_datewise[russia_datewise["Confirmed"]<=max_confirm_india]["Confirmed"].diff().fillna(0),
                    mode='lines',name="Russia"))
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Confirmed"].diff().fillna(0),
                    mode='lines',name="India"))
fig.update_layout(title="Daily Increase in Number of Confirmed Cases",
                 xaxis_title="Date",yaxis_title="Number of Confirmed Cases",
                 legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[ ]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=spain_datewise[spain_datewise["Deaths"]<=max_deaths_india].index, y=spain_datewise[spain_datewise["Deaths"]<=max_deaths_india]["Deaths"].diff().fillna(0),
                    mode='lines',name="Spain"))
fig.add_trace(go.Scatter(x=us_datewise[us_datewise["Deaths"]<=max_deaths_india].index, y=us_datewise[us_datewise["Deaths"]<=max_deaths_india]["Deaths"].diff().fillna(0),
                    mode='lines',name="USA"))
fig.add_trace(go.Scatter(x=italy_datewise[italy_datewise["Deaths"]<=max_deaths_india].index, y=italy_datewise[italy_datewise["Deaths"]<=max_deaths_india]["Deaths"].diff().fillna(0),
                    mode='lines',name="Italy"))
fig.add_trace(go.Scatter(x=iran_datewise[iran_datewise["Deaths"]<=max_deaths_india].index, y=iran_datewise[iran_datewise["Deaths"]<=max_deaths_india]["Deaths"].diff().fillna(0),
                    mode='lines',name="Iran"))
fig.add_trace(go.Scatter(x=france_datewise[france_datewise["Deaths"]<=max_deaths_india].index, y=france_datewise[france_datewise["Deaths"]<=max_deaths_india]["Deaths"].diff().fillna(0),
                    mode='lines',name="France"))
fig.add_trace(go.Scatter(x=uk_datewise[uk_datewise["Deaths"]<=max_deaths_india].index, y=uk_datewise[uk_datewise["Deaths"]<=max_deaths_india]["Deaths"].diff().fillna(0),
                    mode='lines',name="United Kingdom"))
fig.add_trace(go.Scatter(x=brazil_datewise[brazil_datewise["Deaths"]<=max_deaths_india].index, y=brazil_datewise[brazil_datewise["Deaths"]<=max_deaths_india]["Deaths"].diff().fillna(0),
                    mode='lines',name="Brazil"))
fig.add_trace(go.Scatter(x=russia_datewise[russia_datewise["Deaths"]<=max_deaths_india].index, y=russia_datewise[russia_datewise["Deaths"]<=max_deaths_india]["Deaths"].diff().fillna(0),
                    mode='lines',name="Russia"))
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Deaths"].diff().fillna(0),
                    mode='lines',name="India"))
fig.update_layout(title="Daily Increase in Number of Death Cases",
                 xaxis_title="Date",yaxis_title="Number of Death Cases",
                 legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[ ]:


week_num_spain=[]
spain_weekwise_confirmed=[]
spain_weekwise_recovered=[]
spain_weekwise_deaths=[]
w=1
for i in list(spain_datewise["WeekofYear"].unique()):
    spain_weekwise_confirmed.append(spain_datewise[spain_datewise["WeekofYear"]==i]["Confirmed"].iloc[-1])
    spain_weekwise_recovered.append(spain_datewise[spain_datewise["WeekofYear"]==i]["Recovered"].iloc[-1])
    spain_weekwise_deaths.append(spain_datewise[spain_datewise["WeekofYear"]==i]["Deaths"].iloc[-1])
    week_num_spain.append(w)
    w=w+1

week_num_us=[]
us_weekwise_confirmed=[]
us_weekwise_recovered=[]
us_weekwise_deaths=[]
w=1
for i in list(us_datewise["WeekofYear"].unique()):
    us_weekwise_confirmed.append(us_datewise[us_datewise["WeekofYear"]==i]["Confirmed"].iloc[-1])
    us_weekwise_recovered.append(us_datewise[us_datewise["WeekofYear"]==i]["Recovered"].iloc[-1])
    us_weekwise_deaths.append(us_datewise[us_datewise["WeekofYear"]==i]["Deaths"].iloc[-1])
    week_num_us.append(w)
    w=w+1

week_num_italy=[]
italy_weekwise_confirmed=[]
italy_weekwise_recovered=[]
italy_weekwise_deaths=[]
w=1
for i in list(italy_datewise["WeekofYear"].unique()):
    italy_weekwise_confirmed.append(italy_datewise[italy_datewise["WeekofYear"]==i]["Confirmed"].iloc[-1])
    italy_weekwise_recovered.append(italy_datewise[italy_datewise["WeekofYear"]==i]["Recovered"].iloc[-1])
    italy_weekwise_deaths.append(italy_datewise[italy_datewise["WeekofYear"]==i]["Deaths"].iloc[-1])
    week_num_italy.append(w)
    w=w+1
    
week_num_iran=[]
iran_weekwise_confirmed=[]
iran_weekwise_recovered=[]
iran_weekwise_deaths=[]
w=1
for i in list(iran_datewise["WeekofYear"].unique()):
    iran_weekwise_confirmed.append(iran_datewise[iran_datewise["WeekofYear"]==i]["Confirmed"].iloc[-1])
    iran_weekwise_recovered.append(iran_datewise[iran_datewise["WeekofYear"]==i]["Recovered"].iloc[-1])
    iran_weekwise_deaths.append(iran_datewise[iran_datewise["WeekofYear"]==i]["Deaths"].iloc[-1])
    week_num_iran.append(w)
    w=w+1
    
week_num_france=[]
france_weekwise_confirmed=[]
france_weekwise_recovered=[]
france_weekwise_deaths=[]
w=1
for i in list(france_datewise["WeekofYear"].unique()):
    france_weekwise_confirmed.append(france_datewise[france_datewise["WeekofYear"]==i]["Confirmed"].iloc[-1])
    france_weekwise_recovered.append(france_datewise[france_datewise["WeekofYear"]==i]["Recovered"].iloc[-1])
    france_weekwise_deaths.append(france_datewise[france_datewise["WeekofYear"]==i]["Deaths"].iloc[-1])
    week_num_france.append(w)
    w=w+1
    
week_num_uk=[]
uk_weekwise_confirmed=[]
uk_weekwise_recovered=[]
uk_weekwise_deaths=[]
w=1
for i in list(uk_datewise["WeekofYear"].unique()):
    uk_weekwise_confirmed.append(uk_datewise[uk_datewise["WeekofYear"]==i]["Confirmed"].iloc[-1])
    uk_weekwise_recovered.append(uk_datewise[uk_datewise["WeekofYear"]==i]["Recovered"].iloc[-1])
    uk_weekwise_deaths.append(uk_datewise[uk_datewise["WeekofYear"]==i]["Deaths"].iloc[-1])
    week_num_uk.append(w)
    w=w+1
    
week_num_br=[]
br_weekwise_confirmed=[]
br_weekwise_recovered=[]
br_weekwise_deaths=[]
w=1
for i in list(brazil_datewise["WeekofYear"].unique()):
    br_weekwise_confirmed.append(brazil_datewise[brazil_datewise["WeekofYear"]==i]["Confirmed"].iloc[-1])
    br_weekwise_recovered.append(brazil_datewise[brazil_datewise["WeekofYear"]==i]["Recovered"].iloc[-1])
    br_weekwise_deaths.append(brazil_datewise[brazil_datewise["WeekofYear"]==i]["Deaths"].iloc[-1])
    week_num_br.append(w)
    w=w+1
    
week_num_rus=[]
rus_weekwise_confirmed=[]
rus_weekwise_recovered=[]
rus_weekwise_deaths=[]
w=1
for i in list(russia_datewise["WeekofYear"].unique()):
    rus_weekwise_confirmed.append(russia_datewise[russia_datewise["WeekofYear"]==i]["Confirmed"].iloc[-1])
    rus_weekwise_recovered.append(russia_datewise[russia_datewise["WeekofYear"]==i]["Recovered"].iloc[-1])
    rus_weekwise_deaths.append(russia_datewise[russia_datewise["WeekofYear"]==i]["Deaths"].iloc[-1])
    week_num_rus.append(w)
    w=w+1


# In[ ]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=week_num_spain, y=spain_weekwise_confirmed,
                    mode='lines+markers',name="Spain"))
fig.add_trace(go.Scatter(x=week_num_us, y=us_weekwise_confirmed,
                    mode='lines+markers',name="USA"))
fig.add_trace(go.Scatter(x=week_num_italy, y=italy_weekwise_confirmed,
                    mode='lines+markers',name="Italy"))
fig.add_trace(go.Scatter(x=week_num_iran, y=iran_weekwise_confirmed,
                    mode='lines+markers',name="Iran"))
fig.add_trace(go.Scatter(x=week_num_france, y=france_weekwise_confirmed,
                    mode='lines+markers',name="France"))
fig.add_trace(go.Scatter(x=week_num_uk, y=uk_weekwise_confirmed,
                    mode='lines+markers',name="United Kingdom"))
fig.add_trace(go.Scatter(x=week_num_br, y=br_weekwise_confirmed,
                    mode='lines+markers',name="Brazil"))
fig.add_trace(go.Scatter(x=week_num_rus, y=rus_weekwise_confirmed,
                    mode='lines+markers',name="Russia"))
fig.add_trace(go.Scatter(x=week_num_india, y=india_weekwise_confirmed,
                    mode='lines+markers',name="India"))
fig.update_layout(title="Weekly Growth of Confirmed Cases",
                 xaxis_title="Date",yaxis_title="Number of Confirmed Cases",
                 legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[ ]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=week_num_spain, y=spain_weekwise_deaths,
                    mode='lines+markers',name="Spain"))
fig.add_trace(go.Scatter(x=week_num_us, y=us_weekwise_deaths,
                    mode='lines+markers',name="USA"))
fig.add_trace(go.Scatter(x=week_num_italy, y=italy_weekwise_deaths,
                    mode='lines+markers',name="Italy"))
fig.add_trace(go.Scatter(x=week_num_iran, y=iran_weekwise_deaths,
                    mode='lines+markers',name="Iran"))
fig.add_trace(go.Scatter(x=week_num_france, y=france_weekwise_deaths,
                    mode='lines+markers',name="France"))
fig.add_trace(go.Scatter(x=week_num_uk, y=uk_weekwise_deaths,
                    mode='lines+markers',name="United Kingdom"))
fig.add_trace(go.Scatter(x=week_num_br, y=br_weekwise_deaths,
                    mode='lines+markers',name="Brazil"))
fig.add_trace(go.Scatter(x=week_num_rus, y=rus_weekwise_deaths,
                    mode='lines+markers',name="Russia"))
fig.add_trace(go.Scatter(x=week_num_india, y=india_weekwise_deaths,
                    mode='lines+markers',name="India"))
fig.update_layout(title="Weekly Growth of Death Cases",
                 xaxis_title="Date",yaxis_title="Number of Death Cases",
                 legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[ ]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=week_num_spain, y=pd.Series(spain_weekwise_confirmed).diff().fillna(0),
                    mode='lines+markers',name="Spain"))
fig.add_trace(go.Scatter(x=week_num_us, y=pd.Series(us_weekwise_confirmed).diff().fillna(0),
                     mode='lines+markers',name="USA"))
fig.add_trace(go.Scatter(x=week_num_italy, y=pd.Series(italy_weekwise_confirmed).diff().fillna(0),
                    mode='lines+markers',name="Italy"))
fig.add_trace(go.Scatter(x=week_num_iran, y=pd.Series(iran_weekwise_confirmed).diff().fillna(0),
                    mode='lines+markers',name="Iran"))
fig.add_trace(go.Scatter(x=week_num_france, y=pd.Series(france_weekwise_confirmed).diff().fillna(0),
                    mode='lines+markers',name="France"))
fig.add_trace(go.Scatter(x=week_num_uk, y=pd.Series(uk_weekwise_confirmed).diff().fillna(0),
                     mode='lines+markers',name="United Kingdom"))
fig.add_trace(go.Scatter(x=week_num_br, y=pd.Series(br_weekwise_confirmed).diff().fillna(0),
                     mode='lines+markers',name="Brazil"))
fig.add_trace(go.Scatter(x=week_num_rus, y=pd.Series(rus_weekwise_confirmed).diff().fillna(0),
                     mode='lines+markers',name="Russia"))
fig.add_trace(go.Scatter(x=week_num_india, y=pd.Series(india_weekwise_confirmed).diff().fillna(0),
                     mode='lines+markers',name="India"))
fig.update_layout(title="Weekly Growth of Death Cases",
                 xaxis_title="Date",yaxis_title="Number of Death Cases",
                 legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# #### Please note the last data point assoicated with every lineplot of country indicates information about the week that has just started.

# ### Let's perform different feature Analysis for worst affected countries as well
# ![Screenshot%20%28236%29.png](attachment:Screenshot%20%28236%29.png)

# In[ ]:


ac_median_age=[45.5,38.3,47.9,32.4,42,40.8,33.5,39.6,28.2]
ac_tourists=[75315000,76407000,52372000,4942000,82700000,35814000,6547000,24571000,14570000]
ac_weight=[70.556,81.928,69.205,67.608,66.782,75.795,66.093,71.418,52.943]
ac_gdp=[1.62,24.08,2.40,0.56,3.19,3.26,2.54,1.95,3.28]
ac_area=[505992,9833517,301339,1648195,640679,242495,8515767,17098246,3287263]
ac_pd=[93,34,200,51,123,280,25.43,8.58,414]
country_data["Median Age"]=ac_median_age
country_data["Tourists"]=ac_tourists
country_data["GDP"]=ac_gdp
country_data["Area (square km)"]=ac_area
country_data["Average Weight"]=ac_weight
country_data["Population Density (per sq km)"]=ac_pd
country_data.sort_values(["Confirmed"],ascending=False)
country_data.style.background_gradient(cmap='Blues').format("{:.2f}")


# In[ ]:


new_req=country_data[["Confirmed","Deaths","Recovered","Median Age","Tourists","Average Weight",
                     "GDP","Area (square km)","Population Density (per sq km)"]]
plt.figure(figsize=(10,5))
mask = np.triu(np.ones_like(new_req.corr(), dtype=np.bool))
sns.heatmap(new_req.corr(),annot=True, mask=mask)


# **Median Age's correlation with number of Confirmed Cases is positive and prominently high in some countries but not in some, so it's not evident that age has anything to do with people who will be tested positive with COVID-19 all are equally vunerable. But Median age has very high correlation with number of Death Cases in all countries, implying Death Rate is high among old age group.**
# 
# **Number of Tourists has very high corrleation with both with Number of Confirmed Cases and Death cases and that trend is pretty evident in almost all countries. Implying one probable reason for the spread of COVID-19 is Tourism.**
# 
# **Weight plays an important role in all three types of cases, as it decied whether a particular person will be infected by COVID-19 or not, his/her death and will he/she be Recovered**

# In[ ]:


fig=go.Figure()
for country in country_names:
    fig.add_trace(go.Scatter(x=grouped_country.ix[country]["log_confirmed"], y=grouped_country.ix[country]["log_active"],
                    mode='lines',name=country))
fig.update_layout(height=600,title="COVID-19 Journey of some worst affected countries and India",
                 xaxis_title="Confirmed Cases (Logrithmic Scale)",yaxis_title="Active Cases (Logarithmic Scale)",
                 legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# Most of the countries will follow the trejactory of US, which is **Uncontrolled Exponential Growth**
# 
# Iran has started to get control over COVID-19 which is evident from their tajectory.
# 
# Countries like Italy and Spain who were worst affected by the pandemic initally have started showing the a slight dip in the trajectory which is positive sign.

# In[ ]:


fig=go.Figure()
for country in country_names:
    fig.add_trace(go.Scatter(x=grouped_country.ix[country].index, y=grouped_country.ix[country]["Confirmed"].rolling(window=7).mean().diff(),
                    mode='lines',name=country))
fig.update_layout(height=600,title="7 Days Rolling Average of Daily increase of Confirmed Cases for Worst affected countries and India",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",
                 legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[ ]:


fig = px.pie(country_data, values='Confirmed', names=country_data.index, 
             title='Proportion of Confirmed Cases in India and among Worst affected countries')
fig.show()


# In[ ]:


fig = px.pie(country_data, values='Recovered', names=country_data.index, 
             title='Proportion of Recovered Cases in India and among Worst affected countries')
fig.show()


# In[ ]:


fig = px.pie(country_data, values='Deaths', names=country_data.index, 
             title='Proportion of Death Cases in India and among Worst affected countries')
fig.show()


# #### Feature Importance using K-Best Feature selection method

# In[ ]:


model_data=comp_data.drop(["Survival Probability","Mean Mortality Rate","Mean Recovery Rate"],1)
model_data=pd.concat([model_data,country_data])


# In[ ]:


X=model_data.drop(["Confirmed","Recovered","Deaths","Recovery","Mortality"],1)
y1=model_data["Confirmed"]
y2=model_data["Recovered"]
y3=model_data["Deaths"]


# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
k_best_confirmed=SelectKBest(score_func=f_regression,k='all')
k_best_confirmed.fit(X,y1)
k_best_recovered=SelectKBest(score_func=f_regression,k='all')
k_best_recovered.fit(X,y2)
k_best_deaths=SelectKBest(score_func=f_regression,k='all')
k_best_deaths.fit(X,y3)


# In[ ]:


fig = go.Figure(data=[go.Bar(name='Feature Importance for Confirmed Cases', x=k_best_confirmed.scores_, y=pd.Series(list(X)),orientation='h'),
    go.Bar(name='Feature Importance for Recovered Cases', x=k_best_recovered.scores_, y=pd.Series(list(X)),orientation='h'),
    go.Bar(name='Feature Importance for Death Cases', x=k_best_deaths.scores_, y=pd.Series(list(X)),orientation='h')])
fig.update_layout(barmode='group',width=900,legend=dict(x=0,y=-0.5,traceorder="normal"),
                 title="Feature Importance using Select K-Best")
fig.show()


# ## Machine Learning Predictions

# ### Polynomial Regression

# In[ ]:


train_ml=india_datewise.iloc[:int(india_datewise.shape[0]*0.95)]
valid_ml=india_datewise.iloc[int(india_datewise.shape[0]*0.95):]
model_scores=[]


# In[ ]:


poly = PolynomialFeatures(degree = 6) 


# In[ ]:


train_poly=poly.fit_transform(np.array(train_ml["Days Since"]).reshape(-1,1))
valid_poly=poly.fit_transform(np.array(valid_ml["Days Since"]).reshape(-1,1))
y=train_ml["Confirmed"]


# In[ ]:


linreg=LinearRegression(normalize=True)
linreg.fit(train_poly,y)


# In[ ]:


prediction_poly=linreg.predict(valid_poly)
rmse_poly=np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_poly))
model_scores.append(rmse_poly)
print("Root Mean Squared Error for Polynomial Regression: ",rmse_poly)                          


# In[ ]:


comp_data=poly.fit_transform(np.array(india_datewise["Days Since"]).reshape(-1,1))
plt.figure(figsize=(11,6))
predictions_poly=linreg.predict(comp_data)
fig=go.Figure()
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Confirmed"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=india_datewise.index, y=predictions_poly,
                    mode='lines',name="Polynomial Regression Best Fit",
                    line=dict(color='black', dash='dot')))
fig.update_layout(title="Confirmed Cases Polynomial Regression Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",
                 legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[ ]:


new_date=[]
new_prediction_poly=[]
for i in range(1,18):
    new_date.append(india_datewise.index[-1]+timedelta(days=i))
    new_date_poly=poly.fit_transform(np.array(india_datewise["Days Since"].max()+i).reshape(-1,1))
    new_prediction_poly.append(linreg.predict(new_date_poly)[0])


# In[ ]:


model_predictions=pd.DataFrame(zip(new_date,new_prediction_poly),columns=["Date","Polynomial Regression Prediction"])
model_predictions.head()


# ### Support Vector Machine Regressor

# In[ ]:


train_ml=india_datewise.iloc[:int(india_datewise.shape[0]*0.95)]
valid_ml=india_datewise.iloc[int(india_datewise.shape[0]*0.95):]


# In[ ]:


svm=SVR(C=0.01,degree=7,kernel='poly')


# In[ ]:


svm.fit(np.array(train_ml["Days Since"]).reshape(-1,1),train_ml["Confirmed"])


# In[ ]:


prediction_svm=svm.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))
rmse_svm=np.sqrt(mean_squared_error(prediction_svm,valid_ml["Confirmed"]))
model_scores.append(rmse_svm)
print("Root Mean Square Error for SVR Model: ",rmse_svm)


# In[ ]:


plt.figure(figsize=(11,6))
predictions=svm.predict(np.array(india_datewise["Days Since"]).reshape(-1,1))
fig=go.Figure()
fig.add_trace(go.Scatter(x=india_datewise.index, y=india_datewise["Confirmed"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=india_datewise.index, y=predictions,
                    mode='lines',name="Support Vector Machine Best fit Kernel",
                    line=dict(color='black', dash='dot')))
fig.update_layout(title="Confirmed Cases Support Vectore Machine Regressor Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[ ]:


new_date=[]
new_prediction_svm=[]
for i in range(1,18):
    new_date.append(india_datewise.index[-1]+timedelta(days=i))
    new_prediction_svm.append(svm.predict(np.array(india_datewise["Days Since"].max()+i).reshape(-1,1))[0])


# In[ ]:


model_predictions["SVM Prediction"]=new_prediction_svm
model_predictions.head()


# Support Vectore Machine Model doesn't seem to be performing great, as predictions are either understimated or overestimated.

# ## Time Series Forecasting Models

# #### Holt's Linear Model

# In[ ]:


model_train=india_datewise.iloc[:int(india_datewise.shape[0]*0.95)]
valid=india_datewise.iloc[int(india_datewise.shape[0]*0.95):]
y_pred=valid.copy()


# In[ ]:


holt=Holt(np.asarray(model_train["Confirmed"])).fit(smoothing_level=0.3, smoothing_slope=1.2)


# In[ ]:


y_pred["Holt"]=holt.forecast(len(valid))
rmse_holt_linear=np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt"]))
model_scores.append(rmse_holt_linear)
print("Root Mean Square Error Holt's Linear Model: ",rmse_holt_linear)


# In[ ]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Confirmed"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=valid.index, y=valid["Confirmed"],
                    mode='lines+markers',name="Validation Data for Confirmed Cases",))
fig.add_trace(go.Scatter(x=valid.index, y=y_pred["Holt"],
                    mode='lines+markers',name="Prediction of Confirmed Cases",))
fig.update_layout(title="Confirmed Cases Holt's Linear Model Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[ ]:


holt_new_prediction=[]
for i in range(1,18):
    holt_new_prediction.append(holt.forecast((len(valid)+i))[-1])

model_predictions["Holt's Linear Model Prediction"]=holt_new_prediction
model_predictions.head()


# #### Holt's Winter Model

# In[ ]:


model_train=india_datewise.iloc[:int(india_datewise.shape[0]*0.95)]
valid=india_datewise.iloc[int(india_datewise.shape[0]*0.95):]
y_pred=valid.copy()


# In[ ]:


es=ExponentialSmoothing(np.asarray(model_train['Confirmed']),seasonal_periods=15, trend='mul', seasonal='mul').fit()


# In[ ]:


y_pred["Holt's Winter Model"]=es.forecast(len(valid))
rmse_holt_winter=np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt's Winter Model"]))
model_scores.append(rmse_holt_winter)
print("Root Mean Square Error for Holt's Winter Model: ",rmse_holt_winter)


# In[ ]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Confirmed"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=valid.index, y=valid["Confirmed"],
                    mode='lines+markers',name="Validation Data for Confirmed Cases",))
fig.add_trace(go.Scatter(x=valid.index, y=y_pred["Holt\'s Winter Model"],
                    mode='lines+markers',name="Prediction of Confirmed Cases",))
fig.update_layout(title="Confirmed Cases Holt's Winter Model Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[ ]:


holt_winter_new_prediction=[]
for i in range(1,18):
    holt_winter_new_prediction.append(es.forecast((len(valid)+i))[-1])
model_predictions["Holt's Winter Model Prediction"]=holt_winter_new_prediction
model_predictions.head()


# ### AR Model (using AUTO ARIMA)

# In[ ]:


model_train=india_datewise.iloc[:int(india_datewise.shape[0]*0.95)]
valid=india_datewise.iloc[int(india_datewise.shape[0]*0.95):]
y_pred=valid.copy()


# In[ ]:


model_ar= auto_arima(model_train["Confirmed"],trace=True, error_action='ignore', start_p=0,start_q=0,max_p=3,max_q=0,
                   suppress_warnings=True,stepwise=False,seasonal=False)
model_ar.fit(model_train["Confirmed"])


# In[ ]:


prediction_ar=model_ar.predict(len(valid))
y_pred["AR Model Prediction"]=prediction_ar


# In[ ]:


model_scores.append(np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["AR Model Prediction"])))
print("Root Mean Square Error for AR Model: ",np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["AR Model Prediction"])))


# In[ ]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Confirmed"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=valid.index, y=valid["Confirmed"],
                    mode='lines+markers',name="Validation Data for Confirmed Cases",))
fig.add_trace(go.Scatter(x=valid.index, y=y_pred["AR Model Prediction"],
                    mode='lines+markers',name="Prediction of Confirmed Cases",))
fig.update_layout(title="Confirmed Cases AR Model Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[ ]:


AR_model_new_prediction=[]
for i in range(1,18):
    AR_model_new_prediction.append(model_ar.predict(len(valid)+i)[-1])
model_predictions["AR Model Prediction"]=AR_model_new_prediction
model_predictions.head()


# ### MA Model (using AUTO ARIMA)

# In[ ]:


model_train=india_datewise.iloc[:int(india_datewise.shape[0]*0.95)]
valid=india_datewise.iloc[int(india_datewise.shape[0]*0.95):]
y_pred=valid.copy()


# In[ ]:


model_ma= auto_arima(model_train["Confirmed"],trace=True, error_action='ignore', start_p=0,start_q=0,max_p=0,max_q=5,
                   suppress_warnings=True,stepwise=False,seasonal=False)
model_ma.fit(model_train["Confirmed"])


# In[ ]:


prediction_ma=model_ma.predict(len(valid))
y_pred["MA Model Prediction"]=prediction_ma


# In[ ]:


model_scores.append(np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["MA Model Prediction"])))
print("Root Mean Square Error for MA Model: ",np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["MA Model Prediction"])))


# In[ ]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Confirmed"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=valid.index, y=valid["Confirmed"],
                    mode='lines+markers',name="Validation Data for Confirmed Cases",))
fig.add_trace(go.Scatter(x=valid.index, y=y_pred["MA Model Prediction"],
                    mode='lines+markers',name="Prediction for Confirmed Cases",))
fig.update_layout(title="Confirmed Cases MA Model Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[ ]:


MA_model_new_prediction=[]
for i in range(1,18):
    MA_model_new_prediction.append(model_ma.predict(len(valid)+i)[-1])
model_predictions["MA Model Prediction"]=MA_model_new_prediction
model_predictions.head()


# ### ARIMA Model (using AUTO ARIMA)

# In[ ]:


model_train=india_datewise.iloc[:int(india_datewise.shape[0]*0.95)]
valid=india_datewise.iloc[int(india_datewise.shape[0]*0.95):]
y_pred=valid.copy()


# In[ ]:


model_arima= auto_arima(model_train["Confirmed"],trace=True, error_action='ignore', start_p=1,start_q=1,max_p=3,max_q=3,
                   suppress_warnings=True,stepwise=False,seasonal=False)
model_arima.fit(model_train["Confirmed"])


# In[ ]:


prediction_arima=model_arima.predict(len(valid))
y_pred["ARIMA Model Prediction"]=prediction_arima


# In[ ]:


model_scores.append(np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["ARIMA Model Prediction"])))
print("Root Mean Square Error for MA Model: ",np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["ARIMA Model Prediction"])))


# In[ ]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=model_train.index, y=model_train["Confirmed"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=valid.index, y=valid["Confirmed"],
                    mode='lines+markers',name="Validation Data for Confirmed Cases",))
fig.add_trace(go.Scatter(x=valid.index, y=y_pred["ARIMA Model Prediction"],
                    mode='lines+markers',name="Prediction for Confirmed Cases",))
fig.update_layout(title="Confirmed Cases ARIMA Model Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[ ]:


ARIMA_model_new_prediction=[]
for i in range(1,18):
    ARIMA_model_new_prediction.append(model_arima.predict(len(valid)+i)[-1])
model_predictions["ARIMA Model Prediction"]=ARIMA_model_new_prediction
model_predictions.head()


# ### Facebook's Prophet Model

# In[ ]:


prophet_c=Prophet(interval_width=0.95,weekly_seasonality=True,)
prophet_confirmed=pd.DataFrame(zip(list(india_datewise.index),list(india_datewise["Confirmed"])),columns=['ds','y'])


# In[ ]:


prophet_c.fit(prophet_confirmed)


# In[ ]:


forecast_c=prophet_c.make_future_dataframe(periods=17)
forecast_confirmed=forecast_c.copy()


# In[ ]:


confirmed_forecast=prophet_c.predict(forecast_c)


# In[ ]:


rmse_prophet=np.sqrt(mean_squared_error(india_datewise["Confirmed"],confirmed_forecast['yhat'].head(india_datewise.shape[0])))
model_scores.append(rmse_prophet)
print("Root Mean Squared Error for Prophet Model: ",rmse_prophet)


# In[ ]:


print(prophet_c.plot(confirmed_forecast))


# In[ ]:


print(prophet_c.plot_components(confirmed_forecast))


# In[ ]:


model_predictions["Prophet's Prediction"]=list(confirmed_forecast["yhat"].tail(17))
model_predictions["Prophet's Upper Bound"]=list(confirmed_forecast["yhat_upper"].tail(17))
model_predictions.head()


# ## Summarizing Results of all Models

# In[ ]:


models=["Polynomial Regression","Support Vector Machine Regresssor","Holt's Linear Model",
       "Holt's Winter Model","Auto Regressive Model (AR)", "Moving Average Model (MA)","ARIMA Model","Facebook's Prophet Model"]


# In[ ]:


model_evaluation=pd.DataFrame(list(zip(models,model_scores)),columns=["Model Name","Root Mean Squared Error"])
model_evaluation=model_evaluation.sort_values(["Root Mean Squared Error"])
model_evaluation.style.background_gradient(cmap='Reds')


# In[ ]:


model_predictions["Average of Predictions Models"]=model_predictions.mean(axis=1)
show_predictions=model_predictions.head()
show_predictions


# ## Conclusion
# The population and bad hygine practices among majority of country's population are probably the most worrying concerns in accordance to COVID-19 for India. Another concern that might haunt India over and over again is lack of Medical Equipments, Non updgraded Medical technology, negliegence towards Medical facilities and that might play vital role in this pandemic. Less number of Testing, unavailabilty of Medical Hospitals might just add up things to those worries.
# 
# The number still looks good right now considering the population and India. There is also a silver lining to it, India has been able to tackle some serious disease like Plague, Chickenpox, tuberculosis, HIV over a course, mind you which has higher Mortality Rate in comparison to COVID-19. Also India enforced a Nationwide Lockdown at the right moment. "The unity among diversity" is another positive to take away, where people are working to help people below poverty line, people donating money to Government to fight against this pendemic which might play a significant role in this pandemic.
# 
# The course of this pendemic will be decided by the people of this country, the forecasts might look decent in comparison to other countries but that picture can change in just span of few days. It all depends on people how strictly they follow the rules and regulations imposed by the Government of India.
# 
# It will take 12-18 months for vaccine to be available on COVID-19 as per experts. Till then the only possible and effective vaccine on COVID-19 is Social Distancing at public places, Self Isolation in case if you see any symptoms of COVID-19, Quarantine of COVID positive patients, Lockdown and TESTING TESTING AND TESTING!
# 
# ### I will be updating the models and will be adding up visualization over time. Please upvote the Kernel if you have liked my work. Any kind of suggestion, queries and corrections would be highly appreciated.
# 
# ### Please Stay at your homes and Stay Safe. Follow the most basic yet effective hygine practices and I think we will be able to get throught this very soon. 
# ### "There is always a light at the end of every tunnel."
