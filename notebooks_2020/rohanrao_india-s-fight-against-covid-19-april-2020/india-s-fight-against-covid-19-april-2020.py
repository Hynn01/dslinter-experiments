#!/usr/bin/env python
# coding: utf-8

# ## Covid-19 in India
# ![](https://i.imgur.com/0gBVWNm.jpg)
# 
# The [Covid-19](https://en.wikipedia.org/wiki/Coronavirus_disease_2019) disease has shaken the world. It has spread far and wide across the globe leaving no one safe. While most countries have come to a standstill, a lot of healthcase groups, medical professionals, researchers and various other communities are fighting hard to overcome these hard times.
# 
# WHO declared Covid-19 a pandemic and India is fighting back by invoking the Epidemic Diseases Act in March, 2020: https://www.businesstoday.in/latest/trends/coronavirus-outbreak-india-italy-iran-china-total-confirmed-cases/story/397950.html
# 
# I strongly believe and trust the healthcase experts of the country and the world to fight the virus in the best possible way. There are plenty of analysis, dashboards, insights, models, forecasts that have been shared by the community and we are very grateful to all the organzations that collect and share data publicly to facilitate these reports.
# 

# ## Analysis in April
# Each day of April, I will explore data on Covid-19 in India about one particular aspect of it.
# 
# This notebook serves as a daily update of attempting to showcase data regarding many of the questions posed to India. It is, by no means, confirming or denying any hypothesis, suggestion or idea. I highly recommend to share questions, suggestions or feedback in the comments. I will also try to include as many of the suggestions in the coming days.
# 
# * [April 1: Population](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-1:-Population)   
# Can population be used to estimate mask and testing kit requirements in each state?
# * [April 2: Density](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-2:-Density)   
# Can we get city-level data to explore if densely populated cities are seeing more cases?
# * [April 3: Urbanization](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-3:-Urbanization)   
# Can we use geographical urbanization to control spread of cases?
# * [April 4: Monotonicity](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-4:-Monotonicity)   
# Can we improve the quality of data?
# * [April 5: Gender](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-5:-Gender)   
# Why doesn't the government share gender of cases?
# * [April 6: Funds](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-6:-Funds)   
# Can we contribute towards the fight from home?
# * [April 7: Trajectory](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-7:-Trajectory)   
# Has [Kerala's 2018 Nipah virus](https://en.wikipedia.org/wiki/2018_Nipah_virus_outbreak_in_Kerala) experience helped them flatten their curve?
# * [April 8: Testing](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-8:-Testing)   
# Can the four quadrants of lab availability help in determining the preparedness of sample testing?
# * [April 9: Age](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-9:-Age)   
# Are the elderly in more danger?
# * [April 10: Neighbours](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-10:-Neighbours)   
# Why is India worse off than all its neighbouring countries?
# * [April 11: Laboratories](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-11:-Laboratories)   
# Is India scaling enough laboraties to meet Covid-19 requirements?
# * [April 12: Forecasting](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-12:-Forercasting)   
# What should India really be forecasting?
# * [April 13: Mortality](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-13:-Mortality)   
# Why are mortality rates so unreliable?
# * [April 14: Curfew](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-14:-Curfew)   
# Was Janata Curfew useful?
# * [April 15: Lockdown](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-15:-Lockdown)   
# Can lockdowns help India?
# * [April 16: Google](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-16:-Google)   
# Are people slowly moving on from Covid-19?
# * [April 17: Nationality](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-17:-Nationality)   
# How did the proportion of non-Indian cases change over time?
# * [April 18: Closing](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-18:-Closing)   
# How can we use closing rate of cases as a sign of recovery?
# * [April 19: Travel](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-19:-Travel)   
# Can the travel history of positive individuals bring to light some patterns on spread?
# * [April 20: Vaccination](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-20:-Vaccination)   
# How do we fare in the race to find a vaccine?
# * [April 21: Hospitals](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-21:-Hospitals)   
# Are medical professionals taking a huge risk in helping treat patients?
# * [April 22: AarogyaSetu](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-22:-AarogyaSetu)   
# How can the government make the most of technology in this era of apps?
# * [April 23: Education](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-23:-Education)   
# Will schools and colleges die in post-Covid era?
# * [April 24: Beds](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-24:-Beds)   
# Does India have enough beds to treat every severe patient?
# * [April 25: Symptoms](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-25:-Symptoms)   
# Can symptoms be used as early warning signals?
# * [April 26: Masks](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-26:-Masks)   
# Can the never-ending debate on masks be solved using data?
# * [April 27: AQI](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-27:-AQI)   
# Will we get healthier in these times?
# * [April 28: Zones](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-28:-Zones)   
# Can India divide, conquer and re-unite?
# * [April 29: Unemployment](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-29:-Unemployment)   
# Is India reviving its economy steadily?
# * [April 30: Data Science](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-30:-Data Science)   
# How much Data Science can really help?
# 

# In[ ]:


## importing packages
import time

import numpy as np
import pandas as pd
import seaborn as sns

from bokeh.layouts import column, row
from bokeh.models import Panel, Tabs, LinearAxis, Range1d, BoxAnnotation, LabelSet, Span
from bokeh.models.tools import HoverTool
from bokeh.palettes import Category20, Spectral3, Spectral4, Spectral8
from bokeh.plotting import ColumnDataSource, figure, output_notebook, show
from bokeh.transform import dodge

from datetime import datetime as dt
from math import pi

output_notebook()


# In[ ]:


## defining constants
PATH_COVID = "/kaggle/input/covid19-in-india/covid_19_india.csv"
PATH_CENSUS = "/kaggle/input/covid19-in-india/population_india_census2011.csv"
PATH_TESTS = "/kaggle/input/covid19-in-india/ICMRTestingDetails.csv"
PATH_LABS = "/kaggle/input/covid19-in-india/ICMRTestingLabs.csv"
PATH_HOSPITALS = "/kaggle/input/covid19-in-india/HospitalBedsIndia.csv"
PATH_GLOBAL = "/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv"
PATH_METADATA = "/kaggle/input/covid19-forecasting-metadata/region_metadata.csv"
PATH_AQI = "/kaggle/input/air-quality-data-in-india/city_day.csv"

def read_covid_data():
    """
    Reads the main covid-19 India data and preprocesses it.
    """
    
    df = pd.read_csv(PATH_COVID)
    df.rename(columns = {
        "State/UnionTerritory": "state",
        "Confirmed": "cases",
        "Deaths": "deaths",
        "Cured": "recoveries"
    }, inplace = True)

    df.loc[df.state == "Telengana", "state"] = "Telangana"
    df["date"] = pd.to_datetime(df.Date, format = "%d/%m/%y").dt.date.astype(str)

    return df

def read_census_data():
    """
    Reads the 2011 Indian census data and preprocesses it.
    """
    
    df = pd.read_csv(PATH_CENSUS)
    df.rename(columns = {
        "State / Union Territory": "state",
        "Population": "population",
        "Urban population": "urban_population",
        "Gender Ratio": "gender_ratio"
    }, inplace = True)

    df["area"] = df.Area.str.replace(",", "").str.split("km").str[0].astype(int)

    return df

def read_test_samples_data():
    """
    Reads the ICMR test samples data and preprocesses it.
    """
    
    df = pd.read_csv(PATH_TESTS)
    df.drop(index = 0, inplace = True)
    df.rename(columns = {
        "TotalSamplesTested": "samples_tested"
    }, inplace = True)

    df["date"] = pd.to_datetime(df.DateTime, format = "%d/%m/%y %H:%S").dt.date.astype(str)
    
    return df

def read_test_labs_data():
    """
    Reads the ICMR testing labs data and preprocesses it.
    """
    
    df = pd.read_csv(PATH_LABS)
    
    return df

def read_hospitals_data():
    """
    Reads the Hospitals data and preprocesses it.
    """
    
    df = pd.read_csv(PATH_HOSPITALS)
    df.rename(columns = {
        "State/UT": "state"
    }, inplace = True)
    
    df.loc[df.state == "Andaman & Nicobar Islands", "state"] = "Andaman and Nicobar Islands"
    df.loc[df.state == "Jammu & Kashmir", "state"] = "Jammu and Kashmir"
    
    return df

def read_global_data():
    """
    Reads the global covid-19 data and preprocesses it.
    """
    
    df = pd.read_csv(PATH_GLOBAL)
    df_metadata = pd.read_csv(PATH_METADATA)
    
    df.rename(columns = {
        "Country/Region": "country",
        "Confirmed": "cases",
        "Deaths": "deaths",
        "Recovered": "recoveries"
    }, inplace = True)
    
    df_metadata.rename(columns = {
        "Country_Region": "country"
    }, inplace = True)

    df.loc[df.country == "Mainland China", "country"] = "China"
    df["date"] = pd.to_datetime(df.ObservationDate, format = "%m/%d/%Y").dt.date.astype(str)
    
    df = df.merge(df_metadata[["country", "continent"]].drop_duplicates(), on = "country", how = "left")
    
    return df

def read_aqi_data():
    """
    Reads AQI data and preprocesses it.
    """
    
    df = pd.read_csv(PATH_AQI)
    
    return df


# ## April-30: Data Science
# ![](https://i.imgur.com/eo08q1P.png)
# 
# The last topic is about how and why Data Science can help in fighting Covid-19. The world is filled with unknowns. How do we really discover them and make them known? How do we find answers for difficult questions?
# 
# Covid-19 is new with very less historic data available. In order to find answers and solve this pandemic there is no right approach or path but is likely going to be a combination of a lot of different organizations, teams, experiments and data that needs to come together to understand the virus and its effects.
# 
# Data is sparse but it can aid in understanding patterns, however big or small it may seem, help in measuring experiments and monitoring situations. By combining the power of data and algorithms with the intelligence and intuition of humans I strongly believe Covid-19 has no chance to survive for long.
# 
# > May the force be with us.
# 

# ## April-29: Unemployment
# ![](https://i.imgur.com/9lKQxQv.jpg)
# 
# What's the biggest side-effect of the lockdown due to Covid-19? Unemployment is certainly a bigger concern than boredom. India is seeing high numbers of unemployment like many other countries and it is dangerous for the economy. The balance between life and living as people are calling it these days is at a critical stage and there are probably no such thing as a right decision. Personally I'm optimistic about how India is handling the situation.
# 
# The unemployment rate went over 30% in early April but with gradual phasewise lifting of lockdown the rates have improved to ~ 21% at the end of the month: https://www.livemint.com/news/india/india-s-jobless-rate-tapers-to-5-week-low-as-economy-reboots-in-patches-11588140402194.html
# 
# > Remember that lives can also be lost due to unemployment and 20-30% is not a small number.
# 

# ## April-28: Zones
# ![](https://i.imgur.com/wJqnPya.jpg)
# 
# There are always going to be different areas having different severity of Covid-19. It becomes crucial to identify these 'hotspots' so that a phase-wise lifting of the lockdown can be put into effect. That is exactly what the government seems to be doing. As per the zones declared on 28th April, its likely that the red hotspots will continue following strict lockdown while the green zones will see life slowly crawling back and the economy getting revived. The orange zones need to be kept monitoring and probably will see a few services getting resumed.
# 
# The real test will be from 4th May onwards, when the lockdown will be partially lifted and it will be crucial for India to ensure there isn't a drastic rise in cases again. Well, what if there is? Are we ready for another month of complete lockdown?
# 

# ## April-27: AQI
# ![](https://cdn.cnn.com/cnnnext/dam/assets/200401105309-20200401-indian-gate-air-pollution-split-exlarge-169.jpg)
# 
# Covid-19 is known to be air-borne. That begs the question if there is any relation with air pollution. With the lockdown in effect for more than a month there is bound to be lesser pollution. But how much cleaner air are we breathing?
# 
# We'll use an [hourly AQI dataset](https://www.kaggle.com/rohanrao/air-quality-data-in-india) looking at some of the major cities.
# 

# In[ ]:


df_aqi = read_aqi_data()

df = df_aqi[df_aqi.City.isin(["Ahmedabad", "Bengaluru", "Chennai", "Delhi", "Gurugram", "Hyderabad", "Jaipur",
                              "Kolkata", "Lucknow", "Mumbai", "Patna", "Thiruvananthapuram"])]
df = df[((df.Date >= "2019-04-01") & (df.Date < "2019-04-10")) | ((df.Date >= "2020-04-01") & (df.Date < "2020-04-10"))]
df["Year"] = pd.to_datetime(df.Date).dt.year
df = df.groupby(["City", "Year"])["AQI"].mean().reset_index()
df = df.pivot_table(index = "City", columns = "Year", values = "AQI", aggfunc = np.mean).reset_index()
df.rename(columns = {2019: "AQI_2019_April", 2020: "AQI_2020_April"}, inplace = True)
df["Improvement in AQI"] = round((df.AQI_2019_April - df.AQI_2020_April) * 100 / df.AQI_2019_April, 2)
df.AQI_2019_April = df.AQI_2019_April.astype(int).astype(str)
df.AQI_2020_April = df.AQI_2020_April.astype(int).astype(str)
df.sort_values("Improvement in AQI", ascending = False, ignore_index = True, inplace = True)
df.columns.name = None


# In[ ]:


cm = sns.light_palette("green", as_cmap = True)

df.style.background_gradient(cmap = cm)


# Those are some incredible numbers that India has not seen for a long long time.
# 
# * Ahmedabad has over 70% better air than last April while Delhi, Lucknow and Gurugram have ~ 60% better air than last April.
# * Most cities have shown a significant improvement in AQI from last April.
# * Whats up with Kolkata? Either the city is not following the lockdown strictly or else the high AQI is coming due to some other reason. Its worth checking this in detail.

# ## April-26: Masks
# ![](https://i.imgur.com/erRLIkc.jpg)
# 
# There is a lot of debate around the usefulness and impact of using masks. Some say there is no use, some say there is. Some say it reduces contracting, some say it reduces spreading. Some say only N-95 masks are beneficial, some say even DIY homemade-masks are useful. Some say everyone should wear, some say only infected patients should wear.
# 
# Here is one initiative: https://masks4all.co/
# 
# There doesn't seem to be any publicly available data in India to quantify and verify refute these hypothesis. It is a catch-22 situation with no clear answer. Note that there is a downside if everyone start buying masks. If there is a production shortage there will be availability issues which has already been seen and it might lead to someone who really needs it more (like doctors or medical professionals) unable to get a mask on time.
# 
# It would be very interesting to explore data on the production line and supply chain of masks.
# 
# So masks for all or masks for most or masks for priority?
# 

# ## April-25: Symptoms
# ![](https://i.imgur.com/Pp8cqQj.png)
# 
# The symptoms of Covid-19 are evolving as deeper studies and reports are published about the patients. The symptoms are currently very similar to common flu and it is not easy to identify an infected patient without conducting a proper test.
# 
# It would be very interesting to summarize which symptoms were observed in patients and check if there are some symptoms more frequent or severe than others. It could also be useful to check if there were symptoms more often found in patients who died and use that as an early warning indicator.
# 
# Very unlikely India would have or make this data public. But I hope I'm wrong.
# 

# ## April-24: Beds
# ![](https://i.imgur.com/HuPNgod.jpg)
# 
# Proximity to hospitals and health centres are one dimension to look at in terms of access to resources in case someone experiences Covid-19 symptoms. The other is the capacity of the hospitals to admit severe cases for longer treatment.
# 
# Lets look at the availability of beds across states and compare against Covdi-19 cases and expected serious cases.
# 

# In[ ]:


df_covid = read_covid_data()
df_hospitals = read_hospitals_data()

df = df_covid.groupby("state")["cases", "recoveries", "deaths"].max().reset_index().merge(df_hospitals, on = "state")
df["active"] = df.cases - df.recoveries - df.deaths
df.fillna(0, inplace = True)
df.sort_values("NumPublicBeds_HMIS", ascending = False, inplace = True)


# In[ ]:


source = ColumnDataSource(data = dict(
    state = df.state.values,
    active = df.active.values,
    PublicBeds = df.NumPublicBeds_HMIS.values
))

tooltips = [
    ("State", "@state"),
    ("Active Cases", "@active"),
    ("Beds", "@PublicBeds")
]

v = figure(plot_width = 650,plot_height = 400, x_range = df.state.values, tooltips = tooltips, title = "Beds and Cases by State")

v.vbar(x = dodge("state", 0.15, range = v.x_range), top = "active", width = 0.2, source = source, color = "orange", legend_label = "Active Cases")
v.vbar(x = dodge("state", -0.15, range = v.x_range), top = "PublicBeds", width = 0.2, source = source, color = "green", legend_label = "Beds")

v.xaxis.major_label_orientation = -pi / 4

v.xaxis.axis_label = "State"
v.yaxis.axis_label = "Count"

v.legend.location = "top_right"

show(v)


# As seen here India has a lot of beds available compared to Covid-19 cases. Of course the beds need to be used for patients with other health issues as well as the fact that about 20-25% of cases are actually serious enough that a hosital bed is required indicates that there is no immediate danger of hitting the capacity of any state.
# 

# ## April-23: Education
# ![](https://i.imgur.com/ZuV8L8o.jpg)
# 
# March is the main month of all board exams in India. Most have been cancelled. Some postponed indefinitely. There is a lot of uncertainty of how this will be handled in the days to come. While we hope the education department will carve out a plan there is no dearth of material available online for kids, students and even adults to learn.
# 
# If you are home for an entire month, or two, or maybe more, you can always spend time going through e-courses and reading and learning anything of interest. It is a great opportunity to pick up skills and spend time on education that was deprioritized during normal routine.
# 

# ## April-22: AarogyaSetu
# ![](https://i.imgur.com/Fm7g75x.png)
# 
# With the rise in technology the Government of India has launched an app called Aarogya Setu. Official webpage: https://www.mygov.in/covid-19
# 
# Its primary goal is to provide information to the masses, collect data and connect healthcare with society to fasten the fight against Covid-19.
# 
# It recently crossed 75 million downloads: https://economictimes.indiatimes.com/tech/internet/aarogya-setu-app-crosses-75-million-downloads/articleshow/75359890.cms
# 

# ## April-21: Hospitals
# ![](https://i.imgur.com/GYIyWig.jpg)
# 
# Healthcare system play a primary role in fighting against Covid-19. There isn't much of publicly available data on hospitals, healthcare centres and clinics, but we'll look at the one present [here](https://pib.gov.in/PressReleasePage.aspx?PRID=1539877).
# 

# In[ ]:


df_covid = read_covid_data()
df_hospitals = read_hospitals_data()

df = df_covid.groupby("state")["cases", "recoveries", "deaths"].max().reset_index().merge(df_hospitals, on = "state")
df.fillna(0, inplace = True)
df["hospitals"] = df.NumSubDistrictHospitals_HMIS + df.NumDistrictHospitals_HMIS
df["healthcare_facilities"] = df.TotalPublicHealthFacilities_HMIS
df["cases_per_hospital"] = df.cases / df.hospitals
df["cases_per_healthcare_facility"] = df.cases / df.healthcare_facilities
df.sort_values("healthcare_facilities", ascending = False, inplace = True)


# In[ ]:


types = ["PrimaryHealthCenters", "CommunityHealthCenters", "SubDistrictHospitals", "DistrictHospitals"]

source_1 = ColumnDataSource(data = dict(
    state = df.state.values,
    PrimaryHealthCenters = df.NumPrimaryHealthCenters_HMIS.values,
    CommunityHealthCenters = df.NumCommunityHealthCenters_HMIS.values,
    SubDistrictHospitals = df.NumSubDistrictHospitals_HMIS.values,
    DistrictHospitals = df.NumDistrictHospitals_HMIS.values,
    hospitals = df.hospitals.values
))

tooltips_1 = [
    ("State", "@state"),
    ("Primary Health Centers", "@PrimaryHealthCenters"),
    ("Community Health Centers", "@CommunityHealthCenters"),
    ("Sub District Hospitals", "@SubDistrictHospitals"),
    ("District Hospitals", "@DistrictHospitals")
]

v1 = figure(plot_width = 650,plot_height = 400, x_range = df.state.values, tooltips = tooltips_1, title = "Healthcare Facilities by State")

v1.vbar_stack(types, x = "state", width = 0.9, color = Spectral4, source = source_1, legend_label = types)

v1.x_range.range_padding = 0.05
v1.xaxis.major_label_orientation = -pi / 4

v1.xaxis.axis_label = "State"
v1.yaxis.axis_label = "Count"
v1.legend.location = "top_right"

df.sort_values("cases_per_hospital", ascending = False, inplace = True)

source_2 = ColumnDataSource(data = dict(
    state = df.state.values,
    cases_per_healthcare_facility = df.cases_per_healthcare_facility.values,
    cases_per_hospital = df.cases_per_hospital.values
))

tooltips_21 = [
    ("Cases per Healthcare Facility", "@cases_per_healthcare_facility{0}")
]

tooltips_22 = [
    ("Cases per Hospital", "@cases_per_hospital{0}")
]

v2 = figure(plot_width = 650,plot_height = 400, x_range = df.state.values, title = "Cases per Facility / Hospital by State")

v21 = v2.vbar(x = dodge("state", 0.15, range = v2.x_range), top = "cases_per_healthcare_facility", width = 0.2, source = source_2, color = "green", legend_label = "Cases per Healthcare Facility")
v22 = v2.vbar(x = dodge("state", -0.15, range = v2.x_range), top = "cases_per_hospital", width = 0.2, source = source_2, color = "orange", legend_label = "Cases per Hospital")

v2.add_tools(HoverTool(renderers = [v21], tooltips = tooltips_21))
v2.add_tools(HoverTool(renderers = [v22], tooltips = tooltips_22))

v2.xaxis.major_label_orientation = -pi / 4

v2.xaxis.axis_label = "State"
v2.yaxis.axis_label = "Cases"

v2.legend.location = "top_right"

show(column(v1, v2))


# While the hospital numbers are a bit old, the trend gives a relative trend that there is a strain in hospitals in Delhi, Maharashtra and Gujarat due to the high number of Covid-19 cases. Even though Maharashtra has more total number of cases, it has more healthcare facilities to cater to the patients.
# 
# Some hospitals are getting [shut down](https://www.livemint.com/news/india/coronavirus-scare-3-private-hospitals-in-south-mumbai-closed-for-new-patients-11586443524154.html) due to the appearance of cases among the doctors and staff treating the patients.
# 
# These are extremely strenuous times for medical professionals who are nothing short of risking their lives to save others.
# 

# ## April-20: Vaccination
# ![](https://i.imgur.com/stYft0b.jpg)
# 
# There is no cure for Covid-19 as of today. No one knows if there ever will be or if the virus will just eventually die out like other epidemics. A vaccine will surely save a lot of lives and fasten the recovery speed of India and the world.
# 
# There are research teams working day and night to study and understand the virus so as to help develop a vaccine against it. It is not easy but heartening to know that the government as well as private companies are all working together trying to solve this problem.
# 
# https://www.livemint.com/news/india/covid-19-six-indian-companies-working-on-coronavirus-vaccine-11587016987400.html
# 
# Note that even if a vaccine is found in the near future, it is likely to take a few months for it to be produced and made available to the masses.
# 

# ## April-19: Travel
# ![](https://i.imgur.com/Obva2Ua.jpg)
# 
# A large proportion of early cases were of folks who recently travelled back to India from a foreign country. It would be useful to explore travel patterns of individuals who tested positive, along with flights where a positive individual travelled in, internationally as well as within India.
# 
# Its very unlikely to get hold of this data but would be happy to explore if anyone could find something!
# 

# ## April-18: Closing
# ![](https://i.imgur.com/EQiu6Pn.jpg)
# 
# While we continuously look at the number of cases it is also important to know how many cases are getting closed. A closed case is either the individual getting cured or passing away. An increasing close rate is also a sign of improvement.
# 

# In[ ]:


df_covid = read_covid_data()
df_global = read_global_data()

df_country = df_global.groupby(["country", "continent"])["cases", "deaths", "recoveries"].max().reset_index()
df_country["completed_cases"] = df_country.deaths + df_country.recoveries
df_country["close_rate"] = df_country.completed_cases / df_country.cases
df_country.sort_values("close_rate", ascending = False, inplace = True)
df_world = pd.concat([df_country[df_country.cases >= 100].head(10), df_country[df_country.country == "India"]])
df_asia = pd.concat([df_country[(df_country.cases >= 100) & (df_country.continent == "Asia")].head(10), df_country[df_country.country == "India"]])

df_state = df_covid.groupby("state")["cases", "deaths", "recoveries"].max().reset_index()
df_state["completed_cases"] = df_state.deaths + df_state.recoveries
df_state["close_rate"] = df_state.completed_cases / df_state.cases
df_state.sort_values("close_rate", ascending = False, inplace = True)


# In[ ]:


source_1 = ColumnDataSource(data = dict(
    country = df_world.country.values,
    cases = df_world.cases.values,
    completed_cases = df_world.completed_cases.values,
    close_rate = df_world.close_rate.values * 100
))

tooltips_1 = [
    ("Cases", "@cases")
]

tooltips_2 = [
    ("Closed Cases", "@completed_cases")
]

tooltips_3 = [
    ("Close Rate", "@close_rate{0} %")
]

v1 = figure(plot_width = 650, plot_height = 400, x_range = df_world.country.values, y_range = Range1d(0, 1.1 * max(df_world.cases.values)), title = "Covid-19 Top Global Close Rates (At least 100 cases)")
v1.extra_y_ranges = {"Close Rate": Range1d(start = 0, end = 100)}

v11 = v1.vbar(x = dodge("country", 0.15, range = v1.x_range), top = "cases", width = 0.2, source = source_1, color = "blue", legend_label = "Cases")
v12 = v1.vbar(x = dodge("country", -0.15, range = v1.x_range), top = "completed_cases", width = 0.2, source = source_1, color = "green", legend_label = "Closed Cases")
v13 = v1.line("country", "close_rate", source = source_1, color = "orange", y_range_name = "Close Rate", legend_label = "Close Rate")

v1.add_tools(HoverTool(renderers = [v11], tooltips = tooltips_1))
v1.add_tools(HoverTool(renderers = [v12], tooltips = tooltips_2))
v1.add_tools(HoverTool(renderers = [v13], tooltips = tooltips_3))

v1.xaxis.major_label_orientation = pi / 4

v1.xaxis.axis_label = "Country"
v1.yaxis.axis_label = "Count"
v1.add_layout(LinearAxis(y_range_name = "Close Rate", axis_label = "Close Rate"), "right")

v1.legend.location = "top_right"

source_2 = ColumnDataSource(data = dict(
    country = df_asia.country.values,
    cases = df_asia.cases.values,
    completed_cases = df_asia.completed_cases.values,
    close_rate = df_asia.close_rate.values * 100
))

v2 = figure(plot_width = 650, plot_height = 400, x_range = df_asia.country.values, y_range = Range1d(0, 1.1 * max(df_asia.cases.values)), title = "Covid-19 Top Asian Close Rates (At least 100 cases)")
v2.extra_y_ranges = {"Close Rate": Range1d(start = 0, end = 100)}

v21 = v2.vbar(x = dodge("country", 0.15, range = v2.x_range), top = "cases", width = 0.2, source = source_2, color = "blue", legend_label = "Cases")
v22 = v2.vbar(x = dodge("country", -0.15, range = v2.x_range), top = "completed_cases", width = 0.2, source = source_2, color = "green", legend_label = "Closed Cases")
v23 = v2.line("country", "close_rate", source = source_2, color = "orange", y_range_name = "Close Rate", legend_label = "Close Rate")

v2.add_tools(HoverTool(renderers = [v21], tooltips = tooltips_1))
v2.add_tools(HoverTool(renderers = [v22], tooltips = tooltips_2))
v2.add_tools(HoverTool(renderers = [v23], tooltips = tooltips_3))

v2.xaxis.major_label_orientation = pi / 4

v2.xaxis.axis_label = "Country"
v2.yaxis.axis_label = "Count"
v2.add_layout(LinearAxis(y_range_name = "Close Rate", axis_label = "Close Rate"), "right")

v2.legend.location = "top_right"

source_3 = ColumnDataSource(data = dict(
    state = df_state[df_state.cases >= 100].state.values,
    cases = df_state[df_state.cases >= 100].cases.values,
    completed_cases = df_state[df_state.cases >= 100].completed_cases.values,
    close_rate = df_state[df_state.cases >= 100].close_rate.values * 100
))

v3 = figure(plot_width = 650, plot_height = 400, x_range = df_state[df_state.cases >= 100].state.values, y_range = Range1d(0, 1.1 * max(df_state[df_state.cases >= 100].cases.values)), title = "Covid-19 Top State Close Rates (At least 100 cases)")
v3.extra_y_ranges = {"Close Rate": Range1d(start = 0, end = 100)}

v31 = v3.vbar(x = dodge("state", 0.15, range = v3.x_range), top = "cases", width = 0.2, source = source_3, color = "blue", legend_label = "Cases")
v32 = v3.vbar(x = dodge("state", -0.15, range = v3.x_range), top = "completed_cases", width = 0.2, source = source_3, color = "green", legend_label = "Closed Cases")
v33 = v3.line("state", "close_rate", source = source_3, color = "orange", y_range_name = "Close Rate", legend_label = "Close Rate")

v3.add_tools(HoverTool(renderers = [v31], tooltips = tooltips_1))
v3.add_tools(HoverTool(renderers = [v32], tooltips = tooltips_2))
v3.add_tools(HoverTool(renderers = [v33], tooltips = tooltips_3))

v3.xaxis.major_label_orientation = pi / 4

v3.xaxis.axis_label = "State"
v3.yaxis.axis_label = "Count"
v3.add_layout(LinearAxis(y_range_name = "Close Rate", axis_label = "Close Rate"), "right")

v3.legend.location = "top_right"

show(column(v1, v2, v3))


# India has closed ~ 25% of its cases which is on the lower side compared to many other countries. Of course its not the fairest comparison since the outbreak occurred at differents times in different places and the goal would be for most countries to increase their close rate over the cycle of Covid-19.
# 
# Among the states its more comparable since they were all around the same time. **Kerala** had the first case and with its steady decline in number of cases its close rate is above 60% with **Haryana** not far behind. The worst close rates are in **Gujarat, Madhya Pradesh and Rajasthan**.
# 
# I have also observed that some states, especially Madhya Pradesh don't seem to have their recoveries numbers updated frequently and hence the numbers might be exaggerated a bit. But I hope many states start inching towards the 100% mark in the days to come.
# 

# ## April-17: Nationality
# ![](https://i.imgur.com/UDFcKR3.jpg)
# 
# Since Covid-19 was first found in China and subsequently spread throughout the globe primarily due to infected people travelling from one place to another, it is natural that almost all the early cases reported in India were either foreigners or Indians who recently travelled back to India from abroad.
# 
# With the numbers growing rapidly now the share of non-Indian cases is extremely low in proportion. But it would be interested to see how the ratio moved over time and if there was a specific drop when lockdown and the ban on air travel came into effect. This again is similar to the [Gender](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-5:-Gender) and [Age](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-9:-Age) analysis for which individual level data is required that isn't available publicly yet.
# 

# ## April-16: Google
# ![](https://i.imgur.com/ML6M0Cw.png)
# 
# So what are Indians searching on Google during these times? Do they have any correlation with cases?   
# The data used comes from [Google Trends](https://trends.google.com/trends/?geo=IN). Since that is scaled between 0-100, we'll scale the Covid-19 cases in the same range.
# 

# In[ ]:


coronavirus_trend = np.array([7,6,7,6,4,5,5,4,4,4,3,3,4,4,3,3,3,3,2,2,2,2,2,3,3,3,3,4,4,4,8,16,23,19,14,12,12,15,17,20,25,32,31,33,37,38,40,77,60,68,73,81,78,93,93,99,100,98,77,78,74,69,86,87,81,59,56,54,54,55,55,55,76,50,71,70,69,72,70,41,40])

df_covid = read_covid_data()

df = df_covid.groupby("date")["cases"].sum().reset_index()
df["lag_1_cases"] = df.cases.shift(1)
df["day_cases"] = df.cases - df.lag_1_cases
df = df[(df.date >= "2020-02-01") & (df.date <= "2020-04-21")]

df["cases_scaled"] = df.day_cases * 100 / max(df.day_cases)

df["coronavirus_google_trend"] = coronavirus_trend


# In[ ]:


source = ColumnDataSource(data = dict(
    date = np.array(df.date.values, dtype = np.datetime64),
    date_raw = df.date.values,
    cases_scaled = df.cases_scaled.values,
    coronavirus_google_trend = df.coronavirus_google_trend.values
))

tooltips_1 = [
    ("Date", "@date_raw"),
    ("Coronavirus Cases Scaled", "@cases_scaled{0}")
]

tooltips_2 = [
    ("Date", "@date_raw"),
    ("Coronavirus Google Trend", "@coronavirus_google_trend{0}")
]

v = figure(plot_width = 650, plot_height = 400, x_axis_type = "datetime", title = "Coronavirus search trend on Google")

v1 = v.line("date", "cases_scaled", source = source, color = "blue", legend_label = "Coronavirus Cases Scaled")
v2 = v.line("date", "coronavirus_google_trend", source = source, color = "green", legend_label = "Coronavirus Google Trend")

v.add_tools(HoverTool(renderers = [v1], tooltips = tooltips_1))
v.add_tools(HoverTool(renderers = [v2], tooltips = tooltips_2))

v.xaxis.major_label_orientation = pi / 4

v.xaxis.axis_label = "Date"
v.yaxis.axis_label = "Value"

v.legend.location = "top_left"

show(v)


# No clear relationship yet but it seems the interest level of coronavirus has dropped a bit. Are people tired? Has the Covid-19 information already spread to majority of population? Is India moving on in life?
# 

# ## April-15: Lockdown
# ![](https://i.imgur.com/Zsg8591.jpg)
# 
# India was under a 21-day complete lockdown from 25th March 00:00 to 14th April 23:59. The lockdown was further extended till 3rd May.
# 
# These lockdown are measures taken to enforce social distancing and many countries have seen respite from Covid-19 by using these measures. China's Hubei province where Covid-19 was first detected is free of the virus after over 2 months of lockdown. Can lockdown save India too?
# 

# In[ ]:


df_covid = read_covid_data()

df = df_covid.groupby("date")["cases", "deaths", "recoveries"].sum().reset_index()
df = df[df.date >= "2020-03-01"]


# In[ ]:


source = ColumnDataSource(data = dict(
    date = np.array(df.date.values, dtype = np.datetime64),
    date_raw = df.date.values,
    cases = df.cases.values,
    deaths = df.deaths.values,
    recoveries = df.recoveries.values
))

tooltips_1 = [
    ("Date", "@date_raw"),
    ("Cases", "@cases")
]

tooltips_2 = [
    ("Date", "@date_raw"),
    ("Deaths", "@deaths")
]

tooltips_3 = [
    ("Date", "@date_raw"),
    ("Recoveries", "@recoveries")
]

v = figure(plot_width = 650, plot_height = 400, x_axis_type = "datetime", title = "Cumulative metric counts before and during lockdowns")

v1 = v.line("date", "cases", source = source, color = "blue", legend_label = "Cases")
v2 = v.line("date", "deaths", source = source, color = "brown", legend_label = "Deaths")
v3 = v.line("date", "recoveries", source = source, color = "green", legend_label = "Recoveries")

v.add_tools(HoverTool(renderers = [v1], tooltips = tooltips_1))
v.add_tools(HoverTool(renderers = [v2], tooltips = tooltips_2))
v.add_tools(HoverTool(renderers = [v3], tooltips = tooltips_3))

curfew = Span(location = 7.5, dimension = "height", line_color = "grey", line_dash = "dashed", line_width = 2)
v.add_layout(curfew)

lockdown_1_start_date = time.mktime(dt(2020, 3, 25, 0, 0, 0).timetuple()) * 1000
lockdown_2_start_date = time.mktime(dt(2020, 4, 15, 0, 0, 0).timetuple()) * 1000

lockdown_1 = BoxAnnotation(left = lockdown_1_start_date, right = lockdown_2_start_date, fill_alpha = 0.1, fill_color = "yellow")
lockdown_2 = BoxAnnotation(left = lockdown_2_start_date, fill_alpha = 0.1, fill_color = "orange")

v.add_layout(lockdown_1)
v.add_layout(lockdown_2)

v.xaxis.major_label_orientation = pi / 4

v.xaxis.axis_label = "Date"
v.yaxis.axis_label = "Count"

v.legend.location = "top_left"

show(v)


# The yellow section is when the first 21-days lockdown was in effect in India. The orange section is when the lockdown extension is in effect in India.
# 
# Its a little early to go deeper into the impact of the lockdown and I'm leaving this open for now. I will update this towards the end of the month.
# 

# ## April-14: Curfew
# ![](https://i.imgur.com/hQUqjFC.jpg)
# 
# India observerd a nation-wide 14-hour voluntary curfew called [Janata Curfew](https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_India#Closedown_and_curfews) to curb the spread of Covid-19. Due to the enormity of the virus it was quickly realized that a single day's constraint is unlikely to help much and a complete nation-wide [Lockdown](https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_India#Lockdown) for 21-days was announced from 25th March 00:00.
# 
# So did the single day curfew show any benefit?
# 

# In[ ]:


df_covid = read_covid_data()

df = df_covid.groupby("date")["cases", "deaths", "recoveries"].sum().reset_index()
df = df[(df.date >= "2020-03-15") & (df.date <= "2020-04-06")]


# In[ ]:


source = ColumnDataSource(data = dict(
    date = df.date.values,
    cases = df.cases.values,
    deaths = df.deaths.values,
    recoveries = df.recoveries.values
))

tooltips_1 = [
    ("Date", "@date"),
    ("Cases", "@cases")
]

tooltips_2 = [
    ("Date", "@date"),
    ("Deaths", "@deaths")
]

tooltips_3 = [
    ("Date", "@date"),
    ("Recoveries", "@recoveries")
]

v = figure(plot_width = 650, plot_height = 400, x_range = df.date.values, title = "Cumulative metric counts before and after curfew")

v1 = v.line("date", "cases", source = source, color = "blue", legend_label = "Cases")
v2 = v.line("date", "deaths", source = source, color = "orange", legend_label = "Deaths")
v3 = v.line("date", "recoveries", source = source, color = "green", legend_label = "Recoveries")

v.add_tools(HoverTool(renderers = [v1], tooltips = tooltips_1))
v.add_tools(HoverTool(renderers = [v2], tooltips = tooltips_2))
v.add_tools(HoverTool(renderers = [v3], tooltips = tooltips_3))

curfew = Span(location = 7.5, dimension = "height", line_color = "grey", line_dash = "dashed", line_width = 2)
v.add_layout(curfew)

v.xaxis.major_label_orientation = pi / 4

v.xaxis.axis_label = "Date"
v.yaxis.axis_label = "Count"

v.legend.location = "top_left"

show(v)


# You'd barely notice any difference or effect of the curfew. China imposed strict lockdowns for over two months and many of the provinces now are free from Covid-19. You can't expect a single day's curfew to show any impact.
# 
# The intent of the curfew is debatable. Maybe it was just to get the people prepared for the complete lockdown. Maybe it was done with some other goal in mind.
# 

# ## April-13: Mortality
# ![](https://i.imgur.com/m1MjDLs.jpg)
# 
# As the death toll due to Covid-19 is increasing each day it is evidently becoming one of the worst pandemics ever. Where does India lie in its mortality rate compared to other countries? Let's also look at the mortality rate of the Indian states.
# 

# In[ ]:


df_covid = read_covid_data()
df_global = read_global_data()

df_country = df_global.groupby(["country", "continent"])["deaths", "recoveries"].max().reset_index()
df_country["completed_cases"] = df_country.deaths + df_country.recoveries
df_country["mortality_rate"] = df_country.deaths / df_country.completed_cases
df_country.sort_values("mortality_rate", ascending = False, inplace = True)
df_world = pd.concat([df_country[df_country.completed_cases >= 100].head(10), df_country[df_country.country == "India"]])
df_asia = df_country[(df_country.completed_cases >= 100) & (df_country.continent == "Asia")].head(10)

df_state = df_covid.groupby("state")["deaths", "recoveries"].max().reset_index()
df_state["completed_cases"] = df_state.deaths + df_state.recoveries
df_state["mortality_rate"] = df_state.deaths / df_state.completed_cases
df_state.sort_values("mortality_rate", ascending = False, inplace = True)


# In[ ]:


source_1 = ColumnDataSource(data = dict(
    country = df_world.country.values,
    completed_cases = df_world.completed_cases.values,
    deaths = df_world.deaths.values,
    recoveries = df_world.recoveries.values,
    mortality_rate = df_world.mortality_rate.values * 100
))

tooltips_1 = [
    ("Recoveries", "@recoveries")
]

tooltips_2 = [
    ("Deaths", "@deaths")
]

tooltips_3 = [
    ("Mortality Rate", "@mortality_rate{0} %")
]

v1 = figure(plot_width = 650, plot_height = 400, x_range = df_world.country.values, y_range = Range1d(0, 1.1 * max(df_world.deaths.values)), title = "Covid-19 Top Global Mortality Rates (At least 100 completed cases)")
v1.extra_y_ranges = {"Mortality Rate": Range1d(start = 0, end = 100)}

v11 = v1.vbar(x = dodge("country", 0.15, range = v1.x_range), top = "recoveries", width = 0.2, source = source_1, color = "blue", legend_label = "Recoveries")
v12 = v1.vbar(x = dodge("country", -0.15, range = v1.x_range), top = "deaths", width = 0.2, source = source_1, color = "orange", legend_label = "Deaths")
v13 = v1.line("country", "mortality_rate", source = source_1, color = "red", y_range_name = "Mortality Rate", legend_label = "Mortality Rate")

v1.add_tools(HoverTool(renderers = [v11], tooltips = tooltips_1))
v1.add_tools(HoverTool(renderers = [v12], tooltips = tooltips_2))
v1.add_tools(HoverTool(renderers = [v13], tooltips = tooltips_3))

v1.xaxis.major_label_orientation = pi / 4

v1.xaxis.axis_label = "Country"
v1.yaxis.axis_label = "Count"
v1.add_layout(LinearAxis(y_range_name = "Mortality Rate", axis_label = "Mortality Rate"), "right")

v1.legend.location = "top_right"

source_2 = ColumnDataSource(data = dict(
    country = df_asia.country.values,
    completed_cases = df_asia.completed_cases.values,
    deaths = df_asia.deaths.values,
    recoveries = df_asia.recoveries.values,
    mortality_rate = df_asia.mortality_rate.values * 100
))

v2 = figure(plot_width = 650, plot_height = 400, x_range = df_asia.country.values, y_range = Range1d(0, 1.1 * max(df_asia.recoveries.values)), title = "Covid-19 Top Asian Mortality Rates (At least 100 completed cases)")
v2.extra_y_ranges = {"Mortality Rate": Range1d(start = 0, end = 100)}

v21 = v2.vbar(x = dodge("country", 0.15, range = v2.x_range), top = "recoveries", width = 0.2, source = source_2, color = "blue", legend_label = "Recoveries")
v22 = v2.vbar(x = dodge("country", -0.15, range = v2.x_range), top = "deaths", width = 0.2, source = source_2, color = "orange", legend_label = "Deaths")
v23 = v2.line("country", "mortality_rate", source = source_2, color = "red", y_range_name = "Mortality Rate", legend_label = "Mortality Rate")

v2.add_tools(HoverTool(renderers = [v21], tooltips = tooltips_1))
v2.add_tools(HoverTool(renderers = [v22], tooltips = tooltips_2))
v2.add_tools(HoverTool(renderers = [v23], tooltips = tooltips_3))

v2.xaxis.major_label_orientation = pi / 4

v2.xaxis.axis_label = "Country"
v2.yaxis.axis_label = "Count"
v2.add_layout(LinearAxis(y_range_name = "Mortality Rate", axis_label = "Mortality Rate"), "right")

v2.legend.location = "top_right"

source_3 = ColumnDataSource(data = dict(
    state = df_state[df_state.completed_cases >= 40].state.values,
    completed_cases = df_state[df_state.completed_cases >= 40].completed_cases.values,
    deaths = df_state[df_state.completed_cases >= 40].deaths.values,
    recoveries = df_state[df_state.completed_cases >= 40].recoveries.values,
    mortality_rate = df_state[df_state.completed_cases >= 40].mortality_rate.values * 100
))

v3 = figure(plot_width = 650, plot_height = 400, x_range = df_state[df_state.completed_cases >= 40].state.values, y_range = Range1d(0, 1.1 * max(df_state[df_state.completed_cases >= 40].recoveries.values)), title = "Covid-19 Top State Mortality Rates (At least 40 completed cases)")
v3.extra_y_ranges = {"Mortality Rate": Range1d(start = 0, end = 100)}

v31 = v3.vbar(x = dodge("state", 0.15, range = v3.x_range), top = "recoveries", width = 0.2, source = source_3, color = "blue", legend_label = "Recoveries")
v32 = v3.vbar(x = dodge("state", -0.15, range = v3.x_range), top = "deaths", width = 0.2, source = source_3, color = "orange", legend_label = "Deaths")
v33 = v3.line("state", "mortality_rate", source = source_3, color = "red", y_range_name = "Mortality Rate", legend_label = "Mortality Rate")

v3.add_tools(HoverTool(renderers = [v31], tooltips = tooltips_1))
v3.add_tools(HoverTool(renderers = [v32], tooltips = tooltips_2))
v3.add_tools(HoverTool(renderers = [v33], tooltips = tooltips_3))

v3.xaxis.major_label_orientation = pi / 4

v3.xaxis.axis_label = "State"
v3.yaxis.axis_label = "Count"
v3.add_layout(LinearAxis(y_range_name = "Mortality Rate", axis_label = "Mortality Rate"), "right")

v3.legend.location = "top_right"

show(column(v1, v2, v3))


# These numbers are heavily subjected to how accurately countries and states are publishing the number of deaths and recoveries. It is clear that some countries and states have no recoveries and many deaths which is due to the inconsistent data.
# 
# With what we have India's mortality rate is > 12% which is on the higher side among Asian countries. Within India, **Gujarat, Maharashta and West Bengal** have the highest mortality rates among states.
# 
# I'd look at these numbers with a pinch of salt.
# 

# ## April-12: Forecasting
# ![](https://i.imgur.com/ofNHGvU.jpg)
# 
# There are tons of forecasting models published. Globally, nationally as well as regionally. How well do they predict Covid-19? Are these models really useful?   
# 
# Well, no one really knows and a lot of teams are trying to assess and gauge features and trends for better forecasting. Whether it is a [SIER model](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SEIR_model) or an [Exponential Model](https://en.wikipedia.org/wiki/Exponential_growth) or a boosting based model or a deep learning model, they are just different approaches and without an actual objective method of evaluating them, each one is as good or as bad as any other.
# 
# Kaggle are running a series of [Covid-19 Forecasting Challenges](https://www.kaggle.com/c/covid19-global-forecasting-week-4) and many of the top models are simple methods of extrapolation. So does it mean that Covid-19 is still hard to forecast well?
# 
# The other question that arises is what are the useful actions that can be taken using the forecasts? Are we even forecasting the right thing? Should we forecast # testing kits required or # ICU beds needed or the peak of # cases?   
# The answer can even change from country to country and region to region.
# 

# ## April-11: Laboratories
# ![](https://i.imgur.com/bS8eouh.jpg)
# 
# In order to scale up and facilitate [Testing](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-8:-Testing) Covid-19 samples across India, it is very crucial to have sufficient testing laboraties and collection sites. How does India fare? Can it sustain the current growth?
# 
# Let's look at how these laboratories are spread across the country. The list of [ICMR laboratories](https://covid.icmr.org.in/index.php/testing-labs-deatails) are as per the official list.
# 

# In[ ]:


df_census = read_census_data()
df_labs = read_test_labs_data()

df_state = df_labs.groupby("state")["lab"].count().reset_index().rename(columns = {"lab": "labs"}).merge(df_census, on = "state")
df_state["people_per_lab"] = df_state.population / df_state.labs
df_state["area_per_lab"] = df_state.area / df_state.labs

df_state_lab = pd.pivot_table(df_labs, values = "lab", index = "state", columns = "type", aggfunc = "count", fill_value = 0).reset_index()
df_state_lab["labs"] = df_state_lab.sum(axis = 1)
df_state_lab = df_state_lab.sort_values("labs", ascending = False).head(10)

df_city_lab = pd.pivot_table(df_labs, values = "lab", index = "city", columns = "type", aggfunc = "count", fill_value = 0).reset_index()
df_city_lab["labs"] = df_city_lab.sum(axis = 1)
df_city_lab = df_city_lab.sort_values("labs", ascending = False).head(10)


# In[ ]:


source_1 = {
    "state": df_state_lab.state.values,
    "Government Laboratory": df_state_lab["Government Laboratory"].values,
    "Private Laboratory": df_state_lab["Private Laboratory"].values,
    "Collection Site": df_state_lab["Collection Site"].values
}

types = ["Government Laboratory", "Private Laboratory", "Collection Site"]

v1 = figure(plot_width = 300, plot_height = 400, x_range = source_1["state"], title = "Top States with Testing Laboratories")
v1.vbar_stack(types, x = "state", width = 0.81, color = Spectral3, source = source_1, legend_label = types)
v1.xaxis.major_label_orientation = pi / 6
v1.legend.label_text_font_size = "5pt"

source_2 = {
    "city": df_city_lab.city.values,
    "Government Laboratory": df_city_lab["Government Laboratory"].values,
    "Private Laboratory": df_city_lab["Private Laboratory"].values,
    "Collection Site": df_city_lab["Collection Site"].values
}

v2 = figure(plot_width = 300, plot_height = 400, x_range = source_2["city"], title = "Top Cities with Testing Laboratories")
v2.vbar_stack(types, x = "city", width = 0.81, color = Spectral3, source = source_2, legend_label = types)
v2.xaxis.major_label_orientation = pi / 6
v2.legend.label_text_font_size = "5pt"

source_3 = ColumnDataSource(data = dict(
    state = df_state.state.values,
    labs = df_state.labs.values,
    people_per_lab = df_state.people_per_lab.values / 1000000,
    area_per_lab = df_state.area_per_lab.values / 1000
))

tooltips_3 = [
    ("State", "@state"),
    ("Labs", "@labs"),
    ("People per Lab", "@people_per_lab{0.00} M"),
    ("Area per Lab", "@area_per_lab{0.00} K")
]

h_mid = max(df_state.area_per_lab.values / 1000) / 2
v_mid = max(df_state.people_per_lab.values / 1000000) / 2

source_labels = ColumnDataSource(data = dict(
    state = df_state[(df_state.people_per_lab >= v_mid * 1000000) | (df_state.area_per_lab >= h_mid * 1000)].state.values,
    people_per_lab = df_state[(df_state.people_per_lab >= v_mid * 1000000) | (df_state.area_per_lab >= h_mid * 1000)].people_per_lab.values / 1000000,
    area_per_lab = df_state[(df_state.people_per_lab >= v_mid * 1000000) | (df_state.area_per_lab >= h_mid * 1000)].area_per_lab.values / 1000
))

labels = LabelSet(x = "people_per_lab", y = "area_per_lab", text = "state", source = source_labels, level = "glyph", x_offset = -19, y_offset = -23, render_mode = "canvas")

v3 = figure(plot_width = 600, plot_height = 600, tooltips = tooltips_3, title = "People and Area per Lab by State")
v3.circle("people_per_lab", "area_per_lab", source = source_3, size = 13, color = "blue", alpha = 0.41)

tl_box = BoxAnnotation(right = v_mid, bottom = h_mid, fill_alpha = 0.1, fill_color = "orange")
tr_box = BoxAnnotation(left = v_mid, bottom = h_mid, fill_alpha = 0.1, fill_color = "red")
bl_box = BoxAnnotation(right = v_mid, top = h_mid, fill_alpha = 0.1, fill_color = "green")
br_box = BoxAnnotation(left = v_mid, top = h_mid, fill_alpha = 0.1, fill_color = "orange")

v3.add_layout(tl_box)
v3.add_layout(tr_box)
v3.add_layout(bl_box)
v3.add_layout(br_box)

v3.add_layout(labels)

v3.xaxis.axis_label = "People per Lab (in Million)"
v3.yaxis.axis_label = "Area per Lab (in Thousand sq km)"

show(column(row(v1, v2), v3))


# Dividing the grid of people per lab and area per lab into four quadrants, it is ideal to push as many states into the bottom-left quadrant as possible.
# 
# * The **top-left** quadrant are states with high area per lab. It might take longer for people to reach a lab for testing Covid-19. Currently **Ladakh and Arunachal Pradesh** lie here.
# * The **bottom-right** quadrant are states with high population per lab. A sudden outbreak could lead to long queues for testing Covid-19. Currently **Bihar, Uttar Pradesh and Jharkhand** lie here.
# * The **top-right** quadrant are states with high area as well as population per lab. This is a danger zone and **Chhattisgarh** is close.
# * The **bottom-left** quadrant are states with low area as well as population per lab. The testing in these states are in good condition and an ideal scenario is to work towards pushing all the states in this quadrant. Some states are close to orange zones and its worth trying to add a lab in these states.
# 
# This can greatly help in understanding the preparedness of states in handling Covid-19. A further analysis can be done based on samples tested per state to quantify the quality of the labs. Please share if this dataset can be made available from any source.
# 

# ## April-10: Neighbours
# ![india-neighbours.jpg](attachment:india-neighbours.jpg)
# 
# Most countries and governments have imposed lockdown measures, mobility constraints or strict restrictions. This prevents the spread of the virus through physical contact. Let's look at how the Covid-19 numbers in India compare against its neighbouring countries: Bangladesh, Bhutan, Burma, China, Nepal, Pakistan and Sri Lanka.
# 

# In[ ]:


df_global = read_global_data()

country_list = ["Bangladesh", "Bhutan", "Burma", "China", "India", "Nepal", "Pakistan", "Sri Lanka"]

df = df_global[df_global.country.isin(country_list)]
df = df.groupby(["country", "date"])[["cases", "deaths", "recoveries"]].sum().reset_index()


# In[ ]:


v = figure(plot_width = 650, plot_height = 400, x_axis_type = "datetime", title = "Covid-19 log of cumulative cases by country")

tooltips = [
    ("Country", "@country"),
    ("Date", "@date{%F}"),
    ("Cases", "@cases")
]
    
formatters = {
    "@date": "datetime"
}

for i in range(len(country_list)):
    country = country_list[i]
    df_country = df[df.country == country]

    source = ColumnDataSource(data = dict(
        country = df_country.country.values,
        date = np.array(df_country.date.values, dtype = np.datetime64),
        cases = df_country.cases.values,
        log_cases = np.log10(df_country.cases.values)
    ))
    
    v.line("date", "log_cases", source = source, color = Spectral8[i], legend_label = country)

v.add_tools(HoverTool(tooltips = tooltips, formatters = formatters))

v.legend.location = "bottom_left"
v.legend.label_text_font_size = "8pt"

v.xaxis.axis_label = "Date"
v.yaxis.axis_label = "Log Cases"

show(v)


# India seems to be doing the worst. Not only does it have the most cases after China but also looks to be on a steep rise. **China and Sri Lanka** seem to have flattened its curve and **India** needs to do the same to its trajectory as soon as possible.
# 
# The smaller countries like **Nepal, Burma and Bhutan** have their cases under control so far while **Bangladesh and Pakistan** still need to control the spread in the days to come.
# 

# ## April-9: Age
# ![age.jpg](attachment:age.jpg)
# 
# As per this [official report from WHO](https://www.who.int/docs/default-source/coronaviruse/who-china-joint-mission-on-covid-19-final-report.pdf), Covid-19 is particularly deadly for higher age groups. We have seen cases across all age-groups now but a majority of deaths are from the elderly and among those already having medical issues.
# 
# An early dataset of majorly the initial cases shows this histogram of age available on https://www.covid19india.org/deepdive
# 
# Unfortunately, we still do not have enough data on age of Covid-19 cases in India. Getting this will be tremendously useful in understanding and confirming if the age factor seen in China and worldwide is similar to India.
# 
# But even without this data, we must work harder towards treating the elderly.
# 

# ## April-8: Testing
# ![test.jpeg](attachment:test.jpeg)
# 
# A lot of the Covid-19 metrics are in the form of a funnel. You need to have sufficient data at the top of the funnel for the outputs at the bottom to be reliable. So what is at the top of the funnel? Testing!
# 
# If India is not able to test everyone with symptoms, the cases are under-stated. There could be more cases but are unknown due to limited testing. So how is India managing this? Is India meeting the testing requirements of Covid-19 and minimizing this unknown factor?
# 

# In[ ]:


df_covid = read_covid_data()
df_census = read_census_data()
df_testing = read_test_samples_data()
df_labs = read_test_labs_data()

df_country = df_covid.groupby("date")["cases"].sum().reset_index().merge(df_testing[["date", "samples_tested"]], on = "date")
df_country["lag_1_cases"] = df_country.cases.shift(1)
df_country["day_cases"] = df_country.cases - df_country.lag_1_cases
df_country["lag_1_samples_tested"] = df_country.samples_tested.shift(1)
df_country["day_samples_tested"] = df_country.samples_tested - df_country.lag_1_samples_tested

df_country = df_country[df_country.date >= "2020-03-18"]
df_country.dropna(subset = ["day_cases", "day_samples_tested"], inplace = True)
df_country["case_rate"] = df_country.day_cases / df_country.day_samples_tested

df_state = df_labs.groupby("state")["lab"].count().reset_index().rename(columns = {"lab": "labs"}).merge(df_census, on = "state")
df_state["people_per_lab"] = df_state.population / df_state.labs
df_state["area_per_lab"] = df_state.area / df_state.labs


# In[ ]:


source = ColumnDataSource(data = dict(
    date = df_country.date.values,
    day_cases = df_country.day_cases.values,
    day_samples_tested = df_country.day_samples_tested.values,
    case_rate = df_country.case_rate.values
))

tooltips_1 = [
    ("Date", "@date"),
    ("Samples Tested", "@day_samples_tested")
]

tooltips_2 = [
    ("Date", "@date"),
    ("Cases", "@day_cases")
]

tooltips_3 = [
    ("Date", "@date"),
    ("Case Rate", "@case_rate{0.00}")
]

v = figure(plot_width = 650, plot_height = 400, x_range = df_country.date.values, title = "Covid-19 cases and testing from 19th March")
v.extra_y_ranges = {"Case Rate": Range1d(start = 0.0, end = 0.1)}

v1 = v.vbar(x = dodge("date", 0.25, range = v.x_range), top = "day_samples_tested", width = 0.2, source = source, color = "blue", legend_label = "Samples Tested")
v2 = v.vbar(x = dodge("date", -0.25, range = v.x_range), top = "day_cases", width = 0.2, source = source, color = "orange", legend_label = "Cases")
v3 = v.line("date", "case_rate", source = source, color = "red", y_range_name = "Case Rate", legend_label = "Case Rate")

v.add_tools(HoverTool(renderers = [v1], tooltips = tooltips_1))
v.add_tools(HoverTool(renderers = [v2], tooltips = tooltips_2))
v.add_tools(HoverTool(renderers = [v3], tooltips = tooltips_3))

v.xaxis.major_label_orientation = pi/4

v.xaxis.axis_label = "Date"
v.yaxis.axis_label = "Count"
v.add_layout(LinearAxis(y_range_name = "Case Rate", axis_label = "Case Rate"), "right")

v.legend.location = "top_left"

show(v)


# Its a good sign to see the number of samples being tested for Covid-19 is increasing at a good rate. But is it enough?   
# As per [ICMR](https://www.icmr.nic.in/) press releases, they claim that there are enough laboratories and testing kits to cater to anyone needing to be tested.
# 
# About 4-5% of samples being tested are turning out to be positive. Is this in line with the % observed in other countries?
# 

# ## April-7: Trajectory
# ![FTC.jpg](attachment:FTC.jpg)
# 
# You might have heard the phrase 'Flatten The Curve'. But what does it really mean? How can data identify if the curve is flattening or not?
# Let's look at the trend of all the states and the overall numbers of the country.
# 

# In[ ]:


df_covid = read_covid_data()

df = df_covid.copy()
df["log_cases"] = np.log10(df.cases)
df["lag_1_cases"] = df.groupby("state")["cases"].shift(1)
df["day_cases"] = df.cases - df.lag_1_cases
df["lag_1_day_cases"] = df.groupby("state")["day_cases"].shift(1)
df["lag_2_day_cases"] = df.groupby("state")["day_cases"].shift(2)
df["lag_3_day_cases"] = df.groupby("state")["day_cases"].shift(3)
df["lag_4_day_cases"] = df.groupby("state")["day_cases"].shift(4)
df["lag_5_day_cases"] = df.groupby("state")["day_cases"].shift(5)
df["lag_6_day_cases"] = df.groupby("state")["day_cases"].shift(6)
df["ma_7d_day_cases"] = df[["day_cases", "lag_1_day_cases", "lag_2_day_cases", "lag_3_day_cases",
                              "lag_4_day_cases", "lag_5_day_cases", "lag_6_day_cases"]].mean(axis = 1).values
df["log_ma_7d_day_cases"] = np.log10(df.ma_7d_day_cases)
df = df[df.state != "Unassigned"]
df.date = pd.to_datetime(df.date, format = "%Y-%m-%d")
df = df[df.date >= pd.datetime(2020, 3, 21)]


# In[ ]:


v1 = figure(plot_width = 333, plot_height = 333, x_axis_type = "datetime", title = "Covid-19 cases from 21st March")
v2 = figure(plot_width = 333, plot_height = 333, x_axis_type = "datetime", title = "Covid-19 cases from 21st March")

tooltips = [
    ("State", "@state"),
    ("Date", "@date{%F}"),
    ("Cases", "@cases")
]
    
formatters = {
    "@date": "datetime"
}
    
for i in range(len(df.state.unique())):
    state = df.state.unique()[i]
    df_state = df[df.state == state]
    
    source = ColumnDataSource(data = dict(
        state = df_state.state.values,
        date = np.array(df_state.date.values, dtype = np.datetime64),
        cases = df_state.cases.values,
        log_cases = df_state.log_cases.values
    ))
    
    v1.line("date", "cases", source = source, color = Category20[20][i % 20])
    v2.line("date", "log_cases", source = source, color = Category20[20][i % 20])

v1.add_tools(HoverTool(tooltips = tooltips, formatters = formatters))
v2.add_tools(HoverTool(tooltips = tooltips, formatters = formatters))

v1.xaxis.axis_label = "Date"
v1.yaxis.axis_label = "Cases"

v2.xaxis.axis_label = "Date"
v2.yaxis.axis_label = "Cases (Log Scale)"

show(row(v1, v2))


# [John Burn-Murdoch](https://twitter.com/jburnmurdoch) from Financial Times has produced some great visualizations on Covid-19 of which the log-scale view of cases is very popular. That's the second graph above while the first one are raw values of cases.
# 
# Let's look at some particular states in detail:
# 
# **Kerala**: First state to get a case
# **Maharashtra, Tamil Nadu, Delhi**: Top-3 states with most cases
# 

# In[ ]:


state_list = ["Delhi", "Kerala", "Maharashtra", "Tamil Nadu"]

v1 = figure(plot_width = 333, plot_height = 333, x_axis_type = "datetime", title = "Covid-19 cumulative cases since 21st March")
v2 = figure(plot_width = 333, plot_height = 333, x_axis_type = "datetime", title = "Covid-19 MA(7) day cases since 21st March")

tooltips_1 = [
    ("State", "@state"),
    ("Date", "@date{%F}"),
    ("Cases", "@cases")
]
    
tooltips_2 = [
    ("State", "@state"),
    ("Date", "@date{%F}"),
    ("Day Cases", "@ma_7d_day_cases{0}")
]

formatters = {
    "@date": "datetime"
}

for i in range(len(state_list)):
    state = state_list[i]
    df_state = df[df.state == state]

    source = ColumnDataSource(data = dict(
        state = df_state.state.values,
        date = np.array(df_state.date.values, dtype = np.datetime64),
        cases = df_state.cases.values,
        log_cases = df_state.log_cases.values,
        ma_7d_day_cases = df_state.ma_7d_day_cases.values,
        log_ma_7d_day_cases = df_state.log_ma_7d_day_cases.values
    ))
    
    v1.line("date", "log_cases", source = source, color = Spectral4[i], legend_label = state)
    v2.line("date", "log_ma_7d_day_cases", source = source, color = Spectral4[i], legend_label = state)

v1.add_tools(HoverTool(tooltips = tooltips_1, formatters = formatters))
v2.add_tools(HoverTool(tooltips = tooltips_2, formatters = formatters))

v1.legend.location = "top_left"
v2.legend.location = "top_left"

v1.legend.label_text_font_size = "5pt"
v2.legend.label_text_font_size = "5pt"

v1.xaxis.axis_label = "Date"
v1.yaxis.axis_label = "Cases (Log Scale)"

v2.xaxis.axis_label = "Date"
v2.yaxis.axis_label = "MA(7) Day Cases (Log Scale)"

show(row(v1, v2))


# The first plot shows how the cumulative cases increase in log-scale. The second plot shows how the moving average of cases per day in last 7 days changes in log-scale.
# 
# Since Kerala has had cases since January, you can see that its curve has flattened out. The other three states are just beginning to see the flattening.
# 
# Note that Kerala fought the [Nipah virus in 2018](https://en.wikipedia.org/wiki/2018_Nipah_virus_outbreak_in_Kerala). Has that experience helped them against Covid-19?
# 
# Can we learn something from Kerala? Did the government take certain steps that helped the spread of Covid-19 or is this the gradual life we expect to see of the virus in India?
# 

# ## April-6: Funds
# ![fund.jpg](attachment:fund.jpg)
# 
# 
# We can all contribute to fighting against Covid-19 in some way or the other. The advancement of technology and easy of digital payments in India has exposed a large number of individual to contribute to funds with the click of a few buttons. Here is a list of funds available to which you can donate money to (in alphabetical order):
# 
# * Akshaya Patra: [Amazon Cares](https://www.akshayapatra.org/daan-ustaav-with-amazon-cares)
# * Building Dreams: [Razorpay](https://pages.razorpay.com/pl_EWNyAMQujbOKgR/view)
# * Elixir Ahmedabad: [Razorpay](https://pages.razorpay.com/pl_EW357Eyk0tOlaa/view)
# * Elixir Mumbai: [Razorpay](https://pages.razorpay.com/pl_EWwhkkXJ4tIsCG/view)
# * Elixir Vodadara: [Razorpay](https://pages.razorpay.com/pl_EWVhT1xvdhc8qD/view)
# * Give India: [Flipkart](https://flipkart.giveindia.org/)
# * Habitat for Humanity: [Amazon Cares](https://habitatindia.org/covid19appeal/)
# * Helpage India: [Official Website](https://www.helpageindia.org/covid-19/)
# * Hoi Foods: [Official Website](https://donateameal.hoifoods.com/)
# * IAHV India: [Razorpay](https://pages.razorpay.com/pl_EXwrPCmXVbgCPM/view)
# * Jan Sahas: [Razorpay](https://pages.razorpay.com/jansahasdonate)
# * Kriti: [Official Website](https://kriti.org.in/covid-relief.html)
# * KVN Foundation: [Razorpay](https://pages.razorpay.com/feedmybangalore)
# * Narayan Seva: [Razorpay](https://pages.razorpay.com/pl_EWqc9MjE5C5m9u/view)
# * OXFAM India: [Amazon Cares](https://donate.oxfamindia.org/coronavirus-amazoncares)
# * PharmEasy: [Razorpay](https://pages.razorpay.com/COVID-19-Mask-1)
# * PM Cares: [Official Website](https://www.pmindia.gov.in/en/news_updates/appeal-to-generously-donate-to-pms-citizen-assistance-and-relief-in-emergency-situations-fund-pm-cares-fund) | [Paytm](https://paytm.com/helpinghand/pm-cares-fund)
# * Razorpay: [Razorpay](https://pages.razorpay.com/razorpay-covid19-relief)
# * Responsenet: [Official Website](https://www.responsenet.org/donations-for-covid-19-migrant-workers-relief-fund/)
# * Sangati: [Razorpay](https://pages.razorpay.com/pl_EW6aLym55b2cuT/view)
# * Samarthanam: [Official Website](https://samarthanam.org/covid19-rapid-response-for-relief-kit)
# * Seeds India: [Official Website](https://www.seedsindia.org/covid19/)
# * Uber: [Milaap](https://milaap.org/fundraisers/support-uber-driver-fund) (Upto 8th May, 2020)
# * United Way Bengaluru: [Razorpay](https://pages.razorpay.com/pl_EXd8EirttNCYDF/view)
# * United Way Delhi: [Razorpay](https://pages.razorpay.com/pl_EVcjO765jZSsm9/view)
# * United Way Mumbai: [Amazon Cares](https://www.unitedwaymumbai.org/amazoncares)
# * World Vision: [Amazon Cares](https://www.worldvision.in/wvreliefresponse/index.aspx)
# * Feeding India: [Zomato](https://www.feedingindia.org/)
# 
# > ***No contribution is too little or too much***
# 
# Feel free to share more such funds that can be added to the list.
# 

# ## April-5: Gender
# ![gender-icons-wide.jpg](attachment:gender-icons-wide.jpg)
# 
# Does Covid-19 have an affinity to spread or infect individuals of a particular gender? Is there a difference in metrics across states with different gender ratios?
# 

# In[ ]:


df_covid = read_covid_data()
df_census = read_census_data()

df = df_covid.groupby("state")["cases", "deaths", "recoveries"].max().reset_index().merge(df_census[["state", "gender_ratio"]], on = "state")
df["death_rate"] = df.deaths / (df.deaths + df.recoveries)
df["rank_gender_ratio"] = df["gender_ratio"].rank(ascending = False)
df["rank_cases"] = df["cases"].rank(ascending = False)
df["rank_deaths"] = df["deaths"].rank(ascending = False)
df["rank_death_rate"] = df["death_rate"].rank(ascending = False)


# In[ ]:


source = ColumnDataSource(data = dict(
    state = df.state.values,
    gender_ratio = df.gender_ratio.values,
    cases = df.cases.values,
    deaths = df.deaths.values,
    death_rate = df.death_rate.values
))

tooltips_1 = [
    ("state", "@state"),
    ("gender_ratio", "@gender_ratio"),
    ("cases", "@cases")
]

tooltips_2 = [
    ("state", "@state"),
    ("gender_ratio", "@gender_ratio"),
    ("deaths", "@deaths")
]

tooltips_3 = [
    ("state", "@state"),
    ("gender_ratio", "@gender_ratio"),
    ("death_rate", "@death_rate{0.000}")
]

v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Gender Ratio vs Cases by State")
v1.circle("gender_ratio", "cases", source = source, size = 13, color = "blue", alpha = 0.41)
v1.xaxis.axis_label = "Gender Ratio"
v1.yaxis.axis_label = "Cases"

v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Gender Ratio vs Deaths by State")
v2.circle("gender_ratio", "deaths", source = source, size = 13, color = "red", alpha = 0.41)
v2.xaxis.axis_label = "Gender Ratio"
v2.yaxis.axis_label = "Deaths"

v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Gender Ratio vs Death-Rate by State")
v3.circle("gender_ratio", "death_rate", source = source, size = 13, color = "orange", alpha = 0.41)
v3.xaxis.axis_label = "Gender Ratio"
v3.yaxis.axis_label = "Death Rate"

show(row(v1, v2, v3))


# Every circle represent a state's position based on its gender ratio and a Covid-19 metric (cases, deaths and death-rate for the three plots).   
# 
# The data points don't seem to exhibit any particular trend. Let's also look at the relative ranks of the gender ratio of states against the respective ranks in Covid-19 metrics.

# In[ ]:


source = ColumnDataSource(data = dict(
    state = df.state.values,
    rank_gender_ratio = df.rank_gender_ratio.values,
    rank_cases = df.rank_cases.values,
    rank_deaths = df.rank_deaths.values,
    rank_death_rate = df.rank_death_rate.values
))

tooltips_1 = [
    ("State", "@state"),
    ("Rank of Gender Ratio", "@rank_gender_ratio{0}"),
    ("Rank of Cases", "@rank_cases{0}")
]

tooltips_2 = [
    ("State", "@state"),
    ("Rank of Gender Ratio", "@rank_gender_ratio{0}"),
    ("Rank of Cases", "@rank_deaths{0}")
]

tooltips_3 = [
    ("State", "@state"),
    ("Rank of Gender Ratio", "@rank_gender_ratio{0}"),
    ("Rank of Death Rate", "@rank_death_rate{0}")
]

v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Ranks of Gender Ratio vs Cases by State")
v1.circle("rank_gender_ratio", "rank_cases", source = source, size = 13, color = "blue", alpha = 0.41)
v1.x_range.flipped = True
v1.y_range.flipped = True
v1.xaxis.axis_label = "Rank of Gender Ratio"
v1.yaxis.axis_label = "Rank of Cases"

v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Ranks of Gender Ratio vs Deaths by State")
v2.circle("rank_gender_ratio", "rank_deaths", source = source, size = 13, color = "red", alpha = 0.41)
v2.x_range.flipped = True
v2.y_range.flipped = True
v2.xaxis.axis_label = "Rank of Gender Ratio"
v2.yaxis.axis_label = "Rank of Deaths"

v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Ranks of Gender Ratio vs Death-Rate by State")
v3.circle("rank_gender_ratio", "rank_death_rate", source = source, size = 13, color = "orange", alpha = 0.41)
v3.x_range.flipped = True
v3.y_range.flipped = True
v3.xaxis.axis_label = "Rank of Gender Ratio"
v3.yaxis.axis_label = "Rank of Death Rate"

show(row(v1, v2, v3))


# Every circle represent a state's position based on its rank of gender ratio and rank of a Covid-19 metric (cases, deaths and death-rate for the three plots).
# 
# The Covid-19 metrics are pretty evenly spread across states irrespective of their relative gender ratio ranks. It would be very interesting to see and analyze the gender data of the individuals who have been infected with the virus. This is an open question and any dataset that could help with this analysis would be highly appreciated.
# 

# ## April-4: Monotonicity
# ![frequency.png](attachment:frequency.png)
# 
# Garbage-in Garbage-out. Before diving deeper into any dataset, it is crucial to understand it from a sanity perspective. In this case, the primary data is being publicly shared by the Government of India: https://www.mohfw.gov.in/ and is being maintained voluntarily by some amazing Kagglers [here](https://www.kaggle.com/sudalairajkumar/covid19-in-india).
# 
# While we must trust that the numbers shared by the government are their best efforts of providing correct data, can we at least perform some basic checks to see if the data is valid as per expectations. Is the data being updated regularly or daily? Are there some lags in publishing the data since even the government would be scrambling to compile this dataset from multiple sources across the country? Is the data clean enough to analyze trends and create features for deeper study?
# 

# In[ ]:


df_covid = read_covid_data()

df = df_covid.copy()
df = df[df.state != "Unassigned"]
df.date = pd.to_datetime(df.date, format = "%Y-%m-%d")


# In[ ]:


tab_list = []

for state in sorted(df.state.unique()):
    df_state = df[df.state == state]
    
    source = ColumnDataSource(data = dict(
        date = np.array(df_state.date.values, dtype = np.datetime64),
        cases = df_state.cases.values,
        deaths = df_state.deaths.values,
        recoveries = df_state.recoveries.values
    ))
    
    tooltips_1 = [
        ("Date", "@date{%F}"),
        ("Cases", "@cases")
    ]
    
    tooltips_2 = [
        ("Deaths", "@deaths")
    ]
    
    tooltips_3 = [
        ("Recoveries", "@recoveries")
    ]
    
    formatters = {
        "@date": "datetime"
    }
    
    v = figure(plot_width = 650, plot_height = 400, x_axis_type = "datetime", title = "Covid-19 metrics over time")
    v1 = v.line("date", "cases", source = source, color = "blue", legend_label = "Cases")
    v2 = v.line("date", "deaths", source = source, color = "red", legend_label = "Deaths")
    v3 = v.line("date", "recoveries", source = source, color = "green", legend_label = "Recoveries")
    v.legend.location = "top_left"
    v.add_tools(HoverTool(renderers = [v3], tooltips = tooltips_3, formatters = formatters, mode = "vline"))
    v.add_tools(HoverTool(renderers = [v2], tooltips = tooltips_2, formatters = formatters, mode = "vline"))
    v.add_tools(HoverTool(renderers = [v1], tooltips = tooltips_1, formatters = formatters, mode = "vline"))
    v.xaxis.axis_label = "Date"
    v.yaxis.axis_label = "Count"
    tab = Panel(child = v, title = state)
    tab_list.append(tab)

tabs = Tabs(tabs = tab_list)
show(tabs)


# Each tab displays a cumulative plot of the state's total number of cases, deaths and recoveries.
# 
# Since the data is aggregated in a cumulative fashion, we should expect monotonicity for each state. The only inconsistency is with Rajasthan data.   
# I've opened a [thread](https://www.kaggle.com/sudalairajkumar/covid19-in-india/discussion/141379) to resolve this data issue.
# 
# Otherwise it looks good to go!
# 

# ## April-3: Urbanization
# ![mumbai.jpg](attachment:mumbai.jpg)
# 
# While [population](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-1:-Population) and [density](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-2:-Density) looks at an entire area as a whole, urbanization enables dissecting an area based on the characteristic of development.
# 
# So how do more urban areas compare against rural ones? Does Covid-19 have any effect based on urbanization? Is it likely to spread more in urban areas due to presence of airports, railways and higher social connectivity? Or is it likely to affect the rural areas due to limited healthcare facilities and medical supplies?
# 

# In[ ]:


df_covid = read_covid_data()
df_census = read_census_data()

df_census["urbanization"] = df_census.urban_population / df_census.population

df = df_covid.groupby("state")["cases", "deaths", "recoveries"].max().reset_index().merge(df_census[["state", "urbanization"]], on = "state")
df["death_rate"] = df.deaths / (df.deaths + df.recoveries)
df["rank_urbanization"] = df["urbanization"].rank(ascending = False)
df["rank_cases"] = df["cases"].rank(ascending = False)
df["rank_deaths"] = df["deaths"].rank(ascending = False)
df["rank_death_rate"] = df["death_rate"].rank(ascending = False)


# In[ ]:


source = ColumnDataSource(data = dict(
    state = df.state.values,
    urbanization = df.urbanization.values,
    cases = df.cases.values,
    deaths = df.deaths.values,
    death_rate = df.death_rate.values
))

tooltips_1 = [
    ("State", "@state"),
    ("Urbanization", "@urbanization{0.00}"),
    ("Cases", "@cases")
]

tooltips_2 = [
    ("State", "@state"),
    ("Urbanization", "@urbanization{0.00}"),
    ("Deaths", "@deaths")
]

tooltips_3 = [
    ("State", "@state"),
    ("Urbanization", "@urbanization{0.00}"),
    ("Death Rate", "@death_rate{0.000}")
]

v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Urbanization vs Cases by State")
v1.circle("urbanization", "cases", source = source, size = 13, color = "blue", alpha = 0.41)
v1.xaxis.axis_label = "Urbanization"
v1.yaxis.axis_label = "Cases"

v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Urbanization vs Deaths by State")
v2.circle("urbanization", "deaths", source = source, size = 13, color = "red", alpha = 0.41)
v2.xaxis.axis_label = "Urbanization"
v2.yaxis.axis_label = "Deaths"

v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Urbanization vs Death-Rate by State")
v3.circle("urbanization", "death_rate", source = source, size = 13, color = "orange", alpha = 0.41)
v3.xaxis.axis_label = "Urbanization"
v3.yaxis.axis_label = "Death Rate"

show(row(v1, v2, v3))


# Every circle represent a state's position based on its urbanization and a Covid-19 metric (cases, deaths and death-rate for the three plots).   
# 
# There seems to be a little spike in the mid-range of urbanization. Let's look at the relative ranks of the urbanizaton of states against the respective ranks in Covid-19 metrics and check if the spikes still exist.
# 

# In[ ]:


source = ColumnDataSource(data = dict(
    state = df.state.values,
    rank_urbanization = df.rank_urbanization.values,
    rank_cases = df.rank_cases.values,
    rank_deaths = df.rank_deaths.values,
    rank_death_rate = df.rank_death_rate.values
))

tooltips_1 = [
    ("State", "@state"),
    ("Rank of Urbanization", "@rank_urbanization{0}"),
    ("Rank of Cases", "@rank_cases{0}")
]

tooltips_2 = [
    ("State", "@state"),
    ("Rank of Urbanization", "@rank_urbanization{0}"),
    ("Rank of Deaths", "@rank_deaths{0}")
]

tooltips_3 = [
    ("State", "@state"),
    ("Rank of Urbanization", "@rank_urbanization{0}"),
    ("Rank of Death Rate", "@rank_death_rate{0}")
]

v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Ranks of Urbanization vs Cases by State")
v1.circle("rank_urbanization", "rank_cases", source = source, size = 13, color = "blue", alpha = 0.41)
v1.x_range.flipped = True
v1.y_range.flipped = True
v1.xaxis.axis_label = "Rank of Urbanization"
v1.yaxis.axis_label = "Rank of Cases"

v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Ranks of Urbanization vs Deaths by State")
v2.circle("rank_urbanization", "rank_deaths", source = source, size = 13, color = "red", alpha = 0.41)
v2.x_range.flipped = True
v2.y_range.flipped = True
v2.xaxis.axis_label = "Rank of Urbanization"
v2.yaxis.axis_label = "Rank of Deaths"

v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Ranks of Urbanization vs Death-Rate by State")
v3.circle("rank_urbanization", "rank_death_rate", source = source, size = 13, color = "orange", alpha = 0.41)
v3.x_range.flipped = True
v3.y_range.flipped = True
v3.xaxis.axis_label = "Rank of Urbanization"
v3.yaxis.axis_label = "Rank of Death Rate"

show(row(v1, v2, v3))


# Every circle represent a state's position based on its rank of urbanization and rank of a Covid-19 metric (cases, deaths and death-rate for the three plots).
# 
# The plot has values all over the place and we don't see the spikes anymore so at least as of now it seems like urbanization doesn't impact Covid-19 much.   
# How else can we confirm this? What other approaches and ways could we slice the data in?
# 

# ## April-2: Density
# ![density.jpg](attachment:density.jpg)
# 
# [Physical Distancing (or commonly called Social Distancing)](https://en.wikipedia.org/wiki/Social_distancing) has been the immediate strategy taken by many countries to curb the rapid spread of Covid-19. Since it involves minimizes contact with humans, how feasible is it in highly dense areas?
# 
# India is among the [Top-20 densely populated countries](https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population_density) of the world. Hence most of India is dense by default. But how has it affected the spread of Covid-19? Is it better and safer in less dense areas?
# 
# Identifying how density affects Covid-19 can help authorities prioritize physical distancing better.
# 

# In[ ]:


df_covid = read_covid_data()
df_census = read_census_data()

df_census["density"] = df_census.population / df_census.area

df = df_covid.groupby("state")["cases", "deaths", "recoveries"].max().reset_index().merge(df_census[["state", "density"]], on = "state")
df["death_rate"] = df.deaths / (df.deaths + df.recoveries)
df["rank_density"] = df["density"].rank(ascending = False)
df["rank_cases"] = df["cases"].rank(ascending = False)
df["rank_deaths"] = df["deaths"].rank(ascending = False)
df["rank_death_rate"] = df["death_rate"].rank(ascending = False)


# In[ ]:


source = ColumnDataSource(data = dict(
    state = df.state.values,
    density = df.density.values,
    cases = df.cases.values,
    deaths = df.deaths.values,
    death_rate = df.death_rate.values
))

tooltips_1 = [
    ("State", "@state"),
    ("Density", "@density{0}"),
    ("Cases", "@cases")
]

tooltips_2 = [
    ("State", "@state"),
    ("Density", "@density{0}"),
    ("Deaths", "@deaths")
]

tooltips_3 = [
    ("State", "@state"),
    ("Density", "@density{0}"),
    ("Death Rate", "@death_rate{0.000}")
]

v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Density vs Cases by State")
v1.circle("density", "cases", source = source, size = 13, color = "blue", alpha = 0.41)
v1.xaxis.axis_label = "Density"
v1.yaxis.axis_label = "Cases"

v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Density vs Deaths by State")
v2.circle("density", "deaths", source = source, size = 13, color = "red", alpha = 0.41)
v2.xaxis.axis_label = "Density"
v2.yaxis.axis_label = "Deaths"

v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Density vs Death-Rate by State")
v3.circle("density", "death_rate", source = source, size = 13, color = "orange", alpha = 0.41)
v3.xaxis.axis_label = "Density"
v3.yaxis.axis_label = "Death Rate"

show(row(v1, v2, v3))


# Every circle represent a state's position based on its density and a Covid-19 metric (cases, deaths and death-rate for the three plots).   
# 
# Since many of the density number are in a smaller range, we can look at the relative ranks of the density of states against the respective ranks in Covid-19 metrics.
# 

# In[ ]:


source = ColumnDataSource(data = dict(
    state = df.state.values,
    rank_density = df.rank_density.values,
    rank_cases = df.rank_cases.values,
    rank_deaths = df.rank_deaths.values,
    rank_death_rate = df.rank_death_rate.values
))

tooltips_1 = [
    ("State", "@state"),
    ("Rank of Density", "@rank_density{0}"),
    ("Rank of Cases", "@rank_cases{0}")
]

tooltips_2 = [
    ("State", "@state"),
    ("Rank of Density", "@rank_density{0}"),
    ("Rank of Deaths", "@rank_deaths{0}")
]

tooltips_3 = [
    ("State", "@state"),
    ("Rank of Density", "@rank_density{0}"),
    ("Rank of Death Rate", "@rank_death_rate{0}")
]

v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Ranks of Density vs Cases by State")
v1.circle("rank_density", "rank_cases", source = source, size = 13, color = "blue", alpha = 0.41)
v1.x_range.flipped = True
v1.y_range.flipped = True
v1.xaxis.axis_label = "Rank of Density"
v1.yaxis.axis_label = "Rank of Cases"

v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Ranks of Density vs Deaths by State")
v2.circle("rank_density", "rank_deaths", source = source, size = 13, color = "red", alpha = 0.41)
v2.x_range.flipped = True
v2.y_range.flipped = True
v2.xaxis.axis_label = "Rank of Density"
v2.yaxis.axis_label = "Rank of Deaths"

v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Ranks of Density vs Death-Rate by State")
v3.circle("rank_density", "rank_death_rate", source = source, size = 13, color = "orange", alpha = 0.41)
v3.x_range.flipped = True
v3.y_range.flipped = True
v3.xaxis.axis_label = "Rank of Density"
v3.yaxis.axis_label = "Rank of Death Rate"

show(row(v1, v2, v3))


# Every circle represent a state's position based on its rank of density and rank of a Covid-19 metric (cases, deaths and death-rate for the three plots).
# 
# It's hard to conclude anything for density looking at these. These look fairly distributed, right?   
# At least at a state macro-level it doesn't seem to be particularly causing any effect on the spread of Covid-19. Not sure if that is a positive or negative.
# 
# It might be worth exploring the same at a more granular geography like city since some of the recent outbreaks in India have come from populated and dense cities like Delhi, Mumbai. Consider this as an open call for the search and availability of such a dataset.
# 

# ## April-1: Population
# ![population.jpg](attachment:population.jpg)
# 
# Since Covid-19 is an airborne disease and can easily spread from human to human in close contact, does this put populated areas at higher risk? Should we expect to see higher cases and fatalities in populated areas? These are intuitive assumptions.
# 
# Identifying how population affects Covid-19 can help authorities plan processes better in fighting back.
# 

# In[ ]:


df_covid = read_covid_data()
df_census = read_census_data()

df = df_covid.groupby("state")["cases", "deaths", "recoveries"].max().reset_index().merge(df_census[["state", "population"]], on = "state")
df["death_rate"] = df.deaths / (df.deaths + df.recoveries)
df["rank_population"] = df.population.rank(ascending = False)
df["rank_cases"] = df.cases.rank(ascending = False)
df["rank_deaths"] = df.deaths.rank(ascending = False)
df["rank_death_rate"] = df.death_rate.rank(ascending = False)


# In[ ]:


source = ColumnDataSource(data = dict(
    state = df.state.values,
    population = df.population.values / 1000000,
    cases = df.cases.values,
    deaths = df.deaths.values,
    death_rate = df.death_rate.values
))

tooltips_1 = [
    ("State", "@state"),
    ("Population", "@population{0.00} M"),
    ("Cases", "@cases")
]

tooltips_2 = [
    ("State", "@state"),
    ("Population", "@population{0.00} M"),
    ("Deaths", "@deaths")
]

tooltips_3 = [
    ("State", "@state"),
    ("Population", "@population{0.00} M"),
    ("Death Rate", "@death_rate{0.000}")
]

v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Population vs Cases by State")
v1.circle("population", "cases", source = source, size = 13, color = "blue", alpha = 0.41)
v1.xaxis.axis_label = "Population"
v1.yaxis.axis_label = "Cases"

v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Population vs Deaths by State")
v2.circle("population", "deaths", source = source, size = 13, color = "red", alpha = 0.41)
v2.xaxis.axis_label = "Population"
v2.yaxis.axis_label = "Deaths"

v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Population vs Death-Rate by State")
v3.circle("population", "death_rate", source = source, size = 13, color = "orange", alpha = 0.41)
v3.xaxis.axis_label = "Population"
v3.yaxis.axis_label = "Death Rate"

show(row(v1, v2, v3))


# Every circle represent a state's position based on its population and a Covid-19 metric (cases, deaths and death-rate for the three plots).   
# 
# Nothing very clear to take away from this. Instead we can look at the relative ranks of the population of states against the respective ranks in Covid-19 metrics.

# In[ ]:


source = ColumnDataSource(data = dict(
    state = df.state.values,
    rank_population = df.rank_population.values,
    rank_cases = df.rank_cases.values,
    rank_deaths = df.rank_deaths.values,
    rank_death_rate = df.rank_death_rate.values
))

tooltips_1 = [
    ("State", "@state"),
    ("Rank of Population", "@rank_population{0}"),
    ("Rank of Cases", "@rank_cases{0}")
]

tooltips_2 = [
    ("State", "@state"),
    ("Rank of Population", "@rank_population{0}"),
    ("Rank of Deaths", "@rank_deaths{0}")
]

tooltips_3 = [
    ("State", "@state"),
    ("Rank of Population", "@rank_population{0}"),
    ("Rank of Death Rate", "@rank_death_rate{0}")
]

v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Ranks of Population vs Cases by State")
v1.circle("rank_population", "rank_cases", source = source, size = 13, color = "blue", alpha = 0.41)
v1.x_range.flipped = True
v1.y_range.flipped = True
v1.xaxis.axis_label = "Rank of Density"
v1.yaxis.axis_label = "Rank of Cases"

v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Ranks of Population vs Deaths by State")
v2.circle("rank_population", "rank_deaths", source = source, size = 13, color = "red", alpha = 0.41)
v2.x_range.flipped = True
v2.y_range.flipped = True
v2.xaxis.axis_label = "Rank of Density"
v2.yaxis.axis_label = "Rank of Deaths"

v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Ranks of Population vs Death-Rate by State")
v3.circle("rank_population", "rank_death_rate", source = source, size = 13, color = "orange", alpha = 0.41)
v3.x_range.flipped = True
v3.y_range.flipped = True
v3.xaxis.axis_label = "Rank of Density"
v3.yaxis.axis_label = "Rank of Death Rate"

show(row(v1, v2, v3))


# Every circle represent a state's position based on its rank of population and rank of a Covid-19 metric (cases, deaths and death-rate for the three plots).
# 
# These first rank scatterplot shows that higher populated states have higher number of Covid-19 cases, and also to some extent the same with deaths and death-rate. Since the deaths are low in number (I pray it continues to be so), only time will tell how this impacts in the future and whether we will continue seeing such a pattern.
# 
# I'm not sure how useful population as a factor is going to be in the curbing of the disease. It is probably more useful in certain activities around the fight against the disease. Like using it to forecast cases and requirements of testing kits so that inventory is in control.
# 

# ## RIP Covid-19
# > ***Lets contribute in any way possible.   
# > Lets come together as a nation.   
# > Lets beat Covid-19 together.   
# > Lets help the world.***
