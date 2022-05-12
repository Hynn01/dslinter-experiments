#!/usr/bin/env python
# coding: utf-8

# # 1. The Lists of Data Table
# ### 1) Case Data
# - **Case**: Data of COVID-19 infection cases in South Korea
# 
# ### 2) Patient Data
# - **PatientInfo**: Epidemiological data of COVID-19 patients in South Korea
# - **PatientRoute**: Route data of COVID-19 patients in South Korea (currently unavailable)
# 
# ### 3) Time Series Data
# - **Time**: Time series data of COVID-19 status in South Korea
# - **TimeAge**: Time series data of COVID-19 status in terms of the age in South Korea
# - **TimeGender**: Time series data of COVID-19 status in terms of gender in South Korea
# - **TimeProvince**: Time series data of COVID-19 status in terms of the Province in South Korea
# 
# ### 4) Additional Data
# - **Region**: Location and statistical data of the regions in South Korea
# - **Weather**: Data of the weather in the regions of South Korea
# - **SearchTrend**: Trend data of the keywords searched in NAVER which is one of the largest portals in South Korea
# - **SeoulFloating**: Data of floating population in Seoul, South Korea (from SK Telecom Big Data Hub)
# - **Policy**: Data of the government policy for COVID-19 in South Korea

# # 2. The Structure of our Dataset
# - What color means is that they have similar properties.
# - If a line is connected between columns, it means that the values of the columns are partially shared.
# - The dotted lines mean weak relevance.
# ![db_0701](https://user-images.githubusercontent.com/50820635/86225695-8dca0580-bbc5-11ea-9e9b-b0ca33414d8a.PNG)

# # 3. The Detailed Description of each Data Table

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


path = '/kaggle/input/coronavirusdataset/'

case = p_info = pd.read_csv(path+'Case.csv')
p_info = pd.read_csv(path+'PatientInfo.csv')
#p_route = pd.read_csv(path+'PatientRoute.csv')
time = pd.read_csv(path+'Time.csv')
t_age = pd.read_csv(path+'TimeAge.csv')
t_gender = pd.read_csv(path+'TimeGender.csv')
t_provin = pd.read_csv(path+'TimeProvince.csv')
region = pd.read_csv(path+'Region.csv')
weather = pd.read_csv(path+'Weather.csv')
search = pd.read_csv(path+'SearchTrend.csv')
floating = pd.read_csv(path+'SeoulFloating.csv')
policy = pd.read_csv(path+'Policy.csv')


# ##### Before the Start..
# - We make a structured dataset based on the report materials of KCDC and local governments.
# - In Korea, we use the terms named '-do', '-si', '-gun' and '-gu',
# - The meaning of them are explained below.
# 
# ***
# 
# 
# ### Levels of administrative divisions in South Korea
# #### Upper Level (Provincial-level divisions)
# - **Special City**:
# *Seoul*
# - **Metropolitan City**:
# *Busan / Daegu / Daejeon / Gwangju / Incheon / Ulsan*
# - **Province(-do)**:
# *Gyeonggi-do / Gangwon-do / Chungcheongbuk-do / Chungcheongnam-do / Jeollabuk-do / Jeollanam-do / Gyeongsangbuk-do / Gyeongsangnam-do*
# 
# #### Lower Level (Municipal-level divisions)
# - **City(-si)**
# [List of cities in South Korea](https://en.wikipedia.org/wiki/List_of_cities_in_South_Korea)
# - **Country(-gun)**
# [List of counties of South Korea](https://en.wikipedia.org/wiki/List_of_counties_of_South_Korea)
# - **District(-gu)**
# [List of districts in South Korea](https://en.wikipedia.org/wiki/List_of_districts_in_South_Korea)
# 
# ***
# 
# <img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2815958%2F1c50702025f44b0c1ce92460bd2ea3f9%2Fus_hi_30-1.jpg?generation=1582819435038273&amp;alt=media" width=700>
# 
# ***
# 
# Sources
# - http://nationalatlas.ngii.go.kr/pages/page_1266.php
# - https://en.wikipedia.org/wiki/Administrative_divisions_of_South_Korea

# ### 1) Case
# #### Data of COVID-19 infection cases in South Korea
# 1. case_id: the ID of the infection case
#   > - case_id(7) = region_code(5) + case_number(2)  
#   > - You can check the region_code in 'Region.csv'
# - province: Special City / Metropolitan City / Province(-do)
# - city: City(-si) / Country (-gun) / District (-gu)
#   > - The value 'from other city' means that where the group infection started is other city.
# - group: TRUE: group infection / FALSE: not group
#   > - If the value is 'TRUE' in this column, the value of 'infection_cases' means the name of group.  
#   > - The values named 'contact with patient', 'overseas inflow' and 'etc' are not group infection. 
# - infection_case: the infection case (the name of group or other cases)
#   > - The value 'overseas inflow' means that the infection is from other country.  
#   > - The value 'etc' includes individual cases, cases where relevance classification is ongoing after investigation, and cases under investigation.
# - confirmed: the accumulated number of the confirmed
# - latitude: the latitude of the group (WGS84)
# - longitude: the longitude of the group (WGS84)
# 

# In[ ]:


case.head()


# ### 2) PatientInfo
# #### Epidemiological data of COVID-19 patients in South Korea
# 1. patient_id: the ID of the patient
#   > - patient_id(10) = region_code(5) + patient_number(5)
#   > - You can check the region_code in 'Region.csv'
#   > - There are two types of the patient_number  
#       1) local_num: The number given by the local government.  
#       2) global_num: The number given by the KCDC  
# - sex: the sex of the patient
# - age: the age of the patient
#   > - 0s: 0 ~ 9  
#   > - 10s: 10 ~ 19  
#   ...  
#   > - 90s: 90 ~ 99  
#   > - 100s: 100 ~ 109
# - country: the country of the patient
# - province: the province of the patient
# - city: the city of the patient
# - infection_case: the case of infection
# - infected_by: the ID of who infected the patient
#   > - This column refers to the  'patient_id' column. 
# - contact_number: the number of contacts with people
# - symptom_onset_date: the date of symptom onset
# - confirmed_date: the date of being confirmed
# - released_date: the date of being released
# - deceased_date: the date of being deceased
# - state: isolated / released / deceased
#   > - isolated: being isolated in the hospital
#   > - released: being released from the hospital
#   > - deceased: being deceased

# In[ ]:


p_info.head()


# ### 3) PatientRoute
# #### Route data of COVID-19 patients in South Korea
# - patient_id: the ID of the patient
# - date: YYYY-MM-DD
# - province: Special City / Metropolitan City / Province(-do)
# - city: City(-si) / Country (-gun) / District (-gu)
# - latitude: the latitude of the visit (WGS84)
# - longitude: the longitude of the visit (WGS84)

# In[ ]:


#p_route.head()


# ### 4) Time
# #### Time series data of COVID-19 status in South Korea
# - date: YYYY-MM-DD
# - time: Time (0 = AM 12:00 / 16 = PM 04:00)
#   > - The time for KCDC to open the information has been changed from PM 04:00 to AM 12:00 since March 2nd.
# - test: the accumulated number of tests
#   > - A test is a diagnosis of an infection.
# - negative: the accumulated number of negative results
# - confirmed: the accumulated number of positive results
# - released: the accumulated number of releases
# - deceased: the accumulated number of deceases

# In[ ]:


time.head()


# ### 5) TimeAge
# #### Time series data of COVID-19 status in terms of the age in South Korea
# - date: YYYY-MM-DD
#   > - The status in terms of the age has been presented since March 2nd.
# - time: Time
# - age: the age of patients
# - confirmed: the accumulated number of the confirmed
# - deceased: the accumulated number of the deceased

# In[ ]:


t_age.head()


# ### 6) TimeGender
# #### Time series data of COVID-19 status in terms of the gender in South Korea
# - date: YYYY-MM-DD
#   > - The status in terms of the gender has been presented since March 2nd.
# - time: Time
# - sex: the gender of patients
# - confirmed: the accumulated number of the confirmed
# - deceased: the accumulated number of the deceased

# In[ ]:


t_gender.head()


# ### 7) TimeProvince
# #### Time series data of COVID-19 status in terms of the Province in South Korea
# - date: YYYY-MM-DD
# - time: Time
# - province: the province of South Korea
# - confirmed: the accumulated number of the confirmed in the province
#   > - The confirmed status in terms of the provinces has been presented since Feburary 21th.
#   > - The value before Feburary 21th can be different.
# - released: the accumulated number of the released in the province
#   > - The confirmed status in terms of the provinces has been presented since March 5th.
#   > - The value before March 5th can be different.
# - deceased: the accumulated number of the deceased in the province
#   > - The confirmed status in terms of the provinces has been presented since March 5th.
#   > - The value before March 5th can be different.

# In[ ]:


t_provin.head()


# ### 8) Region
# #### Location and statistical data of the regions in South Korea
# - code: the code of the region
# - province: Special City / Metropolitan City / Province(-do)
# - city: City(-si) / Country (-gun) / District (-gu)
# - latitude: the latitude of the visit (WGS84)
# - longitude: the longitude of the visit (WGS84)
# - elementary_school_count: the number of elementary schools
# - kindergarten_count: the number of kindergartens
# - university_count: the number of universities
# - academy_ratio: the ratio of academies
# - elderly_population_ratio: the ratio of the elderly population
# - elderly_alone_ratio: the ratio of elderly households living alone
# - nursing_home_count: the number of nursing homes
# 
# Source of the statistic: [KOSTAT (Statistics Korea)](http://kosis.kr/)

# In[ ]:


region.head()


# ### 9) Weather
# #### Data of the weather in the regions of South Korea
# - code: the code of the region
# - province: Special City / Metropolitan City / Province(-do)
# - date: YYYY-MM-DD
# - avg_temp: the average temperature
# - min_temp: the lowest temperature
# - max_temp: the highest temperature
# - precipitation: the daily precipitation
# - max_wind_speed: the maximum wind speed
# - most_wind_direction: the most frequent wind direction
# - avg_relative_humidity: the average relative humidity
# 
# Source of the weather data: [KMA (Korea Meteorological Administration)](http://data.kma.go.kr)

# In[ ]:


weather.head()


# ### 10) SearchTrend
# #### Trend data of the keywords searched in NAVER which is one of the largest portal in South Korea
# - date: YYYY-MM-DD
# - cold: the search volume of 'cold' in Korean language
#   > - The unit means relative value by setting the highest search volume in the period to 100.
# - flu: the search volume of 'flu' in Korean language
#   > - Same as above.
# - pneumonia: the search volume of 'pneumonia' in Korean language
#   > - Same as above.
# - coronavirus: the search volume of 'coronavirus' in Korean language
#   > - Same as above.
# 
# 
# Source of the data: [NAVER DataLab](https://datalab.naver.com/)

# In[ ]:


search.head()


# ### 11) SeoulFloating
# #### Data of floating population in Seoul, South Korea (from SK Telecom Big Data Hub)
# 
# - date: YYYY-MM-DD
# - hour: Hour
# - birth_year: the birth year of the floating population
# - sext: he sex of the floating population
# - province: Special City / Metropolitan City / Province(-do)
# - city: City(-si) / Country (-gun) / District (-gu)
# - fp_num: the number of floating population
# 
# Source of the data: [SKT Big Data Hub](https://www.bigdatahub.co.kr)

# In[ ]:


floating.head()


# ### 12) Policy
# #### Data of the government policy for COVID-19 in South Korea
# 
# - policy_id: the ID of the policy
# - country: the country that implemented the policy
# - type: the type of the policy
# - gov_policy: the policy of the government
# - detail: the detail of the policy
# - start_date: the start date of the policy
# - end_date: the end date of the policy

# In[ ]:


policy.head()


# In[ ]:




