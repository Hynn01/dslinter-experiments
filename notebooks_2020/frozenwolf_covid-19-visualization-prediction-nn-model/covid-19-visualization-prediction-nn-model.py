#!/usr/bin/env python
# coding: utf-8

# # **COVID-19 DATASET VISUALIZATION & PREDICTION **

# ## **Before starting the notebook**
# 
# 
# ### **Protecting yourself and others from the spread COVID-19 by taking some simple precautions:**
# 
#    Regularly and thoroughly clean your hands with an alcohol-based hand rub or wash them with soap and water. Why? Washing your hands with soap and water or using alcohol-based hand rub kills viruses that may be on your hands.
#    
# 
# <img src="https://www.cdc.gov/handwashing/images/campaign2018/GHD-UVLight-1080x1080.gif" width="600px" height="600px">
#    
#    
#    Maintain at least 2 metre (6 feet) distance between yourself and others. Why? When someone coughs, sneezes, or speaks they spray small liquid droplets from their nose or mouth which may contain virus. If you are too close, you can breathe in the droplets, including the COVID-19 virus if the person has the disease.
#    
#   
# <img src="https://d1nakyqvxb9v71.cloudfront.net/wp-content/uploads/2020/03/Social-no-dots-3-27.gif" width="650px" height="650px">
# 
#     
#    Avoid going to crowded places. Why? Where people come together in crowds, you are more likely to come into close contact with someone that has COIVD-19 and it is more difficult to maintain physical distance of 2 metre (6 feet).
#    
#    Avoid touching eyes, nose and mouth. Why? Hands touch many surfaces and can pick up viruses. Once contaminated, hands can transfer the virus to your eyes, nose or mouth. From there, the virus can enter your body and infect you.
#    
#    <img src="https://i.pinimg.com/originals/5d/d2/a8/5dd2a87cd2a8199e1b13630bdbd17ef1.gif" width="550px" height="550px">
#    
#    
#    Make sure you, and the people around you, follow good respiratory hygiene. This means covering your mouth and nose with your bent elbow or tissue when you cough or sneeze. Then dispose of the used tissue immediately and wash your hands. Why? Droplets spread virus. By following good respiratory hygiene, you protect the people around you from viruses such as cold, flu and COVID-19.
#    
#    Stay home and self-isolate even with minor symptoms such as cough, headache, mild fever, until you recover. Have someone bring you supplies. If you need to leave your house, wear a mask to avoid infecting others. Why? Avoiding contact with others will protect them from possible COVID-19 and other viruses.
#    
#    <img src="https://www.verywellhealth.com/thmb/FXwOsscJpzls46TQPuazu6pkTqA=/400x250/filters:no_upscale():max_bytes(150000):strip_icc()/what-scientists-know-about-covid-19-4800685_final-94a4fe120b4f416a98ee8b134416c939.gif" width="550px" height="550px">
#    
#    If you have a fever, cough and difficulty breathing, seek medical attention, but call by telephone in advance if possible and follow the directions of your local health authority. Why? National and local authorities will have the most up to date information on the situation in your area. Calling in advance will allow your health care provider to quickly direct you to the right health facility. This will also protect you and help prevent spread of viruses and other infections.
# 

# ## What is COVID-19 ?
# 
# Coronaviruses are a large family of viruses which may cause illness in animals or humans.  In humans, several coronaviruses are known to cause respiratory infections ranging from the common cold to more severe diseases such as Middle East Respiratory Syndrome (MERS) and Severe Acute Respiratory Syndrome (SARS). The most recently discovered coronavirus causes coronavirus disease COVID-19.
# 
# ## How does it spread ?
# 
# The virus that causes COVID-19 is mainly transmitted through droplets generated when an infected person coughs, sneezes, or exhales. These droplets are too heavy to hang in the air, and quickly fall on floors or surfaces.
# You can be infected by breathing in the virus if you are within close proximity of someone who has COVID-19, or by touching a contaminated surface and then your eyes, nose or mouth.
# 
# ## What are the symptoms of COVID-19? 
# 
# The most common symptoms of COVID-19 are fever, dry cough, and tiredness. Some patients may have aches and pains, nasal congestion, sore throat or diarrhea. These symptoms are usually mild and begin gradually. Some people become infected but only have very mild symptoms. Most people (about 80%) recover from the disease without needing hospital treatment. Around 1 out of every 5 people who gets COVID-19 becomes seriously ill and develops difficulty breathing. Older people, and those with underlying medical problems like high blood pressure, heart and lung problems, diabetes, or cancer , are at higher risk of developing serious illness. However anyone can catch COVID-19 and become seriously ill. Even people with very mild symptoms of COVID-19 can transmit the virus. People of all ages who experience fever, cough and difficulty breathing should seek medical attention.
# 
# ## What is the curve?
# 
# <img src="https://cdn.mos.cms.futurecdn.net/wMJBgLczxnQQBsEC9QBmuW-970-80.gif" width="750px" height="750px">
# 
# 
# The "curve" researchers are talking about refers to the projected number of people who will contract COVID-19 over a period of time. (To be clear, this is not a hard prediction of how many people will definitely be infected, but a theoretical number that's used to model the virus' spread.)
# 
# The curve takes on different shapes, depending on the virus's infection rate. It could be a steep curve, in which the virus spreads exponentially (that is, case counts keep doubling at a consistent rate), and the total number of cases skyrockets to its peak within a few weeks. Infection curves with a steep rise also have a steep fall; after the virus infects pretty much everyone who can be infected, case numbers begin to drop exponentially, too. 
# 
# ## How do we flatten the curve?
# 
#    <img src="https://upload.wikimedia.org/wikipedia/commons/c/c5/Covid-19-curves-graphic-social-v3.gif" width="750px" height="750px">
#    
# As there is currently no vaccine or specific medication to treat COVID-19, and because testing is so limited in the U.S., the only way to flatten the curve is through collective action. The U.S. Centers for Disease Control and Prevention (CDC) has recommended that all Americans wash their hands frequently, self-isolate when they're sick or suspect they might be, and start "social distancing" (essentially, avoiding other people whenever possible) right away. 
# 
# 

# ## Helpful Resources:
# The Coronavirus Explained & What You Should Do: https://youtu.be/BtN-goy9VOY 
# 
# Coronavirus Curves and Different Outcomes:      https://statisticsbyjim.com/basics/coronavirus/
# 

# ## Objective of the Notebook:
# 
# Objective of this notebook is to study COVID-19 outbreak with the help of some basic visualizations techniques. We will be prediciting the death rate and recovery rate with the help of neural network consisting of Convolution layers.We will compare the graphs and analyse them for different countries.We will try to find the peak and try to explain the reason for the outcome.
# 

# In[ ]:


# Importing libraries for visualization and reading the data

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
from IPython.display import HTML
sns.set_style("darkgrid") # You can choose any style among these : darkgrid, whitegrid, dark, white, ticks


# In[ ]:


os.listdir('../input/novel-corona-virus-2019-dataset') # Datasets is stored in ../input/


# In[ ]:


# Reading the csv files with pandas
deaths_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")
recoverd_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
confirmed_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
us_confirmed_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed_US.csv")
us_death_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths_US.csv")


# # **The Dataset**
# The dataset has information related to death , recovery of patient and new confirmed cases of each day of each 185 countries.It also has these information for some provinces.

# ### **Death in each Country**

# In[ ]:


deaths_data


# ### **Recovery in each Country**

# In[ ]:


recoverd_data


# ### **Cases Confirmed in each Country**

# In[ ]:


confirmed_data


# ### **Cases Confirmed and Deaths in each provinces of US**

# In[ ]:


us_confirmed_data


# In[ ]:


us_death_data


# # **Visualization of Dataset**

# In[ ]:


countries_death = deaths_data['Country/Region']
countries_cured = recoverd_data['Country/Region']
countries_confirmed = confirmed_data['Country/Region']
us_provinces_confirmed = us_confirmed_data['Province_State']
us_provinces_death = us_confirmed_data['Province_State']


# In[ ]:


unique_countries = [] 
for i in countries_death:
    if i  not in unique_countries:
        unique_countries.append(i)
        
print("DATASET CONTAINS INFORMATION ABOUT ",len(unique_countries)," COUNTRIES")


# In[ ]:


unique_provinces = []
for i in us_provinces_death:
    if i not in unique_provinces:
        unique_provinces.append(i)
        
print("DATASET CONTAINS INFORMATION ABOUT ",len(unique_provinces)," PROVINCES")


# In[ ]:


dates = list(deaths_data.keys())[4:]
dates_us_provinces = list(us_death_data.keys())[12:]


# In[ ]:


def get_data(name_of_country, datatype = 'death'): # Defining a function to get data based on country name
    
    if datatype == 'death':
        country_index = []
        for i in range(len(countries_death)):
            if countries_death[int(i)] == name_of_country:
                country_index.append(int(i))     

        data = np.zeros(len(dates))
        for i in country_index:
            temp = []
            for each_date_index in range(len(dates)):
                temp.append(deaths_data[dates[each_date_index]][i])
            data = data + np.asarray(temp)
        
        return data
    
    if datatype == 'recovered':
        country_index = []
        for i in range(len(countries_cured)):
            if countries_cured[int(i)] == name_of_country:
                country_index.append(int(i))     

        data = np.zeros(len(dates))
        for i in country_index:
            temp = []
            for each_date_index in range(len(dates)):
                temp.append(recoverd_data[dates[each_date_index]][i])
            data = data + np.asarray(temp)
        
        return data
    
    if datatype == 'confirmed':
        country_index = []
        for i in range(len(countries_confirmed)):
            if countries_confirmed[i] == name_of_country:
                country_index.append(i)     

        data = np.zeros(len(dates))
        for i in country_index:
            temp = []
            for each_date_index in range(len(dates)):
                temp.append(confirmed_data[dates[int(each_date_index)]][int(i)])
            data = data + np.asarray(temp)
        
        return data
        


# In[ ]:


def get_data_provinces(name_of_provinces, datatype = 'death'): # Defining a function to get data based on provinces name
    
    if datatype == 'death':
        country_index = []
        for i in range(len(us_provinces_death)):
            if us_provinces_death[int(i)] == name_of_provinces:
                country_index.append(int(i))     

        data = np.zeros(len(dates_us_provinces))
        for i in country_index:
            temp = []
            for each_date_index in range(len(dates_us_provinces)):
                temp.append(us_death_data[dates_us_provinces[each_date_index]][i])
            data = data + np.asarray(temp)
        
        return data
    

    
    if datatype == 'confirmed':
        country_index = []
        for i in range(len(us_provinces_death)):
            if us_provinces_confirmed[i] == name_of_provinces:
                country_index.append(i)     

        data = np.zeros(len(dates_us_provinces))
        for i in country_index:
            temp = []
            for each_date_index in range(len(dates_us_provinces)):
                temp.append(us_confirmed_data[dates_us_provinces[int(each_date_index)]][int(i)])
            data = data + np.asarray(temp)
        
        return data
        


# ### **Effect of COVID-19 at a Global Scale**

# In[ ]:


plt.rc('figure', figsize=(15, 7)) # Setting graph size
death_global = np.zeros(len(dates))
recovered_global = np.zeros(len(dates))
confirmed_global = np.zeros(len(dates))

for country_names in unique_countries:
    # Adding the data
    death_global = get_data(country_names,'death') + death_global
    recovered_global = get_data(country_names,'recovered') + recovered_global
    confirmed_global = get_data(country_names,'confirmed') + confirmed_global
    
# Plotting the graph    
sns.lineplot(range(len(dates)),death_global , label = 'death')
sns.lineplot(range(len(dates)),recovered_global , label = 'recovered')
sns.lineplot(range(len(dates)),confirmed_global , label = 'confirmed')
    
# Shding the area
plt.fill_between(range(len(dates)),confirmed_global, color="b", alpha=0.4)   
plt.fill_between(range(len(dates)),recovered_global, color="g", alpha=0.5)
plt.fill_between(range(len(dates)),death_global, color="r", alpha=0.5)

death_global = list(death_global)
recovered_global = list(recovered_global)
confirmed_global = list(confirmed_global)

max_death = max(death_global)
date_of_max_death = death_global.index(max(death_global))
        
max_recovery = max(recovered_global)
date_of_max_recovery = recovered_global.index(max(recovered_global))
        
max_confirmation = max(confirmed_global)
date_of_max_confirmation = confirmed_global.index(max(confirmed_global))
 
# Highlighting the point at maxima
plt.scatter(date_of_max_death,max_death,color = 'r')
plt.scatter(date_of_max_recovery,max_recovery,color = 'g')
plt.scatter(date_of_max_confirmation,max_confirmation,color = 'b')

# Plotting value at maxima
plt.text(date_of_max_death, max_death,str(int(max_death)) , fontsize=12 , color = 'r')
plt.text(date_of_max_recovery, max_recovery,str(int(max_recovery)) , fontsize=12 , color = 'g')
plt.text(date_of_max_confirmation, max_confirmation,str(int(max_confirmation)) , fontsize=12 , color = 'b')

        
plt.legend(loc=2)
plt.title("Global Count")    
plt.xlabel("Time")
plt.ylabel("Count")

plt.show()

print("TOTAL DEATHS: ",sum(death_global))
print("TOTAL RECOVERED PATIENTS: ",sum(recovered_global))
print("TOTAL CONFIRMED CASES: ",sum(confirmed_global))


# In[ ]:


plt.rc('figure', figsize=(15, 7))
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])


ax.bar(range(len(dates)),confirmed_global , label = 'confirmed',color = 'b')
ax.bar(range(len(dates)),recovered_global , label = 'recovered' , color = 'g')
ax.bar(range(len(dates)),death_global , label = 'death' , color = 'r')


ax.set_xlabel("Time")
ax.set_ylabel("Count")
ax.set_title('Global Count')
plt.legend(loc = 2)
plt.show()


# ### **Effect of COVID-19 on different countries**

# In[ ]:


plt.rc('figure', figsize=(15, 7))

for country_names in unique_countries:
    sns.lineplot(range(len(dates)),get_data(country_names,'death'))

        
plt.title("Time vs Death")    
plt.xlabel("Time")
plt.ylabel("Deaths")

plt.show()


# In[ ]:


plt.rc('figure', figsize=(15, 7))

for country_names in unique_countries:
    sns.lineplot(range(len(dates)),get_data(country_names,'recovered'))


        
plt.title("Time vs Recovery")    
plt.xlabel("Time")
plt.ylabel("Recovered patients")

plt.show()


# In[ ]:


plt.rc('figure', figsize=(15, 7))

for country_names in unique_countries:
    sns.lineplot(range(len(dates)),get_data(country_names,'confirmed'))


        
plt.title("Time vs Confirmed Cases")    
plt.xlabel("Time")
plt.ylabel("Confirmed")

plt.show()


# # **Analyising effect of COVID-19 on top 10 countries whose death rate is high **

# In[ ]:


total_death = []
total_confirmed = []
total_recovered = []

from more_itertools import sort_together


for country_names in unique_countries:
    total_death.append(int(sum(get_data(country_names,'death'))))
    
for country_names in unique_countries:
    total_confirmed.append(int(sum(get_data(country_names , 'confirmed'))))

for country_names in unique_countries:
    total_recovered.append(int(sum(get_data(country_names , 'recovered'))))
    
# Sorting the values based on number of deaths , recovery ,confirmed cases

total_death_sorted , countries_sorted_death = tuple(sort_together([total_death,unique_countries]))
total_recovered_sorted , countries_sorted_recovered = tuple(sort_together([total_recovered,unique_countries]))
total_confirmed_sorted , countries_sorted_confirmed = tuple(sort_together([total_confirmed,unique_countries]))

print("COUNTRIES BASED ON NUMBER OF DEATHS:")
for i in countries_sorted_death[:-11:-1]:
    print(" "+i)

print("----------------------------------------")

print("COUNTRIES BASED ON NUMBER OF PATIENTS RECOVERED:")
for i in countries_sorted_recovered[:-11:-1]:
    print(" "+i)
    
print("----------------------------------------")

print("COUNTRIES BASED ON NUMBER OF CONFIRMED CASES:")
for i in countries_sorted_confirmed[:-11:-1]:
    print(" "+i)
    
print("----------------------------------------")


# ### **Each country's death rate**

# In[ ]:


death_each_country = pd.DataFrame(zip(countries_sorted_death,total_death_sorted) , columns = ['Country','Death'])
recovered_each_country = pd.DataFrame(zip(countries_sorted_recovered,total_recovered_sorted) , columns = ['Country','Recovered'])
confirmed_each_country = pd.DataFrame(zip(countries_sorted_confirmed,total_confirmed_sorted) , columns = ['Country','Confirmed'])

display(death_each_country[:-11:-1])

for country_names in countries_sorted_death[:-11:-1]:
    sns.lineplot(range(len(dates)),get_data(country_names,'death'),label = country_names)
        
plt.xlabel("Time")
plt.ylabel("Count")
plt.title('Death')
plt.legend(loc = 2)
plt.show()

# Plotting pie chart
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#77d8d8','#4cbbb9','#efa8e4','#f2ed6f','#fb7b6b','#d7385e','#be79df'] # Choosing light colors
labels = list(countries_sorted_death[:-11:-1]) # Taking only 10 countries
labels.append('Others') 
sizes = list(total_death_sorted[:-11:-1])
sizes.append(sum(total_death_sorted[:-11])) #Adding remaining values
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
         startangle=90 , colors = colors)
plt.legend(loc = 2)
plt.title('Death Percentage')
ax1.axis('equal')  
plt.tight_layout()
plt.show()


# ### **Each country's recovered patients**

# In[ ]:


display(recovered_each_country[:-11:-1])

for country_names in countries_sorted_recovered[:-11:-1]:
    sns.lineplot(range(len(dates)),get_data(country_names,'recovered'),label = country_names)
        
plt.xlabel("Time")
plt.ylabel("Count")
plt.title('Recovered')
plt.legend(loc = 2)
plt.show()


colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#77d8d8','#4cbbb9','#efa8e4','#f2ed6f','#fb7b6b','#d7385e','#be79df']
labels = list(countries_sorted_recovered[:-11:-1])
labels.append('Others')
sizes = list(total_recovered_sorted[:-11:-1])
sizes.append(sum(total_recovered_sorted[:-11]))
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
         startangle=90 , colors = colors)
plt.legend(loc = 2)
plt.title('Recovered Percentage')
ax1.axis('equal')  
plt.tight_layout()
plt.show()


# In[ ]:


display(confirmed_each_country[:-11:-1])


for country_names in countries_sorted_confirmed[:-11:-1]:
    sns.lineplot(range(len(dates)),get_data(country_names,'confirmed'),label = country_names)

  
plt.xlabel("Time")
plt.ylabel("Count")
plt.title('Confirmed')
plt.legend(loc = 2)
plt.show()


colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#77d8d8','#4cbbb9','#efa8e4','#f2ed6f','#fb7b6b','#d7385e','#be79df']
labels = list(countries_sorted_confirmed[:-11:-1])
labels.append('Others')
sizes = list(total_confirmed_sorted[:-11:-1])
sizes.append(sum(total_confirmed_sorted[:-11]))
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
         startangle=90 , colors = colors)
plt.legend(loc = 2)
plt.title('Confirmed Percentage')
ax1.axis('equal')  
plt.tight_layout()
plt.show()


# ### **Each country's confirmed cases**

# In[ ]:


display(confirmed_each_country[:-11:-1])


for country_names in countries_sorted_confirmed[:-11:-1]:
    sns.lineplot(range(len(dates)),get_data(country_names,'confirmed'),label = country_names)

  
plt.xlabel("Time")
plt.ylabel("Count")
plt.title('Confirmed')
plt.legend(loc = 2)
plt.show()


colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#77d8d8','#4cbbb9','#efa8e4','#f2ed6f','#fb7b6b','#d7385e','#be79df']
labels = list(countries_sorted_confirmed[:-11:-1])
labels.append('Others')
sizes = list(total_confirmed_sorted[:-11:-1])
sizes.append(sum(total_confirmed_sorted[:-11]))
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
         startangle=90 , colors = colors)
plt.legend(loc = 2)
plt.title('Confirmed Percentage')
ax1.axis('equal')  
plt.tight_layout()
plt.show()


# ### **Details of all countries**

# In[ ]:


all_countries_data = pd.DataFrame(list(zip(unique_countries,total_death,total_recovered,total_confirmed)),columns = ['Country','Death','Recovered','Confirmed'])


# In[ ]:


# Displaying a table with values highlighted relative to other values with help of gradients

death_color = sns.light_palette("red", as_cmap=True) # Choosing gradient color pallete type
recovered_color = sns.light_palette("green", as_cmap=True)
confirmed_color = sns.light_palette("blue", as_cmap=True)

(all_countries_data.style
  .background_gradient(cmap=death_color, subset=['Death']) # Applying gradient
  .background_gradient(cmap=recovered_color, subset=['Recovered'])
  .background_gradient(cmap=confirmed_color, subset=['Confirmed'])

)


# # **Analysing Individual Countries**

# In[ ]:


def country_info(name,graph_type): # Defining function to plot the graph for a specific country
    
    if graph_type == 'bar':
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
 
        ax.bar(dates,get_data(name , 'confirmed'),label = 'confirmed' , color = 'b')
        ax.bar(dates,get_data(name , 'recovered'),label = 'recovered' , color = 'g')
        ax.bar(dates,get_data(name,'death'),label = 'Death' , color = 'r')


        ax.set_xticklabels([])
        ax.set_xlabel("Date")
        ax.set_ylabel("Count")
        ax.set_title(name)
        plt.legend(loc = 2)
        plt.show()
        
    if graph_type == 'line':
        sns.lineplot(range(len(dates)),get_data(name,'confirmed'),label = 'confirmed',color = 'b')
        sns.lineplot(range(len(dates)),get_data(name,'recovered'),label = 'recovered' , color = 'g')
        sns.lineplot(range(len(dates)),get_data(name,'death'),label = 'death' , color = 'r')
        
        plt.fill_between(range(len(dates)),get_data(name,'confirmed'), color="b", alpha=0.2)
        plt.fill_between(range(len(dates)),get_data(name,'recovered'), color="g", alpha=0.2)
        plt.fill_between(range(len(dates)),get_data(name,'death'), color="r", alpha=0.2)
        
        max_death = max(list(get_data(name,'death')))
        date_of_max_death = list(get_data(name,'death')).index(max(list(get_data(name,'death'))))
        
        max_recovery = max(list(get_data(name,'recovered')))
        date_of_max_recovery = list(get_data(name,'recovered')).index(max(list(get_data(name,'recovered'))))
        
        max_confirmation = max(list(get_data(name,'confirmed')))
        date_of_max_confirmation = list(get_data(name,'confirmed')).index(max(list(get_data(name,'confirmed'))))
        
        plt.scatter(date_of_max_death,max_death,color = 'r')
        plt.scatter(date_of_max_recovery,max_recovery,color = 'g')
        plt.scatter(date_of_max_confirmation,max_confirmation,color = 'b')
        
        plt.text(date_of_max_death, max_death,str(int(max_death)) , fontsize=12 , color = 'r')
        plt.text(date_of_max_recovery, max_recovery,str(int(max_recovery)) , fontsize=12 , color = 'g')
        plt.text(date_of_max_confirmation, max_confirmation,str(int(max_confirmation)) , fontsize=12 , color = 'b')
        
        plt.xlabel("Time")
        plt.ylabel("Count")
        plt.title(name)
        plt.legend(loc = 2)
        plt.show()

        


# ### **US**

# In[ ]:


country_info('US','line')
country_info('US','bar')



print("TOTAL DEATH IN US:",int(sum(get_data('US','death'))))
print("TOTAL PATIENTS RECOVERED IN US:",int(sum(get_data('US','recovered'))))
print("TOTAL CONFIRMED CASES IN US:",int(sum(get_data('US','confirmed'))))


# ### **CHINA**

# In[ ]:


country_info('China','line')
country_info('China','bar')

print("TOTAL DEATH IN China:",int(sum(get_data('China','death'))))
print("TOTAL PATIENTS RECOVERED IN China:",int(sum(get_data('China','recovered'))))
print("TOTAL CONFIRMED CASES IN China:",int(sum(get_data('China','confirmed'))))


# ### **ITALY**

# In[ ]:


country_info('Italy','line')
country_info('Italy','bar')

print("TOTAL DEATH IN Italy:",int(sum(get_data('Italy','death'))))
print("TOTAL PATIENTS RECOVERED IN Italy:",int(sum(get_data('Italy','recovered'))))
print("TOTAL CONFIRMED CASES IN Italy:",int(sum(get_data('Italy','confirmed'))))


# ### **SPAIN**

# In[ ]:


country_info('Spain','line')
country_info('Spain','bar')

print("TOTAL DEATH IN Spain:",int(sum(get_data('Spain','death'))))
print("TOTAL PATIENTS RECOVERED IN Spain:",int(sum(get_data('Spain','recovered'))))
print("TOTAL CONFIRMED CASES IN Spain:",int(sum(get_data('Spain','confirmed'))))


# ### **GERMANY**

# In[ ]:


country_info('Germany','line')
country_info('Germany','bar')

print("TOTAL DEATH IN Germany:",int(sum(get_data('Germany','death'))))
print("TOTAL PATIENTS RECOVERED IN Germany:",int(sum(get_data('Germany','recovered'))))
print("TOTAL CONFIRMED CASES IN Germany:",int(sum(get_data('Germany','confirmed'))))


# ### **FRANCE**

# In[ ]:


country_info('France','line')
country_info('France','bar')

print("TOTAL DEATH IN France:",int(sum(get_data('France','death'))))
print("TOTAL PATIENTS RECOVERED IN France:",int(sum(get_data('France','recovered'))))
print("TOTAL CONFIRMED CASES IN France:",int(sum(get_data('France','confirmed'))))


# ### **UNITED KINGDOM**

# In[ ]:


country_info('United Kingdom','line')
country_info('United Kingdom','bar')

print("TOTAL DEATH IN United Kingdom:",int(sum(get_data('United Kingdom','death'))))
print("TOTAL PATIENTS RECOVERED IN United Kingdom:",int(sum(get_data('United Kingdom','recovered'))))
print("TOTAL CONFIRMED CASES IN United Kingdom:",int(sum(get_data('United Kingdom','confirmed'))))


# ### **IRAN** 

# In[ ]:


country_info('Iran','line')
country_info('Iran','bar')

print("TOTAL DEATH IN Iran:",int(sum(get_data('Iran','death'))))
print("TOTAL PATIENTS RECOVERED IN Iran:",int(sum(get_data('Iran','recovered'))))
print("TOTAL CONFIRMED CASES IN Iran:",int(sum(get_data('Iran','confirmed'))))


# ### **TURKEY**

# In[ ]:


country_info('Turkey','line')
country_info('Turkey','bar')

print("TOTAL DEATH IN Turkey:",int(sum(get_data('Turkey','death'))))
print("TOTAL PATIENTS RECOVERED IN Turkey:",int(sum(get_data('Turkey','recovered'))))
print("TOTAL CONFIRMED CASES IN Turkey:",int(sum(get_data('Turkey','confirmed'))))


# ### **BELGIUM**

# In[ ]:


country_info('Belgium','line')
country_info('Belgium','bar')

print("TOTAL DEATH IN Belgium:",int(sum(get_data('Belgium','death'))))
print("TOTAL PATIENTS RECOVERED IN Belgium:",int(sum(get_data('Belgium','recovered'))))
print("TOTAL CONFIRMED CASES IN Belgium:",int(sum(get_data('Belgium','confirmed'))))


# ### **INDIA**

# In[ ]:


country_info('India','line')
country_info('India','bar')

print("TOTAL DEATH IN India:",int(sum(get_data('India','death'))))
print("TOTAL PATIENTS RECOVERED IN India:",int(sum(get_data('India','recovered'))))
print("TOTAL CONFIRMED CASES IN India:",int(sum(get_data('India','confirmed'))))


# ### Details about each provinces in US

# In[ ]:


plt.rc('figure', figsize=(15, 7))

for country_names in unique_provinces:
    sns.lineplot(range(len(dates_us_provinces)),get_data_provinces(country_names,'confirmed'))


        
plt.title("Time vs Confirmed")    
plt.xlabel("Time")
plt.ylabel("Confirmed count")

plt.show()


# In[ ]:


plt.rc('figure', figsize=(15, 7))

for country_names in unique_provinces:
    sns.lineplot(range(len(dates_us_provinces)),get_data_provinces(country_names,'death'))

        
plt.title("Time vs Death")    
plt.xlabel("Time")
plt.ylabel("Deaths")

plt.show()


# In[ ]:


total_death = []
total_confirmed = []
total_recovered = []

from more_itertools import sort_together


for country_names in unique_provinces:
    total_death.append(int(sum(get_data_provinces(country_names,'death'))))
    
for country_names in unique_provinces:
    total_confirmed.append(int(sum(get_data_provinces(country_names , 'confirmed'))))

    
# Sorting the values based on number of deaths , recovery ,confirmed cases

total_provinces_death_sorted , provinces_sorted_death = tuple(sort_together([total_death,unique_provinces]))
total_provinces_confirmed_sorted , provinces_sorted_confirmed = tuple(sort_together([total_confirmed,unique_provinces]))

print("PROVINCES BASED ON NUMBER OF DEATHS:")
for i in provinces_sorted_death[:-11:-1]:
    print(" "+i)

print("----------------------------------------")



print("PROVINCES BASED ON NUMBER OF CONFIRMED CASES:")
for i in provinces_sorted_confirmed[:-11:-1]:
    print(" "+i)
    
print("----------------------------------------")


# In[ ]:


provinces_sorted_death = pd.DataFrame(zip(provinces_sorted_death,total_provinces_death_sorted) , columns = ['Provinces','Death'])
provinces_sorted_confirmed = pd.DataFrame(zip(provinces_sorted_confirmed,total_provinces_confirmed_sorted) , columns = ['Provinces','Confirmed'])


# In[ ]:


display(provinces_sorted_death[:-11:-1])

for country_names in provinces_sorted_death['Provinces'][:-11:-1]:
    sns.lineplot(range(len(dates_us_provinces)),get_data_provinces(country_names,'death'),label = country_names)
        
plt.xlabel("Time")
plt.ylabel("Count")
plt.title('Deaths')
plt.legend(loc = 2)
plt.show()


colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#77d8d8','#4cbbb9','#efa8e4','#f2ed6f','#fb7b6b','#d7385e','#be79df']
labels = list(provinces_sorted_death['Provinces'][:-11:-1])
labels.append('Others')
sizes = list(total_provinces_death_sorted[:-11:-1])
sizes.append(sum(total_provinces_death_sorted[:-11]))
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
         startangle=90 , colors = colors)
plt.legend(loc = 2)
plt.title('Deaths Percentage')
ax1.axis('equal')  
plt.tight_layout()
plt.show()


# In[ ]:


display(provinces_sorted_confirmed[:-11:-1])

for country_names in provinces_sorted_confirmed['Provinces'][:-11:-1]:
    sns.lineplot(range(len(dates_us_provinces)),get_data_provinces(country_names,'confirmed'),label = country_names)
        
plt.xlabel("Time")
plt.ylabel("Count")
plt.title('Confirmed Cases')
plt.legend(loc = 2)
plt.show()


colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#77d8d8','#4cbbb9','#efa8e4','#f2ed6f','#fb7b6b','#d7385e','#be79df']
labels = list(provinces_sorted_confirmed['Provinces'][:-11:-1])
labels.append('Others')
sizes = list(total_provinces_confirmed_sorted[:-11:-1])
sizes.append(sum(total_provinces_confirmed_sorted[:-11]))
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
         startangle=90 , colors = colors)
plt.legend(loc = 2)
plt.title('Confirmed Percentage')
ax1.axis('equal')  
plt.tight_layout()
plt.show()


# In[ ]:


# Displaying a table with values highlighted relative to other values with help of gradients
all_countries_data = pd.DataFrame(list(zip(unique_provinces,total_death,total_confirmed)),columns = ['Provinces','Death','Confirmed'])

death_color = sns.light_palette("red", as_cmap=True) # Choosing gradient color pallete type
confirmed_color = sns.light_palette("blue", as_cmap=True)

(all_countries_data.style
  .background_gradient(cmap=death_color, subset=['Death']) # Applying gradient
  .background_gradient(cmap=confirmed_color, subset=['Confirmed'])

)


# # Model

# ## PREDICITING THE NUMBER OF DEATHS,RECOVERED PATIENTS & CONFIRMED PATIENTS
# 
# We will be a Neural network to predict the number of deaths and other details in future.
# Time series prediction problems are a difficult type of predictive modeling problem.
# Unlike regression predictive modeling, time series also adds the complexity of a sequence dependence among the input variables.
# Here we will be using Conv1D and MaxPooling1D to make a neural network.
# 
# <img src="https://miro.medium.com/max/1400/1*iJyzEak-RGfpcBC9v-8oAg.png" width="750px" height="750px">
# 

# ## Convolution in Convolutional Neural Networks
# 
# The convolutional neural network, or CNN for short, is a specialized type of neural network model designed for working with two-dimensional image data, although they can be used with one-dimensional and three-dimensional data.
# 
# Central to the convolutional neural network is the convolutional layer that gives the network its name. This layer performs an operation called a “convolution“.
# 
# In the context of a convolutional neural network, a convolution is a linear operation that involves the multiplication of a set of weights with the input, much like a traditional neural network. Given that the technique was designed for two-dimensional input, the multiplication is performed between an array of input data and a two-dimensional array of weights, called a filter or a kernel.
# 
# <img src="https://miro.medium.com/max/790/1*nYf_cUIHFEWU1JXGwnz-Ig.gif" width="250px" height="250px">

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv1D,MaxPooling1D,Dropout
INPUT_SIZE = 10
TARGET_SIZE = 1
PADDING = 40 # Most of the countries didn't have any confirmed cases in first 40 days (approx)
COUNTRIES_N = 50


# ### Reshaping the data

# In[ ]:


X = []
Y=[]

for country_name in countries_sorted_death[:-1*COUNTRIES_N-1:-1]:

    a = get_data(country_name,'death')[PADDING:]
    b = get_data(country_name,'recovered')[PADDING:]
    c = get_data(country_name,'confirmed')[PADDING:]
    
    a = np.asarray(a).reshape(a.shape[0],1)
    b = np.asarray(b).reshape(1,b.shape[0])
    c = np.asarray(c).reshape(1,c.shape[0])


    for i in range(len(dates)-(INPUT_SIZE+TARGET_SIZE)-PADDING):

        temp = []
        x = np.concatenate((np.concatenate((a[i:i+INPUT_SIZE], b[0][i:i+INPUT_SIZE].reshape(1,INPUT_SIZE).T), axis=1),c[0][i:i+INPUT_SIZE].reshape(1,INPUT_SIZE).T),axis = 1)
        X.append(x)
        temp.append(a[i+INPUT_SIZE])
        temp.append(b[0][i+INPUT_SIZE])
        temp.append(c[0][i+INPUT_SIZE])
        
        Y.append(temp)


# In[ ]:


X = np.asarray(X)
Y = np.asarray(Y) 


# In[ ]:


print("Input: ",X[0])
print("Output: ",Y[0])


# In[ ]:


print(X.shape,Y.shape)


# In[ ]:


def plot_(n):
    death_ = []
    recovered_ = []
    confirmed_ = []
    for i in X[n]:
        death_.append(i[0])
        recovered_.append(i[1])
        confirmed_.append(i[2])
        
            
    plt.plot(death_,color = 'r')
    plt.plot(recovered_ , color = 'g')
    plt.plot(confirmed_ , color = 'b')
    
    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+TARGET_SIZE),[death_[-1],Y[n][0]],color = 'r',linestyle = 'dashed')
    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+TARGET_SIZE),[recovered_[-1],Y[n][1]],color = 'g',linestyle = 'dashed')
    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+TARGET_SIZE),[confirmed_[-1],Y[n][2]],color = 'b',linestyle = 'dashed')
    
    plt.legend(labels = ['death','recovery','confirmed','actual death','actual confirmed cases','actual recovered patients'])
    plt.show()


# In[ ]:


import random
plot_(random.randrange(len(X)))


# ## **Inializing Model**

# In[ ]:


model = Sequential()

model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(INPUT_SIZE, 3)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3))

optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='mse',metrics=['mae', 'acc'])


# In[ ]:


model.summary()


# ## **Training Model**

# In[ ]:


history = model.fit(X, Y, epochs=20, verbose=2, validation_split=0.05,shuffle=True)
    
plt.plot(history.history['loss'], label="loss")
plt.plot(history.history['val_loss'], label="val_loss")
plt.legend(loc="upper right")
plt.show()

plt.plot(history.history['acc'], label="accuracy")
plt.plot(history.history['val_acc'], label="val_accuracy")
plt.legend(loc="upper right")
plt.show()


# ## **Testing the Model**

# In[ ]:



def test_random(n):
    death_ = []
    recovered_ = []
    confirmed_ = []
    for i in X[n]:
        death_.append(i[0])
        recovered_.append(i[1])
        confirmed_.append(i[2])
        
            
    plt.plot(range(INPUT_SIZE),death_,color = 'r')
    plt.plot(range(INPUT_SIZE),recovered_ , color = 'g')
    plt.plot(range(INPUT_SIZE),confirmed_ , color = 'b')

    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+1),[death_[-1],Y[n][0]],color = 'm')
    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+1),[recovered_[-1],Y[n][1]],color = 'm')
    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+1),[confirmed_[-1],Y[n][2]],color = 'm')

    out = model.predict(X[n].reshape((1,INPUT_SIZE,3)))

    print("PREDICTED: "+str(out[0][0]),"ACUTAL: "+str(Y[n][0][0]))
    print("PREDICTED: "+str(out[0][1]),"ACUTAL: "+str(Y[n][1]))
    print("PREDICTED: "+str(out[0][2]),"ACUTAL: "+str(Y[n][2]))
    
    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+1),[death_[-1],out[0][0]],color = 'r',linestyle = 'dashed')
    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+1),[recovered_[-1],out[0][1]],color = 'g',linestyle = 'dashed')
    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+1),[confirmed_[-1],out[0][2]],color = 'b',linestyle = 'dashed')

    plt.legend(labels = ['death','crecovery','confirmed','actual death','actual confirmed cases','actual recovered patients','predicted death','predicted confirmed cases','predicted recovered patients'])
    plt.show()


# In[ ]:


for i in range(10):
    n = random.randrange(len(X))
    test_random(n)


# ## Plotting prediction for each country

# In[ ]:


def plot_pred_n_days(country_name,n_days,hide=False):
    predicted_d = []
    predicted_c = []
    predicted_r = []

    X_d = get_data(country_name,'death')[-INPUT_SIZE:]
    X_r = get_data(country_name,'recovered')[-INPUT_SIZE:]
    X_c = get_data(country_name,'confirmed')[-INPUT_SIZE:]
    
    plt.plot(X_d,color = '#ff6252',label = 'death')
    plt.plot(X_r,color = '#00e639',label = 'recovered')
    plt.plot(X_c,color = '#8a4eff',label = 'confirmed')
    
    X_input = []
    
    X_d = np.asarray(X_d).reshape(X_d.shape[0],1)
    X_r = np.asarray(X_r).reshape(1,X_r.shape[0])
    X_c = np.asarray(X_c).reshape(1,X_c.shape[0])
    x = np.concatenate((np.concatenate((X_d, X_r.reshape(1,INPUT_SIZE).T), axis=1),X_c.reshape(1,INPUT_SIZE).T),axis = 1)
    X_input.append(x)
    X_r = np.asarray(X_r).reshape(X_r.shape[1],1)
    X_c = np.asarray(X_c).reshape(X_c.shape[1],1)

    for i in range(n_days):
        predicted = model.predict(np.asarray(X_input).reshape(1,INPUT_SIZE,3))[0]
        predicted_d.append(predicted[0])
        predicted_r.append(predicted[1])
        predicted_c.append(predicted[2])
        X_input = np.concatenate((np.asarray(X_input).reshape(10,3), predicted.reshape(3,1).T), axis=0)[1:]
        
        if hide == False:
            plt.scatter(INPUT_SIZE+i,predicted[0],color = '#ff6252')
            plt.scatter(INPUT_SIZE+i,predicted[1],color = '#00e639')
            plt.scatter(INPUT_SIZE+i,predicted[2],color = '#8a4eff')
    
            plt.text(INPUT_SIZE+i, predicted[0]*102100,str(int(predicted[0])) , fontsize=12 , color = '#ff6252')
            plt.text(INPUT_SIZE+i, predicted[1]*102/100,str(int(predicted[1])) , fontsize=12 , color = '#00e639')
            plt.text(INPUT_SIZE+i, predicted[2]*102/100,str(int(predicted[2])) , fontsize=12 , color = '#8a4eff')
            
    if hide == True:
        plt.scatter(INPUT_SIZE,predicted_d[0],color = '#ff6252')
        plt.scatter(INPUT_SIZE,predicted_r[0],color = '#00e639')
        plt.scatter(INPUT_SIZE,predicted_c[0],color = '#8a4eff')
    
        plt.text(INPUT_SIZE, predicted_d[0]*102/100,str(int(predicted_d[0])) , fontsize=12 , color = '#ff6252')
        plt.text(INPUT_SIZE, predicted_r[0]*102/100,str(int(predicted_r[0])) , fontsize=12 , color = '#00e639')
        plt.text(INPUT_SIZE, predicted_c[0]*102/100,str(int(predicted_c[0])) , fontsize=12 , color = '#8a4eff')
        
        plt.scatter(INPUT_SIZE+n_days-1,predicted_d[-1],color = '#ff6252')
        plt.scatter(INPUT_SIZE+n_days-1,predicted_r[-1],color = '#00e639')
        plt.scatter(INPUT_SIZE+n_days-1,predicted_c[-1],color = '#8a4eff')
    
        plt.text(INPUT_SIZE+n_days-1, predicted_d[-1]*102/100,str(int(predicted_d[-1])) , fontsize=12 , color = '#ff6252')
        plt.text(INPUT_SIZE+n_days-1, predicted_r[-1]*102/100,str(int(predicted_r[-1])) , fontsize=12 , color = '#00e639')
        plt.text(INPUT_SIZE+n_days-1, predicted_c[-1]*102/100,str(int(predicted_c[-1])) , fontsize=12 , color = '#8a4eff')
    
    maximum_d_index = predicted_d.index(max(predicted_d))
    maximum_r_index = predicted_r.index(max(predicted_r))
    maximum_c_index = predicted_c.index(max(predicted_c))
    
    plt.scatter(INPUT_SIZE+maximum_d_index,predicted_d[maximum_d_index],color = 'r',label = 'maximum death')
    plt.scatter(INPUT_SIZE+maximum_r_index,predicted_r[maximum_r_index],color = 'g',label = 'maximum recovered patients')
    plt.scatter(INPUT_SIZE+maximum_c_index,predicted_c[maximum_c_index],color = 'b',label = 'maximum confirmed cases')
    
    plt.text(INPUT_SIZE+maximum_d_index, predicted_d[maximum_d_index]*102/100,str(int(predicted_d[maximum_d_index])) , fontsize=12 , color = 'r')
    plt.text(INPUT_SIZE+maximum_r_index, predicted_r[maximum_r_index]*102/100,str(int(predicted_r[maximum_r_index])) , fontsize=12 , color = 'g')
    plt.text(INPUT_SIZE+maximum_c_index, predicted_c[maximum_c_index]*102/100,str(int(predicted_c[maximum_c_index])) , fontsize=12 , color = 'b')
    

    
    predicted_d.insert(0,X_d[-1])
    predicted_r.insert(0,X_r[-1])
    predicted_c.insert(0,X_c[-1])
    
    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+n_days),predicted_d,linestyle='dashed',color = '#ff6252',label = 'predicted death')
    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+n_days),predicted_r,linestyle='dashed',color = '#00e639',label = 'predicted recovered')
    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+n_days),predicted_c,linestyle='dashed',color = '#8a4eff',label = 'predicted confirmed')
    
    plt.legend(loc=2)
    plt.title("Predicted Death,Confirmed Cases and Recovered patients for "+country_name+" for "+str(n_days)+" days")

    plt.show()
    
    print("Total people died in " + str(n_days) + " days :",int(sum(predicted_d)))
    print("Total patients recovered in " + str(n_days) + " days :",int(sum(predicted_r)))
    print("Total confirmed cases in " + str(n_days) + " days :",int(sum(predicted_c)))
    
    print("Maximum Deaths in a day :",int((predicted_d[maximum_d_index])))
    print("Maximum Recovered patients in a day :",int(predicted_d[maximum_r_index]))
    print("Maximum Confired cases in a day :",int(predicted_d[maximum_c_index]))


# In[ ]:


for country_name in countries_sorted_death[:-15:-1]:
    plot_pred_n_days(country_name,30,hide=True)


# In[ ]:


plot_pred_n_days('India',30,hide=True)


# In[ ]:


plot_pred_n_days('US',1000,hide=True)


# We are getting curve of a exponential function which tends to infinity and the model doesn't seem to tell a peak.This is because of the datset we have provided it.Till now only a few countries have shown a part of the peak so the model tends to adapt the actuall exponential function based on the majority of the dataset ($b.a^x$).It is still possible to make a neural network to find the peak but it would require more and different variety of data.

# You can checkout my other neural network model that classifies if a patient has corona virus based on x-ray [here](https://www.kaggle.com/frozenwolf/coronahack-finetuning-resnet18-pytorch).You can checkout other similair models [here](https://github.com/FrozenWolf-Cyber/Corona-Detection).In future i will be adding models based on LSTM and multi-headed neural network till then peace !!
# 
# 
# 
# <img src="https://i.pinimg.com/originals/34/59/b3/3459b3278915fa5e4cc369136db6cccf.gif" width="450px" height="450px">
