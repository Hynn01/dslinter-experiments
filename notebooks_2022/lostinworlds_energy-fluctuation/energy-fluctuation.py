#!/usr/bin/env python
# coding: utf-8

# # CO2 from the Energy Sector
# **Tracking the changes in CO2 emissions from the energy sector.**
# 
# 
# ## Introduction
# In recent years the discourse around climate change has gained significant momentum. Every sector of the economy has been impacted with governments placing major [targets to curb CO2 emissions](https://www.gov.uk/government/publications/net-zero-strategy). The one key sources of CO2 emissions has been from energy usage. As industries seek alternative carbon neutral energy sources, in this project the CO2 emissions form specific energy sectors will be investigated. 
# 
# 
# Provided by the [Food and Agriculture Organisation of the UN](https://www.fao.org/faostat/en/#data/GN), the dataset reviewed in this project has records covering approx. 50 years from 1970 to 2019.  It holds a breakdown of CO2 emissions for a number of energy sectors from a myriad of countries. As it will become apparent, CO2 emissions from energy industries fluctuates over time and country to country. It is also notable that the energy types used in the fishing sector are particularly polluting. Moreover, the energy reliance differ from nation to nation and therefore the CO2 emissions from such sectors also vary. 
# 
# This project will evaluate the data at a global level and shift its attention to two groups of nations G7 and South American nations to understand the variations in CO2 emissions. 
# 
# ### G7
# G7 consists of some of the wealthiest countries in the world. As such, the populations in these nations are likely to have the biggest demand for energy through the use of luxury and leisure items. On the other hand, the wealth in these nations are likely to enable energy industries to invest in more effective methods of producing energy. Therefore the industries may be able to keep CO2 emissions low.
# 
# 
# ### South American
# Juxtaposing the G7 are the South American countries. They have developing economies and are also in the southern hemisphere. Therefore, these countries are likely to have differing energy needs.
# In the final section, limitations of this dataset are explored. The data has several weaknesses which affect the conclusions which can be drawn.
# 
# ## Overview
# The dataset requires some data progressing. For further details on this, please refer to the appendix. It should be noted that as the appendix will emphasis,  the data has some mathematical outliers which are persistent. Subsequently, the median average has been used though this project as it is less likely to be skewed by extreme outliers.  Furthermore, due to some null values, the data has been capped to include only data from 1971 to 2018, rather than the full scope of the data. 
# 
# 
# Following these steps, the data is as follows: 
# 
# *Please note, as some platforms do not support interactive graphs, the coding provides both static and interactive graphs in some cases*

# In[ ]:


# for data handling
import numpy as np
import pandas as pd

# for static graphs
import matplotlib.pyplot as plt
import seaborn as sns
 
# for interactive graphs
import plotly_express as px

# for kaggle
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


energy = pd.read_csv("/kaggle/input/z-unlocked-challenge-1-data-visualization/energy_use_data_11-29-2021.csv")

# Clean data - exclude unnecessary data and make minor alterations for improved readability

x = ["Year Code", "Domain Code", "Domain", "Element Code", "Element", "Item Code", "Flag", ]
energy.drop( x, inplace = True, axis =1)
energy.drop( energy[energy["Flag Description"]=="Aggregate, may include official, semi-official, estimated or calculated data"].index, inplace = True, axis =0)
energy.rename(columns = {"Area": "Country"}, inplace = True)

energy.drop( energy[energy["Year"].isin([1970, 2019])].index, inplace = True, axis =0)


#Transforming the dataset for use later in the project
energy["Country"].replace("United Kingdom of Great Britain and Northern Ireland", "UK", inplace= True)
energy["Country"].replace("United States of America", "USA", inplace= True)


energy.head()


# ### Establish frequently used functions

# In[ ]:


# Restructure the data by grouping variables 
def restructure_data (Dataset, GroupbyVariable,Value):
    Overview = Dataset.groupby(GroupbyVariable)[Value].median().reset_index()
    return Overview


# In[ ]:


#Extracts a the earliest/ oldest data in the dataset 
def extract_values(Dataset, function):
    if function == "min":
        return Dataset[Dataset["Year"] == Dataset["Year"].min()].reset_index()
    elif function == "max":
        return Dataset[Dataset["Year"] == Dataset["Year"].max()].reset_index()
    else:
        print("incorrect function")


# In[ ]:


#Compares the oldest/latest data against one another 
def comparison_of_min_max(dataset):
    Min = dataset[dataset["Year"] == dataset["Year"].min()]
    Min = Min.groupby("Item")["Value"].median()

    Max = dataset[dataset["Year"] == dataset["Year"].max()]
    Max = Max.groupby("Item")["Value"].median()

    Difference = Max-Min
    Difference =Difference.reset_index().rename(columns = {"index": "Item", 0:"Value"}).fillna(0)
    return Difference


# In[ ]:


#Extracts a cross section of the data based for specific country(s) 
def extract_country_data (Country_list, Dataset):
    extracted_data = Dataset[Dataset["Country"].isin(Country_list)]
    return extracted_data


# ## Global energy emissions
# Beginning at the global level, below is the average CO2 emissions from the energy sector. It illustrates that there are significant fluctuations in the CO2 emissions. This may be due to aspects such as good weather requiring individuals to use less energy for heating or major events placing extra demand on energy. Additionally, since 2016 the CO2 emissions have been decreasing and is more or less on par with the level of emissions recorded in 1971. It, however, remains higher than the extreme low recorded in 1976.

# In[ ]:


#Restructure data for visualisation
energy_group = restructure_data(energy, "Year", "Value")

#Construct visualisation 
fig = px.line(energy_group, x = "Year", y = "Value", title = "Global energy usage in average CO2 emissions (interactive)" )

#customisation
fig.update_xaxes(showticklabels=True, showline=True, linewidth=1, linecolor='black')
fig.update_yaxes(showticklabels=True, showline=True, linewidth=1, linecolor='black')

fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
}, showlegend = False)


fig.add_annotation(text = ("Source: https://www.fao.org/faostat/en/#data/GN <br>Median average used due to extreme outliers"),
                   showarrow = False, x = 0, y = -.30, xref= "paper", yref= "paper", xanchor = "left", yanchor = "bottom", xshift =-1,
                    yshift =-5, font = dict(size =10, color = "grey"), align = "left")


fig.show()


# In[ ]:


#using the transformed data from plotly graph to create a static version

#Construct visualisation 
plt.figure(figsize = (26,8))
sns.lineplot(x = "Year", y = "Value", data = energy_group, lw= 2)

#Customisation
plt.title("Average global energy usage", fontsize = 18, loc='left', y=1.01 )

plt.annotate('Source: https://www.fao.org/faostat/en/#data/GN \nMedian average used due to extreme outliers', (0,-.15), xycoords ='axes fraction' )
plt.ylim(10,)

sns.despine(top = True, right = True, left = False, bottom = False)
plt.xlabel("Year", fontsize=14)
plt.ylabel("CO2 emissions - absolute number in kilotones ", fontsize=14)


plt.show()


# Cumulative total shows the scale of CO2 emissions from the energy industry over approx. 50 years, emphasising the scale of the issue. It is particularly troubling if it is assumed that no new product has been created to absorb even a part of this emission. In fact, reports of [deforestation](https://earth.org/deforestation-facts/) suggest that the existing entities which absorb CO2 may have diminished during the same time period.

# In[ ]:


#Restructure data for visualisation

energy_copy = energy.copy()

energy_copy["Year"] = pd.to_datetime(energy_copy["Year"], format = "%Y")
energy_copy = pd.pivot_table(energy_copy, values='Value', index=["Year"]) 


#Construct visualisation 
ax = energy_copy["Value"].expanding().sum().plot(figsize = (15,8), title = "Cumulative total CO2 emissions from the energy industry")

#Customisation
sns.despine(top = True, right = True, left = False, bottom = False)

ax.set_ylabel("CO2 emissions in kilotonnes")
plt.annotate('Source: https://www.fao.org/faostat/en/#data/GN', (0,-.1), xycoords ='axes fraction')
plt.show()


# Dividing the CO2 emissions to each item indicates that “Gas-diesel oils used in fisheries” had the highest average emissions and is over three times more polluting than the next energy sector with the highest level of CO2 emissions.
# 
# Furthermore, Electricity has an average CO2 emission compared to the other energy types.

# In[ ]:


#Restructure data for visualisation
overall_avg = restructure_data(energy, "Item", "Value")

#Construct visualisation 
plt.figure(figsize = (15,6))
sns.barplot(y = "Item", x = "Value", data = overall_avg, order = overall_avg.sort_values("Value", ascending= False)["Item"], palette = "inferno_r")

#customisation
plt.annotate('Source: https://www.fao.org/faostat/en/#data/GN \nMedian average used due to extreme outliers', (0,-.2), xycoords ='axes fraction' )
sns.despine(top = True, right = True, left = False, bottom = False)
plt.title("Average global energy usage between 1971 and 2018", fontsize = 18, loc='left', y=1.01  )

plt.ylabel("Energy", fontsize=14)
plt.xlabel("CO2 emissions - absolute number in kilotones ", fontsize=14)

plt.show()


# Taking a snapshot of the 1971 data, stresses that Fuel oil used in fisheries was the most polluting, emitting over three times more CO2 than the next polluting energy type, Gas-Diesel oil. In addition Electricity’s role in the CO2 emissions from the energy sector appears to be relatively low.

# In[ ]:


#Restructure data for visualisation
min_data = extract_values(Dataset = energy, function = "min")
min_data_index = restructure_data(min_data, "Item", "Value")
 

#Construct visualisation 
plt.figure(figsize = (15,6))
sns.barplot(y = "Item", x = "Value", data = min_data_index, order = min_data_index.sort_values("Value", ascending= False)["Item"], palette = "inferno_r")

#customisation
plt.title("Average global energy usage in 1971", fontsize = 18, loc='left', y=1.01 )
plt.annotate('Source: https://www.fao.org/faostat/en/#data/GN \nMedian average used due to extreme outliers', (0,-.2), xycoords ='axes fraction' )
sns.despine(top = True, right = True, left = False, bottom = False)
plt.ylabel("Energy", fontsize=14)
plt.xlabel("CO2 emissions - absolute number in kilotones ", fontsize=14)


plt.show()


# Contrasting the above with 2018 data, the CO2 emissions from the electricity have jumped up significantly. An aspect of this is likely to be due to increased demand due to the popularity of items such as PCs to phones and smart devices.
# 
# That being said, the most polluting energy type is “Gas-diesel used for fisheries”. Further investigation will be required in understanding the exact cause of this high level of emission such as the level of supply throughout the sector to determine whether the energy type for fisheries is extremely polluting.

# In[ ]:


#Restructure data for visualisation
max_data = extract_values(Dataset = energy, function = "max")
max_data_index = restructure_data(max_data, "Item", "Value")


#Construct visualisation 
plt.figure(figsize = (15,6))
sns.barplot(y = "Item", x = "Value", data = max_data_index, order = max_data_index.sort_values("Value", ascending= False)["Item"], palette = "inferno_r")

#customisation
plt.annotate('Source: https://www.fao.org/faostat/en/#data/GN \nMedian average used due to extreme outliers', (0,-.2), xycoords ='axes fraction' )
sns.despine(top = True, right = True, left = False, bottom = False)

plt.ylabel("Energy", fontsize=14)
plt.xlabel("CO2 emissions - absolute number in kilotones ", fontsize=14)
plt.title("Average global energy usage in 2018", fontsize = 18, loc='left', y=1.01 )

plt.show()


# The overlay of the 1971 and 2018 data highlights the extent of the difference. It shows that the CO2 emissions by Fuel oil used in fisheries has decreased significantly. This is extremely positive in the context of climate change. However the cause of the decrease is unclear from this dataset.
# 
# Several energy types such as Motor gasoline, LNG and LPG’s CO2 have remained relatively unchanged in the approx. 50 year time period.

# In[ ]:


#Restructure data for visualisation using previously manipulated data
comparison = comparison_of_min_max(energy)

#Construct visualisation 
plt.figure(figsize= (20,8))
sns.barplot(x = "Item", y = "Value", data = comparison, order = comparison.sort_values("Value", ascending= False)["Item"], palette = "inferno_r")

#customisation
plt.axhline(0, ls = "--", color = "black")
plt.annotate('Source: https://www.fao.org/faostat/en/#data/GN \nMedian average used due to extreme outliers', (0,-.4), xycoords ='axes fraction' )
sns.despine(top = True, right = True, left = False, bottom = False)
plt.xticks(rotation=-45)

plt.xlabel("Energy", fontsize=14)
plt.ylabel("CO2 emissions - absolute number in kilotones ", fontsize=14)


plt.title("Comparison of 1971 and 2018 global energy usage in average CO2 emissions", fontsize = 18, loc='left', y=1.01 )
plt.show()


# In[ ]:


#Construct visualisation 
fig= px.bar(comparison, x = "Item", y = "Value", title ="Comparison of 1971 and 2018 global energy usage in average CO2 emissions (interactive)").update_xaxes(categoryorder = "total descending")

#customisation
fig.add_hline(0, line_width = 2, line_dash = "dot")
fig.update_xaxes(showticklabels=True, showline=True, linewidth=1, linecolor='black')
fig.update_yaxes(showticklabels=True, showline=True, linewidth=1, linecolor='black')

fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
}, showlegend = False)


fig.add_annotation(text = ("Source: https://www.fao.org/faostat/en/#data/GN <br>Median average used due to extreme outliers"),
                   showarrow = False, x = 0, y = -.5, xref= "paper", yref= "paper", xanchor = "left", yanchor = "bottom", xshift =-1,
                    yshift =-5, font = dict(size =10, color = "grey"), align = "left")


# ## Energy fluctuations over time
# As a cross section review indicated the energy usage has changed significantly over time. This is reinforced in the below graphs which outline the fluctuations in CO2 for each item over time. It is clear that the “Gas-diesel used for fisheries” and “Fuel oil used in fisheries” has undergone significant changes. “Fuel oil used in fisheries” has seen significant CO2 emission decreases. “Gas-diesel used for fisheries”, on the other hand, increased in the 1990s but has since also recorded year on year falls.
# 
# The fluctuations in other energy types have undergone minor changes.

# In[ ]:


#Restructure data for visualisation
overall_fluctuations = restructure_data(energy,["Year", "Item"], "Value")

#Construct visualisation 
g= sns.FacetGrid(data = overall_fluctuations, col = "Item", col_wrap = 3, height = 4,)
g.map(sns.lineplot, "Year", "Value",)
g.fig.suptitle('Energy usage fluctuations overtime',fontsize = 18, horizontalalignment='right', y = 1.03)
plt.show()


# In[ ]:


#Construct visualisation 
fig = px.line(overall_fluctuations, x = "Year", y = "Value", facet_col = "Item", facet_row_spacing=0.04, facet_col_wrap=2, 
              width=800, height=2000, title = "Global energy usage in average CO2 emissions separated by energy" )

#customisation
fig.update_xaxes(matches = None, showline=True, linewidth=1, linecolor='black')
fig.update_yaxes(showticklabels=True, showline=True, linewidth=1, linecolor='black')
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
}, showlegend = False)
    
    
fig.add_annotation(text = ("Source: https://www.fao.org/faostat/en/#data/GN <br>Median average used due to extreme outliers"),
                   showarrow = False, x = 0, y = -.3, xref= "paper", yref= "paper", xanchor = "left", yanchor = "bottom", xshift =-1,
                    yshift =-5, font = dict(size =10, color = "grey"), align = "left")

fig.show()


# The dynamics of the energy fluctuations are also presented in a heatmap below, demonstrating the fluctuations in a more visual manner. Please see appendix, section 2 for an interactive graph which overlays the fluctuations directly on one graph for direct comparison of the CO2 changes.

# In[ ]:


#Restructure data for visualisation
heat = energy.groupby(["Item", "Year"])["Value"].median().unstack()
heat.dropna(inplace = True)

#Construct visualisation 
plt.figure(figsize = (20,10))
sns.heatmap(heat, cmap = "mako")

#customisation
plt.ylabel("Energy", fontsize=14)
plt.xlabel("Year", fontsize=14)
plt.annotate('Source: https://www.fao.org/faostat/en/#data/GN \nMedian average used due to extreme outliers', (0,-.15), xycoords ='axes fraction' )
plt.title("Heatmap of Energy usage fluctuations overtime", fontsize = 18, horizontalalignment = "right", y=1.01 )

plt.show()


# ## G7
# In this section the data specifically for the G7 countries will be examined. As the information from the UK government highlights, the list of G7 nations are as follows:
# 
# * UK,
# * USA,
# * Canada,
# * Japan,
# * Germany,
# * France,
# * Italy,
# * EU
# 
# As the EU is a collection of nations and some key member countries are also separately in the G7, they have not been included in this graph.
# 
# For G7 countries, the overall CO2 emissions from the energy sector is generally around 400 to 600 kilotons. However in 1978 and in 1990, there were significant spikes in the emission levels. Causes of these spikes are beyond the scope of this project.

# In[ ]:


g7_countries = ( "UK", "USA", "Canada", "Japan", "Germany", "France", "Italy")

g7 = extract_country_data(Country_list = g7_countries, Dataset = energy)


# In[ ]:


#Restructure data for visualisation
G7_Overview = restructure_data(g7, "Year", "Value")

#Construct visualisation 
plt.figure(figsize = (26,8))
sns.lineplot(x = "Year", y = "Value", data = G7_Overview, lw= 2)

#customisation
plt.annotate('Source: https://www.fao.org/faostat/en/#data/GN \nMedian average used due to extreme outliers', (0,-.1), xycoords ='axes fraction' )
plt.ylim(10,)
sns.despine(top = True, right = True, left = False, bottom = False)
plt.title("Average energy usage for G7", fontsize = 18, loc='left', y=1.01 )
plt.xlabel("Year", fontsize=14)
plt.ylabel("CO2 emissions - absolute number in kilotones ", fontsize=14)

plt.show()


# In[ ]:


#Construct visualisation 
fig =px.line(G7_Overview, x="Year", y= "Value", title = "Average energy usage for G7(interactive)" )

#customisation
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

fig.add_annotation(text = ("Source: https://www.fao.org/faostat/en/#data/ET")
    , showarrow = False, x = 0, y = -0.15, xref = "paper", yref = "paper", xanchor = "left", yanchor = "bottom",
    xshift =-1, yshift =-5, font=dict(size=10, color = "grey"), align = "left")
fig.add_annotation(text = ("Source: https://www.fao.org/faostat/en/#data/ET"), showarrow= False, x= 0, y =-.15, 
                  xref = "paper", yref = "paper", xanchor="left", yanchor="bottom", xshift=-1, yshift=-5,
                  font=dict(size = 10, color = "grey"), 
                  align = "left")

fig.show()


# Following is a graph illustrating the overall average emissions for the energy sector for the G7 countries. It highlights that unlike the global trends, Gas-Diesel oil was the most polluting sector for the G7 countries, nearly six times more polluting compared to Coal, the least polluting sector.

# In[ ]:


#Restructure data for visualisation
g7_avg = restructure_data(g7, "Item", "Value")

#Construct visualisation 
plt.figure(figsize = (15,6))
sns.barplot(y = "Item", x = "Value", data = g7_avg, order = g7_avg.sort_values("Value", ascending= False)["Item"], palette = "inferno_r")

#customisation
plt.annotate('Source: https://www.fao.org/faostat/en/#data/GN \nMedian average used due to extreme outliers', (0,-.2), xycoords ='axes fraction' )
sns.despine(top = True, right = True, left = False, bottom = False)
plt.title("Average global energy usage between 1971 and 2018 for G7", fontsize = 18, loc='left', y=1.01  )
plt.ylabel("Energy", fontsize=14)
plt.xlabel("CO2 emissions - absolute number in kilotones ", fontsize=14)

plt.show()


# Comparing the G7 1971 data against the 2018 indicates that Gas-Diesel oil CO2 emissions have greatly increased in the approx. 50 years. This is likely to be due to the [push for Diesel cars](https://www.bbc.co.uk/news/uk-politics-41985715). Also contrary to global trends the CO2 emissions from “Motor Gasoline”.

# In[ ]:


#Restructure data for visualisation
g7_comparison = comparison_of_min_max(g7)

#Construct visualisation 
plt.figure(figsize=(25,10))
sns.barplot(x = "Item", y = "Value", data = g7_comparison, order = g7_comparison.sort_values("Value", ascending= False)["Item"], palette = "inferno_r")

#customisation
plt.axhline(0, ls = "--", color = "black")
plt.annotate('Source: https://www.fao.org/faostat/en/#data/GN \nMedian average used due to extreme outliers', (0,-.2), xycoords ='axes fraction' )
sns.despine(top = True, right = True, left = False, bottom = False)
plt.title("Average total temperature fluctuation for G7 countries between 1961 to 2020", fontsize = 18, loc = "left")

plt.xlabel("Energy", fontsize=14)
plt.ylabel("CO2 emissions - absolute number in kilotones ", fontsize=14)
plt.show()


# Looking at the energy sectors for the G7 countries individually, presents some fascinating trends. Electric energy CO2 emissions, though having some significant fluctuations, have found a stable limit at 2000 kilotons. In contrast, significant work is needed to bring down the CO2 emission from Gas-Diesel as it dominates the missions compared to every other energy type.
# 
# 

# In[ ]:


#Restructure data for visualisation
g7_fluctuations = restructure_data(g7,["Year", "Item"], "Value").reset_index()

#Construct visualisation 
g= sns.FacetGrid(data = g7_fluctuations, col = "Item", col_wrap = 3, height = 4)
g.map(sns.lineplot, "Year", "Value",)
g.fig.suptitle('Energy usage fluctuations overtime for G7',fontsize = 18, horizontalalignment='right', y = 1.03)

plt.show()


# In[ ]:


#Construct visualisation 
fig = px.line(g7_fluctuations, x = "Year", y = "Value", facet_col = "Item", facet_row_spacing=0.04, 
              facet_col_wrap=2, width=800, height=2000,
             title = "Energy usage in average CO2 emissions for G7 separated by energy")

#customisation
fig.update_xaxes(matches = None, showline=True, linewidth=1, linecolor='black')
fig.update_yaxes(showticklabels=True, showline=True, linewidth=1, linecolor='black')
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
}, showlegend = False)

#extracting plotly graphs 
#py.plot(fig, filename = "Energy usage in average CO2 emissions for G7 separated by energy", auto_open = False)

    
fig.show()


# Finally the heatmap clearly demonstrates the high rate of CO2 emission by the Gas-Diesel oil sector. Electricity’s CO2 grows throughout the approx. 50 years but not to the extent of Gas-Diesel. LNG oddly has a spike in 1990s which may be the cause of the overall increase in CO2 for the G7 in 1990s. However, it is still unclear what caused this increase, although 1990 was the year of the [oil crisis](https://www.bis.org/publ/econ31.htm#:~:text=By%20the%20end%20of%201990,respects%2C%20this%20was%20not%20surprising).

# In[ ]:


#Restructure data for visualisation
heat = g7.groupby(["Item", "Year"])["Value"].median().unstack()
heat.dropna(inplace = True)

#Construct visualisation 
plt.figure(figsize = (20,10))
sns.heatmap(heat, cmap = "mako")

#customisation
plt.xlabel("Year", fontsize=14)
plt.ylabel("Energy", fontsize=14)
plt.annotate('Source: https://www.fao.org/faostat/en/#data/GN \nMedian average used due to extreme outliers', (0,-.2), xycoords ='axes fraction' )
plt.title("Heatmap of Energy usage fluctuations overtime for G7", fontsize = 18, horizontalalignment = "right", y=1.01 )

plt.show()


# ## South America
# To directly contrast the G7 data, is the data for the South American nations. Many of the South American nations are regarded as being a part of the developing nations. As such, their reliance on energy is likely to differ from the G7 nations. Moreover, Brazil for instance has had some significant economic growth during the period in concern ([The World Bank, 2022](https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG?locations=BR)). Therefore it may have significant energy needs that differ from nations experiencing differing economic development.
# 
# 
# Firstly it is clear that the overall level of CO2 emission for the South American countries is significantly lower than the G7. The highest average level in South American is just over 92 kilotons. The G7’s highest value was approx. 1,190. Secondly in the late 1980s, there was a major drop in the CO2 emissions before the trend reverting and peaking in 1997. Since then, there has been a plethora of fluctuations but maintain a general downwards trend.

# In[ ]:


South_America_countries = ["Argentina","Bolivia","Brazil","Chile","Colombia","Ecuador","French Guyana",
                 "Guyana","Paraguay","Peru","Suriname","Uruguay","Venezuela"]
# https://www.britannica.com/topic/list-of-countries-in-Latin-America-2061416

South_America = extract_country_data(Country_list = South_America_countries, Dataset = energy)


# In[ ]:


#Restructure data for visualisation
South_America_Overview = restructure_data(South_America, "Year", "Value").reset_index()

#Construct visualisation 
plt.figure(figsize = (26,8))
sns.lineplot(x = "Year", y = "Value", data = South_America_Overview, lw= 2)

#customisation
plt.annotate('Source: https://www.fao.org/faostat/en/#data/GN \nMedian average used due to extreme outliers', (0,-.1), xycoords ='axes fraction' )
plt.ylim(10,)
sns.despine(top = True, right = True, left = False, bottom = False)
plt.title("Average energy usage for South America", fontsize = 18, loc='left', y=1.01 )
plt.xlabel("Year", fontsize=14)
plt.ylabel("CO2 emissions - absolute number in kilotones ", fontsize=14)

plt.show()


# In[ ]:


#Construct visualisation 
fig =px.line(South_America_Overview, x="Year", y= "Value", title = "Average energy usage for South America(interactive)" )

#customisation
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

fig.add_annotation(text = ("Source: https://www.fao.org/faostat/en/#data/ET")
    , showarrow = False, x = 0, y = -0.15, xref = "paper", yref = "paper", xanchor = "left", yanchor = "bottom",
    xshift =-1, yshift =-5, font=dict(size=10, color = "grey"), align = "left")
fig.add_annotation(text = ("Source: https://www.fao.org/faostat/en/#data/ET"), showarrow= False, x= 0, y =-.15, 
                  xref = "paper", yref = "paper", xanchor="left", yanchor="bottom", xshift=-1, yshift=-5,
                  font=dict(size = 10, color = "grey"), 
                  align = "left")

fig.show()


# Similar to the G7, the most polluting energy sector is Gas-Diesel for the approx 50 year period. However, Gas-diesel oils used in fisheries is also prominent, and the scale of difference should be noted between the two groups of countries.

# In[ ]:


#Restructure data for visualisation
South_America_avg = restructure_data(South_America, "Item", "Value").reset_index()

#Construct visualisation 
plt.figure(figsize = (15,6))
sns.barplot(y = "Item", x = "Value", data = South_America_avg, order = South_America_avg.sort_values("Value", ascending= False)["Item"], palette = "inferno_r")

#customisation
plt.annotate('Source: https://www.fao.org/faostat/en/#data/GN \nMedian average used due to extreme outliers', (0,-.2), xycoords ='axes fraction' )
sns.despine(top = True, right = True, left = False, bottom = False)
plt.title("Average global energy usage between 1971 and 2018 for South America", fontsize = 18, loc='left', y=1.01  )
plt.ylabel("Energy", fontsize=14)
plt.xlabel("CO2 emissions - absolute number in kilotones ", fontsize=14)

plt.show()


# Reviewing the 1971 data against the 2018 data for South American indicates that following the global trend “Fuel oil used in fisheries” has undergone the biggest transformation. The changes with other energy sectors also generally follow the global trends previously identified.
# 
# 

# In[ ]:


# Restructuring the data for visualisation 
South_America_comparison =comparison_of_min_max(South_America)

#Construct visualisation 
plt.figure(figsize=(25,10))
sns.barplot(x = "Item", y = "Value", data = South_America_comparison, order = South_America_comparison.sort_values("Value", ascending= False)["Item"], palette = "inferno_r")

#customisation
plt.xlabel("Energy", fontsize=14)
plt.ylabel("CO2 emissions - absolute number in kilotones ", fontsize=14)

plt.title("Average total temperature fluctuation for South American countries between 1961 to 2020", fontsize = 18, loc = "left")


plt.axhline(0, ls = "--", color = "black")
plt.annotate('Source: https://www.fao.org/faostat/en/#data/GN \nMedian average used due to extreme outliers', (0,-.1), xycoords ='axes fraction' )
sns.despine(top = True, right = True, left = False, bottom = False)


# The individual items and its trends for South American shows that it has significantly more fluctuation in a number of energy sectors compared to G7 and even the global trends. As the scales appear to be the same for particularly the global trend, it can be assumed that the presentation of the data is accurate. However, this then leads to inquiries relating to the cause of the fluctuations. One aspect may be the increased demand for energy due to the growth in the economy. However further research examining economic trends against this data is needed to conclusively state this.

# In[ ]:


#Restructure data for visualisation
South_America_fluctuations = restructure_data(South_America, ["Year", "Item"], "Value").reset_index()

#Construct visualisation 
g= sns.FacetGrid(data = South_America_fluctuations, col = "Item", col_wrap = 3, height = 4)
g.map(sns.lineplot, "Year", "Value",)
g.fig.suptitle('Energy usage fluctuations overtime for South America',fontsize = 18, horizontalalignment='right', y = 1.03)
plt.show()


# In[ ]:


#Construct visualisation 
fig = px.line(South_America_fluctuations, x = "Year", y = "Value", facet_col = "Item", 
              facet_row_spacing=0.04, facet_col_wrap=2, width=800, height=2000,
             title = "Energy usage in average CO2 emissions for South America separated by energy")

#customisation
fig.update_xaxes(matches = None, showline=True, linewidth=1, linecolor='black')
fig.update_yaxes(showticklabels=True, showline=True, linewidth=1, linecolor='black')
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
}, showlegend = False)
    
#extracting plotly graphs 
#py.plot(fig, filename = "Energy usage in average CO2 emissions for South America separated by energy", auto_open = False)

fig.show()


# Finally, the heatmap presents the fluctuations in each type of energy sector compared to one another. As was discussed previously, the increase in CO2 emissions in Gas-Diesel is undeniable and major changes will be required for South American nations if they are to bring CO2 emissions and climate change under control. LPG CO2 emissions is particularly low throughout the period concerned.

# In[ ]:


#Restructure data for visualisation
heat = South_America.groupby(["Item", "Year"])["Value"].mean().unstack()
heat.dropna(inplace = True)

#Construct visualisation 
plt.figure(figsize = (20,10))
sns.heatmap(heat, cmap = "mako")

#customisation
plt.xlabel("Year", fontsize=14)
plt.ylabel("Energy", fontsize=14)

plt.annotate('Source: https://www.fao.org/faostat/en/#data/GN \nMedian average used due to extreme outliers', (0,-.15), xycoords ='axes fraction' )
plt.title("Heatmap of Energy usage fluctuations overtime for South America", fontsize = 18, horizontalalignment = "right", y=1.01 )
plt.show()


# ## Limitations of the dataset
# The dataset has numerous limitations that need to be taken into account when drawing conclusions from this dataset. Some of these limitations are outlined below.
# 
# 
# Firstly, the dataset does contain information on the changes in production. For instance, has there been any improvements in the production of one energy type over the other? As such, it cannot be concluded whether the CO2 decrease from 1970 to 2019 for “Gas-diesel oils used in fisheries” is due to decrease in demand/production or due to efficiencies in production.
# 
# 
# Secondly, due to the focus on CO2 emissions, the role of sustainable energy sectors such as wind and solar, and their role is unclear.
# 
# 
# Thirdly, the calculations behind CO2 emission values are likely to have some limitations. Gas, for instance, needs to be burnt to produce the energy they hold. Therefore emitting additional CO2 after they have been produced. Electricity, on the other hand, does not emit further CO2 at the point of usage. Consequently, this dataset could damp down the level of pollution of some energy types.
# 
# 
# Furthermore, linked to the previous point is the extent of calculations. For example the machinery used to create such energy sources are likely to have CO2 emissions in its production. Likewise, transporting energy to the individuals needing it will also incur CO2 emissions. However it is unclear if these aspects have been included in this dataset.
# 
# 
# Moreover, CO2 is not the only negative aspect of the energy sector. Oil spillage and [acid rain](https://www3.epa.gov/acidrain/education/site_students/whatcauses.html#:~:text=Acid%20rain%20is%20caused%20by,pollutants%2C%20known%20as%20acid%20rain) are some aspects which are not captured in this dataset. This further emphasises that conclusions drawn from this dataset must be critically reviewed in the context of the wider discourse concerning climate change.
# 
# 
# Electricity can be produced through the use of fuel such as coal and Gas, i.e. energy listed separately in the dataset([Hausfather, 2019](https://www.carbonbrief.org/analysis-why-the-uks-co2-emissions-have-fallen-38-since-1990)). Once again it is clear how this has been accounted for in the dataset.
# Finally as The Economist ([2011](https://www.amazon.co.uk/Economist-Economics-Making-Modern-Economy/dp/1846684595)) highlights even the GDP is known to have issues with accuracy especially in poorer nations. If this is to be expanded, it can be assumed that these figures may not be fully accurate regardless of the other issues highlighted above. Having said that, the ability to secure more accurate data is likely to face greater issues.
# 
# ## Conclusion
# This has been an extensive examination of the global CO2 emissions for the energy sector for roughly a 50 year period. It highlights the fluctuations in CO2 overall as well as for specific energy industries. Energy centred upon fisheries is particularly polluting though its level of pollution appears to have decreased in the 2000s and onwards.
# In addition, different countries recorded varying levels of CO2 emissions from different energy types.
# However due to a myriad of limitations with the data, further investigation is required to understand the cause of such fluctuations.
# 
# ## Appendix
# 
# ### Understanding the dataset
# This section will examine individual elements of the dataset and provide explanations concerning base assumptions made throughout the project.

# In[ ]:


data= pd.read_csv("/kaggle/input/z-unlocked-challenge-1-data-visualization/energy_use_data_11-29-2021.csv")
data.head()


# On the surface, there does not appear to be any null values.

# In[ ]:


data.info()


# #### Domain, Element and Unit
# Firstly, the Domain Code" and "Domain" contain only one variable, GN/Energy use. As such they are unlikely to provide any insight. Similarly, "Element Code" and "Element" also contain one variable. Therefore these can be removed from the dataset with no loss of understanding/ distorting the overall dataset.
# 
# 
# Whilst the unit column contains one variable, dropping this column will lead to a reduction in the readability of the data. As such, this variable will be retained during the body of the project.

# In[ ]:


data["Domain Code"].value_counts()


# In[ ]:


data["Domain"].value_counts()


# In[ ]:


data["Element Code"].value_counts()


# In[ ]:


data["Element"].value_counts()


# In[ ]:


data["Unit"].value_counts()


# #### Area
# The Area variable holds a list of country names. It should be noted that entries such as "Saint Kitts and Nevis" though may appear to refer to a region are in fact one country as classified by the UN.
# 
# 
# For added readability, the term “Country” has been used for added clarification. In majority of the cases this term holds true under the below [definition of country](https://www.mapsofworld.com/answers/k-12-resources/nation-and-country/#:~:text=In%20broad%20terms%2C%20a%20country,identity%2C%20ethnicity%2C%20history%20etc):
# > “In broad terms, a country is a group of people governed by a government, which is the final authority over those people. There is a political setup that governs everyone in the country.”
# 
# There is, however, a possibility that some entries may be [territories](http://www.differencebetween.net/miscellaneous/politics/political-institutions/difference-between-territory-and-state/#:~:text=A%20state%20is%20also%20sometimes,the%20state%20that%20governs%20them).
# 
# In addition, as the dataset uses “United Kingdom of Great Britain and Northern Ireland” and “United States of America” to represent the UK and US respectively and for readability it may be best to amended these to UK and US.
# 
# 
# It should be noted that names of countries and territories are subject to change over time. The most clear representation of this is the inclusion of “USSR” which was dissolved in 1991 to current day Russia and other nations. ([Office of the Historian, 2021](https://history.state.gov/milestones/1989-1992/collapse-soviet-union#:~:text=On%20December%2025%2C%201991%2C%20the,the%20newly%20independent%20Russian%20state)) This is likely to impact the dataset as countries which ceased to exist or came into existence during the years of 1961 and 2020 are likely to have less data points than established nations.

# In[ ]:


data["Area"].unique()


# In[ ]:


data.rename(columns = {"Area": "Country"}, inplace = True)


# In[ ]:


x = data[data["Country"]== "USSR"]["Year"].max()
print ("Most recent record for USSR is", x)


# #### Duplicated data
# There are some columns in this dataset where the information has been duplicated. For example, “Year Code” and “Year” share the same information. As such, one column can be dropped without affecting the reading of the data.
# 
# Likewise, “Flag” and “Flag Description” hold the same information. However as Flag presents the information in code form it could could pose issues for the reader, thus “Flag” is dropped.
# This is mirrored in the “Item Code” and “Item”. Once again “Item Code” is dropped to ensure readability is not hampered by the changes to the dataset.

# In[ ]:


sum(data["Year Code"]==data["Year"])==data.shape[0]


# #### Year 
# The dataset covers a 49 year time period, starting from 1970.

# In[ ]:


data_group = data.groupby(["Country", "Year"])["Value"].mean().reset_index()

print ("Oldest entry - ", data_group["Year"].min())


# In[ ]:


print ("Newest entry - ", data_group["Year"].max())


# In[ ]:


x = data_group["Year"].to_numpy()
print ( "The dataset contains records for", np.ptp(x), "years, starting from", data_group["Year"].min(), "to", data_group["Year"].max())


# In[ ]:


x = (data_group["Year"].max() +1) - data_group["Year"].min() == data_group["Year"].nunique()

print( "Are there Values for for every year between 1970 and 2019?", x)


# #### Item
# The dataset covers 9 energy industries which include the following:

# In[ ]:


data["Item"].value_counts()


# It appears that the items are not evenly distributed in the dataset. It favours some industries such as Gas-Diesel oil and Motor Gasoline over Fuel oil used in fisheries. This could be the lack of popularity of some energy types over others but comparisons between the industries may be skewed due to this imbalance.

# In[ ]:


#Construct visualisation 
plt.figure(figsize = (20,10))
sns.countplot(x = "Item", data = data)

#customisation
plt.xlabel("Energy", fontsize=14)
plt.annotate('Source: https://www.fao.org/faostat/en/#data/GN', (0,-.3), xycoords ='axes fraction')
plt.title("Distribution Energy type", fontsize = 18, loc='left', y=1.01  )
plt.xticks(rotation=45)
plt.show()


# Furthermore, comparing the items against the year, suggest that there are missing values for “Fuel oil used in fisheries” and “Gas-diesel oils used in fisheries”. This is likely to cause issues in the project.
# 
# Further investigations indicate that the missing values are in 1970 and 2019. As such, it may be best to use only data from 1971 and 2018.

# In[ ]:


pivot = pd.pivot_table(data, index = ["Year"], columns = ["Item"], values = ["Value"])
pivot.info()


# In[ ]:


pivot.isnull().iloc[:,[3, 5]]


# #### Flag description
# The “Flag Description” variable contains three variables, 'FAO estimate', 'International reliable sources', and 'Aggregate, may include official, semi-official, estimated or calculated data’. However they too are not distributed in the dataset evenly. 'Aggregate, may include official, semi-official, estimated or calculated data’ entries are significantly limited in comparison to the other data types.
# 

# In[ ]:


#Construct visualisation 
plt.figure(figsize = (20,8))
sns.countplot(x = "Flag Description", data = data, palette = "mako")

#customisation
sns.despine(top = True, right = True, left = False, bottom = False)
plt.annotate('Source: https://www.fao.org/faostat/en/#data/GN', (0,-.1), xycoords ='axes fraction')
plt.title("Distribution flag description", fontsize = 18, loc='left', y=1.01  )
plt.show()


# In fact the “Aggregate, may include official, semi-official, estimated or calculated data” account for less than 1% of the data. As such, this variable is dropped particularly as the variable is an aggregate of several calculations. For reference, FAO estimates and international reliable sources account for 64% and 35% of the data. Therefore these variables are left in the data source.
# 
# 
# The distribution of the flag description against items is such that there are more records of Motor Gasoline and LNG under the FAO estimates. However international reliable sources provide more data for gas-diesel oil and electric.

# In[ ]:


(sum(data["Flag Description"]=="Aggregate, may include official, semi-official, estimated or calculated data")/data.shape[0])*100


# In[ ]:


(sum(data["Flag Description"]=='FAO estimate')/data.shape[0])*100


# In[ ]:


(sum(data["Flag Description"]=='International reliable sources')/data.shape[0])*100


# In[ ]:


#Construct visualisation 
plt.figure(figsize = (20,8))
sns.countplot(x ="Item", hue= "Flag Description", data = data, palette = "twilight")

#customisation
plt.annotate('Source: https://www.fao.org/faostat/en/#data/GN', (0,-.3), xycoords ='axes fraction')
sns.despine(top = True, right = True, left = False, bottom = False)

plt.xlabel("Energy", fontsize=14)
plt.title("Distribution flag description  and energy", fontsize = 18, loc='left', y=1.01  )
plt.xticks(rotation=45)
plt.show()


# #### Distribution of data
# The data is extremely broadly distributed with many values in the range of 0 to 25000 range with a strong skew to the right.

# In[ ]:


#Construct visualisation 
fig = px.histogram(data, x= "Value", color = "Item",  width = 1000)

#customisation
fig.update_layout(legend = dict(orientation = "h"))
fig.update_xaxes(showticklabels=True, showline=True, linewidth=1, linecolor='black')
fig.update_yaxes(showticklabels=True, showline=True, linewidth=1, linecolor='black')

fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
}, showlegend = False)

#extracting plotly graphs 
#py.plot(fig, filename = "Distribution of CO2 emissions seperated by energy type", auto_open = False)


fig.show()


# In[ ]:


#Construct visualisation 
plt.figure(figsize = (25,10))
sns.histplot(x = "Value", data = data)

#customisation
plt.xlabel("CO2 emissions - absolute number in kilotones ", fontsize=14)
plt.title("Distribution CO2 emissions", fontsize = 18, loc='left', y=1.01  )
plt.annotate('Source: https://www.fao.org/faostat/en/#data/GN', (0,-.1), xycoords ='axes fraction')

plt.show()


# This distribution remains relatively unchanged when each energy industry is examined individually.

# In[ ]:


#Construct visualisation 
fig = px.histogram(data, x= "Value", color = "Item", facet_col = "Item",
                   facet_row_spacing=0.04, facet_col_wrap=1, 
                   width=1000, height=2000,
                  title = "Distribution of CO2 emissions separated by energy ")

#customisation
fig.update_xaxes(matches = None)
fig.update_xaxes(showticklabels=True, )
fig.update_layout(showlegend = False)

fig.update_xaxes(showticklabels=True, showline=True, linewidth=1, linecolor='black')
fig.update_yaxes(showticklabels=True, showline=True, linewidth=1, linecolor='black')

fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
}, showlegend = False)

fig.show()


# In[ ]:


#Construct visualisation 
g= sns.FacetGrid(data = data, col = "Item", col_wrap = 1,margin_titles= False, height = 6,aspect = 4,  sharex=False)
g.map(sns.histplot, "Value",)

#customisation
g.fig.suptitle('Distribution of CO2 emissions separated by energy ',fontsize = 18, horizontalalignment='right', y = 1.03)
plt.show()


# In other words, the data contains a high number of mathematical outliers as further emphasised by the following boxplots. Further research is required to understand the cause of these fluctuations, although causes could include poor weather leading to increased demand for heating.
# 
# Regardless this poses an issue when using mean averages as it is sensitive to extreme outliers.

# In[ ]:


#Construct visualisation 
plt.figure(figsize = (25,10))
sns.boxplot(y = "Value", data = data)

#customisation
sns.despine(top = True, right = True, left = False, bottom = False)
plt.ylabel("CO2 emissions - absolute number in kilotones ", fontsize=14)
plt.title("Box plot distribution of CO2 emissions", fontsize = 18, loc='left', y=1.01  )

plt.show()


# In[ ]:


#Construct visualisation 
fig = px.box(data, x= "Item", y = "Value", title= "Box plot distribution of CO2 emissions")

#customisation
fig.update_xaxes(showticklabels=True, showline=True, linewidth=1, linecolor='black')
fig.update_yaxes(showticklabels=True, showline=True, linewidth=1, linecolor='black')

fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
}, showlegend = False)


# In[ ]:


#Construct visualisation 
plt.figure(figsize = (25,10))
sns.boxplot(data= data, x= "Item", y = "Value")

#customisation
plt.annotate('Source: https://www.fao.org/faostat/en/#data/GN', (0,-.1), xycoords ='axes fraction' )
sns.despine(top = True, right = True, left = False, bottom = False)
plt.title("Box plot distribution of CO2 emissions seperated by Energy", fontsize = 18, loc='left', y=1.01  )


plt.ylabel("CO2 emissions - absolute number in kilotones ", fontsize=14)
plt.xlabel("Energy", fontsize=14)


# This appears to be the same when reviewing the distribution of the data in context of the Flag description type, i.e. data contains outliers.

# In[ ]:


#Restructure data for visualisation
data["Flag Description"].replace('Aggregate, may include official, semi-official, estimated or calculated data',"Aggregate", inplace =True)
data["Flag Description"].replace("International reliable sources", "International", inplace = True)

#Construct visualisation 
fig = px.box(data, x = "Flag Description", y = "Value", title = "Box plot distribution of CO2 emissions seperated by Flag Description")

#customisation
fig.update_xaxes(showticklabels=True, showline=True, linewidth=1, linecolor='black')
fig.update_yaxes(showticklabels=True, showline=True, linewidth=1, linecolor='black')

fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
}, showlegend = False)


# In[ ]:


#Construct visualisation 
plt.figure(figsize = (25,10))
sns.boxplot(data =data, x = "Flag Description", y= "Value", )

#customisation
sns.despine(top = True, right = True, left = False, bottom = False)
plt.title("Box plot distribution of CO2 emissions seperated by Flag Description", fontsize = 18, loc='left', y=1.01  )

plt.ylabel("CO2 emissions - absolute number in kilotones ", fontsize=14)


# ## Works Cited
# Below are some of the particular key resources used in this project.
# 
#     Jee, Ken, and Andrara Olteanu. “Challenge 1 Tutorial - Line Chart (Seaborn).” Kaggle, 2022, Challenge 1 Tutorial - Line Chart (Seaborn). Accessed 7 March 2022.
# 
#     Knaflic, Cole Nussbaumer. Storytelling with Data: A Data Visualization Guide for Business Professionals. Wiley, 2015.
# 
#     Plotly. “Plotly express with Python.” Plotly, 2022, https://plotly.com/python/plotly-express/. Accessed 15 March 2022.
# 
#     Shaikh, Reshama. “Enriching Data Visualizations with Annotations in Plotly using Python.” Medium, 2021, https://medium.com/nerd-for-tech/enriching-data-visualizations-with-annotations-in-plotly-using-python-6127ff6e0f80. Accessed 14 March 2022.
# 
# 
# ## Disclaimer
# *This project was undertaken as part of the HP Unlocked challenge, particularly [challenge one](https://www.hp.com/us-en/workstations/industries/data-science/unlocked-challenge.html)*
# 
# > "Wind and other clean, renewable energy will help end our reliance on fossil fuels and combat the severe threat that climate change poses to humans and wildlife alike." - [Frances Beinecke](https://www.brainyquote.com/quotes/frances_beinecke_736944)

# In[ ]:




