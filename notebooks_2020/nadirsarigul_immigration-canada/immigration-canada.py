#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_excel("/kaggle/input/immigration-to-canada-ibm-dataset/Canada.xlsx",
                     sheet_name='Canada by Citizenship',
                     skiprows = range(20),
                     skipfooter = 2)
                     
           
                   


# In[ ]:


df.shape
df.head()


# In[ ]:


# print the dimensions of the dataframe
print(df.shape)


# In[ ]:


# clean up the dataset to remove unnecessary columns (eg. REG) 
df.drop(["AREA", "REG", "DEV", "Type","Coverage"], axis = 1, inplace=True)
# let's rename the columns so that they make sense
df.rename(columns = {'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace= True)


# In[ ]:


# for sake of consistency, let's also make all column labels of type string
df.columns = list(map(str, df.columns))


# In[ ]:


# set the country name as index - useful for quickly looking up countries using .loc method
df.set_index("Country", inplace=True)


# In[ ]:


df.head()


# In[ ]:


# add total column
df['Total'] = df.sum(axis=1)


# In[ ]:


# years that we will be using in this lesson - useful for plotting later on
years = list(map(str, range(1980, 2014)))


# # Visualizing Data using Matplotlib

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
mpl.style.use("ggplot")
print('Matplotlib version: ', mpl.__version__)


# # Waffle Charts 
# A waffle chart is an interesting visualization that is normally created to display progress toward goals. It is commonly an effective option when you are trying to add interesting visualization features to a visual that consists mainly of cells, such as an Excel dashboard.
# 
# Let's revisit the previous case study about Denmark, Norway, and Sweden.

# In[ ]:


# let's create a new dataframe for these three countries 
three_country = df.loc[['Denmark',"Norway","Sweden"],:]

# let's take a look at our dataframe
three_country


# #  Step 1.
# The first step into creating a waffle chart is determing the proportion of each category with respect to the total.

# In[ ]:


# compute the proportion of each category with respect to the total
total_values = sum(three_country["Total"])
category_proportions =[(float(value) / total_values) for value in three_country["Total"]]

for i, proportion in enumerate(category_proportions):
    print(three_country.index.values[i] + ": " +str(proportion) )


# # Step 2. 
# The second step is defining the overall size of the waffle chart.

# In[ ]:


width = 40 # width of chart
height =10 #height of chart

total_num_tiles = width*height #total number of tiles
print('Total number of tile is ', total_num_tiles)


# # Step 3. 
# The third step is using the proportion of each category to determe it respective number of tiles

# In[ ]:


# compute the number of tiles for each catagory
tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]

# print out number of tiles per category
for i, tiles in enumerate(tiles_per_category):
    print(three_country.index.values[i] + ": " + str(tiles))


# # Step 4. 
# The fourth step is creating a matrix that resembles the waffle chart and populating it.

# In[ ]:


# initialize the waffle chart as an empty matrix
waffle_chart = np.zeros((height, width))

# define indices to loop through waffle chart
category_index = 0
tile_index = 0

# populate the waffle chart
for col in range(width):
    for row in range(height):
        tile_index += 1

        # if the number of tiles populated for the current category is equal to its corresponding allocated tiles...
        if tile_index > sum(tiles_per_category[0:category_index]):
            # ...proceed to the next category
            category_index += 1       
            
        # set the class value to an integer, which increases with class
        waffle_chart[row, col] = category_index
        
print ('Waffle chart populated!')


# In[ ]:


#Let's take a peek at how the matrix looks like.
waffle_chart


# # Step 5.
# Map the waffle chart matrix into a visual.

# In[ ]:


# instantiate a new figure object 
fig = plt.figure()

# use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap =colormap)
plt.colorbar()


# # Step 6. 
# Prettify the chart.

# In[ ]:


# instantiate a new figure object
fig = plt.figure()

# use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap = colormap)
plt.colorbar()

#get the axis
ax = plt.gca()

#set minor ticks
ax.set_xticks(np.arange(-.5, (width),1), minor =True)
ax.set_yticks(np.arange(-.5, (height), 1), minor=True)

#add grilines based on minor ticks
ax.grid(which ="minor", color="w", linestyle = "-", linewidth=2)

plt.xticks([])
plt.yticks([])


# # Step 7. 
# Create a legend and add it to chart.

# In[ ]:


# instantiate a new figure object
fig =plt.figure()

# use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap = colormap)
plt.colorbar

#get the axis
ax = plt.gca()

#set minor ticks
ax.set_xticks(np.arange(-.5, (width), 1), minor = True)
ax.set_yticks(np.arange(-.5, (height), 1), minor =True)

# add gridlines based on minor ticks
ax.grid(which="minor", color="w", linestyle="-", linewidth =2)

plt.xticks([])
plt.yticks([])

# compute cumulative sum of individual categories to match color schemes between chart and legend
values_cumsum  = np.cumsum(three_country["Total"])
total_values = values_cumsum[len(values_cumsum)-1]

#create legend 
legend_handles = []
for i, category in enumerate(three_country.index.values):
    label_str = category + ' (' + str(three_country['Total'][i]) + ')'
    color_val = colormap(float(values_cumsum[i])/total_values)
    legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

# add legend to chart
plt.legend(handles=legend_handles,
           loc='lower center', 
           ncol=len(three_country.index.values),
           bbox_to_anchor=(0., -0.2, 0.95, .1)
          )


# Now it would very inefficient to repeat these seven steps every time we wish to create a waffle chart. So let's combine all seven steps into one function called create_waffle_chart. This function would take the following parameters as input:
# 
# categories: Unique categories or classes in dataframe.    values: Values corresponding to categories or classes.    height: Defined height of waffle chart.   
# width: Defined width of waffle chart.
# colormap: Colormap class
# value_sign: In order to make our function more generalizable, we will add this parameter to address signs that could be associated with a value such as %, $, and so on. value_sign has a default value of empty string.

# In[ ]:


def create_waffle_chart(categories, values, height, width, colormap,value_sign=""):
    
    
    # compute the proportion of each category with respect to the total
    total_values = sum(values)
    category_proportions = [(float(value) / total_values) for value in values]
    # compute the total number of tiles
    total_num_tiles = width*height #total number of tiles
    print("Total number of tiles is ", total_num_tiles)
    
    # print out number of tiles per category
    for i, tiles in enumerate(tiles_per_category):
        print(three_country.index.values[i] + ": " + str(tiles))
        
    # initialize the waffle chart as an empty matrix
    waffle_chart =np.zeros((height, width))
    
    # define indices to loop through waffle chart
    category_index = 0
    tile_index = 0
    
    # populate the waffle chart
    for col in range(width):
        for row in range(height):
            tile_index += 1
            
            # if the number of tiles populated for the current category 
            # is equal to its corresponding allocated tiles...
            if tile_index > sum(tiles_per_category[0:category_index]):
                # ...proceed to the next category
                category_index += 1 
                
            # set the class value to an integer, which increases with class
            waffle_chart[row, col] =category_index
            
    # instantiate a new figure object
    fig =plt.figure()
    
    # use matshow to display the waffle chart
    colormap = plt.cm.coolwarm
    plt.matshow(waffle_chart, cmap = colormap)
    plt.colorbar()
    
    #get the axis
    ax = plt.gca()
    
    # set minor ticks
    ax.set_xticks(np.arange(-.5, (width), 1), minor =True)
    ax.set_yticks(np.arange(-.5, (height), 1), minor = True)
    
    # add dridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    plt.xticks([])
    plt.yticks([])

     # compute cumulative sum of individual categories to match color schemes between chart and legend
    values_cumsum = np.cumsum(values)
    total_values = values_cumsum[len(values_cumsum) - 1]

    # create legend
    legend_handles = []
    for i, category in enumerate(categories):
        if value_sign == '%':
            label_str = category + ' (' + str(values[i]) + value_sign + ')'
        else:
            label_str = category + ' (' + value_sign + str(values[i]) + ')'
            
        color_val = colormap(float(values_cumsum[i])/total_values)
        legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

    # add legend to chart
    plt.legend(
        handles=legend_handles,
        loc='lower center', 
        ncol=len(categories),
        bbox_to_anchor=(0., -0.2, 0.95, .1)
    )   
        


# Now to create a waffle chart, all we have to do is call the function create_waffle_chart. Let's define the input parameters:

# In[ ]:


width = 40
height = 10

categories = three_country.index.values
values = three_country["Total"]

colormap = plt.cm.coolwarm


# In[ ]:


create_waffle_chart(categories, values,height,width, colormap)


# In[ ]:


df.head()


# And what was the total immigration from 1980 to 2013?

# In[ ]:


total_immigration = df["Total"].sum()
total_immigration


# # Word Clouds 
# Word clouds (also known as text clouds or tag clouds) work in a simple way: the more a specific word appears in a source of textual data (such as a speech, blog post, or database), the bigger and bolder it appears in the word cloud.
# 
# Luckily, a Python package already exists in Python for generating word clouds. The package, called word_cloud was developed by Andreas Mueller. You can learn more about the package by following this link.
# 
# Let's use this package to learn how to generate a word cloud for a given text document.
# 
# First, let's install the package.

# In[ ]:


# install wordcloud
get_ipython().system('conda install -c conda-forge wordcloud==1.4.1 --yes')

# import package and its set of stopwords
from wordcloud import WordCloud, STOPWORDS

print ('Wordcloud is installed and imported!')


# Using countries with single-word names, let's duplicate each country's name based on how much they contribute to the total immigration.

# In[ ]:


max_words =90
word_string = ""
for country in df.index.values:
    # check if country's name is a single-word name
    if len(country.split(' ')) == 1:
        repeat_num_times = int(df.loc[country,'Total']/float(total_immigration)*max_words)
        word_string = word_string +((country + " ")*repeat_num_times)
        
# display the generated text
word_string


# We are not dealing with any stopwords here, so there is no need to pass them when creating the word cloud.

# In[ ]:


# create the word cloud
wordcloud = WordCloud(background_color = "black").generate(word_string)
print('Word cloud created!')


# In[ ]:


# display the cloud
fig = plt.figure()
fig.set_figwidth(14)
fig.set_figheight(18)

plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# According to the above word cloud, it looks like the majority of the people who immigrated came from one of 15 countries that are displayed by the word cloud. One cool visual that you could build, is perhaps using the map of Canada and a mask and superimposing the word cloud on top of the map of Canada. That would be an interesting visual to build!

# # Regression Plots 
# Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics. You can learn more about seaborn by following this link and more about seaborn regression plots by following this link.
# 
# In lab Pie Charts, Box Plots, Scatter Plots, and Bubble Plots, we learned how to create a scatter plot and then fit a regression line. It took ~20 lines of code to create the scatter plot along with the regression fit. In this final section, we will explore seaborn and see how efficient it is to create regression lines and fits using this library!
# 
# Let's first install seaborn

# In[ ]:


# install seaborn
get_ipython().system('conda install -c anaconda seaborn --yes')

# import library
import seaborn as sns

print('Seaborn installed and imported!')


# Create a new dataframe that stores that total number of landed immigrants to Canada per year from 1980 to 2013.

# In[ ]:


# we can use the sum() method to get the total population per year
df_tot = pd.DataFrame(df[years].sum(axis=0))

# change the years to type float (useful for regression later on)
df_tot.index = map(float, df_tot.index)
# reset the index to put in back in as a column in the df_tot dataframe
df_tot.reset_index(inplace = True)

#rename columns
df_tot.columns =['year', 'total']
#view the final dataframe
df_tot.head()


# With seaborn, generating a regression plot is as simple as calling the regplot function.

# In[ ]:


import seaborn as sns
ax = sns.regplot(x = 'year', y= "total", data=df_tot)


# This is not magic; it is seaborn! You can also customize the color of the scatter plot and regression line. Let's change the color to green.

# In[ ]:


ax = sns.regplot(x= "year", y = "total", data = df_tot, color="green")


# You can always customize the marker shape, so instead of circular markers, let's use '+'.

# In[ ]:


#Let's blow up the plot a little bit so that it is more appealing to the sight.
plt.figure(figsize=(15, 10))

ax = sns.regplot(x="year", y ="total", data=df_tot, color="green", marker="+")


# And let's increase the size of markers so they match the new size of the figure, and add a title and x- and y-labels.

# In[ ]:


plt.figure(figsize=(15, 10))
ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})

ax.set(xlabel="Year", ylabel ="Total immigration") #add x - and y-labels
ax.set_title("Total Immigration to Canada from 1980-2013") #add title


# And finally increase the font size of the tickmark labels, the title, and the x- and y-labels so they don't feel left out!

# In[ ]:


plt.figure(figsize = (15,10))

sns.set(font_scale=1.5)

ax = sns.regplot(x="year", y="total", data= df_tot, color="green", marker = "+", scatter_kws ={"s":200})
ax.set(xlabel="Year", ylabel= "Total Immigration")
ax.set_title("Total Immigration to Canada from 1980 - 2013")


# **Question:** Use seaborn to create a scatter plot with a regression line to visualize the total immigration from Denmark, Sweden, and Norway to Canada from 1980 to 2013.

# In[ ]:


three_country.head()


# In[ ]:


three_total = pd.DataFrame(three_country[years].sum(axis=0))
three_total.index = map(float, three_total.index)

three_total.reset_index(inplace=True)
three_total.columns = ['year',"total"]
three_total.head()


# In[ ]:


plt.figure(figsize=(10,10))
ax= sns.regplot(x="year", y="total", data= three_total, color="orange", marker="+")

sns.set(font_scale=1.5)
sns.set_style('whitegrid')


ax.set(xlabel ='Year', ylabel="Total Immigration")
ax.set_title("Total immigration of three country to Canada ")


# #         Generating Maps with Python
# # Introduction
# 
# In this lab, we will learn how to create maps for different objectives. To do that, we will part ways with Matplotlib and work with another Python visualization library, namely Folium. What is nice about Folium is that it was developed for the sole purpose of visualizing geospatial data. While other libraries are available to visualize geospatial data, such as plotly, they might have a cap on how many API calls you can make within a defined time frame. Folium, on the other hand, is completely free.

# # Datasets:
# 
# 
# 
# 1. Immigration to Canada from 1980 to 2013 - International migration flows to and from selected countries - The 2015 revision from United Nation's website. The dataset contains annual data on the flows of international migrants as recorded by the countries of destination. The data presents both inflows and outflows according to the place of birth, citizenship or place of previous / next residence both for foreigners and nationals. For this lesson, we will focus on the Canadian Immigration data

# # Introduction to Folium 
# Folium is a powerful Python library that helps you create several types of Leaflet maps. The fact that the Folium results are interactive makes this library very useful for dashboard building.
# 
# From the official Folium documentation page:
# 
# Folium builds on the data wrangling strengths of the Python ecosystem and the mapping strengths of the Leaflet.js library. Manipulate your data in Python, then visualize it in on a Leaflet map via Folium.
# 
# Folium makes it easy to visualize data that's been manipulated in Python on an interactive Leaflet map. It enables both the binding of data to a map for choropleth visualizations as well as passing Vincent/Vega visualizations as markers on the map.
# 
# The library has a number of built-in tilesets from OpenStreetMap, Mapbox, and Stamen, and supports custom tilesets with Mapbox or Cloudmade API keys. Folium supports both GeoJSON and TopoJSON overlays, as well as the binding of data to those overlays to create choropleth maps with color-brewer color schemes.
# 
# Let's install Folium
# Folium is not available by default. So, we first need to install it before we are able to import it.
# 
# 

# In[ ]:


get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')
import folium

print('Folium installed and imported!')


# Generating the world map is straigtforward in Folium. You simply create a Folium Map object and then you display it. What is attactive about Folium maps is that they are interactive, so you can zoom into any region of interest despite the initial zoom level.
# 

# In[ ]:


#define the world map
world_map = folium.Map()

#display world map
world_map


# You can customize this default definition of the world map by specifying the centre of your map and the intial zoom level.
# 
# All locations on a map are defined by their respective Latitude and Longitude values. So you can create a map and pass in a center of Latitude and Longitude values of [0, 0].
# 
# For a defined center, you can also define the intial zoom level into that location when the map is rendered. **The higher the zoom level the more the map is zoomed into the center.**
# 
# 
# Let's create a map centered around Canada and play with the zoom level to see how it affects the rendered map.

# In[ ]:


# define the world map centered around Canada with a low zoom level
world_map = folium.Map(location = [56.130, -106.35], zoom_start=4)

#display world map
world_map


# Let's create the map again with a higher zoom level

# In[ ]:


# define the world map centered around Canada with a higher zoom level
world_map = folium.Map(location = [56.130, -105.35], zoom_start = 8)

world_map


# Question: Create a map of Mexico with a zoom level of 4.
# 

# In[ ]:


mexico_latitude = 23.6345 
mexico_longitude = -102.5528

Mexico_map = folium.Map(location=[mexico_latitude, mexico_longitude], zoom_start=4)
Mexico_map


# # A. Stamen Toner Maps
# These are high-contrast B+W (black and white) maps. They are perfect for data mashups and exploring river meanders and coastal zones.
# 
# Let's create a Stamen Toner map of canada with a zoom level of 4.

# In[ ]:


# create a Stamen Toner map of the world centered around Canada
world_map = folium.Map(location= [56.130, -106.35], zoom_start=4, tiles = 'Stamen Toner')
world_map


# # B. Stamen Terrain Maps
# These are maps that feature hill shading and natural vegetation colors. They showcase advanced labeling and linework generalization of dual-carriageway roads.
# 
# Let's create a Stamen Terrain map of Canada with zoom level 4.

# In[ ]:


# create a Stamen Toner map of the world centered around Canada
world_map = folium.Map(location=[56.130, -103.35], zoom_start=4, tiles="Stamen Terrain")
world_map


# # C. Mapbox Bright Maps
# These are maps that quite similar to the default style, except that the borders are not visible with a low zoom level. Furthermore, unlike the default style where country names are displayed in each country's native language, Mapbox Bright style displays all country names in English.
# 
# Let's create a world map with this style.

# In[ ]:


# create a world map with a Mapbox Bright style.
world_map = folium.Map(tiles= 'Mapbox Bright')

world_map


# **Question:** Create a map of Mexico to visualize its hill shading and natural vegetation. Use a zoom level of 6.

# In[ ]:


Mexico_map = folium.Map(location=[mexico_latitude, mexico_longitude], zoom_start=6, tiles= "Stamen Terrain")
Mexico_map


# # Choropleth Maps
# A Choropleth map is a thematic map in which areas are shaded or patterned in proportion to the measurement of the statistical variable being displayed on the map, such as population density or per-capita income. The choropleth map provides an easy way to visualize how a measurement varies across a geographic area or it shows the level of variability within a region. Below is a Choropleth map of the US depicting the population by square mile per state.
# 
# Now, let's create our own Choropleth map of the world depicting immigration from various countries to Canada.
# 
# Let's first download and import our primary Canadian immigration dataset using pandas read_excel() method. Normally, before we can do that, we would need to download a module which pandas requires to read in excel files. This module is xlrd. For your convenience, we have pre-installed this module, so you would not have to worry about that. Otherwise, you would need to run the following line of code to install the xlrd module:
# 
# !conda install -c anaconda xlrd --yes Download the dataset and read it into a pandas dataframe:
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


df = pd.read_excel("/kaggle/input/immigration-to-canada-ibm-dataset/Canada.xlsx",
                     sheet_name='Canada by Citizenship',
                     skiprows = range(20),
                     skipfooter = 2)
                     


# In[ ]:


df.head()


# In order to create a Choropleth map, we need a GeoJSON file that defines the areas/boundaries of the state, county, or country that we are interested in. In our case, since we are endeavoring to create a world map, we want a GeoJSON that defines the boundaries of all world countries. For your convenience, we will be providing you with this file, so let's go ahead and download it. Let's name it world_countries.json.

# In[ ]:


# download countries geojson file
get_ipython().system('wget --quiet https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/world_countries.json -O world_countries.json')
    
print('GeoJSON file downloaded!')


# In[ ]:


# clean up the dataset to remove unnecessary columns (eg. REG) 
df.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)

# let's rename the columns so that they make sense
df.rename(columns={'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace=True)

# for sake of consistency, let's also make all column labels of type string
df.columns = list(map(str, df.columns))

# add total column
df['Total'] = df.sum(axis=1)

# years that we will be using in this lesson - useful for plotting later on
years = list(map(str, range(1980, 2014)))
print ('data dimensions:', df.shape)


# In[ ]:


df.head()


# In[ ]:


df.columns


# Now that we have the GeoJSON file, let's create a world map, centered around [0, 0] latitude and longitude values, with an intial zoom level of 2, and using Mapbox Bright style.

# In[ ]:


world_geo = r'world_countries.json' # geojson file

# create a plain world map
world_map = folium.Map(location=[0, 0], zoom_start=2, tiles='Mapbox Bright')


# And now to create a Choropleth map, we will use the choropleth method with the following main parameters:
# 1. geo_data, which is the GeoJSON file.
# 1. data, which is the dataframe containing the data.
# 1. columns, which represents the columns in the dataframe that will be used to create the Choropleth map.
# 1. key_on, which is the key or variable in the GeoJSON file that contains the name of the variable of interest. To determine that, you will need to open the GeoJSON file using any text editor and note the name of the key or variable that contains the name of the countries, since the countries are our variable of interest. In this case, name is the key in the GeoJSON file that contains the name of the countries. Note that this key is case_sensitive, so you need to pass exactly as it exists in the GeoJSON file.

# In[ ]:


# generate choropleth map using the total immigration of each country to Canada from 1980 to 2013
world_map.choropleth(
    geo_data=world_geo,
    data=df,
    columns=['Country', 'Total'],
    key_on='feature.properties.name',
    fill_color='Yl0rRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Immigration to Canada'
)
world_map


# As per our Choropleth map legend, the darker the color of a country and the closer the color to red, the higher the number of immigrants from that country. Accordingly, the highest immigration over the course of 33 years (from 1980 to 2013) was from China, India, and the Philippines, followed by Poland, Pakistan, and interestingly, the US.
# 
# Notice how the legend is displaying a negative boundary or threshold. Let's fix that by defining our own thresholds and starting with 0 instead of -6,918!

# 

# In[ ]:


world_geo = r'world_countries.json'

# create a numpy array of length 6 and has linear spacing from the minium total immigration to the maximum total immigration
threshold_scale = np.linspace(df['Total'].min(),
                              df['Total'].max(),
                              6, dtype=int)
threshold_scale = threshold_scale.tolist() # change the numpy array to a list
threshold_scale[-1] = threshold_scale[-1] + 1 # make sure that the last value of the list is greater than the maximum immigration

# let Folium determine the scale.
world_map = folium.Map(location=[0, 0], zoom_start=2, tiles='Mapbox Bright')
world_map.choropleth(
    geo_data=world_geo,
    data=df,
    columns=['Country', 'Total'],
    key_on='feature.properties.name',
    threshold_scale=threshold_scale,
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Immigration to Canada',
    reset=True
)
world_map


# In[ ]:




