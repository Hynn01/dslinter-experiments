#!/usr/bin/env python
# coding: utf-8

# ## A4: Manipulating Data
# This assignment is focused on working with dictionaries and manipulating data. Follow the instructions for each step below. After each step, insert your Code Cell with your solution if necessary (in some steps there will be some code provided for you).The assignment is in two parts (A and B). Part A focuses on data manipulation and using dictionaries. Part B focuses on retrieving data to update the dictionary.
# 
# ### Submission
# When you have finished, submit this homework by sharing your Kaggle notebook. Click Commit in the upper right of your Kaggle notebook screen. Click on Open Version to view it. Make sure to set Sharing permissions to public. Then copy the URL for that version. To submit on Canvas, click Submit Assignment and paste the link into the URL submission field.

# # PART A
# ### Step 1: Accessing values at a specified key in a dictionary.
# 
# We have created a list of cities to keep track of what current populations are and potentially identify which cities are growing fastest, what the weather is like there, etc. Add code to print the number of residents in Chiago the our `citytracker` list. (the value associated with key 'Chicago' in the dictionary dinocount). Hint: this is just one simple line of code. Add the code below.

# In[ ]:


citytracker = {'Atlanta': 486290, 'Boston': 685094, 'Chicago': 2749360, 'Denver': 619968, 'Seattle': 724725}


# ### Step 2: Incrementing the value of a dictionary at a key.
# 
# Write code to increment the number of residents in Seattle by 17,500 (that happened in one month in Summer 2018!). In other words, add 17,500 to the existing value of cities at key 'Seattle').  Then, print out the number of residents in the Seattle.

# ### Step 3: Adding an entry to a dictionary. 
# 
# Our list of cities just got bigger. (What could go wrong?) Write code to insert a new key, 'Los Angeles' into the dictionary, with a value of 45000. Verify that it worked by printing out the value associated with the key 'Los Angeles'

# ### Step 4: Concatenating strings and integers. 
# 
# Write code that creates a string that says 'Denver: X', where X is the number of residents extracted from the `citytracker` dictionary.  Print the string. Hint: you will need to use the + string concatenation operator in conjunction with str() or another string formatting instruction.

# ### Step 5: Iterating over keys in a dictionary.  
# 
# Write code that prints each city (key), one line at a time, using a for loop.

# ### Step 6: iterating over keys to access values in a dictionary. 
# 
# Write code that prints each city (key), followed by a colon and the number of residents (e.g., Seattle : 724725), one line at a time using a for loop.

# ### Step 7: Testing membership in a dictionary.
# 
# Write code to test whether 'New York' is in the `citytracker` dictionary.  If the test yields `true`, print `New York: <x>`, where `<x>` is the current population. If the test yields false, it should print "Sorry, that is not in the Coty Tracker. Do the same thing for the key 'Atlanta'.

# ### Step 8: Default values
# 
# We have a list of potential cities (in the list potentialcities) and we want to check whether the city exists within the City Tracker. If it is, we want to print `city: #`, where city is the city name and # is the population. If the city is not in the dictionary, it should print zero. Add to the code below to do this. *Hint: you can use default values here to make this take less code!*
# 

# In[ ]:


potentialcities = ['Cleveland','Phoenix','Nashville','Philadelphia','Milwaukee']


# ### Step 9: Printing comma separated data from a dictionary.
# 
# You may have worked with comma separated values before: they are basically spreadsheets or tables represented as plain text files, with each row represented on a new line and each cell divided by a comma. Print out the keys and values of the dictionary stored in `citytracker`. The keys and values you print should be separated only by commas (there should be no spaces). Print each `key:value` pair on a different line. *Hint: this is almost identical to Step 6*

# ### Step 10: Saving a dictionary to a CSV file
# Write key and value pairs from `citytracker` out to a file named 'popreport.csv'. *Hint: the procedure is very close to that of Step 9.* You should also include a header to row describe each column, labeling them as "city" and "pop", and subsequent lines should contain the data. (Here is an example to refer to: https://pythonspot.com/save-a-dictionary-to-a-file)

# In[ ]:


import os

### This will print out the list of files in your /working directory to confirm you wrote the file.
### You can also examine the right sidebar to see your file.

for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
### Add your code here


# # PART B
# In this part, you will practice useing API keys to access geographic location and weather information for cities in the `citytracker` dictionary. You need to get the geolocation and weather for each item and store it in the same (or a new) dictionary. You then print out the data and format it to be pretty (whatever that means to you) and finally, write it out as a json file with a timestamp.
# 
# **You will need to enable Internet connections in the sidebar to the right of your notebook. If you get a `Connection error` in the Console, it is because your notebook can't access the internet.**
# 
# You will access two different APIs to get this information:
# 1. OpenCage (https://opencagedata.com/api) which is a reverse-geocoding API (you provide them a location and they give your it's geo-coordinates in Lat and Lon)
# 2. Openweathermap API (https://openweathermap.org/api)
# 
# Note: You can get LAT and LON from Openweathermap, but for this assignment, you should get them from OpenCage.
# 
# ### Step 1: Accessing APIs to retrieve data
# First, you will need to request an API Secret Key from OpenCage (https://opencagedata.com/api) and add it to your Kaggle notebook in the Add-ons menu item. Once you have the Secret Key, you attach it to this notebook (click the checkbox) so you can make the API call. Make sure the **Label** for your key in your Kaggle Secrets file is what you use in your code below.
# 
# You will also an API Secret Key from OpenWeatherMapAPI (https://openweathermap.org/api). Attach it to this notebook and use it in the code. Make sure you have created different labels for each key and use them in the code below. You can see how to make calls to the API here: https://openweathermap.org/current
# 
# Finally, make sure to install the `opencage` module in this notebook. Use the console at the bottom of the window and type `pip install opencage`. You should receive a confirmation message if it installs successfully.
# 
# Then try running the code cells below to see the output. Once the code sucessfully works for Seattle (which has been provided for you below), try typing in different cities instead to see the results to make sure it is working.
# 
# ### Step 2: Retreiving values for each city in your dictionary
# In the code cell below, add some code to try to get information for all of the cities in your `citytracker` dictionary. You can print the information out to make sure it is working. Store the results of `getForecast` for each city in your dictionary.
# 
# ### Step 3: Writing the datafile
# Save the results of your work as a JSON formatted output file in your Kaggle output folder and Commit your notebook. Make sure to make it public and submit the resulting URL in Canvas. (Hint: use the json `dumps()` method)
# 

# In[ ]:


import urllib.error, urllib.parse, urllib.request, json, datetime

# This code retrieves your key from your Kaggle Secret Keys file
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("openCageKey") #replace "openCageKey" with the key name you created!
secret_value_1 = user_secrets.get_secret("openweathermap") #replace "openweathermap" with the key name you created!

from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(secret_value_0)
query = 'Seattle'  # replace this city with cities from the names in your citytracker dictionary
results = geocoder.geocode(query)
lat = str(results[0]['geometry']['lat'])
lng = str(results[0]['geometry']['lng'])
print (f"{query} is located at:")
print (f"Lat: {lat}, Lon: {lng}")


def safeGet(url):
    try:
        return urllib.request.urlopen(url)
    except urllib2.error.URLError as e:
        if hasattr(e,"code"):
            print("The server couldn't fulfill the request.")
            print("Error code: ", e.code)
        elif hasattr(e,'reason'):
            print("We failed to reach a server")
            print("Reason: ", e.reason)
        return None

def getForecast(city="Seattle"):
    key = secret_value_1
    url = "https://api.openweathermap.org/data/2.5/weather?q="+city+"&appid="+key
    print(url)
    return safeGet(url)

data = json.load(getForecast())
print(data)
current_time = datetime.datetime.now() 

print(f"The current weather in Seattle is: {data['weather'][0]['description']}")
print("Retrieved at: %s" %current_time)

### You can add your own code here for Steps 2 and 3

