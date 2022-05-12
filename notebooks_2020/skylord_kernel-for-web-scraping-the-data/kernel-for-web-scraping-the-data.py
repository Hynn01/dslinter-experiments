#!/usr/bin/env python
# coding: utf-8

# ## **Updated Notebook**
# 
# 
# ## Change log
# 
# *Date: 11th May 2020*
# 
# Notebook updated to include the concatenated dataset. This includes all files from previous runs. Check [kernel](https://www.kaggle.com/skylord/kernel-for-concatenating-all-files) here.
# 
# Two files are created: 1# Original raw data 2# Concatenated file
# 
# *Date: 8th May 2020*
# 
# Updated
# 
# *Date: 5th May 2020*
# 
# Updated
# 
# *Date: 1st May 2020*
# 
# Updated till date
# Minor changes in column names
# 
# ```
# Tests -> Tested
# Tests /millionpeople -> Tested /millionpeople
# 
# ```
# 
# New Column `%` added
# 
# *Date:  26th April 2020*
# 
# This was long delayed!  
# 
# *Date:  15th April 2020*
# 
# Changes in table structure: Added columns: Date (change in column name), Positive /millionpeople (instead of Positive / thousands) and other changes
# Commented out extraction from OurWorldInData. Its now available from their github repo
# 
# *Date:  9th April 2020*
# 
# Changes in table structure
# 
# *Created Date: 31st March 2020*
# 
# The notebook has been updated to include web scraping of the new source of information.The updated notebook is appended at the end of the file. The new file takes information from the following source: 
# 
# [Wiki link for Covid-19 Testing](https://en.wikipedia.org/wiki/COVID-19_testing)
# 
# New columns include: positive/confirmed cases, tests conducted per million of population, positive cases per million & source information!
# 

# In[ ]:


# Load the required libraries
import re
import os

import requests
from bs4 import BeautifulSoup
#import datefinder

import pickle
import time
import pandas as pd 


# In[ ]:


import io, requests
from io import StringIO

url = "https://github.com/owid/covid-19-data/blob/master/public/data/testing/covid-testing-all-observations.csv"
content = requests.get(url).content.decode('utf-8')

with open('covid-testing-all-observations.csv', 'w+') as f:
    f.write(content)


# In[ ]:


# tests_df = pd.read_csv('covid-testing-all-observations.csv', index_col=0)

# print(tests_df.shape)
# tests_df.head()


# In[ ]:


#! git clone https://github.com/owid/covid-19-data # <- Uncomment if you want to clone the github repo
#!ls covid-19-data/public


# In[ ]:


# # Define URL for webpage <<- This URL doesn't work so have commented up the old code # Date: 9th April 2020
# url = 'https://ourworldindata.org/coronavirus-testing-source-data#population-estimates-to-calculate-tests-per-million-people'

# try:
#     resp = requests.get(url)
#     soup = BeautifulSoup(resp.text, 'html.parser')
# except:
#     print("Error: In acquiring page data")
    
# # Find table
# allTable = soup.find_all('table')

# First row has the table header.
#headings = [cell.get_text().strip() for cell in allTable[0].find("tr").find_all('td')]

# # Get the remaining rows 
# datasets = list() 
# for row in allTable[0].find_all("tr")[1:]:
#     dataset = dict(zip(headings, (td.get_text().strip() for td in row.find_all("td"))))
# #     datasets.append(dataset)

# # Convert to a dataframe
# dataset_df = pd.DataFrame(datasets)

# # Remove commas from the number of tests 30,098 => 30098
# dataset_df['Total tests'] = dataset_df['Total tests'].apply(lambda x: x.replace(",", ""))

# dataset_df.dtypes
# dataset_df.to_excel("Tests_Conducted.xlsx", index=False)


# ******************************************************************
# 
# # Updated notebook starts from here 
# # Date: 31st March 2020
# ******************************************************************
# 

# In[ ]:


url = 'https://en.wikipedia.org/wiki/COVID-19_testing' # define url to be scraped


# In[ ]:


# Parse the html page
try:
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
except:
    print("Error: In acquiring page data")


# In[ ]:


# Find table
allTable = soup.find_all('table') # allTable[2] has the data of interest


# In[ ]:


# find the idx to the required table
for idx in range(len(allTable)):
    print(allTable[idx]['class'])
    headings = [cell.get_text().strip() for cell in allTable[idx].find("tr").find_all('th')]
    print(headings)
    try:
        if headings[0].__contains__('Country'):
            print(idx)
            break
    except:
        continue
    print("*************************************************")
#     if allTable[idx]['class'][0].find('covid19-testing') == 0:
#         print(allTable[idx]['class'])
#         break


# In[ ]:


# Get the headings & make some changes to their names

headings = [cell.get_text().strip() for cell in allTable[idx].find("tr").find_all('th')]
headings


# In[ ]:


allRows = allTable[idx].find_all("tbody")[0].find_all('tr')
len(allRows)


# In[ ]:


# Main loop for capturing the information

datasets = list() 
for row in allRows[1:]:
    cellValue = row.find_all("td")
    
    dataset = dict(zip(headings[1:], (td.get_text().strip() for td in cellValue)))
    
    try:
        dataset['Country'] = row.find('th').get_text().strip() # Get country or region info
    except:
        print(len(datasets))
        break
    
    getA = cellValue[-1].find_all('a', href=True) # Get citation values
    dataset['Ref.'] = " "
    
    for idx in range(len(getA)):
         dataset['Ref.'] = dataset['Ref.'] + getA[idx]['href'].replace("#", "") + " "
    
    datasets.append(dataset)


# In[ ]:


dataset_df = pd.DataFrame(datasets)
dataset_df.columns


# In[ ]:


dataset_df.rename(columns={'Date[a]':'Date', 'Units[b]': 'Units', 'Confirmed(cases)': 'Positive', 
                          'Confirmed /millionpeople':'Positive /millionpeople'}, inplace=True)

dataset_df = dataset_df[['Country', "Date",'Tested', 'Units', 'Positive', "%", 'Tested /millionpeople', 'Positive /millionpeople', 'Ref.' ] ]
print(dataset_df.columns)
print(dataset_df.shape)
dataset_df.head(10)


# In[ ]:


# Get reference section of the page
refSoup = soup.find_all('ol')
len(refSoup)


# In[ ]:



allRefId = refSoup[-1].find_all('li')
len(allRefId)


# In[ ]:


# parse the reference list 
refList = list()
for refId in allRefId:
    refDict = dict()
    refDict['Ref.'] = refId.get('id')   #getA[0].get('href').strip().replace("ref", "note")
    
    getA = refId.find_all('a')
    for i in range(len(getA)):
        href = getA[i].get('href').strip()
        if "http" in href: 
            refDict['Source'] = href
    
    refList.append(refDict)


# In[ ]:


ref_df = pd.DataFrame(refList)
print(ref_df.shape)
ref_df.head()


# In[ ]:


# Merge the two dataframes, to get source URL
# This is kind of round-about way of doing it, Iff someone can suggest a better method. Pls do it

source1 = list()
source2 = list()

for idx, row in dataset_df.iterrows():
    refs = row['Ref.'].split()
    try:
        source1.append(ref_df[ref_df['Ref.']==refs[0]]['Source'].values[0])
    
        if len(refs) > 1:
            source2.append(ref_df[ref_df['Ref.']==refs[1]]['Source'].values[0])
        else:
            source2.append(" ")
    except:
        source1.append(" ")
        source2.append(" ")
        
assert len(source1) == len(source2)


# In[ ]:


dataset_df.loc[:,('Source_1')] = source1
dataset_df.loc[:,('Source_2')] = source2
dataset_df.head()


# **Now we will try to compress the data & convert it to numerics, so that we can directly use it in our output**

# In[ ]:


dataset_df.dtypes


# In[ ]:


# Replace the commas >>> 2,183 --> 2183
dataset_df['Tested'] = dataset_df['Tested'].apply(lambda x: x.replace(',','').strip()) 
dataset_df.loc[:, ('Positive')] = dataset_df['Positive'].apply(lambda x: x.replace(',','').strip())
dataset_df.loc[:, ('Tested /millionpeople')] = dataset_df['Tested /millionpeople'].apply(lambda x: x.replace(',','').strip())
dataset_df.loc[:, ('Positive /millionpeople')] = dataset_df['Positive /millionpeople'].apply(lambda x: x.replace(',','').strip())


# In[ ]:


#dataset_df.loc[dataset_df['Tests'] == '83800*', 'Tests'] = 83800 # For US-California
#dataset_df['Tests'] = dataset_df['Tests'].astype('int')
dataset_df.loc[:, ('Tested')] = pd.to_numeric(dataset_df['Tested'], errors='coerce')


# In[ ]:


dataset_df.loc[:, ('Positive')] = pd.to_numeric(dataset_df['Positive'], errors='coerce')
dataset_df.loc[:, ('%')] = pd.to_numeric(dataset_df['%'], errors='coerce')

dataset_df.loc[:, ('Tested /millionpeople')] = pd.to_numeric(dataset_df['Tested /millionpeople'], errors='coerce')
dataset_df.loc[:, ('Positive /millionpeople')] = pd.to_numeric(dataset_df['Positive /millionpeople'], errors='coerce')


# In[ ]:


dataset_df.dtypes


# In[ ]:


dataset_df.head(15)


# In[ ]:


dataset_df.to_csv('Tests_conducted_13July2020.csv', index=False)


# In[ ]:


allData = pd.read_csv('/kaggle/input/covid19-tests-conducted-by-country/TestsConducted_AllDates_09June2020.csv')
print(allData.shape)
allData.head()


# In[ ]:


dataset_df.columns


# In[ ]:


dataset_df = dataset_df[['Country', 'Date', 'Tested', 'Units', 'Positive', '%', 'Source_1','Source_2' ]]
dataset_df.rename(columns={'%': 'Positive/Tested %'}, inplace=True)
dataset_df['FileDate'] = '13-July-2020'


# In[ ]:


allData = pd.concat([dataset_df, allData], axis=0, sort=False)
print(allData.shape)
print(allData.columns)
allData.head()


# In[ ]:


allData.to_csv('TestsConducted_AllDates_13July2020.csv', index=False)


# In[ ]:




