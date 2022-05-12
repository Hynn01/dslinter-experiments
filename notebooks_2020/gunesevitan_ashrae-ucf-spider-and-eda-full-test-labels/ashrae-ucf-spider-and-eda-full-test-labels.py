#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tqdm import tqdm
import gc
import os

import requests
from bs4 import BeautifulSoup
import json

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# building_metadata.csv
BUILDINGMETADATA_DTYPES = {'site_id': np.uint8, 'building_id': np.uint16, 'square_feet': np.int32, 'year_built': np.float32, 'floor_count': np.float32}
df_building_metadata = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv', dtype=BUILDINGMETADATA_DTYPES)

# weather_train.csv and weather_test.csv
WEATHER_DTYPES = {'site_id': np.uint8, 'air_temperature': np.float32, 'cloud_coverage': np.float32, 'dew_temperature': np.float32, 
                  'precip_depth_1_hr': np.float32, 'sea_level_pressure': np.float32, 'wind_direction': np.float32, 'wind_speed': np.float32}
df_weather_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv', dtype=WEATHER_DTYPES)
df_weather_test = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv', dtype=WEATHER_DTYPES)
df_weather = pd.concat([df_weather_train, df_weather_test], ignore_index=True)

# train.csv
TRAIN_DTYPES = {'building_id': np.uint16, 'meter': np.uint8, 'meter_reading': np.float32}
df_train = pd.read_csv('../input/ashrae-energy-prediction/train.csv', dtype=TRAIN_DTYPES)

# test.csv
TEST_DTYPES = {'building_id': np.uint16, 'meter': np.uint8}
df_test = pd.read_csv('../input/ashrae-energy-prediction/test.csv', dtype=TEST_DTYPES)
df_test.drop(columns=['row_id'], inplace=True)
    
# Keeping site 0
df_train = df_train[df_train['building_id'] < 105]
df_test = df_test[df_test['building_id'] < 105]
df_site0 = pd.concat([df_train, df_test], ignore_index=True, sort=False)

for df in [df_site0, df_weather]:
    df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)

df_site0 = df_site0.merge(df_building_metadata, on='building_id', how='left')
df_site0 = df_site0.merge(df_weather, on=['site_id', 'timestamp'], how='left')

del df_train, df_test, df_weather_train, df_weather_test, df_weather
gc.collect()

print('Site 0 Shape = {}'.format(df_site0.shape))
print('Site 0 Memory Usage = {:.2f} MB'.format(df_site0.memory_usage().sum() / 1024**2))
print('Site 0 Buildings with Electricity Meter = {}'.format(len(df_site0[df_site0['meter'] == 0]['building_id'].unique())))
print('Site 0 Buildings with Chilled Water Meter = {}'.format(len(df_site0[df_site0['meter'] == 1]['building_id'].unique())))


# ## **1. UCF (University of Central Florida) Spider**
# Building metadata is scraped from https://www.oeis.ucf.edu/buildings, and there are two extra features in the page.
#   * `EUI (kBTU/sqft)`: Energy per square foot per year **Reference**: https://www.energystar.gov/buildings/facility-owners-and-managers/existing-buildings/use-portfolio-manager/understand-metrics/what-energy
#   * `LEED`: (**Leadership in Energy and Environmental Design**) is the most widely used green building rating system in the world

# In[ ]:


SITE_0_START_URL = 'https://www.oeis.ucf.edu/buildings'
SITE_0_AREAS = df_site0['square_feet'].unique()

buildings = BeautifulSoup(requests.get(SITE_0_START_URL).text, 'html.parser')

building_names = [link.text.strip() for link in buildings.select('table#buildings tr th a')]
building_links = [link.get('href') for link in buildings.select('table#buildings tr th a')]
building_types = [link.text.strip() for link in buildings.select('table#buildings tr td:nth-child(3)')]
building_areas = [link.text.strip() for link in buildings.select('table#buildings tr td:nth-child(4)')]
building_euis = [link.text.strip() for link in buildings.select('table#buildings tr td:nth-child(5)')]
building_leeds = [link.text.strip() for link in buildings.select('table#buildings tr td:nth-child(6)')]

site0_building_metadata = {k: v  for k, v in enumerate(zip(building_names, building_links, building_types, building_areas, building_euis, building_leeds))}
df_site0_building_metadata = pd.DataFrame(site0_building_metadata).T.replace('', np.nan)
df_site0_building_metadata.columns = ['building_name', 'building_link', 'building_type', 'square_feet', 'eui', 'leed']
df_site0_building_metadata['building_url_code'] = df_site0_building_metadata['building_link'].str.split('/', expand=True)[4] # Going to use this while sending AJAX requests
df_site0_building_metadata['square_feet'] = df_site0_building_metadata['square_feet'].astype(np.uint32)
df_site0_building_metadata['eui'] = df_site0_building_metadata['eui'].astype(np.float32)
df_site0_building_metadata = df_site0_building_metadata[df_site0_building_metadata['square_feet'].isin(SITE_0_AREAS)] # square_feet values don't exist in competition data are excluded

del building_names, building_links, building_types, building_areas, building_euis, building_leeds, site0_building_metadata, SITE_0_AREAS
gc.collect()


# Building meter readings are scraped from https://www.oeis.ucf.edu/getData. API returns the records of meter readings from given time period in json format. The AJAX request requires a `building_url_code` for returning the records of a specific building. The `building_url_code` values in `df_site0_building_metadata` are iterated and used in the request, and the scraped meter readings concatenated into a DataFrame.

# In[ ]:


SITE_0_AJAX_URL = 'https://www.oeis.ucf.edu/getData'
BUILDING_URL_CODES = df_site0_building_metadata['building_url_code'].unique().tolist()
PARAMS = {
    'building': None,
    'start-date': '01/01/2016',
    'end-date': '01/01/2019',
    'resolution': 'hour',
    'filetype': 'json'    
}

df_site0_labels = pd.DataFrame(columns=['meter', 'meter_reading', 'timestamp', 'building_url_code'])

for building_url_code in tqdm(BUILDING_URL_CODES):
    PARAMS['building'] = building_url_code
    building_readings = json.loads(requests.post(url=SITE_0_AJAX_URL, params=PARAMS).text) 

    for meter_type in building_readings:
        
        if meter_type['key'] == 'Gas' or meter_type['key'] == 'Irrigation' or meter_type['key'] == 'Water':
            continue
        
        timestamps = pd.Series([value['timestamp'] for value in meter_type['values']])
        meter_readings = pd.Series([value['reading'] for value in meter_type['values']])
        meter_types = pd.Series(np.tile(meter_type['key'], len(timestamps)))
        building_url_codes = pd.Series(np.tile(building_url_code, len(timestamps)))

        df_meter_reading = pd.DataFrame(columns=['meter', 'meter_reading', 'timestamp', 'building_url_code'])
        df_meter_reading['timestamp'] = timestamps
        df_meter_reading['meter_reading'] = meter_readings
        df_meter_reading['meter'] = meter_types
        df_meter_reading['building_url_code'] = building_url_codes

        df_site0_labels = pd.concat([df_site0_labels, df_meter_reading], ignore_index=True)
        
df_site0_labels = df_site0_labels[df_site0_labels['timestamp'] < '2019-01-01 00:00:00']


# ## **2. Data Cleaning**
# 
# * Unique `square_feet` values are directly matched with their correct `building_id`
# * Not unique `square_feet` values are manually matched with their correct `building_id` by checking the training set labels

# In[ ]:


# Unique square_feet values in site 0
site0_unique_areas = df_site0_building_metadata['square_feet'].value_counts()[df_site0_building_metadata['square_feet'].value_counts() < 2].index.tolist()
df_site0_unique_areas = df_building_metadata[df_building_metadata['square_feet'].isin(site0_unique_areas) & (df_building_metadata['site_id'] == 0)]
area_building_id_mapping = df_site0_unique_areas.set_index('square_feet')['building_id'].to_dict()
df_site0_building_metadata['building_id'] = df_site0_building_metadata['square_feet'].map(area_building_id_mapping)

# Not Unique square_feet values in site 0
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "86"').index, 'building_id'] = '27'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "149"').index, 'building_id'] = '90'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "12"').index, 'building_id'] = '33'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "92"').index, 'building_id'] = '61'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "68"').index, 'building_id'] = '49'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "142"').index, 'building_id'] = '67'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "131"').index, 'building_id'] = '77'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "7"').index, 'building_id'] = '100'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "28"').index, 'building_id'] = '34'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "79"').index, 'building_id'] = '62'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "44"').index, 'building_id'] = '51'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "125"').index, 'building_id'] = '69'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "140"').index, 'building_id'] = '70'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "52"').index, 'building_id'] = '71'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "100"').index, 'building_id'] = '72'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "74"').index, 'building_id'] = '73'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "130"').index, 'building_id'] = '74'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "10"').index, 'building_id'] = '35'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "134"').index, 'building_id'] = '63'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "105"').index, 'building_id'] = '36'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "69"').index, 'building_id'] = '37'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "138"').index, 'building_id'] = '64'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "24"').index, 'building_id'] = '65'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "6"').index, 'building_id'] = '66'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "71"').index, 'building_id'] = '85'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "146"').index, 'building_id'] = '95'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "98"').index, 'building_id'] = '96'
df_site0_building_metadata.loc[df_site0_building_metadata.query('building_url_code == "30"').index, 'building_id'] = '98'

# building_id mapped to the scraped labels
df_site0_building_metadata['building_id'] = df_site0_building_metadata['building_id'].astype(np.uint16)
url_code_building_id_mapping = df_site0_building_metadata.set_index('building_url_code')['building_id'].to_dict()
df_site0_labels['building_id'] = df_site0_labels['building_url_code'].map(url_code_building_id_mapping)

# Removing unnecessary columns
df_site0_building_metadata.drop(columns=['building_name', 'building_link', 'building_type', 'building_url_code'], inplace=True)
df_site0_labels.drop(columns=['building_url_code'], inplace=True)


# Scraped data are standardized to the competition format
# * Scraped null readings are dropped
# * `meter` is label encoded
# * Chilled water readings are multiplied with **3.51684**  (1 ton of refrigeration) **Reference**: https://www.quora.com/What-is-the-1-tonnes-of-refrigeration-effect
# 
# Finally, standardized data merged to the competition data set.

# In[ ]:


df_site0_labels.drop(df_site0_labels.query('meter_reading.isnull()', engine='python').index, inplace=True)

df_site0_labels['meter'] = df_site0_labels['meter'].map({'Electric': 0, 'Chilled Water': 1})
df_site0_labels['timestamp'] = pd.to_datetime(df_site0_labels['timestamp'], infer_datetime_format=True)
df_site0_labels.loc[df_site0_labels.query('meter == 1').index, 'meter_reading'] = df_site0_labels.loc[df_site0_labels.query('meter == 1').index, 'meter_reading'] * 3.51684

df_site0_labels.sort_values(by=['timestamp', 'building_id', 'meter'], inplace=True)

df_site0 = df_site0.merge(df_site0_labels, on=['timestamp', 'building_id', 'meter'], how='left')    
df_building_metadata_external = df_building_metadata.merge(df_site0_building_metadata, on=['building_id', 'square_feet'], how='left')
df_site0.rename(columns={'meter_reading_x': 'meter_reading_original', 'meter_reading_y':'meter_reading_scraped'}, inplace=True)

del df_site0_labels
gc.collect()


# In[ ]:


df_site0[['building_id', 'meter', 'timestamp', 'meter_reading_original', 'meter_reading_scraped']].sample(10)


# In[ ]:


df_building_metadata_external


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndf_site0[['building_id', 'meter', 'timestamp', 'meter_reading_original', 'meter_reading_scraped']].to_csv('site0.csv.gz', index=False)\ndf_site0[['building_id', 'meter', 'timestamp', 'meter_reading_original', 'meter_reading_scraped']].to_pickle('site0.pkl')\n\ndf_building_metadata_external.to_csv('building_metadata_external.csv', index=False)\ndf_building_metadata_external.to_pickle('building_metadata_external.pkl')")


# ## **3. Scraped Yearly Readings**
# The differences and patterns learned from yearly readings of UCF can be useful, and could be translated to meter readings of other sites. In order to analyze patterns and missing readings, `df_site0` has to be assigned with 3 years of full timeframe index. Data analysis would be more accurate that way.

# In[ ]:


site0_meter0_buildings = sorted(df_site0[df_site0['meter'] == 0]['building_id'].unique())
site0_meter1_buildings = sorted(df_site0[df_site0['meter'] == 1]['building_id'].unique())

building_ids = sorted(np.unique(df_site0['building_id']))
meters = sorted(np.unique(df_site0['meter']))
df_site0 = df_site0.set_index(['building_id', 'timestamp', 'meter'], drop=False).sort_index()

site0_full_index = pd.MultiIndex.from_product([building_ids, pd.date_range(start='2016-01-01 00:00:00', end='2018-12-31 23:00:00', freq='H'), meters])

df_site0 = df_site0.reindex(site0_full_index)
df_site0['building_id'] = df_site0.index.get_level_values(0)
df_site0['timestamp'] = df_site0.index.get_level_values(1)
df_site0['meter'] = df_site0.index.get_level_values(2)

def plot_buildings(building_ids, meter_type):
    
    QUERY_2016 = 'building_id == {} and meter == {} and ("2016-01-01 00:00:00" <= timestamp < "2017-01-01 00:00:00")'
    QUERY_2017 = 'building_id == {} and meter == {} and ("2017-01-01 00:00:00" <= timestamp < "2018-01-01 00:00:00")'
    QUERY_2018 = 'building_id == {} and meter == {} and ("2018-01-01 00:00:00" <= timestamp < "2019-01-01 00:00:00")'
    
    fig, axes = plt.subplots(len(building_ids), figsize=(20, 80), dpi=100)    

    for i, building_id in enumerate(building_ids):
        df_site0.query(QUERY_2016.format(building_id, meter_type)).reset_index().set_index('level_1')['meter_reading_scraped'].plot(label='Training 2016', ax=axes[i])
        df_site0.query(QUERY_2017.format(building_id, meter_type)).reset_index().set_index('level_1')['meter_reading_scraped'].plot(label='Test 2017', ax=axes[i])
        df_site0.query(QUERY_2018.format(building_id, meter_type)).reset_index().set_index('level_1')['meter_reading_scraped'].plot(label='Test 2018', ax=axes[i])

        axes[i].legend()
        axes[i].set_title('building_id {} meter {}'.format(building_id, meter_type), fontsize=13);

        plt.subplots_adjust(hspace=0.45)

    plt.show()


# ### **3.1 Electricity Readings**
# Buildings consume electricity throughout the year with very few interruptions. Electricity readings looks more stable than the Chilled Water readings in this site. There are **14** timestamps at which the entire site has missing Electricity readings. Those **14** timestamps belong to **7** different days and none of those days are in the training set;
#   * **2017-02-14** (Valentine's Day)
#   * **2017-06-19**
#   * **2017-06-20**
#   * **2018-03-11**
#   * **2018-03-12**
#   * **2018-06-26**
#   * **2018-06-27**
#   
# Those dates could be related to planned maintenance days for the entire site or they could be related to meter errors. The other dates of missing electricity readings have fewer buildings. They are more likely to be random power outages, and they are more common in training set.

# In[ ]:


fig = plt.figure(figsize=(18, 10))
                 
df_site0.query('meter == 0 and meter_reading_scraped.isnull() and ("2016-01-01 00:00:00" <= timestamp < "2017-01-01 00:00:00")', engine='python')['timestamp'].value_counts().plot(label='Training 2016')
df_site0.query('meter == 0 and meter_reading_scraped.isnull() and ("2017-01-01 00:00:00" <= timestamp < "2018-01-01 00:00:00")', engine='python')['timestamp'].value_counts().plot(label='Test 2017')
df_site0.query('meter == 0 and meter_reading_scraped.isnull() and ("2018-01-01 00:00:00" <= timestamp < "2019-01-01 00:00:00")', engine='python')['timestamp'].value_counts().plot(label='Test 2018')

plt.title('Site 0 Building Count with Missing Electricity Readings in 2016, 2017 and 2018 ')
plt.ylabel('Building Count')
plt.legend()

plt.show()


# * There are excessive **0** readings and some huge outliers in the beginning of 2016. Those **0** readings are between **2016-01-01 00:00:00** - **2016-05-20 17:00:00** for most of the buildings. This phenomenon can be seen even in June and July for some buildings, and they don't represent the meter readings of next years.
# * There are huge negative outliers in some buildings. They could be related to random power outages because they don't have a date or time pattern. They occur right before a huge positive outlier. Buildings with negative outliers: **(0, 25, 38, 41, 77, 78, 86)**

# In[ ]:


plot_buildings(range(0, 12), 0)


# In[ ]:


plot_buildings(range(12, 24), 0)


# In[ ]:


plot_buildings(range(24, 36), 0)


# In[ ]:


plot_buildings(range(36, 48), 0)


# In[ ]:


plot_buildings(range(48, 60), 0)


# In[ ]:


plot_buildings(range(60, 72), 0)


# In[ ]:


plot_buildings(range(72, 84), 0)


# In[ ]:


plot_buildings(range(84, 96), 0)


# In[ ]:


plot_buildings(range(96, 105), 0)


# ### **3.2 Chilled Water Readings**
# Chilled water readings are less stable than electricity readings in a 3 year timeframe. There are only **24** buildings with chilled water meter in site 0, so it is even harder to find patterns in those buildings. Between some periods, all of the buildings have missing chilled water readings, and a time period without missing chilled water reading is very rare in 3 years. The dates of missing chilled water readings look like totally random except the first couple months of 2016. It was the time when electricity meters were displaying **0** readings.

# In[ ]:


fig = plt.figure(figsize=(18, 10))
                 
(df_site0.query('meter == 1 and meter_reading_scraped.isnull() and ("2016-01-01 00:00:00" <= timestamp < "2017-01-01 00:00:00")', engine='python')['timestamp'].value_counts() - 81).plot(label='Training 2016')
(df_site0.query('meter == 1 and meter_reading_scraped.isnull() and ("2017-01-01 00:00:00" <= timestamp < "2018-01-01 00:00:00")', engine='python')['timestamp'].value_counts() - 81).plot(label='Test 2017')
(df_site0.query('meter == 1 and meter_reading_scraped.isnull() and ("2018-01-01 00:00:00" <= timestamp < "2019-01-01 00:00:00")', engine='python')['timestamp'].value_counts() - 81).plot(label='Test 2018')

plt.title('Site 0 Building Count with Missing Chilled Water Readings in 2016, 2017 and 2018 ')
plt.ylabel('Building Count')
plt.legend()

plt.show()


# * Chilled water readings are recorded on a different scale on UCF's website. They are multiplied with **3.51684** (A ton of refrigeration effect) in the competition data. **A ton of refrigeration** is defined as the amount of refrigeration effect produced by uniform melting of **1 ton** of ice from and at **0º** C in **24** hours.
# * Missing chilled water readings are more common and unpredictable and the periods are longer compared to electricity readings.
# * Positive outliers are also more common in chilled water readings, and unlike electricity readings, there are no negative values.

# In[ ]:


plot_buildings(site0_meter1_buildings[:12], 1)


# In[ ]:


plot_buildings(site0_meter1_buildings[12:], 1)


# ## **4. Scraped Building Metadata**
# 
# There were two new features (`eui` and `leed`) in the scraped building metadata for site 0. Both of those features are dependent to yearly `meter_reading` and `square_feet`.
# * `eui` is not strongly correlated with `square_feet` because there is no causality in their relationship. Apparently, some buildings can be very large and use energy very efficiently at the same time. However `eui` of 2016 can be a predictor of next years.
# * `leed` gives information about `eui`, but most of the buildings don't have certification. Gold certification buildings have higher `eui` than silver certification buildings, but there are other buildings that are using energy more efficiently without any certification. This feature is not reliable because of this reason.

# In[ ]:


df_site0_building_metadata = df_building_metadata_external[df_building_metadata_external['site_id'] == 0]

fig = plt.figure(figsize=(16, 8))

sns.scatterplot(x='square_feet', y='eui', hue='primary_use', data=df_site0_building_metadata)
plt.title('eui vs square_feet')

plt.show()


# In[ ]:


fig = plt.figure(figsize=(16, 8))

sns.boxplot(x='leed', y='eui', data=df_site0_building_metadata.fillna('No Certification'))
plt.title('eui vs leed')

plt.show()

