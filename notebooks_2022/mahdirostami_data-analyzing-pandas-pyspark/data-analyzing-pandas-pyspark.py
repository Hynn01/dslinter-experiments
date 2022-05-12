#!/usr/bin/env python
# coding: utf-8

# **Apartment rental offers in Germany**
# > Rental offers scraped from Germany biggest real estate online plattform 

# # STEPS
# 1. [Import libraries](#Import_libraries)
# 2. [Load data and look at data](#Load_data)
#     1. [Null items](#Null_items)
#     2. [Duplicate](#Duplicate)
#     3. [Outliers](#Outliers)
# 3. [Features](#Features)

# <a id="Import_libraries"></a>
# # Import libraries

# In[ ]:


get_ipython().system('pip install pyspark')


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import PandasUDFType
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructField, StructType, IntegerType, StringType, BooleanType, FloatType


# In[ ]:


# Create SparkSession
spark = SparkSession.builder.appName('Apartment rental offers in Germany').getOrCreate()


# In[ ]:


spark.version


# <a id="Load_data"></a>
# # Load data and look at data

# In[ ]:


pd.set_option('display.max_columns', None)


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


s = time.time()
# load the dataset and create sprk dataframe
df_s = spark.read.csv('/kaggle/input/apartment-rental-offers-in-germany/immo_data.csv',inferSchema=True,header=True)
# Using limit(), or select() or show() to view the data. I often use limit()
# Using toPandas() method to return Pyspark DataFrame as Pandas table
print(f'pyspark {time.time()-s} second')
df_s.limit(3).toPandas()


# In[ ]:


s = time.time()
df_p = pd.read_csv("/kaggle/input/apartment-rental-offers-in-germany/immo_data.csv")
print(f'pandas {time.time()-s} second')
df_p.head(3)


# In[ ]:


print(df_s.count(), len(df_s.columns))


# In[ ]:


df_p.shape


# In[ ]:


df_s.printSchema()


# In[ ]:


df_p.info()


# <a id="Null_items"></a>
# # Null items

# In[ ]:


for col in df_p.columns:
    if df_p[col].isnull().sum()/len(df_p[col])*100 > 0 :
        print(col," =>", df_p[col].isnull().sum()/len(df_p[col])*100)


# In[ ]:


null = ("telekomHybridUploadSpeed", "noParkSpaces", "interiorQual", "petsAllowed", "heatingCosts", "energyEfficiencyClass", "lastRefurbish", "electricityBasePrice", "electricityKwhPrice")


# In[ ]:


s = time.time()
df_s.drop(*null)
print(f'pyspark {time.time()-s} second')


# In[ ]:


s = time.time()
df_p.drop(columns=list(null), axis=1, inplace=True)
print(f'pandas {time.time()-s} second')


# In[ ]:


def fillnapy():
    df_s.na.fill(df_p.serviceCharge.median(), "serviceCharge")
    df_s.na.fill(df_p.heatingType.mode()[0], "heatingType")
    df_s.na.fill("NONE", "telekomTvOffer")
    df_s.na.fill(df_p.pricetrend.median(), "pricetrend")
    df_s.na.fill(df_p.telekomUploadSpeed.median(), "telekomUploadSpeed")
    df_s.na.fill(df_p.totalRent.median(), "totalRent")
    df_s.na.fill(df_p.yearConstructed.median(), "yearConstructed")
    df_s.na.fill(df_p.firingTypes.mode()[0], "firingTypes")
    df_s.na.fill(df_p.yearConstructedRange.median(), "yearConstructedRange")
    df_s.na.fill(0, "houseNumber")
    df_s.na.fill(df_p.condition.mode()[0], "condition")
    df_s.na.fill(df_p.streetPlain.mode()[0], "streetPlain")
    df_s.na.fill(df_p.typeOfFlat.mode()[0], "typeOfFlat")
    df_s.na.fill(df_p.thermalChar.median(), "thermalChar")
    df_s.na.fill(df_p.floor.median(), "floor")
    df_s.na.fill(df_p.numberOfFloors.median(), "numberOfFloors")
    df_s.na.fill('nothing', "description")
    df_s.na.fill('nothing', "facilities")
    return time.time()


# In[ ]:


s = time.time()
time2 = fillnapy()
print(f'pyspark {time2 - s} second')


# In[ ]:


def fillna():
    df_p.serviceCharge.fillna(df_p.serviceCharge.median(), inplace=True)
    df_p.heatingType.fillna(df_p.heatingType.mode()[0], inplace=True)
    df_p.telekomTvOffer.fillna("NONE", inplace=True)
    df_p.pricetrend.fillna(df_p.pricetrend.median(), inplace=True)
    df_p.telekomUploadSpeed.fillna(df_p.telekomUploadSpeed.median(), inplace=True)
    df_p.totalRent.fillna(df_p.totalRent.median(), inplace=True)
    df_p.yearConstructed.fillna(df_p.yearConstructed.median(), inplace=True)
    df_p.firingTypes.fillna(df_p.firingTypes.mode()[0], inplace=True)
    df_p.yearConstructedRange.fillna(df_p.yearConstructedRange.median(), inplace=True)
    df_p.houseNumber.fillna(0, inplace=True)
    df_p.condition.fillna(df_p.condition.mode()[0], inplace=True)
    df_p.streetPlain.fillna(df_p.streetPlain.mode()[0], inplace=True)
    df_p.typeOfFlat.fillna(df_p.typeOfFlat.mode()[0], inplace=True)
    df_p.thermalChar.fillna(df_p.thermalChar.median(), inplace=True)
    df_p.floor.fillna(df_p.floor.median(), inplace=True)
    df_p.numberOfFloors.fillna(df_p.numberOfFloors.median(), inplace=True)
    df_p.description.fillna('nothing', inplace=True)
    df_p.facilities.fillna('nothing', inplace=True)
    return time.time()


# In[ ]:


s = time.time()
time2 = fillna()
print(f'pandas {time2 - s} second')


# <a id="Duplicate"></a>
# # Duplicate

# In[ ]:


df_p.duplicated().sum()


# <a id="Outliers"></a>
# # Outliers 

# In[ ]:


# Return unbiased kurtosis over requested axis using Fisher's definition of kurtosis (kurtosis of normal == 0.0).
# The result is normalized by N-1
kurt = df_p.kurt(numeric_only=True)[:]
kurt


# In[ ]:


outliers = ["serviceCharge", "totalRent", "yearConstructed", "baseRent", "livingSpace", "noRooms", "thermalChar", "floor", "numberOfFloors"]


# In[ ]:


df_p.info()


# In[ ]:


for col in outliers:
    plt.figure(figsize=(4, 4))
    df_p.boxplot(column=[col])


# In[ ]:


def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))


# In[ ]:


for col in outliers:
    print(f'{col}: {len(outliers_iqr(df_p[col])[0])}')


# In[ ]:


for i in outliers:
    quartile_1, quartile_3 = np.percentile(df_p[i], [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    df_p = df_p[df_p[i]<upper_bound]
    df_p = df_p[df_p[i]>lower_bound]


# In[ ]:


def outliers_z_score(ys):
    threshold = 3

    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return np.where(np.abs(z_scores) > threshold)


# In[ ]:


for col in outliers:
    print(f'{col}: {len(outliers_z_score(df_p[col])[0])}')


# In[ ]:


for i in ["serviceCharge", "totalRent", "baseRent"]:
    mean_y = np.mean(df_p[i])
    stdev_y = np.std(df_p[i])
    lower_bound = mean_y - (3 * stdev_y)
    upper_bound = mean_y + (3 * stdev_y)
    df_p = df_p[df_p[i]<upper_bound]
    df_p = df_p[df_p[i]>lower_bound]


# In[ ]:


df_p = df_p[df_p.yearConstructed < 2020]


# <a id="Features"></a>
# # Features

# In[ ]:


df = df_p.copy()


# In[ ]:


# Split numeric and non numeric columns
cat_df = df.select_dtypes(include=['object'])
num_df = df.select_dtypes(exclude=['object'])
print("non numeric:", cat_df.columns)
print("numeric:", num_df.columns)


# In[ ]:


df.head(1)


# ## regio1

# In[ ]:


df.regio1.value_counts()


# In[ ]:


plt.figure(figsize=(10, 10))
data = df.regio1.value_counts()[:10]
x = list(data.index)
y = list(data.values)
x.reverse()
y.reverse()

plt.title("regio1")
plt.ylabel("regio1")
plt.xlabel("Number")

plt.barh(x, y)
plt.show()


# ## regio2

# In[ ]:


df.regio2.value_counts()


# In[ ]:


plt.figure(figsize=(10, 10))
data = df.regio2.value_counts()[:10]
x = list(data.index)
y = list(data.values)
x.reverse()
y.reverse()

plt.title("regio2")
plt.ylabel("regio2")
plt.xlabel("Number")

plt.barh(x, y)
plt.show()


# ## regio3

# In[ ]:


df.regio3.value_counts()


# In[ ]:


plt.figure(figsize=(10, 10))
data = df.regio3.value_counts()[:10]
x = list(data.index)
y = list(data.values)
x.reverse()
y.reverse()

plt.title("regio3")
plt.ylabel("regio3")
plt.xlabel("Number")

plt.barh(x, y)
plt.show()


# ## street

# In[ ]:


df.street.value_counts()


# In[ ]:


plt.figure(figsize=(10, 10))
data = df.street.value_counts()[:10]
x = list(data.index)
y = list(data.values)
x.reverse()
y.reverse()

plt.title("street")
plt.ylabel("street")
plt.xlabel("Number")

plt.barh(x, y)
plt.show()


# ## geo_krs

# In[ ]:


df.geo_krs.value_counts()


# In[ ]:


plt.figure(figsize=(10, 10))
data = df.geo_krs.value_counts()[:10]
x = list(data.index)
y = list(data.values)
x.reverse()
y.reverse()

plt.title("geo_krs")
plt.ylabel("geo_krs")
plt.xlabel("Number")

plt.barh(x, y)
plt.show()


# ## geo_bln

# In[ ]:


df.geo_bln.value_counts()


# In[ ]:


plt.figure(figsize=(10, 10))
data = df.geo_bln.value_counts()[:10]
x = list(data.index)
y = list(data.values)
x.reverse()
y.reverse()

plt.title("geo_bln")
plt.ylabel("geo_bln")
plt.xlabel("Number")

plt.barh(x, y)
plt.show()


# ## typeOfFlat

# In[ ]:


df.typeOfFlat.value_counts()


# In[ ]:


plt.figure(figsize=(10, 10))
data = df.typeOfFlat.value_counts()[:10]
x = list(data.index)
y = list(data.values)
x.reverse()
y.reverse()

plt.title("typeOfFlat")
plt.ylabel("typeOfFlat")
plt.xlabel("Number")

plt.barh(x, y)
plt.show()


# In[ ]:


def filter(x):
    if x in [
    'maisonette', 'raised_ground_floor', 'half_basement', 'terraced_flat', 
    'penthouse', 'loft',
] :
        x = 'other'
        return x
    else:
        return x
df.typeOfFlat = df.typeOfFlat.apply(filter)


# In[ ]:


plt.figure(figsize=(10, 10))
data = df.typeOfFlat.value_counts()[:10]
x = list(data.index)
y = list(data.values)
x.reverse()
y.reverse()

plt.title("typeOfFlat")
plt.ylabel("typeOfFlat")
plt.xlabel("Number")

plt.barh(x, y)
plt.show()


# ## streetPlain

# In[ ]:


df.streetPlain.value_counts()[:10]


# In[ ]:


plt.figure(figsize=(10, 10))
data = df.streetPlain.value_counts()[:10]
x = list(data.index)
y = list(data.values)
x.reverse()
y.reverse()

plt.title("streetPlain")
plt.ylabel("streetPlain")
plt.xlabel("Number")

plt.barh(x, y)
plt.show()


# ## condition

# In[ ]:


df.condition.value_counts()


# In[ ]:


plt.figure(figsize=(10, 10))
data = df.condition.value_counts()[:10]
x = list(data.index)
y = list(data.values)
x.reverse()
y.reverse()

plt.title("condition")
plt.ylabel("condition")
plt.xlabel("Number")

plt.barh(x, y)
plt.show()


# In[ ]:


def filter(x):
    if x in [
    "negotiable"                            ,
"need_of_renovation"                  ,
"ripe_for_demolition"                     ,
] :
        x = 'other'
        return x
    else:
        return x
df.condition = df.condition.apply(filter)


# In[ ]:


plt.figure(figsize=(10, 10))
data = df.condition.value_counts()[:10]
x = list(data.index)
y = list(data.values)
x.reverse()
y.reverse()

plt.title("condition")
plt.ylabel("condition")
plt.xlabel("Number")

plt.barh(x, y)
plt.show()


# ## firingTypes

# In[ ]:


df.firingTypes.value_counts()[:10]


# In[ ]:


plt.figure(figsize=(10, 10))
data = df.firingTypes.value_counts()[:10]
x = list(data.index)
y = list(data.values)
x.reverse()
y.reverse()

plt.title("firingTypes")
plt.ylabel("firingTypes")
plt.xlabel("Number")

plt.barh(x, y)
plt.show()


# ## heatingType

# In[ ]:


df.heatingType.value_counts()


# In[ ]:


plt.figure(figsize=(10, 10))
data = df.heatingType.value_counts()[:10]
x = list(data.index)
y = list(data.values)
x.reverse()
y.reverse()

plt.title("heatingType")
plt.ylabel("heatingType")
plt.xlabel("Number")

plt.barh(x, y)
plt.show()


# In[ ]:


def filter(x):
    if x in [
    'oil_heating', 'combined_heat_and_power_plant', 'heat_pump', 'night_storage_heater', 
    'wood_pellet_heating', 'electric_heating', 'stove_heating', 'solar_heating'
] :
        x = 'other'
        return x
    else:
        return x
df.heatingType = df.heatingType.apply(filter)


# In[ ]:


plt.figure(figsize=(10, 10))
data = df.heatingType.value_counts()[:10]
x = list(data.index)
y = list(data.values)
x.reverse()
y.reverse()

plt.title("heatingType")
plt.ylabel("heatingType")
plt.xlabel("Number")

plt.barh(x, y)
plt.show()


# ## telekomTvOffer

# In[ ]:


df.telekomTvOffer.value_counts()


# In[ ]:


plt.figure(figsize=(10, 10))
data = df.telekomTvOffer.value_counts()[:10]
x = list(data.index)
y = list(data.values)
x.reverse()
y.reverse()

plt.title("telekomTvOffer")
plt.ylabel("telekomTvOffer")
plt.xlabel("Number")

plt.barh(x, y)
plt.show()


# In[ ]:


# Visualizing the distribution for every "feature"
df.hist(edgecolor="black", linewidth=1.2, figsize=(20, 20))
plt.show()


# In[ ]:


# correlation matrix
plt.figure(figsize=(20, 10))
sns.heatmap(df.corr(), annot=True)
plt.show()

