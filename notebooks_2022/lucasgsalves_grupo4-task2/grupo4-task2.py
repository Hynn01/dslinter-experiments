#!/usr/bin/env python
# coding: utf-8

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


# # Welcome!
# <img src="https://thinkingneuron.com/wp-content/uploads/2020/09/Car-price-prediction-case-study.png" alt="Prediction-car">

# # Group 4:
# ### Students:
# #### Lucas Gabriel Sant'ana Alves.
# #### Pedro Azevedo Martins.
# #### Marcos Vinicius Moraes.
# # Institution:
# ### FATEC Ourinhos - Faculdade de Tecnologia de Ourinhos, São Paulo, Brazil.

# # Import of modules

# In[ ]:


import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style  
import altair as alt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import linear_model


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


train_car = pd.read_csv('/kaggle/input/task-02-car-price-prediction/train_car_details.csv')


# In[ ]:


train_car.head(10)


# # EDA

# In[ ]:


train_car.shape


# In[ ]:


train_car.info()


# In[ ]:


train_car.describe()


# In[ ]:


train_car.columns


# In[ ]:


train_car.isnull().sum()


# In[ ]:


train_car.dropna(inplace=True)


# In[ ]:


print('returns the shape of an array in dataframe train_car: ', train_car.shape)


# In[ ]:


print(train_car['fuel'].value_counts())


# In[ ]:


print(train_car['owner'].value_counts())


# In[ ]:


print(train_car['seller_type'].value_counts())
print(train_car['transmission'].value_counts())


# # Creating the graphics

# In[ ]:


get_ipython().system('pip install dataprep')


# In[ ]:


from dataprep.eda import create_report
create_report(train_car)


# In[ ]:


fuel_type = train_car['fuel']
seller_type = train_car['seller_type']
transmission_type = train_car['transmission']
selling_price = train_car['selling_price']


# In[ ]:


style.use('ggplot')
fig = plt.figure(figsize=(15,5))
fig.suptitle('Visualizing categorical data columns')
plt.subplot(1, 3, 1)
plt.bar(fuel_type, selling_price, color='royalblue')
plt.xlabel("Fuel Type")
plt.ylabel("Selling Price")
plt.subplot(1, 3, 2)
plt.bar(seller_type, selling_price, color='red')
plt.xlabel("Seller Type")
plt.subplot(1, 3, 3)
plt.bar(transmission_type, selling_price, color='purple')
plt.xlabel('Transmission Type')
plt.show()


# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle('Visualizing categorical columns')
sns.barplot(x = fuel_type, y = selling_price, ax = axes[0])
sns.barplot(x = seller_type, y = selling_price, ax = axes[1])
sns.barplot(x = transmission_type, y = selling_price, ax = axes[2])


# In[ ]:


plt.figure(figsize = (10,5))
sns.countplot(x = 'fuel', data = train_car, orient = 'selling_price')
plt.show()


# In[ ]:


plt.figure(figsize = (10, 5))
sns.countplot(x = 'seller_type', data = train_car, orient = 'selling_price')
plt.show()


# In[ ]:


plt.figure(figsize = (10,5))
sns.countplot(x = 'transmission', data = train_car, orient = 'selling_price')
plt.show()


# In[ ]:


petrol_data = train_car.groupby('fuel').get_group('Petrol')
petrol_data.describe()


# In[ ]:


seller_data = train_car.groupby('seller_type').get_group('Dealer')
seller_data.describe()


# In[ ]:


# Manual enconding 
train_car.replace({'fuel':{'Diesel':0, 'Petrol':1, 'CNG':2, 'LPG':3}}, inplace=True)

# One hot enconding
train_car = pd.get_dummies(train_car, columns=['seller_type', 'transmission'], drop_first=True)


# In[ ]:


train_car.head()


# In[ ]:


plt.figure(figsize=(10, 7))
sns.heatmap(train_car.corr(), annot=True)
plt.title('Correlation between the columns')
plt.show()


# In[ ]:


fig = plt.figure(figsize = (7, 5))
plt.title('Correlation between present price and KM Driven')
sns.regplot(x = 'km_driven', y = 'selling_price', data=train_car)
plt.show()


# In[ ]:


sns.relplot(x = 'km_driven', y = 'selling_price', data = train_car)
plt.show()


# In[ ]:


fig = plt.figure(figsize = (7, 5))
plt.title('Correlation between present price and year')
sns.regplot(x = 'year', y = 'selling_price', data = train_car)
plt.show()


# In[ ]:


sns.relplot(x = 'year', y = 'selling_price', data = train_car)
plt.show()


# # Looking for other options

# In[ ]:


df_train = pd.read_csv('/kaggle/input/task-02-car-price-prediction/train_car_details.csv')
df_train.head()


# In[ ]:


# Creating Dataframes

inputs = df_train.drop(['name', 'owner', 'seller_type'], axis='columns')
target = df_train.selling_price
target


# In[ ]:


# Enconding
from sklearn.preprocessing import LabelEncoder
Numerics = LabelEncoder()


# In[ ]:


# New Encoded Columns
inputs['Fuel_Type_n'] = Numerics.fit_transform(inputs['fuel'])
inputs['Transmission_n'] = Numerics.fit_transform(inputs['transmission'])
inputs


# In[ ]:


# Droping string columns

inputs_n = inputs.drop(['fuel', 'transmission', 'selling_price', 'mileage', 'engine', 'max_power', 'torque', 'seats', 'Id'], axis='columns')
inputs_n


# In[ ]:


# Linear Regression
model = linear_model.LinearRegression()


# In[ ]:


# Training 
model.fit(inputs_n, target)


# In[ ]:


# Prediction
# The sample values which is from the dataset
pred = model.predict([[1999, 110000, 3, 1]])
print(pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




