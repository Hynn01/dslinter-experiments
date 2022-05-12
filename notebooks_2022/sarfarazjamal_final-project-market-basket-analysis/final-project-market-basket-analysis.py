#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
from statsmodels.tsa.seasonal import seasonal_decompose
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv("../input/market-basket-analysis/Assignment-1_Data.csv", sep = ";")


# # PREPROCESSING

# In[ ]:


data.head(10)


# In[ ]:


data.isnull().sum()
# null customer id does not matter
# Item name has to be removed


# In[ ]:


plt.figure(figsize=(15,6))
sns.heatmap(data.isna().transpose())


# In[ ]:


data.shape


# In[ ]:


data = data.dropna(subset=["Itemname"])


# In[ ]:


data.isnull().sum()


# In[ ]:


plt.figure(figsize=(15,6))
sns.heatmap(data.isna().transpose())


# In[ ]:


data["Itemname"].value_counts()
# low quantity products were removed first.But found out that it does not affect comptational time
# signnificantly


# In[ ]:


data.shape


# In[ ]:


data.dtypes


# In[ ]:


data[data['CustomerID'].isnull()]


# In[ ]:


data = data.fillna(0)


# In[ ]:


data[data["CustomerID"].isnull()]


# In[ ]:


data.isnull().sum()


# In[ ]:


data.shape


# In[ ]:


data["Date"] = pd.to_datetime(data["Date"])


# In[ ]:


data["Price"] = data["Price"].str.replace(",",".")
data["Price"] = pd.to_numeric(data["Price"])


# In[ ]:


data["Price"]


# In[ ]:


data.dtypes


# In[ ]:


data["Sales"] = data["Quantity"] * data["Price"]


# In[ ]:


data.head()


# # EDA

# In[ ]:


top20items = pd.DataFrame(data["Itemname"].value_counts().head(20))
top20items = top20items.reset_index()
top20items.columns = ["Itemname","Frequency"]
labels = top20items["Itemname"]
sizes = top20items["Frequency"]
top20items


# In[ ]:


fig = plt.figure(figsize=(16,6))
colors = sns.color_palette("Spectral",20)
squarify.plot(sizes, label=labels, color =  colors)
plt.title("Top 20 Products")


# In[ ]:


t_light = data[data["Itemname"]== "WHITE HANGING HEART T-LIGHT HOLDER"]
t_light


# In[ ]:


fig = plt.figure(figsize=(24,8))
sns.lineplot(x = t_light["Date"], y = t_light["Quantity"] )
# most of them have been sold in 2011. 


# In[ ]:


startdate = t_light["Date"] >= "20110101"
t_light_2011 = t_light.loc[startdate]
t_light_2011.head()


# In[ ]:


plt.figure(figsize=(16,6))
sns.lineplot(t_light_2011["Date"],t_light_2011["Sales"])


# In[ ]:


country_specific = data.groupby(["Country"]).sum().reset_index()
country_specific = country_specific.sort_values(["Sales"], ascending = False)
country_specific_10 = country_specific.head(10)
country_specific_10


# In[ ]:


plt.figure(figsize=(15,6))
p = sns.barplot(country_specific_10["Country"], country_specific_10["Sales"])
p.set_ylabel("Sales (millions)")


# In[ ]:


basket = data.groupby(["BillNo","Itemname"])["Quantity"].sum().unstack().reset_index().fillna(0).set_index("BillNo")
basket


# In[ ]:


def one_hot_encoding(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
basket = basket.applymap(one_hot_encoding)


# In[ ]:


basket


# # Modelling

# In[ ]:


#frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)


# In[ ]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)


# In[ ]:


rules = rules.sort_values("lift", ascending = False)


# In[ ]:


rules

