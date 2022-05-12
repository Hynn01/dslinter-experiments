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


# In[ ]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().system(' pip install openpyxl')


# In[ ]:


df = pd.read_excel('../input/furniture-sales-and-customer/Office Sales.xlsx')


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.dtypes


# # OrderDate & Delivery Date: determining delivery time

# In[ ]:


plt.figure(figsize = (15,8))
sns.lineplot(df['OrderDate'],df['OrderQuantity'])


# In[ ]:


plt.figure(figsize = (15,8))
sns.lineplot(df['DeliveryDate'],df['OrderQuantity'])


# # Same Day Delivery?

# In[ ]:


plt.figure(figsize=(10, 10))
df['diffInDeliv'] = pd.to_datetime(df['DeliveryDate']) - pd.to_datetime(df['OrderDate'])
df['diffInDeliv'] = df['diffInDeliv'].dt.days
plt.hist(df['diffInDeliv'])


# # ShipMode

# In[ ]:


plt.figure(figsize=(10,10))
sns.barplot(x = df['ShipMode'].unique(), y= df['ShipMode'].value_counts())


# # CategoryName

# In[ ]:


plt.figure(figsize=(10,10))
sns.barplot(x = df['CategoryName'].unique(), y= df['CategoryName'].value_counts())


# # SubCategory

# In[ ]:


fig, axes = plt.subplots(1, 2, figsize = (20,10))
fig.suptitle('Subcategory')

axes[0].set_title('Furniture')
subcat = df[df['CategoryName'] == 'Furniture']['SubcategoryName']
sns.barplot(ax=axes[0],x = subcat.unique(), y = subcat.value_counts())

axes[1].set_title("Office Supplies")
subcat = df[df['CategoryName'] == 'Office Supplies']['SubcategoryName']
sns.barplot(ax = axes[1],x = subcat.unique(), y = subcat.value_counts())


# # ProductName

# In[ ]:


products  = []
df['Product'] = df['ProductName'].map(lambda x: x.split(',')[0])
df['Pro_details'] = df['ProductName'].map(lambda x: x.split(',')[1] if len(x.split(','))>1 else None)


# In[ ]:


df['Product'] = df.Product.str.replace(r' +', ' ')


# In[ ]:


len(df.Product.unique())


# # Total Price = UnitPrice * Quantity % discount + shipping

# In[ ]:


df['total_price'] = ((df['UnitPrice'] * df['OrderQuantity']) * (100-df['Discount %']))/100 + df['ShippingCost']


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize = (20,10))
fig.suptitle('Subcategory with Price')

axes[0].set_title('Furniture')
subcat = df[df['CategoryName'] == 'Furniture']['SubcategoryName']
unit_sum = df[df['CategoryName'] == 'Furniture'].groupby('SubcategoryName').total_price.sum()
sns.barplot(ax=axes[0],x = subcat.unique(), y = unit_sum)

axes[1].set_title("Office Supplies")
subcat = df[df['CategoryName'] == 'Office Supplies']['SubcategoryName']
unit_sum = df[df['CategoryName'] == 'Office Supplies'].groupby('SubcategoryName').total_price.sum()
sns.barplot(ax = axes[1],x = subcat.unique(), y = unit_sum)


# # How does Order Priority determined?

# In[ ]:


plt.figure(figsize=(10,10))
sns.barplot(x = df['OrderPriority'].unique(), y= df['OrderPriority'].value_counts())


# In[ ]:


plt.figure(figsize=(10,10))
sns.barplot(x = df['OrderPriority'].unique(), y= df.groupby('OrderPriority').total_price.sum())


# In[ ]:




