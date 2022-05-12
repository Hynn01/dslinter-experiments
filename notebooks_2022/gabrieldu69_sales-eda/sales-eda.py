#!/usr/bin/env python
# coding: utf-8

# # Kaggle competition predict sales
# https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data

# # Table of Contents
# 
# * [Introduction](#intro)
# * [Cleaning](#clean)
#     * [missing values](#missing)
#     * [outliers](#outliers)
# * [Data visualization](#eda)
#     * [Feature repartition](#repartition)
#     * [top category](#top)
# * [Time series ](#time)
#     * [decomposition](#decomp)
# * [Sales prediction model](#predict)
#     * [Holt Winters](#exp)
#     

# # Introduction <a class="anchor" id="intro"></a>

# Given the following data :
# <ul> 
#     <li>sales_train.csv</li>
#     <li>test.csv</li>  
#     <li>sample_submission.csv</li>
#     <li>items.csv</li>
#     <li>item_categories.csv</li>
#     <li>shops.csv</li>
# </ul>
# <b>The task is to forecast the total amount of products sold in every shop for the test set. Note that the list of shops and products slightly changes every month. Creating a robust model that can handle such situations is part of the challenge.</b>

# <b>Data fields</b>
# <ul>
#     <li><b>ID</b> - an Id that represents a (Shop, Item) tuple within the test set</li>
#     <li><b>shop_id</b> - unique identifier of a shop</li>
#     <li><b>item_id</b> - unique identifier of a product</li>
#     <li><b>item_category_id</b> - unique identifier of item category</li>
#     <li><b>item_cnt_day</b> - number of products sold. You are predicting a monthly amount of this measure</li>
#     <li><b>item_price</b> - current price of an item</li>
#     <li><b>date</b> - date in format dd/mm/yyyy</li>
#     <li><b>date_block_num</b> - a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33</li>
#     <li><b>item_name</b> - name of item</li>
#     <li><b>shop_name</b> - name of shop</li>
#     <li><b>item_category_name</b> - name of item category</li>
# </ul>

# # Importing and cleaning <a class="anchor" id="clean"></a>

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil


# In[ ]:


sales_train=pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv',index_col='date',parse_dates=True)
test=pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
sample_submission=pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')
items=pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
items_category=pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
shops=pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')


# **shapes info**

# In[ ]:


print("shape info of sales_train: {}".format(sales_train.shape))
print("shape info of test: {}".format(test.shape))
print("shape info of sample submission {}".format(sample_submission.shape))
print("shape info of items {}".format(items.shape))
print("shape info of items category {}".format(items_category.shape))
print("shape info of shops {}".format(shops.shape))


# In[ ]:


print(sales_train.head())
print()
print(test.head())
print()
print(items.head())
print()
print(items_category.head())
print()
print(shops.head())
print()


# **merging different table using shop_id and items_id keys**

# In[ ]:


items_merged=pd.merge(items,items_category,on='item_category_id')
sales_train_merged=pd.merge(sales_train,shops,on='shop_id')
sales_train_merged=pd.merge(sales_train_merged,items_merged,on='item_id')


# In[ ]:


items_merged.head(3)


# In[ ]:


sales_train_merged.head(3)


# ## Missing values <a class='anchor' id='missing'></a>

# In[ ]:


print(sales_train.isnull().sum())
print()
print(test.isnull().sum())
print()
print(items.isnull().sum())
print()
print(items_category.isnull().sum())


# **no na or null data**

# ## Outliers detection <a class='anchor' id='outliers'></a>

# In[ ]:


sales_train.dtypes


# In[ ]:


from sklearn.ensemble import IsolationForest


def detect_outliers(dataframe,contamination):
    

    a=list(dataframe.select_dtypes(['int64']).columns)+list(dataframe.select_dtypes(['float64']).columns)
    model=IsolationForest(contamination=contamination)
    model.fit(dataframe[a])
    outliers = model.predict(dataframe[a]) ==-1
    return outliers


# In[ ]:


contamination=0.005
index=detect_outliers(sales_train[['item_price','item_cnt_day']],contamination)
lign,col=sales_train[index].shape
print("number of outliers: {}".format(lign))


# **filtering ->**

# In[ ]:


sales_train[index].head()


# In[ ]:


sales_train=sales_train[index==False]


# In[ ]:


sales_train[['item_price','item_cnt_day']].describe()


# **we can see that we have cleaned data from kaggle and we don't need to clean it much more**

# ## Data visualization <a class='anchor' id='eda'></a>

# we're going to explore the data to guess the nature of the data and make some links between features

# ## Feature repartition <a class='anchor' id='repartition'></a>

# In[ ]:



figure2=plt.figure(2,figsize=(15,10))

plt.subplot(3,1,1)
plt.title("item price",size=15)
sns.boxplot(data=sales_train,x="item_price")
plt.subplot(3,1,2)
plt.title("item count day",size=15)
sns.boxplot(data=sales_train,x="item_cnt_day")

figure2.tight_layout(pad=3.0)

plt.show()


# In[ ]:


print("item price median {:.2f} item price mean : {:.2f}".format(sales_train['item_price'].median(),sales_train['item_price'].mean()))
print("item count day median: {:.2f} item count day mean {:.2f} ".format(sales_train['item_cnt_day'].median(),sales_train['item_cnt_day'].mean()))


# In[ ]:


print(len(sales_train_merged['item_category_name'].unique()))


# ## Top shops and category in sold items <a class='anchor' id='top'></a>

# In[ ]:


top10s=plt.figure(figsize=(10,7))
plt.title('top category')
plt.ylabel('Sales')
sales_train_merged.groupby('item_category_name')['item_cnt_day'].sum().sort_values(ascending=False)[0:10].plot(kind='bar',color='Orange',ls='dashed',edgecolor='Black')
plt.show()


# **top shops**

# In[ ]:


top10s=plt.figure(figsize=(10,7))
plt.title('top shops')
plt.ylabel('Sales')
sales_train_merged.groupby('shop_name')['item_cnt_day'].sum().sort_values(ascending=False)[0:10].plot(kind='bar',color='Red',ls='dashed',edgecolor='Black')
plt.show()


# ## Time series analysis <a class="anchor" id="time"></a>

# **sum of item counts per day**

# In[ ]:


plt.figure(figsize=(15,7))

sales_train['item_cnt_day'].resample('M').agg(['sum']).plot(color='blue',ls='-')
plt.title('total item sold per month')
plt.xlabel('month',size=10)
plt.ylabel('item  count',size=10)
plt.show()




# In[ ]:


plt.figure(figsize=(15,7))
sales_train['item_cnt_day'].resample('M').agg(['sum','min','max','std']).plot(subplots=True,figsize=(15,15))
plt.xlabel('month',size=10)
plt.ylabel('item  count',size=10)

plt.show()


# **max - min**

# In[ ]:


min_max=sales_train['item_cnt_day'].resample('M').agg(['std','min','max'])
min_max['max_min']=min_max['max']-min_max['min']

minfig=plt.figure(figsize=(15,7))
plt.title("maximum - minimum sales over time")
min_max['max_min'].plot()
plt.show()


# ### Seasonal decomposition <a class='anchor' id='decomp'></a>

# In[ ]:


import statsmodels.api as sm

decomposition=sm.tsa.seasonal_decompose(sales_train['item_cnt_day'].resample('M').agg(['sum']),model='additive')

figure=plt.figure(figsize=(20,7))
decomposition.plot()
plt.show()


# **Trend**

# In[ ]:


figtrend=plt.figure(figsize=(15,7))
plt.title("item counts trend")
plt.plot(sales_train['item_cnt_day'].resample('M').sum(),c='blue')
plt.plot(decomposition.trend.index, decomposition.trend, c='red')
plt.legend(["sum of item counts","trend of item count"])
plt.xlabel('time')
plt.ylabel('count')
plt.show()


# In[ ]:


figseason=plt.figure(figsize=(15,7))
plt.title("item counts seasonality")
plt.plot(sales_train['item_cnt_day'].resample('M').sum(),c='blue')
plt.plot(decomposition.seasonal.index, decomposition.seasonal, c='red')
plt.legend(["sum of item counts","seasonal component of item count"])
plt.xlabel('time')
plt.ylabel('count')
plt.show()


# ## Prediction <a class="anchor" id="predict"></a>

# ## Holt winters algorithm for predictions <a class="anchor" id="exp"></a>

# In[ ]:


from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[ ]:


X=sales_train['item_cnt_day'].resample('M').sum()

train=X['2013':'2015']
test=X['2015':]

model = ExponentialSmoothing(train, trend='add',seasonal='add').fit()
pred = model.predict(start=test.index[0], end=test.index[-1])

predfig=plt.figure(figsize=(15,7))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(pred.index, pred, label='Holt-Winters')
plt.legend(loc='best')
plt.show()

