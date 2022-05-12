#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
#         df = pd.read_csv(os.path.join(dirname, filename))
#         print(df.columns)
#         print(df.head(5))


def reduce_mem_usage(train_data):
    start_mem = train_data.memory_usage().sum() / 1024**2
#     print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in train_data.columns:
        col_type = train_data[col].dtype
        if col_type != object:
            c_min = train_data[col].min()
            c_max = train_data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    train_data[col] = train_data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    train_data[col] = train_data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    train_data[col] = train_data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    train_data[col] = train_data[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    train_data[col] = train_data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    train_data[col] = train_data[col].astype(np.float32)
                else:
                    train_data[col] = train_data[col].astype(np.float64)
        else:
            train_data[col] = train_data[col].astype('category')
    
    end_mem = train_data.memory_usage().sum() / 1024**2
#     print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
#     print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return train_data


df_train = reduce_mem_usage(pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/train.csv'))
# print(df_train.info(verbose=False, memory_usage="deep"))

# df_test = reduce_mem_usage(pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/test.csv'))
# df_submission = reduce_mem_usage(pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/sample_submission.csv'))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# **Dataset**
# 

# In[ ]:


# Merge the data with other tables
df_stores = reduce_mem_usage(pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/stores.csv'))
df_trainM = df_train.merge(df_stores, on='store_nbr', how='left')

df_trans = reduce_mem_usage(pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/transactions.csv'))
df_trainM = df_trainM.merge(df_trans, on=['date', 'store_nbr'], how='left')
# df_trainM['date'] = pd.to_datetime(df_trainM['date'])
df_oil = reduce_mem_usage(pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/oil.csv'))
df_trainM = df_trainM.merge(df_oil, on='date', how='left')
df_trainM = df_trainM.rename(columns = {"type" : "store_type"})

df_holiEve = reduce_mem_usage(pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/holidays_events.csv'))
# df_holiEve['date'] = pd.to_datetime(df_holiEve['date'])
df_holiEve.drop(df_holiEve[pd.to_datetime(df_holiEve['date']) < '2013-01-01'].index, inplace=True)
# df_holiEve.drop(df_holiEve[df_holiEve['date'] < '2013-01-01'].index, inplace=True)
df_holiEve.drop(df_holiEve[df_holiEve['type'] == 'Transfer'].index, inplace=True)

df_holiEve.drop_duplicates(subset=['date'], inplace=True, ignore_index=True)
# df_holiEve['date'] = df_holiEve['date'].astype('category')
df_holiEve['type'] = pd.Categorical(df_holiEve['type']).remove_categories('Transfer')

df_trainM = df_trainM.merge(df_holiEve, on='date', how='left')

df_trainM['date'] = pd.to_datetime(df_trainM['date'])
df_trainM['year'] = df_trainM['date'].dt.year
df_trainM['month'] = df_trainM['date'].dt.month
df_trainM['monthname'] = df_trainM['date'].dt.month_name()
df_trainM['week'] = df_trainM['date'].dt.isocalendar().week
df_trainM['quarter'] = df_trainM['date'].dt.quarter
df_trainM['day_of_week'] = df_trainM['date'].dt.day_name()
df_trainM = df_trainM.rename(columns = {"type" : "holiday_type"})

df_trainM.drop(columns = ['locale', 'locale_name', 'description', 'transferred'], axis = 1, inplace=True)
df_trainM['holiday_type'].fillna('Work Day', inplace=True)


# **Sales Analysis**
# 
# Customer (C)> Hey, What do you sell here?
# 
# Sales Manger (SM)> What do you want?
# 
# C> Don't know.
# 
# SM> Come, I will walk you through it.
# We have 33 products, 54 stores across 16 states, and we are into this business for the past 56 months.
# Here are the top 10 products we sell the most.

# In[ ]:



# Intialize the matplotlib figure
f, ax = plt.subplots(figsize=(20,20))
sns.set_context("notebook", font_scale=2)

# Load the dataset
da_fs = df_trainM.groupby('family').agg(mxSum = ('sales',sum), 
                                         meSales = ('sales',np.mean)).reset_index().sort_values(by='mxSum',
                                                                                                ascending=False)
da_fs['pSales'] = (da_fs['mxSum']/sum(df_trainM['sales'])*100)

remlFam = da_fs['family'].tail(23).tolist()

da_fs['family'] = pd.Categorical(da_fs['family']).remove_categories(da_fs['family'].tail(23).tolist())
# print(da_fs.tail(23).index.tolist())
da_fs = da_fs.drop(da_fs.tail(23).index.tolist())

# Plot the sales
ax = sns.barplot(x='pSales', y='family', data=da_fs, label='Family',
             order=da_fs['family'], palette="Blues_r")

# label each bar in barplot
for p in ax.patches:
 height = p.get_height() # height of each horizontal bar is the same
 width = p.get_width() # width (average number of passengers)
 # adding text to each bar
 ax.text(x = width+0.3, # x-coordinate position of data label, padded 3 to right of bar
 y = p.get_y()+(height/2), # # y-coordinate position of data label, padded to be in the middle of the bar
 s = '{:.0f}%'.format(width), # data label, formatted to ignore decimals
 va = 'center', # sets vertical alignment (va) to center
 fontsize=16)
    
# for name, sales in zip(da_fs['family'], da_fs['pSales']):
#     ax.text(name, sales, round(sales, 2), color='white', ha='center')
# Add a legend and informative axis label
# ax.legend(ncol=2, loc='lower right', frameon=True)
ax.set(xlim=(0,max(da_fs['pSales'])+1), ylabel="Goods Family",
       xlabel="% Amount spent on products")

ax.set_title("Overall Sales on 53 Stores, 10/33 Products, 16 States and 56 Months", fontdict={'fontsize':24})
sns.despine(left=True, bottom=True)


# C> I think your business might be affected due to the change in oil prices?
# 
# SM > In fact, our sales have increased each year, irrespective of oil price fluctuations.
# 
# This our growth from past 5 years still sales are high albeit we are in September month of 2017
# 

# In[ ]:


df_ys = df_trainM.groupby(['year', 'family']).agg(mxSum = ('sales',sum), 
                                         meSales = ('sales',np.mean)).reset_index().sort_values(by=['year', 'mxSum'],
                                                                                                ascending=False)


df_oilP = df_trainM.groupby(['year']).agg(mxPrice = ('dcoilwtico',np.mean)).reset_index().sort_values(by=['year'],ascending=False)

# print(da_fs['family'].tolist())
# print(df_ys)
# for x in da_fs['family'].tolist():
#     df_ys = df_ys.filter(like=x, axis=0)
df_ys = df_ys[(df_ys['family'] == 'GROCERY I') | (df_ys['family'] == 'BEVERAGES') | 
        (df_ys['family'] == 'PRODUCE') | (df_ys['family'] == 'CLEANING') |
        (df_ys['family'] == 'DAIRY') | (df_ys['family'] == 'BREAD/BAKERY') |
        (df_ys['family'] == 'POULTRY') | (df_ys['family'] == 'MEATS') |
        (df_ys['family'] == 'PERSONAL CARE') | (df_ys['family'] == 'DELI')]


df_ys['family'] = pd.Categorical(df_ys['family']).remove_categories(remlFam)
df_ys['mxSum'] = df_ys['mxSum']/1000000
df_ys['meSales'] = df_ys['meSales']


f, ax = plt.subplots(figsize=(20,20))
sns.set_context("notebook", font_scale=2)
ax = sns.pointplot(x="year", y="meSales",hue="family",  data=df_ys)
ax = sns.barplot(x='year', y='mxPrice', data=df_oilP, palette="gist_gray")
for p in ax.patches:
 height = p.get_height() # height of each horizontal bar is the same
 width = p.get_width() # width (average number of passengers)
 # adding text to each bar
 ax.text(x = p.get_x()+(width/2), # x-coordinate position of data label, padded 3 to right of bar
 y = height+8,# # y-coordinate position of data label, padded to be in the middle of the bar
 s = '{:.0f}'.format(height), # data label, formatted to ignore decimals
 va = 'center', # sets vertical alignment (va) to center
 fontsize=18)
    
# for name, sales in zip(da_fs['family'], da_fs['pSales']):
#     ax.text(name, sales, round(sales, 2), color='white', ha='center')
# Add a legend and informative axis label
# ax.legend(ncol=2, loc='lower right', frameon=True)

ax.set_title("Average Sales of Products each Year and Oil Price", fontdict={'fontsize':24})

ax.set(ylabel="Average Sales",
       xlabel="Year")
sns.despine(left=True, bottom=True)

C>  Good, I would like to visit your store, let me check my calendar.

SM> You can plan your visit on Workdays.  Stores are overcrowded on holidays.

C> Thank you.

SM> Happy shoping. 

Average sales on working and non-working days.
# In[ ]:


df_ho = df_trainM.groupby(['family','holiday_type']).agg(mxSum = ('sales',sum), 
                                         meSales = ('sales',np.mean)).reset_index().sort_values(by=['holiday_type','meSales'], ascending=False)
# print(df_ho)

df_ho = df_ho[(df_ho['family'] == 'GROCERY I') | (df_ho['family'] == 'BEVERAGES') | 
        (df_ho['family'] == 'PRODUCE') | (df_ho['family'] == 'CLEANING') |
        (df_ho['family'] == 'DAIRY') | (df_ho['family'] == 'BREAD/BAKERY') |
        (df_ho['family'] == 'POULTRY') | (df_ho['family'] == 'MEATS') |
        (df_ho['family'] == 'PERSONAL CARE') | (df_ho['family'] == 'DELI')]

# ms = set(df_trainM['family'])
# s = ['HOME AND KITCHEN II','GROCERY II','HOME AND KITCHEN I','GROCERY I','BEVERAGES','PRODUCE','CLEANING','DAIRY','BREAD/BAKERY','POULTRY','MEATS','PERSONAL CARE','DELI','HOME CARE','EGGS',
#      'FROZEN FOODS','PREPARED FOODS','LIQUOR,WINE,BEER','SEAFOOD','GROCERY II','HOME AND KITCHEN I','HOME AND KITCHEN II']
# ss = set(s)

df_ho['family'] = pd.Categorical(df_ho['family']).remove_categories(remlFam)
# df_ho['family'] = pd.Categorical(df_ho['family']).remove_categories(list(ms.symmetric_difference(ss)))

f, ax = plt.subplots(figsize=(20,20))
sns.set_context("notebook", font_scale=2)
ax = sns.pointplot(x="holiday_type", y="meSales",hue="family",  data=df_ho)

ax.set_title("Average Sales on Working and Non-Working Days", fontdict={'fontsize':24})
ax.set(ylabel="Average Sales",
       xlabel="Working and Non Working Days")
sns.despine(left=True, bottom=True)


# **Annual Sales meeting**
# 
# Stakeholder(SH)> Hey guys, How are we doing?
# 
# SM > We are consistently good in business our sales have increased month on month each year. Especially Christmas (December)

# In[ ]:


df_yrmnth = df_trainM.groupby(['year', 'month']).agg(mxSum = ('sales',sum), 
                                         meSales = ('sales',np.mean)).reset_index().sort_values(by=['month','meSales'], ascending=False)

df_yrmnth['meSales'] = np.round(df_yrmnth['meSales']).astype('int')
df_yrmnth['mxSum'] = np.round(df_yrmnth['mxSum']).astype('int')
# print(df_yrmnth)
sns.set_context("notebook", font_scale=2)
f, ax = plt.subplots(figsize=(20,20))
df_HM = df_yrmnth.pivot("month", "year", "meSales")

# m = {1:'j',2:'f',3:'m',4:'a',5:'ma',6:'j',7:'ju',8:'Au',9:'Se',10:'Oct',11:'Nov',12:'Dec'}
ax.set_title("Average sales by monthly, each year", fontdict={'fontsize':24})
ax.set(ylabel="Months", xlabel="Year")
# Draw a heatmap with the numeric values in each cell
sns.heatmap(df_HM, annot=True, fmt="f", linewidths=.1, ax=ax, cmap="YlGnBu")


# SH>  What are the sales in each store?
# 
# SM > Our Average sales in Store "A" are exceptionally high, but we have to concentrate on B, C, D and E stores.

# In[ ]:


df_stSales = df_trainM.groupby(['store_type']).agg(mxSum = ('sales',sum), 
                                         meSales = ('sales',np.mean)).reset_index().sort_values(by=['meSales'], ascending=False)

# print(df_stSales)
f, ax = plt.subplots(figsize=(20,20))
sns.set_context("notebook", font_scale=2)
# ax = sns.pointplot(x="store_type", y="meSales",hue="family",  data=df_stfa)
ax = sns.barplot(x='store_type', y='meSales', data=df_stSales, palette="gist_gray")

# label each bar in barplot
for p in ax.patches:
 height = p.get_height() # height of each horizontal bar is the same
 width = p.get_width() # width (average number of passengers)
 # adding text to each bar
 ax.text(x = p.get_x()+(width/2), # x-coordinate position of data label, padded 3 to right of bar
 y = height+8,# # y-coordinate position of data label, padded to be in the middle of the bar
 s = '{:.0f}'.format(height), # data label, formatted to ignore decimals
 va = 'center', # sets vertical alignment (va) to center
 fontsize=16)
    
# for name, sales in zip(da_fs['family'], da_fs['pSales']):
#     ax.text(name, sales, round(sales, 2), color='white', ha='center')
# Add a legend and informative axis label
# ax.legend(ncol=2, loc='lower right', frameon=True)
ax.set(ylabel="Average Sales",
       xlabel="Store Type")
ax.set_title("Average sales on each store type", fontdict={'fontsize':24})
sns.despine(left=True, bottom=True)


# SH> What are the stats on the "Promotion" strategy?
# 
# SM> There is 65% growth in sales.

# In[ ]:


df_trainM['promoted'] = np.where(df_trainM['onpromotion']> 0, True, False)

df_cl = df_trainM.groupby(['promoted']).agg(Sales = ('sales',np.sum)).reset_index().sort_values(by=['Sales'],
                                                                                                   ascending=True)
df_cl['Sales'] = np.round(df_cl['Sales']).astype('int')
df_cl['pSales'] = (df_cl['Sales']/sum(df_trainM['sales'])*100)
# print(df_cl)

f, ax = plt.subplots(figsize=(12,10))
sns.set_context("notebook", font_scale=2)
ax = sns.barplot(x='promoted', y='pSales', data=df_cl, palette="BuGn")
for p in ax.patches:
 height = p.get_height() # height of each horizontal bar is the same
 width = p.get_width() # width (average number of passengers)
 # adding text to each bar
 ax.text(x = p.get_x()+(width/2), # x-coordinate position of data label, padded 3 to right of bar
 y = height+1,# # y-coordinate position of data label, padded to be in the middle of the bar
 s = '{:.0f}%'.format(height), # data label, formatted to ignore decimals
 va = 'center', # sets vertical alignment (va) to center
 fontsize=16)
    
# for name, sales in zip(da_fs['family'], da_fs['pSales']):
#     ax.text(name, sales, round(sales, 2), color='white', ha='center')
# Add a legend and informative axis label
# ax.legend(ncol=2, loc='lower right', frameon=True)
ax.set(ylabel="% Sales",
       xlabel="On Promotion")
ax.set_title("% Sales when being promoted", fontdict={'fontsize':24})
sns.despine(left=True, bottom=True)


# **Conclusion**
# 
# SH> Now, we have the data and stats. Let's predict our sales next a couple of years and plan strategies accordingly.
# 
# 
