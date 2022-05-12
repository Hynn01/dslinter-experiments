#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import gc, warnings
warnings.filterwarnings('ignore')


# # Overview
# What is this kernel about?
# * No predictions to make 
# * No features to create
# 
# We will load competition data and look closer on it. We will try to understand what we have in our hands and how we can work with it.
# * * *

#  ## Load train data
#  * * *

# In[ ]:


sale_train = pd.read_csv('../input/sales_train.csv')


# We can view basic DafaFrame information. 
# 
# As you can see, we do not have broken and nan data that is good.

# In[ ]:


print("----------Top-5- Record----------")
print(sale_train.head(5))
print("-----------Information-----------")
print(sale_train.info())
print("-----------Data Types-----------")
print(sale_train.dtypes)
print("----------Missing value-----------")
print(sale_train.isnull().sum())
print("----------Null value-----------")
print(sale_train.isna().sum())
print("----------Shape of Data----------")
print(sale_train.shape)


# We have duplicated rows, but I don't think that it is a mistake.
# 
# It could be different sales methods or client type, etc.
# 
# You can remove it, but I really don't believe that 6 rows of 3m can make the difference.

# In[ ]:


print('Number of duplicates:', len(sale_train[sale_train.duplicated()]))


# I can advise downcasting your DataFrame. It will save you some memory, and believe me you will need all memory possible.
# 
# In our case from 134.4+ MB, we went to 61.6+ MB
# 
# Not a great deal right now but such approach works with bigger DF also.
# 
# #### please see this two links for more tips (I stole that downcast basic snippet from anqitu)))
# * https://www.kaggle.com/anqitu/feature-engineer-and-model-ensemble-top-10
# * https://www.kaggle.com/yuliagm/how-to-work-with-big-datasets-on-16g-ram-dask

# In[ ]:


def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df

sale_train = downcast_dtypes(sale_train)
print(sale_train.info())


# ## 1.1 Item_id
# * * *
# ### Lets group data by item_id and date_block_num and look closer on it.
# 

# In[ ]:


sales_by_item_id = sale_train.pivot_table(index=['item_id'],values=['item_cnt_day'], 
                                        columns='date_block_num', aggfunc=np.sum, fill_value=0).reset_index()
sales_by_item_id.columns = sales_by_item_id.columns.droplevel().map(str)
sales_by_item_id = sales_by_item_id.reset_index(drop=True).rename_axis(None, axis=1)
sales_by_item_id.columns.values[0] = 'item_id'


# ### Simple graph
# What this graph is telling us. Basically nothing.)) I only see that train data has many old products (degradation line) and many 1c products are seasonal and probably release date depended.
# 
# #### I'm not very good with graphs and presentations - there are better data representation examples:
# * https://www.kaggle.com/dimitreoliveira/model-stacking-feature-engineering-and-eda
# * https://www.kaggle.com/jagangupta/time-series-basics-exploring-traditional-ts

# In[ ]:


sales_by_item_id.sum()[1:].plot(legend=True, label="Monthly sum")


# In[ ]:


sales_by_item_id.mean()[1:].plot(legend=True, label="Monthly mean")


# ### Let's see how many products are outdated (no sales for the last 6 months)
# 12391 of 21807 is a huge number. Probably we can set 0 for all that items and do not make any model prediction.

# In[ ]:


outdated_items = sales_by_item_id[sales_by_item_id.loc[:,'27':].sum(axis=1)==0]
print('Outdated items:', len(outdated_items))


# ### How many outdated items in test set?
# 6888 - not much but we have such items

# In[ ]:


test = pd.read_csv('../input/test.csv')
print('Outdated items in test set:', len(test[test['item_id'].isin(outdated_items['item_id'])]))


# ### Outliers by price and sales volume
# We will get rid of them later
# 
# #### please see lovely kernel made by Denis Larionov (I stole few graphs from there)
# * https://www.kaggle.com/dlarionov/feature-engineering-xgboost

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.xlim(-100, 3000)
sns.boxplot(x=sale_train['item_cnt_day'])
print('Sale volume outliers:',sale_train['item_id'][sale_train['item_cnt_day']>500].unique())

plt.figure(figsize=(10,4))
plt.xlim(sale_train['item_price'].min(), sale_train['item_price'].max())
sns.boxplot(x=sale_train['item_price'])
print('Item price outliers:',sale_train['item_id'][sale_train['item_price']>50000].unique())


# ### Possible item_id features:
# 1. Lags
# 2. Release date
# 3. Last month sale
# 4. Days on sale
# 5. Neighbors (items with id 1000 and 1001 could be somehow similar - genre, type, release date)

# ## 1.2 shop_id
# * * *
# ### Lets now group train data by shop_id.
# We can see new shops - probably there will be a sales spike (opening event for example).
# Apparently closed shops (ill call it "outdated shops")  - no sales for last 6 months.

# In[ ]:


sales_by_shop_id = sale_train.pivot_table(index=['shop_id'],values=['item_cnt_day'], 
                                        columns='date_block_num', aggfunc=np.sum, fill_value=0).reset_index()
sales_by_shop_id.columns = sales_by_shop_id.columns.droplevel().map(str)
sales_by_shop_id = sales_by_shop_id.reset_index(drop=True).rename_axis(None, axis=1)
sales_by_shop_id.columns.values[0] = 'shop_id'

for i in range(6,34):
    print('Not exists in month',i,sales_by_shop_id['shop_id'][sales_by_shop_id.loc[:,'0':str(i)].sum(axis=1)==0].unique())

for i in range(6,28):
    print('Shop is outdated for month',i,sales_by_shop_id['shop_id'][sales_by_shop_id.loc[:,str(i):].sum(axis=1)==0].unique())


# In our test set we have 5100 sales in really new shop and no "outdated shops" but anyway it is good feature for future.

# In[ ]:


print('Recently opened shop items:', len(test[test['shop_id']==36]))


# ### Possible shop_id features
# 1. Lags (shop_id/shp_cnt_mth)
# 2. Opening month (possible  opening sales)
# 3. Closed Month (possible stock elimination)

# ## 1.3 Price
# * * *
# ### Possible Price features:
# 1. Price category (1$/10$/20$/ etc.) - obviously (or not obviously),  items with smaller price have greater volumes
# 2. Discount and Discount duration
# 3. Price lag (shows discount)
# 4. Price correction (rubl/usd pair)
# 5. Shop Revenue

# ## 1.4 Dates
# * * *
# ### Possible Date features:
# 1. Weekends and holidays sales (to correct monthly sales)
# 2. Number of days in the month (to correct monthly sales)
# 3. Month number (for seasonal items)

# ## 1.5 Shop info
# * * *
# The structure of the shop information is evident.
# ### Shop City | Shop type | Shop name

# In[ ]:


shops = pd.read_csv('../input/shops.csv')
shops.head()


# With a close look we can find out that some shops have duplicated id/name - probably it changed location (within commercial center), or it has a different type (isle sale point), but I decided to merge it.
# * 11 => 10
# * 1  => 58
# * 0  => 57
# * 40 => 39
# 
# I converted train shop_id to shop_id that is in the test set

# In[ ]:


shops['shop_name'] = shops['shop_name'].apply(lambda x: x.lower()).str.replace('[^\w\s]', '').str.replace('\d+','').str.strip()
shops['shop_city'] = shops['shop_name'].str.partition(' ')[0]
shops['shop_type'] = shops['shop_name'].apply(lambda x: 'мтрц' if 'мтрц' in x else 'трц' if 'трц' in x else 'трк' if 'трк' in x else 'тц' if 'тц' in x else 'тк' if 'тк' in x else 'NO_DATA')
shops.head()


# ### Possible Shop features:
# 1. Shop City
# 2. Shop Type

# ## 1.6 Item info
# * * *
# Let's see what we can get from this file.

# In[ ]:


items = pd.read_csv('../input/items.csv')
items.head()


# We can enconde "features" that many items have.
# 
# The structure is always the same
# ### Item name [category feature] (additional feature)
# we can split it, and "one hot encode it."

# In[ ]:


# Ugly code to show the idea
from collections import Counter
from operator import itemgetter
items['name_1'], items['name_2'] = items['item_name'].str.split('[', 1).str
items['name_1'], items['name_3'] = items['item_name'].str.split('(', 1).str

items['name_2'] = items['name_2'].str.replace('[^A-Za-z0-9А-Яа-я]+', ' ').str.lower()
items['name_3'] = items['name_3'].str.replace('[^A-Za-z0-9А-Яа-я]+', ' ').str.lower()
items = items.fillna('0')

result_1 = Counter(' '.join(items['name_2'].values.tolist()).split(' ')).items()
result_1 = sorted(result_1, key=itemgetter(1))
result_1 = pd.DataFrame(result_1, columns=['feature', 'count'])
result_1 = result_1[(result_1['feature'].str.len() > 1) & (result_1['count'] > 200)]

result_2 = Counter(' '.join(items['name_3'].values.tolist()).split(" ")).items()
result_2 = sorted(result_2, key=itemgetter(1))
result_2 = pd.DataFrame(result_2, columns=['feature', 'count'])
result_2 = result_2[(result_2['feature'].str.len() > 1) & (result_2['count'] > 200)]

result = pd.concat([result_1, result_2])
result = result.drop_duplicates(subset=['feature'])

print('Most common aditional features:', result)


# ### Item name correction
# For our basic "name feature" it is enough to find identical items (not similar but identical),

# In[ ]:


print('Unique item names:', len(items['item_name'].unique()))


# In[ ]:


import re
def name_correction(x):
    x = x.lower()
    x = x.partition('[')[0]
    x = x.partition('(')[0]
    x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x)
    x = x.replace('  ', ' ')
    x = x.strip()
    return x

items['item_name'] = items['item_name'].apply(lambda x: name_correction(x))
items.head()


# In[ ]:


print('Unique item names after correction:', len(items['item_name'].unique()))


# ### Possible Item features:
# 1. Item name
# 2. Encoded aditional feature 

# ## 1.7 Category info
# * * *
# The structure here is
# ### Section name - subsection
# we can split it and have two features from one

# In[ ]:


categories = pd.read_csv('../input/item_categories.csv')
categories.head()


# ### But I did manual feature extraction here to have four features.
# Section / Main Category name / Main SubCategory name / Secondary SubCategory name
# #### Аксессуары / PS2	/ PS / 2

# ### Possible Category features
# 1. Section
# 2. Main Category name
# 3. Main SubCategory name 
# 4. Secondary SubCategory name
# 

# ## 1.8 Test Set
# * * *
# The key to my success was the analysis of Test test data.
# 
# We have three groups of items:
# 1. Item/shop pairs that are in train
# 2. Items without any data
# 3. Items that are in train

# In[ ]:


test = pd.read_csv('../input/test.csv')
good_sales = test.merge(sale_train, on=['item_id','shop_id'], how='left').dropna()
good_pairs = test[test['ID'].isin(good_sales['ID'])]
no_data_items = test[~(test['item_id'].isin(sale_train['item_id']))]

print('1. Number of good pairs:', len(good_pairs))
print('2. No Data Items:', len(no_data_items))
print('3. Only Item_id Info:', len(test)-len(no_data_items)-len(good_pairs))
  


# #### Is it feature? Yes. We need to apply different prediction approach for each type of items in the test set.
# ####  For example - "No Data Items" - it is more likely classification task.

# ### Next part will be about data aggregation and feature preparation.
# ## To be continued...

# 
