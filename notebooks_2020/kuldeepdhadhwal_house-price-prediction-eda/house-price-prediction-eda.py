#!/usr/bin/env python
# coding: utf-8

# # House Prediction EDA

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from scipy.stats import skew
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style("whitegrid")
import os


# In[ ]:


path = '/kaggle/input/home-data-for-ml-course/'


# In[ ]:


train_df = pd.read_csv(path+'train.csv')
test_df = pd.read_csv(path+'/test.csv')
sub_df = pd.read_csv(path+'/sample_submission.csv')


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df.columns


# In[ ]:


test_df.columns


# In[ ]:


train_df.SalePrice.describe()


# In[ ]:


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
sns.distplot(train_df.SalePrice, bins=50)
plt.title('Original')

plt.subplot(1,2,2)
sns.distplot(np.log1p(train_df.SalePrice), bins=50)
plt.title('Log transformed')


# In[ ]:


train_df.SalePrice.skew()


# In[ ]:


train_df.SalePrice.kurt()


# In[ ]:


train_df['GrLivArea']


# In[ ]:


var = 'GrLivArea'
data = pd.concat([train_df["SalePrice"], train_df[var]], axis=1)
data.head()


# In[ ]:


data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))


# In[ ]:


corr_matrix = train_df.corr()


# In[ ]:


sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr_matrix, vmax=.8, square=True)
sns.heatmap


# In[ ]:


k = 10 #number of variables for heatmap
cols = corr_matrix.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:


sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_df[cols], size = 2.5)
plt.show()


# In[ ]:


total = train_df.isnull().sum().sort_values(ascending = False)
percent = (train_df.isnull().sum() / train_df.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)


# In[ ]:


msno.matrix(train_df.sample(500))


# In[ ]:


msno.bar(train_df)


# In[ ]:


msno.heatmap(train_df)


# # Data Cleaning
# 
# Overview
# - remove skewenes of target feature
# - remove skewenes of numeric features is exists
# - handle missing values in categorical features
# - handle missing values in numerical features
# - feature selection

# # Target Variable

# In[ ]:


target = train_df['SalePrice']
target_log = np.log1p(train_df['SalePrice'])


# # Concat train and test dataset in order for pre-processing
# In order to apply transformations on data, we have to concatenate both datasets: train and test

# In[ ]:


# drop target variable from train dataset
train = train_df.drop(["SalePrice"], axis=1)
data = pd.concat([train, test_df], ignore_index=True)


# # Split dataframe into numeric and categorical
# Split dataframe into 2 with:
# 
# - categorical feature space
# - numerical feature space

# In[ ]:


data.head()


# In[ ]:


# save all categorical columns in list
categorical_columns = [col for col in data.columns.values if data[col].dtype == 'object']

# dataframe with categorical features
data_cat = data[categorical_columns]
# dataframe with numerical features
data_num = data.drop(categorical_columns, axis=1)


# In[ ]:


data_num.head(1)


# In[ ]:


data_num.describe()


# In[ ]:


data_cat.head(1)


# # Reduce skewness for numeric features

# In[ ]:


data_num.head()


# In[ ]:


data_num_skew = data_num.apply(lambda x: skew(x.dropna()))
data_num_skew = data_num_skew[data_num_skew > .75]

# apply log + 1 transformation for all numeric features with skewnes over .75
data_num[data_num_skew.index] = np.log1p(data_num[data_num_skew.index])


# In[ ]:


data_num_skew


# In[ ]:


data_num.drop


# # handling missing values in numerical columns

# In[ ]:


data_len = data_num.shape[0]

# check what is percentage of missing values in categorical dataframe
for col in data_num.columns.values:
    missing_values = data_num[col].isnull().sum()
    #print("{} - missing values: {} ({:0.2f}%)".format(col, missing_values, missing_values/data_len*100)) 
    
    # drop column if there is more than 50 missing values
    if missing_values > 50:
        #print("droping column: {}".format(col))
        data_num = data_num.drop(col, axis = 1)
    # if there is less than 50 missing values than fill in with median valu of column
    else:
        #print("filling missing values with median in column: {}".format(col))
        data_num = data_num.fillna(data_num[col].median())


# # handling missing values in categorical columns

# In[ ]:


data_len = data_cat.shape[0]

# check what is percentage of missing values in categorical dataframe
for col in data_cat.columns.values:
    missing_values = data_cat[col].isnull().sum()
    #print("{} - missing values: {} ({:0.2f}%)".format(col, missing_values, missing_values/data_len*100)) 
    
    # drop column if there is more than 50 missing values
    if missing_values > 50:
        print("{}".format(col))
        data_cat.drop(col, axis = 1)
    # if there is less than 50 missing values than fill in with median valu of column
    else:
        #print("filling missing values with XXX: {}".format(col))
        #data_cat = data_cat.fillna('XXX')
        pass


# In[ ]:


data_cat.describe()


# In[ ]:


columns = ['Alley',
'BsmtQual',
'BsmtCond',
'BsmtExposure',
'BsmtFinType1',
'BsmtFinType2',
'FireplaceQu',
'GarageType',
'GarageFinish',
'GarageQual',
'GarageCond',
'PoolQC',
'Fence',
'MiscFeature'
]
data_cat = data_cat.drop(columns, axis=1)


# In[ ]:


data_cat.head()


# In[ ]:


data_cat.dropna()


# In[ ]:


data_num.describe()


# In[ ]:


data_num.columns


# In[ ]:


data_cat = pd.get_dummies(data_cat)


# In[ ]:


frames = [data_cat, data_num]
total_df = pd.concat(frames,  axis=1)


# In[ ]:


total_df.head()


# In[ ]:


# target.head()
sub_df.head()


# # split data into train and test

# In[ ]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error


# In[ ]:


train_df = total_df[:1460]
test_df = total_df[1461:]


# In[ ]:


train_df.tail()


# In[ ]:


test_df.tail()


# In[ ]:


sub_df.head()


# In[ ]:


test_df.head()


# In[ ]:


target.tail()


# InProgress ...
