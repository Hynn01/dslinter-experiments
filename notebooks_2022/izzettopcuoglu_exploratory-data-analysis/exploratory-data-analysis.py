#!/usr/bin/env python
# coding: utf-8

# **Exploratory Data Analysis**

# In this notebook i aim to provide usefull codes for exploratory data analysis. hopefully you'll find it usefull when you face a dataframe that you are not familiar with. I'll use **titanic** dataset for convenience

# In[ ]:


import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[ ]:


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', None)
df = sns.load_dataset('titanic')


# First, a general overview. 
# * What is the shape ? 
# * Are there any null values ? 
# * General look at the head of df
# * Mean, std, and quantile values of numeric values of dataframe

# In[ ]:


print("##################### Shape #####################")
print(df.shape)
print("##################### Types #####################")
print(df.dtypes)
print("##################### Head #####################")
print(df.head())
print("##################### NA #####################")
print(df.isnull().sum())
print("##################### Quantiles #####################")
print(df.describe(percentiles=[0, 0.05, 0.25, 0.50, 0.95, 0.99, 1]).T)


# Now a function to divide the dataframe into categoric, numeric and categoric but cardinal columns, let me briefly explain : 
# 
# **Categoric columns**: dtype is not in int or float and nunique is less than 20 + dtype is in int or float and nunique is less than 10 ( numeric values changeable) 
# 
# **Numeric columns**: dtype is in int or float and nunique is more than 10 
# 
# **Cat but car**: dtype is not in int or float and nunique is more than 20
# 
# This is a very usefull function to divide a dataframe and in rest of this notebook we will inspect the dataframe as categorical columns and numerical columns , categorical but cardinal values are generally names, customer Ids etc.

# In[ ]:


def grab_col_names(dataframe, cat_th=10, car_th=20):


    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes !="O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car


# In[ ]:


cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)


# Lets inspect them real quick

# In[ ]:


df[cat_cols].head(10)


# In[ ]:


df[num_cols].head(10)


# so we have our categorical and numerical columns. we can move on with furthermore inspection on them. 

# In[ ]:


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


# In[ ]:


for col in cat_cols: 
    cat_summary(df,col)


# In[ ]:


for col in num_cols: 
    num_summary(df,col)


# In[ ]:


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


# In[ ]:


for col in cat_cols: 
    target_summary_with_cat(df, 'survived',col)


# In[ ]:


for col in num_cols: 
    target_summary_with_num(df, 'survived', col)


# I find above functions very usefull especially working on features. We can move on with missing values. This function gives us a ratio of missing values and creates a na values list, after that we can group it with our target variable, target variable is the most important column. It will vary depending on your project. Lets say you are working on a ml model that will predict your survival on titanic then survived is your target variable. but if youre working on ticket fees then your target variable is fare

# In[ ]:


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


# In[ ]:


missing_values_table(df,na_name=True)


# In[ ]:


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


# In[ ]:


na_columns = ['age', 'embarked', 'deck', 'embark_town']


# In[ ]:


missing_vs_target(df, "survived", na_columns)


# once you have specified missing values then you can decide what to do with them, you may delete or fill them with mean or median for instance. Proceeding with outliners. 

# In[ ]:


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


# With above 2 functions you can determine the outlier limits and check if there is any outlier within columns

# In[ ]:


for col in num_cols:
    outlier_thresholds(df, col, q1=0.25, q3=0.75)


# In[ ]:


check_outlier(df, 'fare', q1=0.25, q3=0.75)


# In[ ]:


check_outlier(df, 'age', q1=0.25, q3=0.75)


# In[ ]:


check_outlier(df, 'fare', q1=0.25, q3=0.75)


# Once you have settled the outliers you may want to press them.

# This was all for now. I hope you find it helpfull, see you in next notebook
