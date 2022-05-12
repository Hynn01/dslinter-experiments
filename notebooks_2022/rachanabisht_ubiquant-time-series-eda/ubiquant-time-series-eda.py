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
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#import libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import random
import math
from datetime import datetime
from scipy.stats import norm


# # Load dataset
# The size of the competition's dataset is 18.55gb. It is very big so, we will use another dataset which has been converted low memory (about 3.63gb).
# 
# Credits: https://www.kaggle.com/robikscube/ubiquant-parquet

# In[ ]:


train_df = pd.read_parquet('../input/ubiquant-parquet/train_low_mem.parquet')
train_df


# In[ ]:


train_df.shape


# In[ ]:


train_df.head()


# In[ ]:


# finding unique values in each columns
#for col in train_df.columns:
 #   print(col + ":" + str(len(train_df[col].unique())))


# In[ ]:


# check for missing values: the data is free of missing values.
train_df.isnull().any().sum()


# We can see each train_csv has:
# 
# 1. row_id - A unique identifier for the row
# 
# 2. time_id - ID for the time the data was collected. Not all investments have data for all the time IDs
# 
# 3. investment_id - ID for each individual investment
# 
# 4. target - The target
# 
# 5. f_0:f_299 - features generated from the investment data at that time ID

# 
# # EDA

# A. Relationship between investment ids and time ids:

# In[ ]:


#What's the maximum time_id per investment id? Most of the investment ids have a max time_id of 1219. 
train_df.groupby("investment_id").time_id.max().value_counts()


# In[ ]:


#investment_id per time-id:
inv_per_time = train_df.groupby("time_id").investment_id.nunique()
inv_per_time = inv_per_time.reset_index()
inv_per_time.columns = ['time_id', 'inv_id']
inv_per_time


# In[ ]:


# plot a line graph for distrinution of investments with time: 
#The number of unique investment ids seems to be roughly constant in the beginning but has an increasind trend after the 400.
plt.figure(figsize=(15, 5))
sns.lineplot(x='time_id', y='inv_id', data=inv_per_time)
#sns.lineplot(x='inv_id', y='time_id', data=inv_per_time)
plt.xlabel('Investments')
plt.ylabel('Time Stamp')
plt.title('Number of unique Investments with time');


# In[ ]:


# Time_id distribution:
sns.distplot(train_df['time_id'])
plt.figure(figsize=(40,20))


# #Summary:
# 1. The train.csv dataset contains 300 anonymous features that don't have any description, investment_id, and target that is also some anonymous float value.
# 2. It is free from any missing values.
# 3. There are 3141410 samples, of which 3579 unique investment ids, and 1211 unique time ids.  All of the investments doesn't necessarily appear in all time IDs.
# 4. There are no investment ids with an time id > 1219 or < 62. 
# 5. The number of unique investment ids seems to be roughly constant in the beginning but has an increasind trend after the 400.

# # TARGET distribution
# 

# In[ ]:


# Target distribution:
f, axes = plt.subplots(2, 1, figsize=(15, 8))

plt.suptitle("Distribution of the target", size=14)

# Target histogram
train_df["target"].hist(bins=50, ax=axes[0])

# Target Boxplot
sns.boxplot(x="target", data=train_df, ax=axes[1])
plt.show()


# In[ ]:


#How is mean of target distributed over time?
target_time = train_df.groupby('time_id')['target'].mean()
target_time = target_time.reset_index()
target_time.columns = ['time_id', 'target']
plt.figure(figsize=(15, 5))
sns.lineplot(x='time_id', y='target', data=target_time)
plt.xlabel('time stamp')
plt.ylabel('Target')
plt.title("Mean of Target by time");


# In[ ]:


#How is target distributed over time?
target_time = train_df.groupby('time_id')['target'].size()
target_time = target_time.reset_index()
target_time.columns = ['time_id', 'target']
plt.figure(figsize=(15, 5))
sns.lineplot(x='time_id', y='target', data=target_time)
plt.xlabel('time stamp')
plt.ylabel('Target')
plt.title("Number of Unique Target by time");


# In[ ]:


#Distribution of target with respect to investment id
obs_by_assets = train_df.groupby(['investment_id'])['target'].count()

obs_by_assets.plot(kind='hist', bins=100)
plt.title('target by asset distribution')
plt.show()


# There are more targets with investment_id with high values count.

# In[ ]:


#What is the average target value of each investment_id?
mean_targets = train_df.groupby(['investment_id'])['target'].mean()
mean_mean_targets = mean_targets.mean()
mean_targets.plot(kind='hist', bins=100)
plt.title('target mean distribution')
plt.show()

print(f"Mean of all targets: {mean_mean_targets: 0.5f}")


# In[ ]:


#Let's plot the target distribution for some investment_ids :
np.random.seed(1)

# Initiate counter
i = 1

# Initiate plot
plt.figure(figsize=(10, 8))
plt.suptitle("Target distribution for 6 random investment_id", size=14)

# Plot randomly 6 histograms of the target
for j in np.random.choice(train_df["investment_id"].unique(), 6):
    plt.subplot(2, 3, i)
    train_df[train_df["investment_id"] == j]["target"].hist(bins=50)
    plt.title("Target distribution\nfor investment_id {}".format(j), size=10)
    i += 1


# 1. There are 3066513 target values look quite normal without any outliers or long tails. We should not have any problems working with it. 
# 2. Target means in time IDs are centered around 0 and they are quite balanced even though there are some outliers, but outliers look very natural. 
# 3. Target might be correlated with number of samples but it is hard to tell which causes which. Very high and very low target mean values are also observed in the same period (between time_id 350 and 550).
# 4. For individual investment_id, target distribution seems to be less gaussian. Some values are high for values being at the "tail of the distribution" (e.g. investment_id 2441, 1337).

# # Features Distributions
# 

# In[ ]:


# summary statstics:
train_df.describe()


# lets take a look at few random features:

# In[ ]:


f = 'f_67'
train_df[f].hist(bins = 100, figsize = (8,6))


# In[ ]:


f = 'f_80'
train_df[f].hist(bins = 100, figsize = (8,4))


# In[ ]:


f = 'f_150'
train_df[f].hist(bins = 100, figsize = (8,4))


# In[ ]:


f = 'f_234'
train_df[f].hist(bins = 100, figsize = (8,4))


# In[ ]:


#Disrtibution of 300 features:
columns = [f"f_{i}" for i in range(300)]
train_df.hist(column = columns, bins = 100, figsize = (30,30));


# 1. There are 300 anonymized continuous features in dataset and they are named from f_0 to f_299. All of the features are zero-centered and they have standard deviation of one since they are standardized during the anonymization process. Most of the features have symmetrical normal distributions but some of them have very extreme outliers which are skewing their distributions.
# 
# 2. Observing summary stats of DataFrame, we can find that most of the features have mean value close to zero and std close to one.
# 
# 3. Some features look normal, but most have outliers, skewed distribution, and multiple modes. Probably the analysis of features one by one will bring a lot of value later in the competition, but we will not go deep into it in this notebook.
# 
# 4. Some features are centered in zero.
# 
# 5. Some of them get outliers as the distribution is not centered. So maybe, in the future we could consider to normalize data with a Robust Scaler in order to limit the influence of outliers.
# 

# # Target Features Relationship:
# One could also use correlation to examine the LINEAR relationship between the target and feature variables. Usually, features that are highly correlated to the target are useful for the model. However, it does not indicate that lower correlated features should be discarded.
# 

# In[ ]:


n_features = 300
features = [f'f_{i}' for i in range(n_features)] 


# In[ ]:


train_corr = train_df[features+['target']].loc[:1000].corr(method='pearson')


# In[ ]:


train_corr.style.background_gradient(cmap='coolwarm', axis=None)


# In[ ]:


# top ten features which shows a high correlation with target:
train_corr.nlargest(10, 'target').index


# In[ ]:


fig, ax = plt.subplots(1, 2)
plt.style.use('bmh')
plt.rcParams['figure.figsize'] = (16, 10)

train_corr.nlargest(10, 'target').plot(kind='barh', title='Most Predictive Features', legend=False, ax=ax[0])
ax[0].set_xlabel("Pearson Corr with Target")

train_corr.nsmallest(10, 'target').plot(kind='barh', title='Least Predictive Features', legend=False, ax=ax[1])
ax[1].set_xlabel("Pearson Corr with Target");


# 1. Most of correlations between the traget and features are low >0.2.
# 
# 2. We are going to see the highest correlations. Generally, it is considered that high correlation is above 0.8. we find that features :'f_67', 'f_73', 'f_148', 'f_226', 'f_204', 'f_140', 'f_295','f_211', 'f_143'
# 
# 3. We can see that several features are correlated to more than one feature, such as f_4, f_228, f_41, f_95, f_97...
# 
# next we will try to apply ARIMA model in a new notebook.... 

# In[ ]:





# In[ ]:





# References:
# 1. https://www.kaggle.com/code/allunia/ubiquant-eda
# 2. https://www.kaggle.com/code/jnegrini/ubiquant-eda
# 3. https://www.kaggle.com/code/morenovanton/feature-exploration-analytics-ubiquant-market
# 

# In[ ]:




