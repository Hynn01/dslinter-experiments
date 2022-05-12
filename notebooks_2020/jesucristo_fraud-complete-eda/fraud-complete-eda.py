#!/usr/bin/env python
# coding: utf-8

# <span style="font-family:Calibri; font-size:3em; color:blue">IEEE Fraud Detection</span>
# 
# <br>
# <img src="https://cdn.datafloq.com/cache/blog_pictures/878x531/fraud-analytics-protect-banking-sector.jpg" width="500" height="600">
# <br>
# 
# 
# **Why fraud detection?**
# > Fraud is a billion-dollar business and it is increasing every year. The PwC global economic crime survey of 2018[1] found that half (49 percent) of the 7,200 companies they surveyed had experienced fraud of some kind. This is an increase from the PwC 2016 study in which slightly more than a third of organizations surveyed (36%) had experienced economic crime.
# 
# 
# This competition is a **binary classification** problem - i.e. our target variable is a binary attribute (Is the user making the click fraudlent or not?) and our goal is to classify users into "fraudlent" or "not fraudlent" as well as possible.
# 
# Unlike metrics such as ```LogLoss```, the **AUC score** only depends on how well you well you can separate the two classes. In practice, this means that only the order of your predictions matter, as a result of this, any rescaling done to your model's output probabilities will have no effect on your score. [click here to read more about AUC-ROC](https://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it)
# 
# <img src='https://upload.wikimedia.org/wikipedia/commons/6/6b/Roccurves.png' width=300 height=300>
# 
# 
# ### Content
# 
# - Data exploration
# - Missing Data.
# - Imbalanced problem.
# 
# 
# - Plots
#     - Distribution plots
#     - Count plots
#     - Unique values
#     - Groups
#     
#     
# - Memory reduction  
# 
# - PCA
# 
# 
# - Models
#     - XGBoost Model.
#     - LGBM
#     
# **Remember the <span style="color:red">upvote</span> button is next to the fork button, and it's free too! ;)**
# 
# ----
# 
# ### References:
# 
# - https://www.kaggle.com/artgor/eda-and-models/data
# - https://www.kaggle.com/artkulak/ieee-fraud-simple-baseline-0-9383-lb
# - https://www.kaggle.com/robikscube/ieee-fraud-detection-first-look-and-eda
# - https://www.kaggle.com/mjbahmani/reducing-memory-size-for-ieee
# 
# <br>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import xgboost as xgb
import lightgbm as lgb
import catboost
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.gridspec as gridspec
get_ipython().run_line_magic('matplotlib', 'inline')

# Standard plotly imports
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import iplot, init_notebook_mode
import cufflinks
import cufflinks as cf
import plotly.figure_factory as ff

# Using plotly + cufflinks in offline mode
init_notebook_mode(connected=True)
cufflinks.go_offline(connected=True)


import warnings
warnings.filterwarnings("ignore")

import gc
gc.enable()

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print ("Ready!")


# # Data
# 
# 
# In this competition you are predicting the probability that an online transaction is fraudulent, as denoted by the binary target ```isFraud```.
# 
# The data is broken into two files **identity** and **transaction**, which are joined by ```TransactionID```. 
# 
# > Note: Not all transactions have corresponding identity information.
# 
# **Categorical Features - Transaction**
# 
# - ProductCD
# - emaildomain
# - card1 - card6
# - addr1, addr2
# - P_emaildomain
# - R_emaildomain
# - M1 - M9
# 
# **Categorical Features - Identity**
# 
# - DeviceType
# - DeviceInfo
# - id_12 - id_38
# 
# **The TransactionDT feature is a timedelta from a given reference datetime (not an actual timestamp).**
# 
# **Files**
# 
# - train_{transaction, identity}.csv - the training set
# - test_{transaction, identity}.csv - the test set (**you must predict the isFraud value for these observations**)
# - sample_submission.csv - a sample submission file in the correct format
# 

# **Interactive Plots Utils**
# > from https://www.kaggle.com/kabure/baseline-fraud-detection-eda-interactive-views (more about Interactive plots there)

# In[ ]:


# functions from: https://www.kaggle.com/kabure/baseline-fraud-detection-eda-interactive-views

def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary

def plot_distribution(df, var_select=None, title=None, bins=1.0): 
    # Calculate the correlation coefficient between the new variable and the target
    tmp_fraud = df[df['isFraud'] == 1]
    tmp_no_fraud = df[df['isFraud'] == 0]    
    corr = df['isFraud'].corr(df[var_select])
    corr = np.round(corr,3)
    tmp1 = tmp_fraud[var_select].dropna()
    tmp2 = tmp_no_fraud[var_select].dropna()
    hist_data = [tmp1, tmp2]
    
    group_labels = ['Fraud', 'No Fraud']
    colors = ['seagreen','indianred', ]

    fig = ff.create_distplot(hist_data,
                             group_labels,
                             colors = colors, 
                             show_hist = True,
                             curve_type='kde', 
                             bin_size = bins
                            )
    
    fig['layout'].update(title = title+' '+'(corr target ='+ str(corr)+')')

    iplot(fig, filename = 'Density plot')
    
def plot_dist_churn(df, col, binary=None):
    tmp_churn = df[df[binary] == 1]
    tmp_no_churn = df[df[binary] == 0]
    tmp_attr = round(tmp_churn[col].value_counts().sort_index() / df[col].value_counts().sort_index(),2)*100
    print(f'Distribution of {col}: ')
    trace1 = go.Bar(
        x=tmp_churn[col].value_counts().sort_index().index,
        y=tmp_churn[col].value_counts().sort_index().values, 
        name='Fraud',opacity = 0.8, marker=dict(
            color='seagreen',
            line=dict(color='#000000',width=1)))

    trace2 = go.Bar(
        x=tmp_no_churn[col].value_counts().sort_index().index,
        y=tmp_no_churn[col].value_counts().sort_index().values,
        name='No Fraud', opacity = 0.8, 
        marker=dict(
            color='indianred',
            line=dict(color='#000000',
                      width=1)
        )
    )

    trace3 =  go.Scatter(   
        x=tmp_attr.sort_index().index,
        y=tmp_attr.sort_index().values,
        yaxis = 'y2', 
        name='% Fraud', opacity = 0.6, 
        marker=dict(
            color='black',
            line=dict(color='#000000',
                      width=2 )
        )
    )
    
    layout = dict(title =  f'Distribution of {str(col)} feature by %Fraud',
              xaxis=dict(type='category'), 
              yaxis=dict(title= 'Count'), 
              yaxis2=dict(range= [0, 15], 
                          overlaying= 'y', 
                          anchor= 'x', 
                          side= 'right',
                          zeroline=False,
                          showgrid= False, 
                          title= 'Percentual Fraud Transactions'
                         ))

    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    iplot(fig)


# **Load data**

# In[ ]:


print('# File sizes')
for f in os.listdir('../input'):
    if 'zip' not in f:
        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_transaction = pd.read_csv(\'../input/train_transaction.csv\', index_col=\'TransactionID\')\ntest_transaction = pd.read_csv(\'../input/test_transaction.csv\', index_col=\'TransactionID\')\ntrain_identity = pd.read_csv(\'../input/train_identity.csv\', index_col=\'TransactionID\')\ntest_identity = pd.read_csv(\'../input/test_identity.csv\', index_col=\'TransactionID\')\nprint ("Data is loaded!")')


# In[ ]:


print('train_transaction shape is {}'.format(train_transaction.shape))
print('test_transaction shape is {}'.format(test_transaction.shape))
print('train_identity shape is {}'.format(train_identity.shape))
print('test_identity shape is {}'.format(test_identity.shape))


# In[ ]:


train_transaction.head()


# In[ ]:


train_identity.head()


# OK, there are a lot of **NaN** and **interesting columns**: 
# 
# - ``` C1, C2 ... D1, V300, V339 ... ``` 
# - ``` id_01 ... id_38``` 
# 
# The columns with those names don't look friendly.
# Apparently we don't have **dates**.

# ### 1st problem: NaN
# 
# Remember
# > Not all transactions have corresponding identity information

# **train_transaction**

# In[ ]:


missing_values_count = train_transaction.isnull().sum()
print (missing_values_count[0:10])
total_cells = np.product(train_transaction.shape)
total_missing = missing_values_count.sum()
print ("% of missing data = ",(total_missing/total_cells) * 100)


# **train_identity**

# In[ ]:


missing_values_count = train_identity.isnull().sum()
print (missing_values_count[0:10])
total_cells = np.product(train_identity.shape)
total_missing = missing_values_count.sum()
print ("% of missing data = ",(total_missing/total_cells) * 100)


# In[ ]:


del missing_values_count, total_cells, total_missing


# ### 2nd Problem ...
# 
# Notice how **imbalanced** is our original dataset! Most of the transactions are non-fraud. If we use this dataframe as the base for our predictive models and analysis we might get a lot of errors and our algorithms will probably overfit since it will "assume" that most transactions are not fraud. But we don't want our model to assume, we want our model to detect patterns that give signs of fraud!
# 
# **Imbalance** means that the number of data points available for different the classes is different
# 
# <img src='https://www.datascience.com/hs-fs/hubfs/imbdata.png?t=1542328336307&width=487&name=imbdata.png'>

# In[ ]:


x = train_transaction['isFraud'].value_counts().index
y = train_transaction['isFraud'].value_counts().values

trace2 = go.Bar(
     x=x ,
     y=y,
     marker=dict(
         color=y,
         colorscale = 'Viridis',
         reversescale = True
     ),
     name="Imbalance",    
 )
layout = dict(
     title="Data imbalance - isFraud",
     #width = 900, height = 500,
     xaxis=go.layout.XAxis(
     automargin=True),
     yaxis=dict(
         showgrid=False,
         showline=False,
         showticklabels=True,
 #         domain=[0, 0.85],
     ), 
)
fig1 = go.Figure(data=[trace2], layout=layout)
iplot(fig1)


# In[ ]:


del x,y
gc.collect()


# # Time vs fe
# > **The TransactionDT feature is a timedelta from a given reference datetime (not an actual timestamp).**
# 
# **Important ! read the post [The timespan of the dataset is 1 year ?
# ](https://www.kaggle.com/c/ieee-fraud-detection/discussion/100071#latest-577632) by Suchith**
# 
# ```
# Train: min = 86400 max = 15811131
# Test: min = 18403224 max = 34214345
# ```
# 
# The difference train.min() and test.max() is ```x = 34214345 - 86400 = 34127945``` but we don't know is it in seconds,minutes or hours.
# 
# ```
# Time span of the total dataset is 394.9993634259259 days
# Time span of Train dataset is  181.99920138888888 days
# Time span of Test dataset is  182.99908564814814 days
# The gap between train and test is 30.00107638888889 days
# ```
# 
# If it is in seconds then dataset timespan will be ```x/(3600*24*365) = 1.0821``` years which seems reasonable to me. So if the **transactionDT** is in **seconds** then
# 
# ```
# Time span of the total dataset is 394.9993634259259 days
# Time span of Train dataset is  181.99920138888888 days
# Time span of Test dataset is  182.99908564814814 days
# The gap between train and test is 30.00107638888889 days
# ```
# 
# 
# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2370491%2Fc9bf5af5e902595b737df5470adc193b%2Fdownload-1.png?generation=1563312982845419&alt=media)
# 
# **source: [FChmiel](https://www.kaggle.com/fchmiel)**
# <br>

# In[ ]:


# Here we confirm that all of the transactions in `train_identity`
print(np.sum(train_transaction.index.isin(train_identity.index.unique())))
print(np.sum(test_transaction.index.isin(test_identity.index.unique())))


# ```24.4%``` of TransactionIDs in train (144233 / 590540) have an associated train_identity.
# 
# ```28.0%``` of TransactionIDs in test (144233 / 590540) have an associated train_identity.

# In[ ]:


train_transaction['TransactionDT'].head()


# In[ ]:


train_transaction['TransactionDT'].shape[0] , train_transaction['TransactionDT'].nunique()


# **TransactionDT** is not a timestamp, but somehow we use it to measure time.

# In[ ]:


train_transaction['TransactionDT'].value_counts().head(10)


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(18,4))

time_val = train_transaction['TransactionDT'].values

sns.distplot(time_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of TransactionDT', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])

sns.distplot(np.log(time_val), ax=ax[1], color='b')
ax[1].set_title('Distribution of LOG TransactionDT', fontsize=14)
ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])

plt.show()


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(18,4))

time_val = train_transaction.loc[train_transaction['isFraud'] == 1]['TransactionDT'].values

sns.distplot(np.log(time_val), ax=ax[0], color='r')
ax[0].set_title('Distribution of LOG TransactionDT, isFraud=1', fontsize=14)
ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])

time_val = train_transaction.loc[train_transaction['isFraud'] == 0]['TransactionDT'].values

sns.distplot(np.log(time_val), ax=ax[1], color='b')
ax[1].set_title('Distribution of LOG TransactionDT, isFraud=0', fontsize=14)
ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])


plt.show()


# In[ ]:


train_transaction['TransactionDT'].plot(kind='hist',
                                        figsize=(15, 5),
                                        label='train',
                                        bins=50,
                                        title='Train vs Test TransactionDT distribution')
test_transaction['TransactionDT'].plot(kind='hist',
                                       label='test',
                                       bins=50)
plt.legend()
plt.show()


# As you can see it seems that train and test transaction dates don't overlap, so it would be prudent to use time-based split for validation. Rob discovered this here: https://www.kaggle.com/robikscube/ieee-fraud-detection-first-look-and-eda.
# 
# Also we can see the **30 days** gap between train and test.
# 

# In[ ]:


train_transaction.head()


# Also you should read this post by Rob [Plotting features over time shows something.... interesting
# ](https://www.kaggle.com/c/ieee-fraud-detection/discussion/100167#latest-577688) he discovered a weird correlation between C and D features, and that's why I do the following plots :)

# ### isFraud vs time

# In[ ]:


i = 'isFraud'
cor = np.corrcoef(train_transaction['TransactionDT'], train_transaction[i])[0,1]
train_transaction.loc[train_transaction['isFraud'] == 0].set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3), label="isFraud=0")
train_transaction.loc[train_transaction['isFraud'] == 1].set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3), label="isFraud=1")
#test_transaction.set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3))
plt.legend()
plt.show()


# ### C features: C1, C2 ... C14

# In[ ]:


c_features = list(train_transaction.columns[16:30])
for i in c_features:
    cor = np.corrcoef(train_transaction['TransactionDT'], train_transaction[i])[0,1]
    train_transaction.set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3))
    test_transaction.set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3))
    plt.show()


# In[ ]:


c_features = list(train_transaction.columns[16:30])
for i in c_features:
    cor = np.corrcoef(train_transaction['TransactionDT'], train_transaction[i])[0,1]
    train_transaction.set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3))
    test_transaction.set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3))
    plt.show()


# In[ ]:


del c_features
gc.collect()


# ### D features: D1 ... D15

# In[ ]:


d_features = list(train_transaction.columns[30:45])

for i in d_features:
    cor = np.corrcoef(train_transaction['TransactionDT'], train_transaction[i])[0,1]
    train_transaction.set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3))
    test_transaction.set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3))
    plt.show()


# OK, the problem here is that ```D``` features are mostly NaNs!

# In[ ]:


train_transaction[d_features].head()


# In[ ]:


# Click output to see the number of missing values in each column
missing_values_count = train_transaction[d_features].isnull().sum()
missing_values_count


# In[ ]:


# how many total missing values do we have?
total_cells = np.product(train_transaction[d_features].shape)
total_missing = missing_values_count.sum()
# percent of data that is missing
(total_missing/total_cells) * 100


# If we consider D features, de 58.15% are missing values ... Let's plot without missing values

# In[ ]:


for i in d_features:
    cor_tr = np.corrcoef(train_transaction['TransactionDT'], train_transaction[i].fillna(-1))[0,1]
    cor_te = np.corrcoef(test_transaction['TransactionDT'], test_transaction[i].fillna(-1))[0,1]
    train_transaction.set_index('TransactionDT')[i].fillna(-1).plot(style='.', title=i+" corr_tr= "+str(round(cor_tr,3))+" || corr_te= "+str(round(cor_te,3)), figsize=(15, 3))
    test_transaction.set_index('TransactionDT')[i].fillna(-1).plot(style='.', title=i+" corr_tr= "+str(round(cor_tr,3))+"  || corr_te= "+str(round(cor_te,3)), figsize=(15, 3))
    plt.show()


# In[ ]:


del d_features, cor
gc.collect()


# ### M features: M1 .. M9

# In[ ]:


m_features = list(train_transaction.columns[45:54])
train_transaction[m_features].head()


# In[ ]:


del m_features
gc.collect()


# ## V150

# In[ ]:


i = "V150"
cor_tr = np.corrcoef(train_transaction['TransactionDT'], train_transaction[i].fillna(-1))[0,1]
cor_te = np.corrcoef(test_transaction['TransactionDT'], test_transaction[i].fillna(-1))[0,1]
train_transaction.set_index('TransactionDT')[i].fillna(-1).plot(style='.', title=i+" corr_tr= "+str(round(cor_tr,3))+" || corr_te= "+str(round(cor_te,3)), figsize=(15, 3))
test_transaction.set_index('TransactionDT')[i].fillna(-1).plot(style='.', title=i+" corr_tr= "+str(round(cor_tr,3))+"  || corr_te= "+str(round(cor_te,3)), figsize=(15, 3))
plt.show()


# <br>
# # Groups

# Remove ```.head(20)``` and check the entire list.

# In[ ]:


train_transaction.loc[:,train_transaction.columns[train_transaction.columns.str.startswith('V')]].isnull().sum().head(20)


# <br>
# # TransactionAmt

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(18,4))

time_val = train_transaction['TransactionAmt'].values

sns.distplot(time_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of TransactionAmt', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])

sns.distplot(np.log(time_val), ax=ax[1], color='b')
ax[1].set_title('Distribution of LOG TransactionAmt', fontsize=14)
ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])

plt.show()


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(18,4))

time_val = train_transaction.loc[train_transaction['isFraud'] == 1]['TransactionAmt'].values

sns.distplot(np.log(time_val), ax=ax[0], color='r')
ax[0].set_title('Distribution of LOG TransactionAmt, isFraud=1', fontsize=14)
ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])

time_val = train_transaction.loc[train_transaction['isFraud'] == 0]['TransactionAmt'].values

sns.distplot(np.log(time_val), ax=ax[1], color='b')
ax[1].set_title('Distribution of LOG TransactionAmt, isFraud=0', fontsize=14)
ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])


plt.show()


# In[ ]:


del time_val


# In[ ]:


tmp = train_transaction[['TransactionAmt', 'isFraud']][0:100000]
plot_distribution(tmp[(tmp['TransactionAmt'] <= 800)], 'TransactionAmt', 'Transaction Amount Distribution', bins=10.0,)
del tmp


# # Unique Values

# ### D Features

# In[ ]:


plt.figure(figsize=(10, 7))
d_features = list(train_transaction.columns[30:45])
uniques = [len(train_transaction[col].unique()) for col in d_features]
sns.set(font_scale=1.2)
ax = sns.barplot(d_features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature TRAIN')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 


# In[ ]:


plt.figure(figsize=(10, 7))
d_features = list(test_transaction.columns[30:45])
uniques = [len(test_transaction[col].unique()) for col in d_features]
sns.set(font_scale=1.2)
ax = sns.barplot(d_features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature TEST')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 


# ### C features

# In[ ]:


plt.figure(figsize=(10, 7))
c_features = list(train_transaction.columns[16:30])
uniques = [len(train_transaction[col].unique()) for col in c_features]
sns.set(font_scale=1.2)
ax = sns.barplot(c_features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature TRAIN')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 


# In[ ]:


plt.figure(figsize=(10, 7))
c_features = list(test_transaction.columns[16:30])
uniques = [len(test_transaction[col].unique()) for col in c_features]
sns.set(font_scale=1.2)
ax = sns.barplot(c_features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature TEST')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 


# ### V features

# In[ ]:


plt.figure(figsize=(35, 8))
v_features = list(train_transaction.columns[54:120])
uniques = [len(train_transaction[col].unique()) for col in v_features]
sns.set(font_scale=1.2)
ax = sns.barplot(v_features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 


# In[ ]:


plt.figure(figsize=(35, 8))
v_features = list(train_transaction.columns[120:170])
uniques = [len(train_transaction[col].unique()) for col in v_features]
sns.set(font_scale=1.2)
ax = sns.barplot(v_features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 


# In[ ]:


plt.figure(figsize=(35, 8))
v_features = list(train_transaction.columns[170:220])
uniques = [len(train_transaction[col].unique()) for col in v_features]
sns.set(font_scale=1.2)
ax = sns.barplot(v_features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 


# In[ ]:


plt.figure(figsize=(35, 8))
v_features = list(train_transaction.columns[220:270])
uniques = [len(train_transaction[col].unique()) for col in v_features]
sns.set(font_scale=1.2)
ax = sns.barplot(v_features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 


# In[ ]:


plt.figure(figsize=(35, 8))
v_features = list(train_transaction.columns[270:320])
uniques = [len(train_transaction[col].unique()) for col in v_features]
sns.set(font_scale=1.2)
ax = sns.barplot(v_features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 


# In[ ]:


plt.figure(figsize=(38, 8))
v_features = list(train_transaction.columns[320:390])
uniques = [len(train_transaction[col].unique()) for col in v_features]
sns.set(font_scale=1.2)
ax = sns.barplot(v_features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 


# ### id_code

# In[ ]:


train_identity.head(2)


# In[ ]:


plt.figure(figsize=(35, 8))
features = list(train_identity.columns[0:38])
uniques = [len(train_identity[col].unique()) for col in features]
sns.set(font_scale=1.2)
ax = sns.barplot(features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature TRAIN')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 


# In[ ]:


plt.figure(figsize=(35, 8))
features = list(test_identity.columns[0:38])
uniques = [len(test_identity[col].unique()) for col in features]
sns.set(font_scale=1.2)
ax = sns.barplot(features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature TEST')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 


# <br>
# # Categorical Features
# 
# - ProductCD
# - emaildomain
# - card1 - card6
# - addr1, addr2
# - P_emaildomain
# - R_emaildomain
# - M1 - M9
# - DeviceType
# - DeviceInfo
# - id_12 - id_38

# In[ ]:


train_transaction.head(3)


# In[ ]:


train_identity.head(3)


# ### ProductCD

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(20,5))

sns.countplot(x="ProductCD", ax=ax[0], hue = "isFraud", data=train_transaction)
ax[0].set_title('ProductCD train', fontsize=14)
sns.countplot(x="ProductCD", ax=ax[1], data=test_transaction)
ax[1].set_title('ProductCD test', fontsize=14)
plt.show()


# ### Device Type & Device Info

# In[ ]:


ax = sns.countplot(x="DeviceType", data=train_identity)
ax.set_title('DeviceType', fontsize=14)
plt.show()


# **Device information**

# In[ ]:


print ("Unique Devices = ",train_identity['DeviceInfo'].nunique())
train_identity['DeviceInfo'].value_counts().head()


# ### Card

# In[ ]:


cards = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
for i in cards:
    print ("Unique ",i, " = ",train_transaction[i].nunique())


# In[ ]:


fig, ax = plt.subplots(1, 4, figsize=(25,5))

sns.countplot(x="card4", ax=ax[0], data=train_transaction.loc[train_transaction['isFraud'] == 0])
ax[0].set_title('card4 isFraud=0', fontsize=14)
sns.countplot(x="card4", ax=ax[1], data=train_transaction.loc[train_transaction['isFraud'] == 1])
ax[1].set_title('card4 isFraud=1', fontsize=14)
sns.countplot(x="card6", ax=ax[2], data=train_transaction.loc[train_transaction['isFraud'] == 0])
ax[2].set_title('card6 isFraud=0', fontsize=14)
sns.countplot(x="card6", ax=ax[3], data=train_transaction.loc[train_transaction['isFraud'] == 1])
ax[3].set_title('card6 isFraud=1', fontsize=14)
plt.show()


# In[ ]:


cards = train_transaction.iloc[:,4:7].columns

plt.figure(figsize=(18,8*4))
gs = gridspec.GridSpec(8, 4)
for i, cn in enumerate(cards):
    ax = plt.subplot(gs[i])
    sns.distplot(train_transaction.loc[train_transaction['isFraud'] == 1][cn], bins=50)
    sns.distplot(train_transaction.loc[train_transaction['isFraud'] == 0][cn], bins=50)
    ax.set_xlabel('')
    ax.set_title('feature: ' + str(cn))
plt.show()


# As you can see, ``` Card 1``` column is given as Categorical but it is behaving like Continuous Data. Having '13553' unique Values.
# 
# > **From organizer: ** This is a encoded categorical variable. 
# The dataset contains many high-cardinality variables, and it's challenge to model such variable. Meanwhile, it's worthy to see how you talented people deal with them.
# 
# Check this post: https://www.kaggle.com/c/ieee-fraud-detection/discussion/100340#latest-578626

# ### Email Domain

# In[ ]:


"emaildomain" in train_transaction.columns, "emaildomain" in train_identity.columns


# In[ ]:


fig, ax = plt.subplots(1, 3, figsize=(32,10))

sns.countplot(y="P_emaildomain", ax=ax[0], data=train_transaction)
ax[0].set_title('P_emaildomain', fontsize=14)
sns.countplot(y="P_emaildomain", ax=ax[1], data=train_transaction.loc[train_transaction['isFraud'] == 1])
ax[1].set_title('P_emaildomain isFraud = 1', fontsize=14)
sns.countplot(y="P_emaildomain", ax=ax[2], data=train_transaction.loc[train_transaction['isFraud'] == 0])
ax[2].set_title('P_emaildomain isFraud = 0', fontsize=14)
plt.show()


# In[ ]:


fig, ax = plt.subplots(1, 3, figsize=(32,10))

sns.countplot(y="R_emaildomain", ax=ax[0], data=train_transaction)
ax[0].set_title('R_emaildomain', fontsize=14)
sns.countplot(y="R_emaildomain", ax=ax[1], data=train_transaction.loc[train_transaction['isFraud'] == 1])
ax[1].set_title('R_emaildomain isFraud = 1', fontsize=14)
sns.countplot(y="R_emaildomain", ax=ax[2], data=train_transaction.loc[train_transaction['isFraud'] == 0])
ax[2].set_title('R_emaildomain isFraud = 0', fontsize=14)
plt.show()


# It seems that criminals prefer gmail

# # Memory reduction

# **Merge transaction & identity + Label Encoder**

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train = train_transaction.merge(train_identity, how=\'left\', left_index=True, right_index=True)\ny_train = train[\'isFraud\'].astype("uint8").copy()\nprint("Tain: ",train.shape)\ndel train_transaction, train_identity\n\ntest = test_transaction.merge(test_identity, how=\'left\', left_index=True, right_index=True)\nprint("Test: ",test.shape)\ndel test_transaction, test_identity\nprint ("Merged!")')


# In[ ]:


get_ipython().run_cell_magic('time', '', "X_train = train.drop('isFraud', axis=1)\nX_test = test.copy()\n\ndel train, test\ngc.collect()\n\n# Label Encoding\nfor f in X_train.columns:\n    if X_train[f].dtype=='object' or X_test[f].dtype=='object': \n        lbl = preprocessing.LabelEncoder()\n        lbl.fit(list(X_train[f].values) + list(X_test[f].values))\n        X_train[f] = lbl.transform(list(X_train[f].values))\n        X_test[f] = lbl.transform(list(X_test[f].values))  ")


# ### Reduce Memory Usage
# > 2 options
# 
# **Note** Using te option1 the missing values are encoded as -1, you have to update the XGBoost model and set ```missing=-1```

# In[ ]:


def reduce_mem_usage1(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist


# In[ ]:


get_ipython().run_cell_magic('time', '', '# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage\n# WARNING! THIS CAN DAMAGE THE DATA \ndef reduce_mem_usage2(df):\n    """ iterate through all the columns of a dataframe and modify the data type\n        to reduce memory usage.        \n    """\n    start_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage of dataframe is {:.2f} MB\'.format(start_mem))\n    \n    for col in df.columns:\n        col_type = df[col].dtype\n        \n        if col_type != object:\n            c_min = df[col].min()\n            c_max = df[col].max()\n            if str(col_type)[:3] == \'int\':\n                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:\n                    df[col] = df[col].astype(np.int8)\n                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:\n                    df[col] = df[col].astype(np.int16)\n                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:\n                    df[col] = df[col].astype(np.int32)\n                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:\n                    df[col] = df[col].astype(np.int64)  \n            else:\n                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:\n                    df[col] = df[col].astype(np.float16)\n                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:\n                    df[col] = df[col].astype(np.float32)\n                else:\n                    df[col] = df[col].astype(np.float64)\n        else:\n            df[col] = df[col].astype(\'category\')\n\n    end_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage after optimization is: {:.2f} MB\'.format(end_mem))\n    print(\'Decreased by {:.1f}%\'.format(100 * (start_mem - end_mem) / start_mem))\n    \n    return df')


# this takes 6-7 mins. You can click and check the ``` output ```

# In[ ]:


# %%time
# props_train, NAlist_train = reduce_mem_usage1(X_train)
# props_test, NAlist_test = reduce_mem_usage1(X_test)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'X_train = reduce_mem_usage2(X_train)\nX_test = reduce_mem_usage2(X_test)')


# ### Now memory should be around 4 GB !

# In[ ]:


X_train.head(3)


# In[ ]:


X_test.head(3)


# In[ ]:


logging.debug("memory usage!")


# **Drop some columns**
# > from: https://www.kaggle.com/jazivxt/safe-box/notebook

# In[ ]:


#drop_col = ['TransactionDT', 'V300', 'V309', 'V111', 'C3', 'V124', 'V106', 'V125', 'V315', 'V134', 'V102', 'V123', 'V316', 'V113', 'V136', 'V305', 'V110', 'V299', 'V289', 'V286', 'V318', 'V103', 'V304', 'V116', 'V298', 'V284', 'V293', 'V137', 'V295', 'V301', 'V104', 'V311', 'V115', 'V109', 'V119', 'V321', 'V114', 'V133', 'V122', 'V319', 'V105', 'V112', 'V118', 'V117', 'V121', 'V108', 'V135', 'V320', 'V303', 'V297', 'V120']
drop_col = ['TransactionDT']
X_train.drop(drop_col,axis=1, inplace=True)
X_test.drop(drop_col, axis=1, inplace=True)
X_train.head()


# **Fill NaN**

# In[ ]:


X_train.fillna(-1,inplace=True)
X_test.fillna(-1,inplace=True)
X_train.head()


# # PCA

# **PCA 2 components**

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_scaled = scaler.fit_transform(X_train)         
PCA_train_x = PCA(2).fit_transform(train_scaled)
plt.scatter(PCA_train_x[:, 0], PCA_train_x[:, 1], c=y_train, cmap="copper_r")
plt.axis('off')
plt.colorbar()
plt.show()


# In[ ]:


from sklearn.decomposition import KernelPCA

lin_pca = KernelPCA(n_components = 2, kernel="linear", fit_inverse_transform=True)
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
sig_pca = KernelPCA(n_components = 2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)


plt.figure(figsize=(11, 4))
for subplot, pca, title in ((131, lin_pca, "Linear kernel"), (132, rbf_pca, "RBF kernel, $\gamma=0.04$"), 
                            (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):
       
    PCA_train_x = PCA(2).fit_transform(train_scaled)
    plt.subplot(subplot)
    plt.title(title, fontsize=14)
    plt.scatter(PCA_train_x[:, 0], PCA_train_x[:, 1], c=y_train, cmap="nipy_spectral_r")
    plt.xlabel("$z_1$", fontsize=18)
    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)

plt.show()


# In[ ]:


del train_scaled,PCA_train_x,scaler, lin_pca,rbf_pca, sig_pca
gc.collect()


# <br>
# # Models
# ---
# 

# ## XGBoost Model + FE Importance
# 
# > This part is from [can_we_beat_it](https://www.kaggle.com/konradb/can-we-beat-it) by Konrad
# 
# > Also check this kernel [IEEE Fraud Simple Baseline [0.9383 LB]](https://www.kaggle.com/artkulak/ieee-fraud-simple-baseline-0-9383-lb)

# In[ ]:


xgb.XGBClassifier(
        n_estimators=500,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma = 0.1,
        alpha = 4,
        missing = -1,
        tree_method='gpu_hist'
)


# **Important** Check the [XGB official documentation](https://xgboost.readthedocs.io/en/latest/parameter.html) in order to know more about the parameters.
# 
# Also check this thread [CV vs Public LB](https://www.kaggle.com/c/ieee-fraud-detection/discussion/100255#latest-578503)

# In[ ]:


get_ipython().run_cell_magic('time', '', 'NFOLDS = 5\nkf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=123)\n\ny_preds = np.zeros(X_test.shape[0])\ny_oof = np.zeros(X_train.shape[0])\nscore = 0\n  \nfor fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):\n    clf = xgb.XGBClassifier(\n        n_estimators=500,\n        max_depth=9,\n        learning_rate=0.05,\n        subsample=0.9,\n        colsample_bytree=0.9,\n        gamma = 0.2,\n        alpha = 4,\n        missing = -1,\n        tree_method=\'gpu_hist\'\n    )\n    \n    X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]\n    y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]\n    clf.fit(X_tr, y_tr)\n    y_pred_train = clf.predict_proba(X_vl)[:,1]\n    y_oof[val_idx] = y_pred_train\n    print("FOLD: ",fold,\' AUC {}\'.format(roc_auc_score(y_vl, y_pred_train)))\n    score += roc_auc_score(y_vl, y_pred_train) / NFOLDS\n    y_preds+= clf.predict_proba(X_test)[:,1] / NFOLDS\n    \n    del X_tr, X_vl, y_tr, y_vl\n    gc.collect()\n    \n    \nprint("\\nMEAN AUC = {}".format(score))\nprint("OOF AUC = {}".format(roc_auc_score(y_train, y_oof)))')


# ### Importance PLOT
# > last FOLD

# In[ ]:


# Get xgBoost importances
importance_dict = {}
for import_type in ['weight', 'gain', 'cover']:
    importance_dict['xgBoost-'+import_type] = clf.get_booster().get_score(importance_type=import_type)
    
# MinMax scale all importances
importance_df = pd.DataFrame(importance_dict).fillna(0)
importance_df = pd.DataFrame(
    preprocessing.MinMaxScaler().fit_transform(importance_df),
    columns=importance_df.columns,
    index=importance_df.index
)

# Create mean column
importance_df['mean'] = importance_df.mean(axis=1)

# Plot the feature importances
importance_df.sort_values('mean').head(40).plot(kind='bar', figsize=(30, 7))


# In[ ]:


del clf, importance_df
gc.collect()


# # Submission

# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')
sub['isFraud'] = y_preds
sub.to_csv('xgboost.csv')
sub.head()


# In[ ]:


sub.loc[ sub['isFraud']>0.99 , 'isFraud'] = 1
b = plt.hist(sub['isFraud'], bins=50)


# In[ ]:


print ("Predicted {} frauds".format(int(sub[sub['isFraud']==1].sum())))


# In[ ]:


del sub, X_train, X_test, importance_df
gc.collect()


# ### To be continued ...
# **I'll keep updating almost every day :)**
