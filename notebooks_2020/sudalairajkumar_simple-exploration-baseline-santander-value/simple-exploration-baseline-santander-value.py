#!/usr/bin/env python
# coding: utf-8

# **Competition Objective:**
# 
# In their 3rd Kaggle competition, Santander Group is asking Kagglers to help them identify the value of transactions for each potential customer. This is a first step that Santander needs to nail in order to personalize their services at scale.
# 
# **Objective of the Notebook:**
# 
# The objective of the notebook is to explore the data for this competition.! We will be using python for the same.  

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb

color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 9999


# First let us look at the files given for the competition.

# In[2]:


from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))


# This follows the standard format of train, test and sample submission files.
# 
# Now let us read the train and test file and check the number of rows and columns.

# In[3]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train rows and columns : ", train_df.shape)
print("Test rows and columns : ", test_df.shape)


# So we have 4459 rows in train set and 49342 rows in test set. We also have 4993 columns in total including the target and id column.
# 
# *Observations:*
# 1. Test set is almost 10 times as that of train set. 
# 2. Public LB uses 49% of the test set for evaluation. So may be it is better to give some (if not more) weightage to LB scores.
# 3. Number of columns is more than the number of train rows. So need to be careful with feature selection / engineering

# In[4]:


train_df.head()


# *Observations:*
# 1. The column names are anonymized and so we do not know what they mean
# 2. There are many zero values present in the data
# 3. From this [discussion post](https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/59128), the dataset is a sparse tabular one.
# 
# **Target Variable:**
# 
# Let us first do a scatter plot of the target variable to see if there are any visible outliers. 

# In[5]:


plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df['target'].values))
plt.xlabel('index', fontsize=12)
plt.ylabel('Target', fontsize=12)
plt.title("Target Distribution", fontsize=14)
plt.show()


# Looks like there are not any visible outliers in the data but the range is quite high.
# 
# We can now do a histogram plot of the target variable.

# In[6]:


plt.figure(figsize=(12,8))
sns.distplot(train_df["target"].values, bins=50, kde=False)
plt.xlabel('Target', fontsize=12)
plt.title("Target Histogram", fontsize=14)
plt.show()


# This is a right (Thanks to Wesam for pointing out my mistake) skewed distribution with majority of the data points having low value. Our competition admins are aware of this one and so they have chosen the evaluation metric as RMSLE (Root Mean Squared Logarithmic Error.).  
# 
# So let us do a histogram plot on the log of target variables and recheck again.

# In[7]:


plt.figure(figsize=(12,8))
sns.distplot( np.log1p(train_df["target"].values), bins=50, kde=False)
plt.xlabel('Target', fontsize=12)
plt.title("Log of Target Histogram", fontsize=14)
plt.show()


# This looks much better than the old one. 
# 
# **Missing values:**
# 
# Now let us check if there are missing values in the dataset.

# In[8]:


missing_df = train_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')
missing_df


# There are no missing values in the dataset :)
# 
# ** Data Type of Columns:**
# 
# Now let us also check the data type of the columns.

# In[9]:


dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()


# Majority of the columns are of integer type and the rest are float type. There is only one string column which is nothing but 'ID' column.
# 
# ** Columns with constant values: **
# 
# Generally when we get problems with many columns, there might be few columns with constant value in train set. So we can check that one as well.

# In[10]:


unique_df = train_df.nunique().reset_index()
unique_df.columns = ["col_name", "unique_count"]
constant_df = unique_df[unique_df["unique_count"]==1]
constant_df.shape


# So we have 256 columns with constant values in the train set. Probably it is a good idea to remove them from the training. Just printing out the names below for ease.

# In[11]:


str(constant_df.col_name.tolist())


# ** Correlation of features with target:**
# 
# Now let us find the correlation of the variables with target and plot them. 
# 
# Thanks to @Heads or Tails kernel and Tariq's comment, it might be a good idea to use Spearman correlation inplace of pearson since spearman is computed on ranks and so depicts monotonic relationships while pearson is on true values and depicts linear relationships. 
# 
# There are thousands of variables and so plotting all of them will give us a cluttered plot. So let us take only those variables whose absolute spearman correlation coefficient is more than 0.1 (just to reduce the number of variables) and plot them. 

# In[18]:


from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

labels = []
values = []
for col in train_df.columns:
    if col not in ["ID", "target"]:
        labels.append(col)
        values.append(spearmanr(train_df[col].values, train_df["target"].values)[0])
corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')
 
corr_df = corr_df[(corr_df['corr_values']>0.1) | (corr_df['corr_values']<-0.1)]
ind = np.arange(corr_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,30))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='b')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
plt.show()


# There are quite a few variables with absolute correlation greater than 0.1
# 
# **Correlation Heat Map:**
# 
# Now let us take these variables whose absolute value of correlation with the target is greater than 0.11 (just to reduce the number of features fuether) and do a correlation heat map. 
# 
# This is just done to identify if there are any strong monotonic relationships between these important features. If the values are high, then probably we can choose to keep one of those variables in the model building process.  Please note that we are doing this only for the very few features and feel free to add more features to explore more.   

# In[22]:


cols_to_use = corr_df[(corr_df['corr_values']>0.11) | (corr_df['corr_values']<-0.11)].col_labels.tolist()

temp_df = train_df[cols_to_use]
corrmat = temp_df.corr(method='spearman')
f, ax = plt.subplots(figsize=(20, 20))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=1., square=True, cmap="YlGnBu", annot=True)
plt.title("Important variables correlation map", fontsize=15)
plt.show()


# Seems like none of the selected variables have spearman correlation more than 0.7 with each other.  
# 
# The above plots helped us in identifying the important individual variables which are correlated with target. However we generally build many non-linear models in Kaggle competitions. So let us build some non-linear models and get variable importance from them. 
# 
# In this notebook, we will build two models to get the feature importances - Extra trees and Light GBM. It could also help us to see if the important features coming out from both of them are consistent. Let us first start with ET model.  
# 
# **Feature Importance - Extra trees model**
# 
# Our Evaluation metric for the competition is RMSLE. So let us use log of the target variable to build our models. Also please note that we are removing those variables with constant values (that we identified earlier).  

# In[20]:


### Get the X and y variables for building model ###
train_X = train_df.drop(constant_df.col_name.tolist() + ["ID", "target"], axis=1)
test_X = test_df.drop(constant_df.col_name.tolist() + ["ID"], axis=1)
train_y = np.log1p(train_df["target"].values)


# In[15]:


from sklearn import ensemble
model = ensemble.ExtraTreesRegressor(n_estimators=200, max_depth=20, max_features=0.5, n_jobs=-1, random_state=0)
model.fit(train_X, train_y)

## plot the importances ##
feat_names = train_X.columns.values
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12,12))
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()


# 'f190486d6' seems to be the important variable followed by '58e2e02e6'. 
# 
# ** Feature Importance & Baseline - Light GBM:**
# 
# Now let us build a  Light GBM model to get the feature importance. 
# 
# Apart from feature importance, let us also get predictions on the test set using this model and keep them as baseline predictions.
# 
# Below code is a custom helper function for Light GBM.

# In[16]:


def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 30,
        "learning_rate" : 0.01,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=200, evals_result=evals_result)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result


# Let us do KFold cross validation and average the predictions of the test set.

# In[17]:


kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
pred_test_full = 0
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)
    pred_test_full += pred_test
pred_test_full /= 5.
pred_test_full = np.expm1(pred_test_full)


# So the validation set RMSLE of the folds range from 1.40 to 1.46.
# 
# Let us write the predictions of the model and write it to a file

# In[18]:


# Making a submission file #
sub_df = pd.DataFrame({"ID":test_df["ID"].values})
sub_df["target"] = pred_test_full
sub_df.to_csv("baseline_lgb.csv", index=False)


# This model scored **1.47 RMSLE** on the public LB. We did not do any feature selection (apart from removing the constant variables), feature engineering and parameter tuning. So doing that will further imporve the score. We can use this as our baseline model for any further modeling.
# 
# Now let us look at the feature importance of this model.

# In[19]:


### Feature Importance ###
fig, ax = plt.subplots(figsize=(12,18))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()


# Here again the top two important features are same as that of the Extra trees model. 
# 
# So we could also do some form of feature selection using these feature importances and improve our models further. 
# 
# May be in the next versions, let us look at the top variables from the non-linear models and do some more further analysis to understand tham.! 

# **More to come. Stay tuned.!**
