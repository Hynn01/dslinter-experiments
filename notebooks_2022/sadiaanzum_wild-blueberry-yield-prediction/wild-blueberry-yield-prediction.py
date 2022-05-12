#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install dabl')


# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import dabl



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data=pd.read_csv("/kaggle/input/wild-blueberry-yield-prediction/Data in Brief/Data in Brief/WildBlueberryPollinationSimulationData.csv")
data.head(15)


# In[ ]:


data.shape


# In[ ]:


data.drop('Row#', axis='columns', inplace=True)
data.info()


# In[ ]:


# Checking the missing values
data.isnull().sum()


# In[ ]:


#check for duplicated values
data.duplicated()


# In[ ]:


data.duplicated().sum()


# There are no Duplicate Values.

# In[ ]:


data.describe(include="all")


# In[ ]:


data.hist(layout=(5,4), figsize=(20,15), bins=20)
plt.show()


# In[ ]:


#EDA using dabl
dabl.plot(data, target_col="yield")


# In[ ]:


sns.boxplot(data["bumbles"])


# In[ ]:


sns.boxplot(data["honeybee"])


# In[ ]:


plt.figure(figsize=(20,20))
c = data.corr()
plt.figure(figsize=(15,12))
sns.heatmap(c, annot=True, cmap="YlGnBu")
plt.title('Understanding the Correlation between Input Data by a Heatmap', fontsize=15)
plt.show()


# In[ ]:


#splitting into independent and dependent features
X = data.drop(columns=['yield'])
y = data[['yield']]


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


#calculating Inter-quartile Range (IQR)
q1 = X.quantile(0.25)
q3 = X.quantile(0.75)
iqr = q3 -q1
print(iqr)


# In[ ]:


iqr_data = data[~((data < (q1 - 1.5 * iqr)) | (data> (q3 + 1.5 * iqr))).any(axis=1)]
iqr_data.shape


# In[ ]:


iqr_data = iqr_data.reset_index().drop(["index"], axis=1)
iqr_data


# # Feature Selection

# In[ ]:


#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state=0)


# In[ ]:


# Using Pearson Correlation
plt.figure(figsize=(18,12))
cor=X_train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()


# In[ ]:


#With this function, we can select highly correlated features
#The first feature that is correlated with any other feature will be removed

def correlation(dataset, threshold):
    col_corr = set()  #Set of all the names of Correlated Columns
    corr_matrix = dataset.corr()
    for i in range (len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i,j]) > threshold:  
                colname = corr_matrix.columns[i]  #getting the name of the column
                col_corr.add(colname)
    return col_corr


# In[ ]:


corr_features = correlation(X_train, 0.7)
len(set(corr_features))


# In[ ]:


corr_features = {'AverageOfLowerTRange',
 'AverageOfUpperTRange',
 'AverageRainingDays',
 'MaxOfLowerTRange',
 'MinOfLowerTRange',
 'MinOfUpperTRange',
 'honeybee'}
corr_features


# In[ ]:


X_train= X_train.drop(corr_features,axis=1)
X_test= X_test.drop(corr_features,axis=1)


# Analysis on the basis of either Mutual Information Gain or Correlation Regression Values

# In[ ]:


# Mutual Information feature selection 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
 
# feature selection
def select_features_info_based(X_train, y_train, X_test):
	mutual_info = SelectKBest(score_func=mutual_info_regression, k='all')
	# learn relationship from training data
	mutual_info.fit(X_train, y_train)
	# transform train input data
	X_train_fs = mutual_info.transform(X_train)
	# transform test input data
	X_test_fs = mutual_info.transform(X_test)
	return X_train_fs, X_test_fs, mutual_info
 

# Feature Selection
X_train_fs, X_test_fs, fs_info = select_features_info_based(X_train, y_train, X_test)
fs_info
# what are scores for the features
for i in range(len(fs_info.scores_)):
	print('Feature %d: %f' % (i, fs_info.scores_[i]))
# plotting the scores
plt.bar([i for i in range(len(fs_info.scores_))], fs_info.scores_)
plt.show()


# In[ ]:


# Correlation feature selection 

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
 
# Feature Selection
def select_features_corr_based(X_train, y_train, X_test):
	fs = SelectKBest(score_func=f_regression, k='all')
	# learning relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs
 
# feature selection
X_train_fs, X_test_fs, fs_corr = select_features_corr_based(X_train, y_train, X_test)
# what are scores for the features
for i in range(len(fs_corr.scores_)):
	print('Feature %d: %f' % (i, fs_corr.scores_[i]))
# plotting the scores
plt.bar([i for i in range(len(fs_corr.scores_))], fs_corr.scores_)
plt.show()

