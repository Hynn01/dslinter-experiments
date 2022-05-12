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


# import sys
# !cp ../input/rapids/rapids.21.06 /opt/conda/envs/rapids.tar.gz
# !cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null
# sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path
# sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path
# sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 
# !cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/


# In[ ]:


# import cuml


# In[ ]:


from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from cuml.manifold import TSNE
import seaborn as sns
from  sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-nov-2021/train.csv', skiprows=lambda x:x%5!=0)


# In[ ]:


test = pd.read_csv('../input/tabular-playground-series-nov-2021/test.csv')


# In[ ]:


memory_train = sum(train.memory_usage()) / 1e6
print(f'[INFO] Memory usage train_before: {memory_train:.2f} MB.')

memory_test = sum(test.memory_usage()) / 1e6
print(f'[INFO] Memory usage test_before: {memory_test:.2f} MB.\n')

# Downcasting the traind dataset.
for col in train.columns:
    
    if train[col].dtype == "float64":
        train[col] = pd.to_numeric(train[col], downcast="float")
        
    if train[col].dtype == "int64":
        train[col] = pd.to_numeric(train[col], downcast="integer")
        
# Downcasting the test dataset.
for col in test.columns:
    
    if test[col].dtype == "float64":
        test[col] = pd.to_numeric(test[col], downcast="float")
        
    if test[col].dtype == "int64":
        test[col] = pd.to_numeric(test[col], downcast="integer")
        
memory_train = sum(train.memory_usage()) / 1e6
print(f'[INFO] Memory usage train: {memory_train:.2f} MB.')

memory_test = sum(test.memory_usage()) / 1e6
print(f'[INFO] Memory usage test: {memory_test:.2f} MB.')


# In[ ]:


sub = pd.read_csv('../input/tabular-playground-series-nov-2021/sample_submission.csv')


# In[ ]:





# In[ ]:


train.shape


# In[ ]:


train.isna().sum().unique()


# In[ ]:


train.target.unique()


# In[ ]:


import seaborn as sns
sns.countplot(train['target'])
plt.title('Count of each class')
plt.show()


# In[ ]:


X=train.drop(['id','target'], axis=1)
y=train['target']
train.iloc[:,1:].shape


# In[ ]:


train.iloc[:,1:-1].columns


# In[ ]:


fig, axes = plt.subplots(10,10, figsize=(20, 12))
type(axes)
axes = axes.flatten()
type(axes)
for idx, ax in enumerate(axes):
    sns.kdeplot(data=train.iloc[:,1:],hue='target' ,ax=ax, fill=True,
        x=f'f{idx}', palette=['#4DB6AC', 'red'], legend=idx==0
    )
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel('')
    ax.set_ylabel(''); ax.spines['left'].set_visible(False)
    ax.set_title(f'f{idx}', loc='right' , fontsize=11)
plt.show()
    


# In[ ]:


candles_features = [
    'f0','f2','f4','f9','f12','f16','f19','f20','f23','f24','f27',
    'f28','f30','f31','f32','f33','f35','f39','f42','f44','f46','f48',
    'f49','f51','f52','f53','f56','f58','f59','f60','f61','f62','f63',
    'f64','f68','f69','f72','f73','f75','f76','f78','f79','f81','f83',
    'f84','f87','f88','f89','f90','f92','f93','f94','f95','f98','f99'
]


# In[ ]:


train['f1'].plot.kde()


# In[ ]:


train['f2'].plot.kde()


# In[ ]:


sns.boxplot(data=train['f2'])


# In[ ]:


# train.describe().T.sort_values('std' , ascending=False)


# In[ ]:


train[candles_features].describe().T


# In[ ]:



df_candles_log_transform = train[candles_features]

mask_neg = (df_candles_log_transform < 0)
mask_pos = (df_candles_log_transform > 0)

df_candles_log_transform[mask_neg] = np.log(np.abs(df_candles_log_transform)) * (-1)
df_candles_log_transform[mask_pos] = np.log(df_candles_log_transform)


# In[ ]:


df_candles_log_transform.describe().T


# In[ ]:


train[candles_features]=df_candles_log_transform[candles_features]


# In[ ]:


train.describe().T.sort_values('std' , ascending=False)


# In[ ]:


# plt.figure(figsize = (20, 12))
# sns.heatmap(train.corr()) 


# In[ ]:


# https://www.kaggle.com/code/sergiosaharovskiy/tps-nov-2021-a-complete-guide?scriptVersionId=80238862&cellId=24


# In[ ]:


X=train.drop(['id','target'], axis=1)
y=train['target']


# In[ ]:


x_test=test.drop(['id'], axis=1)


# In[ ]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.astype(dtype=int) # TabNet can save model only with int64.


# In[ ]:


x_test = scaler.fit_transform(x_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True,test_size=0.3, random_state = 1)
logreg=LogisticRegression(solver='liblinear')
cv_results=cross_val_score(logreg, X_train,y_train ,cv=10, scoring='roc_auc' )


# In[ ]:


logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)


# In[ ]:


fpr, tpr, _ = roc_curve(y_test, y_pred)
score = auc(fpr, tpr)
print(score)


# In[ ]:


cv_results


# In[ ]:


# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
# dTree=DecisionTreeClassifier()
# cv_results=cross_val_score(dTree, X_train,y_train , scoring='roc_auc' )
# print('mean of 5 cross validation scores:',np.mean(cv_results))


# In[ ]:



# from sklearn.pipeline import Pipeline
# from  sklearn.ensemble import RandomForestClassifier


# In[ ]:


# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
# rnf=RandomForestClassifier()
# rnf_results=cross_val_score(rnf, X_train,y_train,cv=2, scoring='roc_auc')
# rnf_results


# In[ ]:





# In[ ]:


# rnf.fit(X_train,y_train)
# rnf.feature_importances_
# feat_importances = pd.Series(rnf.feature_importances_, index=X.columns)

# feat_importances.nlargest(50).plot(kind='barh',figsize=(20,20))
# important_features=feat_importances.nlargest(50).index
# important_features
# rnf.fit(X[important_features],y)
# y_pred=rnf.predict(X_test[important_features])


# In[ ]:


y_pred=logreg.predict(x_test)
sub['target'] = y_pred
sub.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




