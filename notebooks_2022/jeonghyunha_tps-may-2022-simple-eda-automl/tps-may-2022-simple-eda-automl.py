#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/33105/logos/header.png?t=2021-12-30-01-26-16&quot)

# # **🔍 Data Exploration**

# In[ ]:


train_df = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv')
train_df.head()


# In[ ]:


test_df = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/test.csv')
test_df.head()


# In[ ]:


print('train data shape:', train_df.shape)
print('test data shape:', test_df.shape)


# In[ ]:


train_df.info()


# In[ ]:


# target distribution
sns.countplot('target', data=train_df, palette=sns.color_palette('coolwarm'))


# In[ ]:


train_df.columns


# In[ ]:


# feature heatmap
cat_cols = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_07',
       'f_08', 'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16',
       'f_17', 'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25',
       'f_26', 'f_28', 'f_29', 'f_30']

plt.figure(figsize=(30,30))
sns.heatmap(train_df[cat_cols].corr(), annot=True)


# In[ ]:


# feature distribution
fig, axs = plt.subplots(5, 6, figsize=(20, 20))
for f, ax in zip(cat_cols, axs.ravel()):
    ax.hist(train_df[f], density=True, bins=100)
    ax.set_title(f'Train {f}, std={train_df[f].std():.1f}')
plt.suptitle('Histograms of the float features')
plt.show()


# In[ ]:


# target-feature distribution
fig, axs = plt.subplots(5, 6, figsize=(20, 20))
for f, ax in zip(cat_cols, axs.ravel()):
    temp = pd.DataFrame({f: train_df[f].values,
                         'state': train_df.target.values})
    temp = temp.sort_values(f)
    temp.reset_index(inplace=True)
    ax.scatter(temp[f], temp.state.rolling(15000, center=True).mean(), s=2)
    ax.set_xlabel(f'{f}')
plt.suptitle('How the target probability depends on single features')
plt.show()


# # **⚙️ AutoML**

# In[ ]:


get_ipython().system('pip install -q -U git+https://github.com/mljar/mljar-supervised.git@master')


# In[ ]:


X_train = train_df.drop(['id','f_27','target'], axis=1)
y_train = train_df['target']


# In[ ]:


from supervised.automl import AutoML

automl = AutoML(total_time_limit = 60*15,
                model_time_limit = 300,
                mode='Compete',
                eval_metric='accuracy',
                algorithms=['Xgboost', 'LightGBM', 'Neural Network'],
                ml_task='binary_classification',
                train_ensemble=True)
automl.fit(X_train, y_train)


# In[ ]:


automl.get_leaderboard()


# In[ ]:


submission = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv')
submission.target = automl.predict(test_df)
submission.to_csv('submission_automl.csv')


# In[ ]:


submission.head()

