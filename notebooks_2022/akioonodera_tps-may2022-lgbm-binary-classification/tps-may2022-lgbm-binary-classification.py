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


import lightgbm as lgb
#import optuna.integration.lightgbm as lgb

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
mmscaler = MinMaxScaler(feature_range=(0, 1), copy=True)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error # 平均絶対誤差
#from sklearn.metrics import mean_squared_error # 平均二乗誤差
#from sklearn.metrics import mean_squared_log_error # 対数平均二乗誤差
from sklearn.metrics import r2_score # 決定係数
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import missingno as msno
import plotly.express as px

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


# Pandas setting to display more dataset rows and columns
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 600)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# # 1. Import data

# In[ ]:


sample_submission = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv")
train = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/train.csv")
test = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/test.csv")


# In[ ]:


sample_submission.head()


# In[ ]:


sample_submission.info()


# In[ ]:


sample_submission.describe()


# In[ ]:


train


# In[ ]:


train.info()


# In[ ]:


train.describe()


# In[ ]:


test


# In[ ]:


test.info()


# In[ ]:


test.describe()


# train、testともに、
# f_27だけがアルファベットが連続する文字列になっています。<br>
# f_07からf_18までは0以上の整数値で、f_29は0または1または2、f_30は0または1の値になっているようです。<br>
# ほかの項目はマイナスの値を含む数値のようです。<br>
# For both train and test, f_27 is a string with consecutive alphabets.<br>
# It seems that f_07 to f_18 are integer values greater than or equal to 0, f_29 is 0 or 1 or 2, and f_30 is 0 or 1.<br>
# Other items seem to be numbers with negative values.

# # 2. EDA

# In[ ]:


# Colors to be used for plots
colors = ["lightcoral", "sandybrown", "darkorange", "mediumseagreen", "lightseagreen",
          "cornflowerblue", "mediumpurple", "palevioletred", "lightskyblue", "sandybrown",
          "yellowgreen", "indianred", "lightsteelblue", "mediumorchid", "deepskyblue"]


# In[ ]:



figure = plt.figure(figsize=(16, 8))
for feat in range(31):
    feat_name = f'f_{feat:02d}'
    if(feat_name != 'f_27'): # f_27を除く
        plt.subplot(8, 4, feat+1)
        plt.hist(train[feat_name], bins=100)
        plt.title(f'{feat_name}')
figure.tight_layout(h_pad=1.0, w_pad=0.8)
plt.show()


# In[ ]:



figure = plt.figure(figsize=(16, 8))
for feat in range(31):
    feat_name = f'f_{feat:02d}'
    if(feat_name != 'f_27'): # f_27を除く
        plt.subplot(8, 4, feat+1)
        plt.hist(test[feat_name], bins=100)
        plt.title(f'{feat_name}')
figure.tight_layout(h_pad=1.0, w_pad=0.8)
plt.show()


# 整数値の項目以外は概ね正規分布しているようです。<br>
# It seems that the items other than the items with integer values are normally distributed.

# In[ ]:


corr = train.corr().round(2)
plt.figure(figsize=(20,10))
sns.heatmap(corr, vmin=-1, vmax=1, center=0, square=False, annot=True, cmap='coolwarm')
plt.show()


# ターゲットと強い相関がある特徴量はないようですが、特徴量同士で相関のあるものはあります。<br>
# It seems that there are no features that are strongly correlated with the target, but there are some that are correlated with each other.

# In[ ]:


corr = test.corr().round(2)
plt.figure(figsize=(20,10))
sns.heatmap(corr, vmin=-1, vmax=1, center=0, square=False, annot=True, cmap='coolwarm')
plt.show()


# 特徴量同士で相関のあるものはあります。<br>
# There are some that have a strong correlation between the features.

# In[ ]:


# Concat train and test
all = pd.concat([train,test],ignore_index=True)


# In[ ]:


all.drop(columns=['id', 'target']).describe().T        .style.bar(subset=['mean'], color=px.colors.qualitative.G10[0])        .background_gradient(subset=['std'], cmap='Greens')        .background_gradient(subset=['50%'], cmap='BuGn')


# # 3. Preprosessing

# f_27の特徴量は10文字のアルファベットです。<br>
# 1文字ずつに分けて、ラベルエンコードします。<br>
# The feature of f_27 is a 10-character alphabet. <br>
# Label-encode it by dividing it into characters.

# In[ ]:


get_ipython().run_cell_magic('time', '', "tmp_all = all\nfor i in range(10):\n    temp = []\n    for j in range(len(all)):\n        temp.append(all['f_27'][j][i])\n    tmp_all['f_27_' + str(i + 1)] = temp\ntmp_all")


# In[ ]:


le = LabelEncoder()
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
le.fit(labels)
for i in range(10):
    all['f_27_' + str(i + 1)] = le.transform(tmp_all['f_27_' + str(i + 1)])

all


# In[ ]:


# Split all for train and test
df_train = all.iloc[train.index[0]:train.index[-1]+1].drop(columns=["f_27"])
df_test = all.iloc[train.index[-1]+1:].drop(columns=["f_27", "target"])


# In[ ]:


df_train


# In[ ]:


df_test


# # 4. Modeling

# In[ ]:


X = df_train.drop(columns=['id', 'target'])
value = df_train['target']


# In[ ]:


get_ipython().run_cell_magic('time', '', "X_train, X_test, t_train, t_test = train_test_split(X, value, test_size=0.2, random_state=0)\n\nlgb_train = lgb.Dataset(X_train, t_train)\nlgb_eval = lgb.Dataset(X_test, t_test, reference=lgb_train)\n\nparams = {\n        'task': 'prediction',\n        'boosting_type': 'gbdt',\n        'objective': 'binary',\n        'metric': 'binary_logloss',\n        'learning_rate': 0.1,\n        'max_depth': 9,\n        'bagging_fraction': 0.8,\n        'feature_fraction': 0.8,\n        'num_iterations': 20000,\n        'verbosity': -1\n}\n\nmodel = lgb.train(\n    params,\n    train_set=lgb_train,\n    valid_sets=lgb_eval,\n    early_stopping_rounds=100,\n    verbose_eval=100\n)")


# # 5．Prediction

# In[ ]:


get_ipython().run_cell_magic('time', '', "X_test = df_test.drop(columns=['id'])\nsample_submission['target'] = model.predict(X_test)\nsample_submission")


# # 6. Make submission file

# In[ ]:


sample_submission.to_csv('submission.csv', index=False)

