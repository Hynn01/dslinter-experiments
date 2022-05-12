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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from scipy import stats

from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler,LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
import os, glob, math, cv2, gc, logging, warnings, random

from umap import UMAP



from lightgbm import LGBMClassifier as L
import shap
warnings.filterwarnings("ignore")
print("setup completed")


# In[ ]:


train = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv')
test = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/test.csv')
sub = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv')


# In[ ]:


train


# In[ ]:


sub


# In[ ]:


train.describe()


# In[ ]:


train.nunique()


# In[ ]:


correlation=train.corr()
correlation


# In[ ]:


from sklearn.preprocessing import LabelEncoder as LE


# In[ ]:


encoder = LE()
def encode_features(df, cols = ['f_27']):
    for col in cols:
        df[col + '_enc'] = encoder.fit_transform(df[col])
    return df

train = encode_features(train)
test = encode_features(test)


# In[ ]:


train.head()


# In[ ]:


drop=['id', 'target', 'f_27']
features = [feat for feat in train.columns if feat not in drop]
target_feature = 'target'


# In[ ]:


from sklearn.model_selection import train_test_split
test_size_pct = 0.20
X_train, X_valid, y_train, y_valid = train_test_split(train[features], train[target_feature], test_size = test_size_pct, random_state = 42)


# In[ ]:


model=L()
model.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], eval_metric = ['auc'], early_stopping_rounds = 256, verbose = 250)


# In[ ]:


from sklearn.metrics import roc_auc_score
val_preds = model.predict_proba(X_valid[features])[:, 1]
roc_auc_score(y_valid, val_preds)


# In[ ]:


from sklearn.metrics import roc_auc_score
preds = model.predict_proba(test[features])[:, 1]


# In[ ]:


sub['target'] = preds
sub.to_csv('my_submission_043022.csv', index = False)
sub.head()


# In[ ]:




