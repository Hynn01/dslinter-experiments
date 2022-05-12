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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/train.csv")
test = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/test.csv")
submission = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


submission.head()


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train['f_27'].nunique()


# In[ ]:


for col in train.columns:
    print(col, train[col].nunique(),'\n')


# In[ ]:


train.columns


# In[ ]:


cols = ['f_07',
       'f_08', 'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16',
       'f_17', 'f_18','f_29', 'f_30', 'target']
for col in cols:
    print(col,'\n', len(train[col].unique()), '\n')


# In[ ]:


cols = ['f_07','f_08', 'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16',
       'f_17', 'f_18','f_29', 'f_30']
for col in cols:
    print(col,'\n', len(test[col].unique()), '\n')


# In[ ]:


train = pd.get_dummies(train, columns = ['f_18','f_29', 'f_30'])
test = pd.get_dummies(test, columns = ['f_18','f_29', 'f_30'])


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score , plot_roc_curve
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import SGDClassifier

from sklearn import metrics
from sklearn.metrics import mean_squared_error
rf = RandomForestClassifier(random_state = 45)
ad = AdaBoostClassifier(base_estimator =rf)
dt = DecisionTreeClassifier()
kn = KNeighborsClassifier()
gnb = GaussianProcessClassifier()
svc = SVC()
mlp = MLPClassifier(max_iter=1000, random_state = 44)
gb = GradientBoostingClassifier()
sgd = SGDClassifier()
dt = DecisionTreeClassifier()
et = ExtraTreeClassifier()
ets = ExtraTreesClassifier()
rf = RandomForestClassifier()
knn = KNeighborsClassifier()
rn = RadiusNeighborsClassifier()
import xgboost as xg
from xgboost import XGBClassifier
xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
              gamma=0, gpu_id=-1, importance_type=None,
              interaction_constraints='', learning_rate=0.1300000012,
              max_delta_step=0, max_depth=6, min_child_weight=1,
              monotone_constraints='()', n_estimators=1000, n_jobs=4,
              num_parallel_tree=1, predictor='auto', random_state=45,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
import lightgbm as lgb
lgbm = lgb.LGBMClassifier()
from catboost import CatBoostClassifier
cat = CatBoostClassifier(
    iterations=1000, 
    learning_rate=0.001, 
    loss_function='CrossEntropy'
)


# In[ ]:


from sklearn.preprocessing import LabelEncoder, StandardScaler
sc = StandardScaler()
le = LabelEncoder()
train['f_27'] = le.fit_transform(train['f_27'])
test['f_27'] = le.fit_transform(test['f_27'])
X = train.drop(['id','target'], axis = 1)
X = sc.fit_transform(X)
y = train['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 44)


# In[ ]:


model = [svc]
for model in model:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    roc = roc_auc_score(y_test, y_pred)
    #curve = plot_roc_curve(model, y_test, y_pred)
    #scores = cross_val_score(model, X_train, y_train, cv=5).mean()
    print(model,'\n', 'ROCAUC:', roc,'\n')


# In[ ]:


get_ipython().system('pip install yellowbrick')
from yellowbrick.classifier import ROCAUC
visualizer = ROCAUC(svc, classes=["0", "1"])
visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show()                       # Finalize and show the figure


# In[ ]:


test = test.drop(['id'], axis = 1)
test = sc.transform(test)

y_pred = svc.predict(test).round(3)
test_prediction = pd.DataFrame(y_pred, columns=['target'])
ID = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/test.csv')
test_id = ID['id']
ID = pd.DataFrame(test_id, columns=['id'])
result = pd.concat([ID,test_prediction], axis=1)
result.to_csv('submission_rf.csv',index =False)
    
    
  


# In[ ]:




