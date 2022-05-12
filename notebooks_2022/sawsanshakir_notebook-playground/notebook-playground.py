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


train_dataset = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')
print(train_dataset)


# In[ ]:


test_dataset = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv')
print(test_dataset)


# In[ ]:


train_dataset.info()


# In[ ]:


train_dataset.isnull().sum()


# In[ ]:


import seaborn as sns
sns.distplot(train_dataset['target'])


# In[ ]:


train_dataset['f_27'].value_counts()


# In[ ]:


test_dataset.info()


# In[ ]:


test_dataset.isnull().sum()


# In[ ]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
train_dataset.f_27 = le.fit_transform(train_dataset.f_27)

print(train_dataset)


# In[ ]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
test_dataset.f_27 = le.fit_transform(test_dataset.f_27)

print(test_dataset)


# In[ ]:


y = train_dataset.target
x = train_dataset.drop(['target', 'id'], axis =1)
x_test = test_dataset.drop(['id'], axis =1)

print(x.shape)
print(y.shape)
print(x_test.shape)


# In[ ]:


from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
x = scaler.fit_transform(x)
print(x)


# In[ ]:


from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
x_test = scaler.fit_transform(x_test)
print(x_test)


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split( x, y, test_size=0.33, random_state=42)


# In[ ]:





# In[ ]:


from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

model= XGBClassifier()
name='XGB'


rs_params = {'n_estimators' : [100,300],
              'max_features' : ["sqrt", "log2"],
              'min_samples_split': [2,5]
              
             }                                                                                                      
    
rs_cv = GridSearchCV( estimator = model, 
                           param_grid = rs_params, 
                           cv = 5, n_jobs=-1)


rs_cv.fit( X_train, y_train)
best_parameters = rs_cv.best_params_  
print(best_parameters)


best_result = rs_cv.best_score_  
print(best_result)      
    
## 
##kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
##cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
##print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# In[ ]:


##  # Make predictions on validation dataset
## from sklearn import metrics
## from sklearn.metrics import roc_curve, auc
##  
## model = XGBClassifier(objective= 'binary:logistic')
## model.fit( X_train, y_train)
## false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, model.predict(X_train))
## print(auc(false_positive_rate, true_positive_rate))


# In[ ]:



# Train on training data-
rs_cv.fit(X_train, y_train)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, model.predict(X_train))
print(auc(false_positive_rate, true_positive_rate))


# In[ ]:


#make predictions
predictions = rs_cv.best_estimator_.predict(X_valid)
print(rs_cv.score(X_valid, y_valid))


# In[ ]:


ytest_pred=rs_cv.best_estimator_.predict(x_test)
print(ytest_pred)


# In[ ]:


probability = logistic_reg.predict_proba(x_test)[:,1]
probability


# In[ ]:


final_result = pd.DataFrame({'id': test_dataset.id, 'target': probability})
print(final_result)


# In[ ]:


final_result.to_csv('prop_prediction_file.csv', index=False)

