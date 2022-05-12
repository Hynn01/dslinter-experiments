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
import plotly.express as px


# In[ ]:


train=pd.read_csv('/kaggle/input/data-science-london-scikit-learn/train.csv')
y=pd.read_csv('/kaggle/input/data-science-london-scikit-learn/trainLabels.csv')
test=pd.read_csv('/kaggle/input/data-science-london-scikit-learn/test.csv')
train


# # Information about Data

# In[ ]:


train.info()


# # Number of Unique Values in each columns

# In[ ]:


train.nunique()


# # Consider Inf Value as NAN for entire notebook

# In[ ]:


pd.set_option('mode.use_inf_as_na', True)


# # Columns having Null Values

# In[ ]:


null_train=train.columns[train.isnull().any()]
null_train


# In[ ]:


null_test=test.columns[test.isnull().any()]
null_test


# So, there are NO-NAN values...GOOD THING !

# # Plotting Histogram to see distribution of features

# In[ ]:


import plotly.express as px
for col in train.columns: 
    fig = px.histogram(train, x=train[col],marginal="rug")
    fig.show()


# All Features are Normally Distributed...GOOD THING !

# In[ ]:


import plotly.express as px
fig = px.histogram(y, x="1",marginal="rug",facet_col='1')
fig.show()


# Even the Target is  EVENLY Distrubuted

# # ScatterPlot
# 
# Plots the relation between EACH Pair

# In[ ]:


import plotly.express as px

for col in train.columns:
    fig = px.scatter(train, x=train[col], y=y['1'])
    fig.show()


# ALL Features are Positively Correlated...

# # DistPlot to confirm Distribution of each Features

# In[ ]:


import plotly.figure_factory as ff

fig = ff.create_distplot([train[c] for c in train.columns], train.columns, bin_size=0.5,curve_type='normal')
fig.show()


# all Features are NORMALLY DISTRBUTED...confirmed,,,Good Thing !

# # Heatmap
# Checks the Correlation among ALL Features

# In[ ]:


import plotly.express as px
fig = px.imshow(train.corr(), text_auto=True,aspect="auto")
fig.show()


# # 2D Histogram
# 
# To see relation between EACH Feature and Target Variable

# In[ ]:


import plotly.express as px

for col in train.columns:
    fig = px.density_heatmap(train, x=train[col],y=y['1'], marginal_x="histogram", marginal_y="histogram")
    fig.show()


# # Boxplot
# Vizualize the Distribution of Features

# In[ ]:


import plotly.express as px

for col in train.columns:
    fig = px.box(train, y=train[col], color=y['1'],points="all")
    fig.show()


# **From the EDAs the following can be observed :-**
# 1. All Features are numeric - no need of Label Encoding
# 2. All Features are Normally Distributed & Unimodal - no need of Log / Box-Cox Transformations
# 3. Boxplot reveals presence of some Minor Outliers - we may be FINE including these, but I choose to get rid of these for sake of accuracy

# # Removing Outliers
# 
# In statistics, an outlier is a data point that differs significantly from other observations. An outlier may be due to variability in the measurement or it may indicate experimental error; the latter are sometimes excluded from the data set. An outlier can cause serious problems in statistical analyses.

# In[ ]:


# Using Isolation Forest
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.3)

out = iso.fit_predict(train)

# select all rows that are not outliers
train[out != -1]
train


# In[ ]:


# out = iso.fit_predict(test)

# # select all rows that are not outliers
# test[out != -1
# test


# # Train Test Split
# 
# It splits the train data into 4 parts, X_train, X_test, y_train, y_test.
# 
# * X_train, y_train first used to train the algorithm.
# * X_test is used in that trained algorithms to predict outcomes.
# * Once we get the outcomes, we compare it with y_test
# 

# In[ ]:


X=train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train
# X_test
# y_train
# y_test


# # Train & Fit XGB Model
# 
# To know more about XGB HyperParameters check ;- https://www.kaggle.com/code/pythonkumar/xgboost-hyperparameters-excellent-plots-acc-91?scriptVersionId=94478268&cellId=70

# In[ ]:


from xgboost import XGBClassifier
model = XGBClassifier(
    booster='gbtree', 
    objective='binary:logistic', 
    eval_metric='logloss',
    n_estimators=1000,
    max_depth=15,
    min_split_loss=0.1,
    learning_rate=0.05,
    reg_alpha=0.5,
    reg_lambda=0.5)

model.fit(X_train, y_train)

model.get_params()


# # Predicting from XGB Model

# In[ ]:


pred=model.predict(X_test)
pred


# # Scoring - Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,pred)

from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
cm


# # Scoring - F1 Score

# In[ ]:


from sklearn.metrics import f1_score
f1_score(y_test, pred)


# # Feature Importance

# In[ ]:


# get importance
importance = model.feature_importances_

# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))


# # Plotting Feature Importance

# In[ ]:


# plot
import matplotlib.pyplot as plt
from xgboost import plot_importance
plot_importance(model)
plt.show()


# # Suggestions:-
# * Kaggle - https://www.kaggle.com/pythonkumar
# * GitHub - https://github.com/KumarPython​
# * Twitter - https://twitter.com/KumarPython
# * LinkedIn - https://www.linkedin.com/in/kumarpython/

# # Submission

# In[ ]:


submission=pd.DataFrame({'Id': [i for i in range (1, len(test)+1)],
                         'Solution':model.predict(test)})
# submission
submission.to_csv('submission.csv', index=False)

