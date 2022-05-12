#!/usr/bin/env python
# coding: utf-8

# ![](https://storage.googleapis.com/kaggle-organizations/141/thumbnail.jpg?r=890)
# # Santander Customer Transaction Prediction
# Can you identify who will make a transaction?
# 
# Version6
# - Ensemble : LB 0.899
# - LightGBM : LB 0.898
# - Catboost : LB 0.898 

# In[ ]:


### 패키지 설치 
import pandas as pd #Analysis 
import matplotlib.pyplot as plt #Visulization
import seaborn as sns #Visulization
import numpy as np #Analysis 
from scipy.stats import norm #Analysis 
from sklearn.preprocessing import StandardScaler #Analysis 
from scipy import stats #Analysis 
import warnings 
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
import gc

import os
import string
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

import plotly.graph_objs as go

import time
import random


# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
import gc
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from sklearn import metrics

plt.style.use('seaborn')
sns.set(font_scale=1)
pd.set_option('display.max_columns', 500)


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# In[ ]:


# Taking a look at how many rows and columns the train dataset contains
rows1 = train_df.shape[0]; rows2 = test_df.shape[0]; 
columns1 = train_df.shape[1]; columns2 = test_df.shape[1]
print("The train dataset contains {0} rows and {1} columns".format(rows1, columns1))
print("The test dataset contains {0} rows and {1} columns".format(rows2, columns2))


# There are some check point. 
# - 1. The train and test row are similar.  
# - 2. The column size so many.  

# In[ ]:


train_df.head()


# Wow. All variable name is var_. it means that the variable is identifier !!!. https://www.kaggle.com/c/porto-seguro-safe-driver-prediction porto competition also has identifier variable. This link will help.

# In[ ]:


data = [go.Bar(
            x = train_df["target"].value_counts().index.values,
            y = train_df["target"].value_counts().values,
            text='Distribution of target variable'
    )]

layout = go.Layout(
    title='Target variable distribution'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='basic-bar')


# Target is unbalanced. i'll try upsampling...!

# In[ ]:


#https://www.kaggle.com/ashishpatel26/bird-eye-view-of-two-sigma-nn-approach
def mis_value_graph(data):  
    data = [
    go.Bar(
        x = data.columns,
        y = data.isnull().sum(),
        name = 'Counts of Missing value',
        textfont=dict(size=20),
        marker=dict(
        line=dict(
            color= generate_color(),
            #width= 2,
        ), opacity = 0.45
    )
    ),
    ]
    layout= go.Layout(
        title= '"Total Missing Value By Column"',
        xaxis= dict(title='Columns', ticklen=5, zeroline=False, gridwidth=2),
        yaxis= dict(title='Value Count', ticklen=5, gridwidth=2),
        showlegend=True
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='skin')
    
def generate_color():
    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))
    return color

mis_value_graph(train_df)


# In[ ]:


#https://www.kaggle.com/ashishpatel26/bird-eye-view-of-two-sigma-nn-approach
def mis_value_graph(data):  
    data = [
    go.Bar(
        x = data.columns,
        y = data.isnull().sum(),
        name = 'Counts of Missing value',
        textfont=dict(size=20),
        marker=dict(
        line=dict(
            color= generate_color(),
            #width= 2,
        ), opacity = 0.45
    )
    ),
    ]
    layout= go.Layout(
        title= '"Total Missing Value By Column"',
        xaxis= dict(title='Columns', ticklen=5, zeroline=False, gridwidth=2),
        yaxis= dict(title='Value Count', ticklen=5, gridwidth=2),
        showlegend=True
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='skin')
    
def generate_color():
    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))
    return color

mis_value_graph(test_df)


# Train and Test has no missing value. Very Nice !!!. 

# In[ ]:


train_int = train_df.copy()
del train_int['ID_code']
data = [
    go.Heatmap(
        z= train_int.corr().values,
        x= train_int.columns.values,
        y= train_int.columns.values,
        colorscale='Viridis',
        reversescale = False,
        #text = True ,
        opacity = 1.0 )
]

layout = go.Layout(
    title='Pearson Correlation of Integer-type features',
    xaxis = dict(ticks='', nticks=36),
    yaxis = dict(ticks='' ),
    width = 900, height = 700)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='labelled-heatmap')


# ## LightGBM BaseLine

# In[ ]:


# https://www.kaggle.com/fayzur/customer-transaction-prediction-strong-baseline
# Thanks fayzur. Nice Parameter 
param = {
        'num_leaves': 10,
        'max_bin': 119,
        'min_data_in_leaf': 11,
        'learning_rate': 0.02,
        'min_sum_hessian_in_leaf': 0.00245,
        'bagging_fraction': 1.0, 
        'bagging_freq': 5, 
        'feature_fraction': 0.05,
        'lambda_l1': 4.972,
        'lambda_l2': 2.276,
        'min_gain_to_split': 0.65,
        'max_depth': 14,
        'save_binary': True,
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False,
    }


# In[ ]:


features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
target = train_df['target']


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.metrics import roc_auc_score, roc_curve\nskf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)\noof = np.zeros(len(train_df))\npredictions = np.zeros(len(test_df))\nfeature_importance_df = pd.DataFrame()\n\nstart = time.time()\n\n\nfor fold_, (trn_idx, val_idx) in enumerate(skf.split(train_df.values, target.values)):\n    print("fold n°{}".format(fold_))\n    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])\n    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])\n\n    num_round = 10000\n    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 100)\n    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)\n    \n    fold_importance_df = pd.DataFrame()\n    fold_importance_df["feature"] = features\n    fold_importance_df["importance"] = clf.feature_importance()\n    fold_importance_df["fold"] = fold_ + 1\n    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)\n    \n    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / 5\n\nprint("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))')


# In[ ]:


cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,26))
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('LightGBM Features (averaged over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')


# In[ ]:


##submission
sub_df = pd.DataFrame({"ID_code":test_df["ID_code"].values})
sub_df["target"] = predictions
sub_df.to_csv("lgb_submission.csv", index=False)


# In[ ]:


## Catboost : https://www.kaggle.com/wakamezake/starter-code-catboost-baseline
from catboost import Pool, CatBoostClassifier
model = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC")
kf = KFold(n_splits=5, random_state=42, shuffle=True)

y_valid_pred = 0 * target
y_test_pred = 0

for idx, (train_index, valid_index) in enumerate(kf.split(train_df)):
    y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
    X_train, X_valid = train_df[features].iloc[train_index,:], train_df[features].iloc[valid_index,:]
    _train = Pool(X_train, label=y_train)
    _valid = Pool(X_valid, label=y_valid)
    print( "\nFold ", idx)
    fit_model = model.fit(_train,
                          eval_set=_valid,
                          use_best_model=True,
                          verbose=200
                         )
    pred = fit_model.predict_proba(X_valid)[:,1]
    print( "  auc = ", roc_auc_score(y_valid, pred) )
    y_valid_pred.iloc[valid_index] = pred
    y_test_pred += fit_model.predict_proba(test_df[features])[:,1]
y_test_pred /= 5


# In[ ]:


##submission
sub_df1 = pd.DataFrame({"ID_code":test_df["ID_code"].values})
sub_df1["target"] = y_test_pred
sub_df1.to_csv("cat_submission.csv", index=False)


# In[ ]:


corr_df = pd.merge(sub_df,sub_df1,how='left',on='ID_code')
corr_df.corr()


# In[ ]:


##submission
sub_df2 = pd.DataFrame({"ID_code":test_df["ID_code"].values})
sub_df2["target"] = 0.5*sub_df["target"] + 0.5*sub_df1["target"]
sub_df2.to_csv("lgb_cat_submission.csv", index=False)

