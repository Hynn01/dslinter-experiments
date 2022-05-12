#!/usr/bin/env python
# coding: utf-8

# # Signal of Smoking Visualize Importance

# In[ ]:


import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

from contextlib import contextmanager
from time import time
from tqdm import tqdm
import lightgbm as lgbm
import category_encoders as ce

from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


# # Data preparation

# In[ ]:


data0 = pd.read_csv("../input/body-signal-of-smoking/smoking.csv")
data0[0:2].T


# In[ ]:


cols0=data0.columns.tolist()
print(cols0)


# In[ ]:


data0.info()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

def labelencoder(df):
    for c in df.columns:
        if df[c].dtype=='object': 
            df[c] = df[c].fillna('N')
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(df[c].values)
    return df


# In[ ]:


data1=labelencoder(data0)


# # Target setting

# In[ ]:


target=['smoking']
dataY=data1['smoking']
dataX=data1.drop(['smoking','ID'],axis=1)


# In[ ]:


train_df=dataX
df_columns =train_df.columns.tolist()


# In[ ]:


def create_numeric_feature(input_df):
    use_columns = df_columns 
    return input_df[use_columns].copy()


# In[ ]:


from contextlib import contextmanager
from time import time

class Timer:
    def __init__(self, logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None, sep=' '):

        if prefix: format_str = str(prefix) + sep + format_str
        if suffix: format_str = format_str + sep + str(suffix)
        self.format_str = format_str
        self.logger = logger
        self.start = None
        self.end = None

    @property
    def duration(self):
        if self.end is None:
            return 0
        return self.end - self.start

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        out_str = self.format_str.format(self.duration)
        if self.logger:
            self.logger.info(out_str)
        else:
            print(out_str)


# In[ ]:


from tqdm import tqdm

def to_feature(input_df):

    processors = [
        create_numeric_feature,
    ]
    
    out_df = pd.DataFrame()
    
    for func in tqdm(processors, total=len(processors)):
        with Timer(prefix='create' + func.__name__ + ' '):
            _df = func(input_df)

        assert len(_df) == len(input_df), func.__name__
        out_df = pd.concat([out_df, _df], axis=1)
        
    return out_df


# In[ ]:


train_feat_df = to_feature(train_df)
#test_feat_df = to_feature(test_df)


# # Model

# In[ ]:


import lightgbm as lgbm
from sklearn.metrics import mean_squared_error

def fit_lgbm(X, y, cv, 
             params: dict=None, 
             verbose: int=50):

    if params is None:
        params = {}

    models = []
    oof_pred = np.zeros_like(y, dtype=np.float)

    for i, (idx_train, idx_valid) in enumerate(cv): 
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        clf = lgbm.LGBMRegressor(**params)
        
        with Timer(prefix='fit fold={} '.format(i)):
            clf.fit(x_train, y_train, 
                    eval_set=[(x_valid, y_valid)],  
                    early_stopping_rounds=100,
                    verbose=verbose)

        pred_i = clf.predict(x_valid)
        oof_pred[idx_valid] = pred_i
        models.append(clf)
        print(f'Fold {i} RMSLE: {mean_squared_error(y_valid, pred_i) ** .5:.4f}')
        print()

    score = mean_squared_error(y, oof_pred) ** .5
    print('-' * 50)
    print('FINISHED | Whole RMSLE: {:.4f}'.format(score))
    return oof_pred, models


# In[ ]:


params = {
    'objective': 'rmse', 
    'learning_rate': .1,
    'reg_lambda': 1.,
    'reg_alpha': .1,
    'max_depth': 5, 
    'n_estimators': 10000, 
    'colsample_bytree': .5, 
    'min_child_samples': 10,
    'subsample_freq': 3,
    'subsample': .9,
    'importance_type': 'gain', 
    'random_state': 71,
    'num_leaves': 62
}


# In[ ]:


y = dataY
print(y.shape)
print(y[0:3])


# In[ ]:


ydf=pd.DataFrame(y)
ydf


# In[ ]:


from sklearn.model_selection import KFold

i=0
fold = KFold(n_splits=5, shuffle=True, random_state=71)
ydfi=ydf.iloc[:,i]
y=np.array(ydfi)
cv = list(fold.split(train_feat_df, y))
oof, models = fit_lgbm(train_feat_df.values, y, cv, params=params, verbose=500)

fig,ax = plt.subplots(figsize=(6,6))
ax.set_title(target[i],fontsize=20)
ax.set_xlabel('oof',fontsize=12)
ax.set_ylabel('train_y',fontsize=12)
ax.scatter(oof,y)


# # Visualize Importance

# In[ ]:


def visualize_importance(models, feat_train_df):

    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df['feature_importance'] = model.feature_importances_
        _df['column'] = feat_train_df.columns
        _df['fold'] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, _df], 
                                          axis=0, ignore_index=True)

    order = feature_importance_df.groupby('column')        .sum()[['feature_importance']]        .sort_values('feature_importance', ascending=False).index[:50]

    print(order.values)

    fig, ax = plt.subplots(figsize=(8, max(6, len(order) * .25)))
    sns.boxenplot(data=feature_importance_df, 
                  x='feature_importance', 
                  y='column', 
                  order=order, 
                  ax=ax, 
                  palette='viridis', 
                  orient='h')
    
    ax.tick_params(axis='x', rotation=0)
    #ax.set_title('Importance')
    ax.grid()
    fig.tight_layout()
    
    return fig,ax

#fig, ax = visualize_importance(models, train_feat_df)


# In[ ]:


i=0
fold = KFold(n_splits=5, shuffle=True, random_state=71)
ydfi=ydf.iloc[:,i]
y=np.array(ydfi)
cv = list(fold.split(train_feat_df, y))
oof, models = fit_lgbm(train_feat_df.values, y, cv, params=params, verbose=500)


# In[ ]:


fig, ax = visualize_importance(models, train_feat_df)
ax.set_title(target[i]+' Imortance',fontsize=20)


# In[ ]:


cols=['gender', 'Gtp', 'triglyceride', 'hemoglobin', 'waist(cm)', 'LDL',
       'ALT', 'Cholesterol', 'HDL', 'fasting blood sugar', 'height(cm)', 'age',
       'AST', 'systolic', 'relaxation', 'serum creatinine', 'weight(kg)',
       'eyesight(left)', 'eyesight(right)', 'tartar', 'dental caries',
       'Urine protein', 'hearing(left)', 'hearing(right)', 'oral']


# In[ ]:


fig, ax = plt.subplots(5,2,figsize=(16,16))
for i in tqdm(range(10)):
    r=i//2
    c=i%2
    sns.histplot(data1[data1.smoking==0][cols[i]], label='smoking=0', ax=ax[r,c], color='black',bins=30)
    sns.histplot(data1[data1.smoking==1][cols[i]], label='smoking=1', ax=ax[r,c], color='C1',bins=30)
    ax[r,c].legend()
    ax[r,c].grid()
    
plt.show()


# In[ ]:





# In[ ]:




