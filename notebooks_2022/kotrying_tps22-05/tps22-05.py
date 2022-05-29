#!/usr/bin/env python
# coding: utf-8

# # Idea
#   
# **See the previous kernel for details**
# 
# **success**  
# automating feature generating and selection by AutoFeat (LGBM)   
# Label-Encoding  
# hyperparameter tuning by Optuna   
# split f_27 one character at a time  
# number of unique characters in f_27  
# 
# **failures(not use)**  
# Simple Target-Encoding    
# combine a small number of labels (8,9 or more) of categorical variables into one label  
# count the maximum number of consecutive strings in f_27  

# In[ ]:


# base
import os
import random
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

# encoding
from sklearn.preprocessing import LabelEncoder

# CV
from sklearn.model_selection import KFold

# lgb
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler

# tensorflow/keras
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.callbacks import *

# metrics
from sklearn.metrics import roc_curve, auc

# plot
import matplotlib.pyplot as plt
import seaborn as sns

# warning
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# param
n_splits=5
seed=2022


# In[ ]:


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
set_seed(seed)


# # Data

# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')
sub = pd.read_csv('../input/tabular-playground-series-may-2022/sample_submission.csv')


# In[ ]:


# Features generated by AutoFeat (Vesion8)
train_x_feature_creation = pd.read_csv('../input/tps0522-autofeat/train_x_feature_creation.csv')
test_feature_creation = pd.read_csv('../input/tps0522-autofeat/test_feature_creation.csv')


# In[ ]:


train_x_feature_creation.head(3)


# In[ ]:


test_feature_creation.head(3)


# # Model

# In[ ]:


# lightgbm
class ModelLgb:

    def __init__(self):
        self.model = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        params = {
        'objective':'binary',
        'metric':'auc',
        'seed': seed,
        'verbosity':-1,
        'learning_rate':0.1,
        'reg_alpha':0,
        'reg_lambda':1,
        'num_leaves': 480, 
        'max_depth': 31,
        'feature_fraction': 0.9558908495366608, 
        'bagging_fraction': 0.9018494038054344, 
        'bagging_freq': 5, 
        'min_child_samples': 8,
        }
        
        num_round = 10000
        early_stopping_rounds=50
        
        lgb_train = lgb.Dataset(tr_x, tr_y)
        lgb_eval = lgb.Dataset(va_x, va_y)
        
        self.model = lgb.train(params, lgb_train, valid_sets=lgb_eval, 
                               num_boost_round=num_round, early_stopping_rounds=early_stopping_rounds,
                               verbose_eval=100
                              )
        
        lgb.plot_importance(self.model, figsize=(30,60))
        
    def predict(self, x):
        pred = self.model.predict(x, num_iteration=self.model.best_iteration)
        return pred


# In[ ]:


# NN
class ModelNN:

    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.scaler = StandardScaler()
        self.scaler.fit(tr_x)

        batch_size = 128
        epochs = 100

        tr_x = self.scaler.transform(tr_x)
        va_x = self.scaler.transform(va_x)
        
        model = Sequential()
        model.add(Dense(128, kernel_regularizer=tf.keras.regularizers.l2(30e-6), activation='swish', input_shape=(tr_x.shape[1],)))
        model.add(Dense(64, kernel_regularizer=tf.keras.regularizers.l2(30e-6), activation='swish'))
        model.add(Dense(32, kernel_regularizer=tf.keras.regularizers.l2(30e-6), activation='swish'))
        model.add(Dense(16, kernel_regularizer=tf.keras.regularizers.l2(30e-6), activation='swish'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])
        
        #callbacks
        lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.7, patience=2, verbose=1)
        es = EarlyStopping(monitor='val_loss', patience=5, min_delta=1e-3, restore_best_weights=True)
        
        history = model.fit(tr_x, tr_y,
                            batch_size=batch_size, epochs=epochs,
                            verbose=1, validation_data=(va_x, va_y), callbacks=[lr,es])
        self.model = model

    def predict(self, x):
        x = self.scaler.transform(x)
        pred = self.model.predict(x)
        return pred


# # Run

# In[ ]:


base_col = train_x_feature_creation.columns[:41]
train_x = train_x_feature_creation[base_col]
train_y = train.target


# In[ ]:


def run_training(name='lgb'):
    models=[]
    total_auc = []
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (tr_idx, va_idx) in tqdm(enumerate(kf.split(train_x))):
        
        print('='*15 + f'fold{i+1}' + '='*15)

        tr_x, va_x = train_x.iloc[tr_idx].copy(), train_x.iloc[va_idx].copy()
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        
        if name == 'lgb':
            model = ModelLgb()
        if name == 'nn':
            model = ModelNN()

        model.fit(tr_x, tr_y, va_x, va_y)
        models.append(model)
        
        # val_auc  
        va_pred = model.predict(va_x)
        
        fpr, tpr, _ = roc_curve(va_y, va_pred)
        va_auc = auc(fpr, tpr)
        print(f'AUC : {va_auc}')
        total_auc.append(va_auc)
    
    print(f'Mean AUC : {np.mean(total_auc)}')
         
    return models


# In[ ]:


models_nn = run_training('nn')


# In[ ]:


# models_lgb = run_training('lgb')


# # Predict & Submit

# In[ ]:


# NN
preds=[]

for model in models_nn:
    pred = model.predict(test_feature_creation[base_col])
    preds.append(pred)

test_pred = sum(preds) / len(preds)

sub.target = test_pred
sub.to_csv('submission_nn.csv', index=False)
sub


# In[ ]:


# # LGB
# preds=[]

# for model in models_lgb:
#     pred = model.predict(test_feature_creation)
#     preds.append(pred)

# test_pred = sum(preds) / len(preds)

# sub.target = test_pred
# sub.to_csv('submission_lgb.csv', index=False)
# sub
