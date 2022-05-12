#!/usr/bin/env python
# coding: utf-8

# # <h><center> ⭐️⭐️Tabular Playground Series May 2022⭐️⭐️ </center></h>
# 
# ## **The goal of these competitions is to provide a fun and approachable-for-anyone tabular dataset to model.** 
# 
# <img src='https://4cawmi2va33i3w6dek1d7y1m-wpengine.netdna-ssl.com/wp-content/uploads/2018/08/How-to-improve-your-memory_page-1024x384.png'>
# 
# 
# ## **Try different! I am trying to Tensorflow embedding with xgboost!** 
# 
# ## ***Just try 30000 data and explore new one Try it yourself and get good result***

# # **Import Necessary library**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from math import factorial
import gc
import joblib

import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.model_selection import cross_validate,KFold,train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, roc_auc_score

import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.neighbors import KNeighborsClassifier
from tqdm.notebook import tqdm
from skimage import filters
import os
from sklearn.metrics import plot_roc_curve
from sklearn import metrics


# # **Load, Read, Shape of data**

# In[ ]:


train = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv").sample(30000)
test = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv").sample(20000)
sample = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
print(f'train_shape: {train.shape},test_shape: {test.shape},sample_shape: {sample.shape}')


# In[ ]:


train.head()


# ## **Identify categorial data**

# In[ ]:


cat_cols = list(train.select_dtypes('object').columns.values)
print(cat_cols)
df = pd.concat([train, test], axis=0)


# ## **Cat-Num --> Label Encoder**

# In[ ]:


label_encoder = preprocessing.LabelEncoder()
for col in cat_cols:
    df[col]= label_encoder.fit_transform(df[col])


# In[ ]:


train = df.iloc[:train.shape[0], :]
test = df.iloc[train.shape[0]:, :]
train.info()


# # **Statistical analysis**

# In[ ]:


print(train.shape,test.shape,sample.shape)
train.describe().T


# ## **Correlatation of data**

# In[ ]:


plt.rcParams["figure.figsize"] = (15,6)
sns.heatmap(train.corr(), cmap="rainbow_r", annot=False)
plt.show()


# # **Feature Selection**

# In[ ]:


#Define feature columns
feature_columns = {x for x in train.columns}.difference({'id','target'})
target = 'target'


# # ***Build Tensorflow Embedding XGBoost Model***

# In[ ]:


def get_model(weights=None):
    model = tf.keras.models.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(16, activation='relu' ),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss=tfa.losses.TripletSemiHardLoss(), metrics=[])
    if weights is not None:
        model.load_weights(weights)
        
    return model

def get_split(fold, with_csv=False):
    train_idx, val_idx = fold
    
    _csv_train = train.iloc[train_idx]
    _csv_val = train.iloc[val_idx]

    model = get_model()
    x = _csv_train[feature_columns].to_numpy()
    y = _csv_train[target]
    
    x_val = _csv_val[feature_columns].to_numpy()
    y_val = _csv_val[target]
    
    x, y, x_val, y_val
    
    if with_csv:
        return x, y, x_val, y_val, _csv_train, _csv_val
    
    return x, y, x_val, y_val
    


# ## **Apply tensorflow embedded with xgboost (cross validation)**

# In[ ]:


import xgboost as xgb
folds = list(KFold().split(train))

fold_i = 0

print(f'Fold #{fold_i}')

model = get_model()

x, y, x_val, y_val, csv_train, csv_val = get_split(folds[fold_i], with_csv=True)

assert not any(csv_val.id.isin(csv_train.id))


class LearningRateReducerCb(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        old_lr = self.model.optimizer.lr.read_value()
        new_lr = old_lr * 0.99
        self.model.optimizer.lr.assign(new_lr)

save_cb = tf.keras.callbacks.ModelCheckpoint(f'./best_val_new_{fold_i}', save_best_only=True, monitor='val_loss', save_weights_only=True)
h = model.fit(x, y, validation_data=(x_val, y_val), epochs=10, batch_size=64, verbose=2, callbacks=[save_cb, LearningRateReducerCb()])

del model
model = get_model(f'./best_val_new_{fold_i}')

train_emb = model.predict(x)
val_emb = model.predict(x_val)

xgb_params=  {
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'boosting_type' :	 'gbdt',
              'lambda_l1' :	 1.4679791331431786,
              'lambda_l2' :	 0.008403131421304244,
              'colsample_bytree' :	 1.0,
              'bagging_fraction' :	 0.6,
              'feature_fraction' :	 0.6,
              'learning_rate' :	 0.0015584305322779072,
              'max_depth' :	 7,
              'num_leaves' :88,
              'alpha': 0.5108154566815425,
              'gamma': 1.9276236172849432,
              'reg_lambda': 11.40999855634382,
              'subsample': 0.8386116751473301,
              'min_child_weight': 2.5517043283716605,
              'min_child_samples':	 56
}

xgb =  xgb.XGBClassifier(**xgb_params,n_estimators=5000,random_state=1)
xgb.fit(train_emb, y)

train_acc, val_acc = xgb.score(train_emb, y), xgb.score(val_emb, y_val)

print('#################################################################')
print(f'Train acc: {train_acc} Validation acc: {val_acc}')
print('#################################################################')


test_emb = model.predict(test[feature_columns])
test_pred = xgb.predict(test_emb)
test_probas = xgb.predict_proba(test_emb)


# # **Plot validation,Train Loss**

# In[ ]:


loss = h.history['loss']
val_loss = h.history['val_loss']

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(loss, label='Train loss')
ax.plot(val_loss, label='Validation loss')

ax.grid()
ax.legend()
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss value');
fig.suptitle('Loss visualization', fontsize=16);


# # **ROC_AUC, Confusion_Matrix,ClassificationReport**

# In[ ]:


val = xgb.predict(val_emb)
#Roc_Auc_Score
print('ROC_AUC_XGB_SCORE:')
roc1= metrics.roc_auc_score(val,y_val)
print(f"roc_Xgb:{roc1}")
print('-'*100)
print()

print('REPORT')
# metrics
report = metrics.classification_report(val,y_val)
print(report)
print('-'*100)
print()

print('ConfusionMatrix')
#Confusionmatrix
cf=metrics.plot_confusion_matrix(xgb,val_emb, y_val)  
plt.show(cf)
print('-'*100)
print()


# ## ***Thankyou for visiting guys***
# 
# Reference: https://www.kaggle.com/code/venkatkumar001/apc-5-tensorflow-embed-xgb
# 
# ## **In this tps may 2022, I was trying different package like tensorflow-embed-xgb, Fastai, Pytorch-tabular**
# 
# 1. https://www.kaggle.com/code/venkatkumar001/pytorch-tabular-dl-framework-2
# 2. https://www.kaggle.com/code/venkatkumar001/fast-ai-let-s-try-new-dl-framework-1
