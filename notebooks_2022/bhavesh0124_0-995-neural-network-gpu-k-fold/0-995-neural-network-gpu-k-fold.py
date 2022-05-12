#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import statsmodels.api as sm
import calendar
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import activations,callbacks
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from keras.models import Model

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split

train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')
test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv')
submission  = pd.read_csv('../input/tabular-playground-series-may-2022/sample_submission.csv')


# In[ ]:


import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print('GPU device NOT found')
else:
  print('Found GPU at: {}'.format(device_name))


# In[ ]:


for df in [train, test]:
    for i in range(10):
        df[f'ch{i}'] = df.f_27.str.get(i).apply(ord) - ord('A')
        
    df["unique_characters"] = df.f_27.apply(lambda s: len(set(s)))
    
features = [f for f in test.columns if f != 'id' and f != 'f_27']


# In[ ]:


X_train = train.drop(['target'],axis=1)[features]
Y_train = train['target']
X_test = test[features].copy()


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)


# In[ ]:


def nn():
    inputs = Input(shape=(len(features)))
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(30e-6),activation='swish')(inputs)
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(30e-6),activation='swish')(x)
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(30e-6), activation='swish')(x)
    x = Dense(16, kernel_regularizer=tf.keras.regularizers.l2(30e-6),activation='swish')(x)
    x = Dense(1,activation='sigmoid',)(x)
    model  = Model(inputs,x)
    return model


# In[ ]:


NN = nn()

tqdm_callback = tfa.callbacks.TQDMProgressBar()
loss = tf.keras.losses.BinaryCrossentropy()
optimizer= Adam()
es = tf.keras.callbacks.EarlyStopping( monitor= 'val_loss', patience=5, verbose=0,mode='auto', baseline=None, restore_best_weights=True)

plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=0,mode='auto')


# In[ ]:


from sklearn.model_selection import GroupKFold,KFold
from sklearn.metrics import *

pred_folds_soft_vote = []
pred_train = np.zeros(shape=(X_train.shape[0]))

kf = KFold(n_splits=5)

fold =0
for train_index, val_index in kf.split(X_train_sc, Y_train) :
    fold +=1
    x_tr = X_train_sc[train_index]
    y_tr = Y_train[train_index]
    x_val = X_train_sc[val_index]
    y_val= Y_train[val_index]
    
    model = nn()
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer = keras.optimizers.Adam(learning_rate=0.001))

    model.fit(x_tr,y_tr,
              batch_size = 4096, 
              validation_data=(x_val,y_val),
              epochs=150,
              callbacks=[es, plateau,tqdm_callback],
              verbose =2)
    
    pred_val = model.predict(x_val)
    score_fold = roc_auc_score(y_val,pred_val)
    
    print('SCORE FOLD {} = {}'.format(fold,score_fold))
    pred_train[val_index]= pred_val.squeeze()
    pred = model.predict(X_test_sc)
    pred_folds_soft_vote.append(pred)
    
pred_folds_soft_vote = np.mean(pred_folds_soft_vote,axis=0)


# In[ ]:


test['target']=pred_folds_soft_vote
submit_final = test[['id','target']]
submit_final.to_csv("nn-subsission.csv",index=False)

