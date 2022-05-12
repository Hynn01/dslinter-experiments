#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from tensorflow import keras
import os
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


from sklearn.preprocessing import StandardScaler,RobustScaler
sc = StandardScaler()
rb = RobustScaler()


# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')
test = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")


# In[ ]:


target = train['target'].values
train.drop(['target', 'id', 'f_27'], axis=1, inplace = True)
test.drop(['id', 'f_27'], axis=1, inplace = True)


# In[ ]:


rb.fit(train)
train = rb.transform(train)
test = rb.transform(test)


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(train, target, test_size = 0.25, random_state = 2021)


# In[ ]:


X_train.shape,X_valid.shape, y_train.shape, y_valid.shape


# In[ ]:


opt = tf.keras.optimizers.Adam(learning_rate = 0.002)


# In[ ]:


def build_model():   
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, activation="relu", input_dim=30))

    model.add(layers.Dense(128, activation="relu", kernel_initializer='random_normal'))
    
    model.add(layers.Dense(64, activation="relu", kernel_initializer='random_normal'))
    
    model.add(layers.Dense(64, activation="relu", kernel_initializer='random_normal'))
    
    model.add(layers.Dense(32, activation="relu", kernel_initializer='random_normal'))
    
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer = opt, loss='binary_crossentropy', metrics = ['accuracy'])

    model.summary()

    return model

model = build_model()


# In[ ]:


call_back = tf.keras.callbacks.ModelCheckpoint("Model.h5", monitor='val_accuracy',verbose=1,save_best_only=True)

history = model.fit(X_train, y_train, validation_data=(X_valid,y_valid), callbacks = [call_back], batch_size = 4096, epochs = 100, verbose = 1)


# In[ ]:


############    Prediction   #############


# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')
test = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")
model = tf.keras.models.load_model('./Model.h5')


# In[ ]:


target = train['target'].values
train.drop(['target', 'id', 'f_27'], axis=1, inplace = True)
test.drop(['id', 'f_27'], axis=1, inplace = True)


# In[ ]:


rb.fit(train)
train = rb.transform(train)
test = rb.transform(test)


# In[ ]:


model.evaluate(train,target, verbose=1)


# In[ ]:


pred = model.predict(test, verbose=1)


# In[ ]:


pred.round()


# In[ ]:


submission_file = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
submission_file['target'] = pred.round()
submission_file.to_csv('submission.csv', index=False)


# In[ ]:


submission_file


# In[ ]:




