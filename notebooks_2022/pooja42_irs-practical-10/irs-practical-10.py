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


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_train = pd.read_csv('../input/digit-recognizer/train.csv')
df_test = pd.read_csv('../input/digit-recognizer/test.csv')


# In[ ]:


display(df_train.head(4))
display(df_test.head(4))


# In[ ]:


X_train = df_train.drop(columns=['label'])
X_test = df_test.values 
y_train = df_train["label"].values


# In[ ]:


X_train = X_train/255
X_test = X_test/255


# In[ ]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
    
  tf.keras.layers.Dense(512, activation='sigmoid'),
  tf.keras.layers.BatchNormalization(),
    
  tf.keras.layers.Dense(256, activation='sigmoid'),
  tf.keras.layers.BatchNormalization(),
  
  tf.keras.layers.Dense(128, activation='sigmoid'),
  tf.keras.layers.BatchNormalization(),
    
  tf.keras.layers.Dense(64, activation='sigmoid'),
  tf.keras.layers.BatchNormalization(),
    
  tf.keras.layers.Dense(10),
  tf.keras.layers.Activation('softmax')
])


# In[ ]:


model.compile(
    optimizer=tf.keras.optimizers.SGD(0.1),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


# In[ ]:


model.summary()

tf.keras.utils.plot_model(model)


# In[ ]:


model.fit(
    X_train,y_train, 
    epochs=20
)


# In[ ]:


predictions = model.predict(X_test) 
predictions = np.argmax(predictions,axis=1) 


# In[ ]:


print(predictions)
print(predictions.shape) 


# In[ ]:


sub=pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv', header='infer')
sub["Label"]=predictions
sub.to_csv('submission.csv', index=False)

