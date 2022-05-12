#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')


# In[ ]:


df.head()


# In[ ]:


X = df.drop('label', axis=1).to_numpy()
y = df['label'].to_numpy()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 )


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


X_train = X_train / 255
X_test = X_test / 255


# In[ ]:


model = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10)
])


# In[ ]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[ ]:


history = model.fit(X_train, y_train,
                   epochs=30,
                   batch_size=128, 
                   validation_data=(X_test, y_test))


# In[ ]:


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


test = np.genfromtxt('../input/digit-recognizer/test.csv', delimiter=',', skip_header=1)
test


# In[ ]:


test.shape


# In[ ]:


probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])


# In[ ]:


predictions = probability_model.predict(test)
predictions


# In[ ]:


class_prediction = probability_model.predict_classes(test)
class_prediction


# In[ ]:


submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
submission.head()


# In[ ]:


submission['Label'] = class_prediction
submission.head()


# In[ ]:


submission.set_index('ImageId').to_csv('submission.csv')


# In[ ]:




