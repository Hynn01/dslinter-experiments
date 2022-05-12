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


get_ipython().system('pip install tensorflow-gpu')


# In[ ]:


import tensorflow as tf
print(tf.__version__)


# In[ ]:


## import basics libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


dataset = pd.read_csv("/kaggle/input/churn-modeling-dataset/Churn_Modelling.csv")


# In[ ]:


dataset.head()


# In[ ]:


X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


## Feature Engineering part
geography = pd.get_dummies(X['Geography'], drop_first=True)
gender = pd.get_dummies(X['Gender'], drop_first=True)


# In[ ]:


X = X.drop(['Geography', 'Gender'], axis=1)


# In[ ]:


X.head()


# In[ ]:


## Concatenate variables with dataframe
X = pd.concat([X, geography, gender], axis=1)


# In[ ]:


## Splitting the dataset into train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


## Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[ ]:


X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


X_train


# In[ ]:


X_train.shape


# In[ ]:


## Creatiing ANN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, PReLU, ELU, ReLU


# In[ ]:


## Intialize the ANN
classifier = Sequential()


# In[ ]:


## Adding input layer
classifier.add(Dense(units=11, activation='relu'))


# In[ ]:


## Adding First Hidden layer
classifier.add(Dense(units=22, activation='relu'))


# In[ ]:


## Adding second Hidden layer
classifier.add(Dense(units=6, activation='relu'))


# In[ ]:


## Adding output layer
classifier.add(Dense(units=1, activation='sigmoid'))


# In[ ]:


classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


## Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)


# In[ ]:


model_history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size=10, epochs=100, callbacks=early_stopping)


# In[ ]:


model_history.history.keys()


# In[ ]:


## Plotting Graph for Accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


## Plotting Graph for Loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


## Predicting the test data
y_pred = classifier.predict(X_test)
y_pred = (y_pred >= 0.5)


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:


score = accuracy_score(y_pred, y_test)
score*100


# In[ ]:




