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


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

np.random.seed(42)


# In[ ]:


sub = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')


# In[ ]:


X = train.drop('label',axis=1)


# In[ ]:


X_norm = X/255
test_norm = test/255


# In[ ]:





# In[ ]:


y = train['label']


# In[ ]:


y_dum = pd.get_dummies(y)


# In[ ]:


test_resize = test/255


# In[ ]:


X_norm.max().max()


# In[ ]:


X_norm


# In[ ]:


X_norm.values[0,:].reshape(28,28)


# In[ ]:


X_reshape = []
for i in X_norm.values:
    X_reshape.append(i.reshape(28,28))


# In[ ]:


X_rsarr = np.array(X_reshape)


# In[ ]:


X_2d = X_rsarr.reshape(-1, 28,28,1)


# In[ ]:


X_2d.shape


# In[ ]:


test_reshape = []
for i in test_norm.values:
    test_reshape.append(i.reshape(28,28))
test_rsarr = np.array(test_reshape)
test_2d = test_rsarr.reshape(-1, 28,28,1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_2d, y_dum)


# In[ ]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(32))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('sigmoid'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[ ]:


model.fit(X_train, y_train,
          batch_size=32,
          epochs=10,
          validation_data=(X_test, y_test), 
          verbose=2)


# In[ ]:


model.fit(X_2d, y_dum,
          batch_size=32,
          epochs=5, 
          verbose=2)


# In[ ]:


sub


# In[ ]:


y_pred = model.predict(test_2d)


# In[ ]:


y_pred


# In[ ]:



y_max = np.argmax(y_pred,axis = 1)



# In[ ]:


y_max


# In[ ]:



sub['Label'] = y_max


# In[ ]:


sub


# In[ ]:


sub.to_csv('submission.csv', index=False)


# In[ ]:




