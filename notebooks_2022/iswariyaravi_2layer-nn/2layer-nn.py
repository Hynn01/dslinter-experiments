#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
num_classes = 10
epochs = 20


# **Create dataframes for train and test datasets**

# In[ ]:


train_df = pd.read_csv('../input/mnist-fashion-data-classification/mnist_train.csv',sep=',')
test_df = pd.read_csv('../input/mnist-fashion-data-classification/mnist_test.csv', sep = ',')

X_train = train_df.iloc[:, 1:]

y_train = train_df.iloc[:, 0]

X_test = test_df.iloc[:, 1:]

y_test = test_df.iloc[:, 0]

#full dataset classification
X_train = X_train/255.0
X_test = X_test/255.0


# In[ ]:


from keras.models import Sequential

from keras.layers import Flatten, Dense

model = Sequential()

input_layer = Flatten(input_shape = (28, 28))
model.add(input_layer)

hidden_layer1 = Dense(512, activation = 'relu')
model.add(hidden_layer1)


hidden_layer2 = Dense(256, activation = 'relu')
model.add(hidden_layer2)

output_layer = Dense(10, activation = 'softmax')
model.add(output_layer)

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = 64,
       callbacks = [tf.keras.callbacks.EarlyStopping(patience = 5, restore_best_weights=True)],
       epochs = 10)


# In[ ]:


y_pred = model.predict(X_test)

y_pred


# In[ ]:


labels = tf.argmax(y_pred, axis=1)


# # prediction on test data

# In[ ]:


# labels = tf.argmax(y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, labels)

