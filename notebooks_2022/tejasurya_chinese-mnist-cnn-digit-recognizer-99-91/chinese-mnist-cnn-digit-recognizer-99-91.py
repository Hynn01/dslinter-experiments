#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout,MaxPool2D, LSTM, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from keras.layers import ELU


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Data Preprocessing¶
# 
# ## Load Training Dataset

# In[ ]:


train_data = pd.read_csv("../input/chinese-mnist-digit-recognizer/chineseMNIST.csv")


# In[ ]:


train_data.head()


# In[ ]:


# unique labels
train_data['label'].unique()


# ## Split Dataset

# In[ ]:


img_rows, img_cols = 64, 64
num_classes = 15


# ## Data Preprocessing

# In[ ]:


def data_prep(raw):
    # define X and Y
    X = raw.drop(['label', 'character'], axis = 1)
    Y = raw['label']
    # Normalization
    X = X / 255.0
    # convert data to np.array
    X = X.values
    
    X = X.reshape(-1,img_rows,img_cols,1)
    
    return X, Y

x, y = data_prep(train_data)


# In[ ]:


y.replace(100, 11, inplace=True)
y.replace(1000, 12, inplace=True)
y.replace(10000, 13, inplace=True)
y.replace(100000000, 14, inplace=True)


# In[ ]:


y = tf.keras.utils.to_categorical(y, num_classes = num_classes)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)#ratio 70:30
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = 0.5, random_state = 42)


# In[ ]:


print(f"Training data size is {x_train.shape}")
print(f"Testing data size is {x_test.shape}")


# # Model Summary

# In[ ]:



model = Sequential()

model.add(Conv2D(32, kernel_size=(5, 5), padding='same',input_shape=(img_rows, img_cols, 1),activation='relu'))
model.add(Conv2D(32, kernel_size=(5, 5), padding = 'same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size=(3, 3), padding='same',activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), padding = 'same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3, 3), padding = 'same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))
model.add(Dropout(0.4))
          
model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

model.summary()


# **Data Augmentation**
# 
# One way to avoid overfitting and improve the accuracy is to increase the variability of existing samples. Which is also helps to compensate lack of data.
# Data augmentation generates data from existing samples by applying various transformations to the original dataset. This method aims to increase the number of unique input samples, which, in turn, will allow the model to show better accuracy on the validation dataset.

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


# use data augmentation to improve accuracy and prevent overfitting
augs_gen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False, 
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=10,  
        zoom_range = 0.1, 
        width_shift_range=0.1,  
        height_shift_range=0.1, 
        horizontal_flip=False,  
        vertical_flip=False) 

train_generator = augs_gen.flow(x_train, y_train, batch_size=64)


# For the optimizer I use RMSprop (root mean square propagation), one of the built-in optimizers based on the gradient descent algorithm. In the documentation we can find the formula by which the optimizer updates the model parameters.

# Keeping Learning rate lr = 0.0005 during RootMeanSquare Optimization for better application of optimisation

# In[ ]:


# optimize the model
from tensorflow.keras.optimizers import RMSprop
optimizer = RMSprop( learning_rate = 0.0005, rho = 0.9, epsilon = 0.0000001, decay=0.0, centered=False)


# In[ ]:


epochs= 100

from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0.00001, patience=8, mode='auto', restore_best_weights=True)


# In[ ]:


model.compile(optimizer = optimizer,loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])
model_fit = model.fit(train_generator,
          batch_size=64,
          epochs=epochs,
          validation_data = (x_val, y_val),
          callbacks=[early_stop],
          verbose=1)


# **Save Model**

# In[ ]:


get_ipython().system('mkdir -p saved_model')
model.save('saved_model/my_model')


# In[ ]:


load_model = tf.keras.models.load_model('saved_model/my_model')
load_model.summary()


# # Evaluate

# In[ ]:


evaluate_test = load_model.evaluate(x_test, y_test, verbose=1)

print("\nAccuracy =", "{:.7f}%".format(evaluate_test[1]*100))
print("Loss     =" ,"{:.9f}".format(evaluate_test[0]))

