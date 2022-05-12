#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import join
import cv2
import pandas
import os
import random


# In[ ]:


data = "../input/flowers-recognition/flowers/"
folders = os.listdir(data)
print(folders)


# In[ ]:


image_names = []
train_labels = []
train_images = []

size = 64,64

for folder in folders:
    for file in os.listdir(os.path.join(data,folder)):
        if file.endswith("jpg"):
            image_names.append(os.path.join(data,folder,file))
            train_labels.append(folder)
            img = cv2.imread(os.path.join(data,folder,file))
            im = cv2.resize(img,size)
            train_images.append(im)
        else:
            continue


# In[ ]:


train = np.array(train_images)

train.shape


# In[ ]:


train = train.astype('float32') / 255.0


# In[ ]:


label_dummies = pandas.get_dummies(train_labels)

labels =  label_dummies.values.argmax(1)


# In[ ]:


pandas.unique(train_labels)


# In[ ]:


pandas.unique(labels)


# In[ ]:


union_list = list(zip(train, labels))
random.shuffle(union_list)
train,labels = zip(*union_list)


# In[ ]:


train = np.array(train)
labels = np.array(labels)


# In[ ]:


from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D

model = keras.Sequential()
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', input_shape = (64,64,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))
model.add(Conv2D(128, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))
model.add(Conv2D(128, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))
model.add(Conv2D(128, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(5, activation = 'softmax'))


# In[ ]:


model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train, labels,
          epochs=5, validation_split = 0.2)


# In[ ]:




