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


from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import os
from zipfile import ZipFile
import os
from tqdm.notebook import tqdm_notebook as tqdm
import cv2
import tensorflow as tf
import keras


# In[ ]:


X_train = []  
y_train = []  
os.chdir('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL')
for i in tqdm(os.listdir()):
      img = cv2.imread(i) 
      img = cv2.resize(img,(256,256))  
      X_train.append(img)    
      y_train.append("Normal")  
os.chdir('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA')
for i in tqdm(os.listdir()):
      img = cv2.imread(i)
      img = cv2.resize(img,(256,256))   
      X_train.append(img)      
      y_train.append("PNEUMONIA")  
print(len(X_train))

print(len(y_train))


# In[ ]:


plt.figure(figsize=(5,5))
plt.imshow(X_train[10], cmap="gray")
plt.axis('off')
plt.show()
print(y_train[10])


# In[ ]:


plt.figure(figsize=(5,5))
plt.imshow(X_train[4000], cmap="gray")
plt.axis('off')
plt.show()
print(y_train[4000])


# In[ ]:


from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.models import load_model
from keras.models import Model
vgg19 = VGG19(input_shape=[224,224,3], weights='imagenet', include_top=False)
for layer in vgg19.layers:
    layer.trainable = False
X = Flatten()(vgg19.output) 
output = Dense(2, activation='softmax')(X) 
model = Model(inputs=vgg19.input, outputs=output)


# In[ ]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']) 
train_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/train'
testing_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/test'
train_datagen = ImageDataGenerator(rescale = 1./255,           
                                   shear_range = 0.2,          
                                   zoom_range = 0.2,  
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255) 
train_data = train_datagen.flow_from_directory(train_dir,                      
                                               target_size = (224, 224),      
                                               batch_size = 32,
                                               class_mode = 'categorical') 

test_data = test_datagen.flow_from_directory(testing_dir,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

history = model.fit(train_data,validation_data=test_data,epochs=10) 


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Testing'], loc='upper left')
plt.show()


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Testing'], loc='upper left')
plt.show()


# In[ ]:


model.evaluate(test_data) 


# In[ ]:


model.save("/kaggle/working/model.h5")

