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
        break

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input


# In[ ]:


load_img('/kaggle/input/image-classification/images/images/travel and  adventure/Places365_val_00011185.jpg',target_size=(250,250))


# In[ ]:


train_gen= ImageDataGenerator(rescale=1./255,validation_split=0.3)

train_data=train_gen.flow_from_directory('/kaggle/input/image-classification/images/images',target_size=(50,50),batch_size=32,class_mode='categorical',shuffle=True)


# In[ ]:


test =train_gen.flow_from_directory('/kaggle/input/image-classification/validation/validation',target_size=(50,50),batch_size=1,shuffle=False)


# In[ ]:


train_data.class_indices


# In[ ]:


train_data.classes


# In[ ]:


Base_model=VGG16(input_shape=[50,50,3],weights='imagenet',include_top=False)


# In[ ]:


Base_model.trainable=True
# pred_layer =Dense
layer=10
for i in Base_model.layers[:layer]:
    Base_model.trainable=False
pool = GlobalAveragePooling2D()
mid_layer=Dense(100,activation='softmax')
final_layer =Dense(4,activation='softmax')
model = Sequential([Base_model,pool,mid_layer,final_layer])

model.summary()
Base_model.summary()


# In[ ]:


Base_model.summary()


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics='accuracy')


# In[ ]:


model.fit(train_data,validation_data=test,epochs=1)


# In[ ]:


pred=model.predict(test).argmax(axis=1)   
pred


# In[ ]:


from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout,BatchNormalization

base_model=ResNet50(input_shape=[64,64,3],weights='imagenet',include_top=False)
base_model.trainable=False   # trainable is false bcoz we are training ony the classifiers.
pool=GlobalAveragePooling2D()
mid_layer=Dense(100,activation='relu')
fin=Dense(4,activation='softmax')
model1=Sequential([
    base_model,
    pool,Dropout(0.2),
    mid_layer,
    Dropout(0.2),
    BatchNormalization(),
    fin])


# In[ ]:


base_model.summary()


# In[ ]:


model1.summary()


# In[ ]:


model1.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics='accuracy')


# In[ ]:


model1.fit(train_data,validation_data=test,epochs=1)


# In[ ]:


pred1=model1.predict(test).argmax(axis=1)   
pred1


# In[ ]:




