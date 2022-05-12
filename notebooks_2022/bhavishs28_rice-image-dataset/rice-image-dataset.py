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


# In[ ]:


load_img('/kaggle/input/rice-image-dataset/Rice_Image_Dataset/Basmati/basmati (6626).jpg',target_size=(250,250))


# In[ ]:


train_gen= ImageDataGenerator(rescale=1./255,validation_split=0.3)

train_data=train_gen.flow_from_directory('/kaggle/input/rice-image-dataset/Rice_Image_Dataset',target_size=(50,50),batch_size=32,class_mode='categorical',shuffle=True,subset='training')


# In[ ]:


test = train_gen.flow_from_directory('/kaggle/input/rice-image-dataset/Rice_Image_Dataset',target_size=(50,50),batch_size=1,shuffle=False,subset='validation')


# In[ ]:


train_data.classes


# In[ ]:


train_data.class_indices


# In[ ]:


from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Flatten 
model = Sequential()

model.add(Flatten(input_shape=(50,50,3)))
model.add(Dense(10,activation='sigmoid'))
model.add(Dense(30,activation='relu'))
model.add(Dense(5,activation='sigmoid'))

model.summary()


# In[ ]:


model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics='accuracy')


# In[ ]:


model.fit(train_data,validation_data = test,epochs=5)


# In[ ]:


pred=model.predict(train_data).argmax(axis=1)    
pred


# In[ ]:


dir(train_data)


# In[ ]:


import cv2
import numpy as np

img = cv2.imread ('/kaggle/input/rice-image-dataset/Rice_Image_Dataset/Basmati/basmati (6626).jpg')
img= cv2.resize(img,(50,50))
img= img/255
img=img.reshape(-1,50,50,3)

np.round(model.predict(img))


# In[ ]:


load_img('/kaggle/input/rice-image-dataset/Rice_Image_Dataset/Basmati/basmati (6626).jpg')


# In[ ]:




