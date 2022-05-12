#!/usr/bin/env python
# coding: utf-8

# # Intel Image Classification Using CNN

# # Importing Libraries

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


# # Image Preprocessing

# ## Image Argumentation

# In[ ]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[ ]:


train_datagen = ImageDataGenerator(
                    rescale = 1/255,
                    shear_range = 0.2,
                    zoom_range = 0.2,
                    horizontal_flip = True,
                    vertical_flip = False)
training_set = train_datagen.flow_from_directory('../input/intel-image-classification/seg_train/seg_train',
                                                 target_size = (64, 64),
                                                 class_mode = 'categorical')


# In[ ]:


test_datagen = ImageDataGenerator(rescale = 1/255)
test_set = test_datagen.flow_from_directory('../input/intel-image-classification/seg_test/seg_test',
                                            target_size = (64, 64),
                                            class_mode = 'categorical')


# In[ ]:


training_set.class_indices


# # Building the CNN

# ### Importing the Libraries

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense


# ### Initialising the CNN

# In[ ]:


cnn = Sequential()


# ### Step 1 - Convolution

# In[ ]:


cnn.add(Convolution2D(filters=32, kernel_size=3, activation='relu', input_shape=(64,64,3)))


# ### Step 2 - Pooling

# In[ ]:


cnn.add(MaxPooling2D(pool_size=(2,2)))


# ### Adding a second convolutional layer

# In[ ]:


cnn.add(Convolution2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))


# ### Step 3 - Flattening

# In[ ]:


cnn.add(Flatten())


# ### Step 4 - Adding Hidden Layers

# In[ ]:


## Hidden Layer-1
cnn.add(Dense(units=256, activation='relu'))
## Hidden Layer-2
cnn.add(Dense(units=128, activation='relu'))


# ### Step 5 - Output Layer

# In[ ]:


cnn.add(Dense(units=6, activation='softmax'))


# # Training the CNN

# ### Compiling the CNN

# In[ ]:


cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# ### Training the CNN on the Training set and evaluating it on the Test set

# In[ ]:


cnn.fit_generator(training_set, steps_per_epoch= len(training_set), validation_data = test_set, validation_steps=len(test_set), epochs = 10)


# # Making a single prediction

# In[ ]:


from keras_preprocessing.image.utils import load_img
import numpy as np
from keras.preprocessing import image


# ### Prediction - 1

# In[ ]:


img = image.load_img('../input/intel-image-classification/seg_pred/seg_pred/73.jpg', target_size=(64,64))


# In[ ]:


img


# In[ ]:


x = image.img_to_array(img)
print(x)


# In[ ]:


x = np.expand_dims(x,axis=0)
print(x)


# In[ ]:


pred = np.argmax(cnn.predict(x),axis=1)


# In[ ]:


pred


# In[ ]:


index = ['buildings','forest','glacier','mountain','sea','street']


# In[ ]:


index[pred[0]]


# ### Prediction - 2

# In[ ]:


img = image.load_img('../input/intel-image-classification/seg_pred/seg_pred/251.jpg', target_size=(64,64))


# In[ ]:


img


# In[ ]:


x = image.img_to_array(img)
print(x)


# In[ ]:


x = np.expand_dims(x,axis=0)
print(x)


# In[ ]:


pred = np.argmax(cnn.predict(x),axis=1)
pred


# In[ ]:


index = ['buildings','forest','glacier','mountain','sea','street']


# In[ ]:


index[pred[0]]


# ### Prediction - 3

# In[ ]:


img = image.load_img('../input/intel-image-classification/seg_pred/seg_pred/429.jpg', target_size=(64,64))


# In[ ]:


img


# In[ ]:


x = image.img_to_array(img)
print(x)


# In[ ]:


x = np.expand_dims(x,axis=0)
print(x)


# In[ ]:


pred = np.argmax(cnn.predict(x),axis=1)
pred


# In[ ]:


index = ['buildings','forest','glacier','mountain','sea','street']


# In[ ]:


index[pred[0]]


# ### Prediction - 4

# In[ ]:


img = image.load_img('../input/intel-image-classification/seg_pred/seg_pred/619.jpg', target_size=(64,64))


# In[ ]:


img


# In[ ]:


x = image.img_to_array(img)
print(x)


# In[ ]:


x = np.expand_dims(x,axis=0)
print(x)


# In[ ]:


pred = np.argmax(cnn.predict(x),axis=1)
pred


# In[ ]:


index = ['buildings','forest','glacier','mountain','sea','street']


# In[ ]:


index[pred[0]]


# ### Prediction -5

# In[ ]:


img = image.load_img('../input/intel-image-classification/seg_test/seg_test/mountain/23838.jpg', target_size=(64,64))


# In[ ]:


img


# In[ ]:


x = image.img_to_array(img)
print(x)


# In[ ]:


x = np.expand_dims(x,axis=0)
print(x)


# In[ ]:


pred = np.argmax(cnn.predict(x),axis=1)
pred


# In[ ]:


index = ['buildings','forest','glacier','mountain','sea','street']


# In[ ]:


index[pred[0]]


# ### Prediction - 6

# In[ ]:


img = image.load_img('../input/intel-image-classification/seg_test/seg_test/street/20178.jpg', target_size=(64,64))


# In[ ]:


img


# In[ ]:


x = image.img_to_array(img)
print(x)


# In[ ]:


x = np.expand_dims(x,axis=0)
print(x)


# In[ ]:


pred = np.argmax(cnn.predict(x),axis=1)
pred


# In[ ]:


index = ['buildings','forest','glacier','mountain','sea','street']


# In[ ]:


index[pred[0]]


# ## If you like my notebook, an upvote would be great!
