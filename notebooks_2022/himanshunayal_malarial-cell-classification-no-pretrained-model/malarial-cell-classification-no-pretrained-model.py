#!/usr/bin/env python
# coding: utf-8

# In[ ]:


############ Libraries ############
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

import tensorflow as tf
import tensorflow.keras.layers as layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import accuracy_score


# In[ ]:


image_dir = '../input/cell-images-for-detecting-malaria/cell_images/' 
parasitized_img = os.listdir(image_dir + 'Parasitized/') ### image files
uninfected_img = os.listdir(image_dir + 'Uninfected/') ### image files


# # Combining parasitized and uninfected images with labelling

# In[ ]:


dataset = []
label = []

############ loading parasitized images in dataset list ##############
for idx, image_name in enumerate(parasitized_img):
    if image_name.split('.')[1]=='png':
        image = cv2.imread(image_dir + 'Parasitized/'+image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64,64))
        dataset.append(np.array(image))
        label.append(0)

############ loading uninfected images in dataset list ##############
for idx, image_name in enumerate(uninfected_img):
    if image_name.split('.')[1]=='png':
        image = cv2.imread(image_dir + 'Uninfected/'+image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64,64))
        dataset.append(np.array(image))
        label.append(1)


# In[ ]:


len(dataset), len(label)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(np.array(dataset), to_categorical(np.array(label)), test_size=0.2, random_state=69)


# # Modelling

# In[ ]:



model1 = tf.keras.Sequential([
        layers.Input(dataset[0].shape),
        layers.Conv2D(32,kernel_size=3,activation='relu',padding='same'),
        layers.MaxPooling2D(pool_size=2),
        layers.BatchNormalization(axis=-1),
        layers.Dropout(rate=0.2),


        layers.Conv2D(32,kernel_size=3,activation='relu',padding='same'),
        layers.MaxPooling2D(pool_size=2),
        layers.BatchNormalization(axis=-1),
        layers.Dropout(rate=0.2),

        layers.Flatten(),

        layers.Dense(512,activation='relu'),
        layers.BatchNormalization(axis=-1),
        layers.Dropout(rate=0.2),

        layers.Dense(128,activation='relu'),
        layers.Dense(2,activation='softmax'),


])
model1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model1.summary()


# In[ ]:


from tensorflow.keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 5, verbose=1,factor=0.3, min_lr=0.000001)


# # Training

# In[ ]:


history = model1.fit(X_train, 
                         y_train, 
                         batch_size = 64, 
                         verbose = 1, 
                         epochs = 12,      
                         validation_split = 0.1,
                         shuffle = False,
                         callbacks = [learning_rate_reduction]
                     )


# In[ ]:


preds = model1.predict(X_test)


# In[ ]:


preds = preds.argmax(axis=1)


# In[ ]:


y_test = y_test.argmax(axis=1)


# In[ ]:


score = accuracy_score(y_test,preds)


# In[ ]:


score ## acccuracy


# In[ ]:


img = X_test[0]
plt.imshow(img)
plt.axis('off')


# In[ ]:


fig, axs = plt.subplots(2, 2,figsize=(12, 8))

axs[0, 0].imshow(X_test[0])
axs[0,0].set_title(f'Preds: {preds[0]}, truth: {y_test[0]}')
axs[0, 1].imshow(X_test[1])
axs[0,1].set_title(f'Preds: {preds[1]}, truth: {y_test[1]}')
axs[1, 0].imshow(X_test[2])
axs[1,0].set_title(f'Preds: {preds[2]}, truth: {y_test[2]}')
axs[1, 1].imshow(X_test[3])
axs[1,1].set_title(f'Preds: {preds[3]}, truth: {y_test[3]}')
plt.figure(figsize=(12,23))
plt.show()


# In[ ]:




