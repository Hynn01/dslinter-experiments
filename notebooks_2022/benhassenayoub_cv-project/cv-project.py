#!/usr/bin/env python
# coding: utf-8

# **Libraries**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as scp  #linear algebra
import seaborn as snc #drawing statistical graphics
import os   #creating,removing directory/fetching its contents(interact with operating systems)
import tensorflow as tf
import os 
from PIL import Image
from glob import glob
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import models
from PIL import Image


# **Load data**

# In[ ]:


data_dir='../input/massachusetts-buildings-dataset/tiff'
x_train_dir='../input/massachusetts-buildings-dataset/tiff/train' #x_train_dir=os.path.join(data_dir,'train')
y_tarin_dir='../input/massachusetts-buildings-dataset/tiff/train_labels' #y_train_dir=os.path.join(data_dir,'trainlabels')

x_valid_dir='../input/massachusetts-buildings-dataset/tiff/val' #x_valid_dir=os.path.join(data_dir,'val')
y_valid_dir='../input/massachusetts-buildings-dataset/tiff/val_labels' #y_valid_dir=os.path.join(data_dir,'val_labels')

x_test_dir='../input/massachusetts-buildings-dataset/tiff/test' #x_test_dir=os.path.join(data_dir,'test')
y_test_dir='../input/massachusetts-buildings-dataset/tiff/test_labels' #y_test_dir=os.path.join(dat_dir,'test_labels')

class_dict=pd.read_csv('../input/massachusetts-buildings-dataset/label_class_dict.csv')
print(class_dict)
class_names=class_dict['name'].tolist()
print(class_names)
class_rgb_values=class_dict[['r','g','b']].values.tolist()
print(class_rgb_values)


# In[ ]:


metadata = pd.read_csv("../input/massachusetts-buildings-dataset/metadata.csv")
toy=False
if toy:
   metadata = metadata.sample(50000)

metadata.head()


# **Creating dataframe**

# In[ ]:


def GetData(folder,folder_labels):
    path ="../input/massachusetts-buildings-dataset/tiff"
    img_tiff=os.path.join(path,'./'+folder+'/')#load train
    masked_tiff=os.path.join(path,'./'+folder_labels+'/')#load the labeles
    img_list= os.listdir(img_tiff)
    mask_list=os.listdir(masked_tiff)
    img_list = [img_tiff+i for i in img_list] #create "imgs" list
    mask_list = [masked_tiff+i for i in mask_list] #create "masked "list
    Data = pd.DataFrame({'imgs':img_list,'labels':mask_list})
    Data.head(5)
    return Data


# In[ ]:


train_data=GetData('train','train_labels')
val_data=GetData('val','val_labels')
test_data=GetData('test','test_labels')


# **Data visualisation**

# In[ ]:


import cv2
def VizData(data,N):
    plt.figure(figsize=(10,10))
    plt.subplot(1,3,1)
    img=cv2.imread(data.imgs.iloc[N])
    plt.imshow(img)
    plt.subplot(1,3,2)
    msk=cv2.imread(data.labels.iloc[N])
    plt.imshow(msk)
    plt.subplot(1,3,3)
    plt.imshow(img)
    plt.imshow(msk,alpha=0.65)


# In[ ]:


VizData(train_data,2)
VizData(val_data,2)
VizData(test_data,2)


# 

# **Data augmentation**

# 

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(zoom_range=0.1,
                            rotation_range=30,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.05,
                            horizontal_flip=True,
                            fill_mode='nearest',
                            rescale=1./255)
image_train=datagen.flow_from_dataframe(train_data,  
                                    target_size=(256,256), 
                                    color_mode='rgb',
                                    shuffle=True,
                                    seed=42,
                                    x_col ="imgs", 
                                    batch_size=32,
                                    class_mode=None
)
mask_train=datagen.flow_from_dataframe(train_data, 
                                    target_size=(256,256), 
                                    color_mode='grayscale',
                                    shuffle=True,
                                    seed=42,
                                    x_col ="labels", 
                                    batch_size=32,
                                    class_mode=None)

image_val=datagen.flow_from_dataframe(val_data,  
                                    target_size=(256,256), 
                                    color_mode='rgb',
                                    shuffle=True,
                                    seed=42,
                                    x_col ="imgs", 
                                    batch_size=32,
                                    class_mode=None)
mask_val=datagen.flow_from_dataframe(val_data, 
                                    target_size=(256,256), 
                                    color_mode='grayscale',
                                    shuffle=True,
                                    seed=42,
                                    x_col ="labels", 
                                    batch_size=32,
                                    class_mode=None)


# In[ ]:


train_gen=zip(image_train,mask_train)
valid_gen=zip(image_val,mask_val)


# In[ ]:


def unet(input_size=(256,256,3)):
    inputs = layers.Input(input_size)
    
    conv1 = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    bn1 = layers.Activation('relu')(conv1)
    conv1 = layers.Conv2D(64, (3, 3), padding='same')(bn1)
    bn1 = layers.BatchNormalization(axis=3)(conv1)
    bn1 = layers.Activation('relu')(bn1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = layers.Conv2D(128, (3, 3), padding='same')(pool1)
    bn2 = layers.Activation('relu')(conv2)
    conv2 = layers.Conv2D(128, (3, 3), padding='same')(bn2)
    bn2 = layers.BatchNormalization(axis=3)(conv2)
    bn2 = layers.Activation('relu')(bn2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = layers.Conv2D(256, (3, 3), padding='same')(pool2)
    bn3 = layers.Activation('relu')(conv3)
    conv3 = layers.Conv2D(256, (3, 3), padding='same')(bn3)
    bn3 = layers.BatchNormalization(axis=3)(conv3)
    bn3 = layers.Activation('relu')(bn3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = layers.Conv2D(512, (3, 3), padding='same')(pool3)
    bn4 = layers.Activation('relu')(conv4)
    conv4 = layers.Conv2D(512, (3, 3), padding='same')(bn4)
    bn4 = layers.BatchNormalization(axis=3)(conv4)
    bn4 = layers.Activation('relu')(bn4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = layers.Conv2D(1024, (3, 3), padding='same')(pool4)
    bn5 = layers.Activation('relu')(conv5)
    conv5 = layers.Conv2D(1024, (3, 3), padding='same')(bn5)
    bn5 = layers.BatchNormalization(axis=3)(conv5)
    bn5 = layers.Activation('relu')(bn5)

    up6 = layers.concatenate([layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bn5), conv4], axis=3)
    conv6 = layers.Conv2D(512, (3, 3), padding='same')(up6)
    bn6 = layers.Activation('relu')(conv6)
    conv6 = layers.Conv2D(512, (3, 3), padding='same')(bn6)
    bn6 = layers.BatchNormalization(axis=3)(conv6)
    bn6 = layers.Activation('relu')(bn6)

    up7 = layers.concatenate([layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bn6), conv3], axis=3)
    conv7 = layers.Conv2D(256, (3, 3), padding='same')(up7)
    bn7 = layers.Activation('relu')(conv7)
    conv7 = layers.Conv2D(256, (3, 3), padding='same')(bn7)
    bn7 = layers.BatchNormalization(axis=3)(conv7)
    bn7 = layers.Activation('relu')(bn7)

    up8 = layers.concatenate([layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(bn7), conv2], axis=3)
    conv8 = layers.Conv2D(128, (3, 3), padding='same')(up8)
    bn8 = layers.Activation('relu')(conv8)
    conv8 = layers.Conv2D(128, (3, 3), padding='same')(bn8)
    bn8 = layers.BatchNormalization(axis=3)(conv8)
    bn8 = layers.Activation('relu')(bn8)

    up9 = layers.concatenate([layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(bn8), conv1], axis=3)
    conv9 = layers.Conv2D(64, (3, 3), padding='same')(up9)
    bn9 = layers.Activation('relu')(conv9)
    conv9 = layers.Conv2D(64, (3, 3), padding='same')(bn9)
    bn9 = layers.BatchNormalization(axis=3)(conv9)
    bn9 = layers.Activation('relu')(bn9)

    conv10 = layers.Conv2D(1, (1, 1), activation='sigmoid')(bn9)

    return models.Model(inputs=[inputs], outputs=[conv10])


# In[ ]:


smooth=1
def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return dice_coef_loss(y_true, y_pred) + bce(y_true, y_pred)


# In[ ]:



model = unet(input_size=(256, 256, 3))


# In[ ]:


model.summary()


# In[ ]:


model.compile(
    optimizer='adam',
    loss=bce_dice_loss,
    metrics=[dice_coef,'accuracy'])


# In[ ]:


history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=5,
    validation_steps=len(val_data) /32,
    steps_per_epoch=len(train_data) /32
)


# In[ ]:




