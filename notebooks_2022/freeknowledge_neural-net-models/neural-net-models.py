#!/usr/bin/env python
# coding: utf-8

# hello 
# In this notebook, I have taken some of the CNN architectures and tried to implement them layer by layer.
# I hope it will be useful for you

# In[ ]:


# import the libraries as shown below
 
import cv2
import random
import tensorflow
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as k
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import SpatialDropout2D


# # **AlexNet**

# ![](https://miro.medium.com/max/700/1*vXBvV_Unz3JAxytc5iSeoQ.png)

# In[ ]:


def alexnet(input_shape):
    input = tensorflow.keras.layers.Input(input_shape)
    x = tensorflow.keras.layers.Conv2D(96, (11,11), strides=4)(input)
    x = tensorflow.keras.layers.MaxPooling2D((3,3), strides=2, padding ="same")(x)
    
    x = tensorflow.keras.layers.Conv2D(256, (5,5), strides=1, padding ="same")(x)
    x = tensorflow.keras.layers.MaxPooling2D((5,5), strides=2, padding ="same")(x)
    
    x = tensorflow.keras.layers.Conv2D(384, (3,3), strides=1, padding ="same")(x)
    x = tensorflow.keras.layers.Conv2D(384, (3,3), strides=1, padding ="same")(x)
    x = tensorflow.keras.layers.Conv2D(256, (3,3), strides=1, padding ="same")(x)
    x = tensorflow.keras.layers.MaxPooling2D((3,3), strides=2, padding ="same")(x)
    
    x = tensorflow.keras.layers.Dense(9216)(x)
    x = tensorflow.keras.layers.Dense(4096)(x)
    x = tensorflow.keras.layers.Dense(4096)(x)
    
    model = tensorflow.keras.models.Model(input,x)
    return model


# In[ ]:


input_shape = (227,227,3)
m = alexnet(input_shape)
m.summary()


# # **VGG16** 
# ![](https://miro.medium.com/max/700/1*1gA7d9svzp_jRHPsyy63Iw.png)

# In[ ]:


def vgg16(input_shape):
    input = tensorflow.keras.layers.Input(input_shape)
    
    x = tensorflow.keras.layers.Conv2D(64,(3,3),strides=1, padding="same")(input)
    x = tensorflow.keras.layers.Conv2D(64,(3,3),strides=1, padding="same")(x)
    x = tensorflow.keras.layers.MaxPooling2D((2,2),strides=2)(x)
    
    x = tensorflow.keras.layers.Conv2D(128,(3,3),strides=1, padding="same")(x)
    x = tensorflow.keras.layers.Conv2D(128,(3,3),strides=1, padding="same")(x)
    x = tensorflow.keras.layers.MaxPooling2D((2,2),strides=2)(x)
    
    x = tensorflow.keras.layers.Conv2D(256,(3,3),strides=1, padding="same")(x)
    x = tensorflow.keras.layers.Conv2D(256,(3,3),strides=1, padding="same")(x)
    x = tensorflow.keras.layers.Conv2D(256,(3,3),strides=1, padding="same")(x)
    x = tensorflow.keras.layers.MaxPooling2D((2,2),strides=2)(x)
    
    x = tensorflow.keras.layers.Conv2D(512,(3,3),strides=1, padding="same")(x)
    x = tensorflow.keras.layers.Conv2D(512,(3,3),strides=1, padding="same")(x)
    x = tensorflow.keras.layers.Conv2D(512,(3,3),strides=1, padding="same")(x)
    x = tensorflow.keras.layers.MaxPooling2D((2,2),strides=2)(x)
    
    x = tensorflow.keras.layers.Conv2D(512,(3,3),strides=1, padding="same")(x)
    x = tensorflow.keras.layers.Conv2D(512,(3,3),strides=1, padding="same")(x)
    x = tensorflow.keras.layers.Conv2D(512,(3,3),strides=1, padding="same")(x)
    x = tensorflow.keras.layers.MaxPooling2D((2,2),strides=2)(x)
    x = tensorflow.keras.layers.Flatten()(x)
    x = tensorflow.keras.layers.Dense(4096)(x)
    x = tensorflow.keras.layers.Dense(4096)(x)
    x = tensorflow.keras.layers.Dense(1000,activation='softmax')(x)
  
    model = tensorflow.keras.models.Model(input, x)
    return model


# In[ ]:


input_shape = (224,224,3)
model = vgg16(input_shape)
model.summary()


# # **GoogleNet/Inception** 
# ![](https://cdn-images-1.medium.com/max/1600/1*CWJGqfLiVjHAIan82nPbjg.png)

# In[ ]:


def googlenet_inception(input_shape):
    def inception_block(x,f):
        t1 = tensorflow.keras.layers.Conv2D(f[0], 1, activation='relu')(x)
        t2 = tensorflow.keras.layers.Conv2D(f[1], 1, activation='relu')(x)
        t2 = tensorflow.keras.layers.Conv2D(f[2], 3, padding='same', activation='relu')(t2)
        t3 = tensorflow.keras.layers.Conv2D(f[3], 1, activation='relu')(x)
        t3 = tensorflow.keras.layers.Conv2D(f[4], 5, padding='same', activation='relu')(t3)
        t4 = tensorflow.keras.layers.MaxPooling2D(3, 1, padding='same')(x)
        t4 = tensorflow.keras.layers.Conv2D(f[5], 1, activation='relu')(t4)
        output = tensorflow.keras.layers.Concatenate()([t1, t2, t3, t4])
        return output

    input = tensorflow.keras.layers.Input(input_shape)
    
    x = tensorflow.keras.layers.Conv2D(64, (7,7),strides = 2, padding='same', activation='relu')(input)
    x = tensorflow.keras.layers.MaxPooling2D((3,3),strides = 2, padding='same')(x)
    x = tensorflow.keras.layers.Conv2D(64, (1,1),activation='relu')(x)
    x = tensorflow.keras.layers.Conv2D(192, (3,3), padding='same', activation='relu')(x)
    x = tensorflow.keras.layers.MaxPooling2D((3,3),strides = 2)(x)
    
    x = inception_block(x, [64, 96, 128, 16, 32, 32])
    x = inception_block(x, [128, 128, 192, 32, 96, 64])
    x = tensorflow.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    x = inception_block(x, [192, 96, 208, 16, 48, 64])
    x = inception_block(x, [160, 112, 224, 24, 64, 64])
    x = inception_block(x, [128, 128, 256, 24, 64, 64])
    x = inception_block(x, [112, 144, 288, 32, 64, 64])
    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = tensorflow.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)

    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = inception_block(x, [384, 192, 384, 48, 128, 128])
  
    x = tensorflow.keras.layers.AveragePooling2D(7, strides=1)(x)
    x = tensorflow.keras.layers.Dropout(0.4)(x)
  
    x = tensorflow.keras.layers.Flatten()(x)
    x = tensorflow.keras.layers.Dense(1000, activation='softmax')(x)

    
    model = tensorflow.keras.models.Model(input,x)
    return model


# In[ ]:


input_shape = (227,227,3)
model = googlenet_inception(input_shape)
model.summary()


# # **MobileNet**
# ![](https://miro.medium.com/max/856/1*2IHiEn6SYGgz-p80jhYeGg.png)

# In[ ]:


def MobileNet(input_shape, nb_classes):
    input = tensorflow.keras.layers.Input(input_shape)
    x = tensorflow.keras.layers.Conv2D(32, (3,3), strides=2, padding ="same")(input)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    
    x = tensorflow.keras.layers.DepthwiseConv2D((3,3), strides=1, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    x = tensorflow.keras.layers.Conv2D(64, (1,1), strides=1, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    
    
    x = tensorflow.keras.layers.DepthwiseConv2D((3,3), strides=2, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    x = tensorflow.keras.layers.Conv2D(128, (1,1), strides=1, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    
    x = tensorflow.keras.layers.DepthwiseConv2D((3,3), strides=1, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    x = tensorflow.keras.layers.Conv2D(128, (1,1), strides=1, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    
    x = tensorflow.keras.layers.DepthwiseConv2D((3,3), strides=2, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    x = tensorflow.keras.layers.Conv2D(256, (1,1), strides=1, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    
    x = tensorflow.keras.layers.DepthwiseConv2D((3,3), strides=1, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    x = tensorflow.keras.layers.Conv2D(256, (1,1), strides=1, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    
    x = tensorflow.keras.layers.DepthwiseConv2D((3,3), strides=2, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    x = tensorflow.keras.layers.Conv2D(512, (1,1), strides=1, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
     # five blocks
    x = tensorflow.keras.layers.DepthwiseConv2D((3,3), strides=1, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    x = tensorflow.keras.layers.Conv2D(512, (1,1), strides=1, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    
    x = tensorflow.keras.layers.DepthwiseConv2D((3,3), strides=1, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    x = tensorflow.keras.layers.Conv2D(512, (1,1), strides=1, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    
    x = tensorflow.keras.layers.DepthwiseConv2D((3,3), strides=1, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    x = tensorflow.keras.layers.Conv2D(512, (1,1), strides=1, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    
    x = tensorflow.keras.layers.DepthwiseConv2D((3,3), strides=1, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    x = tensorflow.keras.layers.Conv2D(512, (1,1), strides=1, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    
    x = tensorflow.keras.layers.DepthwiseConv2D((3,3), strides=1, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    x = tensorflow.keras.layers.Conv2D(512, (1,1), strides=1, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    
    # end five blocks
    
    x = tensorflow.keras.layers.DepthwiseConv2D((3,3), strides=2, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    x = tensorflow.keras.layers.Conv2D(1024, (1,1), strides=1, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    
    
    x = tensorflow.keras.layers.DepthwiseConv2D((3,3), strides=1, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    x = tensorflow.keras.layers.Conv2D(1024, (1,1), strides=1, padding='same')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.ReLU()(x)
    
    
    x = tensorflow.keras.layers.GlobalAvgPool2D()(x)
    output = Dense(nb_classes, activation='softmax')(x)
    
    model = tensorflow.keras.models.Model(input,output)
    return model


# In[ ]:


input_shape = (224,224,3)
model = MobileNet(input_shape,1000)
model.summary()


# # **ResNet 50**
# ![](https://iq.opengenus.org/content/images/2020/03/Screenshot-from-2020-03-20-15-49-54.png)

# In[ ]:


# ResNet model creation
def ResNet50(input_w,input_h):
  if tensorflow.keras.backend.image_data_format() == 'channels_first':
    input_shape = (3, input_w, input_h)
  else:
    input_shape = (input_w, input_h,3)

 
  model_input = tensorflow.keras.layers.Input(shape=input_shape)
  
 
  ########### Block 1 ###########
  block_1 = tensorflow.keras.layers.Conv2D(64,kernel_size=7,strides=2, padding='same')(model_input)
  block_1 = tensorflow.keras.layers.BatchNormalization()(block_1)
  block_1 = tensorflow.keras.layers.ReLU()(block_1)
  block_1 = tensorflow.keras.layers.MaxPooling2D(3, strides=2, padding='same')(block_1)
 ############ Block 2 #############
 ### conv_block ###
  block_2_1 = tensorflow.keras.layers.BatchNormalization()(block_1)
  block_2_1 = tensorflow.keras.layers.ReLU()(block_2_1)
  block_2_1 = tensorflow.keras.layers.Conv2D(64,kernel_size=1,strides=1, padding='same')(block_2_1)
  
  block_2_1 = tensorflow.keras.layers.BatchNormalization()(block_2_1)
  block_2_1 = tensorflow.keras.layers.ReLU()(block_2_1)
  block_2_1 = tensorflow.keras.layers.Conv2D(64,kernel_size=3,strides=1,padding='same')(block_2_1)
  
  block_2_1 = tensorflow.keras.layers.BatchNormalization()(block_2_1)
  block_2_1 = tensorflow.keras.layers.ReLU()(block_2_1)
  block_2_1 = tensorflow.keras.layers.Conv2D(256,kernel_size=1,strides=1, padding='same')(block_2_1)

  sh_cut_2 = tensorflow.keras.layers.BatchNormalization()(block_1)
  sh_cut_2 = tensorflow.keras.layers.ReLU()(sh_cut_2)
  sh_cut_2 = tensorflow.keras.layers.Conv2D(256,kernel_size=1,strides=1, padding='same')(sh_cut_2)

  stg1_blok_2_1 = tensorflow.keras.layers.add([sh_cut_2, block_2_1])
  ### identity_block 1 ###
  block_2_2 = tensorflow.keras.layers.BatchNormalization()(stg1_blok_2_1)
  block_2_2 = tensorflow.keras.layers.ReLU()(block_2_2)
  block_2_2 = tensorflow.keras.layers.Conv2D(64,kernel_size=1,strides=1, padding='same')(block_2_2)
  
  block_2_2 = tensorflow.keras.layers.BatchNormalization()(block_2_2)
  block_2_2 = tensorflow.keras.layers.ReLU()(block_2_2)
  block_2_2 = tensorflow.keras.layers.Conv2D(64,kernel_size=3,strides=1,padding='same')(block_2_2)
  
  block_2_2 = tensorflow.keras.layers.BatchNormalization()(block_2_2)
  block_2_2 = tensorflow.keras.layers.ReLU()(block_2_2)
  block_2_2 = tensorflow.keras.layers.Conv2D(256,kernel_size=1,strides=1, padding='same')(block_2_2)

  stg1_blok_2_2 = tensorflow.keras.layers.add([stg1_blok_2_1, block_2_2])

  ### identity_block 2 ###
  block_2_3 = tensorflow.keras.layers.BatchNormalization()(stg1_blok_2_2)
  block_2_3 = tensorflow.keras.layers.ReLU()(block_2_3)
  block_2_3 = tensorflow.keras.layers.Conv2D(64,kernel_size=1,strides=1, padding='same')(block_2_3)
  
  block_2_3 = tensorflow.keras.layers.BatchNormalization()(block_2_3)
  block_2_3 = tensorflow.keras.layers.ReLU()(block_2_3)
  block_2_3 = tensorflow.keras.layers.Conv2D(64,kernel_size=3,strides=1,padding='same')(block_2_3)
  
  block_2_3 = tensorflow.keras.layers.BatchNormalization()(block_2_3)
  block_2_3 = tensorflow.keras.layers.ReLU()(block_2_3)
  block_2_3 = tensorflow.keras.layers.Conv2D(256,kernel_size=1,strides=1, padding='same')(block_2_3)

  stg1_blok_2_3 = tensorflow.keras.layers.add([stg1_blok_2_2, block_2_3])

  block_1 = tensorflow.keras.layers.BatchNormalization()(block_1)
  block_1 = tensorflow.keras.layers.Conv2D(256,kernel_size=1,strides=1, padding='same')(block_1)

  short_cut_2 = tensorflow.keras.layers.add([stg1_blok_2_3, block_1])

############ Block 3 #############
  ### 1 : conv_block ###
  block_3_1 = tensorflow.keras.layers.BatchNormalization()(short_cut_2)
  block_3_1 = tensorflow.keras.layers.ReLU()(block_3_1)
  block_3_1 = tensorflow.keras.layers.Conv2D(128,kernel_size=1,strides=1, padding='same')(block_3_1)
  
  block_3_1 = tensorflow.keras.layers.BatchNormalization()(block_3_1)
  block_3_1 = tensorflow.keras.layers.ReLU()(block_3_1)
  block_3_1 = tensorflow.keras.layers.Conv2D(128,kernel_size=3,strides=1,padding='same')(block_3_1)
  
  block_3_1 = tensorflow.keras.layers.BatchNormalization()(block_3_1)
  block_3_1 = tensorflow.keras.layers.ReLU()(block_3_1)
  block_3_1 = tensorflow.keras.layers.Conv2D(512,kernel_size=1,strides=1, padding='same')(block_3_1)

  sh_cut_3 = tensorflow.keras.layers.BatchNormalization()(short_cut_2)
  sh_cut_3 = tensorflow.keras.layers.ReLU()(sh_cut_3)
  sh_cut_3 = tensorflow.keras.layers.Conv2D(512,kernel_size=1,strides=1, padding='same')(sh_cut_3)

  stg1_blok_3_1 = tensorflow.keras.layers.add([sh_cut_3, block_3_1])
  ### 2 : identity_block  ###
  block_3_2 = tensorflow.keras.layers.BatchNormalization()(stg1_blok_3_1)
  block_3_2 = tensorflow.keras.layers.ReLU()(block_3_2)
  block_3_2 = tensorflow.keras.layers.Conv2D(128,kernel_size=1,strides=1, padding='same')(block_3_2)
  
  block_3_2 = tensorflow.keras.layers.BatchNormalization()(block_3_2)
  block_3_2 = tensorflow.keras.layers.ReLU()(block_3_2)
  block_3_2 = tensorflow.keras.layers.Conv2D(128,kernel_size=3,strides=1,padding='same')(block_3_2)
  
  block_3_2 = tensorflow.keras.layers.BatchNormalization()(block_3_2)
  block_3_2 = tensorflow.keras.layers.ReLU()(block_3_2)
  block_3_2 = tensorflow.keras.layers.Conv2D(512,kernel_size=1,strides=1, padding='same')(block_3_2)

  stg1_blok_3_2 = tensorflow.keras.layers.add([stg1_blok_3_1, block_3_2])

  ### 3 : identity_block  ###
  block_3_3 = tensorflow.keras.layers.BatchNormalization()(stg1_blok_3_2)
  block_3_3 = tensorflow.keras.layers.ReLU()(block_3_3)
  block_3_3 = tensorflow.keras.layers.Conv2D(128,kernel_size=1,strides=1, padding='same')(block_3_3)
  
  block_3_3 = tensorflow.keras.layers.BatchNormalization()(block_3_3)
  block_3_3 = tensorflow.keras.layers.ReLU()(block_3_3)
  block_3_3 = tensorflow.keras.layers.Conv2D(128,kernel_size=3,strides=1,padding='same')(block_3_3)
  
  block_3_3 = tensorflow.keras.layers.BatchNormalization()(block_3_3)
  block_3_3 = tensorflow.keras.layers.ReLU()(block_3_3)
  block_3_3 = tensorflow.keras.layers.Conv2D(512,kernel_size=1,strides=1, padding='same')(block_3_3)
  stg1_blok_3_3 = tensorflow.keras.layers.add([stg1_blok_3_2, block_3_3])

 
  

  ### 4:identity_block  ###
  block_3_4 = tensorflow.keras.layers.BatchNormalization()(stg1_blok_3_3)
  block_3_4 = tensorflow.keras.layers.ReLU()(block_3_4)
  block_3_4 = tensorflow.keras.layers.Conv2D(128,kernel_size=1,strides=1, padding='same')(block_3_4)
  
  block_3_4 = tensorflow.keras.layers.BatchNormalization()(block_3_4)
  block_3_4 = tensorflow.keras.layers.ReLU()(block_3_4)
  block_3_4 = tensorflow.keras.layers.Conv2D(128,kernel_size=3,strides=1,padding='same')(block_3_4)
  
  block_3_4 = tensorflow.keras.layers.BatchNormalization()(block_3_4)
  block_3_4 = tensorflow.keras.layers.ReLU()(block_3_4)
  block_3_4 = tensorflow.keras.layers.Conv2D(512,kernel_size=1,strides=1, padding='same')(block_3_4)

  stg1_blok_3_4 = tensorflow.keras.layers.add([stg1_blok_3_3, block_3_4])

  short_cut_2 = tensorflow.keras.layers.BatchNormalization()(short_cut_2)
  short_cut_2 = tensorflow.keras.layers.Conv2D(512,kernel_size=1,strides=1, padding='same')(short_cut_2)

  short_cut_3 = tensorflow.keras.layers.add([stg1_blok_3_4, short_cut_2])


############ Block 4 #############
  ### 1 : conv_block ###
  block_4_1 = tensorflow.keras.layers.BatchNormalization()(short_cut_3)
  block_4_1 = tensorflow.keras.layers.ReLU()(block_4_1)
  block_4_1 = tensorflow.keras.layers.Conv2D(256,kernel_size=1,strides=1, padding='same')(block_4_1)
  
  block_4_1 = tensorflow.keras.layers.BatchNormalization()(block_4_1)
  block_4_1 = tensorflow.keras.layers.ReLU()(block_4_1)
  block_4_1 = tensorflow.keras.layers.Conv2D(256,kernel_size=3,strides=1,padding='same')(block_4_1)
  
  block_4_1 = tensorflow.keras.layers.BatchNormalization()(block_4_1)
  block_4_1 = tensorflow.keras.layers.ReLU()(block_4_1)
  block_4_1 = tensorflow.keras.layers.Conv2D(1024,kernel_size=1,strides=1, padding='same')(block_4_1)

  sh_cut_4 = tensorflow.keras.layers.BatchNormalization()(short_cut_3)
  sh_cut_4 = tensorflow.keras.layers.ReLU()(sh_cut_4)
  sh_cut_4 = tensorflow.keras.layers.Conv2D(1024,kernel_size=1,strides=1, padding='same')(sh_cut_4)

  stg1_blok_4_1 = tensorflow.keras.layers.add([sh_cut_4, block_4_1])

  ### 2 :identity_block  ###
  block_4_2 = tensorflow.keras.layers.BatchNormalization()(stg1_blok_4_1)
  block_4_2 = tensorflow.keras.layers.ReLU()(block_4_2)
  block_4_1 = tensorflow.keras.layers.Conv2D(256,kernel_size=1,strides=1, padding='same')(block_4_2)
  
  block_4_2 = tensorflow.keras.layers.BatchNormalization()(block_4_2)
  block_4_2 = tensorflow.keras.layers.ReLU()(block_4_2)
  block_4_2 = tensorflow.keras.layers.Conv2D(256,kernel_size=3,strides=1,padding='same')(block_4_2)
  
  block_4_2 = tensorflow.keras.layers.BatchNormalization()(block_4_2)
  block_4_2 = tensorflow.keras.layers.ReLU()(block_4_2)
  block_4_2 = tensorflow.keras.layers.Conv2D(1024,kernel_size=1,strides=1, padding='same')(block_4_2)
  stg1_blok_4_2 = tensorflow.keras.layers.add([stg1_blok_4_1, block_4_2])

  ### 3 : identity_block  ###
  block_4_3 = tensorflow.keras.layers.BatchNormalization()(stg1_blok_4_2)
  block_4_3 = tensorflow.keras.layers.ReLU()(block_4_3)
  block_4_3 = tensorflow.keras.layers.Conv2D(256,kernel_size=1,strides=1, padding='same')(block_4_3)
  
  block_4_3 = tensorflow.keras.layers.BatchNormalization()(block_4_3)
  block_4_3 = tensorflow.keras.layers.ReLU()(block_4_3)
  block_4_3 = tensorflow.keras.layers.Conv2D(256,kernel_size=3,strides=1,padding='same')(block_4_3)
  
  block_4_3 = tensorflow.keras.layers.BatchNormalization()(block_4_3)
  block_4_3 = tensorflow.keras.layers.ReLU()(block_4_3)
  block_4_3 = tensorflow.keras.layers.Conv2D(1024,kernel_size=1,strides=1, padding='same')(block_4_3)
  stg1_blok_4_3 = tensorflow.keras.layers.add([stg1_blok_4_2, block_4_3])

  ### 4 : identity_block  ###
  block_4_4 = tensorflow.keras.layers.BatchNormalization()(stg1_blok_4_3)
  block_4_4 = tensorflow.keras.layers.ReLU()(block_4_4)
  block_4_4 = tensorflow.keras.layers.Conv2D(256,kernel_size=1,strides=1, padding='same')(block_4_4)
  
  block_4_4 = tensorflow.keras.layers.BatchNormalization()(block_4_4)
  block_4_4 = tensorflow.keras.layers.ReLU()(block_4_4)
  block_4_4 = tensorflow.keras.layers.Conv2D(256,kernel_size=3,strides=1,padding='same')(block_4_4)
  
  block_4_4 = tensorflow.keras.layers.BatchNormalization()(block_4_4)
  block_4_4 = tensorflow.keras.layers.ReLU()(block_4_4)
  block_4_4 = tensorflow.keras.layers.Conv2D(1024,kernel_size=1,strides=1, padding='same')(block_4_4)
  stg1_blok_4_4 = tensorflow.keras.layers.add([stg1_blok_4_3, block_4_4])

  ### 5 : :identity_block  ###
  block_4_5 = tensorflow.keras.layers.BatchNormalization()(stg1_blok_4_4)
  block_4_5 = tensorflow.keras.layers.ReLU()(block_4_5)
  block_4_5 = tensorflow.keras.layers.Conv2D(256,kernel_size=1,strides=1, padding='same')(block_4_5)
  
  block_4_5 = tensorflow.keras.layers.BatchNormalization()(block_4_5)
  block_4_5 = tensorflow.keras.layers.ReLU()(block_4_5)
  block_4_5 = tensorflow.keras.layers.Conv2D(256,kernel_size=3,strides=1,padding='same')(block_4_5)
  
  block_4_5 = tensorflow.keras.layers.BatchNormalization()(block_4_5)
  block_4_5 = tensorflow.keras.layers.ReLU()(block_4_5)
  block_4_5 = tensorflow.keras.layers.Conv2D(1024,kernel_size=1,strides=1, padding='same')(block_4_5)
  stg1_blok_4_5 = tensorflow.keras.layers.add([stg1_blok_4_4, block_4_5])


  short_cut_3 = tensorflow.keras.layers.BatchNormalization()(short_cut_3)
  short_cut_3 = tensorflow.keras.layers.Conv2D(1024,kernel_size=1,strides=1, padding='same')(short_cut_3)

  short_cut_4 = tensorflow.keras.layers.add([stg1_blok_4_5, short_cut_3])

############ Block 5 #############
  ### conv_block ###
  block_5_1 = tensorflow.keras.layers.BatchNormalization()(short_cut_4)
  block_5_1 = tensorflow.keras.layers.ReLU()(block_5_1)
  block_5_1 = tensorflow.keras.layers.Conv2D(512,kernel_size=1,strides=1, padding='same')(block_5_1)
  
  block_5_1 = tensorflow.keras.layers.BatchNormalization()(block_5_1)
  block_5_1 = tensorflow.keras.layers.ReLU()(block_5_1)
  block_5_1 = tensorflow.keras.layers.Conv2D(512,kernel_size=3,strides=1,padding='same')(block_5_1)
  
  block_5_1 = tensorflow.keras.layers.BatchNormalization()(block_5_1)
  block_5_1 = tensorflow.keras.layers.ReLU()(block_5_1)
  block_5_1 = tensorflow.keras.layers.Conv2D(2048,kernel_size=1,strides=1, padding='same')(block_5_1)

  sh_cut_5 = tensorflow.keras.layers.BatchNormalization()(short_cut_4)
  sh_cut_5 = tensorflow.keras.layers.ReLU()(sh_cut_5)
  sh_cut_5 = tensorflow.keras.layers.Conv2D(2048,kernel_size=1,strides=1, padding='same')(sh_cut_5)

  stg1_blok_5_1 = tensorflow.keras.layers.add([sh_cut_5, block_5_1])
  ### identity_block 1 ###
  block_5_2 = tensorflow.keras.layers.BatchNormalization()(stg1_blok_5_1)
  block_5_2 = tensorflow.keras.layers.ReLU()(block_5_2)
  block_5_2 = tensorflow.keras.layers.Conv2D(512,kernel_size=1,strides=1, padding='same')(block_5_2)
  
  block_5_2 = tensorflow.keras.layers.BatchNormalization()(block_5_2)
  block_5_2 = tensorflow.keras.layers.ReLU()(block_5_2)
  block_5_2 = tensorflow.keras.layers.Conv2D(512,kernel_size=3,strides=1,padding='same')(block_5_2)
  
  block_5_2 = tensorflow.keras.layers.BatchNormalization()(block_5_2)
  block_5_2 = tensorflow.keras.layers.ReLU()(block_5_2)
  block_5_2 = tensorflow.keras.layers.Conv2D(2048,kernel_size=1,strides=1, padding='same')(block_5_2)

  stg1_blok_5_2 = tensorflow.keras.layers.add([stg1_blok_5_1, block_5_2])

  ### identity_block 2 ###
  block_5_3 = tensorflow.keras.layers.BatchNormalization()(stg1_blok_5_2)
  block_5_3 = tensorflow.keras.layers.ReLU()(block_5_3)
  block_5_3 = tensorflow.keras.layers.Conv2D(512,kernel_size=1,strides=1, padding='same')(block_5_3)
  
  block_5_3 = tensorflow.keras.layers.BatchNormalization()(block_5_3)
  block_5_3 = tensorflow.keras.layers.ReLU()(block_5_3)
  block_5_3 = tensorflow.keras.layers.Conv2D(512,kernel_size=3,strides=1,padding='same')(block_5_3)
  
  block_5_3 = tensorflow.keras.layers.BatchNormalization()(block_5_3)
  block_5_3 = tensorflow.keras.layers.ReLU()(block_5_3)
  block_5_3 = tensorflow.keras.layers.Conv2D(2048,kernel_size=1,strides=1, padding='same')(block_5_3)

  stg1_blok_5_3 = tensorflow.keras.layers.add([stg1_blok_5_2, block_5_3])

  short_cut_4 = tensorflow.keras.layers.BatchNormalization()(short_cut_4)
  short_cut_4 = tensorflow.keras.layers.Conv2D(2048,kernel_size=1,strides=1, padding='same')(short_cut_4)

  short_cut_5 = tensorflow.keras.layers.add([stg1_blok_5_3, short_cut_4])
############ Block 6 #############
  
  pooling = tensorflow.keras.layers.GlobalAveragePooling2D()(short_cut_5)
  model_output = tensorflow.keras.layers.Dense(5,activation='softmax')(pooling)
  
  model = tensorflow.keras.models.Model(model_input,model_output)
  return model 


# In[ ]:


image_w = 256
image_h = 256
n_classes = 5
model = ResNet50(image_w,image_h)
model.summary()


# # **U-Net**
# ![](https://datascientest.com/wp-content/uploads/2021/05/u-net-architecture-1024x682.png)

# In[ ]:


def UNet(nb_classes, input_shape):
    
    input = tensorflow.keras.layers.Input(input_shape)
    
    c1 = tensorflow.keras.layers.Conv2D(64, (3,3), strides=1)(input)
    c1 = tensorflow.keras.layers.BatchNormalization()(c1)
    c1 = tensorflow.keras.layers.ReLU()(c1)
    
    c1 = tensorflow.keras.layers.Conv2D(64, (3,3), strides=1)(c1)
    c1 = tensorflow.keras.layers.BatchNormalization()(c1)
    c1 = tensorflow.keras.layers.ReLU()(c1)
    
    p1 = tensorflow.keras.layers.MaxPool2D(2, strides=2)(c1)
    
    c2 = tensorflow.keras.layers.Conv2D(128, (3,3), strides=1)(p1)
    c2 = tensorflow.keras.layers.BatchNormalization()(c2)
    c2 = tensorflow.keras.layers.ReLU()(c2)

    c2 = tensorflow.keras.layers.Conv2D(128, (3,3), strides=1)(c2)
    c2 = tensorflow.keras.layers.BatchNormalization()(c2)
    c2 = tensorflow.keras.layers.ReLU()(c2)
    
    p2 = tensorflow.keras.layers.MaxPool2D(2, strides=2)(c2)
    
    c3 = tensorflow.keras.layers.Conv2D(256, (3,3) ,strides= 1)(p2)
    c3 = tensorflow.keras.layers.BatchNormalization()(c3)
    c3 = tensorflow.keras.layers.ReLU()(c3)

    c3 = tensorflow.keras.layers.Conv2D(256, (3,3) , strides=1)(c3)
    c3 = tensorflow.keras.layers.BatchNormalization()(c3)
    c3 = tensorflow.keras.layers.ReLU()(c3)
    
    p3 = tensorflow.keras.layers.MaxPool2D(2, strides=2)(c3)
    
    c4 = tensorflow.keras.layers.Conv2D(512, (3,3), strides=1)(p3)
    c4 = tensorflow.keras.layers.BatchNormalization()(c4)
    c4 = tensorflow.keras.layers.ReLU()(c4)

    c4 = tensorflow.keras.layers.Conv2D(512, (3,3), strides=1)(c4)
    c4 = tensorflow.keras.layers.BatchNormalization()(c4)
    c4 = tensorflow.keras.layers.ReLU()(c4)
    
    
    p4 = tensorflow.keras.layers.MaxPool2D(2, strides=2)(c4)
    
    c5 = tensorflow.keras.layers.Conv2D(1024, (3,3), strides=1)(p4)
    c5 = tensorflow.keras.layers.BatchNormalization()(c5)
    c5 = tensorflow.keras.layers.ReLU()(c5)

    c5 = tensorflow.keras.layers.Conv2D(1024, (3,3), strides=1)(c5)
    c5 = tensorflow.keras.layers.BatchNormalization()(c5)
    c5 = tensorflow.keras.layers.ReLU()(c5)
    
    
    c6 = tensorflow.keras.layers.Conv2DTranspose(1024,(2,2), strides=2)(c5)
    u1 = tensorflow.image.resize(c4, ((np.shape(c6)[1]), (np.shape(c6)[2])))
    c6 = tensorflow.keras.layers.concatenate([u1,c6])
    
    c6 = tensorflow.keras.layers.Conv2D(512, (3,3), strides=1)(c6)
    c6 = tensorflow.keras.layers.BatchNormalization()(c6)
    c6 = tensorflow.keras.layers.ReLU()(c6)

    c6 = tensorflow.keras.layers.Conv2D(512, (3,3), strides=1)(c6)
    c6 = tensorflow.keras.layers.BatchNormalization()(c6)
    c6 = tensorflow.keras.layers.ReLU()(c6)
    
    c7 = tensorflow.keras.layers.Conv2DTranspose(512,(2,2), strides=2)(c6)
    u2 = tensorflow.image.resize(c3, ((np.shape(c7)[1]), (np.shape(c7)[2])))
    c7 = tensorflow.keras.layers.concatenate([u2,c7])
    
    c7 = tensorflow.keras.layers.Conv2D(256, (3,3), strides=1)(c7)
    c7 = tensorflow.keras.layers.BatchNormalization()(c7)
    c7 = tensorflow.keras.layers.ReLU()(c7)

    c7 = tensorflow.keras.layers.Conv2D(256, (3,3), strides=1)(c7)
    c7 = tensorflow.keras.layers.BatchNormalization()(c7)
    c7 = tensorflow.keras.layers.ReLU()(c7)
    
    c8 = tensorflow.keras.layers.Conv2DTranspose(256,(2,2), strides=2)(c7)
    u3 = tensorflow.image.resize(c2, ((np.shape(c8)[1]), (np.shape(c8)[2])))
    c8 = tensorflow.keras.layers.concatenate([u3,c8])
    
    c8 = tensorflow.keras.layers.Conv2D(128, (3,3), strides=1)(c8)
    c8 = tensorflow.keras.layers.BatchNormalization()(c8)
    c8 = tensorflow.keras.layers.ReLU()(c8)

    c8 = tensorflow.keras.layers.Conv2D(128, (3,3), strides=1)(c8)
    c8 = tensorflow.keras.layers.BatchNormalization()(c8)
    c8 = tensorflow.keras.layers.ReLU()(c8)
    
    c9 = tensorflow.keras.layers.Conv2DTranspose(128,(2,2), strides=2)(c8)
    u4 = tensorflow.image.resize(c1, ((np.shape(c9)[1]), (np.shape(c9)[2])))
    c9 = tensorflow.keras.layers.concatenate([u4,c9])
    
    c9 = tensorflow.keras.layers.Conv2D(64, (3,3), strides=1)(c9)
    c9 = tensorflow.keras.layers.BatchNormalization()(c9)
    c9 = tensorflow.keras.layers.ReLU()(c9)

    c9 = tensorflow.keras.layers.Conv2D(64, (3,3), strides=1)(c9)
    c9 = tensorflow.keras.layers.BatchNormalization()(c9)
    c9 = tensorflow.keras.layers.ReLU()(c9)
    
    c9 = tensorflow.keras.layers.Conv2D(2, (1,1), strides=1)(c9)
    c9 = tensorflow.keras.layers.BatchNormalization()(c9)
    c9 = tensorflow.keras.layers.ReLU()(c9)
    
    model = tensorflow.keras.models.Model(input,c9)
    return model 


# In[ ]:


input_shape = 572,572,1
nb_classes = 2
m = UNet(nb_classes, input_shape)
m.summary()


# # **se_ResNet50**

# In[ ]:


def se_ResNet50(input_w,input_h):
  if tf.keras.backend.image_data_format() == 'channels_first':
    input_shape = (3, input_w, input_h)
  else:
    input_shape = (input_w, input_h,3)

  model_input = tf.keras.layers.Input(shape=input_shape)
 

  ########### Block 1 ###########
  block_1 = tf.keras.layers.Conv2D(64,kernel_size=3,strides=1,padding='same')(model_input)
  block_1 = tf.keras.layers.BatchNormalization()(block_1)
  block_1 = tf.keras.layers.ReLU()(block_1)
  block_1 = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(block_1)
 ############ Block 2 #############
 ### conv_block ###
  block_2_1 = tf.keras.layers.BatchNormalization()(block_1)
  block_2_1 = tf.keras.layers.ReLU()(block_2_1)
  block_2_1 = tf.keras.layers.Conv2D(64,kernel_size=1,strides=1, padding='same')(block_2_1)
  
  block_2_1 = tf.keras.layers.BatchNormalization()(block_2_1)
  block_2_1 = tf.keras.layers.ReLU()(block_2_1)
  block_2_1 = tf.keras.layers.Conv2D(64,kernel_size=3,strides=1,padding='same')(block_2_1)
  
  block_2_1 = tf.keras.layers.BatchNormalization()(block_2_1)
  block_2_1 = tf.keras.layers.ReLU()(block_2_1)
  block_2_1 = tf.keras.layers.Conv2D(256,kernel_size=1,strides=1, padding='same')(block_2_1)

  sh_cut_2 = tf.keras.layers.BatchNormalization()(block_1)
  sh_cut_2 = tf.keras.layers.ReLU()(sh_cut_2)
  sh_cut_2 = tf.keras.layers.Conv2D(256,kernel_size=1,strides=1, padding='same')(sh_cut_2)

  stg1_blok_2_1 = tf.keras.layers.add([sh_cut_2, block_2_1])
  ### identity_block 1 ###
  block_2_2 = tf.keras.layers.BatchNormalization()(stg1_blok_2_1)
  block_2_2 = tf.keras.layers.ReLU()(block_2_2)
  block_2_2 = tf.keras.layers.Conv2D(64,kernel_size=1,strides=1, padding='same')(block_2_2)
  
  block_2_2 = tf.keras.layers.BatchNormalization()(block_2_2)
  block_2_2 = tf.keras.layers.ReLU()(block_2_2)
  block_2_2 = tf.keras.layers.Conv2D(64,kernel_size=3,strides=1,padding='same')(block_2_2)
  
  block_2_2 = tf.keras.layers.BatchNormalization()(block_2_2)
  block_2_2 = tf.keras.layers.ReLU()(block_2_2)
  block_2_2 = tf.keras.layers.Conv2D(256,kernel_size=1,strides=1, padding='same')(block_2_2)

  stg1_blok_2_2 = tf.keras.layers.add([stg1_blok_2_1, block_2_2])

  ### identity_block 2 ###
  block_2_3 = tf.keras.layers.BatchNormalization()(stg1_blok_2_2)
  block_2_3 = tf.keras.layers.ReLU()(block_2_3)
  block_2_3 = tf.keras.layers.Conv2D(64,kernel_size=1,strides=1, padding='same')(block_2_3)
  
  block_2_3 = tf.keras.layers.BatchNormalization()(block_2_3)
  block_2_3 = tf.keras.layers.ReLU()(block_2_3)
  block_2_3 = tf.keras.layers.Conv2D(64,kernel_size=3,strides=1,padding='same')(block_2_3)
  
  block_2_3 = tf.keras.layers.BatchNormalization()(block_2_3)
  block_2_3 = tf.keras.layers.ReLU()(block_2_3)
  block_2_3 = tf.keras.layers.Conv2D(256,kernel_size=1,strides=1, padding='same')(block_2_3)

  stg1_blok_2_3 = tf.keras.layers.add([stg1_blok_2_2, block_2_3])

  squeeze = tf.keras.layers.GlobalAveragePooling2D()(stg1_blok_2_3)
  excitation = tf.keras.layers.Dense(units=256 / 16, activation='relu')(squeeze)
  excitation = tf.keras.layers.Dense(256,activation='sigmoid')(excitation)
  #excitation = tf.reshape(excitation, [-1,1,1,256])
  scale = tf.keras.layers.multiply([stg1_blok_2_3, excitation])


  block_1 = tf.keras.layers.BatchNormalization()(block_1)
  block_1 = tf.keras.layers.Conv2D(256,kernel_size=1,strides=1, padding='same')(block_1)

  short_cut_2 = tf.keras.layers.add([scale, block_1])

############ Block 3 #############
  ### 1 : conv_block ###
  block_3_1 = tf.keras.layers.BatchNormalization()(short_cut_2)
  block_3_1 = tf.keras.layers.ReLU()(block_3_1)
  block_3_1 = tf.keras.layers.Conv2D(128,kernel_size=1,strides=1, padding='same')(block_3_1)
  
  block_3_1 = tf.keras.layers.BatchNormalization()(block_3_1)
  block_3_1 = tf.keras.layers.ReLU()(block_3_1)
  block_3_1 = tf.keras.layers.Conv2D(128,kernel_size=3,strides=1,padding='same')(block_3_1)
  
  block_3_1 = tf.keras.layers.BatchNormalization()(block_3_1)
  block_3_1 = tf.keras.layers.ReLU()(block_3_1)
  block_3_1 = tf.keras.layers.Conv2D(512,kernel_size=1,strides=1, padding='same')(block_3_1)

  sh_cut_3 = tf.keras.layers.BatchNormalization()(short_cut_2)
  sh_cut_3 = tf.keras.layers.ReLU()(sh_cut_3)
  sh_cut_3 = tf.keras.layers.Conv2D(512,kernel_size=1,strides=1, padding='same')(sh_cut_3)

  stg1_blok_3_1 = tf.keras.layers.add([sh_cut_3, block_3_1])
  ### 2 : identity_block  ###
  block_3_2 = tf.keras.layers.BatchNormalization()(stg1_blok_3_1)
  block_3_2 = tf.keras.layers.ReLU()(block_3_2)
  block_3_2 = tf.keras.layers.Conv2D(128,kernel_size=1,strides=1, padding='same')(block_3_2)
  
  block_3_2 = tf.keras.layers.BatchNormalization()(block_3_2)
  block_3_2 = tf.keras.layers.ReLU()(block_3_2)
  block_3_2 = tf.keras.layers.Conv2D(128,kernel_size=3,strides=1,padding='same')(block_3_2)
  
  block_3_2 = tf.keras.layers.BatchNormalization()(block_3_2)
  block_3_2 = tf.keras.layers.ReLU()(block_3_2)
  block_3_2 = tf.keras.layers.Conv2D(512,kernel_size=1,strides=1, padding='same')(block_3_2)

  stg1_blok_3_2 = tf.keras.layers.add([stg1_blok_3_1, block_3_2])

  ### 3 : identity_block  ###
  block_3_3 = tf.keras.layers.BatchNormalization()(stg1_blok_3_2)
  block_3_3 = tf.keras.layers.ReLU()(block_3_3)
  block_3_3 = tf.keras.layers.Conv2D(128,kernel_size=1,strides=1, padding='same')(block_3_3)
  
  block_3_3 = tf.keras.layers.BatchNormalization()(block_3_3)
  block_3_3 = tf.keras.layers.ReLU()(block_3_3)
  block_3_3 = tf.keras.layers.Conv2D(128,kernel_size=3,strides=1,padding='same')(block_3_3)
  
  block_3_3 = tf.keras.layers.BatchNormalization()(block_3_3)
  block_3_3 = tf.keras.layers.ReLU()(block_3_3)
  block_3_3 = tf.keras.layers.Conv2D(512,kernel_size=1,strides=1, padding='same')(block_3_3)
  stg1_blok_3_3 = tf.keras.layers.add([stg1_blok_3_2, block_3_3])

 
  

  ### 4:identity_block  ###
  block_3_4 = tf.keras.layers.BatchNormalization()(stg1_blok_3_3)
  block_3_4 = tf.keras.layers.ReLU()(block_3_4)
  block_3_4 = tf.keras.layers.Conv2D(128,kernel_size=1,strides=1, padding='same')(block_3_4)
  
  block_3_4 = tf.keras.layers.BatchNormalization()(block_3_4)
  block_3_4 = tf.keras.layers.ReLU()(block_3_4)
  block_3_4 = tf.keras.layers.Conv2D(128,kernel_size=3,strides=1,padding='same')(block_3_4)
  
  block_3_4 = tf.keras.layers.BatchNormalization()(block_3_4)
  block_3_4 = tf.keras.layers.ReLU()(block_3_4)
  block_3_4 = tf.keras.layers.Conv2D(512,kernel_size=1,strides=1, padding='same')(block_3_4)

  stg1_blok_3_4 = tf.keras.layers.add([stg1_blok_3_3, block_3_4])


  squeeze = tf.keras.layers.GlobalAveragePooling2D()(stg1_blok_3_4)
  excitation = tf.keras.layers.Dense(units=512 / 16, activation='relu')(squeeze)
  excitation = tf.keras.layers.Dense(512,activation='sigmoid')(excitation)
  #excitation = tf.reshape(excitation, [-1,1,1,512])
  scale_3 = tf.keras.layers.multiply([stg1_blok_3_4, excitation])

  short_cut_2 = tf.keras.layers.BatchNormalization()(short_cut_2)
  short_cut_2 = tf.keras.layers.Conv2D(512,kernel_size=1,strides=1, padding='same')(short_cut_2)

  short_cut_3 = tf.keras.layers.add([scale_3, short_cut_2])



  ############ Block 4 #############
  ### 1 : conv_block ###
  block_4_1 = tf.keras.layers.BatchNormalization()(short_cut_3)
  block_4_1 = tf.keras.layers.ReLU()(block_4_1)
  block_4_1 = tf.keras.layers.Conv2D(256,kernel_size=1,strides=1, padding='same')(block_4_1)
  
  block_4_1 = tf.keras.layers.BatchNormalization()(block_4_1)
  block_4_1 = tf.keras.layers.ReLU()(block_4_1)
  block_4_1 = tf.keras.layers.Conv2D(256,kernel_size=3,strides=1,padding='same')(block_4_1)
  
  block_4_1 = tf.keras.layers.BatchNormalization()(block_4_1)
  block_4_1 = tf.keras.layers.ReLU()(block_4_1)
  block_4_1 = tf.keras.layers.Conv2D(1024,kernel_size=1,strides=1, padding='same')(block_4_1)

  sh_cut_4 = tf.keras.layers.BatchNormalization()(short_cut_3)
  sh_cut_4 = tf.keras.layers.ReLU()(sh_cut_4)
  sh_cut_4 = tf.keras.layers.Conv2D(1024,kernel_size=1,strides=1, padding='same')(sh_cut_4)

  stg1_blok_4_1 = tf.keras.layers.add([sh_cut_4, block_4_1])

  ### 2 :identity_block  ###
  block_4_2 = tf.keras.layers.BatchNormalization()(stg1_blok_4_1)
  block_4_2 = tf.keras.layers.ReLU()(block_4_2)
  block_4_1 = tf.keras.layers.Conv2D(256,kernel_size=1,strides=1, padding='same')(block_4_2)
  
  block_4_2 = tf.keras.layers.BatchNormalization()(block_4_2)
  block_4_2 = tf.keras.layers.ReLU()(block_4_2)
  block_4_2 = tf.keras.layers.Conv2D(256,kernel_size=3,strides=1,padding='same')(block_4_2)
  
  block_4_2 = tf.keras.layers.BatchNormalization()(block_4_2)
  block_4_2 = tf.keras.layers.ReLU()(block_4_2)
  block_4_2 = tf.keras.layers.Conv2D(1024,kernel_size=1,strides=1, padding='same')(block_4_2)
  stg1_blok_4_2 = tf.keras.layers.add([stg1_blok_4_1, block_4_2])

  ### 3 : identity_block  ###
  block_4_3 = tf.keras.layers.BatchNormalization()(stg1_blok_4_2)
  block_4_3 = tf.keras.layers.ReLU()(block_4_3)
  block_4_3 = tf.keras.layers.Conv2D(256,kernel_size=1,strides=1, padding='same')(block_4_3)
  
  block_4_3 = tf.keras.layers.BatchNormalization()(block_4_3)
  block_4_3 = tf.keras.layers.ReLU()(block_4_3)
  block_4_3 = tf.keras.layers.Conv2D(256,kernel_size=3,strides=1,padding='same')(block_4_3)
  
  block_4_3 = tf.keras.layers.BatchNormalization()(block_4_3)
  block_4_3 = tf.keras.layers.ReLU()(block_4_3)
  block_4_3 = tf.keras.layers.Conv2D(1024,kernel_size=1,strides=1, padding='same')(block_4_3)
  stg1_blok_4_3 = tf.keras.layers.add([stg1_blok_4_2, block_4_3])

  ### 4 : identity_block  ###
  block_4_4 = tf.keras.layers.BatchNormalization()(stg1_blok_4_3)
  block_4_4 = tf.keras.layers.ReLU()(block_4_4)
  block_4_4 = tf.keras.layers.Conv2D(256,kernel_size=1,strides=1, padding='same')(block_4_4)
  
  block_4_4 = tf.keras.layers.BatchNormalization()(block_4_4)
  block_4_4 = tf.keras.layers.ReLU()(block_4_4)
  block_4_4 = tf.keras.layers.Conv2D(256,kernel_size=3,strides=1,padding='same')(block_4_4)
  
  block_4_4 = tf.keras.layers.BatchNormalization()(block_4_4)
  block_4_4 = tf.keras.layers.ReLU()(block_4_4)
  block_4_4 = tf.keras.layers.Conv2D(1024,kernel_size=1,strides=1, padding='same')(block_4_4)
  stg1_blok_4_4 = tf.keras.layers.add([stg1_blok_4_3, block_4_4])

  ### 5 : :identity_block  ###
  block_4_5 = tf.keras.layers.BatchNormalization()(stg1_blok_4_4)
  block_4_5 = tf.keras.layers.ReLU()(block_4_5)
  block_4_5 = tf.keras.layers.Conv2D(256,kernel_size=1,strides=1, padding='same')(block_4_5)
  
  block_4_5 = tf.keras.layers.BatchNormalization()(block_4_5)
  block_4_5 = tf.keras.layers.ReLU()(block_4_5)
  block_4_5 = tf.keras.layers.Conv2D(256,kernel_size=3,strides=1,padding='same')(block_4_5)
  
  block_4_5 = tf.keras.layers.BatchNormalization()(block_4_5)
  block_4_5 = tf.keras.layers.ReLU()(block_4_5)
  block_4_5 = tf.keras.layers.Conv2D(1024,kernel_size=1,strides=1, padding='same')(block_4_5)
  stg1_blok_4_5 = tf.keras.layers.add([stg1_blok_4_4, block_4_5])


  squeeze = tf.keras.layers.GlobalAveragePooling2D()(stg1_blok_4_5)
  excitation = tf.keras.layers.Dense(units=1024 / 16, activation='relu')(squeeze)
  excitation = tf.keras.layers.Dense(1024,activation='sigmoid')(excitation)
  #excitation = tf.reshape(excitation, [-1,1,1,1024])
  scale_4 = tf.keras.layers.multiply([stg1_blok_4_5, excitation])


  short_cut_3 = tf.keras.layers.BatchNormalization()(short_cut_3)
  short_cut_3 = tf.keras.layers.Conv2D(1024,kernel_size=1,strides=1, padding='same')(short_cut_3)

  short_cut_4 = tf.keras.layers.add([scale_4, short_cut_3])


  ############ Block 5 #############
  ### conv_block ###
  block_5_1 = tf.keras.layers.BatchNormalization()(short_cut_4)
  block_5_1 = tf.keras.layers.ReLU()(block_5_1)
  block_5_1 = tf.keras.layers.Conv2D(512,kernel_size=1,strides=1, padding='same')(block_5_1)
  
  block_5_1 = tf.keras.layers.BatchNormalization()(block_5_1)
  block_5_1 = tf.keras.layers.ReLU()(block_5_1)
  block_5_1 = tf.keras.layers.Conv2D(512,kernel_size=3,strides=1,padding='same')(block_5_1)
  
  block_5_1 = tf.keras.layers.BatchNormalization()(block_5_1)
  block_5_1 = tf.keras.layers.ReLU()(block_5_1)
  block_5_1 = tf.keras.layers.Conv2D(2048,kernel_size=1,strides=1, padding='same')(block_5_1)

  sh_cut_5 = tf.keras.layers.BatchNormalization()(short_cut_4)
  sh_cut_5 = tf.keras.layers.ReLU()(sh_cut_5)
  sh_cut_5 = tf.keras.layers.Conv2D(2048,kernel_size=1,strides=1, padding='same')(sh_cut_5)

  stg1_blok_5_1 = tf.keras.layers.add([sh_cut_5, block_5_1])
  ### identity_block 1 ###
  block_5_2 = tf.keras.layers.BatchNormalization()(stg1_blok_5_1)
  block_5_2 = tf.keras.layers.ReLU()(block_5_2)
  block_5_2 = tf.keras.layers.Conv2D(512,kernel_size=1,strides=1, padding='same')(block_5_2)
  
  block_5_2 = tf.keras.layers.BatchNormalization()(block_5_2)
  block_5_2 = tf.keras.layers.ReLU()(block_5_2)
  block_5_2 = tf.keras.layers.Conv2D(512,kernel_size=3,strides=1,padding='same')(block_5_2)
  
  block_5_2 = tf.keras.layers.BatchNormalization()(block_5_2)
  block_5_2 = tf.keras.layers.ReLU()(block_5_2)
  block_5_2 = tf.keras.layers.Conv2D(2048,kernel_size=1,strides=1, padding='same')(block_5_2)

  stg1_blok_5_2 = tf.keras.layers.add([stg1_blok_5_1, block_5_2])

  ### identity_block 2 ###
  block_5_3 = tf.keras.layers.BatchNormalization()(stg1_blok_5_2)
  block_5_3 = tf.keras.layers.ReLU()(block_5_3)
  block_5_3 = tf.keras.layers.Conv2D(512,kernel_size=1,strides=1, padding='same')(block_5_3)
  
  block_5_3 = tf.keras.layers.BatchNormalization()(block_5_3)
  block_5_3 = tf.keras.layers.ReLU()(block_5_3)
  block_5_3 = tf.keras.layers.Conv2D(512,kernel_size=3,strides=1,padding='same')(block_5_3)
  
  block_5_3 = tf.keras.layers.BatchNormalization()(block_5_3)
  block_5_3 = tf.keras.layers.ReLU()(block_5_3)
  block_5_3 = tf.keras.layers.Conv2D(2048,kernel_size=1,strides=1, padding='same')(block_5_3)

  stg1_blok_5_3 = tf.keras.layers.add([stg1_blok_5_2, block_5_3])

  squeeze = tf.keras.layers.GlobalAveragePooling2D()(stg1_blok_5_3)
  excitation = tf.keras.layers.Dense(units=2048 / 16, activation='relu')(squeeze)
  excitation = tf.keras.layers.Dense(2048,activation='sigmoid')(excitation)
  #excitation = tf.reshape(excitation, [-1,1,1,2048])
  scale_5 = tf.keras.layers.multiply([stg1_blok_5_3, excitation])

  short_cut_4 = tf.keras.layers.BatchNormalization()(short_cut_4)
  short_cut_4 = tf.keras.layers.Conv2D(2048,kernel_size=1,strides=1, padding='same')(short_cut_4)

  short_cut_5 = tf.keras.layers.add([scale_5, short_cut_4])
############ Block 6 #############
  
  pooling = tf.keras.layers.GlobalAveragePooling2D()(short_cut_5)
  #pooling = tf.keras.layers.Dense(5,activation='relu')(pooling)
  model_output = tf.keras.layers.Dense(5,activation='softmax')(pooling)


  model = tf.keras.models.Model(model_input,model_output)
 
  return model 


# In[ ]:


image_w = 320
image_h = 256
n_classes = 5
model = se_ResNet50(image_w,image_h)
model.summary()

