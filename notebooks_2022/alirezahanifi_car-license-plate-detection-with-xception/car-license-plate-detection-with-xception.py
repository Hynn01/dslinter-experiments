#!/usr/bin/env python
# coding: utf-8

# - [Import Libraries](#1)
# - [Reading the Dataset](#2)
# - [Data Pre-processing](#3)
# - [Modeling](#4)
# - [Conclusions](#5)
# - [References](#6)

# # Import Libraries <a id = '1'></a>

# In[ ]:


import cv2
import glob
from lxml import etree
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

import tensorflow as tf 
from tensorflow import keras 
from keras.layers import BatchNormalization, Conv2D, Dense,Dropout, Flatten, GlobalAveragePooling2D, MaxPool2D, ReLU
from tensorflow.keras.applications import Xception

import warnings
warnings.filterwarnings("ignore")


# # Reading the Dataset <a id = '2'></a>

# - <a href = 'https://www.kaggle.com/datasets/andrewmvd/car-plate-detection'>Link to the dataset in the Kaggel.</a>
# 
# <i>"This dataset contains 433 images with bounding box annotations of the car license plates within the image.
# Annotations are provided in the PASCAL VOC format."</i>

# In[ ]:


img_list = [] 
annot_list = []

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if os.path.join(dirname, filename)[-3:]!="xml":
            img_list.append(os.path.join(dirname, filename))
        else:
            annot_list.append(os.path.join(dirname, filename))
            
img_list.sort()
annot_list.sort()


# # Data Pre-processing <a id = '3'></a>

# In[ ]:


def get_annotation(annotation):
    
    tree = etree.parse(annotation)
    
    for dim in tree.xpath("size"):
        width = int(dim.xpath("width")[0].text)
        height = int(dim.xpath("height")[0].text)
        
    for dim in tree.xpath("object/bndbox"):
        xmin = int(dim.xpath("xmin")[0].text)
        ymin = int(dim.xpath("ymin")[0].text)
        xmax = int(dim.xpath("xmax")[0].text)
        ymax = int(dim.xpath("ymax")[0].text)
        
    return [int(xmax), int(ymax), int(xmin), int(ymin), int(width), int(height)]

annot_coor = []

for annotation in annot_list:
    annot_coor.append(get_annotation(annotation))


# In[ ]:


images_width = [i[4] for i in annot_coor]
images_height = [i[5] for i in annot_coor]

j_plot = sns.jointplot(x = images_width, y = images_height, kind = 'reg', height = 8)
j_plot.set_axis_labels('width', 'height', fontsize = 16)


# In[ ]:


img_width = 400
img_height = 280


# In[ ]:


X = []

for i in img_list:
    img = cv2.imread(i) 
    img = cv2.resize(img, (img_width, img_height))
    X.append(np.array(img))

X_arr = np.array(X)
print(f'Shape of images(X): {X_arr.shape} (m, height, width, channels)')


# In[ ]:


def resize_annotation(annotation):
    
    tree = etree.parse(annotation)
    
    for dim in tree.xpath("size"):
        width = int(dim.xpath("width")[0].text)
        height = int(dim.xpath("height")[0].text)
        
    for dim in tree.xpath("object/bndbox"):
        xmin = int(dim.xpath("xmin")[0].text)/(width/img_width)
        ymin = int(dim.xpath("ymin")[0].text)/(height/img_height)
        xmax = int(dim.xpath("xmax")[0].text)/(width/img_width)
        ymax = int(dim.xpath("ymax")[0].text)/(height/img_height)
        
    return [int(xmax), int(ymax), int(xmin), int(ymin)]


# In[ ]:


y = []

for annotation in annot_list:
    y.append(resize_annotation(annotation))
    
y_arr = np.array(y)
print(f'Shape of annotations(y): {y_arr.shape} (m, bbox[xmax, ymax, xmin, ymin])')


# In[ ]:


plt.figure(figsize = (20, 15))

for num, i in enumerate(np.random.randint(X_arr.shape[0],size = 9)):
    
    plt.subplot(3, 3, num + 1)
    
    img_rec = cv2.rectangle(
        X_arr[i], #image
        (y_arr[i][0], y_arr[i][1]), #start_point
        (y_arr[i][2], y_arr[i][3]), #end_point
        (255, 0, 0), #color
        2 #thickness
    )
    plt.imshow(img_rec)
    plt.axis('off')


# In[ ]:


X_arr_norm = X_arr / 255.
y_arr_norm = y_arr / float(img_width)

x_train, x_test, y_train, y_test = train_test_split(X_arr_norm, y_arr_norm, test_size = 0.2)
print(f'Shape of x_train: {x_train.shape}')
print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of x_test: {x_test.shape}')
print(f'Shape of y_test: {y_test.shape}')


# # Modeling <a id = '4'></a>

# In[ ]:


base_xception = Xception(
    input_shape = (img_height, img_width, 3),
    include_top = False,
    weights = 'imagenet',
    )


# In[ ]:


x = base_xception.output
x = MaxPool2D(pool_size = 3)(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(64, activation = 'relu')(x)
x = BatchNormalization(axis = -1)(x)
x = Dropout(0.2)(x)
outputs_xception = Dense(4, activation = 'sigmoid')(x)

xception_fine = keras.Model(inputs = base_xception.inputs, outputs = outputs_xception, name = 'xception_pretrained')

xception_fine.summary()


# In[ ]:


for i, layer in enumerate(xception_fine.layers):
    print(f'Layer number {i}: {layer.name}')    


# In[ ]:


for layer in xception_fine.layers[:116]:
    layer.trainable = False
    
for layer in xception_fine.layers[116:]:
    layer.trainable = True


# In[ ]:


xception_fine.compile(
    loss =  'mse',
    optimizer = 'adam'
)

epochs = 50

history_xception_fine = xception_fine.fit(
    x_train,
    y_train,
    batch_size = 64,
    epochs = epochs,
    shuffle = True,
    validation_split = 0.2
)


# In[ ]:


def plot_history(history_of_model):
    plt.figure(figsize = (20, 5))
    
    plt.subplot(1, 1, 1)
    plt.title('Loss on training and validation data')
    sns.lineplot(x = range(1, epochs + 1), y = history_of_model.history['loss'])
    sns.lineplot(x = range(1, epochs + 1), y = history_of_model.history['val_loss'])
    plt.xlabel('epochs')
    plt.xticks(list(range(1, epochs + 1))[::2])
    plt.legend(['training loss', 'validation loss'], loc = 'upper left')
    
    plt.show() 
    
plot_history(history_xception_fine)  


# In[ ]:


test_xception_loss = xception_fine.evaluate(x_test, y_test)
print(f'Loss on testing data with a Xception model: {test_xception_loss:0.4}')


# In[ ]:


y_pred = (xception_fine.predict(x_test) * img_width).astype('int')


# In[ ]:


y_predd = (xception_fine.predict(x_test[16][np.newaxis, ...]) * img_width).astype('int')

print(y_test[16] * 400)
y_predd


# In[ ]:


plt.figure(figsize = (20, 15))

for num, i in enumerate(np.random.randint(x_test.shape[0],size = 9)):
    
    plt.suptitle('Ground Truth bbox: Green, Predicted bbox: Red', size = 20)
   
    plt.subplot(3, 3, num + 1)
    cv2.rectangle(
        x_test[i], #image
        (int(y_test[i][0] * img_width), int(y_test[i][1] * img_width)), #start_point
        (int(y_test[i][2] * img_width), int(y_test[i][3] * img_width)), #end_point
        (0, 255, 0), #color: green
        3 #thickness
    )
    plt.text(
        int(y_test[i][0] * img_width),
        int(y_test[i][1] * img_width) + 8,
        'GT BB',
        fontsize = 14,
        color = 'g'
    )
    
    cv2.rectangle(
        x_test[i], #image
        (y_pred[i][0], y_pred[i][1]), #start_point
        (y_pred[i][2], y_pred[i][3]), #end_point
        (255, 0, 0), #color: red
        3 #thickness
    )
    plt.text(
        y_pred[i][2],
        y_pred[i][3] - 8,
        'P BB',
        fontsize = 14,
        color = 'r'
    )    
    
    plt.imshow(x_test[i])
    plt.axis('off')


# # Conclusions <a id = '5'></a>

# From the plots above, it is clear that this model does not recognize license plates well. Finding metrics and losses is currently one of the biggest challenges. Yolo must be used in the next step!

# # References <a id = '6'></a>

# - <a href = 'https://www.kaggle.com/code/mclikmb4/vehicle-license-plate-detection-vgg16'>Vehicle License Plate Detection | VGG16</a>
