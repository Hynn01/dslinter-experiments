#!/usr/bin/env python
# coding: utf-8

# ###### Kaggle Project - MSDS686 - Amanda Kimball April 2020
# 
# **Overview of the assignment:**
# 
# In this project I will overview the dataset 100-bird-species. I will explore the data, transform it into a format for modeling, and then create classification models to predict the species of birds given a picture of a bird. My models will  
# 
# **Description of the data:**
# 
# For this assignment, I choose to create a deep learning CNN model for the classification of 180 bird species (which was continuously increasing so I adjusted the code appropriately). This dataset is located here: https://www.kaggle.com/gpiosenka/100-bird-species. The first step is to add the data at right to your kaggle input. The dataset consists of 2 directories 175 and 180 - (Now changed to a single directory - continuously increasing). Directory 175 is not explored during my analysis, but has the same structure as directory 180. 180 has train, test, valid, consolidated, and a predictor test set. I used the first 3 directories for training my model, validating my model and then a final test of the model. Keeping these 3 datasets seperate is a key part of any good modeling method.  The consolidated file puts all of the photos into one folder, and could verywell be used for a larger trainig set with seperate test/valid sets or to split the data differently than the original split. Gerry Piosenka (2020) indicates that he compiled the data from photos on the internet, evaluated them so that there are no duplicates, cropped them so that the bird was 50% of the image, and then sized them to the 224x224x3 jpg format. The jpg files are all in thier 
# label/species name directory, which you will see that I use to find my favorite bird -- the hummingbird -- early in this tutorial. The author notes a bias in photos of males 80% to females. As an avid bird watcher, I can attest that male birds have better coloring and are easier to speciate. Most female hummingbirds for example are easier to decipher based on bird calls and size. I hypothsize, that a grey-scale evaluation would be better for predicting female birds, but will leave that for future research. 
# 
# **Summary of Methods:**
# 
# For each model, I used the imagedatagenerator in keras to move the data from the 100-bird-species/180 directory inot the model. I created each model as described below. Each model is compiled using a batchsize/samples ratio for the epochs_per_step (on training and validation steps). The compiled models are fit to the generator data for epochs required with patience of 5 (max was 50 but none of the model fit executions went for that long). I graphed the validation and training data loss and accuracy as a function of epochs. Finally, the third dataset - test - was compared to the model to give a true accuracy and loss value for each fit model.
# 
# **Summary of Model:**
# 
# I present my best from scratch convolution model and the VGG16 model after a few iterations (and MobileNet which performed much better). The different methods attempted are detailed at the top with hash for future investigation. I did find that the SGD optimizer did better than ADAM. 
# 
# **Analysis of Results:**
# 
# The best accuracy achieved was 86% (94% achieved with MobileNet).

# In[ ]:


#Used to make data more uniform across screen.
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# In[ ]:


#Import packages used here:
# for initial data exploration:
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from IPython.display import Image, display
import random
import math

#For modeling and model viewing. 
import tensorflow as tf
from keras.preprocessing.image import load_img,img_to_array,ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model 
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation,Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical #Image generator used for transformation to categorical
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras import backend, models
#from sklearn.model_selection import train_test_split  #could have used on the consolidated file.
from sklearn.metrics import confusion_matrix

from tensorflow.keras.applications import VGG16, MobileNet
#from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input


# In[ ]:


#This will setup my directories for all of the data files in the 100-bird-species dataset. 
BASE_DIR = '/kaggle/input/100-bird-species'
print('BASE_DIR contains ', os.listdir(BASE_DIR))
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALIDATION_DIR = os.path.join(BASE_DIR, 'valid')
TEST_DIR = os.path.join(BASE_DIR, 'test')


# In[ ]:


#This will establish the prediction groups for the model.
CATEGORIES = os.listdir(TRAIN_DIR)
print(str(len(CATEGORIES)),'CATEGORIES are ', CATEGORIES)

Category_count = len(CATEGORIES)


# I choose to load a humming bird image as my favorite bird. I also displayed the shape of the image so that I can use it in my model.

# In[ ]:


#Load an image and determine image shape for analysis.
IMAGE = load_img("/kaggle/input/100-bird-species/train/ANNAS HUMMINGBIRD/025.jpg")
plt.imshow(IMAGE)
plt.axis("off")
plt.show()

IMAGEDATA = img_to_array(IMAGE)
SHAPE = IMAGEDATA.shape
print('Figures are ', SHAPE)



# I'll create instances of ImageDataGenerators. One for all of the data being processed and more if I decide to augment my training data.

# In[ ]:


#This will be used on training, test, and valid data
General_datagen = ImageDataGenerator(rescale=1./255, )


# The directories are not direct links to the data so I used the IMAGEDATAGENERATOR in Keras to consolidate the images for each train/test/validation set. I left the defaults as follows: batch_size = 32, class_mode = 'categorical', shuffle = TRUE(in flow).

# In[ ]:


train_data = General_datagen.flow_from_directory(TRAIN_DIR, target_size=(224,224))
print('data groups:', len(train_data)) #Will be used to determine steps_per_epoch in my models.
Train_groups = len(train_data)
validation_data = General_datagen.flow_from_directory(VALIDATION_DIR, target_size=(224,224),)
image_qty = len(validation_data.filenames)
print('data groups:', len(validation_data))
print('validation image qty:',str(image_qty))
Valid_groups = len(validation_data)
test_data = General_datagen.flow_from_directory(TEST_DIR, target_size=(224,224),)
print('data groups:', len(test_data))


# So to make certain my new datasets still had images and label seperation, I printed a few more images from the test set.

# In[ ]:


#create seperate labels for images 
def label_images2(DIR, dataset):
    label = []
    image = []
    j=0
    for i in range (0,30):
        j = random.randint(0, len(dataset.filenames))
        label.append(dataset.filenames[j].split('/')[0])
        image.append(DIR + '/' + dataset.filenames[j])
    return [label,image]

#plot the random images.
y,x = label_images2(TEST_DIR, test_data)

for i in range(0,6):
    X = load_img(x[i])
    plt.subplot(2,3,+1 + i)
    plt.axis(False)
    plt.title(y[i], fontsize=8)
    plt.imshow(X)
plt.show()


# In[ ]:


#This was my Sequential model from the CIFAR10 dataset - seemed like a good starting point. -65% accuracy
#With 2 epochs I got: Test loss: 2.3443613751181243 Test accuracy: 0.4788889
#With 50 epochs/stopped at 13 Test loss: 1.7568193797407479, Test accuracy: 0.5733333..Not so great. I will move on to pretrained models.
#Increased from 32 to 64 nodes in CONV2D layers: Test loss: 4.270853807186258, Test accuracy: 0.5377778
#Changed from Adam to sgd for optimizer:Test loss: 1.4400342908398858, Test accuracy: 0.65444446 - 65%
backend.clear_session()
model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same',input_shape=SHAPE)) #224X224
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3))) #222x222
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #111x111
model.add(BatchNormalization())
model.add(Dropout(0.35)) #Doesn't appear to be working in the model summary.

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization()) 

model.add(Conv2D(64, (3, 3))) #109x109
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #54x54
model.add(BatchNormalization())
model.add(Dropout(0.35)) #64 --> 42

model.add(Conv2D(64, (3, 3), padding='same')) #54x54
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Flatten()) 
model.add(Dropout(0.5)) 
model.add(Dense(512)) 
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(Category_count)) #Updated for number of classes
model.add(Activation('softmax'))

model.summary()

#Compile
model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'])
#fit model
history = model.fit_generator( 
    train_data, 
    steps_per_epoch = Train_groups, 
    epochs = 50,
    validation_data = validation_data,
    validation_steps = Valid_groups,
    verbose = 1,
    callbacks=[EarlyStopping(monitor='val_accuracy', patience = 5, restore_best_weights = True),
               ReduceLROnPlateau(monitor = 'val_loss', factor = 0.7, #0.2 to 0.5 dropped to fast 0.7
                                 patience = 2, verbose = 1)])


# In[ ]:


#plot accuracy vs epoch
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot loss values vs epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluate against test data.
scores = model.evaluate(test_data, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# In[ ]:


#Let's try the vgg16 - 86% accuracy
#Initial run with 2 epochs est loss: 27.959821766820447, Test accuracy: 0.40666667
# Increased to 50 epochs to Test loss: 26.42553789862271, Test accuracy: 0.79888886 - 80%
# Added pooling Max to the vgg16 model -78%
# Removedpooling and add sgd in place of adam optimizer: Test loss: 0.5881293734599804, Test accuracy: 0.8611111
backend.clear_session()


#Bring in the imagenet dataset training weights for the VGG16 CNN model, remove the classification, the default shape is correct (3,224,224) for my purposes.
base_vgg16 = VGG16(weights = 'imagenet', include_top = False, input_shape = SHAPE)
base_vgg16.trainable = False # Freeze the VGG16 weights.

model = Sequential()
model.add(base_vgg16)

model.add(Flatten()) #1024#model.add(Dense(256)) 
model.add(Activation('relu'))
model.add(Dense(Category_count)) 
model.add(Activation('softmax'))

model.summary()

#Compile
model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'])
history = model.fit_generator( 
    train_data, 
    steps_per_epoch = Train_groups, 
    epochs = 50,
    validation_data = validation_data,
    validation_steps = Valid_groups,
    verbose = 1,
    callbacks=[EarlyStopping(monitor='val_accuracy', patience = 5, restore_best_weights = True),ReduceLROnPlateau(monitor = 'val_loss', factor = 0.7, #0.2 to 0.5 dropped to fast 0.7
                                 patience = 2, verbose = 1)])  


# In[ ]:


#plot accuracy vs epoch
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot loss values vs epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluate against test data.
scores = model.evaluate(test_data, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# In[ ]:


#Let's try the mobilenet with ReduceLROnPlateau - 93% accuracy
backend.clear_session()

#Bring in the imagenet dataset training weights for the Mobilenet CNN model.
#Remove the classification top.
base_mobilenet = MobileNet(weights = 'imagenet', include_top = False, 
                           input_shape = SHAPE)
base_mobilenet.trainable = False # Freeze the mobilenet weights.

model = Sequential()
model.add(base_mobilenet)

model.add(Flatten()) 
model.add(Activation('relu'))
model.add(Dense(Category_count)) 
model.add(Activation('softmax'))

model.summary()

#Compile
model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.001, 
                                                  momentum=0.9, nesterov=True),
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'])
#fit model
history = model.fit_generator( 
    train_data, 
    steps_per_epoch = Train_groups, 
    epochs = 50,
    validation_data = validation_data,
    validation_steps = Valid_groups,
    verbose = 1,
    callbacks=[EarlyStopping(monitor = 'val_accuracy', patience = 5, 
                             restore_best_weights = True),
               ReduceLROnPlateau(monitor = 'val_loss', factor = 0.7, #0.2 to 0.5 dropped to fast 0.7
                                 patience = 2, verbose = 1)]) 
                # left verbose 1 so I could see the learning rate decay


# In[ ]:


#plot accuracy vs epoch
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot loss values vs epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluate against test data.
scores = model.evaluate(test_data, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# Let's use this model to look at how augmentation might improve the accuracy.

# In[ ]:


#This would only be applied to my training data. looking at the data rotated within 40 degrees would give more data without a change to the mostly vertically sitting birds. 
Augment_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, # Rotate the images randomly by 40 degrees
    width_shift_range=0.2, # Shift the image horizontally by 20%
    height_shift_range=0.2, # Shift the image veritcally by 20%
    zoom_range=0.2, # Zoom in on image by 20% 
    horizontal_flip=True, # Flip image horizontally 
    fill_mode='nearest') 
Augmentation_train = Augment_datagen.flow_from_directory(TRAIN_DIR, target_size=(224,224))

print('data groups:', len(Augmentation_train)) #Will be used to determine steps_per_epoch in my models.


# An API model is generally used when you have multiple inputs or multiple outputs. This example dataset doesn't appear to have that type of need directly, but I created one to show it is an option.

# In[ ]:


#Let's try the mobilenet with ReduceLROnPlateau with augmentation - 93% accuracy
backend.clear_session()

#Bring in the imagenet dataset training weights for the Mobilenet CNN model.
#Remove the classification top.
base_mobilenet = MobileNet(weights = 'imagenet', include_top = False, 
                           input_shape = SHAPE)
base_mobilenet.trainable = False # Freeze the mobilenet weights.

model = Sequential()
model.add(base_mobilenet)

model.add(Flatten()) 
model.add(Activation('relu'))
model.add(Dense(Category_count)) 
model.add(Activation('softmax'))

model.summary()

#Compile
model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.001, 
                                                  momentum=0.9, nesterov=True),
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'])
#fit model
history = model.fit_generator( 
    Augmentation_train, 
    steps_per_epoch = Train_groups, 
    epochs = 50,
    validation_data = validation_data,
    validation_steps = Valid_groups,
    verbose = 1,
    callbacks=[EarlyStopping(monitor = 'val_accuracy', patience = 5, 
                             restore_best_weights = True),
               ReduceLROnPlateau(monitor = 'val_loss', factor = 0.7, #0.2 to 0.5 dropped to fast 0.7
                                 patience = 2, verbose = 1)]) 
                # left verbose 1 so I could see the learning rate decay


# In[ ]:


#plot accuracy vs epoch
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot loss values vs epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluate against test data.
scores = model.evaluate(test_data, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

