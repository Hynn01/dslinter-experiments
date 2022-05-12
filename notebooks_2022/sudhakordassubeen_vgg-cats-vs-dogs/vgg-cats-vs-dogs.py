#!/usr/bin/env python
# coding: utf-8

# ---

# # <center>★ AI / ML Project - CNN Image Classification (Cats & Dogs) ★

# ---

# <center><img src="https://raw.githubusercontent.com/Masterx-AI/Project_Cat_vs_Dog_Classification_using_DL/main/wp5592665.jpg" style="width: 700px;"/>

# ---

# ### Description:
# 
# A convolutional neural network (CNN) is a type of artificial neural network used in image recognition and processing that is specifically designed to process pixel data.
# 
# CNNs are powerful image processing, artificial intelligence (AI) that use deep learning to perform both generative and descriptive tasks, often using machine vison that includes image and video recognition, along with recommender systems and natural language processing (NLP).
# 
# The dataset consists of multiple samples of images of dogs & cats. Can you build an optimal Deep Neural Network Model to classify them?
# 
# 
# ### Acknowledgement: 
# The dataset is referred from tensorflow datasets.
# 
# ### Objective:
# - Understand the Dataset & perform necessary Preprocessing.
# - Design a Neural Network Architecture to classify Cats & Dogs.

# ---

# # <center> Stractegic Plan of Action:

# **We aim to solve the problem statement by creating a plan of action, Here are some of the necessary steps:**
# 
# Data Collection
# 
# Data Visualization
# 
# Data Pre-processing
# 
# Deep Neural Network Architecture

# ---

# # <center> Data Collection

# In[ ]:


#Importing the basic librarires

import math
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10,6]

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

import warnings 
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 50)


# In[ ]:


#Defining constants

random_seed=123
batch_size = 64
img_height = 128
img_width = 128


# In[ ]:


#Getting the dataset information

import tensorflow_datasets as tfds
builder = tfds.builder('cats_vs_dogs')
info = builder.info


# In[ ]:


# Getting the Class label
class_names = builder.info.features['label'].names
class_names


# In[ ]:


# Load the cats vs dogs dataset

train_ds, validation_ds, test_ds = tfds.load(
    "cats_vs_dogs",
    # Reserve 10% for validation and 10% for test
    split=["train[:80%]", "train[80%:90%]", "train[90%:100%]"],
    as_supervised=True,  # Include labels
)


# In[ ]:


# Shape of the image
for image_batch, labels_batch in train_ds.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())


# In[ ]:


# Image sample

image, label = next((iter(train_ds)))
print(label)
print(image)
plt.imshow(image.numpy().astype("uint8"))


# ---

# ---

# # <center>  Data Preprocessing and Visualization

# In[ ]:


# function for resizing and rescaling the data
def resize(image, label):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, [img_height, img_width])
  return image, label

def rescale(image, label):
  image = (image / 255.0)
  return image, label

# functions for augmenting the data
def img_augment(image, label):
    
    #Random brightness
    #image = tf.image.stateless_random_brightness(image, max_delta=0.25, seed = (0,1))
    # Random contrast
   # image = tf.image.stateless_random_contrast(image, lower=0.1, upper=0.9, seed = (0,1))
    # Use tensorflow addons to randomly rotate images
    deg = np.random.uniform(-20,20)
    image = tfa.image.rotate(image, deg)
    #Random crop
    #image = tf.image.stateless_random_crop(image, size=[128, 128, 3])
    # Random left right flip
    image = tf.image.random_flip_left_right(image)
    #Random fil up down
    image = tf.image.stateless_random_flip_up_down(image, seed = (0,1))
    #Random Saturation
    #image = tf.image.stateless_random_saturation(image, lower = .1, upper = .9, seed = (0,1))

    return image, label


# In[ ]:


#Preprocessing the Dataset

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).map(resize).map(rescale).map(img_augment).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
val_ds = validation_ds.cache().shuffle(1000).map(resize).map(rescale).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).map(resize).batch(batch_size).prefetch(buffer_size=AUTOTUNE)


# In[ ]:


# shape of image after processing

for image_batch, labels_batch in test_ds.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())


# In[ ]:


# Visualizing the data from test split

plt.figure(figsize=(20, 15))
for image_batch, labels_batch in test_ds.take(1):
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[labels_batch[i]])
        plt.axis("off")


# In[ ]:


# function for plotting the model history

def perf_plot(history):
  train_accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  train_loss = history.history['loss']
  val_loss = history.history['val_loss']

  

  fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

  ax[0].set_title('Training Accuracy vs. Epochs')
  ax[0].plot(train_accuracy, 'o-', label='Train Accuracy')
  ax[0].plot(val_accuracy, 'o-', label='Validation Accuracy')
  ax[0].set_xlabel('Epochs')
  ax[0].set_ylabel('Accuracy')
  ax[0].legend(loc='best')

  ax[1].set_title('Training/Validation Loss vs. Epochs')
  ax[1].plot(train_loss, 'o-', label='Train Loss')
  ax[1].plot(val_loss, 'o-', label='Validation Loss')
  ax[1].set_xlabel('Epochs')
  ax[1].set_ylabel('Loss')
  ax[1].legend(loc='best')


  plt.tight_layout()
  plt.show()


# In[ ]:


# Function for visualizing the predicted result

def prediction(model):
  for image_batch, labels_batch in test_ds.take(1):
        plt.figure(figsize=(20, 30))
        for i in range(24):
          ax = plt.subplot(6, 4, i + 1)
          image = image_batch[i].numpy().astype('uint8')
          label = labels_batch[i].numpy()
          plt.imshow(image)
          Actual_label = (class_names[label])
          pred = model.predict(image_batch)
         # print(np.argmax(pred[0]))
          Predicted_label = class_names[np.argmax(pred[i])]
          plt.title(f"Actual: {Actual_label}, \nPredicted: {Predicted_label}")
          plt.axis('off')
        
    
def wrong_prediction(model):
  for image_batch, labels_batch in test_ds.take(1):
        plt.figure(figsize=(20, 35))
        for i in range(24):
          ax = plt.subplot(6, 4, i + 1)
          image = image_batch[i].numpy().astype('uint8')
          nmlabel = labels_batch[i].numpy()

          Actual_label = (class_names[nmlabel])
          pred = model.predict(image_batch)
          pdlabel = np.argmax(pred[i])
          #print(np.argmax(pred[i]))
          Predicted_label = class_names[pdlabel]
          if nmlabel != pdlabel:
            plt.imshow(image)
            plt.title(f"Actual: {Actual_label}, \nPredicted: {Predicted_label}")
            plt.axis('off')

            
              


# ---

# # <center>  Deep Neural Network Architecture

# # **CNN_Model: 3**

# In[ ]:


get_ipython().system('pip install -U tensorboard_plugin_profile')


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
early = EarlyStopping(monitor="loss", mode="min",min_delta = 0,
                          patience = 10,
                          verbose = 1,
                          restore_best_weights = True)
learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='CNN_Model_3.h5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

from datetime import datetime
# Create a TensorBoard callback
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

# You'll need to edit the profile_batch here so that it profiles 10 batches
# in the second epoch of your training
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '200,300')

callbacks_list = [ early, learning_rate_reduction,model_checkpoint_callback, tboard_callback ]


# In[ ]:


# Develop a sequential model using tensorflow keras
from tensorflow.keras import models
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(img_height, img_width,3)),
    tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu" ),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu" ),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])


model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
             optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
             metrics=['accuracy'])

model.summary()


# In[ ]:


epochs=10
history_3 = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks_list)


# In[ ]:


#Making the prediction
score = model.evaluate(test_ds)
score


# In[ ]:


#Plotting results of Iter-1

perf_plot(history_3)


# In[ ]:


# predicted result

prediction(model)


# In[ ]:


# wrong prediction made by model
wrong_prediction(model)


# In[ ]:


# Load the TensorBoard notebook extension.
get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[ ]:


# Launch TensorBoard and navigate to the Profile tab to view performance profile
get_ipython().run_line_magic('tensorboard', '--logdir=logs')


# # **CNN-Model:2**

# In[ ]:





# In[ ]:


epochs=10
history_3 = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks_list)


# In[ ]:


#Making the prediction
score = model.predict(test_ds)
score


# In[ ]:


#Plotting results of Iter-1

perf_plot(history_3)


# In[ ]:


# Develop a sequential model using tensorflow keras

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(img_height, img_width,3)),
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu" ),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu" ),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu" ),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
             optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
             metrics=['accuracy'])

model.summary()


# In[ ]:


epochs=20
history_6 = model.fit(train_ds, validation_data=val_ds, epochs=epochs)


# In[ ]:


#Plotting results of Iter-1

perf_plot(history_6)


# ---

# # <center>5. Outcomes & Project Conclusion

# ### Here are some of the key outcomes of the project:
# - The  Image Dataset was large enough with 6k training samples & 2k testing samples.
# - Visualising the image samples & it's distribution, helped us to get some insights into the dataset.
# - The classes were not imbalanced, hence we did not perform data augmentation.
# - The performance of basic Vanilla Neural Network Architecture was pretty good, as the results obtained had 97.4% of Training Accuracy & 81.7% of Testing Accuracy.
# - Further improvisations can include the usage of CNN layers, adding dropouts & batch-normalization layers, or even utilizing the Transfer Learning Methodology to train Prominent Models like VGG, ResNet, AlexNet, GoogleNet, Etc.

# In[ ]:


#<<<--------------------------------------THE END---------------------------------------->>>

