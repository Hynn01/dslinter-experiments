#!/usr/bin/env python
# coding: utf-8

# # TensorFlow Image Classification Guide
# 
# **In this notebook, I will demonstrate how to create image classification models with Tensorflow.**
# 
#  --------------------------------------------------------------------------------------------------
# **Notebook Prerequisites:**
# - Python                               > https://www.kaggle.com/learn/python
# - Data Visualization with Matplotlib   > https://www.kaggle.com/learn/data-visualization
# - Basic Linear Algebra                 > https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
# - Basic Knowledge of Machine Learning  > https://www.kaggle.com/learn/intro-to-machine-learning
# 
# 
# ---------------------------------------------------------------------------------------------------
# 
# 
# **This notebook will cover the following topics:**
# - Loading datasets from the Tensorflow Datasets API
# - Creating an `ImageDataGenerator` to process images
# - Performing image augmentation in memory to reduce overfitting
# - Building deep convolutional neural networks from scratch
# - Evaluating a model's performance
# - Creating custom callbacks
# - Using premade models with transfer learning
# 
# 
# 
# *Note: You should research anything in this notebook that you do not understand. Some links will be provided*

# # Load Horses or Humans Dataset
# 
# We will load the `horses_or_humans` dataset from the Tensorflow Dataset API. Datasets loaded in this way are already processed in a way where they can be sent directly to a model. However, because images are rarely ever in such a nice format, we will save the images into folders to replicate real-world conditions.
# 
# In order to use the `ImageDataGenerator`, the images need to be organized in a specific folder organization (shown below).
# 
#                       Data                      <-----------------Root Directory 
#                        /\
#                       /  \
#                      /    \
#                     /      \
#                    /        \
#                Train        Test                <-----------------Data Subset Directory (train/val/test)
#                /   \         /  \
#               /     \       /    \
#              /       \     /      \
#         Horses   Humans  Horses   Humans        <-----------------Class Directory (name of data classes)

# In[ ]:


#Import Libraries
import tensorflow as tf
import tensorflow_datasets as tfds #Dataset API
import numpy as np #Linear Algebra
import matplotlib.pyplot as plt #Data visualization
import os #Manipulate Files
from PIL import Image #Manipulate Images

import warnings
warnings.filterwarnings('ignore') #ignores warnings

#Make sure Tensorflow is version 2.0 or higher
print('Tensorflow Version:', tf.__version__)


# In[ ]:


#Makes Folders to store images
os.makedirs('Data', exist_ok=True)
os.makedirs('Data/Train/Horses', exist_ok=True)
os.makedirs('Data/Train/Humans', exist_ok=True)
os.makedirs('Data/Test/Horses', exist_ok=True)
os.makedirs('Data/Test/Humans', exist_ok=True)

base_path = os.getcwd()
horse_counter = 0
human_counter = 0
#The below code will save the dataset images into the folders created above
#Note: This step is not required when using Tensorflow datasets but will be required when 
# using datasets that are in the wild or possibly on Kaggle
#see horse or humans doc here ->https://www.tensorflow.org/datasets/catalog/horses_or_humans
for i, dataset in enumerate(tfds.load('horses_or_humans', split=['train', 'test'])):
    if i==0: #training set
        set_path = os.path.join(base_path, 'Data/Train')
    else: #test set
        set_path = os.path.join(base_path, 'Data/Test')
        
    for row in list(dataset):
        im = Image.fromarray(row['image'].numpy())
        if row['label'] == 0: #0 is horse and 1 is human
            class_path = os.path.join(set_path, 'Horses')
            file_path = os.path.join(class_path, "horse_{}.jpeg".format(horse_counter))
            horse_counter += 1
        elif row['label'] == 1: #0 is horse and 1 is human
            class_path = os.path.join(set_path, 'Humans')
            file_path = os.path.join(class_path, "human_{}.jpeg".format(horse_counter))
            human_counter += 1
        im.save(file_path) #saves the image in the proper folder


# In[ ]:


print('Number of Horse Images in the Training Set:', len(os.listdir('Data/Train/Horses')))
print('Number of Human Images in the Training Set:', len(os.listdir('Data/Train/Humans')))
print('\n')
print('Number of Horse Images in the Testing Set:', len(os.listdir('Data/Test/Horses')))
print('Number of Human Images in the Testing Set:', len(os.listdir('Data/Test/Humans')))


# In[ ]:


#Print Sample Images
horse_imgs = []
human_imgs = []

for i in range(5):
    horse_im = Image.open(os.path.join('Data/Train/Horses', os.listdir('Data/Train/Horses')[i]))
    human_im = Image.open(os.path.join('Data/Train/Humans', os.listdir('Data/Train/Humans')[i]))
    horse_imgs.append(horse_im)
    human_imgs.append(human_im)
    

plt.rcParams["figure.figsize"] = (20,5)
fig, axs = plt.subplots(2, 5)
for i in range(2):
    for j in range(5):
        if i == 0:
            axs[i, j].imshow(horse_imgs[j])
        else:
            axs[i, j].imshow(human_imgs[j])
plt.show()


# # `ImageDataGenerator` that Loads Images and Performs Image Augmentation
# 
# The `ImageDataGenerator` will be able automatically detect the different classes in our dataset from the folder structure that was setup in the previous section. The `ImageDataGenerator` will then take each of these images and apply transformations (such as rotations) as to augment our image dataset. After this, the data will be ready to be fed to a machine learning model.

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Here we initialize an image generator that will conduct in-memory image augmentation
#see here for docs -> https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
print('Training Set:')
train_gen = ImageDataGenerator(
    rescale=(1./255), #Rescales pixel values (originally 0-256) to 0-1
    rotation_range=0.4, #Rotates the image up to 40 degrees in either direction
    shear_range=0.2, #shears the image up to 20 degrees
    width_shift_range=0.2, #shifts the width by up to 20 %
    height_shift_range=0.2, #shifts the height by up to 20 %
    horizontal_flip=True, #flips the image along the horizontal axis
    fill_mode='nearest' #fills pixels lost during transformations with its nearest pixel
    )

train_generator = train_gen.flow_from_directory(
    'Data/Train',
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

#Now we do the same thing for the test set but do not include any augmentations except for pixel rescaling
print('Testing Set:')
test_gen = ImageDataGenerator(rescale=(1./255))

test_generator = test_gen.flow_from_directory(
    'Data/Test',
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)


# # Custom Callback
# 
# Callbacks are used in Tensorflow to allow user intervention during model training. A callback can be executed at a number of specific intances during model training. 
# For example: 
# - `on_batch_begin`/`end`
# - `on_epoch_begin`/`end`
# - `on_predict_batch_begin`/`end`
# - `on_predict_begin`/`end`
# - `on_test_batch_begin`/`end`
# - `on_test_begin`/`end`
# - `on_train_batch_begin`/`end`
# - `on_train_begin`/`end`
# 
# We will create `CustomCallback` which will stop the model from training once the model reaches 95% acccuracy on the training set.
# 
# Link: https://keras.io/api/callbacks/

# In[ ]:


from tensorflow.keras.callbacks import Callback

#creates a custom callback class
class CustomCallback(Callback):
    """
    This callback will stop the model from training once the model reaches 95% accuracy on the training data
    """
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            print('Accuracy above 95% -- Stopping Training')
            self.model.stop_training = True #stops model training

my_callback = CustomCallback()


# # Predefined Callback - `LearningRateScheduler`
# 
# There are also a number of predefined callbacks. We will use the `LearningRateScheduler` to dynamically update the learning rate of our optimizer.

# In[ ]:


from tensorflow.keras.callbacks import LearningRateScheduler

#creates a function that updates the learning rate based on the epoch number
def lr_update(epoch, lr):
    """
    For the first 5 epochs the learning rate will be 0.005.
    From epoch 6 and on, the learning rate will be reduced 1% per epoch
    """
    if epoch <= 5:
        return 0.005
    else:
        return lr * 0.99
    
lr_scheduler = LearningRateScheduler(lr_update)


# # Image Classifier Model from Scratch
# 
# We will create a basic convolution neural network to classifiy the images. Convolution neural networks typically follow the following pattern:
# 
# 
# **Convolution Layer -> Pooling Layer :: Repeated a number of times followed by -> Flatten -> Dense**
# 
# We will create a model with this architecture and train it on the training data.
# For more information on what these layers are visit -> https://www.youtube.com/watch?v=YRhxdVk_sIs

# In[ ]:


#Creates a model with the architecture mentioned above

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout


model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPool2D((2,2)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPool2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPool2D((2,2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.summary()


# In[ ]:


#Compiles and trains the model

from tensorflow.keras.optimizers import Adam

model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=20,
    callbacks=[my_callback, lr_scheduler]
)


# In[ ]:


#Plots model training history

fig, axs = plt.subplots(1, 2)

axs[0].plot(history.history['loss'])
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].set_title('Training Loss')

axs[1].plot(history.history['accuracy'])
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Training Accuracy')

plt.show()


# In[ ]:


#Evaluate Custom CNN on Test Data
#We will now see how well our model performs on test data. 
#It is likely that the model is overfitting the training data.
test_acc = model.evaluate(test_generator, verbose=0)[1]
print('Model Accuracy on Test Data:', round(test_acc,3))


# # Transfer Learning to Increase Accuracy
# 
# The accuracy of our basic CNN was not very high. Luckily, we are able to use models created and trained by others on our classification problem. 
# 
# The steps to conduct transfer learning are as follows:
# 1. Download the base model and weights
# 2. Make the base model untrainable (lock the weights)
# 3. Add a few layers to the end of the model 
# 4. Train the new model

# In[ ]:


#1. Download the base model which we will use for transfer learning 'MobileNetV2'
from tensorflow.keras.applications import MobileNetV2

base_model = MobileNetV2(
            input_shape=(150,150,3),
            include_top=False,
            weights='imagenet')

#2. Locks all the base model's weights
for layer in base_model.layers:
    layer.trainable = False

#3. Adds a few layers to the end of the model
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(1, activation='sigmoid')(x)


new_model = tf.keras.Model(base_model.input, x)

new_model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

#4. Trains the new model
history = new_model.fit(
    train_generator,
    epochs=20,
    callbacks=[my_callback, lr_scheduler]
)


# In[ ]:


#Plots training history of the new model
fig, axs = plt.subplots(1, 2)

axs[0].plot(history.history['loss'])
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].set_title('Transfer Model Training Loss')

axs[1].plot(history.history['accuracy'])
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Transfer Model Training Accuracy')

plt.show()


# In[ ]:


#Evaluation of the new model
test_acc = new_model.evaluate(test_generator, verbose=0)[1]
print('Model Accuracy on Test Data:', round(test_acc,3))


# **Wow!! The transfer learning model performed extremely well with just a few epochs!**

# ### Now try out these methods for yourself!
# 
# ### Similar Notebooks
# **TensorFlow Natural Language Processing Guide**: https://www.kaggle.com/code/calebreigada/tensorflow-natural-language-processing-guide
# 
# **TensorFlow Time Series Forecasting Guide**:
# https://www.kaggle.com/code/calebreigada/tensorflow-time-series-forecasting-guide
