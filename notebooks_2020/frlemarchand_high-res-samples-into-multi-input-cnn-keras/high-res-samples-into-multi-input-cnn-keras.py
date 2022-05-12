#!/usr/bin/env python
# coding: utf-8

# # High-res samples into multi-input CNN
# This notebook will go through the steps required to take some high-resolution samples from an image and feed them into a multi-input CNN. The model will never see the entire image and only three 256x256 pixels samples. This approach assumes that there is more value to learn from high-resolution local areas in an image than a resized version of the entire image. To do so, we will use a custom generator which will prepare the images "live" as we build the batches during training.
# In previous versions of this notebook, I tried to stack the three samples together and feed as one input into a ResNet50 but the model never learnt.

# Version notes:
# * **V7**: Stacked images randonly sampled and feed them into a ResNet50 - *score:0.0*
# * **V8**: Randomly samples 3 images at full resolution and feed them into a multi-input CNN - *score:0.15*
# * **V9**: Apply to the model 3 times over each image at prediction time - *score:0.32*
# * **V11**: Take the mean from the predicted isup score over 5 runs instead of the max over 3 runs - *score:0.34*
# * **V12**: Now train the input branches first, before training a model with 3 input branches, with the input branches mostly frozen. 
# * **V13**: It appeared that the model training was sometimes unstable as seen in V12's log. This version brings so minor changes to try to fix that.
# * **V14**: Reduce the number of epochs in order to run in less than 6 hours and submit the notebook.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from sklearn.utils import shuffle
import openslide

import os
import sys
from shutil import copyfile, move
from tqdm import tqdm
import h5py
import random
from random import randint

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, Reshape, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import ResNet50, VGG16
from keras.losses import mean_squared_error
import keras as K
from sklearn.metrics import cohen_kappa_score


# In[ ]:


train_df = pd.read_csv("../input/prostate-cancer-grade-assessment/train.csv")
image_path = "../input/prostate-cancer-grade-assessment/train_images/"


# In[ ]:


train_df.head()


# In[ ]:


len(train_df)


# Generate paths to the input files and a list with corresponding groundtruth

# In[ ]:


image_size = 256
training_sample_percentage = 0.9
training_item_count = int(len(train_df)*training_sample_percentage)
train_df["image_path"] = [image_path+image_id+".tiff" for image_id in train_df["image_id"]]


# In[ ]:


#remove all image file that don't have a mask file
index_to_drop = []
for idx, row in train_df.iterrows():
    mask_path = row.image_path.replace("train_images","train_label_masks").replace(".tiff","_mask.tiff")

    if not os.path.isfile(mask_path):
        index_to_drop.append(idx)

train_df.drop(index_to_drop,0,inplace=True)


# In[ ]:


example = openslide.OpenSlide(train_df.iloc[0].image_path)
print(example.dimensions)
clipped_example = example.read_region((5000, 5000), 0, (256, 256))
plt.imshow(clipped_example)
plt.show()


# In[ ]:


train_df.head()


# We're quickly shuffling the data and split the dataframe in two in order to keep a validation set.

# In[ ]:


train_df = shuffle(train_df)
validation_df = train_df[training_item_count:]
train_df = train_df[:training_item_count]


# # Image preparation

# In[ ]:


def get_single_sample(image_path,image_size=256,training=False,display=False):
    '''
    Return a single 256x256 sample
    with possibility of returning a gleason score using the masks
    '''
    
    image = openslide.OpenSlide(image_path)
    
    mask_path = image_path.replace("train_images","train_label_masks").replace(".tiff","_mask.tiff")
    mask = openslide.OpenSlide(mask_path)
    
    stacked_image = []
    groundtruth_per_image = []
    
    maximum_iteration = 0
    selected_sample = False
    while not selected_sample:
        sampling_start_x = randint(image_size,image.dimensions[0]-image_size)
        sampling_start_y = randint(image_size,image.dimensions[1]-image_size)

        clipped_sample = image.read_region((sampling_start_x, sampling_start_y), 0, (256, 256))
        clipped_array = np.asarray(clipped_sample)
        
        #check that the sample is not empty
        #and use the standard deviation to make sure
        #there is something happening in the sample
        if (not np.all(clipped_array==255) and np.std(clipped_array)>20) or maximum_iteration>200:
            if display:
                plt.imshow(clipped_sample)
                plt.show()
                
            sampled_image = clipped_array[:,:,:3]
            
            if training:
                clipped_mask = mask.read_region((sampling_start_x, sampling_start_y), 0, (256, 256))
                groundtruth_per_image.append(np.mean(np.asarray(clipped_mask)[:,:,0]))
            
            selected_sample = True
        maximum_iteration+=1
    
    if training: 
        return np.array(sampled_image), np.array(groundtruth_per_image)
    else:
        return np.array(sampled_image)


# In[ ]:


def get_random_samples(image_path,image_size=256,display=False):
    '''
    Load an image and select random areas.
    Return a list of 3 images from areas where there is data.
    '''
    
    image = openslide.OpenSlide(image_path)
    stacked_image = []
    
    selected_samples = 0
    maximum_iteration = 0
    while selected_samples<3:
        sampling_start_x = randint(image_size,image.dimensions[0]-image_size)
        sampling_start_y = randint(image_size,image.dimensions[1]-image_size)

        clipped_sample = image.read_region((sampling_start_x, sampling_start_y), 0, (256, 256))
        clipped_array = np.asarray(clipped_sample)
        
        #check that the sample is not empty
        #and use the standard deviation to make sure
        #there is something happening in the sample
        if (not np.all(clipped_array==255) and np.std(clipped_array)>20) or maximum_iteration>200:
            if display:
                plt.imshow(clipped_sample)
                plt.show()

            stacked_image.append(clipped_array[:,:,:3])
            selected_samples+=1
        maximum_iteration+=1
    return np.array(stacked_image)


# Just to double-check. We know generate images with 3 images with 3 channels each.

# In[ ]:


get_random_samples(train_df.iloc[0].image_path).shape


# We can check that the function works by displaying the random samples that will be used as inputs.

# In[ ]:


_ = get_random_samples(train_df.iloc[0].image_path, display=True)


# In[ ]:


output = get_single_sample(train_df.iloc[0].image_path, display=True, training=True)
print(output[1])


# We create a custom generator that will randomly take samples from each full-scale image. It ensures there is actually some kind of content in the returned sample and give the associated groundtruth.

# In[ ]:


def custom_single_image_generator(image_path_list, batch_size=16):
    '''
    return an image and a corresponding gleason score from the mask
    '''
    
    while True:
        for start in range(0, len(image_path_list), batch_size):
            X_batch = []
            Y_batch = []
            end = min(start + batch_size, training_item_count)

            image_info_list = [get_single_sample(image_path, training=True) for image_path in image_path_list[start:end]]
            X_batch = np.array([image_info[0]/255. for image_info in image_info_list])
            Y_batch = np.array([image_info[1] for image_info in image_info_list])
            
            yield X_batch, Y_batch


# # Train our future input branch

# We build a fairly "basic" CNN that will be trained using random 256x256pixel samples and the gleason score. The gleason score will be calculcated from the corresponding masks for each image.

# In[ ]:


num_channel = 3
image_shape = (image_size, image_size, num_channel)

def branch(input_image):
    x = Conv2D(128, (3, 3))(input_image)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(64, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(32, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    
    x = layers.Dense(256)(x)
    x = Activation('relu')(x)
    
    return layers.Dropout(0.3)(x)


# In[ ]:


input_image = Input(shape=image_shape)
core_branch = branch(input_image)
output = Dense(1, activation='linear')(core_branch)

branch_model = Model(input_image,output)


# In[ ]:


branch_model.compile(loss="mse",
                      optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001))
callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.5),
             EarlyStopping(monitor='val_loss', patience=3),
             ModelCheckpoint(filepath='best_branch.h5', monitor='val_loss', save_best_only=True)]

batch_size = 16

history = branch_model.fit_generator(custom_single_image_generator(train_df["image_path"], batch_size=batch_size),
                        steps_per_epoch = int(len(train_df)/batch_size),
                        validation_data=custom_single_image_generator(validation_df["image_path"], batch_size=batch_size),
                        validation_steps= int(len(validation_df)/batch_size),
                        epochs=2,
                        callbacks=callbacks)


# # Build the core model

# Below, we have our custom generator which will use the function to generate original-scale 256x256 images from random samples.

# In[ ]:


def custom_generator(image_path_list, groundtruth_list, batch_size=16):
    num_classes=6
    while True:
        for start in range(0, len(image_path_list), batch_size):
            X_batch = []
            Y_batch = []
            end = min(start + batch_size, training_item_count)
            
            X_batch = np.array([get_random_samples(image_path)/255. for image_path in image_path_list[start:end]])
            input_image1 = X_batch[:,0,:,:]
            input_image2 = X_batch[:,1,:,:]
            input_image3 = X_batch[:,2,:,:]
            
            Y_batch = tf.keras.utils.to_categorical(np.array(groundtruth_list[start:end]),num_classes) 
            
            yield [input_image1,input_image2,input_image3], Y_batch


# Now, time to build the model. We have an `input_branch` function that will generate the 3 input branches with our pretrained weights, before merging them together. The 3 input branches have their weight mostly frozen as they have already been trained using the gleason score. The architecture itself will need some additional tuning later. The activation layer for our output is set to `softmax` with 6 units, representing the gradient from 0 to 5. It could also be set to `linear` if we approached this as a regression problem.

# In[ ]:


def input_branch(input_image):
    '''
    Generate a new input branch using our previous weights
    
    '''
    input_image = Input(shape=image_shape)
    core_branch = branch(input_image)
    output = Dense(1, activation='linear')(core_branch)
    branch_model = Model(input_image,output)
    branch_model.load_weights("../working/best_branch.h5")
        
    new_branch = Model(inputs=branch_model.input, outputs=branch_model.layers[-2].output)
    
    for layer in new_branch.layers[:-3]:
        layer.trainable = False
    
    return new_branch


# In[ ]:


input_image1 = Input(shape=image_shape)
input_image2 = Input(shape=image_shape)
input_image3 = Input(shape=image_shape)

first_branch = branch(input_image1)
second_branch = branch(input_image2)
third_branch = branch(input_image3)

merge = layers.Concatenate(axis=-1)([first_branch,second_branch,third_branch])
dense = layers.Dense(256)(merge)
dropout = layers.Dropout(0.3)(dense)
output = Dense(6, activation='softmax')(dropout)

model = Model([input_image1,input_image2,input_image3],output)
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))


# * Time to use our custom generator which loads 3 samples into the branches. We need to use it both for our training data and validation data. Be careful if you decide to add data augmentation into your custom generator as your performance on the validation set will be biased. In this case, add a parameter to your custom generator to activate the augmentation.

# In[ ]:


callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.5),
             EarlyStopping(monitor='val_loss', patience=3),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

batch_size = 16
history = model.fit_generator(custom_generator(train_df["image_path"], train_df["isup_grade"], batch_size=batch_size),
                        steps_per_epoch = int(len(train_df)/batch_size),
                        validation_data=custom_generator(validation_df["image_path"],np.array(validation_df["isup_grade"]), batch_size=batch_size),
                        validation_steps=int(len(validation_df)/batch_size),
                        epochs=3,
                        callbacks=callbacks)


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss over epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()


# # Make predictions using our newly trained model

# In[ ]:


model.load_weights("best_model.h5")


# Time to predict on new data! We use the function to retrieve random samples from the original full-size images and simply feed it to the model. The `passes` parameter will decide how many times we attempt to predict on a given image. Due to the random sampling, we will do 3 passes on each image to increase our chances of finding the cancerous zones and keep the mean of all grades. In future version, we will try to optimise the sampling strategy.

# In[ ]:


def predict_submission(df, path, passes=1):
    
    df["image_path"] = [path+image_id+".tiff" for image_id in df["image_id"]]
    df["isup_grade"] = 0
    
    for idx, row in df.iterrows():
        prediction_per_pass = []
        for i in range(passes):
            model_input = np.array([get_random_samples(row.image_path)/255.])
            input_image1 = model_input[:,0,:,:]
            input_image2 = model_input[:,1,:,:]
            input_image3 = model_input[:,2,:,:]

            prediction = model.predict([input_image1,input_image2,input_image3])
            prediction_per_pass.append(np.argmax(prediction))
            
        df.at[idx,"isup_grade"] = np.mean(prediction_per_pass)
    df = df.drop('image_path', 1)
    return df[["image_id","isup_grade"]]


# One last check to ensure that the `predict_submission` function works as intended. We use 100 images from the training set.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'test_from_training_df = pd.read_csv("../input/prostate-cancer-grade-assessment/train.csv")[:20]\npredict_submission(test_from_training_df, image_path, passes=5)')


# To finish, we add a condition that will check that the test images are available when running the notebook in `submission` mode. If it finds the folder, it will load `test.csv` and apply predictions before saving the submission file.

# In[ ]:


test_path = "../input/prostate-cancer-grade-assessment/test_images/"
submission_df = pd.read_csv("../input/prostate-cancer-grade-assessment/sample_submission.csv")

if os.path.exists(test_path):
    test_df = pd.read_csv("../input/prostate-cancer-grade-assessment/test.csv")
    submission_df = predict_submission(test_df, test_path, passes=5)

submission_df.to_csv('submission.csv', index=False)
submission_df.head()


# ### If you found this notebook helpful, please give it an upvote. It is always greatly appreciated!
