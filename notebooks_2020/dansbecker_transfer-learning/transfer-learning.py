#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# At the end of this lesson, you will be able to use transfer learning to build highly accurate computer vision models for your custom purposes, even when you have relatively little data.
# 
# # Lesson
# 

# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo('mPFq5KMxKVw', width=800, height=450)


# # Sample Code
# 
# ### Specify Model

# In[ ]:


from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

num_classes = 2
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False


# ### Compile Model

# In[ ]:


my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# ### Fit Model

# In[ ]:


from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = data_generator.flow_from_directory(
        '../input/urban-and-rural-photos/train',
        target_size=(image_size, image_size),
        batch_size=24,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        '../input/urban-and-rural-photos/val',
        target_size=(image_size, image_size),
        class_mode='categorical')

my_new_model.fit_generator(
        train_generator,
        steps_per_epoch=3,
        validation_data=validation_generator,
        validation_steps=1)


# ### Note on Results:
# The printed validation accuracy can be meaningfully better than the training accuracy at this stage. This can be puzzling at first.
# 
# It occurs because the training accuracy was calculated at multiple points as the network was improving (the numbers in the convolutions were being updated to make the model more accurate).  The network was inaccurate when the model saw the first training images, since the weights hadn't been trained/improved much yet.  Those first training results were averaged into the measure above.
# 
# The validation loss and accuracy measures were calculated **after** the model had gone through all the data.  So the network had been fully trained when these scores were calculated.
# 
# This isn't a serious issue in practice, and we tend not to worry about it.

# # Your Turn
# **[Try transfer learning](https://www.kaggle.com/kernels/fork/532365)** yourself.
# 

# ---
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum/161321) to chat with other Learners.*
