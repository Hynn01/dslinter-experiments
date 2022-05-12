#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from keras import models

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTj3R5YMGdjis_UAt10xWgqfsxRy_nPhfhwcfxzDETZL1bgDQj2ddktMxozYurCaL3Vfg&usqp=CAU)analyticsvidhya.com

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
import pathlib
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing import image_dataset_from_directory
import itertools


# #How to Visualize Deep Learning Models using Visualkeras?
# 
# By Yugesh Verma
# 
# "Deep Learning models are considered black-box models. It is not easy to understand how a defined model is functioning with the data. visualizing the deep learning models can help in improve interpretability."
# 
# https://analyticsindiamag.com/how-to-visualize-deep-learning-models-using-visualkeras/

# In[ ]:


#Code by Gonzalo Recio https://www.kaggle.com/code/gonzalorecioc/alzheimer-brain-mri-classifier-effnetb0-99-acc  

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
    print(f"{len(dirnames)} dirs and {len(filenames)} images in '{dirpath}'")


# In[ ]:


#Code by Gonzalo Recio https://www.kaggle.com/code/gonzalorecioc/alzheimer-brain-mri-classifier-effnetb0-99-acc

data_dir = "/kaggle/input/abstract-paintings/"
path_dir = pathlib.Path("/kaggle/input/abstract-paintings/img/") 
class_names = np.array(sorted([item.name for item in path_dir.glob('*')]))
print(class_names)


# In[ ]:


#Code by Gonzalo Recio https://www.kaggle.com/code/gonzalorecioc/alzheimer-brain-mri-classifier-effnetb0-99-acc

def view_random_image(target_dir, target_class):
    target_folder = target_dir + target_class
    random_image = random.sample(os.listdir(target_folder), 1)
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off");

    print(f"Image shape: {img.shape}")
    return img


# In[ ]:


#Code by Gonzalo Recio https://www.kaggle.com/code/gonzalorecioc/alzheimer-brain-mri-classifier-effnetb0-99-acc

#img = view_random_image(data_dir, class_names[3])#Save for the next time


# In[ ]:


#Code by Gonzalo Recio https://www.kaggle.com/code/gonzalorecioc/alzheimer-brain-mri-classifier-effnetb0-99-acc

# For replicable results
SEED = 0
# Size of the images is (128,128)
IMAGE_SIZE = (128, 128)
# Default batch size
BATCH_SIZE = 32
# Images are grayscale
COLOR_MODE = "grayscale"
# 20% test split
VAL_SPLIT = 0.2

tf.random.set_seed(SEED)
np.random.seed(SEED)
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    label_mode='categorical',
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
    color_mode=COLOR_MODE,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)
valid_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=VAL_SPLIT,
    subset="validation",
    label_mode='categorical',
    seed=SEED,
    color_mode=COLOR_MODE,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)


# In[ ]:


#Code by Gonzalo Recio https://www.kaggle.com/code/gonzalorecioc/alzheimer-brain-mri-classifier-effnetb0-99-acc

base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = True
inputs = tf.keras.layers.Input(shape=(IMAGE_SIZE+(1,)), name="input_layer")
# Efficient net model has the normalizing layer builtin
x = base_model(inputs)
x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax", name="output_layer")(x)


# In[ ]:


#Code by Gonzalo Recio https://www.kaggle.com/code/gonzalorecioc/alzheimer-brain-mri-classifier-effnetb0-99-acc

model = tf.keras.Model(inputs, outputs)


# In[ ]:


#Code by Gonzalo Recio https://www.kaggle.com/code/gonzalorecioc/alzheimer-brain-mri-classifier-effnetb0-99-acc

# Default Learning rate
LR = 0.001

model.compile(loss="categorical_crossentropy", 
                optimizer=tf.keras.optimizers.Adam(learning_rate=LR), 
                metrics=["accuracy"])


# In[ ]:


#Code by Gonzalo Recio https://www.kaggle.com/code/gonzalorecioc/alzheimer-brain-mri-classifier-effnetb0-99-acc

model.summary()


# #Almost identical with my DL script.

# #Create a NN

# In[ ]:


#Code by Devashree Madhugiri https://www.kaggle.com/code/devsubhash/visualize-deep-learning-models-using-visualkeras

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# #Build a CNN

# In[ ]:


#Code by Devashree Madhugiri https://www.kaggle.com/code/devsubhash/visualize-deep-learning-models-using-visualkeras

from keras import models  
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout
#from keras_visualizer import visualizer
from keras import layers 
cnn = Sequential()
cnn.add(Conv2D(64,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(64,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.2))
cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(10, activation='softmax'))


# In[ ]:


#Code by Devashree Madhugiri https://www.kaggle.com/code/devsubhash/visualize-deep-learning-models-using-visualkeras

from keras.utils.vis_utils import plot_model
plot_model(cnn, to_file='cnn_plot.png', show_shapes=True, show_layer_names=True)


# #Install VisualKeras

# In[ ]:


get_ipython().system('pip install visualkeras')


# In[ ]:


#Code by Devashree Madhugiri https://www.kaggle.com/code/devsubhash/visualize-deep-learning-models-using-visualkeras

import keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, optimizers
import warnings 
warnings.filterwarnings('ignore')
print("Tensorflow version:",tf.__version__)
print("Keras version:",keras.__version__)


# #From now till the end everything is the same as Devashree Madhugiri. I've No clue how to fix it.

# In[ ]:


#Code by Devashree Madhugiri https://www.kaggle.com/code/devsubhash/visualize-deep-learning-models-using-visualkeras

model = Sequential()
model.add(Conv2D(64,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.summary()


# In[ ]:


#Code by Devashree Madhugiri https://www.kaggle.com/code/devsubhash/visualize-deep-learning-models-using-visualkeras

import visualkeras
visualkeras.layered_view(model)


# In[ ]:


#Code by Devashree Madhugiri https://www.kaggle.com/code/devsubhash/visualize-deep-learning-models-using-visualkeras

model = Sequential()
model.add(Conv2D(64,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(128,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.35))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()


# In[ ]:


#Code by Devashree Madhugiri https://www.kaggle.com/code/devsubhash/visualize-deep-learning-models-using-visualkeras

visualkeras.layered_view(model)


# #Neural Network model in 2D space or in flat style.

# In[ ]:


#Code by Devashree Madhugiri https://www.kaggle.com/code/devsubhash/visualize-deep-learning-models-using-visualkeras

visualkeras.layered_view(model, legend=True) # without custom font
from PIL import ImageFont
visualkeras.layered_view(model, legend=True) 


# In[ ]:


#Code by Devashree Madhugiri https://www.kaggle.com/code/devsubhash/visualize-deep-learning-models-using-visualkeras

visualkeras.layered_view(model, legend=True, draw_volume=False)


# #Below, the spacing between the layers (avove) can be adjusted using the 'spacing' variable

# In[ ]:


#Code by Devashree Madhugiri https://www.kaggle.com/code/devsubhash/visualize-deep-learning-models-using-visualkeras

visualkeras.layered_view(model, legend=True, draw_volume=False,spacing=30)


# In[ ]:


#Code by https://github.com/paulgavrikov/visualkeras

from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D
from collections import defaultdict

color_map = defaultdict(dict)
color_map[Conv2D]['fill'] = 'orange'
color_map[ZeroPadding2D]['fill'] = 'gray'
color_map[Dropout]['fill'] = 'pink'
color_map[MaxPooling2D]['fill'] = 'red'
color_map[Dense]['fill'] = 'green'
color_map[Flatten]['fill'] = 'teal'

visualkeras.layered_view(model, color_map=color_map)


# In[ ]:


#Code by https://github.com/paulgavrikov/visualkeras

visualkeras.layered_view(model, scale_xy=1, scale_z=1, max_z=1000)


# In[ ]:


#https://github.com/paulgavrikov/visualkeras

model.add(visualkeras.SpacingDummyLayer(spacing=100))
...

visualkeras.layered_view(model, spacing=0)


# #Customized Colors for the layers

# In[ ]:


#Code by Devashree Madhugiri https://www.kaggle.com/code/devsubhash/visualize-deep-learning-models-using-visualkeras

from tensorflow.keras import layers
from collections import defaultdict
color_map = defaultdict(dict)
color_map[layers.Conv2D]['fill'] = '#00f5d4'
color_map[layers.MaxPooling2D]['fill'] = '#8338ec'
color_map[layers.Dropout]['fill'] = '#03045e'
color_map[layers.Dense]['fill'] = '#fb5607'
color_map[layers.Flatten]['fill'] = '#ffbe0b'
visualkeras.layered_view(model, legend=True,color_map=color_map)


# #Since those charts are the same of Devashree Madhugiri in another code, I'd hope to understand it.

# #Acknowledgement:
# 
# https://github.com/paulgavrikov/visualkeras
# 
# https://analyticsindiamag.com/how-to-visualize-deep-learning-models-using-visualkeras/
# 
# Devashree Madhugiri https://www.kaggle.com/code/devsubhash/visualize-deep-learning-models-using-visualkeras/notebook
# 
# https://www.analyticsvidhya.com/blog/2022/03/visualize-deep-learning-models-using-visualkeras/
# 
# Gonzalo Recio https://www.kaggle.com/code/gonzalorecioc/alzheimer-brain-mri-classifier-effnetb0-99-acc
