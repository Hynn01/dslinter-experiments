#!/usr/bin/env python
# coding: utf-8

# # <b>1 <span style='color:#E71414'>|</span> Importing libraries</b>
# - **For ML Models**: tensorflow, keras
# - **For Data Manipulation**: numpy, sklearn, PIL
# - **For Data Visualization**: matplotlib, seaborn, plotly

# In[ ]:


# For Data Processing & ML Models
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.models import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image, ImageEnhance

# For Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Miscellaneous
from tqdm import tqdm
import os
import random

# Turn off warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# # <b>2 <span style='color:#E71414'>| </span> Reading the Dataset</b>

# In[ ]:


unique_labels = ['Karacadag', 'Basmati', 'Jasmine', 'Arborio', 'Ipsala']

data_dir = '/kaggle/input/rice-image-dataset/Rice_Image_Dataset/'

all_paths = []
all_labels = []

for label in unique_labels:
    for image_path in os.listdir(data_dir+label):
        all_paths.append(data_dir+label+'/'+image_path)
        all_labels.append(label)
'''
An image of path all_paths[i] has the label all_labels[i], where i is an index
'''
all_paths, all_labels = shuffle(all_paths, all_labels)


# In[ ]:


values = [len([x for x in all_labels if x==label]) for label in unique_labels]
fig = go.Figure(data=[go.Pie(labels=unique_labels, values=values, rotation=-45, hole=.3, textinfo='label+percent')])
fig.update_layout(showlegend=False)
fig.show()


# #### The dataset is perfectly balanced

# # <b>3 <span style='color:#E71414; font-weight: bold;'>|</span> Data Preprocessing</b>

# <h2>3.1 <span style='color:#E71414; font-weight: bold;'>|</span> Train-Val Split</h2>  

# - 90% for training
# - 10% for validation

# In[ ]:


x_train_paths, x_val_paths, y_train, y_val = train_test_split(all_paths, all_labels,
                                                              test_size=0.1, random_state=42,
                                                              stratify=all_labels)


# <h2>3.2 <span style='color:#E71414; font-weight: bold;'>|</span> Image Data Augmentation</h2>  

# - Random Brightness from 60% to 140%
# - Random Contrast from 60% to 140%

# In[ ]:


BRIGHTNESS = (0.6, 1.4)
CONTRAST   = (0.6, 1.4)

def augment_image(image):
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(BRIGHTNESS[0],BRIGHTNESS[1]))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(CONTRAST[0],CONTRAST[1]))
    return image


# <h2>3.3 <span style='color:#E71414; font-weight: bold;'>|</span> Label encoder-decoder</h2>  

# In[ ]:


def encode_labels(labels):
    encoded = []
    for x in labels:
        encoded.append(unique_labels.index(x))
    return np.array(encoded)

def decode_labels(labels):
    decoded = []
    for x in labels:
        decoded.append(unique_labels[x])
    return np.array(decoded)


# <h2>3.4 <span style='color:#E71414; font-weight: bold;'>|</span> Load images</h2>  

# In[ ]:


IMAGE_SIZE = 96


# In[ ]:


def open_images(paths, augment=True):
    '''
    Given a list of paths to images, this function returns the images as arrays, and conditionally augments them
    '''
    images = []
    for path in paths:
        image = load_img(path, target_size=(IMAGE_SIZE,IMAGE_SIZE))
        if augment:
            image = augment_image(image)
        image = np.array(image)/255.0
        images.append(image)
    return np.array(images)


# Example usage of `open_images` function

# In[ ]:


# Load images and their labels
images = open_images(x_train_paths[50:59])
labels = y_train[50:59]

# Plot images with their labels
fig = plt.figure(figsize=(12, 6))
for x in range(1, 9):
    fig.add_subplot(2, 4, x)
    plt.axis('off')
    plt.title(labels[x])
    plt.imshow(images[x])
plt.show()


# <h2>3.5 <span style='color:#E71414; font-weight: bold;'>|</span> Data Generator</h2>  
# <p style="font-size:15px; line-height: 1.7em">
#     Given a list of paths to images, and the labels, <br>
#     this function augments the images, normalizes them, encodes the label, and then returns the batch on which the model can train on. <br>
# </p>

# In[ ]:


def datagen(paths, labels, batch_size=12, epochs=3, augment=True):
    for _ in range(epochs):
        for x in range(0, len(paths), batch_size):
            batch_paths = paths[x:x+batch_size]
            batch_images = open_images(batch_paths, augment=augment)
            batch_labels = labels[x:x+batch_size]
            batch_labels = encode_labels(batch_labels)
            yield batch_images, batch_labels


# # <b>4 <span style='color:#E71414; font-weight: bold;'>|</span> Model</b>

# <h2>4.1 <span style='color:#E71414; font-weight: bold;'>|</span> Build Model</h2>  

# <h3>I am using <span style = "color:#E71414; font-weight: normal;">VGG16</span> for transfer learning</h3>

# In[ ]:


base_model = VGG16(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3), include_top=False, weights='imagenet')
# Set all layers to non trainable
for layer in base_model.layers:
    layer.trainable = False
# Set the last VGG block to trainable
base_model.layers[-2].trainable = True
base_model.layers[-3].trainable = True
base_model.layers[-4].trainable = True


# In[ ]:


model = Sequential()
model.add(Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
model.add(base_model)
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(unique_labels), activation='softmax'))


# In[ ]:


model.summary()


# <h2>4.2 <span style='color:#E71414; font-weight: bold;'>|</span> Compile Model</h2>  

# `SparseCategoricalCrossentropy` and `CategoricalCrossentropy` are basically the same loss functions, just their input formats are different.  
# 
# $$\mathrm{Loss} = -\cfrac{1}{N}\sum_{i=1}^N [y_i\text{log}(\hat y_i) + (1-y_i)\text{log}(1-\hat y_i)]$$
# where,  
# $\hat y$ is the predicted label, and $y$ is the actual label  
# $y_i$ is the $i^\mathbf{th}$ sample of $y$  and $\hat y_i$ is the $i^\mathbf{th}$ sample of $\hat y$  
# $N$ is the number of samples
# 
# If $y_i$ is **one-hot encoded**, we use `CategoricalCrossentropy`, and if $y_i$ is **integer-encoded**, we use `SparseCategoricalCrossentropy`
# 
# In our case, our labels are **integer-encoded**, so we are using `SparseCategoricalCrossentropy`  

# In[ ]:


model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])


# <h2>4.3 <span style='color:#E71414; font-weight: bold;'>|</span> Train Model</h2>  

# In[ ]:


batch_size = 64
steps = int(len(x_train_paths)/batch_size)
epochs = 5
history = model.fit(datagen(x_train_paths, y_train, batch_size=batch_size, epochs=epochs),
                    epochs=epochs, steps_per_epoch=steps)


# <h2>4.4 <span style='color:#E71414; font-weight: bold;'>|</span> Evaluate Model</h2>  

# In[ ]:


batch_size=128
steps = int(len(x_val_paths)/batch_size)
y_pred = []
y_true = []
for x,y in tqdm(datagen(x_val_paths, y_val, batch_size=batch_size, epochs=1, augment=False), total=steps):
    pred = model.predict(x)
    pred = np.argmax(pred, axis=-1)
    for i in decode_labels(pred):
        y_pred.append(i)
    for i in decode_labels(y):
        y_true.append(i)


# In[ ]:


print(classification_report(y_true, y_pred, digits=3))


# In[ ]:


cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,8))

ax = sns.heatmap(cm/np.sum(cm),fmt='.2%', annot=True, cmap='Blues')

ax.set_title('Confusion Matrix with labels\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

ax.xaxis.set_ticklabels(unique_labels)
ax.yaxis.set_ticklabels(unique_labels)

plt.show()


# # <b>5 <span style='color:#E71414; font-weight: bold;'>|</span> Inference</b>

# In[ ]:


def predict(images):
    pred = model.predict(images)
    pred = np.argmax(pred, axis=-1)
    pred = decode_labels(pred)
    return pred


# In[ ]:


NUM_IMAGES = 8
idx = random.sample(range(len(y_val)), NUM_IMAGES)

labels = [y_val[x] for x in idx]
image_paths = [x_val_paths[x] for x in idx]
images = open_images(image_paths, augment=False)
pred = predict(images)

cols = 4
rows = 2
fig = plt.figure(figsize=(12, 7))

for x in range(NUM_IMAGES):
    fig.add_subplot(rows, cols, x+1)
    plt.axis('off')
    plt.title('Predicted:'+str(labels[x])+'\nActual:'+str(labels[x]))
    plt.imshow(images[x])
plt.show()


# ### Please Upvote this notebook as it encourages me in doing better.
# ![](http://68.media.tumblr.com/e1aed171ded2bd78cc8dc0e73b594eaf/tumblr_o17frv0cdu1u9u459o1_500.gif)
