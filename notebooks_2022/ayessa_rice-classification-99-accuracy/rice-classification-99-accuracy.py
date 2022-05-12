#!/usr/bin/env python
# coding: utf-8

# # <b>1 <span style='color:#78D118'>|</span> Introduction</b>
# ![](http://www.isaaa.org/kc/cropbiotechupdate/files/images/9252019125259AM.jpg)
# 
# ### What to Expect?
# In this notebook I'm gonna be using Transfer Learning MobileNetv2 by Keras to make a classification model for our dataset.
# 
# ### Dataset Overview
# This dataset includes 5 different rice types images with 15000 images for every category. And our task is to make a classification model that could correctly predict the 5 kinds of rice.
# 
# #### Rice Types
# * Arborio
# * Basmati
# * Ipsala
# * Jasmine
# * Karacadag

# # <b>2 <span style='color:#78D118'>|</span> Preparing the Data</b>

# In[ ]:


# Importing necessary libraries

# Building deep learning models
import tensorflow as tf 
from tensorflow import keras 
# For accessing pre-trained models
import tensorflow_hub as hub 
# For separating train and test sets
from sklearn.model_selection import train_test_split

# For visualizations
import matplotlib.pyplot as plt
import matplotlib.image as img
import PIL.Image as Image
import cv2

import os
import numpy as np
import pathlib


# **Preparing our dataset**

# In[ ]:


data_dir = "../input/rice-image-dataset/Rice_Image_Dataset" # Datasets path
data_dir = pathlib.Path(data_dir)
data_dir


# **Separating the categories**

# In[ ]:


arborio = list(data_dir.glob('Arborio/*'))[:600]
basmati = list(data_dir.glob('Basmati/*'))[:600]
ipsala = list(data_dir.glob('Ipsala/*'))[:600]
jasmine = list(data_dir.glob('Jasmine/*'))[:600]
karacadag = list(data_dir.glob('Karacadag/*'))[:600]


# **Checking samples**

# In[ ]:


fig, ax = plt.subplots(ncols=5, figsize=(20,5))
fig.suptitle('Rice Category')
arborio_image = img.imread(arborio[0])
basmati_image = img.imread(basmati[0])
ipsala_image = img.imread(ipsala[0])
jasmine_image = img.imread(jasmine[0])
karacadag_image = img.imread(karacadag[0])

ax[0].set_title('arborio')
ax[1].set_title('basmati')
ax[2].set_title('ipsala')
ax[3].set_title('jasmine')
ax[4].set_title('karacadag')


ax[0].imshow(arborio_image)
ax[1].imshow(basmati_image)
ax[2].imshow(ipsala_image)
ax[3].imshow(jasmine_image)
ax[4].imshow(karacadag_image)


# **Assigning a separate dictionary for images and their corresponding labels**

# In[ ]:


# Contains the images path
df_images = {
    'arborio' : arborio,
    'basmati' : basmati,
    'ipsala' : ipsala,
    'jasmine' : jasmine,
    'karacadag': karacadag
}

# Contains numerical labels for the categories
df_labels = {
    'arborio' : 0,
    'basmati' : 1,
    'ipsala' : 2,
    'jasmine' : 3,
    'karacadag': 4
}


# **Since the MobileNetv2 training images dimensions are 224 by 224 by 3, we have to reshape our categories into that**

# In[ ]:


img = cv2.imread(str(df_images['arborio'][0])) # Converting it into numerical arrays
img.shape # Its currently 250 by 250 by 3


# In[ ]:


X, y = [], [] # X = images, y = labels
for label, images in df_images.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img, (224, 224)) # Resizing the images to be able to pass on MobileNetv2 model
        X.append(resized_img) 
        y.append(df_labels[label])


# **Splitting the data and standarization**

# In[ ]:


# Standarizing
X = np.array(X)
X = X/255
y = np.array(y)


# In[ ]:


# Separating data into training, test and validation sets
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val)


# # <b>3 <span style='color:#78D118'>|</span> Creating the Model</b>

# In[ ]:


mobile_net = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4' # MobileNetv4 link
mobile_net = hub.KerasLayer(
        mobile_net, input_shape=(224,224, 3), trainable=False) # Removing the last layer


# In[ ]:


num_label = 5 # number of labels

model = keras.Sequential([
    mobile_net,
    keras.layers.Dense(num_label)
])

model.summary()


# # <b>4 <span style='color:#78D118'>|</span> Training the Model</b>

# In[ ]:


model.compile(
  optimizer="adam",
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])


history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))


# # <b>5 <span style='color:#78D118'>|</span> Evaluate the Model</b>
# 
# #### I've evaluated the model using accuracy, recall, precision and f1-score

# In[ ]:


model.evaluate(X_test,y_test)


# In[ ]:


from sklearn.metrics import classification_report

y_pred = model.predict(X_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_bool))


# # <b>6 <span style='color:#78D118'>|</span> Visualizing the Model</b>
# #### On how the models accuracy and loss changed through-out the 5 epochs

# In[ ]:


plt.plot(history.history['acc'], marker='o')
plt.plot(history.history['val_acc'], marker='o')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.show()


# In[ ]:


plt.plot(history.history['loss'], marker='o')
plt.plot(history.history['val_loss'], marker='o')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()


# # <b>7 <span style='color:#78D118'>|</span> Authors Message</b>
# 
# * If you find this helpful, I would really appreciate the upvote!
# * If you see something wrong please let me know.
# * And lastly Im happy to hear your thoughts about the notebook for me to also improve!
