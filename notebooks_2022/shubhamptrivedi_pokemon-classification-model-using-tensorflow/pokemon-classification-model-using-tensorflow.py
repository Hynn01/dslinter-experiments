#!/usr/bin/env python
# coding: utf-8

# # Building a pokemon classification model using tensorflow
# ![](http://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSGtg9sZRxYS28nr3N8KteLJRDtDK8Apslu4Q&usqp=CAU)

# ## Import libraries

# In[ ]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 
import matplotlib.pyplot as plt


# ## Import dataset
# 
# #### Go to Add data button and import https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types dataset

# 
# #### List all the image filenames present in "../input/pokemon-images-and-types/images/images" location and show first five images using matplotib subplots
# 

# In[ ]:


# defining root directory
from PIL import Image

root_dir = "../input/pokemon-images-and-types/images/images"

files =  os.path.join(root_dir)
File_names = os.listdir(files)
print("This is the list of all the files present in the path given to us:\n")
print(File_names)

# plot here
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
first_five = File_names[0:6]

def subplots():
# Use the axes for plotting
    i = 0
    j = 0
    k = 0
    for k in range(5):
        state = os.path.join(root_dir, first_five[k])
        img = Image.open(state)
        axes[i,j].imshow(img)
        
        if k==2:
            i +=1
            j = 0
        else:
            j += 1


    plt.tight_layout(pad=2);
    
subplots()


# In[ ]:


## Run the below cells as it is
data = pd.read_csv("../input/pokemon-images-and-types/pokemon.csv")

data.head()


# ## We are going to use Type1 column as our labels. Each Name is unique and classified into 18 Type1 types. 

# In[ ]:


## Run the below cells as it is
data_dict = {}

for key, val in zip(data["Name"], data["Type1"]):
    data_dict[key] = val
print(data_dict)


# In[ ]:


labels = data["Type1"].unique()
print(labels)


# #### Create a dictionary and assign each label in labels list a unique id from 1 to 18. Name the dictionary as "labels_idx"

# In[ ]:


ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
labels_idx = dict(zip(labels,ids))

print(labels_idx)


# #### Understand and Complete the below code

# In[ ]:


final_images = []
final_labels = []
count = 0
files =  os.path.join(root_dir)
for file in File_names:
    count += 1
    img = cv2.imread(os.path.join(root_dir, file), cv2.COLOR_BGR2GRAY) 
    label = labels_idx[data_dict[file.split(".")[0]]] 
    # append img in final_images list
    final_images.append(np.array(img))
    # append label in final_labels list
    final_labels.append(np.array(label))
    
    
# converting lists into numpy arrayn
# normalizing and reshaping the data 
final_images = np.array(final_images, dtype = np.float32)/255.0
final_labels = np.array(final_labels, dtype = np.int8).reshape(809, 1)


# ### We have segregated our data into images and labels and is the time to build our model using tensorflow

# #### Complete the following code to create a 1 input, 3 layer fully connected and an output layer network to provide final output of 18 classes. 

# In[ ]:


# import necessary libraries
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(120, 120,3)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(18)
])
# print model summary and check trainable parameters
model.summary()


# In[ ]:


# compile model (Use: Adam optimizer, categorical_crossentropy loss and metrics as Accuracy)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# fit model (use images and labels)
history = model.fit(final_images, final_labels, epochs=50)


# In[ ]:


probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(final_images)

print("\n",predictions[0])
id = np.argmax(predictions[0])
print("\nid that we got from the model as prediction: {}\nType of pokemon associted with that id: {} ".format(id,labels[id]))
print("accuracy of the model",history.history['accuracy'][-1])


# # Thank you for checking out the notebook. If you like it please upvote and share
