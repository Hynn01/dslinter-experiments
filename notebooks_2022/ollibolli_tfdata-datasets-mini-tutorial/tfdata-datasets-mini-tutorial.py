#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.data import AUTOTUNE
import tensorflow as tf
from glob import glob
import matplotlib.pyplot as plt


# ### Get our datasets as list of filepaths

# In[ ]:


cats = glob("../input/microsoft-catsvsdogs-dataset/PetImages/Cat/*.jpg")
dogs = glob("../input/microsoft-catsvsdogs-dataset/PetImages/Dog/*.jpg")
print(f"#cats: {len(cats)}, #dogs: {len(dogs)}")


# In[ ]:


#create a fake dataset
labels = np.random.choice([0, 1], len(cats))
catset = tf.data.Dataset.from_tensor_slices(({'cats_input': cats}, labels))


# ### Create functions just like you would for a single data instance (image in this case)

# In[ ]:


#preprocessing images
# @tf.function
def load_cat_image(inputs, labels):
    #inputs is a dictionary get the filename
    filename = inputs['cats_input'] 
    #read file 
    file = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(file)
    #make the image [0,1]
    image = tf.cast(image, tf.float32) / 255.0
    # reize image
    image = tf.image.resize(image, size = [64, 64])
    #note we are not returning the labels we do that later
    return image


# In[ ]:


#yield a batch from dataset
next(iter(catset))


# In[ ]:


#increase the batch size .batch(4)
catset = catset.batch(4)
#should return batch size number of datapoints
next(iter(catset))


# In[ ]:


#reset
catset = tf.data.Dataset.from_tensor_slices(({'cats_input': cats}, labels))
#apply (map) function to every point in the dataset
catset = catset.map(load_cat_image)
next(iter(catset))


# In[ ]:


image = next(iter(catset)).numpy()
plt.imshow(image)
plt.title("Blushie pic")
plt.show()


# In[ ]:


#reset
catset = tf.data.Dataset.from_tensor_slices(({'cats_input': cats}, labels))
#apply (map) function to every point in the dataset then tell the batch size
catset = catset.map(load_cat_image).batch(2)
images = next(iter(catset)).numpy()
print(f"shape of batch: {images.shape} \n")
fig, ax = plt.subplots(1, 2)
ax[0].imshow(images[0])
ax[1].imshow(images[1])
plt.title("Multi Blushie")
plt.show()


# ### ok we got that down!
# lets try with creating a dataset that yields an image of a cat and a dog!

# In[ ]:


#this decorate makes stuff faster: https://www.tensorflow.org/guide/function
# @tf.function
def load_images(inputs, labels):
    filename = inputs['cats_input']
    cat = tf.io.read_file(filename)
    cat = tf.image.decode_jpeg(cat)
    cat = tf.cast(cat, tf.float32) / 255.0
    cat = tf.image.resize(cat, size = [64, 64])

    dog_filename = inputs['dogs_input']
    dog = tf.io.read_file(dog_filename)
    dog = tf.image.decode_jpeg(dog)
    dog = tf.cast(dog, tf.float32) / 255.0
    dog = tf.image.resize(dog, size = [64, 64])
    # now we also return the labels since I want to show you how to use it
    return {'cats_input': cat, 'dogs_input': dog}, labels


# In[ ]:


dataset = tf.data.Dataset.from_tensor_slices(({'cats_input': cats, 'dogs_input': dogs}, labels))


# In[ ]:


print(f"one cat and one dog + label coming up:")
(cat, dog), label = next(iter(dataset))
print(f"a dog {dog} and a cat {cat} walk into a bar and see a label {label}")


# ### we are experts now, skip above steps and return a batch of 2 = 4 images

# In[ ]:


dataset = dataset.map(load_images).batch(2)
images, catdog_labels = next(iter(dataset)) #this is a dict now:

print(f"shape of batch: {images['cats_input'].shape} \n")
print(f"shape of batch: {images['dogs_input'].shape} \n")
print(f"shape of batch: {catdog_labels.shape} \n")
fig, ax = plt.subplots(1, 4)
ax[0].imshow(images['cats_input'][0].numpy())
ax[1].imshow(images['cats_input'][1].numpy())
ax[2].imshow(images['dogs_input'][0].numpy())
ax[3].imshow(images['dogs_input'][1].numpy())
plt.title("Multi Blushie and Woofie")
plt.show()


# In[ ]:


shape = images['cats_input'][0].numpy().shape


# # How to use in Keras

# In[ ]:


catinput = Input(shape, name = "cats_input")
doginput = Input(shape, name = "dogs_input")
cat1 = Conv2D(filters = 1, kernel_size = 2, strides = 1)(catinput)
dog1 = Conv2D(filters = 1, kernel_size = 2, strides = 1)(doginput)
out = concatenate([cat1, dog1])
out = Flatten()(out)
out = Dense(1, activation = 'sigmoid')(out)
model = Model(inputs = [catinput, doginput], outputs = out, name = "Convolution_Model")
model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy',optimizer= 'sgd',metrics=['categorical_accuracy'])


# In[ ]:


#this decorate makes stuff faster: https://www.tensorflow.org/guide/function
@tf.function
def load_images(inputs, labels):
    filename = inputs['cats_input']
    cat = tf.io.read_file(filename)
    cat = tf.image.decode_jpeg(cat)
    cat = tf.cast(cat, tf.float32) / 255.0
    cat = tf.image.resize(cat, size = [64, 64])

    dog_filename = inputs['dogs_input']
    dog = tf.io.read_file(dog_filename)
    dog = tf.image.decode_jpeg(dog)
    dog = tf.cast(dog, tf.float32) / 255.0
    dog = tf.image.resize(dog, size = [64, 64])
    # now we also return the labels since I want to show you how to use it
    return {'cats_input': cat, 'dogs_input': dog}, labels


# In[ ]:


#there is a broken image in the dataset so I just use the first 32 to not deal with that
#sorry picked a bad dataset for the example and realized too late
cats = glob("../input/microsoft-catsvsdogs-dataset/PetImages/Cat/*.jpg")[:32]
dogs = glob("../input/microsoft-catsvsdogs-dataset/PetImages/Dog/*.jpg")[:32]
print(f"#cats: {len(cats)}, #dogs: {len(dogs)}")
dataset = tf.data.Dataset.from_tensor_slices(({'cats_input': cats, 'dogs_input': dogs}, labels[:32]))
dataset = dataset.map(load_images, num_parallel_calls = AUTOTUNE).cache().batch(32).prefetch(AUTOTUNE)


# In[ ]:


model.fit(dataset)


# ### read the guide on how to use prefetch, cache, shuffle map and in which order

# In[ ]:




