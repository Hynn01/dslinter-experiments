#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# #In fact, that whole Kaggle Notebook should be deleted. GPU strikes Epochs back.
# 
# ![](https://i.ytimg.com/vi/p4349CICxhA/maxresdefault.jpg)youtube.com

# In[ ]:


import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from zipfile import ZipFile as zipper
import tensorflow as tf
from tensorflow.keras.layers import *
from tqdm import tqdm
import tensorflow.keras.layers as layers
import time
from IPython import display
from tensorflow.keras.models import Model
from glob import glob
from tqdm import tqdm
from IPython import display
import time


# In[ ]:


#Code by Daniel Valyano https://www.kaggle.com/code/danielvalyano/i-am-something-of-a-painter-myself/notebook

destination = '../input/images-from-star-wars-movies'
data = pd.DataFrame({'file': os.listdir(destination)})
data.head()


# In[ ]:


#Code by Daniel Valyano https://www.kaggle.com/code/danielvalyano/i-am-something-of-a-painter-myself/notebook

IMG_H = 128
IMG_W = 128
IMG_C = 3
batch_size = 64
latent_dim = 128
EPOCHS = 350
w_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
images_path = glob('../input/images-from-star-wars-movies/*/*.jpg')


# In[ ]:


#Code by Daniel Valyano https://www.kaggle.com/code/danielvalyano/i-am-something-of-a-painter-myself/notebook

def load_image(image):
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) / 127.5
    return image


# In[ ]:


#Code by Daniel Valyano https://www.kaggle.com/code/danielvalyano/i-am-something-of-a-painter-myself/notebook

import tensorflow_datasets as tfds

data_builder = tf.keras.utils.image_dataset_from_directory(
    destination,
    label_mode = None,
    batch_size = batch_size,
    image_size = (IMG_H, IMG_W),
    color_mode = 'rgb'
)
data_builder


# In[ ]:


#Code by Daniel Valyano https://www.kaggle.com/code/danielvalyano/i-am-something-of-a-painter-myself/notebook

data_builder = data_builder.map(load_image)
data_builder


# In[ ]:


#Code by Daniel Valyano https://www.kaggle.com/code/danielvalyano/i-am-something-of-a-painter-myself/notebook

def deconv_block(inputs, num_filters, kernel_size, strides, bn=True):
    x = Conv2DTranspose(
        filters=num_filters,
        kernel_size=kernel_size,
        kernel_initializer=w_init,
        padding="same",
        strides=strides,
        use_bias=False
        )(inputs)

    if bn:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
    return x


def conv_block(inputs, num_filters, kernel_size, padding="same", strides=2, activation=True):
    x = Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        kernel_initializer=w_init,
        padding=padding,
        strides=strides,
    )(inputs)

    if activation:
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
    return x


# In[ ]:


#Code by Daniel Valyano https://www.kaggle.com/code/danielvalyano/i-am-something-of-a-painter-myself/notebook

def build_generator(latent_dim):
    f = [2**i for i in range(5)][::-1]
    filters = 32
    output_strides = 16
    h_output = IMG_H // output_strides
    w_output = IMG_W // output_strides

    noise = tf.keras.layers.Input(shape=(latent_dim,), name="generator_noise_input")

    x = Dense(f[0] * filters * h_output * w_output, use_bias=False)(noise)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((h_output, w_output, 16 * filters))(x)

    for i in range(1, 5):
        x = deconv_block(x,
            num_filters=f[i] * filters,
            kernel_size=5,
            strides=2,
            bn=True
        )

    x = conv_block(x,
        num_filters=3,  ## Change this to 1 for grayscale.
        kernel_size=5,
        strides=1,
        activation=False
    )
    fake_output = tf.keras.layers.Activation("tanh")(x)

    return Model(noise, fake_output, name="generator")

def build_discriminator():
    f = [2**i for i in range(4)]
    image_input = Input(shape=(IMG_H, IMG_W, IMG_C))
    x = image_input
    filters = 64
    output_strides = 16
    h_output = IMG_H // output_strides
    w_output = IMG_W // output_strides

    for i in range(0, 4):
        x = conv_block(x, num_filters=f[i] * filters, kernel_size=5, strides=2)

    x = Flatten()(x)
    x = Dense(1)(x)

    return Model(image_input, x, name="discriminator")


# In[ ]:


#Code by Daniel Valyano https://www.kaggle.com/code/danielvalyano/i-am-something-of-a-painter-myself/notebook

gen = build_generator(latent_dim)
gen.summary()


# In[ ]:


#Code by Daniel Valyano https://www.kaggle.com/code/danielvalyano/i-am-something-of-a-painter-myself/notebook

disc = build_discriminator()
disc.summary()


# In[ ]:


get_ipython().system('mkdir generated_samples')


# In[ ]:


#Code by Daniel Valyano https://www.kaggle.com/code/danielvalyano/i-am-something-of-a-painter-myself/notebook

class GAN(Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        for _ in range(2):
            ## Train the discriminator
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            generated_images = self.generator(random_latent_vectors)
            generated_labels = tf.zeros((batch_size, 1))

            with tf.GradientTape() as ftape:
                predictions = self.discriminator(generated_images)
                d1_loss = self.loss_fn(generated_labels, predictions)
            grads = ftape.gradient(d1_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

            ## Train the discriminator
            labels = tf.ones((batch_size, 1))

            with tf.GradientTape() as rtape:
                predictions = self.discriminator(real_images)
                d2_loss = self.loss_fn(labels, predictions)
            grads = rtape.gradient(d2_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        ## Train the generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as gtape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = gtape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d1_loss": d1_loss, "d2_loss": d2_loss, "g_loss": g_loss}

def save_plot(examples, epoch, n):
    examples = (examples  + 1) / 2.0
    fig1 = plt.figure(figsize=(25, 15)) 
    for i in range(n * n):
        plt.subplot(n, n, i+1)
        plt.axis("off")
        plt.imshow(examples[i])
    plt.suptitle(f'Epoch {epoch+1}')
    filename = f"./generated_samples/generated_plot_epoch-{epoch+1}.png"
    plt.savefig(filename)
    plt.show()
    plt.close()


# In[ ]:


#Code by Daniel Valyano https://www.kaggle.com/code/danielvalyano/i-am-something-of-a-painter-myself/notebook

gan = GAN(disc, gen, latent_dim)

bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
gan.compile(d_optimizer, g_optimizer, tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1))

images_dataset = data_builder

with tf.device('/device:GPU:0'):
    for epoch in range(EPOCHS):
        gan.fit(images_dataset, epochs=1)
        gen.save("./generated_samples/gen.h5")
        disc.save("./generated_samples/disc.h5")

        n_samples = 9
        noise = np.random.normal(size=(n_samples, latent_dim))
        examples = gen.predict(noise)
        display.clear_output(wait=True)
        save_plot(examples, epoch, int(np.sqrt(n_samples)))


# #It Seems that Darth Epochs Vader striked back and win. MarkDown Hamill confirmed it below.

# ![](https://static0.srcdn.com/wordpress/wp-content/uploads/2019/09/facebook.jpg?q=50&fit=crop&w=740&h=750&dpr=1.5)screenrant.com

# #Acknowledgement:
# 
# Daniel Valyano https://www.kaggle.com/code/danielvalyano/i-am-something-of-a-painter-myself/notebook
