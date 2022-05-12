#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import display


# In[ ]:


fashion_mnist = keras.datasets.fashion_mnist


# In[ ]:


(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()


# In[ ]:


X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
X_train_target= X_train
X_valid_target = X_valid


# # Stacked Autoencoders
# stacked autoencoders(or deep autoencoders): Adding more layers helps the autoencoder learn more complex codings. That said, one must be careful not to make the autoencoder too powerful.

# In[ ]:


stacked_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="selu"),
])

stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[30]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])

stacked_ae = keras.models.Sequential([stacked_encoder, stacked_decoder])
stacked_ae.compile(loss="binary_crossentropy",optimizer=keras.optimizers.SGD(learning_rate=1.5))


# In[ ]:


history = stacked_ae.fit(X_train, X_train_target, epochs=10, validation_data=(X_valid, X_valid_target))


# In[ ]:


def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")

def show_reconstructions(model, n_images=5):
    reconstructions = model.predict(X_valid[:n_images])
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plot_image(X_valid[image_index])
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plot_image(reconstructions[image_index])


show_reconstructions(stacked_ae)


# # Convolutional Autoencoder
# If you want to build an autoencoder for images,you will need to build a convolutional autoencoder. The
# encoder is a regular CNN composed of convolutional layers and pooling layers. It typically reduces the spatial dimensionality of the inputs (i.e., height and width) while increasing the depth (i.e., the number of feature maps). The decoder must do the reverse (upscale the image and reduce its depth back to the original dimensions),and for this you can use transpose convolutional layers (alternatively, you could combine upsampling layers with convolutional layers).

# In[ ]:


conv_encoder = keras.models.Sequential([
     keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),
     keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="selu"),
     keras.layers.MaxPool2D(pool_size=2),
     keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="selu"),
     keras.layers.MaxPool2D(pool_size=2),
     keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="selu"),
     keras.layers.MaxPool2D(pool_size=2)
])
conv_decoder = keras.models.Sequential([
     keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="valid",
     activation="selu",
     input_shape=[3, 3, 64]),
     keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding="same", activation="selu"),
     keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="same", activation="sigmoid"),
     keras.layers.Reshape([28, 28])
])

conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])


# In[ ]:


conv_ae.compile(loss="binary_crossentropy",optimizer=keras.optimizers.SGD(learning_rate=1.5))


# In[ ]:


history = conv_ae.fit(X_train, X_train_target, epochs=10, validation_data=(X_valid, X_valid_target))


# In[ ]:


show_reconstructions(conv_ae)


# # Recurrent Autoencoders
# If you want to build an autoencoder for sequences, such as time series or text (e.g., for unsupervised learning or dimensionality reduction),then recurrent neural networks (see Chapter 15) may be better suited than dense networks. Building a recurrent autoencoder is straightforward: the encoder is typically a sequence-to-vector RNN which compresses the input sequence down to a single vector. The decoder is a vector-to-sequence RNN that does the reverse.

# In[ ]:


recurrent_encoder = keras.models.Sequential([
     keras.layers.LSTM(100, return_sequences=True, input_shape=[None, 28]),
     keras.layers.LSTM(30)
])

recurrent_decoder = keras.models.Sequential([
     keras.layers.RepeatVector(28, input_shape=[30]),
     keras.layers.LSTM(100, return_sequences=True),
     keras.layers.TimeDistributed(keras.layers.Dense(28, activation="sigmoid"))
])

recurrent_ae = keras.models.Sequential([recurrent_encoder, recurrent_decoder])


# In[ ]:


recurrent_ae.compile(loss="binary_crossentropy",optimizer=keras.optimizers.SGD(learning_rate=1.5))


# In[ ]:


history = recurrent_ae.fit(X_train, X_train_target, epochs=10, validation_data=(X_valid, X_valid_target))


# In[ ]:


show_reconstructions(recurrent_ae)


# # Denoising Autoencoders
# Another way to force the autoencoder to learn useful features is to add noise to its inputs, training it to recover the original, noise-free inputs. Autoencoders could also be used for feature extraction. 
# The noise can be pure Gaussian noise added to the inputs, or it can be randomly switched-off inputs, just like in dropout. 

# In[ ]:


denoising_encoder = keras.models.Sequential([
     keras.layers.Flatten(input_shape=[28, 28]),
     keras.layers.Dropout(0.5),
     keras.layers.Dense(100, activation="selu"),
     keras.layers.Dense(30, activation="selu")
])

denoising_decoder = keras.models.Sequential([
     keras.layers.Dense(100, activation="selu", input_shape=[30]),
     keras.layers.Dense(28 * 28, activation="sigmoid"),
     keras.layers.Reshape([28, 28])
])

denoising_ae = keras.models.Sequential([denoising_encoder, denoising_decoder])


# In[ ]:


denoising_ae.compile(loss="binary_crossentropy",optimizer=keras.optimizers.SGD(learning_rate=1.5))


# In[ ]:


history = denoising_ae.fit(X_train, X_train_target, epochs=10, validation_data=(X_valid, X_valid_target))


# In[ ]:


show_reconstructions(denoising_ae)


# # Sparse Autoencoders
# Another kind of constraint that often leads to good feature extraction is sparsity: by adding an appropriate term to the cost function, the autoencoder is pushed to reduce the number of active neurons in the coding layer. For example, it may be pushed to have on average only 5% significantly active neurons in the coding layer. This forces the autoencoder to represent each input as a combination of a small number of acti‐ vations. As a result, each neuron in the coding layer typically ends up representing a useful feature.

# In[ ]:


sparse_l1_encoder = keras.models.Sequential([
     keras.layers.Flatten(input_shape=[28, 28]),
     keras.layers.Dense(100, activation="selu"),
     keras.layers.Dense(300, activation="sigmoid"),
     keras.layers.ActivityRegularization(l1=1e-3)
])

sparse_l1_decoder = keras.models.Sequential([
     keras.layers.Dense(100, activation="selu", input_shape=[300]),
     keras.layers.Dense(28 * 28, activation="sigmoid"),
     keras.layers.Reshape([28, 28])
])

sparse_l1_ae = keras.models.Sequential([sparse_l1_encoder, sparse_l1_decoder])


# In[ ]:


sparse_l1_ae.compile(loss="binary_crossentropy",optimizer=keras.optimizers.SGD(learning_rate=1.5))


# In[ ]:


history = sparse_l1_ae.fit(X_train, X_train_target, epochs=10, validation_data=(X_valid, X_valid_target))


# In[ ]:


show_reconstructions(sparse_l1_ae)

