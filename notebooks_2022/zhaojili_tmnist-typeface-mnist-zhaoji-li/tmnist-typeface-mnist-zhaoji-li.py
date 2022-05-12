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


# Name:Zhaoji Li
# 
# NUID 002198196

# # > **Abstruct**
# 
# TMNIST: A database of Typeface based digits
# This dataset is inspired by the MNIST database for handwritten digits. It consists of images representing digits from 0-9 produced using 2,990 google fonts files.
# 
# The dataset consists of a single file:
# 
# TMNIST_Data.csv
# This file consists of 29,900 examples with labels and font names. Each row contains 786 elements: the first element represents the font name (ex-Chivo-Italic, Sen-Bold), the second element represents the label (a number from 0-9) and the remaining 784 elements represent the grayscale pixel values (from 0-255) for the 28x28 pixel image.

# # # import dataset

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


df = pd.read_csv("../input/tmnist-typeface-mnist/TMNIST_Data.csv")
df.head()


# In[ ]:


# Counting the number of labels for each character
print(df.labels.value_counts())


# # # Data formalize

# **random sequence**

# In[ ]:


import random
N=list(range(len(df)))
n=len(df)
print(n)
random.seed(2022)
random.shuffle(N)


# **Data Split**

# In[ ]:


trainY=df.loc[N[0:(n//5)*4],'labels']
testY=df.loc[N[(n//5)*4:],'labels']
X=df.drop(['names','labels'],axis=1)
trainX=X.loc[N[0:(n//5)*4]]
testX=X.loc[N[(n//5)*4:]]


# **Labels Binarize**

# In[ ]:


from sklearn.preprocessing import LabelBinarizer
lb=LabelBinarizer()
y=lb.fit_transform(trainY)
y.shape


# **Data Reshape**

# In[ ]:


X_images=trainX.values.reshape(-1,28,28)
test=testX.values.reshape(-1,28,28)


# In[ ]:


X_images.shape, test.shape


# **Data Samples**

# In[ ]:


import matplotlib.pyplot as plt
fig,axs = plt.subplots(3,3,figsize=(9,9))
for i in range(9):
    h=i//3
    w=i%3
    axs[h][w].set_xticks([])
    axs[h][w].set_yticks([])
    axs[h][w].imshow(X_images[i])
plt.show()


# # # Training

# **Training and Testing Split**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_images, y, test_size=0.2, random_state=66)


# In[ ]:


X_train.shape, X_test.shape


# In[ ]:


y_train.shape, y_test.shape


# In[ ]:


X_train = (X_train/255).reshape(-1,28,28,1).astype('float32')
X_test = (X_test/255).reshape(-1,28,28,1).astype('float32')


# **CNN model**

# **1.convolutional layer**
# 
# A convolutional layer can produce a set of parallel feature maps, which are formed by sliding different convolution kernels on the input image and performing certain operations. In addition, at each sliding position, an element-wise product and sum operation is performed between the convolution kernel and the input image to project the information in the receptive field to an element in the feature map. This sliding process can be called stride Z_s, which is a factor that controls the size of the output feature map. The size of the convolution kernel is much smaller than the input image, and it overlaps or acts on the input image in parallel. All elements in a feature map are calculated by a convolution kernel, that is, a feature map. shared the same weights and bias terms.
# 
# **2.Linear rectifier layer**
# 
# The Rectified Linear Units layer (ReLU layer) uses Rectified Linear Units (ReLU) f(x)=max(0,x) as the activation function of this layer of nerves. It can enhance the non-linearity of the decision function and the entire neural network without changing the convolutional layer itself.
# 
# **3.pooling layer**
# 
# Pooling is another important concept in convolutional neural networks, which is actually a non-linear form of downsampling. There are many different forms of nonlinear pooling functions, of which "Max pooling" is the most common. It divides the input image into several rectangular areas, and outputs the maximum value for each sub-area.
# Intuitively, this mechanism works because the precise location of a feature is far less important than its rough location relative to other features. The pooling layer will continuously reduce the size of the data space, so the number of parameters and the amount of computation will also decrease, which also controls overfitting to a certain extent. Generally speaking, pooling layers are periodically inserted between the convolutional layers in the CNN network structure. Pooling operations provide another form of translation invariance. Because the convolution kernel is a feature finder, we can easily discover various edges in the image through the convolution layer. However, the features found by the convolutional layer are often too accurate. Even if we shoot an object at high speed, the edge pixel positions of the object in the photo are unlikely to be exactly the same. Through the pooling layer, we can reduce the sensitivity of the convolutional layer to edges. .
# The pooling layer computes the output on a pooling window (depth slice) at a time, and then moves the pooling window according to the stride.
# 
# **4.fully connected layer**
# 
# Finally, after several convolutional and max-pooling layers, high-level inference in the neural network is done through fully connected layers. Just like in a regular non-convolutional artificial neural network, neurons in a fully connected layer have connections to all activations in the previous layer. Therefore, their activations can be computed as affine transformations, i.e. by multiplying by a matrix and then adding a bias offset (a vector plus a fixed or learned bias).

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils


# We can use the Dropout method, that is, randomly "squeeze" some neurons and block their input and output each time the neurons are trained in the forward inference, which can play a role in regularization.
# 
# It can be understood that the emperor is exposed to rain and dew, and he is favored today, and may be put into the cold palace tomorrow, which prevents Yang Guifei from being "loved by three thousand people in one body", thereby preventing some neurons from becoming dominant and becoming topic leaders. , covering the sky with one hand.
# 
# All neurons are on equal footing, preventing overfitting.

# In[ ]:


model = Sequential()
model.add(Conv2D(32,(4,4),input_shape = (28,28,1),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(50, activation='relu'))

model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])


# In[ ]:


model.summary()


# **Train**

# In[ ]:


result = model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=92, verbose=2)
result


# **Model Evaluation**

# In[ ]:


evaluation=model.evaluate(X_test, y_test, verbose=1)
print(f'Accuracy: {evaluation[1] * 100}%')


# In[ ]:


def Plott (data):
    fig, ax = plt.subplots(1,2 , figsize = (20,7))
    # summarize history for accuracy
    ax[0].plot(data.history['accuracy'])
    ax[0].plot(data.history['val_accuracy'])
    ax[0].set_title('model accuracy')
    ax[0].legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    ax[1].plot(data.history['loss'], label =['loss'])
    ax[1].plot(data.history['val_loss'] ,label =['val_loss'])
    ax[1].set_title('model loss')
    ax[1].legend(['train', 'test'], loc='upper left')
    plt.show()


# In[ ]:


Plott(result)


# # # Explanation and Conclusion

# We divided the dataset into two subsets: images set (X) and lable set (y). For each item in X, we reshape the data into a 28x28 matrix. And each result in y is a 10 array. Therefore, we created a 9 layers CNN model that the first layer accept a 28x28 matrix and the output is a 10 array. The overall accuracy is 99.3%.

# **Reference**
# 
# https://www.kaggle.com/code/yuxinliustella/93-6-tmnist
# https://github.com/TommyZihao/zihaopytorch/blob/master/%E5%AF%B9Fashion-MNIST%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%AD%E7%9A%84%E6%97%B6%E5%B0%9A%E7%89%A9%E5%93%81%E8%BF%9B%E8%A1%8C%E5%88%86%E7%B1%BB.ipynb
