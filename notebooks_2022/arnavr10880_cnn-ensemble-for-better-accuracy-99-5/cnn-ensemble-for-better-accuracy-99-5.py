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


# # <b><span style="color:#27aee3; font-weight:1200">|</span> About Dataset
#     
# MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike. 
#     
# In this notebook, Modified LeNet CNN along with ensembling will be used to predict the digits.

# # 1 <b><span style="color:#27aee3; font-weight:1200">|</span> Required Libraries

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# # 2 <b><span style="color:#27aee3; font-weight:1200">|</span> Data

# In[ ]:


train = pd.read_csv('../input/digit-recognizer/train.csv')

print(f'{train.shape}\n')
train.head()


# In[ ]:


test = pd.read_csv('../input/digit-recognizer/test.csv')

print(f'{test.shape}\n')
test.head()


# ### <b><span style="color:#27aee3; font-weight:1200">※</span> Classes Distribution
# 
# To check the amount of images belonging to each digit from 0-9, **Counter** function can be used. It is used to get the count of individual elements. For example, suppose the list is: 
# 
# > `from collections import Counter`
# >
# > `a = [1, 1, 2 ,1 ,1 , 2, 1]`
# > 
# > `print(Counter(a))`
# 
# will print:
# 
# > `Counter({'1': 5, '2': 2})`
# 

# In[ ]:


print("Total values for each digit:\n")
Counter(train["label"])


# In[ ]:


sns.countplot(data=train, x='label')


# The no. of classes of digits are balanced. So, no need to worry about upsampling or downsampling! **Hooray!**
# 

# ### <b><span style="color:#27aee3; font-weight:1200">※</span> Checking for any missing pixel value
# 

# In[ ]:


print(f'Null values (training data): {train.isnull().sum().sum()}\n')
print(f'Null values (testing data): {test.isnull().sum().sum()}')


# # 3 <b><span style="color:#27aee3; font-weight:1200">|</span> Data Preprocessing
# 
# Our dataframes are in the following form:
#  
# ### <b><span style="color:#27aee3; font-weight:1200">※</span> Train dataframe
# 
#     
#  <table style="width:100%; border: 0.1px solid black">
#   <tr>
#     <th>label</th>
#     <th>pixel0</th>
#     <th>pixel1</th>
#     <th>...</th>
#     <th>pixel785</th>
#   </tr>
# </table>
#     
# From the shape of the train dataframe, total columns are 42000 x 785. The image size that is required is 28 x 28 which means the source should have 784 dimensions. As train set has 785 dimensions including the **label** column, hence **label** needs to be seperated from the training set so that the remaining columns can be reshaped into the required 28 x 28 size.
#     
#     
#     
#    
# ### <b><span style="color:#27aee3; font-weight:1200">※</span> Test dataframe
#     
#  <table style="width:100%; border: 0.1px solid black">
#   <tr>
#     <th>pixel0</th>
#     <th>pixel1</th>
#     <th>...</th>
#     <th>pixel784</th>
#   </tr>
# </table>
#     
# The test dataframe already has 784 dimensions, **label** being the column which is to be predicted, hence it can be easily reshaped into 28 x 28 without any transformation.

# In[ ]:


x_train = train.values[:, 1:] # get all values from 1st index onwards
y_train = train.values[:, 0]  # get the label column

x_test = test.values[:, 0:]   # get all values starting from 0th index

del train # delete train and test set to free up memory
del test 


# ### <b><span style="color:#27aee3; font-weight:1200">※</span> Some of the training set contents<br/>

# In[ ]:


fig = plt.figure(figsize=[14, 10])

for i in range(16):
    ax = fig.add_subplot(4 , 4, i + 1, xticks=[], yticks=[])
    ax.imshow(x_train[i].reshape((28,28)))
    ax.set_title(str(y_train[i]))


# ### <b><span style="color:#27aee3; font-weight:1200">※</span> Normalizing the pixel values
# 
# Normalizing the pixel values is a necessary step. There may be pixels representing the values say **244** while there may be some having the value **2**, so passing them as it is to a machine learning or deep learning algorithm will make the models perform poorly as there are some values enormously larger than the others. It creates an uneveness among the values. Also, it also messes with our beloved **gradient descent**. 🙂 <br/>
# 
# <center><img height=500 width=500 src='https://raw.githubusercontent.com/moelgendy/deep_learning_for_vision_systems/master/chapter_03/normalized.jpg'></img><center>
#     
# The formula used for normalization is:
#     
# <center><img height=400 width=400 src='https://www.spreadsheetweb.com/wp-content/uploads/2020/07/How-to-normalize-data-in-Excel-011.png'></img></center>

# In[ ]:


mean = np.mean(x_train) # take the mean
std = np.std(x_train)   # take the standard deviation
x_train = (x_train-mean)/(std+1e-7)    # normalizing the values
x_test = (x_test-mean)/(std+1e-7)

x_train = x_train.reshape(-1, 28, 28, 1) # reshaping them
x_test = x_test.reshape(-1, 28, 28, 1)

y_train


# In[ ]:


plt.imshow(x_train[2])


# In[ ]:


y_train[2]


# ### <b><span style="color:#27aee3; font-weight:1200">※</span> One-hot encoding & Validation Set
#     
# 
# One-hot encoding is a process by which categorical targets are converted to a binary form (1's & 0's). The best way to understand is using an example. Suppose like in this set, there are 10 classes, each number representing the equivalent digit:
# 
# <h3 align='center'> [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ] </h3>
#     
# Now, once one-hot encoding is done using tensorflow, the result can be interpreted in the following way:
#     
# <b>Image at index 3</b>
# 
# <code>plt.imshow(x_train[3])</code>
#     
# which is the digit <b> 4 </b> (look in the coming cells). Now, to check the <code>y_train</code> for the answer, the result is as shown below:
# 
# <code>y_train[3]</code>
#     
# > $ \begin{bmatrix}
# 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.  \\
# \end{bmatrix}  $
#     
# Here, 0 means the digit is not present at this index while 1 means the digit is present at this index (which in this case is it's value). So, the above result means that the value is at index **4** which in this case is it's value.
#     
# Well, this may take a little time to wrap the head around :)

# In[ ]:


num_classes = 10

y_train = to_categorical(y_train, num_classes=num_classes)

x_train, x_val = x_train[:37000], x_train[37000:]
y_train, y_val = y_train[:37000], y_train[37000:]

print(f'Training samples: {x_train.shape}\nValidation samples: {x_val.shape}\nTesting samples: {x_test.shape}')


# In[ ]:


plt.imshow(x_train[3])


# In[ ]:


y_train[3]


# # 4 <b><span style="color:#27aee3; font-weight:1200">|</span> CNN Model

# ### <b><span style="color:#27aee3; font-weight:1200">※</span> How CNN (Convolutional Neural Network) works?<br/>
# 
# <center><img width=600 height=600 src='https://www.mdpi.com/entropy/entropy-19-00242/article_deploy/html/images/entropy-19-00242-g001.png'></img></center>
# <br/>
# 
# In mathematics, convolution is the operation of two functions to produce a third
# modified function. In the context of CNNs, the first function is the input image, and
# the second function is the convolutional filter. Some mathematical
# operations are performed to produce a modified image with new pixel values.
# The above image summarizes the whole process of a CNN.
# 
# The basic components of a CNN involves:
# - CONV layer
# - POOLING layer
# - FULLY CONNECTED layer
# 
# 
# 
# #### <b><span style="color:#27aee3; font-weight:1200">※</span> CONV Layer<br/>
# 
# - The image pixel values are passed to the 1st convolutional layer of a CNN;
# - All the Convolutional layers try to find the relations between the pixels and their neighborhood pixels (with the help of kernels) and extract the different features of the image;
# - Then, the result is passed on to the subsequent layers (the kernel works like below):
# 
# <center><img height=700 width=700 src='https://1.cms.s81c.com/sites/default/files/2021-01-06/ICLH_Diagram_Batch_02_17A-ConvolutionalNeuralNetworks-WHITEBG.png'></img></center><br/>
# 
# 
# 
# #### <b><span style="color:#27aee3; font-weight:1200">※</span> POOLING Layer<br/>
# 
# - Adding more convolutional layers increases the depth of the output layer, which leads to increase in the number of parameters that the network needs to optimize (learn). This in turn increases the dimensions and hence training may become computationally expensive. So, pooling helps reduce the size of the network by reducing the number of parameters passed to the next layer.
# 
# <center><img height=500 width=500 src='https://cs231n.github.io/assets/cnn/maxpool.jpeg'></img></center>
# <br/>
# 
# 
# 
# #### <b><span style="color:#27aee3; font-weight:1200">※</span> FULLY CONNECTED Layer<br/>
#     
# <center><img src='https://miro.medium.com/max/441/1*yjy3dwRL-vmSpmUG7UNJYg@2x.png'></img></center>  
#   
# - Finally, after passing the image through the feature-learning process using convolutional and pooling layers, all the features are extracted and are put in a long tube. Hence, the extracted features are ready to be used for classification with the help of the usual fully connected layers.

# ### <b><span style="color:#27aee3; font-weight:1200">※</span> Modified LeNet Architecture<br/>
# <br/>
# <center><img src='https://raw.githubusercontent.com/moelgendy/deep_learning_for_vision_systems/2c9d077b43003657cd8f6d5ddfb6f83ee8bae1f3/chapter_05/images/lenet_architecture.png'></img></center>
# 
# The CNN Model that is used is the **LeNet** architecture with the following modifications:
# 
# - The 5x5 layers are replaced by two 3x3 layers for better feature extraction;
# - BatchNormalization is added;
# - The pooling layers are replaced by convolutional layers of stride 2;
# - **sigmoid** activation is replaced by **relu** one;
# - Dropout is added; 
# - The network is made more deeper; and
# - Ensembling of the CNN's is done.

# In[ ]:


nets = 4  # change here the amount of CNN ensembles
model = [0] * nets

for j in range(nets):
    model[j] = Sequential()

    model[j].add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))

    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))

    model[j].add(Conv2D(128, kernel_size = 4, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Flatten())
    model[j].add(Dropout(0.4))
    model[j].add(Dense(10, activation='softmax'))
    
    model[j].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    
model[0].summary() # summary of one of the models


# # 5 <b><span style="color:#27aee3; font-weight:1200">|</span> Data Augmentation

# Neural Networks are data hungry. The more they are fed with, the more better they perform. Sometimes the data that is available is not that large, so in order to increase the amount of data fed, Data Augmentation is used.
# 
# Data Augmentation doesn't add new images to the present data but when they are passed to the network, along with them different variations of the images are also passed. Hence, we generate more images out of thin air!
# 
# 
# <center><img src='https://i.gifer.com/origin/a5/a51dfe73c77cdcbac1b33fe8009bd0bc_w200.gif'></img></center>

# In[ ]:


datagen = ImageDataGenerator(
                rotation_range=10,  
                zoom_range = 0.10,  
                width_shift_range=0.1, 
                height_shift_range=0.1
)

aug = datagen.flow(x_train[6].reshape(-1, 28, 28, 1))

fig = plt.figure(figsize=[10, 8])
for i in range(24):
    
    ax = fig.add_subplot(3, 8, i+1, xticks=[], yticks=[])
    aug_img = next(aug)[0]
    ax.imshow(aug_img, cmap = 'gray')
    
plt.show()


# # 6 <b><span style="color:#27aee3; font-weight:1200">|</span> Network Training

# LearningRateScheduler is used to update the learning rate with each new epoch.

# In[ ]:


annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

hist = [0] * nets
epochs = 20
batch_size = 64

for j in range(nets):
    hist[j] = model[j].fit(
        datagen.flow(x_train, y_train, batch_size=batch_size), 
        epochs = epochs,
        steps_per_epoch=x_train.shape[0] // batch_size,
        validation_data = (x_val, y_val),
        callbacks = [annealer],
        verbose = 1)
    
    print(f'CNN {j+1}: Epochs = {epochs}, Training accuracy = {max(hist[j].history["accuracy"])}, Val accuracy = {max(hist[j].history["val_accuracy"])}')
    print()
    


# # 7 <b><span style="color:#27aee3; font-weight:1200">|</span> Ensembling Predictions
#     

# The results of all the CNN models are combined to get the maximun accuracy. 
# 
# <center><img src='https://editor.analyticsvidhya.com/uploads/990813.jpg'></img></center>

# In[ ]:


results = np.zeros((x_test.shape[0], 10))

for j in range(nets):
    results += model[j].predict(x_test)
    
results = np.argmax(results, axis=1)
results = pd.Series(results, name="Label")
submission = pd.concat([pd.Series(range(1,28001), name="ImageId"), results], axis=1)
submission.to_csv('sub.csv', index=False) # creating the submission file


# # 8 <b><span style="color:#27aee3; font-weight:1200">|</span> Accuracy & Loss Plots

# ### <b><span style="color:#27aee3; font-weight:1200">※</span> Accuracy<br/>

# In[ ]:


fig = plt.figure(figsize=[15, 10])

for i in range(4):
    ax = fig.add_subplot(4, 2, i+1)
    
    ax.plot([None] + hist[i].history['accuracy'], 'o-')
    ax.plot([None] + hist[i].history['val_accuracy'], 'x-')
    
    ax.legend(['Train acc', 'Validation acc'], loc = 0)
    ax.set_title(f'Model {i+1} Training/Validation acc per Epoch')
    ax.set_xlabel('Epoch')
    plt.tight_layout()


# ### <b><span style="color:#27aee3; font-weight:1200">※</span> Loss<br/>

# In[ ]:


fig = plt.figure(figsize=[15, 10])

for i in range(4):
    ax = fig.add_subplot(4, 2, i+1)
    
    ax.plot([None] + hist[i].history['loss'], 'o-')
    ax.plot([None] + hist[i].history['val_loss'], 'x-')
    
    ax.legend(['Train loss', 'Validation loss'], loc = 0)
    ax.set_title(f'Model {i+1} Training/Validation loss per Epoch')
    ax.set_xlabel('Epoch')
    plt.tight_layout()


# # 9 <b><span style="color:#27aee3; font-weight:1200">|</span> Some of the predictions

# In[ ]:


fig = plt.figure(figsize=[15, 10])


for i in range(20):
    img = x_test[i];
    ax = fig.add_subplot(2, 10, i+1)
    ax.grid(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.title.set_text(f'Pred:{results[i]}')
    plt.imshow(img, cmap='gray')
    
plt.show()


# # 10 <b><span style="color:#27aee3; font-weight:1200">|</span> Acknowledgements
#     
# - https://www.kaggle.com/code/cdeotte/25-million-images-0-99757-mnist
# - https://www.kaggle.com/code/samuelcortinhas/mnist-cnn-data-augmentation-99-6-accuracy

# ## Thanks for reading this. Would love to hear your views! 😄
