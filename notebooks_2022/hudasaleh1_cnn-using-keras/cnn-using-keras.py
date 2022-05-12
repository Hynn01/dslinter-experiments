#!/usr/bin/env python
# coding: utf-8

# <h1 style="font-family:verdana;"> <center> <b>Building Convolution Neural Networks For Digit Recognition</b> 🎰🖊🖊 </center> </h1>

# <a id="What is CNN?"></a>
# 
# <h1> <span style="color:blue;"> What is CNN? </span> </h1>

# <p style="font-size:15px; font-family:verdana; line-height: 1.7em">
# Convolutional Neural Network(CNN) is a deep learning algorithm that is commonly used for images related problems such as image classification, object detection, instance segmentation, and semantic segmentation. It can find various importance elements in the images which helps in differentiating one from other. A CNN consists of an input, an output layer, and multiple hidden layers. The hidden layers of a CNN typically consist of convolutional layers, pooling layers and dense layers. Batch Normalization and drop out layers can also be used to avoid overfitting.</p>
# 
# ![1_vkQ0hXDaQv57sALXAJquxA.jpg](attachment:9f57ac60-0700-43a2-835b-72e02fe5e4e8.jpg)

# <h1> <span style="color:blue;">Basic Layers in Convolutional Neural Networks </span> </h1>
# <h2> <span style="color:gray;">Convolutional layers </span> </h2>
# 
# <p> <span style="font-size:15px; font-family:verdana; line-height: 1.7em"> The main purpose of a convolutional layer is to detect features or visual features in images such as edges, lines, color drops, etc. It utilized various <b>filters</b> (also known as <b>kernels</b>, <b>feature detectors</b>), to detect features are present throughout an image. A filter is just a matrix of values, called weights, that are trained to detect specific features.
# The filter carries out a <b>convolution operation</b>, which is an element-wise product and sum between two matrices. The objective of the Convolution Operation is to extract the <b>high-level features</b> such as edges, from the input image. CNNs can include one or multiple Convolutional Layer. Where, the first ConvLayer is responsible for capturing the Low-Level features such as edges, color, gradient orientation, etc. With added layers, the architecture adapts to the High-Level features as well. </span></p>

# ![1_GcI7G-JLAQiEoCON7xFbhg.gif](attachment:f9b3557f-641d-427d-8019-c5b4eac07d34.gif) 

# <p> <span style="font-size:15px; font-family:verdana; line-height: 1.7em">The output matrix from the convolution layer called <b>feature maps or convolved features</b>. Since we have multiple filters, we end up with a <b>3D output</b>: one 2D feature map per filter. The filter must have the same number of channels as the input image so that the element-wise multiplication can take place.</span></p>

# <h3><span style="color:gray;"> Conv layer main Arguments </span><h/3>
# <h4>1.kernel_size: </h4>
# <p><span style="font-size:15px; font-family:verdana; line-height: 1.7em"> filter dimensions</span></p>
# <h4>2.strides:</h4>
# <p><span style="font-size:15px; font-family:verdana; line-height: 1.7em">The stride value dictates by how much the filter should move at each step.</span></p>
# <h4>3.Padding:</h4>
# <p><span style="font-size:15px; font-family:verdana; line-height: 1.7em">Typically convolution results in a bit small image than the input one. One solution to resolve it, pad the image with zeros(zero-padding) to allow for more space for the kernel to cover the image. Adding padding to an image processed by a CNN allows for a more accurate analysis of images.</span></p>
# <h4>4.Activation Function:</h4>
# <p><span style="font-size:15px; font-family:verdana; line-height: 1.7em">The feature maps are summed with a bias term and passed through a non-linear activation function.  The purpose of the activation function is to introduce non-linearity into our network because the images are made of different objects that are not linear to each other, so the images are highly non-linear. Rectified Linear Unit(ReLU) is mostly preferred because it provides sparsity and a reduces likelihood of vanishing gradient problems.</span> </p>
# 

# <h2> <span style="color:gray;">Pooling Layer </span></h2>
# <p style="font-size:15px; font-family:verdana; line-height: 1.7em">Pooling Layer down samples the feature maps (to save on processing time), while also reducing the size of the image. This helps reduce overfitting, which would occur if CNN is given too much information, especially if that information is not relevant in classifying the image.
# There are different types of pooling, for example, max pooling and min pooling
# <b>Max Pooling</b> returns the maximum value from the portion of the image covered by the Kernel. On the other hand, <b>Average Pooling</b> returns the average of all the values from the portion of the image covered by the Kernel.
# Max Pooling also performs as a <b>Noise Suppressant</b>. It discards the noisy activations altogether and also performs de-noising along with dimensionality reduction. On the other hand, Average Pooling simply performs dimensionality reduction as a noise suppressing mechanism. Hence, we can say that Max Pooling performs a lot better than Average Pooling.</p>
# 

# ![1_KQIEqhxzICU7thjaQBfPBQ.jpg](attachment:96f7057b-2a7f-448a-b7e8-5209b8ecdb65.jpg)

# <p style="font-size:15px; font-family:verdana; line-height: 1.7em">These values then form a new matrix called a pooled feature map.</p>

# <h2 style=color:gray>Flattening Layer</h2>
# <p style="font-size:15px; font-family:verdana; line-height: 1.7em">After extracting features using multiple convolution and pooling layers, we are going to flatten the final output and feed it to a regular Neural Network for classification purposes. Flattening layer convert the 3D representation into a long feature vector.</p>
# <h2 style=color:gray>Dense layers</h2>
# <p style="font-size:15px; font-family:verdana; line-height: 1.7em">The flattened output is fed to a feed-forward neural network and backpropagation applied to every iteration of training. Sigmod and Softmax are added in the last dense layer for binary and multiclass classification respectively. </p>
# 

# <h2 style=color:gray>Global Pooling Layers</h2>
# <p style="font-size:15px; font-family:verdana; line-height: 1.7em">You can use another strategy called global pooling to replace the Flatten layers in CNN. It generates one feature map for each corresponding category of the classification task in the last Conv layer.
# One advantage of global pooling over the fully connected layers is that it is more native to the convolution structure by enforcing correspondences between feature maps and categories.
# Another advantage is that there is no parameter to optimize in the global pooling thus overfitting is avoided at this layer. It also can be used to reduce the dimensionality of the feature maps.</p>
# 

# <img src="attachment:769424d2-ce79-4621-b795-411aa1d5757f.png" width="900"/>
# 

# <p style="font-size:15px; font-family:verdana; line-height: 1.7em">There are 2 types of Global pooling: </p>
# <p style="font-size:15px; font-family:verdana; line-height: 1.7em"><b>Global Average Pooling:</b> which can be viewed as a structural regularizer that explicitly enforces feature maps to be confidence maps of concepts (categories).
# <p style="font-size:15px; font-family:verdana; line-height: 1.7em"><b>Global Max Pooling:</b> downsamples the input representation by taking the maximum value over the time dimension. </p>

# <h2 style=color:gray>Batch normalization</h2>
# <p style="font-size:15px; font-family:verdana; line-height: 1.7em">Batch normalization is a layer that allows every layer of the network to do learning more independently. It is used to normalize the output of the previous layers to avoid having instability in network and having high weight cascading down to the output. The activations scale the input layer in normalization. Using batch normalization learning becomes efficient also it can be used as regularization to avoid overfitting of the model.
# It can be used at several points in between the layers of the model. It is often placed after the convolution and pooling layers.</p>
# <h2 style=color:gray>Dropout</h2>
# <p style="font-size:15px; font-family:verdana; line-height: 1.7em">Dropouts are the regularization technique that is used to prevent overfitting in the model. Dropouts randomly drop some percentage of neurons of the network. This is done to enhance the learning of the model. Dropouts are usually advised not to use after the convolution layers, they are mostly used after the dense layers of the network.</p>

# ![pytorch-validation-of-cnn4.png](attachment:028cf0c2-6447-4dbc-8a88-a5c09c8c5639.png)
# ***

# <h1> <span style="color:blue;">Convolutional Neural Networks in Keras</span> </h1>
# 

# <h2 style=color:gray>EDA</h2>
# <p style="font-size:15px; font-family:verdana; line-height: 1.7em">The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology dataset. It is a dataset of small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9 </p>

# In[ ]:


# import libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import random


# In[ ]:


# dataframes creation for both training and testing datasets 
df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
print(df_train.head())


# In[ ]:


df_train.describe()


# <p style="font-size:15px; font-family:verdana; line-height: 1.7em">Each image is a row of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255. </p>
# 

# In[ ]:


# check that there is no missing data
print(df_train.isna().sum())


# In[ ]:


# check if the target classes are balanced
plt.figure(figsize=(10,6))
sns.countplot(x='label', data=df_train)


# In[ ]:


# Create training and testing arrays
training = np.array(df_train, dtype = 'float32')
testing = np.array(df_test, dtype='float32')


# In[ ]:


# visualize the training data
# this requires reshaping the images from 1D to 2D
W = 10
H = 10
fig, axes = plt.subplots(W, H, figsize = (17,17))

axes = axes.ravel() # flaten the matrix into array

n_training = len(training) # get the length of the training dataset

# Select a random number from 0 to n_training/ images will be selected randomly
for i in np.arange(0, W * H): 
    # Select a random number
    index = np.random.randint(0, n_training)
    # read and display an image with the selected index    
    axes[i].imshow( training[index,1:].reshape((28,28)) )
    axes[i].set_title(training[index,0], fontsize = 8) # the label
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)


# In[ ]:


df_test.head()


# In[ ]:


# visualize the testing data
# this requires reshaping the images from 1D to 2D
W = 5
H = 5
fig, axes = plt.subplots(W, H, figsize = (17,17))

axes = axes.ravel() # flaten the matrix into array

n_testing = len(testing) # get the length of the training dataset

# Select a random number from 0 to n_training/ images will be selected randomly
for i in np.arange(0, W * H): 
    # Select a random number
    index = np.random.randint(0, n_testing)
    # read and display an image with the selected index    
    axes[i].imshow( testing[index,:].reshape((28,28)) )
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)


# <h2 style=color:gray>Model Training</h2>
# 

# In[ ]:


from sklearn.model_selection import train_test_split

# Normalization -> pixels values from [0-255] to [0-1]
X_train = training[:,1:]/255
y_train = training[:,0]

x_test = testing/255


# create training and validation sets
x_train,x_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.2,random_state=101)


# In[ ]:


# reshape the images 
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_val   = x_val.reshape(x_val.shape[0],28,28,1)
x_test  = x_test.reshape(x_test.shape[0],28,28,1)


# In[ ]:


print('training shape',x_train.shape)


# In[ ]:


# CNN Model in keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense,Dropout
from tensorflow.keras.optimizers import Adam


model1 = Sequential([Conv2D(64,(3,3),input_shape=(28,28,1),activation='relu',padding = 'same'),
                    MaxPool2D(2,2),
                    Conv2D(64,3,activation='relu',padding = 'same'),  
                    MaxPool2D(2,2),
                    Flatten(),
                    Dropout(0.2),
                    Dense(128,activation = 'relu'),
                    Dense(64, activation = 'relu'),
                    Dense(10,activation='softmax')])
model1.summary()


# In[ ]:


# compile
model1.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001),metrics =['accuracy'])


# In[ ]:


# fit the model
from tensorflow.keras.callbacks import EarlyStopping
# early stopping: we can use EarlyStopping which is one of keras callbackes to stop training when a monitored metric has stopped improving.
earlystop = EarlyStopping(monitor='loss', patience=10)
epoch = 200 # we can set it to a large value because there is early stopping

history = model1.fit(x_train, y_train, batch_size = 64, 
                                 epochs = epoch, 
                                 validation_data = (x_val, y_val), 
                                 verbose = 1,
                                 steps_per_epoch = x_train.shape[0] // 64,
                                 callbacks = [earlystop])


# In[ ]:


from tensorflow.keras.models import load_model
model1.save('cnn_model.h5')
model = load_model('cnn_model.h5')
y_pred = np.argmax(model.predict(x_val), axis = 1)
print(y_pred)


# <h2 style=color:gray>Model Performance</h2>
# 

# In[ ]:


performance = pd.DataFrame(history.history)
plt.figure(figsize=(10,4))
performance[['loss','val_loss']].plot(figsize=(12,6))
performance[['accuracy','val_accuracy']].plot(figsize=(12,6))


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_val, y_pred))


# In[ ]:


plt.figure(figsize=(10,8))
plt.title('Predicted digits', size=14)
sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt = '.0f',linewidths=.5)
plt.show()


# In[ ]:


#y_test = cnn1.predict(test_array)
#y_test = np.array(pd.DataFrame(y_test).idxmax(axis=1))
y_test = np.argmax(model.predict(x_test), axis = 1)
print(y_test)


# In[ ]:


# visualize the prediction 
# this requires reshaping the images from 1D to 2D
W = 10
H = 10
fig, axes = plt.subplots(W, H, figsize = (17,17))

axes = axes.ravel() # flaten the matrix into array

n_test = len(testing) # get the length of the training dataset

# Select a random number from 0 to n_training/ images will be selected randomly
for i in np.arange(0, W * H): 
    # Select a random number
    index = np.random.randint(0, n_test)
    # read and display an image with the selected index    
    axes[i].imshow( testing[index,0:].reshape((28,28)) )
    axes[i].set_title(y_test[index], fontsize = 8) # the label
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)


# In[ ]:



df_submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
df_submission['Label']=y_test
df_submission.to_csv('submission.csv',index=False)


# <h2 style=color:gray>Error Analysis</h2>
# 

# https://androidkt.com/explain-pooling-layers-max-pooling-average-pooling-global-average-pooling-and-global-max-pooling/?msclkid=2d2b5b2ab38c11ec894c8e9559d7f4ad
# 
# https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
# 
# https://analyticsindiamag.com/everything-you-should-know-about-dropouts-and-batchnormalization-in-cnn/
# 
# https://keras.io/api/layers/convolution_layers/convolution2d/?msclkid=9d10fb6bb38c11ec960084a7bd401d0b
# 
# 

# In[ ]:




