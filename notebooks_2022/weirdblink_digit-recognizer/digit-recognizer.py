#!/usr/bin/env python
# coding: utf-8

# # Digit Recognizer Project
# #### BUDT 737 - Group 16
# #### Team Members: Trang Ngo, Stone Heyman, Xin Lan, Kai-Wen Chen
# #### Accuracy: 97.88%

# # Table of contents
# 1. [Introduction](#introduction)
#     1. [Example use case](#example_use_case)
#     2. [AI-Driven solution](#AI_solution)
#     3. [Simple Neural Network overview](#NN_overview)
# 2. [Data Preparation](#data_preparation)
#     1. [Import required libraries](#libraries)
#     2. [Load the data](#load_data)
#     3. [Explore the data structure](#explore_data)
# 3. [Data Preprocessing](#data_preprocessing)
#     1. [Normalization](#normalization)
#     2. [Reshape](#reshape)
#     3. [Label encoding](#label_encoding)
#     4. [Split training and testing set](#partition)
# 4. [Setting Up the Neural Network Architecture](#setup_NN)
#     1. [Design the neural network architecture](#design)
#     2. [Define a baseline model](#baseline)
# 5. [Train Fully Connected Neural Network for Digit Recognition on Train Set](#training)
# 6. [Evaluate the Model Performance on the Test Set](#evaluating) 
#     1. [Overall model accuracy](#accuracy)
#     2. [Plot model accuracy and model loss by epoch](#plot)
#     3. [Display the actual label and the predicted label to check model performance](#recheck)
# 7. [Findings and Submission](#findings_and_submission)
#     1. [Overall findings](#findings)
#     2. [Submission](#submission)

# <a id="introduction"></a>
# # 1. Introduction
# In this project, we will implement a simple fully connected neural network with TensorFlow.
# 
# <a id="example_use_case"></a>
# ## <img align=left src="https://i.ibb.co/4sQ9tHg/thinking.png" alt="thinking" height="40" width="40" border="0"> 1.1. Example Use Case  
# 
# There is value in being able to record prescriptions into an electronic system, such as for record keeping at the doctor's office or in a pharmacy. This can be done by a human reading the paper prescriptions and manually typing them into the system. This however is time consuming, tedious work, and requires paying for that labor. An alterative could be to develop a model capable of recognizing handwriting and entering it automatically into the system. In this experiment, we will attempt to develop a model capable of accurately recognizing handwritten digits that could be applied to this use case.
# 
# <a id="AI_solution"></a>
# ## <img align=left src="https://i.ibb.co/yy67PkZ/ai.png" alt="ai" height="40" width="40" border="0"> 1.2. AI-Driven Solution: Simple Fully Connected Neural Network
# 
# By building a simple neural network model, we can use AI to help us capture the features in each picture and train our model to recognize the digits. 
# 
# <a id="NN_overview"></a>
# ## <img align=left src="https://i.ibb.co/QYsW6M0/neuralnet.png" alt="neuralnet" height="40" width="40" border="0"> 1.3. Simple Neural Network Overview
# This example is using MNIST handwritten digits. The dataset contains 42,000 examples for training and 28,000 examples for testing. The labels are numbers from 0 to 9.

# <a id="data_preparation"></a>
# # 2. Data Preparation 
# 
# <a id="libraries"></a>
# ## <img align=left src="https://i.ibb.co/mDp4kXB/library.png" alt="library" height="40" width="40" border="0"> 2.1. Import required libraries
# 
# Python packages enable different functions, providing easy ways of manipulating data and building models. As a first step, we 'import' packages to set up our environment in a way that allows us to take advantage of different capabilities. 
#  

# In[ ]:


import numpy
import pandas as pd 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils import np_utils # utils is a package toolkit. np_utils is a module that deal with np array. 
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras import backend as K
K.set_image_data_format('channels_first')


# <a id="load_data"></a>
# ## <img align=left src="https://i.ibb.co/Hnb80Dh/file.png" alt="file" height="40" width="50" border="0"> 2.2. Load the data
# <br>
# To start, we read the provided data.

# In[ ]:


import os
print(os.listdir('../input/digit-recognizer'))


# In[ ]:


#lfixed_blurboad the data
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
sub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
print("Data are Ready!!")


# In[ ]:


train.head()


# In[ ]:


test.head()


# <a id="explore_data"></a>
# ## <img align=left src="https://i.ibb.co/zZr9ZxC/explore.png" alt="explore" height="40" width="40" border="0"> 2.3. Explore the data structure

# In[ ]:


print('In the Train dataset, there are 42,000 images (data points). Each image (data point) is a 28 * 28 (= 784) matrix \nA column named label is also included to show the value of digits.')
print(train.shape)
print('\n')

print('In the Test dataset, there are 28,000 images (data points). Each image (data point) is a 28 * 28 matrix.')
print(test.shape)


# In[ ]:


# put labels into y_train variable
y_train = train["label"]
# Drop 'label' column
x_train = train.drop(labels = ["label"],axis = 1) 
x_test = test


# In[ ]:


# visualize number of digits classes
plt.figure(figsize=(15,7))
g = sns.countplot(y_train, palette="Paired")
plt.title("Number of digit classes")
y_train.value_counts()


# <a id="data_preprocessing"></a>
# # 3. Data Preprocessing 
# 
# <a id="normalization"></a>
# ## <img align=left src="https://i.ibb.co/cYbp0qv/balance.png" alt="balance" height="40" width="40" border="0"> 3.1. Normalization
# Why normalize?
# 
# Neural networks process inputs using small weight values, and inputs with large integer values can disrupt or slow down the learning process. As such, it is good practice to normalize the pixel values so that each pixel value has a value between 0 and 1.

# In[ ]:


#X: Normalize the Data to [0,1]
x_train = x_train / 255
x_test  = x_test / 255


# In[ ]:


print('x_train shape:', x_train.shape)
print('x_test.shape', x_test.shape)


# <a id="reshape"></a>
# ## <img align=left src="https://i.ibb.co/qddb4XS/reshape.png" alt="reshape" height="40" width="40" border="0"> 3.2. Reshape
# 
# * Train and test images (28 x 28)
# * We reshape all data to 28x28x1 3D matrices.
# * Keras needs an extra dimension in the end which correspond to channels. Our images are gray scaled so it use only one channel.
# 
# 

# In[ ]:


# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
x_train = x_train.values.reshape(-1,28,28,1)
x_test = x_test.values.reshape(-1,28,28,1)
print("x_train shape: ",x_train.shape)
print("x_test shape: ",x_test.shape)


# In[ ]:


# Confirm that the pixel values are now between 0 and 1:
# We are going to display the first image in the dataset
plt.figure()
plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)


# <a id="label_encoding"></a>
# ## <img align=left src="https://i.ibb.co/nBVZyc7/matrix.png" alt="matrix" height="40" width="40" border="0"> 3.3. One Hot encoding of labels
# Labels are 10 digits numbers from 0 to 9. We need to encode these lables to one hot vectors. A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension. In this case, the nth digit will be represented as a vector which is 1 in the nth dimension.
# 
# For example, 3 would be [0,0,0,1,0,0,0,0,0,0].

# In[ ]:


# Label Encoding 
y_train = to_categorical(y_train, num_classes = 10)


# <a id="partition"></a>
# ## <img align=left src="https://i.ibb.co/x1wr4c8/split.png" alt="split" height="40" width="40" border="0"> 3.4. Split training and testing set
# 
# We split the data into training and validation sets.
# * training size is 90%. 
# * testing size is 10%.

# In[ ]:


# fix random seed for reproducibility
seed = 43


# In[ ]:


# Split the train and the validation set for the fitting
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=2)
print("x_train shape",X_train.shape)
print("x_test shape",X_test.shape)
print("y_train shape",y_train.shape)
print("y_test shape",y_test.shape)


# In[ ]:


X_train__ = X_train.reshape(X_train.shape[0], 28, 28)

fig, axis = plt.subplots(1, 4, figsize=(10, 10))
for i, ax in enumerate(axis.flat):
    ax.imshow(X_train__[i], cmap='binary')
    digit = y_train[i].argmax()
    ax.set(title = f"Real Number is {digit}");


# <a id="setup_NN"></a>
# # 4. Setting Up the Neural Network Architecture
# 
# <a id="design"></a>
# ## <img align=left src="https://i.ibb.co/FqTL0Lf/build.png" alt="build" height="40" width="40" border="0"> 4.1. Design the neural network architecture
# 1. First, we are going to create the model framework. It is like the basic structure of a house without furniture and room layouts
# 
# 2. Next we will add the flatten layer which we can input an image. This layer will flatten the image to a vector so we can pass it in to the neural network. Each pixel of the image will become an individual input that feeds in the next hidden layer
# 
# 3. The next layer is a fully connected with 128 neurons. Because the neurons are all close to each other, we call it the "dense" layer. Each neuron on this layer is fully connected to the last layer. It takes the input from last layer, aggregates them and runs them through a "Relu" function. "Relu" is an activation function which will transform the aggregated result in each neuron.
# 
# 4. The next layer is our output layer. The output layer has 10 neurons because we have 10 different digits in this classification problem. We feed the results from last layer to a softmax activation function to output probability-like predictions for each class. The digits with the highest probability is what the algorithm thought the digit to be.
# 
# 5. The model is trained using categorical cross entropy loss function and the ADAM optimizer for gradient descent.

# <a id="baseline"></a>
# ## <img align=left src="https://i.ibb.co/XL1xmpt/model.png" alt="model" height="40" width="50" border="0"> 4.2. Define a baseline model
# <br>
# Now we are going to define a function to help us create a base model with the layers we dicussed.
# <br>
# Quick note: we use the "def" keyword in python to define a function. By defining a function, it increases the reusability and saves us time when we want to create the same model. So that we don't have to write out all the layers every time.

# In[ ]:


def baseline_model():
    # Create a model
    model = Sequential()
    # Add different layers to your model
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# <a id="training"></a>
# # 5. Train Fully Connected Neural Network for Digit Recognition on Train Set
# With our environment set up and data loaded, we can now train, test, and evaluate our model to recognize the 10 different digits. To do this, we will use the train set and test set we created above to train our model. Running the example, the accuracy on the training and validation test is printed each epoch and at the end of the classification error rate is printed.

# In[ ]:


# Call the function we defined earlier to create the model
# The trainable parameters within the new model are set by the default initialization methods from Keras.
model = baseline_model()
model.summary()


# In[ ]:


# Train the model using the picture stored in X_train and the corresponding labels stored in Y_train
history = model.fit(X_train, y_train, epochs=20, batch_size=512, verbose=1, validation_split=0.2) #verbose =1 will show the progress bar


# <a id="evaluating"></a>
# # 6. Evaluate the Model Performance on the Test Set
# * X_test include the new digit images the model has never seen
# * y_test include the correct labels of the test set
# 
# <a id="accuracy"></a>
# ## <img align=left src="https://i.ibb.co/QfdFxJr/excellence.png" alt="excellence" height="40" width="40" border="0"> 6.1. Overall Model Accuracy   

# In[ ]:


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0) # scores has [loss] and [accuracy]
#Print out the error rate
print('Simple NN Accuracy: %.2f%%'% (scores[1]*100))
print("Simple NN Error:    %.2f%%"% (100-scores[1]*100))


# <a id="plot"></a>
# ## <img align=left src="https://i.ibb.co/xzWSP1k/plot.png" alt="plot" width="40" height="10" border="0"> 6.2. Plot Model Accuracy and Model Loss by Epoch

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# <a id="recheck"></a>
# ## <img align=left src="https://i.ibb.co/4fBwCsq/search.png" alt="search" height="40" width="40" border="0"> 6.3. Display the Actual Label and the Predicted Label to check Model Performance  

# In[ ]:


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = numpy.argmax(predictions_array)
    # if the algorithm makes the right prediction, it will show a blue text, 
    # otherwise show in red text
    if predicted_label == true_label:
      color = 'blue'
    else:
      color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                  100*numpy.max(predictions_array),
                                  class_names[true_label]),
                                  color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = numpy.argmax(predictions_array)
  
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# In[ ]:


# Make predictions using the model. Each row in the output matrix contain probability of the image belogning to one of 10 numbers.
predictions = model.predict(X_test)


# In[ ]:


# Convert validation observations to one hot vectors
y_test  = numpy.argmax(y_test, axis = 1) 


# In[ ]:


# Load the data and create a list of class names
class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Let's examine the 1st and the 19th image to check for accuracy
image_sample_idxes = [0, 18]
for i in image_sample_idxes:
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions, y_test, X_test)


# Time to check the ones that the algorithm missed!

# In[ ]:


# Time to check the ones that the algorithm missed

def check_wrong(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]

    predicted_label = numpy.argmax(predictions_array)
    if predicted_label == true_label:
        return False
    else: 
        return True


wrong_prediction_images_index = []
for i in range(len(predictions)):
    if check_wrong(i, predictions, y_test, X_test) == True:
        wrong_prediction_images_index.append(i)

print(wrong_prediction_images_index[:10])
print(len(wrong_prediction_images_index))


# In[ ]:


def plot_wrong(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]

    predicted_label = numpy.argmax(predictions_array)
    if predicted_label == true_label:
        pass
    else:
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plot_image(i, predictions, y_test, X_test)

# We are going to check the first 10 images to see which one the model predicted wrong

top_k = 3
for i in wrong_prediction_images_index[:top_k]:
    plot_wrong(i, predictions, y_test, X_test)


# In[ ]:


from ipywidgets import interact, widgets
#application
probs = model.predict(X_test) #Each row in the output matrix contain probability of the image belogning to one of 10 numbers.

predicts = model.predict(X_test).argsort()[:,-1] #To get the number in which the image belongs to, need to find out in which column the maximum probability is present.
img_idx_slider = widgets.IntSlider(value=0, min=0, max=len(X_test) - 1, description="Image index")
@interact(index=img_idx_slider)
def visualize_prediction(index=0):
    fix, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(X_test[index], cmap=plt.cm.binary)
    ax1.set_title("label: %s" % class_names[y_test[index]])
    ax1.set_xlabel("predict: %s" % class_names[predicts[index]])
    ax2.bar(x=[class_names[index] for index in range(10)], height=probs[index]*100)
    plt.xticks()


# <div class="alert alert-block alert-info">
# <b>Note:</b> We have a image index slider as the input widget. Whenever the slider is changed, the visualize_prediction function is called with the value of the slider assigned to the input parameter i. However, the interactive slider doesn't appear to work on Kaggle although the code is still working fine when running locally on Google Colab. 
# </div>

# <a id="findings_and_submission"></a>
# # 7. Findings and Submission
# 
# <a id="findings"></a>
# ## <img align=left src="https://i.ibb.co/mNTpdY5/finding.png" alt="finding" width="30" height="20" border="0"> 7.1. Findings

# Overall, the model turned out very accurate, attaining an accuracy of 97.55% on the test set. When examining some of the predictions the model got wrong, it is not surprising that the model had difficulties with these cases. These records with incorrect predictions are written in a way that makes them partially look like a different number. 
# 
# For example, the model incorrectly predicted a record as an "8" in one case instead of a "3". In that particular example, the "3" had been written with the bottom half as a full enclosed circle. In another example, a "9" was mistaken as a "0" because the "9" curved in on itself so dramatically that the top and bottm were almost connecting. In some examples even a human may have trouble discerning which number was really written. 

# <a id="submission"></a>
# ## <img align=left src="https://i.ibb.co/1TgVvfL/submit.png" alt="submit" height="40" width="50" border="0"> 7.2. Submission

# In[ ]:


# predict the raw test data for submission
predict_x=model.predict(x_test) 
classes_x=numpy.argmax(predict_x,axis=1)


# In[ ]:


# submissions
sub['Label'] = classes_x
sub.to_csv("digit_recognizer.csv", index=False)
sub.head()

