#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from zipfile import ZipFile
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


import cv2
import random
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt


# In[ ]:


import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


# In[ ]:



with ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip', 'r') as zip:
    #zip.printdir()
    zip.extractall()
    print('done')


# In[ ]:


with ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip', 'r') as zip:
    #zip.printdir()
    zip.extractall()
    print('done')


# In[ ]:


PATH = '/kaggle/working/train'
filename = os.listdir(PATH)
IMG_SIZE = 100

plt.figure(figsize=(10,10))
for i in range(1, 7):
    img_array = cv2.imread(os.path.join(PATH, filename[i]))
    resize_image = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    plt.subplot(3,3, i)
    image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    #plt.figure()
    plt.axis('off')
    plt.imshow(image)    


# In[ ]:



training_data = []
IMG_SIZE = 100
path = '/kaggle/working/train'
filenames = os.listdir('/kaggle/working/train')
for img in tqdm(filenames):
    try:
        if img.find('cat') == -1:
            category = 0
        else:
            category = 1
        
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        training_data.append([new_array, category])
    except Exception as e:
        pass


# In[ ]:


plt.figure(figsize=(10,10))
for i in range(1, 7):
    plt.subplot(3,3,i)
    plt.axis('off')
    if training_data[10+i][1] == 0:
        plt.title('Dog')
    else:
        plt.title('Cat')
    plt.imshow(training_data[10+i][0], cmap='gray_r')


# In[ ]:


testing_data = []
IMG_SIZE = 100

path = '/kaggle/working/test'
filenames = os.listdir('/kaggle/working/test')
for img in tqdm(filenames):
    try:        
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        testing_data.append([new_array])
    except Exception as e:
        pass


# In[ ]:


len(training_data)


# In[ ]:


random.shuffle(training_data)


# In[ ]:



X = []
y = []

for features, label  in training_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# In[ ]:



pickle_out =open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[ ]:


pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)


# In[ ]:


X = X/255.0

X = np.array(X)
y = np.array(y)


# In[ ]:


len(X)


# In[ ]:


len(y)


# In[ ]:


X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=50)


# In[ ]:


len(X_train)


# In[ ]:


len(y_train)


# In[ ]:


model = Sequential()

model.add(Conv2D(256, (3,3), input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))
model.add(Dense(1))

model.add(Activation('sigmoid'))


# In[ ]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'],)

history = model.fit(X_train, y_train, batch_size=32, epochs=15, validation_data = (x_test, y_test))


# In[ ]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:





# In[ ]:


# Plotting our loss charts
# Use the history object we created to get our saved perormace result.
history_dict = history.history

# Extract the loss and validation losses
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values)+1)

line1 = plt.plot(epochs, val_loss_values, label ='Validation/Test Loss')
line2 = plt.plot(epochs, loss_values, label='Training Loss')
plt.setp(line1, linewidth = 2.0, marker='+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker='4', markersize=10.0)
plt.legend()
plt.grid(True)
plt.show()


# ### Accuracy Charts

# In[ ]:


history_dict = history.history

acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(loss_values)+1)

line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Accuracy')
line2 = plt.plot(epochs, acc_values, label ='Training Accuracy')

plt.setp(line1, linewidth=2.0, marker='+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker='4', markersize=10.0)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()


# In[ ]:


IMG_SIZE = 100
test = []

for features  in testing_data:
    test.append(features)
    
test = np.array(test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# In[ ]:


test = test/255.0

test = np.array(test)


# In[ ]:


prediction = model.predict(test) 


# In[ ]:


#predict_xtest = model.predict(x_test) 


# In[ ]:


#predict_xtest 


# In[ ]:


prediction


# In[ ]:


prediction.shape


# In[ ]:


my_submission = pd.read_csv('/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')


# In[ ]:


my_submission.head()


# In[ ]:


my_submission['label'] = prediction


# In[ ]:


my_submission['label'] = my_submission['label'].map(lambda x: 1 if x >= 0.5 else 0 )


# In[ ]:


my_submission.head()


# In[ ]:


my_submission['label'] = my_submission['label'].round(1)


# In[ ]:


my_submission.to_csv('my_submission.csv', index=False)


# In[ ]:




