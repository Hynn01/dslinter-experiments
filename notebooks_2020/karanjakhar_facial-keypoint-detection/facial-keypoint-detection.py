#!/usr/bin/env python
# coding: utf-8

#  <h2>Facial Keypoint Detection</h2>         
#  First of all let's discuss what we are given.        
# We are given three CSV files.        
# training.csv :- Its has coordinates of facial keypoints like left eye, rigth eye etc and also the image.      
# test.csv :- Its has image only and we have to give coordinates of various facial keypoints by looking at third csv file which is IdLookupTable.csv     
# Rest everything is explained below.      
# **I would really appreciate if you could upvote this kernel.**
# 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from time import sleep
import os


# In[ ]:


Train_Dir = '../input/training/training.csv'
Test_Dir = '../input/test/test.csv'
lookid_dir = '../input/IdLookupTable.csv'
train_data = pd.read_csv(Train_Dir)  
test_data = pd.read_csv(Test_Dir)
lookid_data = pd.read_csv(lookid_dir)
os.listdir('../input')


# Lets explore our dataset

# In[ ]:


train_data.head().T


# Lets check for missing values

# In[ ]:


train_data.isnull().any().value_counts()


# So there are missing values in 28 columns. We can do two things here one remove the rows having missing values and another is the fill missing values with something. I used two option as removing rows will reduce our dataset. 
# I filled the missing values with the previous values in that row.

# In[ ]:



train_data.fillna(method = 'ffill',inplace = True)
#train_data.reset_index(drop = True,inplace = True)


# Lets check for missing values now

# In[ ]:


train_data.isnull().any().value_counts()


# As there is no missing values we can now separate the labels and features.
# The image is our feature and other values are labes that we have to predict later.
# As image column values are in string format and there is also some missing values so we have to split the string by space and append it and also handling missing values

# In[ ]:



imag = []
for i in range(0,7049):
    img = train_data['Image'][i].split(' ')
    img = ['0' if x == '' else x for x in img]
    imag.append(img)
    
    


# Lets reshape and convert it into float value.

# In[ ]:


image_list = np.array(imag,dtype = 'float')
X_train = image_list.reshape(-1,96,96,1)


# Lets see what is the first image.

# In[ ]:


plt.imshow(X_train[0].reshape(96,96),cmap='gray')
plt.show()


# Now lets separate labels.

# In[ ]:


training = train_data.drop('Image',axis = 1)

y_train = []
for i in range(0,7049):
    y = training.iloc[i,:]

    y_train.append(y)
y_train = np.array(y_train,dtype = 'float')


# As our data is ready for training , lets define our model. I am using keras and simple dense layers. For loss function I am using 'mse' ( mean squared error ) as we have to predict new values. Our result evaluted on the basics of 'mae' ( mean absolute error ) . 

# In[ ]:


from keras.layers import Conv2D,Dropout,Dense,Flatten
from keras.models import Sequential

model = Sequential([Flatten(input_shape=(96,96)),
                         Dense(128, activation="relu"),
                         Dropout(0.1),
                         Dense(64, activation="relu"),
                         Dense(30)
                         ])



# In[ ]:


from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D


# In[ ]:


model = Sequential()

model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
# model.add(BatchNormalization())
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())


model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30))
model.summary()


# In[ ]:


model.compile(optimizer='adam', 
              loss='mean_squared_error',
              metrics=['mae'])


# Now our model is defined and we will train it by calling fit method. I ran it for 500 iteration keeping batch size and validtion set size as 20% ( 20% of the training data will be kept for validating the model ).

# In[ ]:


model.fit(X_train,y_train,epochs = 50,batch_size = 256,validation_split = 0.2)


# Now lets prepare our testing data

# In[ ]:


#preparing test data
timag = []
for i in range(0,1783):
    timg = test_data['Image'][i].split(' ')
    timg = ['0' if x == '' else x for x in timg]
    
    timag.append(timg)


# Reshaping and converting 

# In[ ]:


timage_list = np.array(timag,dtype = 'float')
X_test = timage_list.reshape(-1,96,96,1) 


# Lets see first image in out test data

# In[ ]:


plt.imshow(X_test[0].reshape(96,96),cmap = 'gray')
plt.show()


# Lets predict our results

# In[ ]:


pred = model.predict(X_test)


# Now the last step is the create our submission file keeping in the mind required format.
# There should be two columns :- RowId and Location
# Location column values should be filled according the lookup table provided ( IdLookupTable.csv)
# 

# In[ ]:


lookid_list = list(lookid_data['FeatureName'])
imageID = list(lookid_data['ImageId']-1)
pre_list = list(pred)


# In[ ]:


rowid = lookid_data['RowId']
rowid=list(rowid)


# In[ ]:


feature = []
for f in list(lookid_data['FeatureName']):
    feature.append(lookid_list.index(f))


# In[ ]:


preded = []
for x,y in zip(imageID,feature):
    preded.append(pre_list[x][y])


# In[ ]:


rowid = pd.Series(rowid,name = 'RowId')


# In[ ]:


loc = pd.Series(preded,name = 'Location')


# In[ ]:


submission = pd.concat([rowid,loc],axis = 1)


# In[ ]:


submission.to_csv('face_key_detection_submission.csv',index = False)


# In[ ]:




