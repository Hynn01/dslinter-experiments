#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd 
df = pd.read_csv('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')
df.head()


# In[ ]:


df.drop_duplicates(inplace=True)
df['Target'].value_counts()


# In[ ]:


import os 
print(len(os.listdir('../input/rsna-pneumonia-detection-challenge/stage_2_train_images/')))


# In[ ]:


trainfiles=[]
for file in os.listdir('../input/rsna-pneumonia-detection-challenge/stage_2_train_images/'):
    trainfiles.append(file)


# In[ ]:


print(len(trainfiles))


# In[ ]:


sick= df[df.Target==1]
sick= sick[['patientId','Target']]
sick.head()


# In[ ]:


sick.drop_duplicates(inplace = True)
sick.shape


# In[ ]:


not_sick= df[df.Target ==0]
not_sick= not_sick[['patientId','Target']]
not_sick.head()


# In[ ]:


not_sick.shape


# In[ ]:


get_ipython().system('pip install pydicom')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('mkdir train')


# In[ ]:


cd train


# In[ ]:


get_ipython().system('mkdir sick')
get_ipython().system('mkdir notsick')


# In[ ]:


get_ipython().system('ls')


# In[ ]:





# In[ ]:


get_ipython().system('ls')


# In[ ]:


cd-


# In[ ]:


get_ipython().system('mkdir test')


# In[ ]:


cd  test 


# In[ ]:


get_ipython().system('mkdir sick')
get_ipython().system('mkdir notsick')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


import pydicom, numpy as np
import matplotlib.pyplot as plt
import shutil
import cv2
import os
train_notsick="/kaggle/working/train/notsick/"
test_notsick= "/kaggle/working/test/notsick/"
train_dir = '../input/rsna-pneumonia-detection-challenge/stage_2_train_images/'
i=0
for idx, row in not_sick.iterrows():
    patientId = row['patientId']
    file='%s.dcm' % patientId
    dcm_file = train_dir + file
    ds =  pydicom.dcmread(dcm_file)
    pixel_array_numpy = ds.pixel_array
    dcmfile = '%s.png' % patientId
    i=i+1
    if i<5000 and file in trainfiles:
        cv2.imwrite(os.path.join(train_notsick, dcmfile), pixel_array_numpy)
    elif i >5000 and i<7000 and file in trainfiles:
        cv2.imwrite(os.path.join(test_notsick, dcmfile), pixel_array_numpy)


# In[ ]:


get_ipython().system('ls')


# In[ ]:


cd train/notsick


# In[ ]:


import os 
print(len(os.listdir(test_notsick)))


# In[ ]:


train_sick="/kaggle/working/train/sick/"
test_sick= "/kaggle/working/test/sick/"
i=0
for idx, row in sick.iterrows():
    patientId = row['patientId']
    file ='%s.dcm' % patientId
    dcm_file = train_dir + file
    ds =  pydicom.dcmread(dcm_file)
    pixel_array_numpy = ds.pixel_array
    dcmfile = '%s.png' % patientId
    i=i+1
    if i<4000 and file in trainfiles:
        cv2.imwrite(os.path.join(train_sick, dcmfile), pixel_array_numpy)
    elif i>4000 and file in trainfiles:
        cv2.imwrite(os.path.join(test_sick, dcmfile), pixel_array_numpy)


# In[ ]:


print(trainfiles[0])


# In[ ]:


print(len(os.listdir(test_sick)))


# In[ ]:


from os import listdir
import imageio
path='/kaggle/working/train/sick/'
for file in listdir(path):
    dcm_data = pydicom.read_file(path+file)
    im = dcm_data.pixel_array
    imageio.imwrite(path+file.replace('.dcm','.png'), im)
    


# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()
model.add(Conv2D(512, (3, 3), activation='relu',input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 
model.summary()


# In[ ]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('rsna_model', monitor='loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2,  horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/kaggle/working/train/',  target_size = (256, 256),
                                                 batch_size = 32, class_mode = 'binary')

test_set = test_datagen.flow_from_directory('/kaggle/working/test/', target_size = (256, 256), 
                                            batch_size = 32,  class_mode = 'binary')

model.fit_generator(training_set, steps_per_epoch = 282,epochs = 20,validation_data = test_set,validation_steps = 125
                    ,callbacks=callbacks_list)

