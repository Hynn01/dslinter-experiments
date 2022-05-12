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


# In[ ]:


get_ipython().system(' pip install py7zr')


# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


from py7zr import unpack_7zarchive
import shutil

shutil.register_unpack_format('7zip',['.7z'],unpack_7zarchive)


# In[ ]:


#unpack train images to temp directory
shutil.unpack_archive('/kaggle/input/cifar-10/train.7z','/kaggle/temp/')


# In[ ]:


train_labels = pd.read_csv('/kaggle/input/cifar-10/trainLabels.csv', header= 'infer')

#unique labels
classes = train_labels['label'].unique()

print(classes)

import os

if not os.path.exists("/kaggle/temp/validation/"):
    os.mkdir("/kaggle/temp/validation/")

parent_path_train = "/kaggle/temp/train/"
parent_path_validation = "/kaggle/temp/validation/"
parent_path_test = "/kaggle/temp/test/"


for class1 in classes:
    path_train = os.path.join(parent_path_train,class1)
    if not os.path.exists(path_train):
        os.mkdir(path_train)
        
    path_validation = os.path.join(parent_path_validation,class1)
    if not os.path.exists(path_validation):
        os.mkdir(path_validation)
        
for (int_ind,row) in train_labels.iterrows():
    id = str(row["id"]) + ".png"
    source_path = os.path.join(parent_path_train,id)
    
    p = np.random.random()
    if p <= 0.80:
        target_path = os.path.join(parent_path_train,row["label"],id)
        os.replace(source_path,target_path)
    else:
        target_path = os.path.join(parent_path_validation,row["label"],id)
        os.replace(source_path,target_path)


# In[ ]:


import tensorflow as tf
from tensorflow import keras

from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Activation,BatchNormalization,GlobalAveragePooling2D,Flatten
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


def my_model():
    model = Sequential()
    model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='valid',activation=None,use_bias=False,input_shape=(32,32,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(filters=48, kernel_size=(3,3), strides=(1,1), padding='valid', activation=None, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid', activation=None, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(filters=80, kernel_size=(3,3), strides=(1,1), padding='valid', activation=None, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding='valid', activation=None, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='valid', activation=None, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(units=10))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[ ]:


model = my_model()


# In[ ]:


train_datagen = ImageDataGenerator(featurewise_center= False,
                                  samplewise_center= False,
                                  featurewise_std_normalization=False,
                                  samplewise_std_normalization=False,
                                  zca_whitening=False,
                                  rotation_range=10,
                                  zoom_range=0.1,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  horizontal_flip=False,
                                  vertical_flip=False,
                                  rescale=1./255
                                  )

train_gen = train_datagen.flow_from_directory(directory='/kaggle/temp/train/',target_size=(32,32),batch_size=128)
validation_datagen = ImageDataGenerator(rescale=1./255)
valid_gen = validation_datagen.flow_from_directory(directory='/kaggle/temp/validation/',target_size=(32,32),batch_size=128)


# In[ ]:


model.fit(train_gen,epochs=50,validation_data=valid_gen,steps_per_epoch=train_gen.n//train_gen.batch_size,
         validation_steps= valid_gen.n//valid_gen.batch_size,workers=8,use_multiprocessing=True)


# In[ ]:


shutil.unpack_archive('/kaggle/input/cifar-10/test.7z','/kaggle/temp/test')


# In[ ]:


shutil.unregister_unpack_format('7zip')


# In[ ]:


test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(directory='/kaggle/temp/test',target_size=(32,32),batch_size=64,class_mode=None,shuffle=False)


# In[ ]:


import os

os.chdir('/kaggle/temp/test/test')
print(os.getcwd())


# In[ ]:


test_gen.filenames


# In[ ]:


predictions_final = model.predict(test_gen)


# In[ ]:


predictions_final[1]


# In[ ]:


print(type(train_gen.class_indices))
print(train_gen.class_indices)

classes = {value:key for (key,value) in train_gen.class_indices.items()}
print(classes)

predicted_classes=np.empty(shape=300000,dtype=np.dtype('U20'))

ind=0
for i in predictions_final.tolist():
    predicted_classes[ind]=classes[np.argmax(i)]
    ind=ind+1


# In[ ]:


predicted_classes[9]


# In[ ]:


filenames_wo_ext = []
for fname in test_gen.filenames:
    filenames_wo_ext.append(int(fname.split(sep="/")[-1].split(sep=".")[0])-1)

# print(filenames_wo_ext)
# print(len(predicted_classes))

predicted_classes_final = np.empty(shape=300000,dtype=np.dtype('U20'))
# for i in range(0,300000) :
#     predicted_classes_final[i]= predicted_classes[i]
predicted_classes_final[filenames_wo_ext]=predicted_classes


# In[ ]:


predicted_classes_final


# In[ ]:


sub = pd.read_csv('/kaggle/input/cifar-10/sampleSubmission.csv',header='infer')
sub.info()


# In[ ]:


sub['label'] = predicted_classes_final
sub.to_csv('/kaggle/working/submission.csv',index=False)


# In[ ]:


sub.head(10)

