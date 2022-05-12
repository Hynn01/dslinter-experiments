#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import cv2 as cv
from path import Path
import os 
import glob
import tensorflow_hub as hub
import os 
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from keras import applications


# In[ ]:


train_df= pd.read_csv('../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv')
sample_df = pd.read_csv('../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv')


# In[ ]:


def load_dicom(path):
    dicom=pydicom.read_file(path)
    data=dicom.pixel_array
    data=data-np.min(data)
    if np.max(data) != 0:
        data=data/np.max(data)
    data=(data*255).astype(np.uint8)
    return data


# In[ ]:


train_dir='../input/rsna-miccai-brain-tumor-radiogenomic-classification/train'
trainset=[]
trainlabel=[]
trainidt=[]
for i in tqdm(range(len(train_df))):
    idt=train_df.loc[i,'BraTS21ID']
    idt2=('00000'+str(idt))[-5:]
    path=os.path.join(train_dir,idt2,'T1wCE')              
    for im in os.listdir(path):
        img=load_dicom(os.path.join(path,im)) 
        img=cv.resize(img,(64,64)) 
        image=img_to_array(img)
        image=image/255.0
        trainset+=[image]
        trainlabel+=[train_df.loc[i,'MGMT_value']]
        trainidt+=[idt]


# In[ ]:


test_dir='../input/rsna-miccai-brain-tumor-radiogenomic-classification/test'
testset=[]
testidt=[]
for i in tqdm(range(len(sample_df))):
    idt=sample_df.loc[i,'BraTS21ID']
    idt2=('00000'+str(idt))[-5:]
    path=os.path.join(test_dir,idt2,'T1wCE')               
    for im in os.listdir(path):   
        img=load_dicom(os.path.join(path,im))
        img=cv.resize(img,(64,64)) 
        image=img_to_array(img)
        image=image/255.0
        testset+=[image]
        testidt+=[idt]


# In[ ]:


y=np.array(trainlabel)
Y_train=to_categorical(y)
X_train=np.array(trainset)
X_test=np.array(testset)


# In[ ]:


img_height,img_width = 64,64 
num_classes = 2
base_model = applications.resnet.ResNet50(weights= None, include_top=False, input_shape= (img_height,img_width,1))


# In[ ]:


x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.7)(x)
predictions = keras.layers.Dense(num_classes, activation= 'softmax')(x)
model = keras.models.Model(inputs = base_model.input, outputs = predictions)
model.summary()


# In[ ]:


model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


history = model.fit(X_train, Y_train, epochs = 50, batch_size = 60)


# In[ ]:


get_ac = history.history['accuracy']
get_los = history.history['loss']


# In[ ]:


epochs = range(len(get_ac))
plt.plot(epochs, get_ac, 'g', label='Accuracy of Training data')
plt.plot(epochs, get_los, 'r', label='Loss of Training data')
plt.title('Training data accuracy and loss')
plt.legend(loc=0)
plt.figure()
plt.show()


# In[ ]:


y_pred=model.predict(X_test)
pred=np.argmax(y_pred,axis=1)
result=pd.DataFrame(testidt)
result[1]=pred
result.columns=['BraTS21ID','MGMT_value']
result2=result.groupby('BraTS21ID',as_index=False).mean()
result2


# In[ ]:


result2['BraTS21ID']=sample_df['BraTS21ID']
result2['MGMT_value']=result2['MGMT_value'].apply(lambda x:round(x*10)/10)
result2.to_csv('submission.csv',index=False)


# In[ ]:


from IPython.display import FileLink
FileLink(r'submission.csv')

