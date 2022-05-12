#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
plt.rcParams['figure.figsize'] = (12,7)

# Input data files are available in the "../input/" directory.

import os
print(os.listdir("../input/cell-images-for-detecting-malaria/cell_images"))


# In[ ]:


infected = os.listdir("../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized")
infected_path = "../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized"
print("Length of infected data = ",len(infected),'images')
uninfected = os.listdir("../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected")
uninfected_path = "../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected"
print("Length of uninfected data = ",len(uninfected),'images')


# <h3>**Data Visualisation - ** </h3>
# 
# <h4>&nbsp;&nbsp;&nbsp;&nbsp; Infected Data - </h4>
# 

# In[ ]:


for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(cv2.imread(infected_path+'/'+infected[i]))
    plt.title('PARASITIZED CELL')
    plt.tight_layout()
plt.show()


# <h4>&nbsp;&nbsp;&nbsp;&nbsp; Uninfected Data - </h4>

# In[ ]:


for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(cv2.imread(uninfected_path+'/'+uninfected[i]))
    plt.title('UNINFECTED CELL')
    plt.tight_layout()
plt.show()


# As we can see that there is clear difference between infected and uninfected cells. There is a small clot inside infected cells while unifected cells are clean. 
# Now, let's explore the dimension of cells as we have to reshape them before feeding them to algorithm.

# In[ ]:


dim1 = []
dim2 = []
for file in infected:
    try:
        imag = imread(infected_path+'/'+file)
        d1,d2,colors = imag.shape
        dim1.append(d1)
        dim2.append(d2)
    except:
        None


# In[ ]:


sns.jointplot(dim1,dim2)


# Let's check the mean of dimensions. We can use this value to reshape all the images.

# In[ ]:


print('Mean of X dimensions - ',np.mean(dim1))
print('Mean of Y dimensions - ',np.mean(dim2))


# Also let's check if the array values of images are normalised.

# In[ ]:


cv2.imread(infected_path+'/'+infected[0]).max()


# So, values are not normalised too.
# Let's create a ImageGenerator and we will use this to divide the data into train and validation set, normalise the dataset and reshape the imeges - 

# In[ ]:


img_shape = (130,130,3)
image_gen = ImageDataGenerator(rotation_range = 20,
                              width_shift_range = 0.1,
                              height_shift_range=0.1,
                              rescale=1 / 255,
                              shear_range=0.1,
                              zoom_range=0.1,
                              horizontal_flip=True,
                              fill_mode='nearest',
                              validation_split=0.2)


# In[ ]:


image_gen.flow_from_directory('../input/cell-images-for-detecting-malaria/cell_images/cell_images')


# Let's generate the test and validation dataset - 
# 

# In[ ]:


train = image_gen.flow_from_directory('../input/cell-images-for-detecting-malaria/cell_images/cell_images',
                                     target_size =img_shape[:2],
                                     color_mode='rgb',
                                     batch_size = 16,
                                     class_mode='binary',shuffle=True,
                                     subset="training")

validation = image_gen.flow_from_directory('../input/cell-images-for-detecting-malaria/cell_images/cell_images',
                                     target_size = img_shape[:2],
                                     color_mode='rgb',
                                     batch_size = 16,
                                     class_mode='binary',
                                     subset="validation",shuffle=False)


# In[ ]:


train.class_indices


# So, classes 0 are labelled as Parasitized and 1 as Uninfected.

# Let's create the model - 

# In[ ]:


# Model 1 ---
model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape = (130,130,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[ ]:


model.summary()


# Adding Early Stopping Parameter - 

# In[ ]:


early = EarlyStopping(monitor='val_loss',patience=2,verbose=1)


# In[ ]:


model.metrics_names


# In[ ]:


model.fit_generator(train,
                   epochs=20,
                   validation_data=validation,
                   callbacks=[early])


# In[ ]:


losses = pd.DataFrame(model.history.history)


# In[ ]:


losses[['loss','val_loss']].plot()


# In[ ]:


losses[['accuracy','val_accuracy']].plot()


# In[ ]:


predictions = model.predict_generator(validation)


# In[ ]:


predictions = predictions>0.5 # The most important factor, directly control precision and recall.


# In[ ]:


print('Confusion Matrix: \n',confusion_matrix(validation.classes,predictions),'\n')
print('Classification Report: \n\n',classification_report(validation.classes,predictions))


# Saving the model - 

# In[ ]:


model.save('model.h5')


# Let's see how to use this model - 

# In[ ]:


img = image.load_img(infected_path+'/'+infected[22],target_size = img_shape)
img


# In[ ]:


img_arr = image.img_to_array(img)


# In[ ]:


model.predict_classes(img_arr.reshape(1,130,130,3))


# So, our model predicted this image to be of class 0 which is Parasitized and this image is of infected cell. So our model prediction is right.

# In[ ]:





# In[ ]:




