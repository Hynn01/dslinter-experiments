#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


#imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import numpy as np
import tensorflow as tf


# # Explore the data

# In[ ]:


# define the important paths
train_dir = '../input/chest-xray-pneumonia/chest_xray/train/'
val_dir = '../input/chest-xray-pneumonia/chest_xray/val/'
test_dir = '../input/chest-xray-pneumonia/chest_xray/test/'


# In[ ]:


# viualize some images from the training data
# visualize the Normal Class
W = 5
H = 5
fig, axes = plt.subplots(W, H, figsize = (17,17))

axes = axes.ravel() # flaten the matrix into array
for i in np.arange(0, W * H): 
    label ='NORMAL'
    class_dir = os.path.join(train_dir,label)
    # Select a random image
    image = random.choice(os.listdir(class_dir))
    # read and display an image with the selected index    
    img = plt.imread(os.path.join(class_dir,image))
    axes[i].imshow( img )
    axes[i].set_title(label, fontsize = 8) # the label
    axes[i].axis('off')


# In[ ]:


# viualize some images from the training data
# visualize the PNEUMONIA Class
W = 5
H = 5
fig, axes = plt.subplots(W, H, figsize = (17,17))

axes = axes.ravel() # flaten the matrix into array
for i in np.arange(0, W * H): 
    label ='PNEUMONIA'
    class_dir = os.path.join(train_dir,label)
    # Select a random image
    image = random.choice(os.listdir(class_dir))
    # read and display an image with the selected index    
    img = plt.imread(os.path.join(class_dir,image))
    axes[i].imshow( img )
    axes[i].set_title(label, fontsize = 8) # the label
    axes[i].axis('off')


# There are several image sizes. We need to rescale the images later before fitting the model

# In[ ]:


## count number of images in each class for training data
DF = pd.DataFrame(columns=['class','count'])
DF['class']=pd.Series([os.listdir(train_dir)[x] for x in range(0,2)])
DF['count']=pd.Series([len(os.listdir(os.path.join(train_dir,os.listdir(train_dir)[x]))) for x in range(0,2)])
plt.figure(figsize=(8,6))
g=sns.barplot(x='class', y='count',data=DF)
g.set_xticklabels(g.get_xticklabels(), rotation=0)
plt.tight_layout()


# In[ ]:


plt.figure(figsize=(8,6))
plt.tight_layout()
plt.pie(DF['count'],
        labels=DF['class'],
        autopct='%1.1f%%')
plt.axis('equal')
plt.title('Proportion of each observed category')
plt.show()


# The data is imbalanced. only 1/4 of the images are in Normal class and the rest is Pneumonia. 

# # Data Augmantation

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator( rescale = 1.0/255.,
                                  rotation_range=20 ,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  brightness_range=[0.6,0.9],
                                  fill_mode='nearest')
test_datagen = ImageDataGenerator( rescale = 1.0/255)


# In[ ]:


# --------------------
# Flow training images in batches of 32 using train_datagen generator
# --------------------
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=32,
                                                    class_mode='binary',
                                                    target_size=(300, 300))     
# --------------------
# Flow validation images using test_datagen generator
# --------------------
validation_generator =  test_datagen.flow_from_directory(val_dir,
                                                         shuffle=False,
                                                         class_mode  = 'binary',
                                                         target_size = (300, 300))

# --------------------
# Flow validation images using test_datagen generator
# --------------------
test_generator =  test_datagen.flow_from_directory(test_dir,
                                                         shuffle=False,
                                                         class_mode  = 'binary',
                                                         target_size = (300, 300))


# In[ ]:


train_generator.class_indices


# In[ ]:


# viualize some images after the augmentation
x_batch, y_batch = next(train_generator)
W = 5
H = 5
fig, axes = plt.subplots(W, H, figsize = (17,17))

axes = axes.ravel() # flaten the matrix into array
for i in np.arange(0, W * H): 

    # Select a random image
    image = x_batch[i]
    # read and display an image with the selected index    
    axes[i].imshow( image )
    axes[i].set_title(y_batch[i], fontsize = 8) # the label
    axes[i].axis('off')


# # Transfer Leaarning

# In[ ]:


# Model

Vgg = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False,input_shape=(300, 300, 3))
Vgg.trainable = False
#Create new model on top
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout

model=tf.keras.models.Sequential()
model.add(Vgg)
model.tf.keras.layers.GlobalAveragePooling2D()
#model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


# In[ ]:


Vgg.summary()


# The learning rate controls how much the weights are updated according to the estimated error. Choose too small of a value and your model will train forever and likely get stuck. Opt for a too large learning rate and your model might skip the optimal set of weights during training.
# 
# 

# In[ ]:


opt = tf.keras.optimizers.Adam()
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)
model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=opt)
history = model.fit(train_generator ,epochs = 10 , validation_data = validation_generator, steps_per_epoch =163)


# In[ ]:


def evaluation(history):
    # evaluation
    #-----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    #-----------------------------------------------------------
    acc      = history.history['accuracy']
    val_acc  = history.history['val_accuracy']
    loss     = history.history['loss']
    val_loss = history.history['val_loss']

    epochs   = range(len(acc)) # Get number of epochs

    #------------------------------------------------
    # Plot training and validation accuracy per epoch
    #------------------------------------------------
    plt.figure(figsize=(10,6))
    plt.plot  ( epochs,     acc )
    plt.plot  ( epochs, val_acc )
    plt.title ('Training and validation accuracy')
    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    plt.figure(figsize=(10,6))
    plt.plot  ( epochs,     loss )
    plt.plot  ( epochs, val_loss )
    plt.title ('Training and validation loss'   )


# In[ ]:



evaluation(history)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report

#Confution Matrix and Classification Report
Y_pred = model.predict(validation_generator)
model.evaluate(validation_generator)
y_pred= np.where(Y_pred>0.5, 1, 0)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = os.listdir(train_dir)
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
plt.figure(figsize=(10,8))
plt.title('Predicted classes', size=14)
sns.heatmap(confusion_matrix(validation_generator.classes, y_pred), annot=True, fmt = '.0f',linewidths=.5)
plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report

#Confution Matrix and Classification Report
Y_pred = model.predict(test_generator)
model.evaluate(test_generator)
y_pred= np.where(Y_pred>0.5, 1, 0)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = os.listdir(train_dir)
print(classification_report(test_generator.classes, y_pred, target_names=target_names))
plt.figure(figsize=(10,8))
plt.title('Predicted classes', size=14)
sns.heatmap(confusion_matrix(test_generator.classes, y_pred), annot=True, fmt = '.0f',linewidths=.5)
plt.show()

