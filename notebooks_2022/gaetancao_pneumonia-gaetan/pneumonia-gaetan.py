#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPool2D

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical


# In[ ]:


import cv2
import os
import glob
import gc

def lire_images(img_dir, xdim, ydim, nmax=5000) :
    """ 
    Lit les images dans les sous répertoires de img_dir
    nmax images lues dans chaque répertoire au maximum
    Renvoie :
    X : liste des images lues, matrices xdim*ydim
    y : liste des labels numériques
    label : nombre de labels
    label_names : liste des noms des répertoires lus
    """
    label = 0
    label_names = []
    X = []
    y=[]
    for dirname in os.listdir(img_dir):
        print(dirname)
        label_names.append(dirname)
        data_path = os.path.join(img_dir + "/" + dirname,'*g')
        files = glob.glob(data_path)
        n=0
        for f1 in files:
            if n>nmax : break
            img = cv2.imread(f1) # Lecture de l'image dans le repertoire
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Conversion couleur RGB
            img = cv2.resize(img, (xdim,ydim)) # Redimensionnement de l'image
            X.append(np.array(img)) # Conversion en tableau et ajout a la liste des images
            y.append(label) # Ajout de l'etiquette de l'image a la liste des etiquettes
            n=n+1
        print(n,' images lues')
        label = label+1
    X = np.array(X)
    y = np.array(y)
    gc.collect() # Récupération de mémoire
    return X,y, label, label_names


# In[ ]:


X_train,y_train,Nombre_classes,Classes = lire_images("../input/chest-xray-pneumonia/chest_xray/train", 224, 224, 1000)
X_test,y_test,Nombre_classes,Classes = lire_images("../input/chest-xray-pneumonia/chest_xray/test", 224, 224, 1000)
X_val,y_val,Nombre_classes,Classes = lire_images("../input/chest-xray-pneumonia/chest_xray/val", 224, 224, 1000)


# In[ ]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(X_val.shape)
print(y_val.shape)
print(Nombre_classes)


# In[ ]:


plt.figure(figsize=(10,20))
for i in range(0, 10) :
    plt.subplot(10,5,i+1)
    i *= -1
    plt.axis('off')
    plt.imshow(X_train[i])
    plt.title(Classes[int(y_train[i])])


# In[ ]:


X_train = X_train / 255
X_test = X_test / 255
X_val = X_val / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)


# In[ ]:


model = Sequential()
model.add(Conv2D(16, (7, 7), input_shape=(224, 224, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(Nombre_classes, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


train = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=128,
    verbose=1
)


# In[ ]:


scores = model.evaluate(X_test, y_test, verbose=0)
print("Score : %.2f%%" % (scores[1]*100))


# In[ ]:


def plot_scores(train) :
    accuracy = train.history['accuracy']
    val_accuracy = train.history['val_accuracy']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'b', label='Score apprentissage')
    plt.plot(epochs, val_accuracy, 'r', label='Score validation')
    plt.title('Scores')
    plt.legend()
    plt.show()


# In[ ]:


plot_scores(train)

