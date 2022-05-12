#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tqdm import trange
from keras import Model,Sequential
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Activation, BatchNormalization, Flatten


# In[ ]:


(X_train,Y_train), (X_test,Y_test)=mnist.load_data()
# Dimension of Training and Test set
print('',f'Training set: {X_train.shape}, {Y_train.shape}', f'Testing set: {X_test.shape}, {Y_test.shape}','',sep='\n'+('*'*40)+'\n')


# In[ ]:


X_train = X_train.reshape(X_train.shape[0],28*28)
X_test = X_test.reshape(X_test.shape[0],28*28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train=(X_train-127.5)/127.5
X_test=(X_test-127.5)/127.5


# In[ ]:


# Convert Ground Truth to one hot encode vector 
Y_train = tf.keras.utils.to_categorical(Y_train)
Y_test = tf.keras.utils.to_categorical(Y_test)
print(Y_train.shape)


# In[ ]:


Dim=100


# In[ ]:


Generator=Sequential()
Generator.add(Dense(input_shape=(Dim,),units=128,activation='relu',kernel_initializer=keras.initializers.RandomNormal(stddev=0.02)))
Generator.add(Dense(units=256,activation='relu'))
Generator.add(Dense(units=512,activation='relu'))
Generator.add(Dense(units=1024,activation='relu'))
Generator.add(Dense(units=784,activation='tanh'))


# In[ ]:


Generator.summary()


# In[ ]:


Discriminator=Sequential()
Discriminator.add(Dense(input_shape=(784,),units=512,activation='relu'))
Discriminator.add(Dropout(0.3))
Discriminator.add(Dense(units=256,activation='relu'))
Discriminator.add(Dropout(0.3))
Discriminator.add(Dense(units=128,activation='relu'))
Discriminator.add(Dropout(0.3))
Discriminator.add(Dense(units=1,activation='sigmoid'))


# In[ ]:


Discriminator.summary()


# In[ ]:


adam=keras.optimizers.Adam(learning_rate=0.0002,beta_1=0.5)
Discriminator.compile(loss='binary_crossentropy',optimizer=adam)


# In[ ]:


generator_input=Input(shape=(Dim,))
Discriminator.trainable=False
x=Generator(generator_input)
GAN_Output=Discriminator(x)
GAN=Model(inputs=generator_input,outputs=GAN_Output)
GAN.compile(loss='binary_crossentropy',optimizer='adam')


# In[ ]:


y=np.random.normal(0,1,size=(1,Dim))
g_n=Generator.predict(y)
g_n=g_n.reshape(1,28,28)
plt.imshow(g_n[0],interpolation='nearest')


# In[ ]:


discriminator_loss=[]
generative_loss=[]
def plot_Loss(epoch):
  plt.figure(figsize=(10,10))
  plt.plot(discriminator_loss,label='Discriminative Loss')
  plt.plot(generative_loss,label='Generator Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()

def plot_Generated_Images(epoch,examples=16,dim=(4,4),figsize=(10,10)):
  noise=np.random.normal(0,1,size=(examples,Dim))
  generated_images=Generator.predict(noise)
  generated_images=generated_images.reshape(examples,28,28)
  plt.figure(figsize=figsize)
  for i in range(0,generated_images.shape[0]):
    plt.subplot(dim[0],dim[1],i+1)
    plt.imshow(generated_images[i],interpolation='nearest',cmap='gray_r',)
  plt.tight_layout()
  


# In[ ]:


def train(epochs=1,batch_size=128):
  batch_count=X_train.shape[0]//batch_size
  print('',f'Total_Epochs: {epochs}\n',f'Batch Size: {batch_size}\n',f'Batches per epoch: {batch_count}\n')

  for e in range(1,epochs+1):
    print('',f'Epoch {e}')
    for min_batch in trange(batch_count):
        
      noise=np.random.normal(0,1,size=[batch_size,Dim])
      imageBatch=X_train[np.random.randint(0,X_train.shape[0],size=batch_size)]
        
#       print(imageBatch.shape)
        
      #Generate Mnist IMAGE
      generated_Images=Generator.predict(noise)
      imageBatch=imageBatch.reshape(batch_size,784)
      X=np.concatenate([imageBatch,generated_Images])
      #Labels for generated and real data
      ydis=np.zeros(2*batch_size)
      #One Sided Label Smoothing instead of  label it 0.9
      ydis[:batch_size]=0.9

      shuffle=np.arange(2*batch_size)
      np.random.shuffle(shuffle)
      
      X=X[shuffle]
      ydis=ydis[shuffle]

      #Train Discriminator
      Discriminator.trainable=True
      d_loss=Discriminator.train_on_batch(X,ydis)

      #Train Generator
      noise=np.random.normal(0,1,size=[batch_size,Dim])
      yGen=np.ones(batch_size)
      Discriminator.trainable=False
      g_loss=GAN.train_on_batch(noise,yGen)
      
      #Append Loss of current epoch for ploting
      discriminator_loss.append(d_loss)
      generative_loss.append(g_loss)

    if e==1 or e==35 or e==50:
      plot_Generated_Images(e,100,(10,10))
    


# In[ ]:


train(50,256)


# In[ ]:





# In[ ]:




