#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow import keras
from keras.layers import Dense, Input
from keras.models import Model, Sequential
from keras.datasets import mnist
import numpy as np
from tensorflow.keras.utils import to_categorical


# In[ ]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, newshape=(60000, 784)).astype('float32')
x_test = np.reshape(x_test, newshape=(10000, 784)).astype('float32')
y_train=to_categorical(y_train,num_classes=10)
y_test=to_categorical(y_test,num_classes=10)
x_train=x_train/255
x_test=x_test/255


# In[ ]:


input_main=Input(shape=(784,))
h1=Dense(units=100, activation='sigmoid')(input_main)
o1=Dense(units=784, activation='sigmoid')(h1)
autoencoder1=Model(inputs=input_main, outputs=o1)
autoencoder1.summary()
autoencoder1.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])


# In[ ]:


print("Training Autoencoder1:")
autoencoder1.fit(x_train,x_train,epochs=5)

autoencoder1_hidden_output=autoencoder1.layers[1].output
trimmed_autoencoder1=Model(inputs=input_main, outputs=autoencoder1_hidden_output)
x_train_ae2=trimmed_autoencoder1.predict(x_train)
x_test_ae2=trimmed_autoencoder1.predict(x_test)
print(x_train_ae2.shape)


inputs_ae2=Input(shape=(100,))
h2=Dense(units=50, activation='sigmoid')(inputs_ae2)
o2=Dense(units=100, activation='sigmoid')(h2)
autoencoder2=Model(inputs=inputs_ae2, outputs=o2)
autoencoder2.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])


# In[ ]:


print("Training Autoencoder2:")
autoencoder2.fit(x_train_ae2, x_train_ae2, epochs=5)

autoencoder2_hidden_output=autoencoder2.layers[1].output
trimmed_autoencoder2=Model(inputs=inputs_ae2, outputs=autoencoder2_hidden_output)
x_train_clf=trimmed_autoencoder2.predict(x_train_ae2)
x_test_clf=trimmed_autoencoder2.predict(x_test_ae2)
print(x_train_clf.shape)

inputs_clf=Input(shape=(50,))
f_output=Dense(units=10, activation='softmax')(inputs_clf)
clf=Model(inputs=inputs_clf, outputs=f_output)
clf.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(clf.layers[1].get_weights()[0].shape)


# In[ ]:


print("Training Classifier:")
clf.fit(x_train_clf, y_train, epochs=5)


print(clf.evaluate(x_test_clf,y_test))


new_model=Sequential()
new_model.add(autoencoder1.layers[0])
new_model.add(autoencoder1.layers[1])
new_model.add(autoencoder2.layers[1])
new_model.add(clf.layers[-1])
new_model.summary()
new_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
print(new_model.layers[1].get_weights()[0].shape)
print(autoencoder2.layers[1].get_weights()[0].shape)
print("Fine tuning:")
new_model.fit(x_train, y_train, epochs=5)


print(new_model.evaluate(x_test, y_test))

