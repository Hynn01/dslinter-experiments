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


# Здесь мы будем использовать библиотеку Keras. Keras представляет собой надстройку над фреймворками Deeplearning4j, TensorFlow и Theano. Поскольку она хорошо описана, кодирование для задач глубокого 
# обучения будет несложным.

# In[ ]:


import keras


# MNIST – база данных рукописных изображений символов. Одно 
# изобра жение представляет собой квадрат размером 28×28 пикселей со 
# значением насыщенности от 0 до 255

# In[ ]:


from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# Поскольку на входе нейронной сети будет небольшая область значений от 0 до 1, нет необходимости на первом этапе настраивать вес или 
# разряды коэффициентов для обучения. Поскольку максимальная величина пиксела равна 255, после преобразования целочисленного типа в тип с плавающей запятой выполнится операция деления всех 
# данных на 255

# In[ ]:


img_rows, img_cols = 28, 28
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') /255


# Далее настроим выход. Правильной меткой будет целое число от 0 до 
# 9, которым обозначаются изображения, поэтому нужен десятимерный 
# вектор, называющийся one-hot
# При использовании унитарного кодирования только один выход равен 1, а оставшиеся – 0. На выходном слое нейронной сети количество 
# классов (на этот раз 10) будет преобразовано в формат, который будет 
# легко подаваться в качестве учительского сигнала

# In[ ]:


from tensorflow.keras.utils import to_categorical 
Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)


# А теперь определим структуру сверточной нейронной сети. У нас будет 
# по два слоя свертки (фильтр размером 3×3) и пулинга (размер 2×2), выход будет конвертирован в одномерный вектор и перенаправлен на 
# двухслойную нейронную сеть для классификации. В качестве функции 
# активации на выходном слое используется softmax, а на остальных 
# RELU.

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
n_out = len(Y_train[0]) # 10
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),
 activation='relu',
 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(n_out, activation='softmax'))
model.summary()


# Применяя метод summary Keras, посмотрим структуру полученной 
# нейросети

# При помощи метода compile найдем categorical cross entropy и rmsprop, 
# а также проведем обучение методом fit.

# In[ ]:


model.compile(loss = 'categorical_crossentropy',
 optimizer = 'rmsprop',
 metrics = ['accuracy'])
model.fit(X_train, Y_train, epochs=10, batch_size=200)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Точность распознавания – 98,78 %. Она очень высокая.
