#!/usr/bin/env python
# coding: utf-8

# ## MDS-MISIS-DL Car classification
# В третьем соревновании вам снова нужно решить задачу компьютерного зрения. На сайтах по продаже подержаных автомобилей необходимо верификация того, что на фото к объявлению находится именно та машина. Один из способов проверить это - классифицировать, какая марка машины изображена на фото.
# 
# Вам будет доступно несколько тысяч изображений автомобилей 10 марок, которые соответствуют числам от 0 до 9.

# In[ ]:


# Будем использовать в работе оптимизированные сети (EfficientNets) для повышения accuracy
get_ipython().system('pip install -q efficientnet')


# In[ ]:


# Игнорирование предупреждений
import warnings
warnings.filterwarnings('ignore')

# Загрузка необходимых библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import scipy.io
import tarfile
import zipfile
import csv
import sys
import os


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as M
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as C
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import optimizers
import efficientnet.tfkeras as efn

from sklearn.model_selection import train_test_split

import PIL
from PIL import ImageOps, ImageFilter
#увеличим дефолтный размер графиков
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5
get_ipython().run_line_magic('matplotlib', 'inline')

print(os.listdir("../input"))
print('Python       :', sys.version.split('\n')[0])
print('Numpy        :', np.__version__)
print('Tensorflow   :', tf.__version__)
print('Keras        :', keras.__version__)


# In[ ]:


# Проверка работы GPU
tf.test.gpu_device_name()


# In[ ]:


# Задаваемые параметры для задачи
EPOCHS               = 8
BATCH_SIZE           = 16
LR                   = 1e-4
VAL_SPLIT            = 0.15 #15%
RANDOM_SEED          = 42
CLASS_NUM            = 10
IMG_SIZE             = 250
IMG_CHANNELS         = 3
INPUT_SHAPE          = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)

# Определение директорий с исходными данными
DATA_PATH = '../input/mds-misis-dl-car-classificationn/'
DATA_DIR = '../input/mds-misis-dl-car-classificationn/train/train'
TEST_DIR = '../input/mds-misis-dl-car-classificationn/test/test_upload'
PATH = "../working/car/"


# In[ ]:


# Загрузим исходные данные
train_df = pd.read_csv(DATA_PATH+"train.csv")
sample_submission = pd.read_csv(DATA_PATH+"sample-submission.csv")
train_df.head()


# In[ ]:


# Представим распределение примеров по классам
sns.set_style("darkgrid")
sns.set_palette('tab10', n_colors=3)
sns.barplot(x = train_df.Category.value_counts().index, y = train_df.Category.value_counts());


# In[ ]:


train_df.info()


# In[ ]:


# Выведем примеры исходных картинок
print('Пример картинок (random sample)')
plt.figure(figsize=(16,10))
           
random_image = train_df.sample(n=9)
random_image_paths = random_image['Id'].values
random_image_cat = random_image['Category'].values

for index, path in enumerate(random_image_paths):
    im = PIL.Image.open(DATA_PATH+f'/train/train/{random_image_cat[index]}/{path}')
    plt.subplot(3,3, index+1)
    plt.imshow(im)
    plt.title('Class: '+str(random_image_cat[index])+ '; Size: ' + str(im.size[0]) + 'x' +str(im.size[1]))
    plt.axis('off')
plt.show()


# In[ ]:


# Для повышения качества работы модели используем аугментацию
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range = 50,
                                   shear_range=0.2,
                                   zoom_range=[0.75,1.25],
                                   brightness_range=[0.5, 1.5],
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True,
                                   validation_split=VAL_SPLIT)

train_ds = train_datagen.flow_from_directory(
    DATA_PATH+'train/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True, seed=RANDOM_SEED,
    subset='training') # set as training data

val_ds = train_datagen.flow_from_directory(
    DATA_PATH+'train/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True, seed=RANDOM_SEED,
    subset='validation') # set as validation data


# In[ ]:


test_datagen = ImageDataGenerator(rescale=1. / 255)
test_ds = test_datagen.flow_from_dataframe(
    dataframe=sample_submission,
    directory='../input/mds-misis-dl-car-classificationn/test/test_upload',
    x_col="Id",
    y_col=None,
    shuffle=False,
    class_mode=None,
    seed=RANDOM_SEED,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,)


# In[ ]:


# Выведем преобразованные фотографии
from skimage import io

def imshow(image_RGB):
  io.imshow(image_RGB)
  io.show()

x,y = train_ds.next()
print('Пример картинок из train_ds')
plt.figure(figsize=(16,10))

for i in range(0,6):
    image = x[i]
    plt.subplot(3,3, i+1)
    plt.imshow(image)
    #plt.title('Class: '+str(y[i]))
    #plt.axis('off')
plt.show()


# ### Модель

# In[ ]:


tf.keras.backend.clear_session()
print(INPUT_SHAPE)


# In[ ]:


base_model = efn.EfficientNetB6(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)


# In[ ]:


base_model.summary()


# In[ ]:


tf.keras.utils.plot_model(base_model, show_shapes = True)


# In[ ]:


# first: train only the top layers (which were randomly initialized)
base_model.trainable = False


# In[ ]:


model=M.Sequential()
model.add(base_model)
model.add(L.GlobalAveragePooling2D(),)
model.add(L.Dense(CLASS_NUM, activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


# сколько слоев
print(len(model.layers))


# In[ ]:


len(model.trainable_variables)


# In[ ]:


# Check the trainable status of the individual layers
for layer in model.layers:
    print(layer, layer.trainable)


# ### Обучение

# In[ ]:


model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=LR), metrics=["accuracy"])


# In[ ]:


checkpoint = ModelCheckpoint('best_model.hdf5' , monitor = ['val_accuracy'] , verbose = 1  , mode = 'max')
earlystop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
callbacks_list = [checkpoint, earlystop]


# In[ ]:


scores = model.evaluate_generator(val_ds, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


# Обучаем
history = model.fit_generator(
                    train_ds,
                    steps_per_epoch = train_ds.samples//train_ds.batch_size,
                    validation_data = val_ds, 
                    validation_steps = val_ds.samples//val_ds.batch_size,
                    epochs = EPOCHS,
                    callbacks = callbacks_list
                    )


# In[ ]:


model.save('../working/model_step1.hdf5')
model.load_weights('best_model.hdf5')


# In[ ]:


scores = model.evaluate_generator(val_ds, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# ### Улучшение модели

# In[ ]:


# Количество слоев в base_model
print("Number of layers in the base model: ", len(base_model.layers))


# In[ ]:


# Изменим количество необучаемых слоев в base_model
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = len(base_model.layers)//2

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False


# In[ ]:


len(base_model.trainable_variables)


# In[ ]:


# Check the trainable status of the individual layers
for layer in model.layers:
    print(layer, layer.trainable)


# In[ ]:


LR=0.0001
model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=LR), metrics=["accuracy"])


# In[ ]:


model.summary()


# In[ ]:


scores = model.evaluate_generator(val_ds, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


# Обучаем
history = model.fit_generator(
                    train_ds,
                    steps_per_epoch = train_ds.samples//train_ds.batch_size,
                    validation_data = val_ds, 
                    validation_steps = val_ds.samples//val_ds.batch_size,
                    epochs = EPOCHS,
                    callbacks = callbacks_list
                    )


# In[ ]:


model.save('../working/model_step2.hdf5')
model.load_weights('best_model.hdf5')


# In[ ]:


scores = model.evaluate_generator(val_ds, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[ ]:


predictions = model.predict(test_ds, steps=len(test_ds), verbose=1)
print(predictions.shape)


# In[ ]:


predictions = np.argmax(predictions, axis=-1) #multiple categories
label_map = (train_ds.class_indices)
label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
predictions = [label_map[k] for k in predictions]


# In[ ]:


filenames_with_dir=test_ds.filenames
submission = pd.DataFrame({'Id':filenames_with_dir, 'Category':predictions}, columns=['Id', 'Category'])
submission['Id'] = submission['Id'].replace('test_upload/','')
submission.to_csv('submission1.csv', index=False)
print('Save submit')

