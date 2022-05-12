#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras import layers, applications, optimizers
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# In[ ]:


image_size = 350
batch_size = 16
save_model_filename = 'effnet_(1).h5'


# In[ ]:


train = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')
train_path = '../input/cassava-leaf-disease-classification/train_images'


# In[ ]:


def image_path(image):
    return os.path.join(train_path,image)

train['image_id'] = train['image_id'].apply(image_path)


# In[ ]:


train['label'] = train['label'].astype('str')


# In[ ]:


image_gen = ImageDataGenerator(preprocessing_function = tf.keras.applications.efficientnet.preprocess_input,
                                horizontal_flip=True, vertical_flip=True, fill_mode='nearest', brightness_range=[0.7, 1.3],
                                rotation_range=270, zoom_range=0.2, shear_range=10, width_shift_range=0.2, height_shift_range=0.2,
                                validation_split=0.2, rescale = 1./255)

test_gen = ImageDataGenerator(rescale=1./255)


# In[ ]:


train_generator = image_gen.flow_from_dataframe(dataframe=train, directory=None, x_col='image_id', y_col='label',
                                                subset='training', batch_size=batch_size, seed=1,
                                                shuffle=True, class_mode='categorical', target_size=(image_size,image_size))

validation_generator = image_gen.flow_from_dataframe(dataframe=train, directory=None, x_col='image_id', y_col='label',
                                                   subset='validation', batch_size=batch_size, seed=1,
                                                   shuffle=False, class_mode='categorical', target_size=(image_size,image_size))


# In[ ]:


def build_efficientnet_b3():
    model = Sequential()
    model.add(EfficientNetB3(input_shape=(image_size,image_size,3), include_top=False, weights='imagenet'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    
    return model


# In[ ]:


model = build_efficientnet_b3()
model.summary()


# In[ ]:


model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy']
)


# In[ ]:


from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(save_model_filename, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.2, patience=2, min_lr=0, verbose=1)


# In[ ]:


import seaborn as sns

sns.countplot(train['label'])
plt.title('Count of disease types')
plt.grid()
plt.show()


# In[ ]:


epoch = 30

history = model.fit(
    train_generator,
    epochs=epoch,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)


# In[ ]:


print('train loss:', history.history['loss'][-1])
print('train accuracy:', history.history['accuracy'][-1])

print('dev loss:', history.history['val_loss'][-1])
print('dev accuracy:', history.history['val_accuracy'][-1])

results = pd.DataFrame(history.history)

fig, axs = plt.subplots(1,2,figsize=(15,5))

axs[0].plot(results[['loss', 'val_loss']])
axs[0].set_title('Model Loss')

axs[1].plot(results[['accuracy', 'val_accuracy']])
axs[1].set_title('Model Accuracy')

plt.show()


# In[ ]:


pred = model.predict(validation_generator)
predictions = np.argmax(pred, axis=1)
actual = validation_generator.classes

from sklearn.metrics import classification_report

report = classification_report(actual, predictions, target_names=['CBB', 'CBSD', 'CGM', 'CMD', 'Healthy'])
print(report)


# In[ ]:


test = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')


# In[ ]:


test_path = '../input/cassava-leaf-disease-classification/test_images'

def test_image_path(image):
    return os.path.join(test_path,image)

test['image_id'] = test['image_id'].apply(test_image_path)

test['label'] = test['label'].astype('str')


# In[ ]:


test_generator = test_gen.flow_from_dataframe(dataframe=test, directory=None, x_col='image_id', y_col='label',
                                              preprocessing_function=applications.efficientnet.preprocess_input,
                                              class_mode='categorical', target_size=(image_size,image_size))


# In[ ]:


output = model.predict(test_generator)


# In[ ]:


submission = pd.DataFrame()
submission['image_id'] = list(os.listdir(test_path))
submission['label'] = np.argmax(output, axis=1)
submission.to_csv("submission.csv", index=False)


# In[ ]:


submission.head()


# In[ ]:


model.save('model_final.h5')


# In[ ]:




