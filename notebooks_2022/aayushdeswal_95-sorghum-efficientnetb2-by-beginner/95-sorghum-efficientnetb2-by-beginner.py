#!/usr/bin/env python
# coding: utf-8

# # **Import Libraries**

# In[ ]:


import numpy as np
import os
import tensorflow as tf
import pandas as pd
import glob
import math
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from PIL import Image
from collections import Counter


# # Train data preprocessing

# In[ ]:


train_cultivar= pd.read_csv('/kaggle/input/sorghum-id-fgvc-9/train_cultivar_mapping.csv')
train_cultivar.dropna(inplace=True)
train_cultivar['cultivar']=train_cultivar['cultivar'].astype(str)
train_cultivar.head()


# In[ ]:


labels=tf.unique(train_cultivar['cultivar'])
label=labels[0]
label


# In[ ]:


import seaborn as sns
a=pd.DataFrame({'cultivar':train_cultivar['cultivar']})
plt.figure(figsize=(30,10))
sns.histplot(a,x='cultivar')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=0.9, horizontal_flip=True, shear_range=0.2)
train_data=train_datagen.flow_from_dataframe(train_cultivar, "/kaggle/input/sorghum-id-fgvc-9/train_images/",color_mode='rgb', batch_size=32, x_col='image', y_col='cultivar', class_mode='categorical', target_size=(300, 300))


# # **EfficientnetB2(Base Model)**

# In[ ]:


base_model_2= tf.keras.applications.efficientnet.EfficientNetB2(include_top=False, weights='imagenet', input_shape=(300,300,3))


# In[ ]:


image_batch,label_image= next(iter(train_data))
feature_batch=base_model_2(image_batch)
feature_batch.shape


# # **Design the Model**

# In[ ]:


inputs=tf.keras.Input(shape=(300,300,3))
x=base_model_2(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x=tf.keras.layers.Dropout(0.3)(x)
outputs=tf.keras.layers.Dense(100, activation='softmax')(x)
model_2=tf.keras.Model(inputs, outputs)


# In[ ]:


model_2.summary()


# # **Callbacks**

# In[ ]:


callback1=tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)


# In[ ]:


class callback2(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs={}):
        if (logs.get('accuracy')>0.945):
            print('\nStopping training since train accuracy is greater than 94.5%')
            self.model.stop_training=True


# # **Cyclical Learning Rate**

# In[ ]:


len_dataset=len(train_cultivar)
batch_size=32
total_steps=round(len_dataset/batch_size)
clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=1e-4,
    maximal_learning_rate=1e-3,
    scale_fn=lambda x: 1/(2.**(x-1)),
    step_size=2 * total_steps
)


# # **Compile the model**

# In[ ]:


model_2.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=clr), metrics=['accuracy'])


# # **Fit the model and Visualize**

# In[ ]:


tf.keras.utils.plot_model(model_2)


# In[ ]:


history=model_2.fit(train_data, epochs=7, steps_per_epoch=total_steps, callbacks=[callback1, callback2()])


# In[ ]:


pd.DataFrame(history.history).plot()
plt.ylabel('loss & accuracy')
plt.xlabel('epochs')


# # **Prepare Test Data**

# In[ ]:


l={'image':[]}
for filename in glob.glob("/kaggle/input/sorghum-id-fgvc-9/test/*"):
    l['image'].append(filename[37:])
    
l['image'].sort()
test_data=pd.DataFrame(l)
test_data


# In[ ]:


test_datagen=tf.keras.preprocessing.image.ImageDataGenerator()
test_dataset=test_datagen.flow_from_dataframe(test_data,"/kaggle/input/sorghum-id-fgvc-9/test/", x_col='image', y_col=None, color_mode='rgb', target_size=(300,300), class_mode=None)


# # **Predict**

# In[ ]:


hehe=model_2.predict(test_dataset)


# In[ ]:


hehe.shape


# In[ ]:


predict1=[]
for j in hehe:
    predict1.append(tf.argmax(j))


# In[ ]:


predictions=[]
for k in predict1:
    predictions.append(label[k].numpy().decode("utf-8"))
  


# In[ ]:


final=pd.DataFrame({'filename': test_data.image, 'cultivar': predictions})
final.to_csv('submission.csv', index=False)

