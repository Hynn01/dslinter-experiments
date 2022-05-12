#!/usr/bin/env python
# coding: utf-8

# **Import dependency**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report, multilabel_confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
import seaborn as sns

import warnings 
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train.csv')
test = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test.csv')


# **Cleanimg Data**

# In[ ]:


train.head()


# In[ ]:


train.isnull()


# In[ ]:


train.isna().sum()


# In[ ]:


train_df_original = train.copy()

# Split into training, test and validation sets
val_index = int(train.shape[0]*0.2)

train_df = train_df_original.iloc[val_index:]
val_df = train_df_original.iloc[:val_index]


# In[ ]:


y = np.array(train_df['label'])
X = np.array(train_df.drop(columns='label'))


# In[ ]:


X.shape,y.shape


# In[ ]:


import random
r = random.randint(0,(21964-1))
def show_img():
  arr = np.array(X)
  some_value = arr[r]
  some_img = some_value.reshape(28,28)
  plt.imshow(some_img, cmap="gray")
  plt.axis("off")
  plt.show()  

show_img()
print(y[r])


# **Preprocessing data for training**

# In[ ]:


y_train = pd.get_dummies(y)
y_train.head(5)


# In[ ]:


y_val = val_df['label']
X_val = val_df.drop(columns="label",axis=1)


# In[ ]:


y_val = pd.get_dummies(y_val)


# In[ ]:


y_train.shape # else it's shape was (21964,)


# In[ ]:


X_val = pd.DataFrame(X_val).values.reshape(X_val.shape[0] ,28, 28, 1)


# In[ ]:


X_train = pd.DataFrame(X).values.reshape(X.shape[0] ,28, 28, 1)

# reshaping X_train into (27455,28,28,1)


# In[ ]:


X_train.shape,y_train.shape


# In[ ]:


#  Accepts a batch of images used for training.
generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=False,
    fill_mode="nearest"
)

X_train_flow = generator.flow(X_train, y_train, batch_size=32)

X_val_flow = generator.flow(X_val, y_val, batch_size=32)


# **Defining a model**

# In[ ]:


# Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization

model = Sequential()

model.add(Conv2D(filters=32,  kernel_size=(3,3), activation="relu", input_shape=(28,28,1)))
model.add(MaxPool2D((2,2),padding='SAME'))
model.add(Dropout(rate=0.2))


model.add(Conv2D(filters=64,  kernel_size=(3,3), activation="relu", input_shape=(28,28,1)))
model.add(MaxPool2D((2,2),padding='SAME'))
model.add(Dropout(rate=0.2))


model.add(Conv2D(filters=521,  kernel_size=(3,3), activation="relu", input_shape=(28,28,1)))
model.add(MaxPool2D((2,2),padding='SAME'))
model.add(Dropout(rate=0.2))



model.add(Flatten())
model.add(Dense(units=521, activation="relu"))
model.add(Dense(units=256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=24, activation="softmax"))


model.compile(loss="categorical_crossentropy", optimizer='adam',  metrics=["accuracy"])


# In[ ]:


model.summary()


# **Evaluating the model**

# In[ ]:


learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001
)


# In[ ]:


history = model.fit(
    X_train_flow,
    validation_data=X_val_flow,
    # epochs=100,
    epochs=50,
    callbacks=[
               tf.keras.callbacks.EarlyStopping(
                   monitor='val_loss',
                   patience=5,
                   restore_best_weights=True
                   ),
      learning_rate_reduction
    ])


# **Visualizing the loss**

# In[ ]:


fig, axes = plt.subplots(2, 1, figsize=(15, 10))
ax = axes.flat

pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot(ax=ax[0])
ax[0].set_title("Accuracy", fontsize = 15)
ax[0].set_ylim(0,1.1)

pd.DataFrame(history.history)[['loss','val_loss']].plot(ax=ax[1])
ax[1].set_title("Loss", fontsize = 15)
plt.show()


# **Checking the accuracy**

# In[ ]:


y_test = np.array(test['label'])
X_test = np.array(test.drop(columns='label'))

y_test = pd.get_dummies(y_test)
X_test = pd.DataFrame(X_test).values.reshape(X_test.shape[0] ,28, 28, 1)

# X_test_flow = generator.flow(X_test, y_test, batch_size=32)
# X_test.shape,X_train.shape

y_test = pd.get_dummies(y_test)


# In[ ]:


from sklearn.metrics import classification_report

# predictions
pred = model.predict(X_test)

y_pred = np.argmax(pred,axis=1)
y_test = np.argmax(y_test.values,axis=1)


# In[ ]:


acc = accuracy_score(y_test,y_pred)

# # Display the results
print(f'## {acc*100:.2f}% accuracy on the test set')

