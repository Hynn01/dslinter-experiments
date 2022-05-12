#!/usr/bin/env python
# coding: utf-8

# # MNIST Digit Recognizer

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout,MaxPool2D, LSTM, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ReduceLROnPlateau
from keras.layers import ELU


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Data Preprocessing

# ## Load Training Dataset

# In[ ]:


train_data = pd.read_csv("../input/digit-recognizer/train.csv")


# In[ ]:


test_file = "../input/digit-recognizer/test.csv"
test_data = pd.read_csv(test_file)

submission_data = pd.read_csv('../input/digit-recognizer/sample_submission.csv')


# Shape

# In[ ]:


print(f"train.csv size is {train_data.shape}")
print(f"test.csv size is {test_data.shape}")


# # Split Dataset

# In[ ]:


img_rows, img_cols = 28, 28
num_classes = 10


# # Data Preprocessing

# 
# **ONE HOT ENCODING:**
# 
# One hot encoding is one method of converting data to prepare it for an algorithm and get a better prediction. With one-hot, we convert each categorical value into a new categorical column and assign a binary value of 1 or 0 to those columns. Each integer value is represented as a binary vector.

# In[ ]:


def data_prep(raw):
    out_y = tf.keras.utils.to_categorical(raw.label, num_classes)

    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    # normalization
    out_x = x_shaped_array / 255
    return out_x, out_y

x, y = data_prep(train_data)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)#ratio 90:10
x_train, x_val, y_train, y_val= train_test_split(x_train, y_train, test_size = 1/9, random_state=42)


# In[ ]:


test_data = test_data / 255
test_data = test_data.values.reshape(-1,28,28,1)


# **Data Dimension**

# In[ ]:


print(f"Training data size is {x_train.shape}")
print(f"Testing data size is {x_test.shape}")


# **Visualize Data**

# In[ ]:


plt.figure(figsize=(7,9))
for i in range(1, 10):
    plt.subplot(330 + i)
    plt.imshow(x_train[i], cmap=plt.get_cmap('Greys'))
    plt.title(y_train[i])
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
plt.tight_layout()


# # **Activation Function**
# - ELU

# In[ ]:


def elu(z,alpha):
    if z >= 0:
        return z
    else :
        return alpha*(e^z -1)


# In[ ]:


elu = keras.activations.elu(x, alpha=1.0)


# - LeakyReLu

# In[ ]:


def LeakyReLu(z,alpha):
    if z >= 0:
        return z
    else :
        return alpha*z


# # Model Summary

# In[ ]:



model = Sequential()

model.add(Conv2D(64, kernel_size=(5, 5), padding='same',input_shape=(img_rows, img_cols, 1),activation='relu'))
model.add(Conv2D(64, kernel_size=(5, 5), padding = 'same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), padding='same',activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), padding = 'same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

model.summary()


# In[ ]:


lr_reduction = ReduceLROnPlateau(monitor='val_accuracy',patience=4, verbose=1,  factor=0.4, min_lr=0.0001)


# In[ ]:


epochs=25


# In[ ]:


model.compile(optimizer = 'adam',loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])
model_fit = model.fit(x_train, y_train, epochs=epochs,batch_size =300 ,validation_data=(x_val, y_val), verbose =2,callbacks=[lr_reduction])


# In[ ]:


# Defining Figure
f = plt.figure(figsize=(20,7))

#Adding Subplot 1 (For Accuracy)
f.add_subplot(121)

plt.plot(model_fit.epoch,model_fit.history['accuracy'],label = "accuracy") # Accuracy curve for training set
#plt.plot(history_1.epoch,history_1.history['val_accuracy'],label = "val_accuracy") # Accuracy curve for validation set

plt.title("Accuracy Curve",fontsize=18)
plt.xlabel("Epochs",fontsize=15)
plt.ylabel("Accuracy",fontsize=15)
plt.grid(alpha=0.3)
plt.legend()

#Adding Subplot 1 (For Loss)
f.add_subplot(122)

plt.plot(model_fit.epoch,model_fit.history['loss'],label="loss") # Loss curve for training set

plt.title("Loss Curve",fontsize=18)
plt.xlabel("Epochs",fontsize=15)
plt.ylabel("Loss",fontsize=15)
plt.grid(alpha=0.3)
plt.legend()

plt.show()


# # **Save Model**

# In[ ]:


get_ipython().system('mkdir -p saved_model')
model.save('saved_model/my_model')
model.save('saved_model/my_model.h5')


# In[ ]:


load_model = tf.keras.models.load_model('saved_model/my_model.h5')

from tensorflow.keras.utils import plot_model
plot_model(load_model, to_file='model.png', show_shapes=True)
from IPython.display import Image
Image("model.png")


# # Evaluate

# In[ ]:


evaluate_test = load_model.evaluate(x_test, y_test, verbose=1)

print("\nAccuracy =", "{:.7f}%".format(evaluate_test[1]*100))
print("Loss     =" ,"{:.9f}".format(evaluate_test[0]))


# In[ ]:


y_predict = load_model.predict(x_test)


# In[ ]:


y_predict_max = np.argmax(y_predict,axis=1) 
y_predict_max


# ## Submission

# In[ ]:


submission_label = np.argmax(load_model.predict(test_data), axis=1)
submission_label = pd.Series(submission_label, name="Label")

image_id = pd.Series(range(1,len(test_data)+1))
image_id = pd.Series(image_id, name="ImageId")


# In[ ]:


submission = pd.concat([image_id,submission_label],axis = 1)
submission.to_csv("submission.csv", index=False)
pd.read_csv("submission.csv").head()

