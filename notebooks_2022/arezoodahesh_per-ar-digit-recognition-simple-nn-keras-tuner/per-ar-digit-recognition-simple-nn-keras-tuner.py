#!/usr/bin/env python
# coding: utf-8

# <div style="border-radius:10px;
#             border: #900c3f solid;
#             background-color:#fff6f9;
#             font-size:110%;
#             letter-spacing:0.5px;
#             text-align: center">
# 
# <center><h1 style="padding: 25px 0px 10px 0; color:#900c3f; font-weight: bold; font-family: Cursive">
# Simple neural network with tensorflow</h1></center>
# <center><h1 style="padding-bottom:25px; color:#900c3f; font-weight: bold; font-family: Cursive">
# and hyperparameter tuning with keras_tuner</h1></center>
# <center><h3 style="padding-bottom: 35px; color:#900c3f; font-weight: bold; font-family: Cursive">
# (Persian/Arabic handwritten digit recognition 🧐)</h3></center>     
# 
# </div>

# In[ ]:


import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().system('mkdir dataset')


# In[ ]:


get_ipython().system('wget https://github.com/Alireza-Akhavan/SRU-deeplearning-workshop/raw/master/dataset/Data_hoda_full.mat -P dataset')


# In[ ]:


get_ipython().system('pip install -q -U keras-tuner')


# # Import Libraries

# In[ ]:


from scipy import io
from skimage.transform import resize
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# -----------------------------------

import keras
from tensorflow import keras
import keras_tuner as kt
from keras.models import Sequential
from keras.backend import shape
from keras.layers import Dense, Dropout, Activation


# # Data Understanding

# <div style="padding: 10px; font-family: Cursive; border: solid 2px #900c3f;
#             font-size:15.5px;padding: 25px 10px; border-radius:8px;">
# <p>In <strong>Introducing a very large dataset of handwritten Farsi digits and a study on their varieties</strong> paper: A very large dataset of handwritten Farsi digits is introduced. Binary images of 102,352 digits were extracted from about 12,000 registration forms of two types, filled by B.Sc. and senior high school students. These forms were scanned at 200 dpi with a high-speed scanner.</p>
#     
# <p>It should be noted that there are 60,000 random data in the dataset used in this notebook.</p>
# ref : <a href="https://www.sciencedirect.com/science/article/abs/pii/S0167865507000037">https://www.sciencedirect.com/science/article/abs/pii/S0167865507000037</a>
#    
# </div>

# In[ ]:


dataset = io.loadmat('./dataset/Data_hoda_full.mat')
len(dataset['Data'])


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

index = 55
label = dataset['labels']
data = dataset['Data']
np.set_printoptions(linewidth=320)

print(f'LABEL: {label[index]}')
print(f'\nIMAGE PIXEL ARRAY:\n {data[index]}')


# # Preprocessing images and Train/Test split

# In[ ]:


x_train = np.squeeze(dataset['Data'][:50000])
y_train = np.squeeze(dataset['labels'][:50000])
x_test = np.squeeze(dataset['Data'][50000:])
y_test = np.squeeze(dataset['labels'][50000:])


# In[ ]:


indexes = [0,55,60,90,600]
plt.figure(figsize=(10,10))
for i in range(len(indexes)):
    plt.subplot(1,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('{}th image is: {}'.format(indexes[i], y_train[indexes[i]]))
    plt.imshow(x_train[indexes[i]], cmap=plt.cm.binary)
plt.show()


# In[ ]:


x_train_8x8 = [resize(img, (8, 8)) for img in x_train]
x_test_8x8 = [resize(img, (8, 8)) for img in x_test]


# In[ ]:


indexes = [0,55,60,90,600]
plt.figure(figsize=(10,10))
for i in range(len(indexes)):
    plt.subplot(1,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('{}th image is: {}'.format(indexes[i], y_train[indexes[i]]))
    plt.imshow(x_train_8x8[indexes[i]], cmap=plt.cm.binary)
plt.show()


# In[ ]:


x_train_new = [x.reshape(64) for x in x_train_8x8]
x_test_new = [x.reshape(64) for x in x_test_8x8]

#---------------------------------------------
x_train_new[index].shape


# In[ ]:


x_train = np.array(x_train_new)
x_test = np.array(x_test_new)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# In[ ]:


y_train_new = keras.utils.to_categorical(y_train, num_classes=10)
y_test_new = keras.utils.to_categorical(y_test, num_classes=10)


# In[ ]:


x_val = x_test[:5000]
y_val = y_test_new[:5000]
# ---------------------------------

x_test = x_test[5000:]
y_test_new = y_test_new[5000:]


# # Hyperparameter tuning (with keras_tuner) and Build Model

# In[ ]:


hp = kt.HyperParameters()

def model_builder(hp):
    model = keras.Sequential()
    # Input
    model.add(keras.Input(shape = 64))
    
    # Hidden layer
    hp_units = hp.Choice('units', values=[16, 32, 64, 128, 256, 512])
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))

    # Output layer
    model.add(keras.layers.Dense(10, activation="softmax")) 

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model


# In[ ]:


tuner = kt.Hyperband(model_builder,
                     objective="val_accuracy",
                     max_epochs=10,
                     factor=3)


# In[ ]:


stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)


# In[ ]:


tuner.search_space_summary()


# In[ ]:


tuner.search(x_train, y_train_new, epochs=20,
             validation_data = (x_val, y_val), 
             callbacks=[stop_early])

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""The optimal number of units in the first densely-connected
layer is -> {best_hps.get('units')} and the optimal learning rate for the optimizer
is -> {best_hps.get('learning_rate')}.""")


# In[ ]:


model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train_new, epochs=50, 
                    validation_data = (x_val, y_val))

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch is -> %d' % (best_epoch,))


# In[ ]:


class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.98):
            print("\nReached 98% accuracy so cancelling training ;)")
            self.model.stop_training = True
            
callbacks = myCallback()


# In[ ]:


hypermodel = tuner.hypermodel.build(best_hps)
#-----------------------------------------------

hypermodel.summary()


# In[ ]:


history = hypermodel.fit(x_train, y_train_new, epochs=best_epoch, 
                         validation_data = (x_val, y_val), 
                         callbacks=[callbacks])


# # Evaluation and Prediction

# In[ ]:


eval_result = hypermodel.evaluate(x_test, y_test_new)
print("[test loss, test accuracy]:", eval_result)


# In[ ]:


pred = hypermodel.predict(x_test)

print(pred[12])
print(y_test_new[12])


# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss'] 

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# # References
# 
# * https://www.tensorflow.org/tutorials/keras/keras_tuner
# * https://keras.io/keras_tuner/
# * https://towardsdatascience.com/hyperparameter-tuning-with-kerastuner-and-tensorflow-c4a4d690b31a
# * https://www.kaggle.com/code/jebathuraiibarnabas/wine-quality-ann-using-keras-tuner#Building-ANN-models-using-Keras-Tuner
# * https://www.sciencedirect.com/science/article/abs/pii/S0167865507000037

# <div style="border-radius:10px;
#             background-color:#ffffff;
#             border-style: solid;
#             border-color: #900c3f;
#             letter-spacing:0.5px;">
# 
# <center><h4 style="padding: 5px 0px; color:#900c3f; font-weight: bold; font-family: Cursive">
#     Thanks for your attention and for reviewing my notebook.🙌 <br><br>Please write your comments for me.📝</h4></center>
# <center><h4 style="padding: 5px 0px; color:#900c3f; font-weight: bold; font-family: Cursive">
# If you liked my work and found it useful, please upvote. Thank you🙏</h4></center>
# </div>
