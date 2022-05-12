#!/usr/bin/env python
# coding: utf-8

# # About this NoteBook
# 
# In this notebook, I review a multi-class classification problem.
# 
# In the following, you can find the explanation of the trained model, dataset, metric I used, and the focus of the study.
# 
# Also, as always, you can have free access to complete documentation of this NoteBook on my [Medium](https://samanemami.medium.com/) profile.
# 
# This Notebook only has the last version, and I do not update it.
# 
# ## The focus of this study 
# 
# The focus of this notebook is on applying a different statistical method to analyze the model's performance. 
# 
# ## Model training
# The model I used in this example is from the Keras library. Using `tf.keras.Sequential`. 
# 
# The deep Neural Network is implemented from sequential steps of the Keras. 
# Note that to use the Keras model, I recommend using the pre-processing steps from Keras.
# 
# <hr>
# 
# #### GitHub Package
# 
# To have access to the my package on GitHub, please refer to [here](https://github.com/samanemami/)
# 
# <hr>
# 
# 

# ### Author: [Seyedsaman Emami](https://github.com/samanemami)
# 
# If you want to have this method or use the outputs of the notebook, you can fork the Notebook as following (copy and Edit Kernel).
# 
# <img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1101107%2F8187a9b84c9dde4921900f794c6c6ff9%2FScreenshot%202020-06-28%20at%201.51.53%20AM.png?generation=1593289404499991&alt=media" alt="Copyandedit" width="300" height="300" class="center">
# 
# <hr>
# 
# ##### You can find some of my developments [here](https://github.com/samanemami?tab=repositories).
# 
# <hr>

# <a id='top'></a>
# # Contents
# 
# * [Importing libraries](#lib)
# * [Dataset](#dt)
#     * [Describe](#des)
#     * [Normalization](#Normalization)
# * [model](#model)

# <a id='lib'></a>
# # Importing libraries

# In[ ]:


import os
import numpy as np
import pandas as pd
import sklearn.datasets as dts
import matplotlib.pyplot as plt
from itertools import permutations
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[ ]:


import warnings
random_state = 123
np.random.seed(random_state)

warnings.simplefilter('ignore')

np.set_printoptions(precision=4, suppress=True)


# <a id='dt'></a>
# # Importing dataset

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        path = os.path.join(dirname, filename)
df = pd.read_csv(path)
df.head()


# <hr>
# 
# #### [Scroll Back To Top](#top)
# 
# <hr>

# <a id='des'></a>
# ## Describe the dataset

# In[ ]:


df.describe().T


# In[ ]:


X = (df.drop(columns=df[['Y1', 'Y2']], axis=0))
y = (df.iloc[:, -2:])

print('\n', 'X shape:',
      X.shape, '\n',
      'y shape:', y.shape)


# <a id='Normalization'></a>
# ## Normalization

# In[ ]:


norm = tf.keras.layers.Normalization(axis=-1)
norm.adapt(np.array(X))


# In[ ]:


line = X.iloc[1, :]
with np.printoptions(precision=2, suppress=True):
    print('real values', line, '\n', 'normalized values', norm(X))


# In[ ]:


X = np.array(norm(X))


# <hr>
# 
# #### [Scroll Back To Top](#top)
# 
# <hr>

# <a id='model'></a>
# # Model definition
# 
# The model I used in this example is from the Keras library. Using  tf.keras.Sequential.
# 
# multiple-input DNN models

# In[ ]:


model = keras.Sequential([norm,
                          layers.Dense(64, activation='relu'),
                          layers.Dense(64, activation='relu'),
                          layers.Dense(1)
                          ])

model.summary()


# In[ ]:


model.compile(loss='MeanSquaredError',
              optimizer=tf.keras.optimizers.Adam(0.001))


# In[ ]:


history = model.fit(X,
                    y,
                    validation_split=0.2,
                    verbose=1, epochs=100)


# In[ ]:


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.ylim([0, 10])
plt.xlabel('Epoch')
plt.ylabel('loss curve')
plt.legend()
plt.grid(True)

