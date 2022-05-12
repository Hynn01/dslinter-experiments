#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd

SEED = 1234 # global seed for random operations

df = pd.read_csv("/kaggle/input/spaceship-titanic-data-cleaning/train.csv")
df.head()


# # Introduction

# In this notebook data from [Spaceship Titanic competition](https://www.kaggle.com/competitions/spaceship-titanic), previously preprocessed in another [notebook](https://www.kaggle.com/code/cavfiumella/spaceship-titanic-data-cleaning), are predicted using a SVM, a Random Forest and a Multilayer Perceptron.

# ## Data normalization

# SVM, Decision Trees and Multilayer Perceptrons work better on normalized data. There is not a unique normalization; in this notebook _standard normalization_ is used:
# 
# <center>$x' = (x - \langle x \rangle)\ /\ \sigma$</center>
# 
# Data has to be normalized so that features have a similar range for their values.

# ## Train and test data

# To avoid supervised model adapting too much to training data, loosing precision in generic data prediction (i.e. _overfitting_) let's split our data into one subset for training and one for testing. Only training data are going to be used to build the model, while testing subset is going to be used to determine the most accurate object.

# In[ ]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(df, train_size = 0.8, random_state = SEED, stratify = df.Transported)

# let's split x from y (i.e. features from targets)
x_train, y_train = train.drop(columns="Transported").values, train.Transported.values
x_test, y_test = test.drop(columns="Transported").values, test.Transported.values


# In[ ]:


models = {} # all built models


# # Support Vector Machine

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

models["svm"] = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC())
])

models["svm"] = models["svm"].fit(x_train, y_train)


# # Random Forest

# Random Forest use multiple Decision Trees to improve the performance of a single tree.

# In[ ]:


#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

models["tree"] = Pipeline([
    ("scaler", StandardScaler()),
    ("tree", RandomForestClassifier(n_jobs = -1, random_state = SEED))
])

models["tree"] = models["tree"].fit(x_train, y_train)


# # Multilayer perceptron

# In[ ]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, BatchNormalization, Dense

models["mlp"] = Sequential([
    Input((x_train.shape[-1],), name = "input"),
    BatchNormalization(name = "norm"),
    Dense(2048, activation = "relu", name = "dense1"),
    Dense(1, name = "dense2")
])

models["mlp"].summary()


# Let's compile the model choosing optimizer, loss and metric to be used during training.

# In[ ]:


models["mlp"].compile(
    optimizer = "adam",                # tensorflow.keras.optimizers.Adam
    loss      = "binary_crossentropy", # tensorflow.keras.losses.BinaryCrossentropy
    metrics   = "accuracy"             # tensorflow.keras.metrics.accuracy
)


# In[ ]:


BATCH_SIZE = 1000
EPOCHS = 30

history = models["mlp"].fit(
    x_train.astype(float), y_train.astype(float),
    validation_data = (x_test.astype(float), y_test.astype(float)),
    batch_size = BATCH_SIZE, epochs = EPOCHS,
    workers = 4, use_multiprocessing = True
).history


# In[ ]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(2,1, sharex = True, figsize = (8,8))

for i, key in enumerate(["loss", "accuracy"]):
    ax[i].plot(np.arange(1, EPOCHS+1), history[key], label = "train")
    ax[i].plot(np.arange(1, EPOCHS+1), history[f"val_{key}"], label = "test")
    ax[i].legend()
    ax[i].grid(True)
    if i != 0: ax[i].set_xlabel("epoch")
    ax[i].set_ylabel(key)


# # Model evaluation

# Let's compare different models accuracy on testing data.

# In[ ]:


from sklearn.metrics import accuracy_score

scores = []
for key, model in models.items():
    if key == "mlp":
        scores += [[
            key, accuracy_score(y_true = y_test, y_pred = [round(a) for a in model.predict(x_test.astype(float)).ravel()])
        ]]
    else:
        scores += [[
            key, accuracy_score(y_true = y_test, y_pred = model.predict(x_test))
        ]]

pd.DataFrame(scores, columns = ["model", "accuracy"])


# In[ ]:




