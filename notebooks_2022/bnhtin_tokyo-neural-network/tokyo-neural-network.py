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


# In[ ]:


import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import jpx_tokyo_market_prediction


# In[ ]:


df = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
print(df[:5])
print(df.shape)


# In[ ]:


print("Start date: ",df.Date.unique().min())
print("End date: ",df.Date.unique().max())


# Look at the dataframe shape. We have 2332531 rows and start, end date from 2017-01-04 to 2021-12-03

# Next I will devide data into 2 part. That is train and test.

# In[ ]:


train = df[df['Date'] < '2021-02-30'].copy()
print(train.shape)
test = df[df['Date'] >= '2021-02-30'].copy()
print(test.shape)
print("test data percent: ",test.shape[0] / train.shape[0] * 100, "%")


# And now, we've 19.3% data for testing

# **The next step,I'll select feature from dataset**

# In[ ]:


#select feature
features = ['Open', 'High', 'Low', 'Close', 'Volume']
target = ['Target']
train = train[features + target].reset_index(drop=True).copy()
test = test[features + target].reset_index(drop=True).copy()


#  Now, I'll print train and test data to make sure it was select features

# In[ ]:


train.head(5)


# In[ ]:


test.head(5)


# I want to check the missing data for building my model

# In[ ]:


#check missing data
df.isnull().sum()


# We've missing data so to handle it I just delete it.

# In[ ]:


train.dropna(subset=features + target, axis=0, inplace=True)
test.dropna(subset=features + target, axis=0, inplace=True)


# Now let's check the data again to make sure

# In[ ]:


#check again after delete missing data
print(train.isnull().sum() + test.isnull().sum())


# Perfectly, We have a clean data

# The next step is preprocessing, so I'll do that code following this list: https://keras.io/examples/structured_data/structured_data_classification_from_scratch/#preparing-the-data . That is the nice examples, For each of these features, I will use a Normalization() layer to make sure the mean of each feature is 0 and its standard deviation is 1.

# In[ ]:


# Define encoding function for numerical features
def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = layers.Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


# I'll transfer dataframe to dataset with 2 part: features = {Open,High,Low,Close,Volume} , label = Target

# In[ ]:


# Generate tensorflow dataset
def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("Target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

train_dataset= dataframe_to_dataset(train)
test_dataset = dataframe_to_dataset(test)


# In[ ]:


# Batch the dataset
train_dataset = train_dataset.batch(1024)
test_dataset = test_dataset.batch(1024)


# **And Now, I'll build my model**
# First I define the input layers.

# In[ ]:


# Raw numerical features
Open = keras.Input(shape=(1,), name="Open")
High = keras.Input(shape=(1,), name="High")
Low = keras.Input(shape=(1,), name="Low")
Close = keras.Input(shape=(1,), name="Close")
Volume = keras.Input(shape=(1,), name="Volume")
all_inputs = [Open, High, Low, Close, Volume]


# Second , I encode them to add to layer

# In[ ]:


# Encode numerical features
open_encoded = encode_numerical_feature(Open, "Open", train_dataset)
high_encoded = encode_numerical_feature(High, "High", train_dataset)
low_encoded = encode_numerical_feature(Low, "Low", train_dataset)
close_encoded = encode_numerical_feature(Close, "Close", train_dataset)
volume_encoded = encode_numerical_feature(Volume, "Volume", train_dataset)


# In[ ]:


all_features = layers.concatenate(
    [
        open_encoded,
        high_encoded,
        low_encoded,
        close_encoded,
        volume_encoded,
    ]
)
# Add several hidden layers with batch_norm and dropout
x = layers.Dense(256, activation="relu")(all_features)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)

# Output layer for regression task
output = layers.Dense(1, activation="linear")(x)

# Create our NN model
model = keras.Model(all_inputs, output)
model.compile("adam", "mse", metrics=[tf.keras.metrics.RootMeanSquaredError()])


# If validation loss does not improve for some number of epochs, stop training and restore best model weights. So that is the step to save the trainning time for model before doing that

# In[ ]:


# Set early_stopping callbacks, if val_loss does not improve for 10 epochs, stop training and restore best model weights
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    min_delta=1e-3,
    restore_best_weights=True,
)


# In[ ]:


# Model training 
model.fit(train_dataset, epochs=50, validation_data=test_dataset, callbacks=[early_stopping])


# Transfer dataframe to dataset for test

# In[ ]:


# Generate tensorflow dataset for test data
def dataframe_to_dataset_test(dataframe):
    dataframe = dataframe.copy()
    ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))
    return ds


# Prediction:

# In[ ]:


env = jpx_tokyo_market_prediction.make_env()   # initialize the environment


# In[ ]:


iter_test = env.iter_test()    # an iterator which loops over the test files


# In[ ]:


for (prices, options, financials, trades, secondary_prices, prediction) in iter_test:
    test_ds = dataframe_to_dataset_test(prices)
    prediction['target_pred'] = model.predict(test_ds)
    prediction = prediction.sort_values(by="target_pred", ascending=False)
    prediction['Rank'] = np.arange(2000)
    prediction = prediction.sort_values(by="SecuritiesCode", ascending=True)
    prediction.drop(['target_pred'], axis=1, inplace=True)
    display(prediction)
    env.predict(prediction)   #your predictions

