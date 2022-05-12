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
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Adamax

from transformers import DistilBertTokenizer, TFDistilBertModel

from sklearn.model_selection import train_test_split

import gc

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import warnings,json
warnings.filterwarnings('ignore')


# In[ ]:


os.environ["WANDB_API_KEY"] = "0"


# In[ ]:


def Init_TPU():  

    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
        REPLICAS = strategy.num_replicas_in_sync
        print("Connected to TPU Successfully:\n TPUs Initialised with Replicas:",REPLICAS)
        
        return strategy
    
    except ValueError:
        
        print("Connection to TPU Falied")
        print("Using default strategy for CPU and single GPU")
        strategy = tf.distribute.get_strategy()
        
        return strategy
    
strategy=Init_TPU()


# In[ ]:


path = '../input/contradictory-my-dear-watson/'


# In[ ]:



train_url = os.path.join(path,'train.csv')
train_data = pd.read_csv(train_url, header='infer')

sample_sub_url = os.path.join(path,'sample_submission.csv')
sample_sub = pd.read_csv(sample_sub_url, header='infer')

test_url = os.path.join(path,'test.csv')
test_data = pd.read_csv(test_url, header='infer')


# In[ ]:


train_data.head()


# In[ ]:


gc.collect()


# In[ ]:


# Transformer Model Name
transformer_model = 'distilbert-base-multilingual-cased'

# Define Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(transformer_model)


# In[ ]:


# Checking the output of tokenizer
tokenizer.convert_tokens_to_ids(list(tokenizer.tokenize("Elementary, My Dear Watson!")))


# In[ ]:


# Create seperate list from Train & Test Dataframes with only Premise & Hypothesis
train = train_data[['premise','hypothesis']].values.tolist()
test = test_data[['premise','hypothesis']].values.tolist()


# In[ ]:


# Define Max Length
max_len = 80   # << change if you wish

# Encode the training & test data 
train_encode = tokenizer.batch_encode_plus(train, pad_to_max_length=True, max_length=max_len)
test_encode = tokenizer.batch_encode_plus(test, pad_to_max_length=True, max_length=max_len)


# In[ ]:


# Split the Training Data into Training (90%) & Validation (10%)

test_size = 0.1  # << change if you wish
x_train, x_val, y_train, y_val = train_test_split(train_encode['input_ids'], train_data.label.values, test_size=test_size)


# Split Test Data
x_test = test_encode['input_ids']


# In[ ]:


#garbage collect
gc.collect()


# In[ ]:


# Loading Data Into TensorFlow Dataset
AUTO = tf.data.experimental.AUTOTUNE
batch_size = 16 * strategy.num_replicas_in_sync

train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train)).repeat().shuffle(3072).batch(batch_size).prefetch(AUTO))
val_ds = (tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).prefetch(AUTO))

test_ds = (tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size))


# In[ ]:


# Garbage Collection
gc.collect()


# In[ ]:


def build_model(strategy,transformer):
    with strategy.scope():
        transformer_encoder = TFDistilBertModel.from_pretrained(transformer)  #Pretrained BERT Transformer Model
        
        input_layer = Input(shape=(max_len,), dtype=tf.int32, name="input_layer")
        
        sequence_output = transformer_encoder(input_layer)[0]
        
        cls_token = sequence_output[:, 0, :]
        
        output_layer = Dense(3, activation='softmax')(cls_token)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        
        model.compile(
            Adamax(lr=1e-5), 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        return model
    

# Applying the build model function
model = build_model(strategy,transformer_model)


# In[ ]:


# Model Summary
model.summary()


# In[ ]:


# Train the Model

epochs = 30  # < change if you wish
n_steps = len(train_data) // batch_size 

model.fit(train_ds, 
          steps_per_epoch = n_steps, 
          validation_data = val_ds,
          epochs = epochs)


# In[ ]:


# Garbage Collection
gc.collect()


# In[ ]:


prediction = model.predict(test_ds, verbose=0)
sample_sub['prediction'] = prediction.argmax(axis=1)


# In[ ]:


sample_sub.to_csv("submission.csv", index=False)


# In[ ]:


sample_sub.head()


# In[ ]:




