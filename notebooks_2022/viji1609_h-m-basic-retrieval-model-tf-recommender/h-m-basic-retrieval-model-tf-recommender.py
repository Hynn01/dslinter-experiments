#!/usr/bin/env python
# coding: utf-8

# # Tensorflow Recommenders:
# 
# In this notebook we peek into the possibility of using Tensorflow recommender system (tfrs) -  Retrieval models for H&M product recommendations.
# 
# Retrieval models have typically query and candidate models in which features are embedded. Affinity score is calculated by a factorized retrieval model.The retrieval task is for selecting an initial set of candidates from all possible candidates.
# 
# Tensorflow has easy to implement modules such as *tfrs.tasks.Retrieval* along with metrics such as *tfrs.metrics.FactorizedTopK* for retrieval task.
# 
# Tensorflow *ScaNN* library can be used to retrieve the best candidates for a given query. In our case we can get the 12 recommendations required using this library.

# In[ ]:


get_ipython().system('pip install -q tensorflow-recommenders')
get_ipython().system('pip install -q scann')


# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

from pathlib import Path
from typing import Dict, Text


# # The Dataset

# In[ ]:


data_dir = Path('../input/h-and-m-personalized-fashion-recommendations')
train0 = pd.read_csv(data_dir/'transactions_train.csv')
train0 = train0[train0['t_dat'] >='2020-09-01']

# add 0 in article_id column (string)
train0['article_id'] = train0['article_id'].astype(str)
train0['article_id'] = train0['article_id'].apply(lambda x: x.zfill(10))
train0.head()


# In[ ]:


customer_df = pd.read_csv(data_dir/'customers.csv')
customer_df.head()


# In[ ]:


article_df = pd.read_csv(data_dir/'articles.csv')

# add 0 in article_id column (string) similar to train0
article_df['article_id'] = article_df['article_id'].astype(str)
article_df['article_id'] = article_df['article_id'].apply(lambda x: x.zfill(10))
article_df.head()


# We select only two features for training. Also generate data for embedding in both query and candidate models.
# 

# In[ ]:


#get data for embedding and task

unique_customer_ids = customer_df.customer_id.unique()
unique_article_ids = article_df.article_id.unique()

article_ds = tf.data.Dataset.from_tensor_slices(dict(article_df[['article_id']]))
articles = article_ds.map(lambda x: x['article_id'])


# # Query, Candidate and H&M model 

# In[ ]:


embedding_dimension = 64

# Query Model
customer_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_customer_ids, mask_token=None),  
  tf.keras.layers.Embedding(len(unique_customer_ids) + 1, embedding_dimension)
])


# In[ ]:


# Candidate Model
article_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_article_ids, mask_token=None),
  tf.keras.layers.Embedding(len(unique_article_ids) + 1, embedding_dimension)
])


# In[ ]:


# Retrieval Model

class HandMModel(tfrs.Model):
    
    def __init__(self, customer_model, article_model):
        super().__init__()
        self.article_model: tf.keras.Model = article_model
        self.customer_model: tf.keras.Model = customer_model
        self.task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=articles.batch(128).map(self.article_model),            
            ),
        )        

    def compute_loss(self, features: Dict[str, tf.Tensor], training=False) -> tf.Tensor:
    
        customer_embeddings = self.customer_model(features["customer_id"])    
        article_embeddings = self.article_model(features["article_id"])

        # The task computes the loss and the metrics.
        return self.task(customer_embeddings, article_embeddings,compute_metrics=not training)


# # Train & Validate

# In[ ]:


model = HandMModel(customer_model, article_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))


# In[ ]:


train = train0[train0['t_dat']<='2020-09-15']
test = train0[train0['t_dat'] >='2020-09-15']

train_ds = tf.data.Dataset.from_tensor_slices(dict(train[['customer_id','article_id']])).shuffle(100_000).batch(256).cache()
test_ds = tf.data.Dataset.from_tensor_slices(dict(test[['customer_id','article_id']])).batch(256).cache()

num_epochs = 5

'''

history = model.fit(
    train_ds, 
    validation_data = test_ds,
    validation_freq=5,
    epochs=num_epochs,
    verbose=1)

'''


# **A word on metrics:**
# 
# Calculation of factorized top K metric is highly time intensive. Even with the option 'compute_metrics=not training' and 
# computing validation metrics only every 5 epochs, it still takes a lot of time. You can check this by running above model. Another option 
# may be by reducing the number of retrievals from standard 100.(may cost accuracy?)
# 
# self.task = tfrs.tasks.Retrieval(
#         metrics=tfrs.metrics.FactorizedTopK(
#         candidates=articles.batch(128).map(self.article_model),
#         k = (any value less than 100)
#         )

# # Retrieve & Submit

# In[ ]:


# train without validation

train_ds = tf.data.Dataset.from_tensor_slices(dict(train0[['customer_id','article_id']])).shuffle(100_000).batch(256).cache()

num_epochs = 5

history = model.fit(
    train_ds,    
    epochs=num_epochs,
    verbose=1)


# In[ ]:


scann_index = tfrs.layers.factorized_top_k.ScaNN(model.customer_model, k = 12 )
scann_index.index_from_dataset(
  tf.data.Dataset.zip((articles.batch(100), articles.batch(100).map(model.article_model)))
)


# In[ ]:


sub = pd.read_csv(data_dir/'sample_submission.csv')
_,articles = scann_index(sub.customer_id.values)
preds = articles.numpy().astype(str)
preds = pd.Series(map(' '.join, preds,))
sub['prediction'] = preds
sub.to_csv('submission.csv',index=False)


# This notebook is based on recommender models in the Tensorflow official site. This model can be further refined by adding more features, deep layers and with different model hyperparameters. Examples can be referred at https://www.tensorflow.org/recommenders
# 
# **Thank you for your time!**
# 
