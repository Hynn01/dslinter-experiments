#!/usr/bin/env python
# coding: utf-8

# ## Please note that despite the accuracy of this notebook, it used a BERT transformer for text encoding. 
# ## So, it is ineligible on a technicality for using an online resource.
# 
# ## However, I was able to achieve validation accuracy scores between 85% to 95% with this method and I thought I would publish it.

# # It was rough to not be able to use this in the competition so please leave an upvote if you found this interesting! 🥲

# In[ ]:


import pandas as pd
df_train=pd.read_csv('../input/us-patent-phrase-to-phrase-matching/train.csv')
df_test=pd.read_csv('../input/us-patent-phrase-to-phrase-matching/test.csv')
print(df_train.shape)
print(df_test.shape)


# In[ ]:


df_train_col=df_train.columns
df_test_col=df_test.columns

df_train['set']='train'
df_test['set']='test'
df_test['score']=0

df_train_test=df_train.append(df_test).reset_index(drop=True)
traindummy = pd.get_dummies(df_train_test['context'], prefix='context_')
df_train_test = pd.merge(
    df_train_test,
    traindummy,
    left_index=True,
    right_index=True,)

cat_var_list=[]
for col in df_train_test.columns:
  if 'context__' in col:
    cat_var_list.append(col)
print(len(cat_var_list))

df_train=df_train_test[df_train_test['set']=='train'].reset_index(drop=True)
df_test=df_train_test[df_train_test['set']=='test'].reset_index(drop=True)
df_train=df_train[[*df_train_col, *cat_var_list]]
df_test=df_test[[*df_test_col, *cat_var_list]]
print(df_train.shape)
print(df_test.shape)


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


get_ipython().system('pip install tensorflow_text')
get_ipython().system('pip install tensorflow_hub')


# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
from tensorflow import keras
from sklearn.model_selection import train_test_split

train_dfsam, val_df = train_test_split(df_train, test_size=0.10, stratify=df_train["context"].values, random_state=42)


# In[ ]:


# https://www.tensorflow.org/text/tutorials/bert_glue#loading_models_from_tensorflow_hub

#lite:
bert_model_path="https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1"

# heavy
# bert_model_path="https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_large/2"

bert_preprocess_path="https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"


# In[ ]:


def make_bert_preprocessing_model(sentence_features, seq_length=128):
    input_segments = [
        tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft)
        for ft in sentence_features]
    bert_preprocess = hub.load(bert_preprocess_path)
    tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name="tokenizer")
    segments = [tokenizer(s) for s in input_segments]
    truncated_segments = segments
    packer = hub.KerasLayer(bert_preprocess.bert_pack_inputs, arguments=dict(seq_length=seq_length), name="packer")
    model_inputs = packer(truncated_segments)
    return keras.Model(input_segments, model_inputs)


bert_preprocess_model = make_bert_preprocessing_model(["text_1", "text_2"])
keras.utils.plot_model(bert_preprocess_model, show_shapes=True, show_dtype=True)


# In[ ]:


def preprocess_text(text_1_series, text_2_series):
    output = bert_preprocess_model([np.array(text_1_series.to_list()), np.array(text_2_series.to_list())])
    output = {feature: tf.squeeze(output[feature]) for feature in bert_input_features}
    return output


# In[ ]:


def project_embeddings(embeddings, num_projection_layers, projection_dims, dropout_rate):
    projected_embeddings = keras.layers.Dense(units=projection_dims)(embeddings)
    for _ in range(num_projection_layers):
        x = tf.nn.gelu(projected_embeddings)
        x = keras.layers.Dense(projection_dims)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Add()([projected_embeddings, x])
        projected_embeddings = keras.layers.LayerNormalization()(x)
    return projected_embeddings
def create_text_encoder(
    num_projection_layers, projection_dims, dropout_rate, trainable=False
):
    bert = hub.KerasLayer(bert_model_path, name="bert",)
    bert.trainable = trainable

    bert_input_features = ["input_type_ids", "input_mask", "input_word_ids"]
    inputs = {
        feature: keras.Input(shape=(128,), dtype=tf.int32, name=feature)
        for feature in bert_input_features
    }
    embeddings = bert(inputs)["pooled_output"]
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
    )
    return keras.Model(inputs, outputs, name="text_encoder")


# In[ ]:


from keras.models import Sequential
from keras import layers

bert_input_features = ["input_type_ids", "input_mask", "input_word_ids"]
text_inputs = {
    feature: keras.Input(shape=(128,), dtype=tf.int32, name=feature)
    for feature in bert_input_features}

text_encoder = create_text_encoder(1,256,.1,True)
text_projections = text_encoder(text_inputs)
text_projections


# In[ ]:


from tensorflow.keras.layers import *
##possibly try leaky relu
cat_var=keras.Input(shape=(len(cat_var_list),))
transformer_embedding = keras.layers.Concatenate()([text_encoder(text_inputs), cat_var])
dense1 = keras.layers.Dense(256, activation="relu")(transformer_embedding)
dropout1 = keras.layers.Dropout(0.2)(dense1)
dense2 = keras.layers.Dense(128, activation="relu")(dropout1)
dropout2 = keras.layers.Dropout(0.1)(dense2)
dense3 = keras.layers.Dense(64, activation="relu")(dropout2)
dropout3 = keras.layers.Dropout(0.05)(dense3)
dense4 = keras.layers.Dense(16, activation="relu")(dropout3)
outputs = keras.layers.Dense(1, activation="sigmoid")(dense4)

model=keras.Model([text_inputs, cat_var], outputs)
keras.utils.plot_model(model, show_shapes=True)


# In[ ]:


#will use pearson correlation coefficent for loss function as the competition is using it for an accuracy score:
# tf.enable_eager_execution()

def t(a):
  return tf.constant(a, dtype=tf.float64)
def tmean(x, axis=-1):
  x = tf.convert_to_tensor(x)
  sum = tf.reduce_sum(x, axis=axis)
  n = tf.cast(tf.shape(x)[axis], x.dtype)
  return sum / n

tmean(t([[1.0],[2.0],[3.0]]), axis=-2)


# Pearson correlation coefficlent : $r_{xy} = \frac{\sum\left((x-\overline{x})(y-\overline{y})\right)}{\sqrt{\sum(x-\overline{x})^2\sum(y-\overline{y})^2}} $

# In[ ]:


from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import squared_difference

def correlationLoss(x,y, axis=-2):
  x = tf.convert_to_tensor(x)
  y = math_ops.cast(y, x.dtype)
  n = tf.cast(tf.shape(x)[axis], x.dtype)
  xsum = tf.reduce_sum(x, axis=axis)
  ysum = tf.reduce_sum(y, axis=axis)
  xmean = xsum / n
  ymean = ysum / n
  xsqsum = tf.reduce_sum( math_ops.squared_difference(x, xmean), axis=axis)
  ysqsum = tf.reduce_sum( math_ops.squared_difference(y, ymean), axis=axis)
  cov = tf.reduce_sum( (x - xmean) * (y - ymean), axis=axis)
  corr = cov / tf.sqrt(xsqsum * ysqsum)
  sqdif = tf.reduce_sum(tf.squared_difference(x, y), axis=axis) / n / tf.sqrt(ysqsum / n)
  return tf.convert_to_tensor( K.mean(tf.constant(1.0, dtype=x.dtype) - corr + (0.01 * sqdif)) , dtype=tf.float32 )


def correlationMetric(x, y, axis=-2):
  x = tf.convert_to_tensor(x)
  y = math_ops.cast(y, x.dtype)
  n = tf.cast(tf.shape(x)[axis], x.dtype)
  xsum = tf.reduce_sum(x, axis=axis)
  ysum = tf.reduce_sum(y, axis=axis)
  xmean = xsum / n
  ymean = ysum / n
  xvar = tf.reduce_sum(math_ops.squared_difference(x, xmean), axis=axis)
  yvar = tf.reduce_sum(math_ops.squared_difference(y, ymean), axis=axis)
  cov = tf.reduce_sum( (x - xmean) * (y - ymean), axis=axis)
  corr = cov / tf.sqrt(xvar * yvar)
  return tf.constant(1.0, dtype=x.dtype) - corr

correlationMetric(tf.constant([[0.0, 1.0, 2.0]]), tf.constant([[1.0, 3.0, 2.0]]), axis=-1)
correlationMetric(tf.constant([[0.0, 2.0, 1.0]]), tf.constant([[1.0, 3.0, 2.0]]), axis=-1)
correlationMetric(tf.constant([[0.0], [2.0], [1.0]]), tf.constant([[1.0], [3.0], [2.0]]), axis=-2)

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=[correlationMetric, 'mean_squared_error'])


# In[ ]:


train_text=bert_preprocess_model([np.array(train_dfsam['anchor'].to_list()), np.array(train_dfsam['target'].to_list())])
train_cat=np.array(train_dfsam[cat_var_list])
val_text=bert_preprocess_model([np.array(val_df['anchor'].to_list()), np.array(val_df['target'].to_list())])
val_cat=np.array(val_df[cat_var_list])


# In[ ]:


y_train=np.array(pd.to_numeric(train_dfsam['score']))
y_val=np.array(pd.to_numeric(val_df['score']))


# In[ ]:


# optimizer = keras.optimizers.Adam(lr=5e-5)
# model.compile(optimizer=optimizer, loss=[loss, loss])
# rmsprop
# sgd
# model.compile(loss='cosine_similarity', optimizer='sgd', metrics=[correlationMetric, 'accuracy'])


# In[ ]:


history = model.fit([train_text, train_cat], y_train, validation_data=([val_text, val_cat], y_val), epochs=90)

