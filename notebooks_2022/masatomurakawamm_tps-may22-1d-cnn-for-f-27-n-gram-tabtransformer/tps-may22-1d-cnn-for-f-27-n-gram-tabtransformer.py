#!/usr/bin/env python
# coding: utf-8

# ---
# # [Tabular Playground Series - May 2022][1]
# 
# - The goal of this competition is to predict whether the machine is in state 0 or state 1. The data has various feature interactions that may be important in determining the machine state.
# 
# ---
# #### **The aim of this notebook is to**
# - **1. Conduct Exploratory Data Analysis (EDA).**
# - **2. Create unigram and bigram from 'f_27' column, and apply the 1D-CNN module to encode text features.**
# - **3. Build and train a TabTransformer model.**
# 
# ---
# **References:** Thanks to previous great codes and notebooks.
# - [🔥🔥[TensorFlow]TabTransformer🔥🔥][2]
# - [Sachin's Blog Tensorflow Learning Rate Finder][3]
# 
# ---
# ### **If you find this notebook useful, please do give me an upvote. It helps me keep up my motivation.**
# #### **Also, I would appreciate it if you find any mistakes and help me correct them.**
# 
# ---
# [1]: https://www.kaggle.com/competitions/tabular-playground-series-may-2022
# [2]: https://www.kaggle.com/code/usharengaraju/tensorflow-tabtransformer
# [3]: https://sachinruk.github.io/blog/tensorflow/learning%20rate/2021/02/15/Tensorflow-Learning-Rate-Finder.html

# <h1 style="background:#a1a8f0; border:0; border-radius: 10px; color:black"><center>TABLE OF CONTENTS</center></h1>
# 
# <ul class="list-group" style="list-style-type:none;">
#     <li><a href="#0" class="list-group-item list-group-item-action">0. Settings</a></li>
#     <li><a href="#1" class="list-group-item list-group-item-action">1. Data Loading</a></li>
#     <li><a href="#2" class="list-group-item list-group-item-action">2. Exploratory Data Analysis</a>
#         <ul class="list-group" style="list-style-type:none;">
#             <li><a href="#2.1" class="list-group-item list-group-item-action">2.1 Target Distribution</a></li>
#             <li><a href="#2.2" class="list-group-item list-group-item-action">2.2 Feature Distributions</a></li>
#             <li><a href="#2.3" class="list-group-item list-group-item-action">2.3 Exploring f_27 Feature</a></li>
#         </ul>
#     </li>
#     <li><a href="#3" class="list-group-item list-group-item-action">3. Model Building</a>
#         <ul class="list-group" style="list-style-type:none;">
#             <li><a href="#3.1" class="list-group-item list-group-item-action">3.1 Validation Split</a></li>
#             <li><a href="#3.2" class="list-group-item list-group-item-action">3.2 Dataset</a></li>
#             <li><a href="#3.3" class="list-group-item list-group-item-action">3.3 Preprocessing Model</a></li>
#             <li><a href="#3.4" class="list-group-item list-group-item-action">3.4 Training Model</a></li>
#         </ul>
#     </li>
#     <li><a href="#4" class="list-group-item list-group-item-action">4. Model Training</a>
#         <ul class="list-group" style="list-style-type:none;">
#             <li><a href="#4.1" class="list-group-item list-group-item-action">4.1 Learning Rate Finder</a></li>
#             <li><a href="#4.2" class="list-group-item list-group-item-action">4.2 Model Training</a></li>
#         </ul>
#     </li>
#     <li><a href="#5" class="list-group-item list-group-item-action">5. Inference</a></li>
# </ul>
# 

# <a id ="0"></a><h1 style="background:#a1a8f0; border:0; border-radius: 10px; color:black"><center>0. Settings</center></h1>

# In[ ]:


## Import dependencies 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
import pathlib
import gc
import sys
import re
import math 
import random
import time 
import datetime as dt
from tqdm import tqdm 

import sklearn
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

import warnings
warnings.filterwarnings('ignore')

print('import done!')


# In[ ]:


## For reproducible results    
def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['PYTHONHASHSEED'] = str(s) 
    print('Seeds setted!')
    
global_seed = 42
seed_all(global_seed)


# <a id ="1"></a><h1 style="background:#a1a8f0; border:0; border-radius: 10px; color:black"><center>1. Data Loading</center></h1>

# ---
# ### [Files Descriptions](https://www.kaggle.com/competitions/tabular-playground-series-may-2022/data)
# 
# - **train.csv** - the training data, which includes normalized continuous data and categorical data
# 
# - **test.csv** - the test set; your task is to predict binary target variable which represents the state of a manufacturing process
# 
# - **sample_submission.csv** -  a sample submission file in the correct format.
# 
# ---
# ### [Submission & Evaluation](https://www.kaggle.com/competitions/tabular-playground-series-may-2022/overview/evaluation)
# 
# - For each id in the test set, you must predict a probability for the target variable.
# - Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.
# 
# ---

# In[ ]:


## Data Loading
data_config = {'train_csv_path': '../input/tabular-playground-series-may-2022/train.csv',
               'test_csv_path': '../input/tabular-playground-series-may-2022/test.csv',
               'sample_submission_path': '../input/tabular-playground-series-may-2022/sample_submission.csv',
              }

train_df = pd.read_csv(data_config['train_csv_path'])
test_df = pd.read_csv(data_config['test_csv_path'])
submission_df = pd.read_csv(data_config['sample_submission_path'])

print(f'train_length: {len(train_df)}')
print(f'test_lenght: {len(test_df)}')
print(f'submission_length: {len(submission_df)}')


# In[ ]:


## train_df Check
train_df.head()


# In[ ]:


## Null Value Check
print('train_df.info()'); print(train_df.info(), '\n')


# <a id ="2"></a><h1 style="background:#a1a8f0; border:0; border-radius: 10px; color:black"><center>2. Exploratory Data Analysis</center></h1>

# <a id ="2.1"></a><h2 style="background:#d9afed; border:0; border-radius: 8px; color:black"><center>2.1 Target Distribution</center></h2>

# In[ ]:


## Target Distribution
target_count = train_df.groupby(['target'])['id'].count()
target_percent = target_count / target_count.sum()

## Make Figure object
fig = go.Figure()

## Make trace (graph object)
data = go.Bar(x=target_count.index.astype(str).values, 
              y=target_count.values)

## Add the trace to the Figure
fig.add_trace(data)

## Setting layouts
fig.update_layout(title = dict(text='target distribution'),
                  xaxis = dict(title='target values'),
                  yaxis = dict(title='counts'))

## Show the Figure
fig.show()


# In[ ]:


## Heat map of Correlation Matrix
fig = px.imshow(train_df.drop(['id'], axis=1).corr(),
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0, 
                aspect='auto')
fig.update_layout(height=750, 
                  title = "Heatmap",                  
                  showlegend=False)
fig.show()


# <a id ="2.2"></a><h2 style="background:#d9afed; border:0; border-radius: 8px; color:black"><center>2.2 Feature Distributions</center></h2>

# In[ ]:


## Preparing dataframes for EDA
train_pos_df = train_df.query('target==1')
train_neg_df = train_df.query('target==0')

numerical_columns = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06',
                 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25',
                 'f_26', 'f_28']
categorical_columns = ['f_07', 'f_08', 'f_09', 'f_10', 'f_11', 'f_12',
               'f_13', 'f_14', 'f_15', 'f_16', 'f_17', 'f_18',
               'f_29', 'f_30']
obj_columns = ['f_27']

print(f'numerical_columns: {len(numerical_columns)},  categorical_columns: {len(categorical_columns)},  obj_columns: {len(obj_columns)}')


# In[ ]:


## Numerical Features Statistics
train_df[numerical_columns].describe()


# In[ ]:


## Numerical Features Distribution
fig = plt.figure(figsize=(16, 10))
for i, c in enumerate(numerical_columns):
    ax = fig.add_subplot(4, 4, i+1)
    ax.hist(train_pos_df[c], color='b', alpha=0.5, bins=50)
    ax.hist(train_neg_df[c], color='r', alpha=0.5, bins=50)
    ax.set_title(numerical_columns[i])
    
fig.suptitle('Distributions of Numerical Features (Blue: "target=1", red: "target=0")', fontsize=20)
fig.tight_layout()
plt.show()


# In[ ]:


## Categorical Features Statistics
train_df[categorical_columns].describe()


# In[ ]:


## Categorical Features Distribution
fig = plt.figure(figsize=(16, 10))
for i, c in enumerate(categorical_columns):
    ax = fig.add_subplot(4, 4, i+1)
    x_range = (train_df[c].min(), train_df[c].max())
    #bins = train_df[c].max() - train_df[c].min() + 1
    bins = 50
    ax.hist(train_pos_df[c], color='b', alpha=0.5, range=x_range, bins=bins)
    ax.hist(train_neg_df[c], color='r', alpha=0.5, range=x_range, bins=bins)
    ax.set_title(categorical_columns[i])
    
fig.suptitle('Distributions of Categorical Features (Blue: "target=1", red: "target=0")', fontsize=20)
fig.tight_layout()
plt.show()


# <a id ="2.3"></a><h2 style="background:#d9afed; border:0; border-radius: 8px; color:black"><center>2.3 Exploring f_27 Feature</center></h2>

# In[ ]:


## Preparing dataframes for EDA
f_27_df = train_df[['f_27', 'target']]
f_27_feature_df = f_27_df.drop(['f_27'], axis=1)
f_27_feature_df['n_char'] = f_27_df['f_27'].map(lambda x: len(x))

for i in range(65, 91): ## ASCII of A to Z
    f_27_feature_df[chr(i)] = f_27_df['f_27'].map(lambda x: x.count(chr(i)))
    
f_27_feature_df.describe()


# ---
# 
# - All f_27 values have 10 characters.
# - There are no 'U', 'V', 'W', 'X', 'Y' and 'Z' in f_27.
# 
# ---

# In[ ]:


## Plot the number of appearance of each characters
tmp_df = f_27_feature_df.groupby(['target']).sum()
tmp_df = tmp_df.drop(['n_char'], axis=1)

fig = make_subplots(rows=2, cols=1,
                    subplot_titles=['target=0', 'target=1'],
                    shared_xaxes='all',
                    shared_yaxes='all')
for row in range(2):
    for col in range(1):
        data = go.Bar(x=tmp_df.columns.astype(str).values,
                      y=tmp_df.query(f'target=={row}').values.squeeze())
        fig.add_trace(data, row=row+1, col=col+1)
fig.update_layout(title='Count of Characters',
                  showlegend=False)
fig.show()


# In[ ]:


## Heat map of Correlation Matrix
f_27_feature_df = f_27_feature_df.drop(['U', 'V', 'W', 'X', 'Y', 'Z', 'n_char'], axis=1)
fig = px.imshow(f_27_feature_df.corr(),
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0,
                aspect='auto')
fig.update_layout(height=750, 
                  title = "Heatmap",                  
                  showlegend=False)
fig.show()


# In[ ]:


## The Ratio between characters.
f_27_feature_df['A_B'] = f_27_feature_df['A'] / (f_27_feature_df['B'] + 1)
f_27_feature_df['A_C'] = f_27_feature_df['A'] / (f_27_feature_df['C'] + 1)
f_27_feature_df['A_D'] = f_27_feature_df['A'] / (f_27_feature_df['D'] + 1)
f_27_feature_df['B_C'] = f_27_feature_df['B'] / (f_27_feature_df['C'] + 1)
f_27_feature_df['B_D'] = f_27_feature_df['B'] / (f_27_feature_df['D'] + 1)
f_27_feature_df['B_E'] = f_27_feature_df['B'] / (f_27_feature_df['E'] + 1)
f_27_feature_df['C_D'] = f_27_feature_df['C'] / (f_27_feature_df['D'] + 1)
f_27_feature_df['C_E'] = f_27_feature_df['C'] / (f_27_feature_df['E'] + 1)
f_27_feature_df['D_E'] = f_27_feature_df['D'] / (f_27_feature_df['E'] + 1)

tmp_df = f_27_feature_df[['target', 'A_B', 'A_C', 'A_D', 'B_C', 'B_D', 'B_E', 'C_D', 'C_E', 'D_E']].groupby(['target']).sum()
fig = make_subplots(rows=2, cols=1,
                    subplot_titles=['target=0', 'target=1'],
                    shared_xaxes='all',
                    shared_yaxes='all')
for row in range(2):
    for col in range(1):
        data = go.Bar(x=tmp_df.columns.astype(str).values,
                      y=tmp_df.query(f'target=={row}').values.squeeze())
        fig.add_trace(data, row=row+1, col=col+1)
fig.update_layout(title='Count of Character Raito',
                  showlegend=False)
fig.show()


# <a id ="3"></a><h1 style="background:#a1a8f0; border:0; border-radius: 10px; color:black"><center>3. Model Building</center></h1>

# <a id ="3.1"></a><h2 style="background:#d9afed; border:0; border-radius: 8px; color:black"><center>3.1 Validation Split</center></h2>

# In[ ]:


## Split train samples for cross-validation
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits)
train_df['k_folds'] = -1
for fold, (train_idx, valid_idx) in enumerate(skf.split(X=train_df, y=train_df['target'])):
    train_df['k_folds'][valid_idx] = fold
    
## Check split samples
for i in range(n_splits):
    print(f"fold {i}: {len(train_df.query('k_folds == @i'))} samples")


# In[ ]:


train = train_df.query(f'k_folds != 0').reset_index(drop=True)
valid = train_df.query(f'k_folds == 0').reset_index(drop=True)

print(len(train), len(valid))


# <a id ="3.2"></a><h2 style="background:#d9afed; border:0; border-radius: 8px; color:black"><center>3.2 Dataset</center></h2>

# In[ ]:


def df_to_dataset(dataframe, target=None, shuffle=False,
                  batch_size=5, drop_remainder=False):
    df = dataframe.copy()
    if target is not None:
        labels = df.pop(target)
        df = {key: value[:, tf.newaxis] for key, value in df.items()}
        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    else:
        df = {key: value[:, tf.newaxis] for key, value in df.items()}
        ds = tf.data.Dataset.from_tensor_slices(dict(df))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(batch_size)
    return ds

## Create datasets
batch_size = 512
train_ds = df_to_dataset(train,
                         target='target',
                         shuffle=True,
                         batch_size=batch_size,
                         drop_remainder=True)
valid_ds = df_to_dataset(valid,
                         target='target',
                         shuffle=False,
                         batch_size=batch_size,
                         drop_remainder=True)

## Display a batch sample
example = next(iter(train_ds))[0]
for key in example:
    print(f'{key}, shape:{example[key].shape}, {example[key].dtype}')


# <a id ="3.3"></a><h2 style="background:#d9afed; border:0; border-radius: 8px; color:black"><center>3.3 Preprocessing Model</center></h2>

# In[ ]:


numerical_columns = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06',
                 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25',
                 'f_26', 'f_28']
categorical_columns = ['f_07', 'f_08', 'f_09', 'f_10', 'f_11', 'f_12',
               'f_13', 'f_14', 'f_15', 'f_16', 'f_17', 'f_18',
               'f_29', 'f_30']
text_columns = ['f_27']

def create_preprocess_inputs(numerical, categorical, text):
    preprocess_inputs = {}
    numerical_inputs = {key: layers.Input(shape=(1, ), dtype='float64') for key in numerical}
    categorical_inputs = {key: layers.Input(shape=(1, ), dtype='int64') for key in categorical}
    text_inputs = {key: layers.Input(shape=(1, ), dtype='string') for key in text}
    preprocess_inputs.update(**numerical_inputs, **categorical_inputs, **text_inputs)
    return preprocess_inputs

preprocess_inputs = create_preprocess_inputs(numerical_columns,
                                             categorical_columns,
                                             text_columns)
preprocess_inputs


# In[ ]:


## Preprocess layers for numerical_features
normalize_layers = {}
for nc in numerical_columns:
    normalize_layer = layers.Normalization(mean=train_df[nc].mean(), variance=train_df[nc].var())
    normalize_layers[nc] = normalize_layer
normalize_layers


# In[ ]:


## Preprocess layers for categorical_features
lookup_layers = {}
for cc in categorical_columns:
    lookup_layer = layers.IntegerLookup(vocabulary=train_df[cc].unique(),
                                       output_mode='int')
    lookup_layers[cc] = lookup_layer
lookup_layers


# In[ ]:


## Split the text_feature into unigram
def split_unigram(input_data):
    s = tf.strings.regex_replace(input_data, '', ' ')
    s = tf.strings.strip(s)
    s = tf.strings.split(s, sep = ' ')
    return s

uni_vocabulary = [chr(i) for i in range(65, 91)] ## ASCII of A to Z

## Preprocess layers for unigram
vectorize_layers = {}
for tc in text_columns:
    uni_vectorize_layer = layers.TextVectorization(standardize=None,
                                                   split=split_unigram,
                                                   vocabulary=uni_vocabulary)
    vectorize_layers[f'{tc}_unigram'] = uni_vectorize_layer
vectorize_layers


# In[ ]:


## Split the text_feature into bigram
def split_bigram(input_data):
    s = tf.strings.regex_replace(input_data, '', ' ')
    s = tf.strings.strip(s)
    s = tf.strings.split(s, sep = ' ')
    s = tf.strings.ngrams(s, 2, separator='')
    return s

bi_vocabulary = {uni1+uni2 for uni1 in uni_vocabulary for uni2 in uni_vocabulary}
bi_vocabulary = list(bi_vocabulary)
bi_vocabulary.sort()

## Preprocess layers for bigram
for tc in text_columns:
    bi_vectorize_layer = layers.TextVectorization(standardize=None,
                                                  split=split_bigram,
                                                  vocabulary=bi_vocabulary)
    vectorize_layers[f'{tc}_bigram'] = bi_vectorize_layer
vectorize_layers


# In[ ]:


preprocess_outputs = {}
for key in preprocess_inputs:
    if key in normalize_layers:
        output = normalize_layers[key](preprocess_inputs[key])
        normalize_layers[key]
        preprocess_outputs[key] = output
    elif key in lookup_layers:
        output = lookup_layers[key](preprocess_inputs[key])
        preprocess_outputs[key] = output
    else:
        uni_output = vectorize_layers[f'{key}_unigram'](preprocess_inputs[key])
        bi_output = vectorize_layers[f'{key}_bigram'](preprocess_inputs[key])
        preprocess_outputs[f'{key}_unigram'] = uni_output
        preprocess_outputs[f'{key}_bigram'] = bi_output
        
preprocess_outputs


# In[ ]:


## Create the preprocessing model
preprocessing_model = tf.keras.Model(preprocess_inputs,
                                     preprocess_outputs)

## Apply the preprocessing model in tf.data.Dataset.map
train_ds = train_ds.map(lambda x, y: (preprocessing_model(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
valid_ds = valid_ds.map(lambda x, y: (preprocessing_model(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE)

## Display a preprocessed input sample
example = next(train_ds.take(1).as_numpy_iterator())
for key in example[0]:
    print(f'{key}, shape:{example[0][key].shape}, {example[0][key].dtype}')


# <a id ="3.4"></a><h2 style="background:#d9afed; border:0; border-radius: 8px; color:black"><center>3.4 Training Model</center></h2>

# In[ ]:


## Training model inputs
def create_model_inputs(numerical, categorical, unigram_keys, bigram_keys):
    model_inputs = {}
    
    normalized_inputs = {key: layers.Input(shape=(1, ), dtype='float64') for key in numerical}
    lookup_inputs = {key: layers.Input(shape=(1, ), dtype='int64') for key in categorical}
    #vectorized_inputs = {key: layers.Input(shape=(10, ), dtype='int64') for key in text}
    unigram_inputs = {key: layers.Input(shape=(10, ), dtype='int64') for key in unigram_keys}
    bigram_inputs = {key: layers.Input(shape=(9, ), dtype='int64') for key in bigram_keys}
    
    model_inputs.update(**normalized_inputs, **lookup_inputs, **unigram_inputs, **bigram_inputs)
    return model_inputs

unigram_keys = ['f_27_unigram']
bigram_keys = ['f_27_bigram']
model_inputs = create_model_inputs(numerical_columns,
                                   categorical_columns,
                                   unigram_keys,
                                   bigram_keys)
model_inputs


# In[ ]:


## Create Embedding Layers
cat_embedding_dim = 16
unigram_embedding_dim = 16
bigram_embedding_dim = 32

numerical_feature_list = []
encoded_categorical_feature_list = []
encoded_text_feature_list = []
for key in model_inputs:
    if key in numerical_columns:
        numerical_feature_list.append(model_inputs[key])
    elif key in categorical_columns:
        embedding = layers.Embedding(input_dim=lookup_layers[key].vocabulary_size(),
                                     output_dim=cat_embedding_dim)
        encoded_categorical_feature = embedding(model_inputs[key])
        encoded_categorical_feature_list.append(encoded_categorical_feature)
    elif key in unigram_keys:
        embedding = layers.Embedding(input_dim=vectorize_layers[key].vocabulary_size(),
                                   output_dim=unigram_embedding_dim)
        encoded_text_feature = embedding(model_inputs[key])
        encoded_text_feature_list.append(encoded_text_feature)
    elif key in bigram_keys:
        embedding = layers.Embedding(input_dim=vectorize_layers[key].vocabulary_size(),
                                   output_dim=bigram_embedding_dim)
        encoded_text_feature = embedding(model_inputs[key])
        encoded_text_feature_list.append(encoded_text_feature)

encoded_categorical_features = tf.concat(encoded_categorical_feature_list, axis=1)
encoded_categorical_features.shape


# ### 1D-CNN for unigram and bigram features

# In[ ]:


## 1D-CNN for N-gram features
def create_conv1d(n_filters, kernels):
    conv1d = tf.keras.models.Sequential([
        layers.Conv1D(filters=n_filters[0], kernel_size=kernels[0], strides=1, use_bias=False), 
        layers.BatchNormalization(), 
        layers.ReLU(),
        layers.Conv1D(filters=n_filters[1], kernel_size=kernels[1], strides=1, use_bias=False), 
        layers.BatchNormalization(), 
        layers.ReLU(),
        layers.Conv1D(filters=n_filters[2], kernel_size=kernels[2], strides=1, use_bias=False), 
        layers.BatchNormalization(), 
        layers.ReLU(),
        layers.GlobalAveragePooling1D()
    ])
    return conv1d
    
for encoded_text_feature in encoded_text_feature_list:
    if encoded_text_feature.shape[1] == 10: ##unigram
        conv1d_uni = create_conv1d([32, 64, 128], [4, 4, 4])
        uni_numerical = conv1d_uni(encoded_text_feature)
        numerical_feature_list.append(uni_numerical)
    elif encoded_text_feature.shape[1] == 9: ##bigram
        conv1d_bi = create_conv1d([64, 128, 256], [4, 4, 3])
        bi_numerical = conv1d_bi(encoded_text_feature)
        numerical_feature_list.append(bi_numerical)
        
numerical_features = layers.concatenate(numerical_feature_list)
numerical_features.shape


# ### Tab Transformer
# 
# The TabTransformer architecture works as follows:
# 
# - All the categorical features are encoded as embeddings, using the same embedding_dims. This means that each value in each categorical feature will have its own embedding vector.
# 
# - A column embedding, one embedding vector for each categorical feature, is added (point-wise) to the categorical feature embedding.
# 
# - The embedded categorical features are fed into a stack of Transformer blocks. Each Transformer block consists of a multi-head self-attention layer followed by a feed-forward layer.
# 
# - The outputs of the final Transformer layer, which are the contextual embeddings of the categorical features, are concatenated with the input numerical features, and fed into a final MLP block.
# 
# <img src="https://raw.githubusercontent.com/keras-team/keras-io/master/examples/structured_data/img/tabtransformer/tabtransformer.png" width="500"/>
# 

# In[ ]:


def create_mlp(hidden_units, dropout_rate, 
               activation, normalization_layer,
               name=None):
    mlp_layers = []
    for units in hidden_units:
        mlp_layers.append(normalization_layer),
        mlp_layers.append(layers.Dense(units,
                                       activation=activation))
        mlp_layers.append(layers.Dropout(dropout_rate))
    return keras.Sequential(mlp_layers, name=name)


# In[ ]:


## Create TabTransformer model
num_transformer_blocks = 4
num_heads = 4
dropout_rate = 0.2
mlp_hidden_units_factors = [2, 1] 

for block_idx in range(num_transformer_blocks):
    ## Create a multi-head attention layer
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=cat_embedding_dim,
        dropout=dropout_rate,
        name=f'multi-head_attention_{block_idx}'
    )(encoded_categorical_features, encoded_categorical_features)
    ## Skip connection 1
    x = layers.Add(
        name=f'skip_connection1_{block_idx}'
    )([attention_output, encoded_categorical_features])
    ## Layer normalization 1
    x = layers.LayerNormalization(
        name=f'layer_norm1_{block_idx}', epsilon=1e-6
    )(x)
    ## Feedforward
    feedforward_output = keras.Sequential([
        layers.Dense(cat_embedding_dim, activation=keras.activations.gelu),
        layers.Dropout(dropout_rate),
    ], name=f'feedforward_{block_idx}'
    )(x)
    ## Skip connection 2
    x = layers.Add(
        name=f'skip_connection2_{block_idx}'
    )([feedforward_output, x])
    ## Layer normalization 2
    encoded_categorical_features = layers.LayerNormalization(
        name=f'layer_norm2_{block_idx}', epsilon=1e-6
    )(x)
    
contextualized_categorical_features = layers.Flatten(
)(encoded_categorical_features)
    
## Numerical features
numerical_features = layers.LayerNormalization(
    name='numerical_norm', epsilon=1e-6
)(numerical_features)

## Concatenate categorical features with numerical features
features = layers.Concatenate()([
    contextualized_categorical_features,
    numerical_features])

## Final MLP
mlp_hidden_units = [
    factor * features.shape[-1] for factor in mlp_hidden_units_factors
]

features = create_mlp(
    hidden_units=mlp_hidden_units, 
    dropout_rate=dropout_rate, 
    activation=keras.activations.selu, 
    normalization_layer=layers.BatchNormalization(),
    name='MLP'
)(features)

# Add a sigmoid to cap the output from 0 to 1
model_outputs = layers.Dense(
    units=1, 
    activation='sigmoid', 
    name='sigmoid'
)(features)

training_model = keras.Model(inputs=model_inputs,
                             outputs=model_outputs)


# In[ ]:


LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0001

optimizer = tfa.optimizers.AdamW(
    learning_rate=LEARNING_RATE, 
    weight_decay=WEIGHT_DECAY)

loss_fn = keras.losses.BinaryCrossentropy(
    from_logits=False)

training_model.compile(optimizer=optimizer,
                       loss=loss_fn,
                       metrics=[keras.metrics.AUC()])

training_model.summary()


# <a id ="4"></a><h1 style="background:#a1a8f0; border:0; border-radius: 10px; color:black"><center>4. Model Training</center></h1>

# <a id ="4.1"></a><h2 style="background:#d9afed; border:0; border-radius: 8px; color:black"><center>4.1 Learning Rate Finder</center></h2>

# In[ ]:


class LRFind(tf.keras.callbacks.Callback):
    def __init__(self, min_lr, max_lr, n_rounds):
        self.min_lr = tf.constant(min_lr)
        self.max_lr = tf.constant(max_lr)
        self.step_up = tf.constant((max_lr / min_lr) ** (1 / n_rounds))
        self.lrs = []
        self.losses = []

    def on_train_begin(self, logs=None):
        self.weights= self.model.get_weights()
        self.model.optimizer.lr = self.min_lr 

    def on_train_batch_end(self, batch, logs=None):
        self.lrs.append(self.model.optimizer.lr.numpy())
        self.losses.append(logs['loss'])
        self.model.optimizer.lr = self.model.optimizer.lr * self.step_up 
        if self.model.optimizer.lr > self.max_lr:
            self.model.stop_training = True 

    def on_train_end(self, logs=None):
        self.model.set_weights(self.weights)


# In[ ]:


lr_find_epochs = 1
lr_finder_steps = 500 
lr_find = LRFind(1e-6, 1e1, lr_finder_steps)

lr_find_batch_size = 512
lr_find_ds = df_to_dataset(train_df, target='target', batch_size=lr_find_batch_size) 
lr_find_ds = lr_find_ds.map(lambda x, y: (preprocessing_model(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE)

training_model.fit(lr_find_ds,
                   steps_per_epoch=lr_finder_steps,
                   epochs=lr_find_epochs,
                   callbacks=[lr_find])

plt.plot(lr_find.lrs, lr_find.losses)
plt.xscale('log')
plt.show()


# <a id ="4.2"></a><h2 style="background:#d9afed; border:0; border-radius: 8px; color:black"><center>4.2 Model Training</center></h2>

# In[ ]:


epochs = 10
steps_per_epoch = len(train)//batch_size

## Re-construct the model
model_config = training_model.get_config()
training_model = tf.keras.Model.from_config(model_config)

## Model compile
learning_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-2,
    decay_steps=epochs*steps_per_epoch,
    alpha=0.0)

weight_decay = 0.0001

optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_schedule,
        weight_decay=weight_decay)

loss_fn = keras.losses.BinaryCrossentropy(
    from_logits=False)

training_model.compile(optimizer=optimizer,
                       loss=loss_fn,
                       metrics=[keras.metrics.AUC()])

## Checkpoint callback
checkpoint_filepath = './tmp/model/exp_ckpt'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, 
    save_weights_only=True,
    monitor='val_loss', 
    mode='min', 
    save_best_only=True)


# In[ ]:


training_model.fit(train_ds, epochs=epochs, shuffle=True,
                   validation_data=valid_ds,
                   callbacks=[model_checkpoint_callback])

training_model.load_weights(checkpoint_filepath)


# <a id ="5"></a><h1 style="background:#a1a8f0; border:0; border-radius: 10px; color:black"><center>5. Inference  </center></h1>

# In[ ]:


## Inference model = preprocessing model + training model
inference_inputs = preprocessing_model.input
inference_outputs = training_model(preprocessing_model(inference_inputs))
inference_model = tf.keras.Model(inputs=inference_inputs,
                                 outputs=inference_outputs)


# In[ ]:


## Test Dataset
test = test_df
test_ds = df_to_dataset(test,
                        target=None,
                        shuffle=False,
                        batch_size=batch_size,
                        drop_remainder=False)

## Display a test sample
example = next(test_ds.take(1).as_numpy_iterator())
for key in example:
    print(f'{key}, shape:{example[key].shape}, {example[key].dtype}')


# In[ ]:


## Inference and submission
preds = inference_model.predict(test_ds)
preds = np.squeeze(preds)
submission_df['target'] = preds
submission_df.to_csv('submission.csv', index=False)
submission_df.head()


# In[ ]:




