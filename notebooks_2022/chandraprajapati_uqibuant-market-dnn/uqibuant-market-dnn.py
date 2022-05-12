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
#for dirname, _, filenames in os.walk('/kaggle/input'):
 #   for filename in filenames:
  #      print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import gc


# In[ ]:


get_ipython().run_cell_magic('time', '', 'n_features = 300\nfeatures = [f"f_{i}" for i in range(300)]\nfeature_column = [\'investment_id\', \'time_id\' ] + features\ntrain = pd.read_pickle(\'../input/ubiquant-market-prediction-half-precision-pickle/train.pkl\' )\ntrain.head()')


# In[ ]:


print(train.shape)
#print(train.info())


# Let us try to understand the data
# 1. There are 1211 time_id's recorded with min value 0 to some value and max value 1219.
# 1. There are 3579 investment_id's recorded with min 0 and max 3773.

# In[ ]:


print(train['time_id'].nunique())
print(train['time_id'].min())
print(train['time_id'].max())
print(train['investment_id'].nunique())
print(train['investment_id'].min())
print(train['investment_id'].max())


# In[ ]:


import tensorflow as tf
from keras import Input, layers, Model
from keras.metrics import RootMeanSquaredError


# In[ ]:


investment_id = train['investment_id']
list(investment_id.unique())
investment_ids = list(investment_id.unique())
#print(len(investment_ids))
#investment_ids


# In[ ]:


investment_id_size = len(investment_ids) + 1
print(investment_id_size)
# an IntegerLookup layer maps integer features to contiguous ranges. This maps a set of arbitrary integer input token 
# into indexed integer output via a table based vocabulary lookup. The output indices will be contigously arranged upto
# maximum vocab size. The layer supports multiple options for encoding the output via output_mode 
investment_id_lookup_layer = layers.IntegerLookup(max_tokens = investment_id_size) # lookup layer with adapted vocab
investment_id_lookup_layer.adapt(pd.DataFrame({'investment_ids':investment_ids})) # 
#investment_id_lookup_layer.get_vocabulary()


# In[ ]:


# lets split the features and target
investment_id = train['investment_id'] 
target = train['target']
train = train.drop(['time_id','investment_id','target'], axis=1)
train.head()


# In[ ]:


# prepare the dataset into X and Y, with X being features and investment_id and Y being the target
def preprocess(X,Y):
    return X,Y

def make_dataset(features, investment_id, target, batch_size = 1024, mode = 'train'):#investment_id, 
    ds = tf.data.Dataset.from_tensor_slices(((investment_id, features), target)) #, investment_id
    #ds = tf.constant(((features, investment_id), target))
    ds = ds.map(preprocess)
    if mode == 'train':
        ds = ds.shuffle(4096)
    
    ds = ds.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
    # cache transform keeps the dataset either into menory or on local storage after they are loaded after firts epoch
    # prefetch makes sure to overlap the preprocessing the dataset and model execution while training
    return ds


# In[ ]:


# build the deep neural network model with the help of helper funtion build_model
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay( initial_learning_rate = 0.001,
                                                              decay_steps = 4000, decay_rate = 0.98)
def build_model():
    # the inputs are investment_id and featuresso we need to create two separate inputs 
    input_investment_id = Input((1,), dtype=tf.uint16)
    input_features = Input((300,), dtype=tf.float16)
    
    investment_id_x = investment_id_lookup_layer(input_investment_id)
    # create embedding layer of shape (1 X 32)
    investment_id_x = layers.Embedding(investment_id_size, 32, input_length = 1)(investment_id_x)
    investment_id_x = layers.Reshape((-1,))(investment_id_x)  # get layer of shape (32)
    investment_id_x = layers.Dense(64, activation='swish')(investment_id_x) # size = 64
    investment_id_x = layers.Dense(64, activation='swish')(investment_id_x) # size = 64
    investment_id_x = layers.Dense(64, activation='swish')(investment_id_x)
    investment_id_x = layers.Dense(64, activation='swish')(investment_id_x)
    
    #feature_x = layers.Reshape((-1,))(input_features)
    feature_x = layers.Dense(256, activation = 'swish')(input_features)
    feature_x = layers.Dense(256, activation = 'swish')(feature_x)
    feature_x = layers.Dense(256, activation = 'swish')(feature_x)
    feature_x = layers.Dense(256, activation = 'swish')(feature_x)
    
    # concatenate the two input layers
    x = layers.Concatenate(axis=1)([investment_id_x, feature_x]) #investment_id_x, 
    x = layers.Dense(512, activation='swish',kernel_regularizer='l2')(x)
    x = layers.Dense(128, activation='swish',kernel_regularizer='l2')(x)
    x = layers.Dense(32, activation='swish',kernel_regularizer='l2')(x) 
    
    output = layers.Dense(1)(x)
    rmse = RootMeanSquaredError(name = 'rmse')
    model = Model(inputs = [input_investment_id, input_features], outputs = [output]) #input_investment_id, 
    model.compile(optimizer=tf.optimizers.Adam(lr_schedule), loss = 'mse', metrics=['mse', 'mae', 'mape', rmse])
    return model
    


# In[ ]:


from keras.callbacks import EarlyStopping, ModelCheckpoint
import scipy.stats as stats
from sklearn.model_selection import train_test_split


# In[ ]:


get_ipython().run_cell_magic('time', '', '# lets use stratified K-Fold to do cross validation\nfrom sklearn.model_selection import StratifiedKFold\nkfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=112)\n#kfold.get_n_splits([investment_id, features], target)\nmodels = []\nfor index, (train_index, val_index) in enumerate(kfold.split(train, investment_id)): #\n    #print(\'index: \', index, \'train_index: \', train_index, \'test_index: \', val_index)\n    X_train, X_val = train.iloc[train_index], train.iloc[val_index]\n    Y_train, Y_val = target[train_index], target[val_index]#print(X_train)\n    investment_id_train = investment_id[train_index]\n    investment_id_val = investment_id[val_index]\n    \n    train_ds = make_dataset(X_train, investment_id_train, Y_train) #investment_id_train,\n    val_ds = make_dataset(X_val, investment_id_val, Y_val, mode = \'validation\') #investment_id_val, \n    checkpoint = ModelCheckpoint(f"model_{index}", save_best_only=True)\n    earlystoping = EarlyStopping(patience=10)\n    \n    model = build_model()\n    print(model.summary())\n    history = model.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=[checkpoint, earlystoping])\n    model = tf.keras.models.load_model(f"model_{index}")\n    models.append(model)\n    pearson_coef = stats.pearsonr(model.predict(val_ds).ravel(), Y_val.values)[0]\n    print(\'pearson coefficients is: \', pearson_coef)\n    \n    del investment_id_train\n    del investment_id_val\n    del X_train\n    del X_val\n    del Y_train\n    del Y_val\n    del train_ds\n    del val_ds\n    gc.collect()\n    break')


# In[ ]:


#tf.keras.utils.plot_model(model, show_shapes=True)


# In[ ]:


test = pd.read_csv('../input/ubiquant-market-test/example_test.csv')
print(test.shape)
test.head()
X_test = test.drop(['row_id','time_id','investment_id'],axis=1)
X_test.head()


# In[ ]:


# preprocess test dataset
def preprocess_test(investment_id, test):
    return (investment_id, test), 0
def make_test_dataset(features, investment_id, batch_size = 1024):
    test_ds = tf.data.Dataset.from_tensor_slices((investment_id, features))
    test_ds = test_ds.map(preprocess_test)
    test_ds = test_ds.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
    return test_ds
def predict_test(model, ds):
    test_preds = []
    for model in models:
        test_pred = model.predict(ds)
        test_preds.append(test_pred)
    return np.mean(test_preds, axis=0)

test_ds = make_test_dataset(X_test, test['investment_id']) 
test_pred = predict_test(model, test_ds)
test_pred


# In[ ]:


sample_submission = pd.read_csv('../input/ubiquant-market-test/example_sample_submission.csv')
sample_submission['target'] = test_pred
sample_submission
sample_submission.to_csv('sample_submission.csv')


# In[ ]:


import ubiquant
env = ubiquant.make_env()   # initialize the environment
iter_test = env.iter_test()    # an iterator which loops over the test set and sample submission
for (test_df, sample_prediction_df) in iter_test:
    test_ds = make_test_dataset(test_df[features], test_df['investment_id'])
    sample_prediction_df['target'] = predict_test(model, test_ds) # make your predictions here
    env.predict(sample_prediction_df)   # register your predictionsa


# In[ ]:




