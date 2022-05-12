#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[ ]:


import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import GlobalMaxPooling1D, Add
from tensorflow.keras.layers import Bidirectional, LSTM, GRU
from tensorflow.keras.layers import Input, BatchNormalization
from tensorflow.keras.layers import Dense, Concatenate, Multiply

np.random.seed(42)
tf.random.set_seed(42)


# ## Load source datasets

# In[ ]:


train = pd.read_csv("../input/tabular-playground-series-apr-2022/train.csv")
train.sort_values(by=['sequence','step'], inplace=True)
print(f"train: {train.shape}")
train.head()


# In[ ]:


train_labels = pd.read_csv("../input/tabular-playground-series-apr-2022/train_labels.csv")
train_labels.sort_values(by=['sequence'], inplace=True)
print(f"train_labels: {train_labels.shape}")
train_labels.head()


# In[ ]:


test = pd.read_csv("../input/tabular-playground-series-apr-2022/test.csv")
test.sort_values(by=['sequence','step'], inplace=True)
print(f"test: {test.shape}")
test.head()


# In[ ]:


submission = pd.read_csv("../input/tabular-playground-series-apr-2022/sample_submission.csv")
print(f"submission: {submission.shape}")
submission.head()


# ## Feature Engineering

# In[ ]:


def sub_imp(x):
    if x < 25:
        return 0
    elif x > 95:
        return 2
    else:
        return 1


# In[ ]:


def add_features(df):
    for col in tqdm(sensor_cols):
        
        for window in [1,2,3,6]:
            df[f'{col}_lead_diff{window}'] = df[col] - df.groupby('sequence')[col].shift(window).fillna(0)
            df[f'{col}_lag_diff{window}'] = df[col] - df.groupby('sequence')[col].shift(-1*window).fillna(0)
        
        for window in [3,6,12,24]:
            df[col+'_roll_'+str(window)+'_mean'] = df.groupby('sequence')[col]                                                     .rolling(window=window, min_periods=1)                                                     .mean().reset_index(level=0,drop=True)
            
            df[col+'_roll_'+str(window)+'_std'] = df.groupby('sequence')[col]                                                    .rolling(window=window, min_periods=1)                                                    .std().reset_index(level=0,drop=True)
            
            df[col+'_roll_'+str(window)+'_sum'] = df.groupby('sequence')[col]                                                    .rolling(window=window, min_periods=1)                                                    .sum().reset_index(level=0,drop=True)
    
    df.fillna(0, inplace=True)
    
    sub_stat = df[['sequence', 'subject']]                .drop_duplicates()                .groupby('subject')                .agg({'sequence': 'count'})                .rename(columns={'sequence': 'count'}).reset_index()
    
    df = df.merge(sub_stat, on='subject', how='left')
    df['sub_imp'] = df['count'].apply(lambda x: sub_imp(x))
    df.drop('count', axis=1, inplace=True)
        
    return df


# In[ ]:


sensor_cols = [col for col in train.columns if 'sensor' in col]
train = add_features(train)
test = add_features(test)
print(f"train: {train.shape} \ntest: {test.shape}")


# In[ ]:


train.drop(["sequence","step","subject"], axis=1, inplace=True)
test.drop(["sequence","step","subject"], axis=1, inplace=True)


# In[ ]:


scaler = QuantileTransformer(n_quantiles=2000, 
                             output_distribution='normal', 
                             random_state=42).fit(train)
train = scaler.transform(train)
test = scaler.transform(test)


# In[ ]:


train = train.reshape(-1, 60, train.shape[-1]).copy()
test = test.reshape(-1, 60, train.shape[-1]).copy()
train_labels = train_labels['state'].values.reshape(-1,1)
print(f"train: {train.shape} \ntest: {test.shape} \ntrain_labels {train_labels.shape}")


# In[ ]:


del scaler
gc.collect()


# ## Hardware config

# In[ ]:


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    BATCH_SIZE = strategy.num_replicas_in_sync * 32
    print("Running on TPU:", tpu.master())
    print(f"Batch Size: {BATCH_SIZE}")
    
except ValueError:
    strategy = tf.distribute.get_strategy()
    BATCH_SIZE = 256
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    print(f"Batch Size: {BATCH_SIZE}")


# ## Keras Model

# In[ ]:


def dnn_model():
    
    x_input = Input(shape=(train.shape[-2:]))
    
    x1 = Bidirectional(LSTM(units=768, return_sequences=True))(x_input)
    x2 = Bidirectional(LSTM(units=512, return_sequences=True))(x1)
    x3 = Bidirectional(LSTM(units=384, return_sequences=True))(x2)
    x4 = Bidirectional(LSTM(units=256, return_sequences=True))(x3)
    x5 = Bidirectional(LSTM(units=128, return_sequences=True))(x4)
    
    z1 = Bidirectional(GRU(units=384, return_sequences=True))(x2)
    z2 = Multiply()([x3, z1])
    
    z3 = Bidirectional(GRU(units=256, return_sequences=True))(z2)
    z4 = Multiply()([x4, z3])
    
    z5 = Bidirectional(GRU(units=128, return_sequences=True))(z4)
    
    x = Concatenate(axis=2)([x1, x3, x5, z1, z3, z5])
    x = GlobalMaxPooling1D()(x)
    
    x = Dense(units=1024, activation='selu')(x)
    x = Dense(units=128, activation='selu')(x)
    
    x_output = Dense(units=1, activation='sigmoid')(x)

    model = Model(inputs=x_input, outputs=x_output, 
                  name='TPS_Apr22_TFv2_Model')
    return model


# In[ ]:


model = dnn_model()
model.summary()


# In[ ]:


plot_model(
    model, 
    to_file='TPS_Apr22_TFv2_Model.png', 
    show_shapes=True,
    show_layer_names=True
)


# In[ ]:


with strategy.scope():
    
    VERBOSE = 0
    test_preds = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(train, train_labels)):
        X_train, X_valid = train[train_idx], train[test_idx]
        y_train, y_valid = train_labels[train_idx], train_labels[test_idx]
        
        model = dnn_model()
        model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['AUC'])

        lr = ReduceLROnPlateau(monitor="val_auc", factor=0.75, 
                               patience=4, verbose=VERBOSE)
        
        save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
        chk_point = ModelCheckpoint(f'./TPS_Apr22_TFv2_Model_{fold+1}C.h5', options=save_locally, 
                                    monitor='val_auc', verbose=VERBOSE, 
                                    save_best_only=True, mode='max')

        es = EarlyStopping(monitor="val_auc", patience=10, 
                           verbose=VERBOSE, mode="max", 
                           restore_best_weights=True)
        
        model.fit(X_train, y_train, 
                  validation_data=(X_valid, y_valid), 
                  epochs=100,
                  verbose=VERBOSE,
                  batch_size=BATCH_SIZE, 
                  callbacks=[lr, chk_point, es])
        
        load_locally = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
        model = load_model(f'./TPS_Apr22_TFv2_Model_{fold+1}C.h5', options=load_locally)
        
        y_pred = model.predict(X_valid, batch_size=BATCH_SIZE, verbose=VERBOSE).squeeze()
        score = roc_auc_score(y_valid, y_pred)
        print(f"Fold-{fold+1} | OOF Score: {score}")
        
        test_preds.append(model.predict(test, batch_size=BATCH_SIZE, verbose=VERBOSE).squeeze())
        
        del model, y_pred
        del X_train, X_valid
        del y_train, y_valid
        gc.collect()


# ## Create submission file

# In[ ]:


submission["state"] = np.mean(np.vstack(test_preds), axis=0)
submission.to_csv('mean_submission.csv', index=False)
submission.head()


# In[ ]:


submission["state"] = np.median(np.vstack(test_preds), axis=0)
submission.to_csv('median_submission.csv', index=False)
submission.head()


# In[ ]:


# Good Day!!

