#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold


# In[ ]:


# Detect hardware
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
except ValueError:
    tpu = None
    gpus = tf.config.experimental.list_logical_devices("GPU")
    
# Select appropriate distribution strategy
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu) # Going back and forth between TPU and host is expensive. Better to run 128 batches on the TPU before reporting back.
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])  
elif len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
elif len(gpus) == 1:
    strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
    print('Running on single GPU ', gpus[0].name)
else:
    strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
    print('Running on CPU')
print("Number of accelerators: ", strategy.num_replicas_in_sync)


# In[ ]:


batch_size_per_replica = 256
batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
batch_size


# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-apr-2022/train.csv')
test = pd.read_csv('../input/tabular-playground-series-apr-2022/test.csv')
train_labels = pd.read_csv('../input/tabular-playground-series-apr-2022/train_labels.csv')
sample = pd.read_csv('../input/tabular-playground-series-apr-2022/sample_submission.csv')


# In[ ]:


def feat_eng(df):
    
    seq_df=pd.DataFrame()
    sensors=[col for col in df if col.startswith('sensor')]
    print('Processing New DF')
    
    temp = df.subject.value_counts().sort_values() // 60
    subject_count = df.merge(temp, left_on='subject', right_index=True, how='left').iloc[:, -1]
    df['subject_count'] = subject_count
    
    for sensor in tqdm(sensors):
#         if sensor != 'sensor_02':
#             df['{}_groupby_sensor_2'.format(sensor)] = df.merge(df.groupby('sensor_02')[sensor].median(), 
#                                                                  left_on='sensor_02', right_index=True, how='left').iloc[:, -1]
#             df['{}_groupby_sensor_2_diff'.format(sensor)] = df[sensor] - df['{}_groupby_sensor_2'.format(sensor)]
        df['{}_lag1'.format(sensor)] = df.groupby('sequence')[sensor].shift(1)
        df['{}_lag1'.format(sensor)].fillna(df[sensor].median(), inplace=True)
        df['{}_diff'.format(sensor)] = df[sensor] - df['{}_lag1'.format(sensor)] 
        df['{}_roll_mean3'.format(sensor)]=df['{}'.format(sensor)].rolling(window=3).mean()
        df['{}_roll_mean6'.format(sensor)]=df['{}'.format(sensor)].rolling(window=6).mean()
        df['{}_roll_mean9'.format(sensor)]=df['{}'.format(sensor)].rolling(window=9).mean()
        df['{}_roll_mean3'.format(sensor)].fillna(df['{}_roll_mean3'.format(sensor)].median(), inplace=True)
        df['{}_roll_mean6'.format(sensor)].fillna(df['{}_roll_mean6'.format(sensor)].median(), inplace=True)
        df['{}_roll_mean9'.format(sensor)].fillna(df['{}_roll_mean9'.format(sensor)].median(), inplace=True)
        
        if sensor == 'sensor_02':
            df['{}_lag2'.format(sensor)] = df.groupby('sequence')[sensor].shift(2)
            df['{}_lag2'.format(sensor)].fillna(df[sensor].median(), inplace=True)
            df['{}_diff_lag2'.format(sensor)] = df[sensor] - df['{}_lag2'.format(sensor)]
            df['{}_lag3'.format(sensor)] = df.groupby('sequence')[sensor].shift(3)
            df['{}_lag3'.format(sensor)].fillna(df[sensor].median(), inplace=True)
            df['{}_diff_lag3'.format(sensor)] = df[sensor] - df['{}_lag3'.format(sensor)]
            df['{}_lag5'.format(sensor)] = df.groupby('sequence')[sensor].shift(5)
            df['{}_lag5'.format(sensor)].fillna(df[sensor].median(), inplace=True)
            df['{}_diff_lag5'.format(sensor)] = df[sensor] - df['{}_lag5'.format(sensor)]

            df['{}_lead1'.format(sensor)] = df.groupby('sequence')[sensor].shift(-1)
            df['{}_lead1'.format(sensor)].fillna(df[sensor].median(), inplace=True)
            df['{}_diff_lead1'.format(sensor)] = df[sensor] - df['{}_lead1'.format(sensor)]
            df['{}_lead2'.format(sensor)] = df.groupby('sequence')[sensor].shift(-2)
            df['{}_lead2'.format(sensor)].fillna(df[sensor].median(), inplace=True)
            df['{}_diff_lead2'.format(sensor)] = df[sensor] - df['{}_lead2'.format(sensor)]
            df['{}_lead3'.format(sensor)] = df.groupby('sequence')[sensor].shift(-3)
            df['{}_lead3'.format(sensor)].fillna(df[sensor].median(), inplace=True)
            df['{}_diff_lead3'.format(sensor)] = df[sensor] - df['{}_lead3'.format(sensor)]
            df['{}_lead5'.format(sensor)] = df.groupby('sequence')[sensor].shift(-5)
            df['{}_lead5'.format(sensor)].fillna(df[sensor].median(), inplace=True)
            df['{}_diff_lead5'.format(sensor)] = df[sensor] - df['{}_lead5'.format(sensor)]
        
        
        
#         aa = df.groupby(['sequence','subject'], as_index=False)[sensor].mean()
#         aa = aa.sort_values(by=['sequence', 'subject'])
#         train2 = train.copy()
#         df = df.merge(aa, left_on=['sequence', 'subject'], 
#                       right_on=['sequence', 'subject'], how='left', suffixes=['', '_grouped_by_seq_sub_mean'])
#         df['diff_'] = df[sensor] - df[f'{sensor}_grouped_by_seq_sub_mean']

        if sensor == 'sensor_02':
            df['{}_groupby_count'.format(sensor)] = df.merge(df.groupby('subject_count')[sensor].mean(), 
                                                                     left_on='subject_count', right_index=True, how='left').iloc[:, -1]

            df['{}_groupby_count_diff'.format(sensor)] = df[sensor] - df['{}_groupby_count'.format(sensor)]
        
        s_diff='{}_diff'.format(sensor)
        seq_df['{}_mean'.format(sensor)] = df.groupby(['sequence','subject'])[sensor].mean()
        seq_df['{}_diff_mean'.format(sensor)] = df.groupby(['sequence','subject'])[s_diff].mean()
        seq_df['{}_med'.format(sensor)] = df.groupby(['sequence','subject'])[sensor].median()
        seq_df['{}_std'.format(sensor)] = df.groupby(['sequence','subject'])[sensor].std()
        seq_df['{}_skew'.format(sensor)] = df.groupby(['sequence','subject'])[sensor].skew()
        seq_df['{}_kurt'.format(sensor)] = df.groupby(['sequence','subject'])[sensor].apply(pd.DataFrame.kurt)
        seq_df['{}_min'.format(sensor)] = df.groupby(['sequence','subject'])[sensor].min()
        seq_df['{}_max'.format(sensor)] = df.groupby(['sequence','subject'])[sensor].max()
    
    
    bucketized_00 = pd.qcut(df['sensor_00'], q=10, labels=list(range(10)))
    df['bucketized_00'] = bucketized_00
    df['{}_groupby_sensor_00'.format('sensor_02')] = df.merge(df.groupby('bucketized_00')['sensor_02'].mean(), 
                                                                 left_on='bucketized_00', right_index=True, how='left').iloc[:, -1]
    df['{}_groupby_sensor_00_diff'.format('sensor_02')] = df['sensor_02'] - df['{}_groupby_sensor_00'.format('sensor_02')]
    df.drop('bucketized_00', axis=1, inplace=True)
    
    
#     bucketized_09 = pd.qcut(df['sensor_09'], q=10, labels=list(range(10)))
#     df['bucketized_09'] = bucketized_09
#     df['{}_groupby_sensor_09'.format('sensor_02')] = df.merge(df.groupby('bucketized_09')['sensor_02'].mean(), 
#                                                                  left_on='bucketized_09', right_index=True, how='left').iloc[:, -1]
#     df['{}_groupby_sensor_09_diff'.format('sensor_02')] = df['sensor_02'] - df['{}_groupby_sensor_09'.format('sensor_02')]
#     df.drop('bucketized_09', axis=1, inplace=True)
    
    return df, seq_df.reset_index()

warnings.filterwarnings('ignore')
train, train_seq_df =feat_eng(df=train)
test, test_seq_df =feat_eng(df=test)


# In[ ]:


# train_seq_df.shape, test_seq_df.shape


# In[ ]:


# train_seq_df = train_seq_df.values
# test_seq_df = test_seq_df.values


# In[ ]:


train = train.iloc[:, 3:]
test = test.iloc[:, 3:]

len_train = len(train)
concat = pd.concat([train, test], axis=0)
del train, test
# time.sleep(10)

scaler = StandardScaler()
concat_scaled = scaler.fit_transform(concat)
del concat 
time.sleep(10)

# train_scaled = scaler.fit_transform(train)
# del train
# time.sleep(10)

# test_scaled = scaler.transform(test)
# del test
time.sleep(10)

train_scaled = concat_scaled[:len_train, :]
test_scaled = concat_scaled[len_train:, :]

train_scaled = train_scaled.reshape(-1, 60, train_scaled.shape[-1])
test_scaled = test_scaled.reshape(-1, 60, train_scaled.shape[-1])
train_scaled.shape, test_scaled.shape


# In[ ]:


# def transformer_encoder(num_blocks=12, linear_shape=64, num_heads=8, dropout_rate=0.0):
#     inputs = layers.Input(shape=(train_scaled.shape[-2:]))
#     random_mask = layers.Input(shape=(num_heads, 60, 60))
#     indexes = tf.range(inputs.shape[-2])
#     pos_encoding = layers.Embedding(input_dim=60, output_dim=inputs.shape[-1], trainable=True)(indexes)
#     encoded_inputs = inputs + pos_encoding
    
#     encoder = trasformer_block(encoded_inputs, linear_shape=linear_shape, num_heads=num_heads, attention_mask=random_mask)
#     encoder = 0.3*encoded_inputs + 0.7*encoder
#     encoder = layers.Dropout(dropout_rate)(encoder)
#     if num_blocks > 1:
#         for i in range(1, num_blocks):
#             x = encoder
#             encoder = trasformer_block(x, linear_shape=linear_shape, num_heads=num_heads, attention_mask=random_mask)
#             encoder = 0.3*x + 0.7*encoder
#             encoder = layers.Dropout(dropout_rate)(encoder)
            
#     pooled = layers.GlobalAveragePooling1D()(encoder)
#     dropout = layers.Dropout(0.0)(pooled)
#     output = layers.Dense(1, activation='sigmoid')(dropout)
#     model = tf.keras.Model(inputs=[inputs, random_mask], outputs=output)
#     metric1 = tf.keras.metrics.AUC(name='auc')
#     metric2 = tf.keras.metrics.BinaryAccuracy()
#     model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
#                   optimizer=tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=0.0), #tf.keras.optimizers.Adam(learning_rate=5e-4),
#                   metrics=[metric1, metric2])
#     return model


# In[ ]:


import math

LR_START = 1e-4
LR_MAX = 1e-3
LR_MIN = 5e-5
LR_RAMPUP_EPOCHS = 0
LR_SUSTAIN_EPOCHS = 0
EPOCHS = 50
STEPS = [50]


def lrfn(epoch):
    if epoch<STEPS[0]:
        epoch2 = epoch
        EPOCHS2 = STEPS[0]
    elif epoch<STEPS[0]+STEPS[1]:
        epoch2 = epoch-STEPS[0]
        EPOCHS2 = STEPS[1]
    elif epoch<STEPS[0]+STEPS[1]+STEPS[2]:
        epoch2 = epoch-STEPS[0]-STEPS[1]
        EPOCHS2 = STEPS[2]
    
    if epoch2 < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch2 + LR_START
    elif epoch2 < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        decay_total_epochs = EPOCHS2 - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS - 1
        decay_epoch_index = epoch2 - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        phase = math.pi * decay_epoch_index / decay_total_epochs
        cosine_decay = 0.5 * (1 + math.cos(phase))
        lr = (LR_MAX - LR_MIN) * cosine_decay + LR_MIN
    return lr

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)


# In[ ]:


# def model():
#     x_input = layers.Input(shape=(train_scaled.shape[-2:]))
#     x1 = layers.Bidirectional(layers.LSTM(units=512, return_sequences=True))(x_input)

#     l1 = layers.Bidirectional(layers.LSTM(units=384, return_sequences=True))(x1)
#     l2 = layers.Bidirectional(layers.LSTM(units=384, return_sequences=True))(x_input)

#     c1 = layers.Concatenate(axis=2)([l1,l2])

#     l3 = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(c1)
#     l4 = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(l2)

#     c2 = layers.Concatenate(axis=2)([l3,l4])

#     l6 = layers.GlobalMaxPooling1D()(c2)
#     l7 = layers.Dense(units=128, activation='selu')(l6)
#     l8 = layers.Dropout(0.05)(l7)

#     output = layers.Dense(1, activation='sigmoid')(l8)
    
#     model = tf.keras.Model(inputs=x_input, outputs=output)
#     model.compile(optimizer='adam', 
#                   loss='binary_crossentropy', 
#                   metrics=[tf.keras.metrics.AUC(name = 'auc')])
#     return model


# In[ ]:


# model = transformer_encoder(num_blocks=1, linear_shape=64, num_heads=4)
# model.summary()


# In[ ]:


# with strategy.scope():
#     loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
#     train_metric = tf.keras.metrics.AUC(name='auc')
#     valid_metric = tf.keras.metrics.AUC(name='auc')
#     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# @tf.function
# def train_step(x, y):
#     with tf.GradientTape() as tape:
#         preds = model(x, training=True)
#         loss_value = loss_object(y, preds)
#     grads = tape.gradient(loss_value, model.trainable_weights)
#     optimizer.apply_gradients((zip(grads, model.trainable_weights)))
#     train_metric.update_state(y, preds)
#     return loss_value

# @tf.function
# def valid_step(x, y):
#     preds = model(x, training=True)
#     loss_value = loss_object(y, preds)
#     valid_metric.update_state(y, preds)
#     return loss_value


# n_splits = 3
# skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1443)
# for fold, (train_idx, valid_idx) in enumerate(skf.split(train_scaled, train_labels.iloc[:, -1].values)):
#     print('*'*30, f'Fold {fold+1}', '*'*30)
#     x_train, y_train = train_scaled[train_idx], train_labels.iloc[train_idx, -1].values
#     x_valid, y_valid = train_scaled[valid_idx], train_labels.iloc[valid_idx, -1].values
    
#     train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#     train_ds = train_ds.shuffle(1024)
#     train_ds = train_ds.batch(128).prefetch(-1)
    
#     valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
#     valid_ds = valid_ds.batch(128).prefetch(-1)
    
#     with strategy.scope():
#         model = transformer_encoder(num_blocks=12, linear_shape=64, num_heads=8, dropout_rate=0.2)
    
#     for epoch in range(100):
#         print(f'Epoch --------> {epoch+1}')
#         train_loss_list = []
#         epoch_random_mask = tf.random.uniform(shape=(1, 60, 60)) + 0.2
#         epoch_random_mask = tf.broadcast_to(epoch_random_mask, shape=(8, 60, 60))
#         epoch_random_mask = tf.where(epoch_random_mask<0.5, 0.0, 1.0)
#         for x, y in tqdm(train_ds, total=len(train_ds)):
#             random_mask = tf.broadcast_to(epoch_random_mask, shape=(x.shape[0], 8, 60, 60))
#             inputs = (x, random_mask)
#             train_loss_list.append(train_step(inputs, y))
#         print('Train', 'Loss:', np.mean(train_loss_list), 'AUC:', train_metric.result().numpy())
#         train_metric.reset_state()
        
#         valid_loss_list = []
#         for x, y in tqdm(valid_ds, total=len(valid_ds)):
#             ones_mask = tf.ones(shape=(8, 60, 60))
#             ones_mask = tf.broadcast_to(ones_mask, shape=(x.shape[0], 8, 60, 60))
#             inputs = (x, ones_mask)
#             valid_loss_list.append(valid_step(inputs, y))
#         print('Valid', 'Loss:', np.mean(valid_loss_list), 'AUC:', valid_metric.result().numpy())
#         valid_metric.reset_state()


# In[ ]:


# def trasformer_block(inputs, linear_shape=512, num_heads=8, dropout_rate=0.0):
#     x = layers.MultiHeadAttention(num_heads=num_heads,
#                                   key_dim=linear_shape,
#                                   value_dim=linear_shape,
#                                   dropout=dropout_rate)(inputs, inputs)
#     x = layers.Add()([inputs, x])
#     x1 = layers.LayerNormalization()(x)
    
#     x = layers.Dense(linear_shape, activation='gelu')(x1)
#     x = layers.Dense(inputs.shape[-1])(x)
    
#     x = layers.Add()([x1, x])
#     x2 = layers.LayerNormalization()(x)
#     return x2


# In[ ]:


# def transformer_encoder(num_blocks=12, linear_shape=64, num_heads=8, dropout_rate=0.0):
#     inputs = layers.Input(shape=(train_scaled.shape[-2:]))
#     indexes = tf.range(inputs.shape[-2])
#     pos_encoding = layers.Embedding(input_dim=60, output_dim=inputs.shape[-1], trainable=True)(indexes)
#     encoded_inputs = inputs + pos_encoding
    
#     encoder = trasformer_block(encoded_inputs, linear_shape=linear_shape, num_heads=num_heads, dropout_rate=dropout_rate)
# #     encoder = 0.3*encoded_inputs + 0.7*encoder
#     encoder = layers.BatchNormalization()(encoder)
#     encoder = layers.Dropout(dropout_rate)(encoder)
#     if num_blocks > 1:
#         for i in range(1, num_blocks):
#             x = encoder
#             encoder = trasformer_block(x, linear_shape=linear_shape, num_heads=num_heads, dropout_rate=dropout_rate)
# #             encoder = 0.3*x + 0.7*encoder
#             encoder = layers.BatchNormalization()(encoder)
#             encoder = layers.Dropout(dropout_rate)(encoder)
            
#     pooled = layers.GlobalAveragePooling1D()(encoder)
#     dropout = layers.Dropout(0.5)(pooled)
#     output = layers.Dense(1, activation='sigmoid')(dropout)
#     model = tf.keras.Model(inputs=inputs, outputs=output)
#     metric1 = tf.keras.metrics.AUC(name='auc')
#     metric2 = tf.keras.metrics.BinaryAccuracy()
#     model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
#                   optimizer=tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=5e-4), #tf.keras.optimizers.Adam(learning_rate=5e-4),
#                   metrics=[metric1, metric2])
#     return model


# In[ ]:


# def transformer_encoder(num_blocks=12, linear_shape=64, num_heads=8, dropout_rate=0.0):
#     inputs = layers.Input(shape=(train_scaled.shape[-2:]))
# #     indexes = tf.range(inputs.shape[-2])
# #     pos_encoding = layers.Embedding(input_dim=60, output_dim=inputs.shape[-1], trainable=True)(indexes)
# #     encoded_inputs = inputs + pos_encoding
    
#     lstm = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(inputs)
#     lstm = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(lstm)
    
#     encoder = trasformer_block(lstm, linear_shape=linear_shape, num_heads=num_heads, dropout_rate=dropout_rate)
# #     encoder = lstm + encoder
#     encoder = layers.BatchNormalization()(encoder)
#     encoder = layers.Dropout(dropout_rate)(encoder)
#     if num_blocks > 1:
#         for i in range(1, num_blocks):
#             x = encoder
#             encoder = trasformer_block(x, linear_shape=linear_shape, num_heads=num_heads, dropout_rate=dropout_rate)
# #             encoder = x + encoder
#             encoder = layers.BatchNormalization()(encoder)
#             encoder = layers.Dropout(dropout_rate)(encoder)
        
#     pooled = layers.GlobalAveragePooling1D()(encoder)
#     dropout = layers.Dropout(0.5)(pooled)
#     output = layers.Dense(1, activation='sigmoid')(dropout)
#     model = tf.keras.Model(inputs=inputs, outputs=output)
#     metric1 = tf.keras.metrics.AUC(name='auc')
#     metric2 = tf.keras.metrics.BinaryAccuracy()
#     model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
#                   optimizer=tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=5e-4), #tf.keras.optimizers.Adam(learning_rate=5e-4),
#                   metrics=[metric1, metric2])
#     return model


# In[ ]:


# model = transformer_encoder(num_blocks=2, linear_shape=64, num_heads=4, dropout_rate=0.0)
# model.summary()


# In[ ]:


# n_splits = 3
# skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1443)
# for fold, (train_idx, valid_idx) in enumerate(skf.split(train_scaled, train_labels.iloc[:, -1].values)):
#     print('*'*30, f'Fold {fold+1}', '*'*30)
#     x_train, y_train = train_scaled[train_idx], train_labels.iloc[train_idx, -1].values
#     x_valid, y_valid = train_scaled[valid_idx], train_labels.iloc[valid_idx, -1].values
    
#     file_path = f'Fold_{fold+1}_weights.h5'
#     ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=file_path,
#                                               monitor='val_auc',
#                                               mode='max',
#                                               save_best_only=True,
#                                               save_weights_only=True,
#                                              verbose=1)
    
#     lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.4,  patience=10, verbose=True)
#     with strategy.scope():
#         model = transformer_encoder(num_blocks=2, linear_shape=32, num_heads=8, dropout_rate=0.0)
#     model.fit(x_train, y_train, 
#               validation_data=(x_valid, y_valid),
#               epochs=120, batch_size=256, callbacks=[ckpt])
#     break


# In[ ]:


# n_splits = 5
# skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1443)
# for fold, (train_idx, valid_idx) in enumerate(skf.split(train_scaled, train_labels.iloc[:, -1].values)):
#     print('*'*30, f'Fold {fold+1}', '*'*30)
#     x_train, y_train = train_scaled[train_idx], train_labels.iloc[train_idx, -1].values
#     x_valid, y_valid = train_scaled[valid_idx], train_labels.iloc[valid_idx, -1].values
    
#     file_path = f'Fold_{fold+1}_weights.h5'
#     ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=file_path,
#                                               monitor='val_auc',
#                                               mode='max',
#                                               save_best_only=True,
#                                               save_weights_only=True,
#                                              verbose=1)
    
#     lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.4,  patience=4, verbose=True, mode='max')
#     with strategy.scope():
#         model = transformer_encoder(num_blocks=4, linear_shape=64, num_heads=4, dropout_rate=0.0)
#     model.fit(x_train, y_train, 
#               validation_data=(x_valid, y_valid),
#               epochs=20, batch_size=256, callbacks=[ckpt, lr])


# In[ ]:


# def dense_block(inputs):
#     x = layers.Dense(256, activation='gelu')(inputs)
#     x = layers.Dense(256, activation='gelu')(x)
#     x = layers.Dense(128, activation='gelu')(x)
#     x = layers.Dense(128, activation='gelu')(x)
#     x = layers.Dense(64, activation='gelu')(x)
#     return x


# In[ ]:


# from tensorflow.keras.layers import Input, Bidirectional, LSTM, Concatenate, GlobalMaxPooling1D, Dense, Dropout

# def lstm_model():
#     inputs = Input(shape=(train_scaled.shape[-2:]))
#     dense = Dense(128, activation='gelu')(inputs)
#     dense = Dense(128, activation='gelu')(dense)
#     dropout = Dropout(0.35)(dense)
#     dense = Dense(128, activation='linear')(dropout)
# #     dense = layers.Add()([x_input, dense])
#     x1 = Bidirectional(LSTM(units=512, return_sequences=True))(dense)

#     l1 = Bidirectional(LSTM(units=384, return_sequences=True))(x1)
#     l2 = Bidirectional(LSTM(units=384, return_sequences=True))(inputs)

#     c1 = Concatenate(axis=2)([l1,l2])

#     l3 = Bidirectional(LSTM(units=256, return_sequences=True))(c1)
#     l4 = Bidirectional(LSTM(units=256, return_sequences=True))(l2)

#     c2 = Concatenate(axis=2)([l3,l4])

#     l6 = GlobalMaxPooling1D()(c2)
    
#     l7 = Dense(units=128, activation='gelu')(l6)
#     l8 = Dropout(0.0)(l7)

#     output = Dense(1, activation='sigmoid')(l8)
    
    
#     model = tf.keras.Model(inputs=inputs, outputs=output)
#     metric1 = tf.keras.metrics.AUC(name='auc')
#     metric2 = tf.keras.metrics.BinaryAccuracy()
#     model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
#                   optimizer=tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-3), #tf.keras.optimizers.Adam(learning_rate=5e-4),
#                   metrics=[metric1, metric2])
#     return model


# In[ ]:


def cnn_block(inputs, filters=128, dropout=0.4):
    x = layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    return x


# In[ ]:


class GeMPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, p=1., train_p=False):
        super().__init__()
        if train_p:
            self.p = tf.Variable(p, dtype=tf.float32)
        else:
            self.p = p
        self.eps = 1e-6

    def call(self, inputs: tf.Tensor, **kwargs):
        inputs = tf.clip_by_value(inputs, clip_value_min=1e-6, clip_value_max=tf.reduce_max(inputs))
        inputs = tf.pow(inputs, self.p)
        inputs = tf.reduce_mean(inputs, axis=[1], keepdims=False)
        inputs = tf.pow(inputs, 1./self.p)
        return inputs


# In[ ]:


from tensorflow.keras.layers import Input, Bidirectional, LSTM, Concatenate, GlobalMaxPooling1D, Dense, Dropout, GRU

def lstm_model():
    inputs = Input(shape=(train_scaled.shape[-2:]))
#     dense = Dense(128, activation='gelu')(inputs)
#     dense = Dense(128, activation='gelu')(dense)
#     dropout = Dropout(0.35)(dense)
#     dense = Dense(128, activation='linear')(dropout)
    processed = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(inputs)
    cnn = cnn_block(processed, filters=128, dropout=0.40)
    cnn = cnn_block(cnn, filters=64, dropout=0.35)
    cnn = cnn_block(cnn, filters=32, dropout=0.20)
    cnn = cnn_block(cnn, filters=1, dropout=0.0)
    cnn = layers.MaxPooling2D()(cnn)
    cnn = layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(cnn)
    
    x1 = Bidirectional(GRU(units=512, return_sequences=True))(cnn)

    l1 = Bidirectional(GRU(units=384, return_sequences=True))(x1)
    l2 = Bidirectional(GRU(units=384, return_sequences=True))(cnn)

    c1 = Concatenate(axis=2)([l1,l2])

    l3 = Bidirectional(GRU(units=256, return_sequences=True))(c1)
    l4 = Bidirectional(GRU(units=256, return_sequences=True))(l2)

    c2 = Concatenate(axis=2)([l3,l4])
#     c2 = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(c2)
#     c2 = layers.Conv2D(filters=1, kernel_size=3, strides=1, activation='gelu', padding='valid')(c2)
#     c2 = layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(c2)
    l6 = GlobalMaxPooling1D()(c2)
#     l6 = GeMPoolingLayer(p=1., train_p=True)(c2)
    
    l7 = Dense(units=128, activation='gelu')(l6)
    l8 = Dropout(0.0)(l7)

    output = Dense(1, activation='sigmoid')(l8)
    
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    metric1 = tf.keras.metrics.AUC(name='auc')
    metric2 = tf.keras.metrics.BinaryAccuracy()
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  optimizer=tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-3), #tf.keras.optimizers.Adam(learning_rate=5e-4),
                  metrics=[metric1, metric2])
    return model


# In[ ]:


# from tensorflow.keras.layers import Input, Bidirectional, LSTM, Concatenate, GlobalMaxPooling1D, Dense, Dropout, GRU

# def lstm_model():
#     inputs = Input(shape=(train_scaled.shape[-2:]))
# #     dense = Dense(128, activation='gelu')(inputs)
# #     dense = Dense(128, activation='gelu')(dense)
# #     dropout = Dropout(0.35)(dense)
# #     dense = Dense(128, activation='linear')(dropout)
#     processed = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(inputs)
#     cnn = cnn_block(processed, filters=128, dropout=0.40)
#     cnn = cnn_block(cnn, filters=64, dropout=0.35)
#     cnn = cnn_block(cnn, filters=32, dropout=0.20)
#     cnn = cnn_block(cnn, filters=1, dropout=0.0)
#     cnn = layers.MaxPooling2D()(cnn)
#     cnn = layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(cnn)
    
#     x1 = Bidirectional(LSTM(units=512, return_sequences=True))(cnn)
    
#     l1 = Bidirectional(LSTM(units=384, return_sequences=True))(x1)
#     l2 = Bidirectional(LSTM(units=384, return_sequences=True))(cnn)

#     c1 = Concatenate(axis=2)([l1,l2])

#     l3 = layers.MultiHeadAttention(num_heads=4,
#                                   key_dim=64,
#                                   value_dim=64,
#                                   dropout=0.0)(c1, c1)
    
#     l3 = layers.MultiHeadAttention(num_heads=4,
#                                   key_dim=64,
#                                   value_dim=64,
#                                   dropout=0.0)(l3, l3)
    
#     l4 = Bidirectional(LSTM(units=256, return_sequences=True))(l3)

# #     c2 = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(c2)
# #     c2 = layers.Conv2D(filters=1, kernel_size=3, strides=1, activation='gelu', padding='valid')(c2)
# #     c2 = layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(c2)
#     l6 = GeMPoolingLayer(p=1., train_p=True)(l4)
    
#     l7 = Dense(units=128, activation='gelu')(l6)
#     l8 = Dropout(0.0)(l7)

#     output = Dense(1, activation='sigmoid')(l8)
    
    
#     model = tf.keras.Model(inputs=inputs, outputs=output)
#     metric1 = tf.keras.metrics.AUC(name='auc')
#     metric2 = tf.keras.metrics.BinaryAccuracy()
#     model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
#                   optimizer=tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-3), #tf.keras.optimizers.Adam(learning_rate=5e-4),
#                   metrics=[metric1, metric2])
#     return model


# In[ ]:


# from tensorflow.keras.layers import Input, Bidirectional, LSTM, Concatenate, GlobalMaxPooling1D, Dense, Dropout, GRU

# def lstm_model():
#     inputs = Input(shape=(train_scaled.shape[-2:]))
# #     dense = Dense(128, activation='gelu')(inputs)
# #     dense = Dense(128, activation='gelu')(dense)
# #     dropout = Dropout(0.35)(dense)
# #     dense = Dense(128, activation='linear')(dropout)
#     processed = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(inputs)
#     cnn = cnn_block(processed, filters=128, dropout=0.20)
#     cnn = cnn_block(cnn, filters=64, dropout=0.10)
#     cnn = cnn_block(cnn, filters=32, dropout=0.05)
#     cnn = cnn_block(cnn, filters=1, dropout=0.0)
#     cnn = layers.MaxPooling2D()(cnn)
#     cnn = layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(cnn) 
    
#     x1 = Bidirectional(LSTM(units=512, return_sequences=True))(cnn)

#     l1 = Bidirectional(LSTM(units=384, return_sequences=True))(x1)
#     l2 = Bidirectional(LSTM(units=384, return_sequences=True))(cnn)

#     c1 = Concatenate(axis=2)([l1,l2])
    
#     processed = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(c1)
#     cnn = cnn_block(processed, filters=128, dropout=0.30)
#     cnn = cnn_block(cnn, filters=128, dropout=0.25)
#     cnn = cnn_block(cnn, filters=64, dropout=0.10)
#     cnn = cnn_block(cnn, filters=64, dropout=0.10)
#     cnn = layers.MaxPooling2D()(cnn)

#     l6 = layers.GlobalMaxPooling2D()(cnn)
    
#     l7 = Dense(units=128, activation='gelu')(l6)
#     l8 = Dropout(0.0)(l7)

#     output = Dense(1, activation='sigmoid')(l8)
    
    
#     model = tf.keras.Model(inputs=inputs, outputs=output)
#     metric1 = tf.keras.metrics.AUC(name='auc')
#     metric2 = tf.keras.metrics.BinaryAccuracy()
#     model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
#                   optimizer=tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-3), #tf.keras.optimizers.Adam(learning_rate=5e-4),
#                   metrics=[metric1, metric2])
#     return model


# In[ ]:


tf.keras.backend.clear_session()
model = lstm_model()
model.summary()


# In[ ]:


# from tensorflow.keras.layers import Input, Bidirectional, LSTM, Concatenate, GlobalMaxPooling1D, Dense, Dropout

# def lstm_model():
#     inputs_1 = Input(shape=(train_scaled.shape[-2:]))
#     inputs_2 = Input(shape=(train_seq_df.shape[-1],))
#     dense = Dense(128, activation='gelu')(inputs_1)
#     dense = Dense(128, activation='gelu')(dense)
#     dropout = Dropout(0.35)(dense)
#     dense = Dense(128, activation='linear')(dropout)
# #     dense = layers.Add()([x_input, dense])
#     x1 = Bidirectional(LSTM(units=512, return_sequences=True))(dense)

#     l1 = Bidirectional(LSTM(units=384, return_sequences=True))(x1)
#     l2 = Bidirectional(LSTM(units=384, return_sequences=True))(inputs_1)

#     c1 = Concatenate(axis=2)([l1,l2])

#     l3 = Bidirectional(LSTM(units=256, return_sequences=True))(c1)
#     l4 = Bidirectional(LSTM(units=256, return_sequences=True))(l2)

#     c2 = Concatenate(axis=2)([l3,l4])

#     l6 = GlobalMaxPooling1D()(c2)
    
#     dense_outputs = dense_block(inputs_2)
#     c3 = Concatenate()([l6, dense_outputs])
    
#     l7 = Dense(units=128, activation='gelu')(c3)
#     l8 = Dropout(0.0)(l7)

#     output = Dense(1, activation='sigmoid')(l8)
    
    
#     model = tf.keras.Model(inputs=[inputs_1, inputs_2], outputs=output)
#     metric1 = tf.keras.metrics.AUC(name='auc')
#     metric2 = tf.keras.metrics.BinaryAccuracy()
#     model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
#                   optimizer=tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-3), #tf.keras.optimizers.Adam(learning_rate=5e-4),
#                   metrics=[metric1, metric2])
#     return model


# In[ ]:


# model = lstm_model()
# model.summary()


# In[ ]:


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1443)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_scaled, train_labels.iloc[:, -1].values)):
    for rep in range(3):
        print('*'*30, f'Fold {fold+1} Trial {rep+1}', '*'*30)
        x_train, y_train = train_scaled[train_idx], train_labels.iloc[train_idx, -1].values
        x_valid, y_valid = train_scaled[valid_idx], train_labels.iloc[valid_idx, -1].values

        file_path = f'Fold_{fold+1}_{rep+1}_weights.h5'
        ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                  monitor='val_auc',
                                                  mode='max',
                                                  save_best_only=True,
                                                  save_weights_only=True,
                                                 verbose=1)

        tf.keras.backend.clear_session()
        lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5,  patience=4, verbose=True, mode='max')
        with strategy.scope():
            model = lstm_model()
        model.fit(x_train, y_train, 
                  validation_data=(x_valid, y_valid),
                  epochs=40, batch_size=32*8, callbacks=[ckpt, lr])


# In[ ]:


best = [1, 2, 2, 2, 3]


# In[ ]:


# n_splits = 5
# skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1443)
# for fold, (train_idx, valid_idx) in enumerate(skf.split(train_scaled, train_labels.iloc[:, -1].values)):
#     print('*'*30, f'Fold {fold+1}', '*'*30)
#     x_train_1, x_train_2, y_train = train_scaled[train_idx], train_seq_df[train_idx], train_labels.iloc[train_idx, -1].values
#     x_valid_1, x_valid_2, y_valid = train_scaled[valid_idx], train_seq_df[valid_idx], train_labels.iloc[valid_idx, -1].values
    
#     file_path = f'Fold_{fold+1}_weights.h5'
#     ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=file_path,
#                                               monitor='val_auc',
#                                               mode='max',
#                                               save_best_only=True,
#                                               save_weights_only=True,
#                                              verbose=1)
    
#     lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.4,  patience=4, verbose=True, mode='max')
#     with strategy.scope():
#         model = lstm_model()
#     model.fit((x_train_1, x_train_2), y_train, 
#               validation_data=((x_valid_1, x_valid_2), y_valid),
#               epochs=20, batch_size=256, callbacks=[ckpt, lr])


# In[ ]:


valid_preds_array = np.zeros(len(train_scaled))
test_preds_array = np.zeros(len(test_scaled))

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1443)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_scaled, train_labels.iloc[:, -1].values)):
    print('*'*30, f'Fold {fold+1}', '*'*30)
    x_train, y_train = train_scaled[train_idx], train_labels.iloc[train_idx, -1].values
    x_valid, y_valid = train_scaled[valid_idx], train_labels.iloc[valid_idx, -1].values
    
    tf.keras.backend.clear_session()
    with strategy.scope():
        model = lstm_model()
        print('Weights:', f'./Fold_{fold+1}_{best[fold]}_weights.h5')
        model.load_weights(f'./Fold_{fold+1}_{best[fold]}_weights.h5')
    valid_preds = model.predict(x_valid, batch_size=256, verbose=True)
    model.evaluate(x=x_valid, y=y_valid, batch_size=256, verbose=True)
    valid_preds_array[valid_idx] = np.squeeze(valid_preds)

    test_preds = model.predict(test_scaled, batch_size=256, verbose=True)
    test_preds_array += np.squeeze(test_preds) / n_splits


# In[ ]:


np.mean([0.9851, 0.9861, 0.9874, 0.9873, 0.9878])


# In[ ]:


pd.DataFrame({'preds': valid_preds_array}).to_csv('valid_preds_array_9867.csv', index=False)
pd.DataFrame({'preds': test_preds_array}).to_csv('test_preds_array_9867.csv', index=False)
# # !rm ./Fold*


# In[ ]:


# 256 ----> 0.9591
# 512 ----> 0.9612
# 512, 4, 64, 4, 0.0 ----> 0.9637
# 0.9689


# In[ ]:


# model.load_weights('./Fold_1_weights.h5')
# preds = model.predict(test_scaled, batch_size=256, verbose=1)


# In[ ]:


# sample.iloc[:, -1] = preds.reshape(-1,)
# sample.to_csv('submission.csv', index=False)


# In[ ]:




