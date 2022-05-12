#!/usr/bin/env python
# coding: utf-8

# # <center>RNNs, Encoder-Decoder, 1d CNN</center>
# ## <div align=right>Made by Ihor Markevych</div>

# **Note:** Model selection cycle was not included due to its size. Each model presented is best found one in its subclass.

# -----------------
# -----------------

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Convolution1D, Flatten, Dropout,                                     LSTM, GRU, Input, RepeatVector, SimpleRNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os

import sklearn.metrics
from sklearn.preprocessing import MinMaxScaler
import sklearn.model_selection
from sklearn.linear_model import LinearRegression
np.random.seed(123)


# In[ ]:


data = pd.read_csv('/kaggle/input/uop-aml-hw3/admData20.csv')


# In[ ]:


data


# In[ ]:


plt.plot(data.AdmittedNum)
plt.plot()


# ## Encoder-decoder (taking each admission date separately)

# ### Preprocessing

# In[ ]:


grouped = data.groupby(by='ExpStartDate')

X =  []
X_aux = []
y = []
for g in grouped.groups:
    X.append(grouped.get_group(g).AppliedNum.values)
    X_aux.append(grouped.get_group(g).Budget.iloc[0])
    y.append(grouped.get_group(g).AdmittedNum)


# In[ ]:


X = pd.DataFrame([[i for i in j] for j in X])
y = pd.DataFrame([[i for i in j] for j in y])
X = np.array(X)
y = np.array(y)
X_aux = np.array(X_aux)

X = X / X_aux[:, None]
y = y / X_aux[:, None]

X_train, X_test, X_aux_train, X_aux_test, y_train, y_test = sklearn.model_selection.train_test_split(X, X_aux, y)


# ### Helper functions

# Generate training datasets of sequences of variable length.

# In[ ]:


def train_gen(X, y, foresight=5):
    train = []
    target = []
    for i in range(len(X)):
        X_seq = X[i, ~np.isnan(X[i,:])]
        y_seq = y[i, ~np.isnan(y[i,:])]
        
        X_seq_train = X_seq[:-foresight]
        y_seq_train = y_seq[:-foresight]
        y_seq_test = y_seq[-foresight:]
    
        train.append(np.array([X_seq_train, y_seq_train]))
        target.append(y_seq_test)
    return train, target


# Custom train cycle.  
# **NOTE:** validation is not included here due to sample size - after grouping training is 6 samples, splitting it into additional set looks inappropriate.

# In[ ]:


def train_enc_dec(model, train_gen=train_gen, X_train=X_train, y_train=y_train, epochs=50):
    epochs = epochs
    history = []
#     val_history = []
    X_train_seq, y_train_seq = train_gen(X_train, y_train)
    
#     X_t, X_val, y_t, y_val = sklearn.model_selection.train_test_split(X_train_seq, y_train_seq, test_size=0.1)
    
    for e in range(epochs):
        for i in range(len(X_train_seq)):
            hist = model.fit(np.array([X_train_seq[i].T]), [[y_train_seq[i]]], 
                             epochs=1, 
                             batch_size=1, 
                             validation_split=0,
                             verbose=0)
            history.append(hist)
        
#         val_temp = []
#         for i in range(len(X_val)):
#             val_temp.append(model.evaluate(np.array([X_val[i].T]), [[y_val[i]]], 
#                                            verbose=0))
    return history


# Reporting MSE and MAE.

# In[ ]:


def print_report(y_true, y_pred):
    print(f'MSE: {np.mean((np.array(y_true) - y_pred) ** 2) / len(y_true[0])}')
    print(f'MAE: {np.mean(np.abs(np.array(y_true) - y_pred)) / len(y_true[0])}')


# ### LSTM-based model

# In[ ]:


# https://stackoverflow.com/questions/43117654/many-to-many-sequence-prediction-with-different-sequence-length

def create_LSTM_model():

    seq_input = Input(shape=(None, 2), name='seq_input') # unknown timespan, fixed feature size of 2
    x = LSTM(4, return_sequences=True, activation='relu')(seq_input)
    x = LSTM(4, return_sequences=True, activation='relu')(x)
    x = LSTM(1, return_sequences=False, activation='relu')(x)
    # aux_input = Input(shape=(1), name='aux_input')

    # x = tf.keras.layers.concatenate([aux_input, seq_x])
    # x = Dense(1, activation='relu')(x)
    x = RepeatVector(5)(x)

    out = LSTM(1, return_sequences=True)(x)

    model = Model(seq_input, out)

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mae'])
    return model


# In[ ]:


model = create_LSTM_model()
model.summary()


# In[ ]:


history = train_enc_dec(model)


# In[ ]:


X_test_seq, y_test_seq = train_gen(X_test, y_test)
y_pred = np.array([model.predict(np.array([X_test_seq[i].T])) for i in range(len(X_test))])
y_pred = np.squeeze(y_pred)


# In[ ]:


print_report(y_test_seq, y_pred)


# In[ ]:


y_pred = y_pred * X_aux_test[:, None]
y_test_seq = y_test_seq * X_aux_test[:, None]
print('For unscaled:')
print_report(y_test_seq, y_pred)


# In[ ]:


y_test_unscaled = [ts * aux for ts, aux in zip([y_t[1] for y_t in X_test_seq], X_aux_test)]
for i in range(len(y_test_seq)):
    plt.plot(np.append(y_test_unscaled[i], y_test_seq[i]), label='True')
    plt.plot(np.append(y_test_unscaled[i], y_pred[i]), label='Prediction')
    plt.legend()
    plt.show()


# ### GRU-based model

# In[ ]:


def create_GRU_model():

    seq_input = Input(shape=(None, 2), name='seq_input') # unknown timespan, fixed feature size of 2
    x = GRU(4, return_sequences=True, activation='relu')(seq_input)
    x = GRU(4, return_sequences=True, activation='relu')(x)
    x = GRU(1, return_sequences=False, activation='relu')(x)
    # aux_input = Input(shape=(1), name='aux_input')

    # x = tf.keras.layers.concatenate([aux_input, seq_x])
    # x = Dense(1, activation='relu')(x)
    x = RepeatVector(5)(x)

    out = GRU(1, return_sequences=True)(x)

    model = Model(seq_input, out)

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mae'])
    return model


# In[ ]:


model = create_GRU_model()
model.summary()


# In[ ]:


history = train_enc_dec(model)


# In[ ]:


X_test_seq, y_test_seq = train_gen(X_test, y_test)
y_pred = np.array([model.predict(np.array([X_test_seq[i].T])) for i in range(len(X_test))])
y_pred = np.squeeze(y_pred)


# In[ ]:


print_report(y_test_seq, y_pred)


# In[ ]:


y_pred = y_pred * X_aux_test[:, None]
y_test_seq = y_test_seq * X_aux_test[:, None]
print('For unscaled:')
print_report(y_test_seq, y_pred)


# In[ ]:


y_test_unscaled = [ts * aux for ts, aux in zip([y_t[1] for y_t in X_test_seq], X_aux_test)]
for i in range(len(y_test_seq)):
    plt.plot(np.append(y_test_unscaled[i], y_test_seq[i]), label='True')
    plt.plot(np.append(y_test_unscaled[i], y_pred[i]), label='Prediction')
    plt.legend()
    plt.show()


# ----------------------
# ----------------------
# ----------------------

# ## Taking whole timespan

# **NOTE:** in printed report, in predictions plots red lines correspond to predicted sequences, blue lines to true data.

# ### Helper functions

# In[ ]:


def plot_preds(model):

    plt.figure(figsize=(20, 8))
    plt.subplot(1, 3, 1)
    axes = plt.gca()
    axes.set_ylim([0, 2])

    for j, y in enumerate(y_train):
        row = [None for i in range(j)]
        row += list(y)
        plt.plot(row, 'b', label='true')

    plt.subplot(1, 3, 2)
    axes = plt.gca()
    axes.set_ylim([0, 2])
    y_pred = model.predict(X_val)

    plt.plot(X_val[0,:5,0], 'b')
    
    for j, prediction in enumerate(y_pred):
        row = [None for i in range(j + 5)]
        row += list(prediction)
        plt.plot(row, 'r')


    for j, y in enumerate(y_val):
        row = [None for i in range(j + 5)]
        row += list(y)
        plt.plot(row, 'b')

    plt.legend()

    plt.subplot(1, 3, 3)
    axes = plt.gca()
    axes.set_ylim([0, 2])
    y_pred = model.predict(X_test)

    plt.plot(X_test[0,:5,0], 'b')
    
    for j, prediction in enumerate(y_pred):
        row = [None for i in range(j + 5)]
        row += list(prediction)
        plt.plot(row, 'r')


    for j, y in enumerate(y_test):
        row = [None for i in range(j + 5)]
        row += list(y)
        plt.plot(row, 'b')

    plt.legend()
    plt.plot()


# In[ ]:


OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.001)
EPOCHS = 50
BATCH_SIZE = 64
LOOK_BACK = 20
es2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=400, restore_best_weights=True)


# In[ ]:


def train_and_report(creation_func):
    model = creation_func()
    print(model.summary())
    
    history = model.fit(X_train, y_train,
                        batch_size=128, 
                        epochs=700, 
                        validation_split=0.0,
                        use_multiprocessing=True,
                        verbose=0,
                        validation_data=(X_val, y_val), 
                        callbacks=[es2]
                       )
    
    plt.title('Loss')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    try:
        y_pred_val = model.predict(X_val)[:,:,0]
        y_pred_test = model.predict(X_test)[:,:,0]
    except:
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
    print('For validation set:')
    print_report(y_val, y_pred_val)

    print('For test set:')
    print_report(y_test, y_pred_test)
    
    plot_preds(model)


# In[ ]:


def inverse_transform(y, scaler, column):
    return y * (scaler.data_max_[column] - scaler.data_min_[column]) + scaler.data_min_[column]


# In[ ]:


def create_seq(dataset, look_back=LOOK_BACK, foresight=5):
    X, y = [], []
    for i in range(len(dataset) - look_back - foresight):
        X.append(np.array(dataset[i:(i + look_back)].loc[:, ['AdmittedNum', 'AppliedNum', 'WeeksBeforeStart']]) 
                 / np.array(dataset.Budget[i:(i + look_back)])[:, None])
        y.append(dataset.iloc[(i + look_back) : (i + look_back + foresight), 0] 
                 / np.array(dataset.Budget[(i + look_back) : (i + look_back + foresight)]))
    return np.array(X), np.array(y)


# In[ ]:


data2 = data.loc[:, ['AdmittedNum', 'AppliedNum', 'WeeksBeforeStart', 'Budget']]

trainTr = round(0.6 * len(data))
valTr = round((0.8 * len(data)))
data_train, data_val, data_test = data2[:trainTr], data2[trainTr:valTr], data2[valTr:]

X_train, y_train = create_seq(data_train)
X_val, y_val = create_seq(data_val)
X_test, y_test = create_seq(data_test)

X_train = np.reshape(np.array(X_train), (X_train.shape[0], X_train.shape[1], 3))
X_val = np.reshape(np.array(X_val), (X_val.shape[0], X_val.shape[1], 3))
X_test = np.reshape(np.array(X_test), (X_test.shape[0], X_test.shape[1], 3))


# ---------
# ---------

# ## GRU-based model

# In[ ]:


def create_GRU():
    model = Sequential()
    
    seq_input = Input(shape=(LOOK_BACK, 3), name='seq_input')
    x = GRU(4, return_sequences=False, activation='relu', dropout=0.3)(seq_input)
#     x = GRU(16, return_sequences=True, activation='relu')(x)
#     x = GRU(4, return_sequences=False, activation='relu')(x)

    # aux_input = Input(shape=(1), name='aux_input')
    # x = tf.keras.layers.concatenate([aux_input, seq_x])
    # x = Dense(1, activation='relu')(x)
    
    x = RepeatVector(5)(x)

#     x = GRU(2, return_sequences=True, activation='relu')(x)
#     x = GRU(2, return_sequences=True, activation='relu')(x)
    out = GRU(1, return_sequences=True)(x)

    model = Model(seq_input, out)
    
    model.compile(loss='mean_squared_error', optimizer=OPTIMIZER, metrics=["mae"])
    
    return model


# In[ ]:


train_and_report(create_GRU)


# ---------

# ### LSTM-based model

# In[ ]:


def create_LSTM():
    model = Sequential()
    
    seq_input = Input(shape=(LOOK_BACK, 3), name='seq_input')
    x = LSTM(4, return_sequences=False, activation='relu', dropout=0.3)(seq_input)
#     x = LSTM(16, return_sequences=True, activation='relu')(x)
#     x = LSTM(4, return_sequences=False, activation='relu')(x)

    # aux_input = Input(shape=(1), name='aux_input')
    # x = tf.keras.layers.concatenate([aux_input, seq_x])
    # x = Dense(1, activation='relu')(x)
    
    x = RepeatVector(5)(x)

#     x = LSTM(2, return_sequences=True, activation='relu')(x)
#     x = LSTM(2, return_sequences=True, activation='relu')(x)
    out = LSTM(1, return_sequences=True)(x)

    model = Model(seq_input, out)
    
    model.compile(loss='mean_squared_error', optimizer=OPTIMIZER, metrics=["mae"])
    
    return model


# In[ ]:


train_and_report(create_LSTM)


# -----------
# -----------

# ### 1d CNN

# In[ ]:


def create_CNN():
    model = Sequential()
    
    seq_input = Input(shape=(LOOK_BACK, 3), name='seq_input')
    x = Convolution1D(filters=2, kernel_size=5, input_shape=(2, LOOK_BACK), activation='relu')(seq_input)
#     x = Dropout(0.3)(x)
#     x = Convolution1D(filters=4, kernel_size=3, activation='relu')(x)
#     x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(8, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4, activation='relu')(x)
#     x = Dropout(0.3)(x)
    
    out = Dense(5)(x)

    model = Model(seq_input, out)
    
    model.compile(loss='mean_squared_error', optimizer=OPTIMIZER, metrics=["mae"])
    
    return model


# In[ ]:


train_and_report(create_CNN)


# ### 1d CNN with GRU output

# In[ ]:


def create_CNN_GRU():
    model = Sequential()
    
    seq_input = Input(shape=(LOOK_BACK, 3), name='seq_input')
    x = Convolution1D(filters=2, kernel_size=5, input_shape=(2, LOOK_BACK), activation='relu')(seq_input)
    x = Convolution1D(filters=4, kernel_size=3, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(8, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(4, activation='relu')(x)
    x = Dropout(0.1)(x)

    x = RepeatVector(5)(x)

    out = GRU(1, return_sequences=True)(x)

    model = Model(seq_input, out)
    
    model.compile(loss='mean_squared_error', optimizer=OPTIMIZER, metrics=["mae"])
    
    return model


# In[ ]:


train_and_report(create_CNN_GRU)


# --------

# ## 1d CNN with LSTM output

# In[ ]:


def create_CNN_LSTM():
    model = Sequential()
    
    seq_input = Input(shape=(LOOK_BACK, 3), name='seq_input')
    x = Convolution1D(filters=2, kernel_size=5, input_shape=(2, LOOK_BACK), activation='relu')(seq_input)
    x = Convolution1D(filters=4, kernel_size=3, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(8, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(4, activation='relu')(x)
    x = Dropout(0.1)(x)
    
    x = RepeatVector(5)(x)

    out = LSTM(1, return_sequences=True)(x)

    model = Model(seq_input, out)
    
    model.compile(loss='mean_squared_error', optimizer=OPTIMIZER, metrics=["mae"])
    
    return model


# In[ ]:


train_and_report(create_CNN_LSTM)


# -----------
# -----------

# ### Dense NN

# In[ ]:


def create_DNN():
    model = Sequential()
    
    seq_input = Input(shape=(LOOK_BACK, 3), name='seq_input')
    x = Flatten()(seq_input)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(8, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(4, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(5, activation='relu')(x)

    model = Model(seq_input, out)
    
    model.compile(loss='mean_squared_error', optimizer=OPTIMIZER, metrics=["mae"])
    
    return model


# In[ ]:


train_and_report(create_DNN)


# -----------

# ### DNN with GRU output

# In[ ]:


def create_DNN_with_GRU_out():
    model = Sequential()
    
    seq_input = Input(shape=(LOOK_BACK, 3), name='seq_input')
    x = Flatten()(seq_input)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(8, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(4, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = RepeatVector(5)(x)

    out = GRU(1, return_sequences=True)(x)

    model = Model(seq_input, out)
    
    model.compile(loss='mean_squared_error', optimizer=OPTIMIZER, metrics=["mae"])
    
    return model


# In[ ]:


train_and_report(create_DNN_with_GRU_out)


# ------------

# ### DNN with LSTM output

# In[ ]:


def create_DNN_with_LSTM_out():
    model = Sequential()
    
    seq_input = Input(shape=(LOOK_BACK, 3), name='seq_input')
    x = Flatten()(seq_input)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(8, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(4, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = RepeatVector(5)(x)

    out = LSTM(1, return_sequences=True)(x)

    model = Model(seq_input, out)
    
    model.compile(loss='mean_squared_error', optimizer=OPTIMIZER, metrics=["mae"])
    
    return model


# In[ ]:


train_and_report(create_DNN_with_LSTM_out)


# -----------
# -----------
# -----------

# ## Conclusion

# Overall, all models gave comparable performance.  
# Model that fully LSTM-based model gave best performance, however, as it was said, all models had comparable error.  
# This fact can be explained by cyclic pattern of the data and small data sample.  
#   
# Stacking recurrent layers was explored, it made models worse, which may be explained by sample size.  
#   
# Auxilary input was tried. However, it appeared that best way of incorporating it is to scale each value by Budget - in this case we assume that we now Budget for future (which is usually the case).
# 
# Encoder-decoder model was also explored. This model takes an input of a sequence of variable length and predicts next five weeks (limited by single admission date). This is the most accurate model and the most logical from the use case - after the start of new admission date user should wait for two-three weeks. Then model can start making predictions and adjust them by taking into account newly available data each week.

# In[ ]:




