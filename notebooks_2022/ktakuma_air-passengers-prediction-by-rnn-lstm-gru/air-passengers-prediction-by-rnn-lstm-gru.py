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


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import MinMaxScaler
import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import warnings 
warnings.filterwarnings('ignore')


# In[ ]:


# Import and show dataset
data = pd.read_csv("../input/air-passengers-forecast-dataset/AirPassengers.csv")
print("Shape of Data: ", data.shape)
data.head()


# In[ ]:


# Plot data
data.plot()


# In[ ]:


# Information of the data
data.info()


# ## Data Preparation

# In[ ]:


# Create a dataframe with only the number of passengers
df=data.filter(['#Passengers'])
# Convert the dataframe to a numpy array
df=df.values
print(data[:5])


# In[ ]:


# Scale the data to make it applicable for RNN
scaler=MinMaxScaler(feature_range=(0,1))
df_scaled=scaler.fit_transform(df)
df_scaled[:5]


# In[ ]:


# Split data into predictors and outcomes
# predict the number of passengers using by the past 6 months' number of passengers
X=[]
y=[]
sequence=6
for i in range(len(df_scaled) - sequence):
    X.append(df_scaled[i:(i + sequence),0]) 
    y.append(df_scaled[i + sequence,0])
X,y=np.array(X),np.array(y)
print(X)
print(y)


# In[ ]:


# Reshape the predictor 
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
X.shape


# In[ ]:


# Split data into training and test sets 

# Set the size of training and test data
# Use 75% of the data for training
train_size = math.ceil(len(X) * 0.75)
test_size = len(X - train_size)

# Split X and y into training and test sets
X_train = X[:train_size, :]
y_train = y[:train_size]

X_test = X[train_size:len(X),:]
y_test = y[train_size:len(y)]


# In[ ]:


# Show the size of training and test sets
print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test : ", X_test.shape)
print("y_test : ", y_test.shape)


# ## Modeling

# ### Simple RNN

# In[ ]:


# Build Simple RNN model
rnn=Sequential()
rnn.add(SimpleRNN(units=32, return_sequences=True, input_shape=(X_train.shape[1],1)))
rnn.add(SimpleRNN(units=32, return_sequences=True))
rnn.add(SimpleRNN(units=32, return_sequences=True))
rnn.add(SimpleRNN(units=32))
rnn.add(Dense(units=1))
rnn.compile(optimizer='adam', loss='mean_squared_error')
rnn.summary()


# In[ ]:


# Set Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=40)


# In[ ]:


rnn_history = rnn.fit(X_train, y_train,
                      batch_size=16,
                      epochs=1000,
                      validation_split=0.2,
                      callbacks=[early_stop],
                      verbose=1)


# In[ ]:


# Plot the learning history
plt.plot(rnn_history.history['loss'], label='train loss')
plt.plot(rnn_history.history['val_loss'], label='val loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.ylim([0,0.05])
plt.show()


# ### LSTM

# In[ ]:


# Build LSTM model
lstm=Sequential()
lstm.add(LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1],1)))
lstm.add(LSTM(units=32, return_sequences=True))
lstm.add(LSTM(units=32, return_sequences=True))
lstm.add(LSTM(units=32))
lstm.add(Dense(units=1))
lstm.compile(optimizer='adam', loss='mean_squared_error')
lstm.summary()


# In[ ]:


lstm_history = lstm.fit(X_train, y_train,
                        batch_size=16,
                        epochs=1000,
                        validation_split=0.2,
                        callbacks=[early_stop],
                        verbose=1)


# In[ ]:


# Plot the learning history
plt.plot(lstm_history.history['loss'], label='train loss')
plt.plot(lstm_history.history['val_loss'], label='val loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.ylim([0,0.05])
plt.show()


# ### GRU

# In[ ]:


# Build GRU model
gru=Sequential()
gru.add(GRU(units=32, return_sequences=True, input_shape=(X_train.shape[1],1)))
gru.add(GRU(units=32, return_sequences=True))
gru.add(GRU(units=32, return_sequences=True))
gru.add(GRU(units=32))
gru.add(Dense(units=1))
gru.compile(optimizer='adam', loss='mean_squared_error')
gru.summary()


# In[ ]:


gru_history = gru.fit(X_train, y_train,
                      batch_size=16,
                      epochs=1000,
                      validation_split=0.2,
                      callbacks=[early_stop],
                      verbose=1)


# In[ ]:


# Making predictions

# Predict with RNN model
rnn_y_pred=rnn.predict(X_test)
rnn_y_pred=scaler.inverse_transform(rnn_y_pred)
# Predict with LSTM model
lstm_y_pred=lstm.predict(X_test)
lstm_y_pred=scaler.inverse_transform(lstm_y_pred)
# Predict with GRU model
gru_y_pred=gru.predict(X_test)
gru_y_pred=scaler.inverse_transform(gru_y_pred)

# Reverse test data to real number
y_test=y_test.reshape(y_test.shape[0],1)
y_test=scaler.inverse_transform(y_test)


# In[ ]:


# Visualize the results
plt.figure(figsize=(16,8))
plt.plot(y_test, color='blue',label='Real N of Passenger')
plt.plot(rnn_y_pred, color='tomato',label='RNN Prediction')
plt.plot(lstm_y_pred, color='green', label='LSTM Prediction')
plt.plot(gru_y_pred, color='black', label='GRU Prediction')
plt.title('Air Passenger Prediction')
plt.xlabel('Time')
plt.ylabel('N of Passengers')
plt.legend()
plt.show()


# In[ ]:


# Set subplot subttitles
titles = ['RNN', 'LSTM', 'GRU']

# Create a list of prediction models
models = [rnn_y_pred, lstm_y_pred, gru_y_pred]

# Set the plot area
fig, ax = plt.subplots(1, 3, figsize=(16,4), tight_layout=True)  

# Set the title
plt.suptitle('Air Passenger Prediction')

# Create and show subplots
for i in range(0, 3):
    plt.subplot(1, 3,i+1)
    plt.title(titles[i], fontsize=10) 
    plt.plot(y_test, label='N of Passengers')
    plt.plot(models[i], label=titles[i]+' Prediction')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('N of Passengers') 

