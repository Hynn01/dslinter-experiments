#!/usr/bin/env python
# coding: utf-8

# # First, we will go with CNN(Convolutional Neural Netoworks) by creating a sample dataset of sequences.

# In[ ]:


from numpy import array
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers.convolutional import Conv1D, MaxPooling1D


# In[ ]:


x = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = array([40,50,60,70])


# Note: Reshape from [samples, timesteps] into [samples, timesteps, features]

# In[ ]:


x = x.reshape((x.shape[0], x.shape[1], 1))


# # Build a CNN model with 1 Convolutional and MaxPooling layer

# In[ ]:


model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=1000, verbose=1)


# Test_Data or Driver code to test your model

# In[ ]:


ip = array([50, 60, 70])
ip =ip.reshape((1, 3, 1))
y_pred = model.predict(ip, verbose=1)
print(y_pred)


# # Let's check for different types of Numerical Sequences!

# Let's try with 5s table now!

# In[ ]:


x = array([[5,10,15], [10,15,20], [20,25,30], [35,40,45]])
y = array([20,25,35,50])


# In[ ]:


x = x.reshape((x.shape[0], x.shape[1], 1))


# In[ ]:


model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=1000, verbose=2)


# In[ ]:


ip = array([75,80,85])
ip =ip.reshape((1, 3, 1))
y_pred = model.predict(ip, verbose=1)
print(y_pred)


# # Let's go into the world of RNNs and LSTMs

# In[ ]:


from keras.layers import LSTM
from sklearn.metrics import mean_squared_error


# In[ ]:


model = Sequential()
model.add(LSTM(4, input_shape=(3,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(x,y,epochs=1000,batch_size=1,verbose=2)


# In[ ]:


ip = array([50,55,60])
ip =ip.reshape((1, 3, 1))
y_pred = model.predict(ip, verbose=1)
print(y_pred)


# CASE_2

# In[ ]:


X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = array([40, 50, 60, 70])
# reshape from [samples, timesteps] into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))
# define model
model = Sequential()
model.add(LSTM(4, input_shape=(3,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(X,y,epochs=1000,batch_size=1,verbose=2)


# In[ ]:


# demonstrate prediction
x_input = array([40,50,60])
x_input = x_input.reshape((1, 3, 1))
yhat = model.predict(x_input, verbose=0)
print(yhat)


# # Recurrent Neural Networks were overpowered by a simple CNN model! So, i guess that for predicting next digits in a set of numbers, it is best to go with CNNs. 

# This is really shocking as i heard and learn that RNNs work well for sequential data and helps to predicT the next value in a sequence. But this is not the case here! Suggest your opinions guys and do UPVOTE!
