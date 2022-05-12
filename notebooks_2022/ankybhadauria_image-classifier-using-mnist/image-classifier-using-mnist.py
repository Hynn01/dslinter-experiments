#!/usr/bin/env python
# coding: utf-8

# # Image Classifier using MNIST 
# 
# In this notebook, I implemented a simple neural network with one hidden layer, Activation and Sigmoid functions and trained it on the MNIST dataset. This features both Forward and Reverse Propogation to calibrate all Weights and Biases accordingly.

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')


# In[ ]:


data = np.array(data)
m, n = data.shape
np.random.shuffle(data) 

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape


# In[ ]:


Y_train 


# Our Neural Network will have a simple two-layer architecture. Input layer $a^{[0]}$ will have 784 units corresponding to the 784 pixels in each 28x28 input image. A hidden layer $a^{[1]}$ will have 10 units with ReLU activation, and finally our output layer $a^{[2]}$ will have 10 units corresponding to the ten digit classes with normalized exponential function.
# 
# **Forward propagation**
# 
# $$Z^{[1]} = W^{[1]} X + b^{[1]}$$
# $$A^{[1]} = g_{\text{ReLU}}(Z^{[1]}))$$
# $$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$$
# $$A^{[2]} = g_{\text{softmax}}(Z^{[2]})$$
# 
# **Backward propagation**
# 
# $$dZ^{[2]} = A^{[2]} - Y$$
# $$dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T}$$
# $$dB^{[2]} = \frac{1}{m} \Sigma {dZ^{[2]}}$$
# $$dZ^{[1]} = W^{[2]T} dZ^{[2]} .* g^{[1]\prime} (z^{[1]})$$
# $$dW^{[1]} = \frac{1}{m} dZ^{[1]} A^{[0]T}$$
# $$dB^{[1]} = \frac{1}{m} \Sigma {dZ^{[1]}}$$
# 
# **Parameter updates**
# 
# $$W^{[2]} := W^{[2]} - \alpha dW^{[2]}$$
# $$b^{[2]} := b^{[2]} - \alpha db^{[2]}$$
# $$W^{[1]} := W^{[1]} - \alpha dW^{[1]}$$
# $$b^{[1]} := b^{[1]} - \alpha db^{[1]}$$
# 
# **Vars and shapes**
# 
# Forward prop
# 
# - $A^{[0]} = X$: 784 x m
# - $Z^{[1]} \sim A^{[1]}$: 10 x m
# - $W^{[1]}$: 10 x 784 (as $W^{[1]} A^{[0]} \sim Z^{[1]}$)
# - $B^{[1]}$: 10 x 1
# - $Z^{[2]} \sim A^{[2]}$: 10 x m
# - $W^{[1]}$: 10 x 10 (as $W^{[2]} A^{[1]} \sim Z^{[2]}$)
# - $B^{[2]}$: 10 x 1
# 
# Backprop
# 
# - $dZ^{[2]}$: 10 x m ($~A^{[2]}$)
# - $dW^{[2]}$: 10 x 10
# - $dB^{[2]}$: 10 x 1
# - $dZ^{[1]}$: 10 x m ($~A^{[1]}$)
# - $dW^{[1]}$: 10 x 10
# - $dB^{[1]}$: 10 x 1

# In[ ]:


def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2


# In[ ]:


def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2


# In[ ]:


W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)


# ~85% accuracy on training set.

# In[ ]:


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


# In[ ]:


test_prediction(0, W1, b1, W2, b2)
test_prediction(4800, W1, b1, W2, b2)
test_prediction(40900, W1, b1, W2, b2)
test_prediction(9999, W1, b1, W2, b2)


# Finally, let's check the accuracy

# In[ ]:


dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_dev)


# Our Neural Network is around 85% accurate, which is quite good for one with a single Hidden layer.
