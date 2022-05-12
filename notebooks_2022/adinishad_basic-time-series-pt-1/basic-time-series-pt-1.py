#!/usr/bin/env python
# coding: utf-8

# # Let a time forecasting problem equation is, 
# 
# $T_{t} = 300 + 0.2t + 5\sin (\frac{t}{5}) + 20\cos (\frac{t}{24}) + 100\sin (\frac{t}{120}) + 20R_{t} $
# 
# **This equation has a Trend, random deviation and Three Seasonable Period**

# In[ ]:


import random
import copy
from math import sin, cos
from scipy import interpolate
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn import model_selection
import torch.nn.functional as F
import numpy as np


# In[ ]:


def get_time_series_data(length):
    a = 0.2
    b = 300
    c = 20
    ls = 5
    ms = 20
    gs = 100
    ts = []
    for i in range(length):
        ts.append(b + a * i + ls * sin(i / 5) + ms * cos(i / 24) + gs * sin(i / 120) + c * random.random())
    return ts

if __name__ == "__main__":
    data = get_time_series_data(3000)
    plt.figure(figsize=(10, 8))
    plt.plot(data)
    plt.title("Dataset")
    plt.grid()
    plt.show()


# ## Now we have to preprare a time series dataset as the series of inputs and outputs, and it is done using sliding window technique.

# In[ ]:


def get_time_series_datasets(features, ts_len):
    X = []
    Y = []
    for i in range(features + 1, ts_len):
        ts = get_time_series_data(ts_len)
        X.append(ts[i-(features+1):i-1])
        Y.append([ts[i]])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, Y, test_size=0.3, shuffle=False
    )

    X_val, X_test, y_val, y_test = model_selection.train_test_split(
        X_test, y_test, test_size=0.5, shuffle=False
    )
    
    X_train = torch.tensor(data=X_train)
    X_test = torch.tensor(data=X_test)
    y_train = torch.tensor(data=y_train)
    y_test = torch.tensor(data=y_test)
    X_val = torch.tensor(data=X_val)
    y_val = torch.tensor(data=y_val)

    return X_train, X_val, X_test, y_train, y_val, y_test


# # Create a fully connected neural network

# In[ ]:


class FCNN(torch.nn.Module):
    def __init__(self, n_inp, l_1, l_2, n_out):
        super(FCNN, self).__init__()
        self.lin1 = nn.Linear(n_inp, l_1)
        self.lin2 = nn.Linear(l_1, l_2)
        self.lin3 = nn.Linear(l_2, n_out)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


# ## Better to compare with another model and understand how effective it is.
# 
# ## A straight forward model, which always predict last observer value.

# In[ ]:


class DummyPredictor(nn.Module):
    def forward(self, x):
        last_values = []
        for r in x.tolist():
            last_values.append(r[-1])
        return torch.tensor(data=last_values)


# ## Another one is linear interpolation

# In[ ]:


class InterpolationPredictor(nn.Module):
    def forward(self, x):
        last_values = []
        values = x.tolist()
        for v in values:
            x = np.arange(0, len(v))
            y = interpolate.interp1d(x, v, fill_value='extrapolate')
            last_values.append([y(len(v)).tolist()])
        return torch.tensor(data=last_values)


# ## Classical HWES method

# In[ ]:


class HwesPredictor(nn.Module):
    def forward(self, x):
        last_value = []
        for r in x.tolist():
            model = ExponentialSmoothing(r)
            results = model.fit()
            forecast = results.forecast()
            last_value.append([forecast[0]])
        return torch.tensor(data=last_value)


# In[ ]:


random.seed(1)
torch.manual_seed(1)

# We will use 256 sliding window and 3000 as time series length
features = 256
ts_len = 3000

# Dataset for training, validation and testing
x_train, x_val, x_test, y_train, y_val, y_test = get_time_series_datasets(
    features, ts_len	
)

# initialize prediction models
net = FCNN(n_inp=features, l_1=64, l_2=32, n_out=1)
net.train()

dummy_predictor = DummyPredictor()
interpolation_predictor = InterpolationPredictor()
hwes_predictor = HwesPredictor()

optimizer = torch.optim.Adam(net.parameters())
loss_fn = nn.MSELoss()

# we will choose the model that shown the best results on validation set
best_model = None
min_val_loss = 1000000

# Trainning Process
trainning_loss = []
validation_loss = []

# Start Training
for t in range(10000):
    prediction = net(x_train)
    loss = loss_fn(prediction, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    val_prediction = net(x_val)
    val_loss = loss_fn(val_prediction, y_val)
    trainning_loss.append(loss.item())
    validation_loss.append(val_loss.item())

    if val_loss.item() < min_val_loss:
        best_model = copy.deepcopy(net)
        min_val_loss = val_loss.item()
    if t % 1000 == 0:
        print(f'EPOCH - {t}:             train - {round(loss.item(), 4)}             val - {round(val_loss.item(), 4)}')
net.eval()


# ## Let's visualize results. Our FCNN shows the best result

# In[ ]:


print('Testing')
print(f'FCNN Loss: {loss_fn(best_model(x_test), y_test).item()}')
print(f'Dummy Loss: {loss_fn(dummy_predictor(x_test), y_test).item()}')
print(f'Linear Interpolation Loss:{loss_fn(interpolation_predictor(x_test), y_test).item()}')
print(f"HWES Loss: {loss_fn(hwes_predictor(x_test), y_test).item()}")

plt.figure(figsize=(12, 8))
plt.title("trainning progress")
plt.yscale("log")
plt.plot(trainning_loss, label="trainning loss")
plt.plot(validation_loss, label="validation loss")
plt.grid()
plt.show()

plt.figure(figsize=(12, 8))
plt.title("FCNN on Train Data")
plt.plot(y_test, label="actual")
plt.plot(best_model(x_test).tolist(), label="FCNN")
plt.plot(hwes_predictor(x_test).tolist(), label="HWES")
plt.plot()
plt.grid()
plt.legend()
plt.show()


# In[ ]:




