#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


with open("/kaggle/input/ml2021-2022-2-nn/train.csv") as f:
    train_data = np.loadtxt(f, delimiter=",", skiprows=1,dtype=int)
print(train_data.shape)
train_label=train_data[:,0]
train_data=train_data[:,1:]
print(train_label.shape)
print(train_data.shape)
print(np.max(train_data))
train_data=train_data/255-0.5
with open("/kaggle/input/ml2021-2022-2-nn/test.csv") as f:
    result_data = np.loadtxt(f, delimiter=",", skiprows=1,dtype=int)
result_data=result_data/255-0.5


# In[ ]:


size = [784, 50, 30, 10]
weight = []
layer = []
train_size = int(0.8 * len(train_data))
ep = 500000
learning_rate = 0.1


def sigmoid(v):
    return 1 / (1 + pow(math.e, -v))


def de_sigmoid(v):
    return v * (1 - v)


for i in range(len(size) - 1):
    weight.append(np.random.rand(size[i], size[i + 1]) - 0.5)


def forward(v):
    v = np.atleast_2d(v)
    layer.clear()
    layer.append(v)
    for i in weight:
        v = sigmoid(v.dot(i))
        layer.append(v)
    return v


def backward(v):
    for i in range(len(weight) - 1, -1, -1):
        # print(v)
        v *= de_sigmoid(layer[i + 1])
        dw = layer[i].T.dot(v)
        v = v.dot(weight[i].T)
        weight[i] -= learning_rate * dw


def predict(v):
    return np.argmax(forward(v), axis=1)


def train():
    for i in range(ep):
        index = np.random.randint(train_size)
        x = train_data[index]
        y = forward(x)
        # print(y)
        loss = y - np.eye(10)[train_label[index]]
        backward(loss)
        if (i % 1000 == 0):
            print(i, test())


def test():
    x = train_data[train_size:]
    y = forward(x)
    y = np.argmax(y, axis=1)
    y = (y == train_label[train_size:])

    return sum(y) / len(y)


train()
test()



# In[ ]:


result=predict(result_data)
print(result)
with open("./submission.csv","w") as f:
    f.write("id,label\n")
    for i in range(len(result)):
        f.write("%d,%d\n"%(i,result[i]))

