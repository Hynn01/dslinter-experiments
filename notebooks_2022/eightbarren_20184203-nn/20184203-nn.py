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


# 取数据
import torch.utils.data as Data


def dataloader(batch_size, shuffle=True):
    # from sklearn.datasets import load_digits
    # digits = load_digits()
    # X = digits.data
    # y = digits.target
    # Y = []
    # for i in y:
    #     Y.append([i])

    train_data = pd.read_csv('/kaggle/input/ml2021-2022-2-nn/train.csv').values
    test_data = pd.read_csv('/kaggle/input/ml2021-2022-2-nn/test.csv').values
    train_X = train_data[:, 1:]
    train_y = train_data[:, 0]
    test_X = test_data
    test_y = test_data[:, 0]
    len_train_data = len(train_X)
    len_test_data = len(test_X)
    train_dataset = []
    test_dataset = []
    for i in range(len(train_X)):
        train_dataset.append((train_X[i], train_y[i]))
    for i in range(len(test_X)):
        test_dataset.append((test_X[i], test_y[i]))
    train_iter = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_iter = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_iter, test_iter, len_train_data, len_test_data


# In[ ]:


# 定义一个线性类
class LinearLayer:
    def __init__(self, n_in, n_out, batch_size, activation=None, lr=0.001):
        self.W = np.random.normal(scale=0.01, size=(n_in, n_out))
        self.b = np.zeros((batch_size, n_out))
        self.activation = activation
        self.lr = lr
        self.batch_size = batch_size
        self.parameter = {'name':'Linear', 'size':[n_in, n_out], 'activation':activation}

    def forward(self, x):
        self.x = x
        output = np.dot(x, self.W) + self.b
        if self.activation is 'relu':
            output = np.maximum(0, output)
        if self.activation is 'sigmoid':
            output = 1 / (1 + np.exp(-output))
        if self.activation is 'tanh':
            output = np.tanh(output)
        self.activated_output = output
        return output

    def backward(self, dout):
        if self.activation is 'relu':
            self.activated_output[self.activated_output <= 0] = 0
            self.activated_output[self.activated_output > 0] = 1
            dout = dout * self.activated_output
        if self.activation is 'sigmoid':
            dout = self.activated_output * (1 - self.activated_output) * dout
        if self.activation is 'tanh':
            dout = (1 - self.activated_output ** 2) * dout
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = dout
        self.W = self.W - self.dW * self.lr / self.batch_size
        self.b = self.b - self.db * self.lr / self.batch_size
        return dx
    
# 定义softmax类
class SoftMax:
    y_hat = []

    def __init__(self):
        super(SoftMax, self).__init__()
        self.parameter = {'name':'SoftMax'}

    def forward(self, x):
        x_exp = np.exp(x)
        partition = np.sum(x_exp, axis=1, keepdims=True)
        self.y_hat = x_exp / partition
        return self.y_hat

    def backward(self, y):
        dout = self.y_hat - y
        return dout


# In[ ]:


# 定义MLP类
class MLP:
    def __init__(self, input_size, batch_size, num_classes, lr=0.001, hidden_layer_sizes=(), activation='relu'):

        self.layer_list = [[hidden_layer_sizes[i], hidden_layer_sizes[i + 1]]
                           for i in range(len(hidden_layer_sizes) - 1)]
        self.input_layer = LinearLayer(input_size, hidden_layer_sizes[0], batch_size, activation, lr=lr)
        self.classifier = LinearLayer(hidden_layer_sizes[-1], num_classes, batch_size, activation, lr=lr)
        self.softmax = SoftMax()
        self.batch_size = batch_size
        self.lr = lr

        self.layers = [self.input_layer]
        for i in range(len(self.layer_list)):
            self.layers.append(LinearLayer(self.layer_list[i][0], self.layer_list[i][1], batch_size, activation, lr=lr))
        self.layers.append(self.classifier)
        self.layers.append(self.softmax)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y):
        for layer in reversed(self.layers):
            y = layer.backward(y)

    def parameter(self):
        for i in range(len(self.layers)):
            print("layer {}: {}".format(i + 1, self.layers[i].parameter))


# In[ ]:


# 设置超参数
num_epochs = 10
batch_size = 200

# 取数据
train_iter, test_iter, len_train_data, len_test_data = dataloader(batch_size)

# 实例化MLP模型
model = MLP(input_size=784, batch_size=batch_size, num_classes=10, lr=0.001, hidden_layer_sizes=(256,),
            activation='tanh')

# 打印一下模型参数
model.parameter()


# In[ ]:


from tqdm import tqdm

# 开始训练
acc_list = [0.,]
for epoch in range(num_epochs):
    acc = 0
    with tqdm(train_iter, unit='batch') as tepoch:
        for data, label in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1} train")
            if data.shape[0] < batch_size:
                break
            data = data.numpy()
            label = label.numpy()
            outputs = model.forward(data)
            acc += (outputs.argmax(1) == label).sum() / len_train_data
            model.backward(np.eye(10)[label])
            tepoch.set_postfix(acc=acc)
    acc_list.append(acc)


# In[ ]:


#做预测输出
predictions = []
with tqdm(test_iter, unit='batch') as tepoch:
    for data, _ in tepoch:
        tepoch.set_description(f"Prediction")
        data = data.numpy()
        outputs = model.forward(data)
        outputs_label = outputs.argmax(1)
        for label in outputs_label:
            predictions.append(label)


# In[ ]:


out_dict = {
    'id':list(np.arange(len_test_data)),
    'label':predictions
}
out = pd.DataFrame(out_dict)
out.to_csv('submission.csv',index=False)

